"""
model_unet_advanced.py
======================

Advanced U-Net for complex-spectrogram speech enhancement via residual
noise prediction.

    Input  : (B, T)   noisy waveform
    Output : (B, T)   enhanced waveform  =  noisy  −  predicted_noise

Architecture summary
--------------------
1. Internal STFT
   - (B, T) → (B, 2, F, T) complex spectrogram, fp32 even under AMP

2. Encoder (5 stages)
   - Each stage: ResidualConvBlock (Conv2d -> GroupNorm -> LeakyReLU x2 + skip)
   - Frequency-only downsampling with stride (2, 1); time resolution preserved
   - Skip tensors saved BEFORE downsampling so decoder gets full resolution

3. Bottleneck
   - Stack of dilated 2D convs (dilations 1, 2, 4, 8) with residual links
   - Squeeze-Excitation (SE) channel attention afterwards
   - Two dilation sweeps for wider receptive field (second sweep: 1,2,4,8,16)

4. Decoder (5 stages, mirrored)
   - ConvTranspose2d for frequency upsampling
   - Gated skip connections: sigmoid gate decides how much encoder
     information to admit (mitigates noise leakage through skips)
   - ResidualConvBlock after every merge

5. Mask head
   - Predicts a raw Complex Ratio Mask (CRM) of shape (B, 2, F, T)
   - Magnitude-bounded mask: |mask| = tanh(|raw|) * mask_bound
     Bounds the noise-prediction energy so the model cannot predict
     more noise energy than mask_bound × noisy energy — prevents mask
     inversion where the model accidentally learns to predict speech.
   - mask_bound=1.2: allows up to 20% overshoot vs. noisy magnitude

6. Residual subtraction
   - noise_spec = apply_mask(noisy_spec, mask)
   - noise_wav  = iSTFT(noise_spec)
   - enhanced   = noisy_wav − noise_wav

Stability tricks
----------------
- Kaiming-normal init for all convs, GroupNorm gamma=1, beta=0
- Mask head initialised to (0, 0) bias: mask starts at near-zero,
  noise_pred ≈ 0, enhanced ≈ noisy — a clean pass-through at epoch 0
- Magnitude-bounded mask prevents gradient explosion and mask inversion
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Helpers
# =============================================================================

def _groups(channels: int, target: int = 8) -> int:
    """Pick the largest group count <= target that divides `channels`."""
    g = min(target, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


def _crop_or_pad(x: torch.Tensor, F_target: int, T_target: int) -> torch.Tensor:
    """Center-crop or zero-pad x along the last two dims to (F_target, T_target).

    Decoder upsampling can be off by 1 vs. the encoder skip when F or T are
    odd; this aligns shapes losslessly.
    """
    Fx, Tx = x.size(-2), x.size(-1)

    # Frequency
    if Fx > F_target:
        start = (Fx - F_target) // 2
        x = x[..., start:start + F_target, :]
    elif Fx < F_target:
        diff = F_target - Fx
        x = F.pad(x, (0, 0, diff // 2, diff - diff // 2))

    # Time
    Tx = x.size(-1)
    if Tx > T_target:
        start = (Tx - T_target) // 2
        x = x[..., :, start:start + T_target]
    elif Tx < T_target:
        diff = T_target - Tx
        x = F.pad(x, (diff // 2, diff - diff // 2, 0, 0))

    return x


# =============================================================================
# Building blocks
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class ResidualConvBlock(nn.Module):
    """Depthwise-separable two-conv residual block.

    Replaces expensive full Conv2d(5×3) with a PW+DW pair per conv:
        conv1 : PW(in→out, 1×1) → DW(out, 5×3, groups=out)
        conv2 : DW(out, 5×3, groups=out) → PW(out→out, 1×1)

    Parameter savings at out_ch=256 vs. standard 5×3 Conv2d:
        conv1 full=128×256×15=491 520  →  DS=128×256+256×15=36 608  (13×)
        conv2 full=256×256×15=983 040  →  DS=256×15+256×256=69 376  (14×)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int] = (5, 3),
    ) -> None:
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        # conv1: project channels then mix spatially (depthwise)
        self.entry = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                      if in_ch != out_ch else nn.Identity())
        self.dw1   = nn.Conv2d(out_ch, out_ch, kernel_size,
                               padding=pad, groups=out_ch, bias=False)
        self.gn1   = nn.GroupNorm(_groups(out_ch), out_ch)
        # conv2: mix spatially (depthwise) then mix channels (pointwise)
        self.dw2   = nn.Conv2d(out_ch, out_ch, kernel_size,
                               padding=pad, groups=out_ch, bias=False)
        self.pw2   = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.gn2   = nn.GroupNorm(_groups(out_ch), out_ch)
        self.act   = nn.LeakyReLU(0.1, inplace=True)
        self.proj  = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                      if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.act(self.gn1(self.dw1(self.entry(x))))
        out = self.gn2(self.pw2(self.dw2(out)))
        return self.act(out + identity)


class EncoderStage(nn.Module):
    """ResidualConvBlock at full resolution + frequency downsampling."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int] = (5, 3),
    ) -> None:
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch, kernel_size)
        self.down = nn.Conv2d(
            out_ch, out_ch, kernel_size=(3, 1), stride=(2, 1),
            padding=(1, 0), bias=False,
        )
        self.down_norm = nn.GroupNorm(_groups(out_ch), out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x)
        x = self.act(self.down_norm(self.down(skip)))
        return x, skip


class GatedSkip(nn.Module):
    """Learnable gating for skip connections.

    lf_direct_bins > 0 adds a zero-initialized additive residual for the
    lowest `lf_direct_bins` frequency bins, bypassing the sigmoid gate.
    This ensures low-frequency encoder features (HVAC/fan/hum patterns)
    are never fully suppressed by the gate and have a direct gradient path
    to the mask head regardless of what the gate learns.
    """

    def __init__(self, channels: int, lf_direct_bins: int = 0) -> None:
        super().__init__()
        self.gate = nn.Conv2d(2 * channels, channels, 1, bias=True)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.lf_direct_bins = lf_direct_bins
        if lf_direct_bins > 0:
            # Scalar gate initialized to 0: sigmoid(0)=0.5 but starts adding
            # only after the gated path stabilizes. True no-op needs zeros init
            # on the gated proj and direct path together, which _init_weights handles.
            self.lf_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, dec: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([dec, skip], dim=1)))
        out = dec + g * self.proj(skip)
        if self.lf_direct_bins > 0 and self.lf_direct_bins <= skip.shape[-2]:
            lf_gain = torch.sigmoid(self.lf_alpha)
            out = torch.cat([
                out[:, :, :self.lf_direct_bins, :] + lf_gain * skip[:, :, :self.lf_direct_bins, :],
                out[:, :, self.lf_direct_bins:, :]
            ], dim=-2)
        return out


class DecoderStage(nn.Module):
    """Frequency-only upsample -> shape align -> gated skip merge -> residual block."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int] = (5, 3),
        lf_direct_bins: int = 0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=(3, 1), stride=(2, 1),
            padding=(1, 0), output_padding=(0, 0), bias=False,
        )
        self.up_norm = nn.GroupNorm(_groups(out_ch), out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.skip_proj = (
            nn.Conv2d(skip_ch, out_ch, 1, bias=False)
            if skip_ch != out_ch else nn.Identity()
        )
        self.gated = GatedSkip(out_ch, lf_direct_bins=lf_direct_bins)
        self.block = ResidualConvBlock(out_ch, out_ch, kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.up_norm(self.up(x)))
        skip = self.skip_proj(skip)
        if x.shape[-2:] != skip.shape[-2:]:
            x = _crop_or_pad(x, skip.size(-2), skip.size(-1))
        x = self.gated(x, skip)
        x = self.block(x)
        return x


class DilatedBottleneck(nn.Module):
    """Dilated-conv bottleneck with SE attention.

    Each layer is a pre-activation residual unit:
        x = x + GN( Conv_dilated( x ) ) -> LeakyReLU
    Dilation is applied along the TIME axis only (frequency is tiny at depth=5).

    Two sweeps of dilations are used by default:
        sweep 1: (1, 2, 4, 8)   — base receptive field
        sweep 2: (1, 2, 4, 8, 16) — wider field for longer-range noise patterns
    """

    def __init__(
        self,
        channels: int,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
        extra_dilations: Tuple[int, ...] = (1, 2, 4, 8, 16),
        kernel_size: Tuple[int, int] = (5, 3),
    ) -> None:
        super().__init__()
        kF, kT = kernel_size

        def _make_sweep(dils):
            layers: List[nn.Module] = []
            for d in dils:
                pad = (kF // 2, (kT // 2) * d)
                layers.append(nn.Sequential(
                    # Depthwise: spatial mixing within each channel, dilated in time.
                    # Replaces a standard channels×channels×15 conv (~983K at ch=256) with
                    # channels×15 depthwise + channels×channels pointwise (~69K at ch=256).
                    nn.Conv2d(channels, channels, kernel_size,
                              padding=pad, dilation=(1, d),
                              groups=channels, bias=False),
                    # Pointwise: channel mixing (no spatial footprint).
                    nn.Conv2d(channels, channels, 1, bias=False),
                    nn.GroupNorm(_groups(channels), channels),
                    nn.LeakyReLU(0.1, inplace=True),
                ))
            return nn.ModuleList(layers)

        self.sweep1  = _make_sweep(dilations)
        self.se1     = SEBlock(channels)
        self.sweep2  = _make_sweep(extra_dilations) if extra_dilations else nn.ModuleList()
        self.se2     = SEBlock(channels) if extra_dilations else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.sweep1:
            x = x + layer(x)
        x = self.se1(x)
        for layer in self.sweep2:
            x = x + layer(x)
        if self.se2 is not None:
            x = self.se2(x)
        return x


# =============================================================================
# LF-bypass branch
# =============================================================================

class LFBypass(nn.Module):
    """Dedicated bypass branch for low-frequency bins crushed in the main UNet.

    The 5 encoder stages downsample 257 freq bins → 9 at the bottleneck.
    Bins 0-6 (< 200 Hz at 16 kHz / 512-pt STFT) collapse to ~0-1 bins there,
    so the main path has very weak gradient signal for HVAC/fan/hum suppression.

    This branch takes the lowest `lf_bins` frequency bins (covering 0-500 Hz)
    from the ORIGINAL spectrogram (no downsampling) and processes them through
    temporal-only dilated convolutions, producing a mask correction for those
    bins that is added to the main U-Net raw mask BEFORE magnitude bounding.

    Zero-initialized output (proj_out): starts as a strict no-op at epoch 0,
    gradually activates as training discovers LF suppression opportunities.
    """

    def __init__(self, lf_bins: int = 16, hidden: int = 32,
                 dilations: Tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.lf_bins  = lf_bins
        self.proj_in  = nn.Conv2d(2, hidden, 1, bias=False)
        self.gn_in    = nn.GroupNorm(_groups(hidden), hidden)
        self.blocks   = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(hidden, hidden, (1, 5), padding=(0, 2 * d), dilation=(1, d),
                          groups=hidden, bias=False),
                nn.Conv2d(hidden, hidden, 1, bias=False),
                nn.GroupNorm(_groups(hidden), hidden),
                nn.LeakyReLU(0.1, inplace=True),
            ))
        self.proj_out = nn.Conv2d(hidden, 2, 1, bias=True)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, lf_spec: torch.Tensor) -> torch.Tensor:
        """lf_spec: (B, 2, lf_bins, T) → (B, 2, lf_bins, T) mask correction."""
        h = F.leaky_relu(self.gn_in(self.proj_in(lf_spec)), 0.1, inplace=True)
        for blk in self.blocks:
            h = h + blk(h)
        return self.proj_out(h)


# =============================================================================
# Main model
# =============================================================================

class AdvancedUNetSE(nn.Module):
    """Advanced U-Net with SE attention, dilated bottleneck and gated skips.

    Residual noise-prediction architecture:
        enhanced = noisy_wave − iSTFT( mask ⊙ STFT(noisy_wave) )

    where mask is magnitude-bounded:  |mask| = tanh(|raw_mask|) * mask_bound

    Args
    ----
    n_fft, hop_length, win_length : STFT params used internally.
    base_channels  : channels in the first encoder stage (doubled per level, capped).
    depth          : number of encoder/decoder stages (default 5; 257 -> 9 in F).
    mask_bound     : maximum magnitude of complex mask (tanh-scaled).
                     1.2 allows 20% overshoot so the model can suppress tonal noise
                     that has slightly higher magnitude than the noisy signal.
    kernel_size    : 2D kernel for residual blocks; default (5, 3) = wide-F, narrow-T.

    Forward
    -------
    wav    : (B, T) noisy waveform
    returns: (B, T) enhanced waveform
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        base_channels: int = 32,
        depth: int = 5,
        mask_bound: float = 1.2,
        kernel_size: Tuple[int, int] = (5, 3),
        max_channels: int = 256,
        wide_bottleneck: bool = True,
        sample_rate: int = 16_000,
    ) -> None:
        super().__init__()
        assert depth >= 3, "depth >= 3 required for a useful receptive field."

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mask_bound = float(mask_bound)
        self.depth = depth
        self.wide_bottleneck = wide_bottleneck

        channels: List[int] = []
        c = base_channels
        for _ in range(depth):
            channels.append(c)
            c = min(c * 2, max_channels)
        self.enc_channels = channels

        # ----- Encoder -----
        self.encoders = nn.ModuleList()
        in_ch = 2
        for out_ch in channels:
            self.encoders.append(EncoderStage(in_ch, out_ch, kernel_size))
            in_ch = out_ch

        # ----- Bottleneck -----
        extra = (1, 2, 4, 8, 16) if wide_bottleneck else ()
        self.bottleneck = DilatedBottleneck(
            channels[-1], dilations=(1, 2, 4, 8),
            extra_dilations=extra, kernel_size=kernel_size,
        )

        # Precompute lf_bins so decoder and LFBypass share the same value.
        # Target strictly <200 Hz to avoid suppressing speech harmonics (80-300 Hz).
        hz_per_bin = sample_rate / n_fft
        _lf_bins = max(4, round(200.0 / hz_per_bin) + 1)   # ≈7 for n_fft=512 @ 16 kHz

        # ----- Decoder (mirror) -----
        self.decoders = nn.ModuleList()
        in_c = channels[-1]
        for i in range(depth):
            skip_idx = depth - 1 - i
            skip_c = channels[skip_idx]
            out_c = channels[max(skip_idx - 1, 0)]
            # LF direct path only in the 2 shallowest stages (highest freq resolution)
            lf_dec = _lf_bins if i >= depth - 2 else 0
            self.decoders.append(
                DecoderStage(in_c, skip_c, out_c, kernel_size, lf_direct_bins=lf_dec)
            )
            in_c = out_c

        # ----- Mask head: predicts raw (real, imag) channels -----
        self.mask_head = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], (5, 3), padding=(2, 1), bias=False),
            nn.GroupNorm(_groups(channels[0]), channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[0], 2, kernel_size=1, bias=True),
        )

        # LFBypass: direct gradient path for bins < ~500 Hz (bypasses lossy bottleneck)
        self.lf_bins   = _lf_bins
        self.lf_bypass = LFBypass(lf_bins=self.lf_bins, hidden=32)

        # STFT/iSTFT window registered as buffer so .to(device) moves it automatically.
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        self._init_weights()

    # ---------------------------------------------------------------- init
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Noise-prediction init: bias = (0, 0)
        # → raw mask = (0, 0) → |raw| ≈ 0 → bounded_mag ≈ 0 → noise_spec ≈ 0
        # → noise_wav ≈ 0 → enhanced ≈ noisy (clean pass-through at epoch 0)
        last = self.mask_head[-1]
        assert isinstance(last, nn.Conv2d)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        # LFBypass output: zero-init overrides kaiming pass above — keeps no-op at epoch 0
        nn.init.zeros_(self.lf_bypass.proj_out.weight)

    # ------------------------------------------------------------- STFT/iSTFT
    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, T) → (B, 2, F, T) in fp32, safe under AMP autocast."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            w = wav.float()
            spec = torch.stft(
                w,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                return_complex=True,
            )
            return torch.stack([spec.real, spec.imag], dim=1)

    def istft(self, spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """(B, 2, F, T) → (B, L) in fp32, safe under AMP autocast."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            s = spec.float()
            complex_spec = torch.complex(s[:, 0], s[:, 1])
            wav = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                length=length,
                center=True,
            )
        return wav

    # ------------------------------------------------------------- internals
    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], Tuple[int, int]]:
        skips: List[torch.Tensor] = []
        orig_F, orig_T = x.size(-2), x.size(-1)
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)
        return x, skips, (orig_F, orig_T)

    def _decode(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor],
        orig_size: Tuple[int, int],
    ) -> torch.Tensor:
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = dec(x, skip)
        x = _crop_or_pad(x, orig_size[0], orig_size[1])
        return x

    # --------------------------------------------------------------- public
    def forward_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """Predict the magnitude-bounded complex noise mask. Shape (B, 2, F, T).

        Magnitude-bounded mask:
            raw  = mask_head(features)         # (B, 2, F, T) — unbounded
            mag  = sqrt(raw_re² + raw_im² + ε)
            bounded_mag = tanh(mag) * mask_bound
            mask = raw / mag * bounded_mag     # preserves direction, bounds magnitude
        """
        x, skips, orig = self._encode(spec)
        x = self.bottleneck(x)
        x = self._decode(x, skips, orig)
        raw     = self.mask_head(x)
        lf_corr = self.lf_bypass(spec[:, :, :self.lf_bins, :])
        raw = torch.cat([raw[:, :, :self.lf_bins, :] + lf_corr,
                         raw[:, :, self.lf_bins:, :]], dim=-2)
        re, im  = raw[:, 0:1], raw[:, 1:2]
        mag     = torch.sqrt(re ** 2 + im ** 2 + 1e-7)
        bounded = torch.tanh(mag) * self.mask_bound
        return torch.cat([re / mag * bounded, im / mag * bounded], dim=1)

    @staticmethod
    def apply_mask(spec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Complex multiply: noise_spec = noisy_spec ⊙ mask  (element-wise)."""
        nr, ni = spec[:, 0:1], spec[:, 1:2]
        mr, mi = mask[:, 0:1], mask[:, 1:2]
        er = nr * mr - ni * mi
        ei = nr * mi + ni * mr
        return torch.cat([er, ei], dim=1)

    def forward_noise_spec(self, spec: torch.Tensor) -> torch.Tensor:
        """Return the predicted noise complex STFT (B, 2, F, T)."""
        mask = self.forward_mask(spec)
        return self.apply_mask(spec, mask)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """End-to-end enhancement: waveform → enhanced waveform.

        enhanced = noisy_wave − iSTFT(mask ⊙ STFT(noisy_wave))
        """
        L = wav.shape[-1]
        spec      = self._stft(wav)                    # (B, 2, F, T)
        noise_spec = self.forward_noise_spec(spec)     # (B, 2, F, T)
        noise_wav  = self.istft(noise_spec, length=L)  # (B, L)
        return wav - noise_wav


# Convenience aliases — pick whichever name your train.py expects.
Model = AdvancedUNetSE
UNetAdvanced = AdvancedUNetSE


# =============================================================================
# Quick self-test
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L = 2, 16_000 * 4   # 4-second clips at 16 kHz

    # base_channels=32 (default) → channels [32,64,128,256,256]
    # DS ResidualConvBlock cuts encoder+decoder cost 10-14×; target ~2M total
    model = AdvancedUNetSE(n_fft=512, hop_length=128, win_length=512, mask_bound=1.2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.2f} M  (target: 2-4 M)")

    wav = torch.randn(B, L)
    with torch.no_grad():
        enhanced = model(wav)
    assert enhanced.shape == wav.shape, enhanced.shape
    print("Input  shape:", tuple(wav.shape))
    print("Output shape:", tuple(enhanced.shape))

    # Sanity: at init, enhanced ≈ noisy (mask ≈ 0 → noise ≈ 0)
    diff = (wav - enhanced).abs().mean().item()
    wav_rms = wav.abs().mean().item()
    print(f"Mean |noisy - enhanced| at init: {diff:.6f}  (noisy RMS ~ {wav_rms:.4f})")
    print(f"Pass-through ratio: {diff / (wav_rms + 1e-8):.4f}  (should be << 1)")
