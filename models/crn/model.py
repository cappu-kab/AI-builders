"""
model.py
========
Convolutional Recurrent Network (CRN) for speech enhancement,
operating on the **complex** STFT (real + imaginary channels).

Architecture
------------
        noisy waveform
             │
        STFT  → complex (B, F, T) → stacked (B, 2, F, T)        # 2 = re/im
             │
   ┌─────────┴─────────┐
   │  CNN encoder      │  5× stride-2 freq downsampling,
   │  (Conv2d+BN+ELU)  │  kernel (5,3), doubling channels each block,
   │  + SE attention   │  2→16→32→64→128→256 channels
   └─────────┬─────────┘
             │
        proj_in  2304 → bottleneck_dim (384)   ← keeps GPU compute tractable
             │
        BiLSTM bottleneck (per time-step)       ← default: 2-layer, hidden=256
          output dim = rnn_hidden × 2 (bidirectional)
             │
        proj_out  rnn_hidden×2 → 2304
             │
   ┌─────────┴─────────┐
   │  CNN decoder      │  5× transposed conv blocks
   │  with U-Net skips │  skip-concat at every encoder level
   └─────────┬─────────┘
             │
        complex noise mask (B, 2, F, T)   # tanh-bounded real+imag mask
             │
        noise_spec = mask ⊗ noisy_complex  (complex multiply)
             │
        iSTFT(noise_spec) → predicted_noise
             │
        enhanced = noisy_waveform − predicted_noise   # residual learning

Bottleneck options (rnn_type)
-----------------------------
  "bilstm"  2-layer bidirectional LSTM  [default]
            proj_in → BiLSTM(hidden) → proj_out
            output dim = rnn_hidden × 2, so proj_out takes rnn_hidden*2
  "gru"     1-layer unidirectional GRU  (legacy / fast preset)
  "lstm"    N-layer unidirectional LSTM

Full-band coverage (n_fft=512, 16 kHz):
  Input:     F = 257 bins  (0–8 kHz, bin width 31.25 Hz)
  Enc level 1: 129 bins — captures everything above 4 kHz (sibilants, fricatives)
  Enc level 2:  65 bins — preserves harmonics up to 2 kHz
  Enc level 3:  33 bins
  Enc level 4:  17 bins
  Bottleneck:    9 bins — compressed representation, all bands present
  Decoder restores full 257 bins via transposed conv + skip connections.
  No low-pass bias: all 257 bins feed directly into every skip connection,
  so the decoder always has access to the full-band input representation.

Reference for complex masking:  Tan & Wang, "Complex Spectral Mapping with a
Convolutional Recurrent Network", ICASSP 2019.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# STFT helpers (kept as nn.Modules so they live on the right device)
# ---------------------------------------------------------------------------
class STFT(nn.Module):
    def __init__(self, n_fft: int = 512, hop_length: int = 128, win_length: int | None = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """waveform (B, T) -> complex spec (B, F, T'). Forces fp32 under AMP."""
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.float()
            spec = torch.stft(x32, n_fft=self.n_fft, hop_length=self.hop_length,
                              win_length=self.win_length, window=self.window,
                              return_complex=True, center=True)
        return spec


class ISTFT(nn.Module):
    def __init__(self, n_fft: int = 512, hop_length: int = 128, win_length: int | None = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    def forward(self, spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            spec32 = spec.to(torch.complex64)
            return torch.istft(spec32, n_fft=self.n_fft, hop_length=self.hop_length,
                               win_length=self.win_length, window=self.window,
                               length=length, center=True)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018).

    Learns per-channel importance weights from global average pooling.
    Costs ~2 × C² / reduction params — for our encoder channels
    [16,32,64,128,256] that's <22 K total, negligible.
    """
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(channels, mid, bias=True)
        self.fc2  = nn.Linear(mid, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x).flatten(1)                             # (B, C)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s.view(s.size(0), s.size(1), 1, 1)


class EncBlock(nn.Module):
    """Stride-2 (along freq) downsampling block with SE channel attention.

    Kernel (5, 3) with padding (2, 1): preserves time length exactly,
    halves the freq dim (rounded up).
    SE attention is applied after activation; disable with use_se=False
    to load checkpoints trained without it.
    """
    def __init__(self, in_ch: int, out_ch: int, use_se: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(5, 3),
                              stride=(2, 1), padding=(2, 1))
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ELU()
        self.se   = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.act(self.bn(self.conv(x))))


class DecBlock(nn.Module):
    """Stride-2 (along freq) upsampling block.

    Kernel (5, 3) with padding (2, 1) and output_padding (1, 0):
    preserves time length, doubles freq (we trim with _pad_freq later).
    """
    def __init__(self, in_ch: int, out_ch: int, last: bool = False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(5, 3),
                                         stride=(2, 1), padding=(2, 1),
                                         output_padding=(1, 0))
        self.bn = nn.BatchNorm2d(out_ch) if not last else nn.Identity()
        self.act = nn.ELU() if not last else nn.Identity()
        self.last = last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))


# ---------------------------------------------------------------------------
# LF-bypass branch
# ---------------------------------------------------------------------------

def _lf_groups(channels: int, target: int = 8) -> int:
    g = min(target, channels)
    while channels % g and g > 1:
        g -= 1
    return g


class LFBypass(nn.Module):
    """Direct bypass for low-frequency bins that collapse at the 9-bin bottleneck.

    5 EncBlocks squeeze 257 freq bins → 9 at the BiLSTM, crushing bins 0-6
    (< 200 Hz at 16 kHz).  This branch reads the lowest `lf_bins` bins from
    the original input spectrum (no downsampling) and produces a mask correction
    that is added to the decoder output before magnitude bounding.
    Zero-initialized — strict no-op at epoch 0, activates gradually via training.
    """

    def __init__(self, lf_bins: int = 16, hidden: int = 32,
                 dilations: Tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.lf_bins  = lf_bins
        self.proj_in  = nn.Conv2d(2, hidden, 1, bias=False)
        self.gn_in    = nn.GroupNorm(_lf_groups(hidden), hidden)
        self.blocks   = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(hidden, hidden, (1, 5), padding=(0, 2 * d), dilation=(1, d),
                          groups=hidden, bias=False),
                nn.Conv2d(hidden, hidden, 1, bias=False),
                nn.GroupNorm(_lf_groups(hidden), hidden),
                nn.ELU(),
            ))
        self.proj_out = nn.Conv2d(hidden, 2, 1, bias=True)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, lf_spec: torch.Tensor) -> torch.Tensor:
        """lf_spec: (B, 2, lf_bins, T) → (B, 2, lf_bins, T) mask correction."""
        h = F.elu(self.gn_in(self.proj_in(lf_spec)))
        for blk in self.blocks:
            h = h + blk(h)
        return self.proj_out(h)


class LearnableSkipGain(nn.Module):
    """Per-channel scalar gate on skip connections (init=1 → identity).

    Lets the decoder learn how much encoder detail to pass at each U-Net level
    without paying the cost of a full attention layer. ~C parameters per skip.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain   # (B, C, F, T) * (C, 1, 1)


class FreqAttention(nn.Module):
    """Cross-frequency self-attention at the encoder bottleneck.

    With n_fft=512 and 5 encoder blocks, the bottleneck has only Fb=9 frequency
    bins.  Multi-head attention over these 9 tokens captures cross-bin harmonic
    relationships (fundamental vs. overtones, HVAC harmonics) that the BiLSTM
    misses because it flattens all C×Fb features into one vector per time step.

    Zero-initialised output projection → strict no-op at epoch 0, so existing
    checkpoints load and run correctly with strict=False without any accuracy change.
    """

    def __init__(self, channels: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        # Clamp heads so channels is always divisible (e.g. channels=16 → 2 heads max)
        n_heads = min(n_heads, channels // 4) or 1
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads,
                                          dropout=dropout, batch_first=True)
        self.proj = nn.Linear(channels, channels, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, Fb, T)
        B, C, Fb, T = x.shape
        # (B, C, Fb, T) → (B*T, Fb, C): attend over frequency bins per time step
        q = x.permute(0, 3, 2, 1).reshape(B * T, Fb, C)
        q_norm = self.norm(q)
        attn_out, _ = self.attn(q_norm, q_norm, q_norm)
        out = (q + self.proj(attn_out)).reshape(B, T, Fb, C).permute(0, 3, 2, 1)
        return out.contiguous()


# ---------------------------------------------------------------------------
# CRN
# ---------------------------------------------------------------------------
class CRN(nn.Module):
    """
    Complex-spectrogram CRN denoiser with U-Net-style skip connections.
    """

    def __init__(self,
                 n_fft: int = 512,
                 hop_length: int = 128,
                 win_length: int | None = None,
                 base_channels: int = 32,
                 rnn_hidden: int = 192,
                 rnn_layers: int = 1,
                 rnn_type: str = "gru",
                 bottleneck_dim: int = 384,
                 mask_bound: float = 1.2,
                 dropout_p: float = 0.1,
                 use_se: bool = True,
                 use_freq_attn: bool = True,
                 sample_rate: int = 16_000,
                 # legacy kwarg names — still accepted
                 lstm_hidden: int | None = None,
                 lstm_layers: int | None = None):
        super().__init__()
        if lstm_hidden is not None: rnn_hidden = lstm_hidden
        if lstm_layers is not None: rnn_layers = lstm_layers
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.mask_bound = mask_bound

        self.stft  = STFT(n_fft, hop_length, self.win_length)
        self.istft = ISTFT(n_fft, hop_length, self.win_length)

        # 5 encoder blocks: 2 -> 16 -> 32 -> 64 -> 128 -> 256
        ch = [2, base_channels, base_channels * 2, base_channels * 4,
              base_channels * 8, base_channels * 16]
        self.encoders = nn.ModuleList(
            [EncBlock(ch[i], ch[i + 1], use_se=use_se) for i in range(5)]
        )
        self.skip_gains = nn.ModuleList(
            [LearnableSkipGain(ch[i + 1]) for i in range(5)]
        )

        # bottleneck on (B, T, feat) where feat = C * F'
        F_in = n_fft // 2 + 1
        F_b = F_in
        for _ in range(5):
            F_b = (F_b + 1) // 2
        self.bottleneck_F = F_b
        self.bottleneck_C = ch[-1]
        feat = self.bottleneck_C * self.bottleneck_F
        self.bottleneck_se  = SEBlock(ch[-1])       # global channel attention before BiLSTM
        # FreqAttention is opt-in: checkpoints trained BEFORE this block was
        # added do not contain freq_attn.* keys; load_model() auto-detects
        # their absence and constructs CRN(use_freq_attn=False) so there are
        # no missing keys and the architecture exactly matches the checkpoint.
        self.use_freq_attn = bool(use_freq_attn)
        if self.use_freq_attn:
            self.freq_attn = FreqAttention(self.bottleneck_C, n_heads=4)

        # ---- project (B,T,feat) -> (B,T,bottleneck_dim) before recurrent layer ----
        # CRITICAL: do NOT skip this projection. The raw feature dim (2304) makes
        # even a small LSTM prohibitively slow. proj_in compresses to bottleneck_dim
        # first, keeping the recurrent compute tractable on a single GPU.
        self.proj_in  = nn.Linear(feat, bottleneck_dim)
        self.dropout  = nn.Dropout(p=dropout_p)

        _rnn = rnn_type.lower()
        _bidirectional = _rnn == "bilstm"
        if _rnn == "gru":
            self.rnn = nn.GRU(input_size=bottleneck_dim, hidden_size=rnn_hidden,
                              num_layers=rnn_layers, batch_first=True)
        elif _bidirectional:
            # Bidirectional LSTM: each time-step attends to the full sequence context
            # in both directions — better at capturing long-range harmonic patterns
            # and precisely localising transient / low-freq events.
            # Output dim = rnn_hidden × 2 (forward + backward concatenated).
            self.rnn = nn.LSTM(input_size=bottleneck_dim, hidden_size=rnn_hidden,
                               num_layers=rnn_layers, batch_first=True,
                               bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_size=bottleneck_dim, hidden_size=rnn_hidden,
                               num_layers=rnn_layers, batch_first=True)

        rnn_out_dim = rnn_hidden * (2 if _bidirectional else 1)
        self.proj_out = nn.Linear(rnn_out_dim, feat)
        # alias for back-compat with old checkpoints that named it `proj`
        self.proj = self.proj_out

        # 5 decoder blocks (input doubles because of skip-concat)
        self.decoders = nn.ModuleList([
            DecBlock(ch[5] * 2, ch[4]),
            DecBlock(ch[4] * 2, ch[3]),
            DecBlock(ch[3] * 2, ch[2]),
            DecBlock(ch[2] * 2, ch[1]),
            DecBlock(ch[1] * 2, 2, last=True),       # back to 2 channels (re/im mask)
        ])

        # LFBypass: direct gradient path for bins the bottleneck crushes most.
        # Target strictly < 200 Hz — the HVAC/hum zone — so the bypass never
        # touches speech fundamentals (80–300 Hz for normal voices).
        # Covering 0–469 Hz (old lf_bins=16) overlapped the fundamental range
        # and caused the bypass to suppress speech harmonics (over-suppression).
        hz_per_bin = sample_rate / n_fft
        self.lf_bins   = max(4, round(200.0 / hz_per_bin) + 1)   # ≈7 for n_fft=512
        self.lf_bypass = LFBypass(lf_bins=self.lf_bins, hidden=32)

    # ------------------------------------------------------------------
    @staticmethod
    def _pad_freq(x: torch.Tensor, target: int) -> torch.Tensor:
        diff = target - x.shape[2]
        if diff <= 0:
            return x[:, :, :target]
        return F.pad(x, (0, 0, 0, diff))

    # ------------------------------------------------------------------
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform : (B, T)
        returns  : (B, T) enhanced waveform
        """
        L = waveform.shape[-1]
        spec = self.stft(waveform)                            # (B, F, T')  complex
        re, im = spec.real, spec.imag
        x = torch.stack([re, im], dim=1)                      # (B, 2, F, T')

        # encoder + skip cache
        skips: List[torch.Tensor] = []
        h = x
        for enc, gain in zip(self.encoders, self.skip_gains):
            h = enc(h)
            skips.append(gain(h))

        # bottleneck along time:  feat -> proj_in -> RNN -> proj_out -> feat
        h = self.bottleneck_se(h)             # channel attention: LF vs HF separation
        if self.use_freq_attn:
            h = self.freq_attn(h)             # cross-frequency harmonic attention
        B, C, Fb, T = h.shape
        seq = h.permute(0, 3, 1, 2).reshape(B, T, C * Fb)
        seq = self.proj_in(seq)
        seq = self.dropout(seq)   # breaks trivial identity path; off during eval
        seq, _ = self.rnn(seq)
        seq = self.proj_out(seq)
        h = seq.reshape(B, T, C, Fb).permute(0, 2, 3, 1).contiguous()

        # decoder with skip-concat
        for dec, skip in zip(self.decoders, reversed(skips)):
            h = self._pad_freq(h, skip.shape[2])              # align freq dim
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        # match exactly the input freq size
        h = self._pad_freq(h, spec.shape[1])

        # LFBypass: inject correction from raw input for low-freq bins only
        lf_corr = self.lf_bypass(x[:, :, :self.lf_bins, :])
        h = torch.cat([h[:, :, :self.lf_bins, :] + lf_corr,
                       h[:, :, self.lf_bins:, :]], dim=-2)

        mask_re, mask_im = h[:, 0], h[:, 1]

        # Magnitude-bounded complex mask — prevents mask inversion.
        #
        # Independent tanh per component allows |mask| up to mask_bound*sqrt(2),
        # which lets the model extract more energy than the input contains.
        # That makes predicting clean speech (large, structured) energetically
        # cheaper than predicting noise, causing "target leakage" and eventual
        # inversion where noise_pred ≈ clean and enhanced ≈ noise.
        #
        # Bounding the MAGNITUDE preserves the phase direction chosen by the
        # decoder while capping how much energy can be predicted as noise.
        # With mask_bound=1.2: |noise_pred| ≤ 1.2 * |noisy|, so the model can
        # never "invent" energy it has not seen in the input.
        # eps=1e-6 → min denominator 1e-3. Prevents the 10 000× gradient
        # amplification that occurs at initialisation when mask_mag ≈ 1e-4.
        mask_mag    = torch.sqrt(mask_re ** 2 + mask_im ** 2 + 1e-6)
        bounded_mag = torch.tanh(mask_mag) * self.mask_bound
        scale       = bounded_mag / mask_mag          # ≈ tanh(x)/x ≈ 1 at init
        mask_re     = mask_re * scale
        mask_im     = mask_im * scale

        # Residual learning: mask predicts NOISE (not clean speech).
        # complex multiplication: (a+bi)(c+di) = (ac-bd)+(ad+bc)i
        noise_re = mask_re * re - mask_im * im
        noise_im = mask_re * im + mask_im * re
        noise_spec = torch.complex(noise_re, noise_im)
        noise_wave = self.istft(noise_spec, length=L)

        # Subtract predicted noise — at init (mask≈0) this passes through noisy unchanged,
        # which is a far better starting point than predicting silence (old behaviour).
        return waveform - noise_wave


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for rnn_type, rnn_hidden, rnn_layers in [
        ("gru",    192, 1),   # legacy fast preset
        ("bilstm", 256, 2),   # new default — 2-layer BiLSTM
    ]:
        net = CRN(rnn_type=rnn_type, rnn_hidden=rnn_hidden, rnn_layers=rnn_layers,
                  sample_rate=16_000)
        x   = torch.randn(2, 16_000 * 4)
        y   = net(x)
        n   = sum(p.numel() for p in net.parameters())
        print(f"{rnn_type:6s}  hidden={rnn_hidden}×{rnn_layers}  lf_bins={net.lf_bins}"
              f"  input={x.shape}  output={y.shape}  params={n/1e6:.2f}M")
