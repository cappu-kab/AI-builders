#!/usr/bin/env python3
"""
export_crn_onnx.py
==================
Export checkpoints_crn2/best.pt to crn_crn2.onnx for ONNX Runtime GPU on Jetson.

WHY A WRAPPER (two reasons):
1. torch.stft/istft use complex tensors — not exportable by the legacy ONNX tracer.
   Fix: STFT and iSTFT are handled in numpy on the host (each <1 ms).
2. nn.MultiheadAttention dispatches to aten::_native_multi_head_attention (fused
   kernel) which has no ONNX symbolic. Fix: FreqAttnForONNX decomposes MHA
   into separate Q/K/V linear projections + matmul + softmax — all ONNX-native.

ONNX interface (all real-valued float32, no complex types):
  Input  x       : (1, 2, F, T_stft)   noisy STFT [ch0=re, ch1=im]
  Output noise_x : (1, 2, F, T_stft)   noise spectrum [ch0=re, ch1=im]

STFT shape for T_wave=47872, n_fft=512, hop=128, center=True:
  F = 257 bins,  T_stft = 375 frames  => shapes (1, 2, 257, 375)

Host code (numpy/scipy) is responsible for STFT and iSTFT:
    enhanced_wave = noisy_wave - istft( noise_x[:,0] + j*noise_x[:,1] )

Run from AI_builders/:
    python export_crn_onnx.py
"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── paths ─────────────────────────────────────────────────────────────────────
_AI_DIR  = Path(__file__).resolve().parent
_CRN_DIR = _AI_DIR / "Run" / "CRN"
_CKPT    = _CRN_DIR / "checkpoints" / "best.pt"
_OUT     = _AI_DIR / "crn_bilstm.onnx"
_OUT15   = _AI_DIR / "crn_bilstm_opset15.onnx"

sys.path.insert(0, str(_CRN_DIR))
from model import CRN           # noqa: E402
from inference import load_model  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
_CRN_CTX_N = 47_616
CHUNK_SIZE  = 256
T_WAVE      = _CRN_CTX_N + CHUNK_SIZE   # 47,872 samples
N_FFT       = 512
HOP         = 128
F_BINS      = N_FFT // 2 + 1            # 257
T_STFT      = T_WAVE // HOP + 1         # 375  (center=True pads n_fft//2 each side)


# ── FreqAttention with manually decomposed MHA ────────────────────────────────
class FreqAttnForONNX(nn.Module):
    """
    FreqAttention with nn.MultiheadAttention replaced by manual Q/K/V matmul.

    aten::_native_multi_head_attention (PyTorch 2.x fused kernel) has no ONNX
    symbolic for opset 17.  Manual decomposition uses only:
        nn.Linear / F.linear, torch.matmul, torch.softmax, reshape/permute
    — all of which export cleanly.

    Weights are copied from the trained MHA module; numerical output is
    identical to the original FreqAttention in eval mode (dropout=0).
    """

    def __init__(self, fa: nn.Module) -> None:
        super().__init__()
        mha = fa.attn
        self.norm       = fa.norm
        self.fa_proj    = fa.proj          # FreqAttention's own output projection
        self.n_heads    = mha.num_heads
        self.head_dim   = mha.head_dim
        self.embed_dim  = mha.embed_dim
        self.scale      = self.head_dim ** -0.5

        # Split the fused in_proj_weight (3*E, E) into three separate Linear layers
        E = self.embed_dim
        W = mha.in_proj_weight.detach()   # (3*E, E)
        b = mha.in_proj_bias.detach()     # (3*E,)

        self.q_proj = nn.Linear(E, E, bias=True)
        self.k_proj = nn.Linear(E, E, bias=True)
        self.v_proj = nn.Linear(E, E, bias=True)
        self.q_proj.weight = nn.Parameter(W[0*E:1*E])
        self.k_proj.weight = nn.Parameter(W[1*E:2*E])
        self.v_proj.weight = nn.Parameter(W[2*E:3*E])
        self.q_proj.bias   = nn.Parameter(b[0*E:1*E])
        self.k_proj.bias   = nn.Parameter(b[1*E:2*E])
        self.v_proj.bias   = nn.Parameter(b[2*E:3*E])

        # MHA output projection (trained, not zero-init)
        self.out_proj = nn.Linear(E, E, bias=True)
        self.out_proj.weight = nn.Parameter(mha.out_proj.weight.detach())
        self.out_proj.bias   = nn.Parameter(mha.out_proj.bias.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, Fb, T)
        B, C, Fb, T = x.shape
        E, H, D = self.embed_dim, self.n_heads, self.head_dim

        # flatten batch * time -> query sequence of Fb tokens
        q = x.permute(0, 3, 2, 1).reshape(B * T, Fb, C)   # (B*T, Fb, E)
        q_norm = self.norm(q)                               # (B*T, Fb, E)

        BT, S = B * T, Fb
        # project Q, K, V and split into heads
        Q = self.q_proj(q_norm).reshape(BT, S, H, D).transpose(1, 2)  # (BT, H, S, D)
        K = self.k_proj(q_norm).reshape(BT, S, H, D).transpose(1, 2)
        V = self.v_proj(q_norm).reshape(BT, S, H, D).transpose(1, 2)

        # scaled dot-product attention (no dropout in eval mode)
        attn_w = torch.matmul(Q, K.transpose(-2, -1)) * self.scale    # (BT, H, S, S)
        attn_w = torch.softmax(attn_w, dim=-1)
        attn_o = torch.matmul(attn_w, V)                               # (BT, H, S, D)

        # merge heads and project output
        attn_o = attn_o.transpose(1, 2).reshape(BT, S, E)             # (BT, S, E)
        attn_o = self.out_proj(attn_o)                                 # (BT, S, E)

        # residual (uses q before normalisation, matching original FreqAttention)
        out = (q + self.fa_proj(attn_o)).reshape(B, T, Fb, C).permute(0, 3, 2, 1)
        return out.contiguous()


# ── CRN body without STFT / iSTFT ────────────────────────────────────────────
class CRNBodyForONNX(nn.Module):
    """
    CRN encoder -> BiLSTM bottleneck -> decoder -> mask bounding.
    STFT and iSTFT removed; all ops are real-valued float32.

    Input  x       : (B, 2, F, T_stft)  noisy STFT [ch0=re, ch1=im]
    Output noise_x : (B, 2, F, T_stft)  noise spectrum [ch0=re, ch1=im]
    """

    def __init__(self, crn: CRN) -> None:
        super().__init__()
        self.encoders      = crn.encoders
        self.skip_gains    = crn.skip_gains
        self.bottleneck_se = crn.bottleneck_se
        self.use_freq_attn = crn.use_freq_attn
        if self.use_freq_attn:
            self.freq_attn = FreqAttnForONNX(crn.freq_attn)
        self.proj_in       = crn.proj_in
        self.dropout       = crn.dropout
        self.rnn           = crn.rnn
        self.proj_out      = crn.proj_out
        self.decoders      = crn.decoders
        self.lf_bins       = crn.lf_bins
        self.lf_bypass     = crn.lf_bypass
        self.mask_bound    = crn.mask_bound

    @staticmethod
    def _pad_freq(x: torch.Tensor, target: int) -> torch.Tensor:
        diff = target - x.shape[2]
        if diff <= 0:
            return x[:, :, :target]
        return F.pad(x, (0, 0, 0, diff))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        re = x[:, 0]    # (B, F, T_stft) — noisy STFT real part
        im = x[:, 1]    # (B, F, T_stft) — noisy STFT imag part
        Fq = x.shape[2]

        # encoder + U-Net skip connections
        skips: List[torch.Tensor] = []
        h = x
        for enc, gain in zip(self.encoders, self.skip_gains):
            h = enc(h)
            skips.append(gain(h))

        # BiLSTM bottleneck
        h = self.bottleneck_se(h)
        if self.use_freq_attn:
            h = self.freq_attn(h)
        B, C, Fb, T = h.shape
        seq = h.permute(0, 3, 1, 2).reshape(B, T, C * Fb)
        seq = self.proj_in(seq)
        seq = self.dropout(seq)
        seq, _ = self.rnn(seq)
        seq = self.proj_out(seq)
        h = seq.reshape(B, T, C, Fb).permute(0, 2, 3, 1).contiguous()

        # decoder with U-Net skip-concat
        for dec, skip in zip(self.decoders, reversed(skips)):
            h = self._pad_freq(h, skip.shape[2])
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        h = self._pad_freq(h, Fq)

        # LF bypass: direct path for bins 0-6 (<200 Hz ANC target band)
        lf_corr = self.lf_bypass(x[:, :, :self.lf_bins, :])
        h = torch.cat([h[:, :, :self.lf_bins, :] + lf_corr,
                       h[:, :, self.lf_bins:, :]], dim=-2)

        mask_re, mask_im = h[:, 0], h[:, 1]

        # magnitude-bounded complex mask
        mask_mag    = torch.sqrt(mask_re ** 2 + mask_im ** 2 + 1e-6)
        bounded_mag = torch.tanh(mask_mag) * self.mask_bound
        scale       = bounded_mag / mask_mag
        mask_re     = mask_re * scale
        mask_im     = mask_im * scale

        # complex multiply: mask * noisy_stft = noise_spectrum
        noise_re = mask_re * re - mask_im * im
        noise_im = mask_re * im + mask_im * re

        return torch.stack([noise_re, noise_im], dim=1)   # (B, 2, F, T_stft)


# ── load model ────────────────────────────────────────────────────────────────
print(f"Checkpoint    : {_CKPT}")
if not _CKPT.exists():
    sys.exit(f"ERROR: checkpoint not found at {_CKPT}")

device = torch.device("cpu")
crn, ckpt = load_model(_CKPT, device)
crn.eval()

saved_args = ckpt.get("args", {}) or {}
print(f"rnn_type      : {saved_args.get('rnn_type', '?')}")
print(f"rnn_hidden    : {saved_args.get('rnn_hidden', '?')}  "
      f"rnn_layers : {saved_args.get('rnn_layers', '?')}")
print(f"use_freq_attn : {crn.use_freq_attn}")
print()

body = CRNBodyForONNX(crn)
body.eval()

# ── verify FreqAttnForONNX matches original FreqAttention ─────────────────────
if crn.use_freq_attn:
    print("Verifying FreqAttnForONNX matches original ...")
    fa_in = torch.randn(1, 512, 9, 20)  # (B, C, Fb, T) at bottleneck
    with torch.no_grad():
        ref_fa  = crn.freq_attn(fa_in)
        new_fa  = body.freq_attn(fa_in)
    diff_fa = float((ref_fa - new_fa).abs().max())
    print(f"  Max |delta| : {diff_fa:.2e}")
    if diff_fa > 1e-5:
        print("  ERROR - FreqAttnForONNX diverges from original")
        sys.exit(1)
    print("  OK")
    print()

# ── verify full wrapper matches full CRN output ───────────────────────────────
print("Verifying CRNBodyForONNX matches full CRN output ...")
window    = torch.hann_window(N_FFT)
dummy_w   = torch.randn(1, T_WAVE)
with torch.no_grad():
    ref_enhanced = crn(dummy_w)
    spec    = torch.stft(dummy_w, n_fft=N_FFT, hop_length=HOP,
                         win_length=N_FFT, window=window,
                         return_complex=True, center=True)
    x_stft  = torch.stack([spec.real, spec.imag], dim=1)   # (1, 2, 257, 375)
    noise_x = body(x_stft)
    ns      = torch.complex(noise_x[:, 0], noise_x[:, 1])
    n_wave  = torch.istft(ns, n_fft=N_FFT, hop_length=HOP,
                          win_length=N_FFT, window=window,
                          length=T_WAVE, center=True)
    wrapper_enhanced = dummy_w - n_wave

diff_full = float((ref_enhanced - wrapper_enhanced).abs().max())
print(f"  Max |delta| vs full CRN : {diff_full:.2e}")
if diff_full > 1e-5:
    print("  ERROR - wrapper diverges from full CRN")
    sys.exit(1)
print("  OK")
print()

# ── ONNX export ───────────────────────────────────────────────────────────────
dummy_stft = torch.zeros(1, 2, F_BINS, T_STFT, dtype=torch.float32)
print(f"Input shape   : (1, 2, {F_BINS}, {T_STFT})  [batch, re/im, freq, stft_frames]")

def export_at_opset(out_path: Path, opset: int) -> bool:
    print(f"Exporting opset {opset} -> {out_path}")
    t0 = time.perf_counter()
    ok = False
    for cf in (True, False):
        label = "" if cf else " (do_constant_folding=False)"
        try:
            with torch.no_grad():
                torch.onnx.export(
                    body,
                    dummy_stft,
                    str(out_path),
                    input_names=["x"],
                    output_names=["noise_x"],
                    dynamic_axes={"x": {3: "T_stft"}, "noise_x": {3: "T_stft"}},
                    opset_version=opset,
                    do_constant_folding=cf,
                    dynamo=False,
                    verbose=False,
                )
            ok = True
            print(f"  Exported{label}")
            break
        except Exception as e:
            print(f"  FAILED{label}: {e.__class__.__name__}: {str(e)[:300]}")
    ms = (time.perf_counter() - t0) * 1000
    if ok:
        print(f"  Time : {ms:.0f} ms  |  Size : {out_path.stat().st_size / 1e6:.1f} MB")
    print()
    return ok

if not export_at_opset(_OUT, 17):
    sys.exit(1)
if not export_at_opset(_OUT15, 15):
    sys.exit(1)

# ── ONNX Runtime verification (CPU) ───────────────────────────────────────────
try:
    import onnxruntime as ort
    inp_np = dummy_stft.numpy()
    with torch.no_grad():
        ref_np = body(dummy_stft).numpy()
    for label, path in [("opset 17", _OUT), ("opset 15", _OUT15)]:
        print(f"Verifying {label} ONNX output against PyTorch body (CPU ORT) ...")
        sess    = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(["noise_x"], {"x": inp_np})[0]
        max_diff = float(np.abs(ref_np - ort_out).max())
        status   = "PASS" if max_diff < 1e-3 else ("WARN" if max_diff < 1e-2 else "FAIL")
        print(f"  Max |delta| PyTorch vs ORT : {max_diff:.2e}  [{status}]")
except Exception as e:
    print(f"  ORT verification skipped: {e}")

print()
for label, path in [("opset 17", _OUT), ("opset 15", _OUT15)]:
    print("=" * 64)
    print(f"  ONNX file ({label}) : {path}")
    print(f"  Size        : {path.stat().st_size / 1e6:.1f} MB")
    print(f"  Input  'x'      : (1, 2, {F_BINS}, {T_STFT})  float32  [re, im stacked]")
    print(f"  Output 'noise_x': (1, 2, {F_BINS}, {T_STFT})  float32  [re, im stacked]")
print("=" * 64)
print()
print("  Jetson usage per chunk:")
print("    stft   = np.fft.rfft(...)                     # CPU numpy")
print("    x_in   = np.stack([stft.real, stft.imag])[None]")
print("    noise  = ort_sess.run(['noise_x'], {'x': x_in})[0]  # GPU")
print("    n_wave = scipy.signal.istft(noise[0,0]+1j*noise[0,1], ...)  # CPU")
print("    enhanced = noisy_wave - n_wave")
print("=" * 64)

# ── Fixed-shape exports for context sweep ────────────────────────────────────
# Required by profile_latency.py on Jetson. Two separate ONNX files, one per
# context size, with no dynamic axes. TensorRT can optimise these at compile time.
#
# T_STFT formula (matches torch.stft center=True, hop=128):
#   T_STFT = T_WAVE // HOP + 1  where T_WAVE = ctx_n + CHUNK_SIZE
#   ctx_n=8000: T_WAVE=8256  → T_STFT=65   (500 ms — START HERE)
#   ctx_n=3200: T_WAVE=3456  → T_STFT=28   (200 ms — edge case)

print()
print("── Fixed-shape exports (for context sweep) ──────────────────────")

SWEEP_CONFIGS = [
    (47616, 375, "3000ms"),  # Stage-2 baseline equivalent (full 3s context)
    (16000, 128, "1000ms"),  # backup if 500ms quality degrades
    (8000,  65,  "500ms"),   # recommended start (LFBypass fully saturated)
    (3200,  28,  "200ms"),   # edge case / curiosity
]

for ctx_n, t_stft, label in SWEEP_CONFIGS:
    out_fixed   = _AI_DIR / f"crn_bilstm_{label}.onnx"
    dummy_fixed = torch.zeros(1, 2, F_BINS, t_stft, dtype=torch.float32)
    print(f"Exporting {label}  ctx={ctx_n}  T_STFT={t_stft} ...", flush=True)

    exported = False
    for cf in (True, False):
        tag = "" if cf else " (do_constant_folding=False)"
        print(f"  trying opset=15{tag} ...", end="  ", flush=True)
        try:
            with torch.no_grad():
                torch.onnx.export(
                    body,
                    dummy_fixed,
                    str(out_fixed),
                    input_names=["x"],
                    output_names=["noise_x"],
                    # No dynamic_axes → strictly fixed (1, 2, 257, t_stft) shapes
                    opset_version=15,       # IR version 8, compatible with ORT 1.12.1
                    do_constant_folding=cf,
                    dynamo=False,           # force legacy exporter — dynamo path writes opset 18+
                    verbose=False,
                )
            size_mb = out_fixed.stat().st_size / 1e6
            if size_mb < 10:
                # Opset 17+ dynamo export silently produces a tiny stub (~0.3 MB).
                # Treat anything under 10 MB as a failed export.
                print(f"CORRUPT ({size_mb:.1f} MB — unexpected tiny file, retrying)")
                out_fixed.unlink(missing_ok=True)
                continue
            print(f"OK  ->  {out_fixed.name}  ({size_mb:.1f} MB)")
            exported = True
            break
        except Exception as exc:
            print(f"FAILED: {exc.__class__.__name__}: {str(exc)[:200]}")

    if not exported:
        print(f"  ERROR: could not export {label} — see messages above")

print()
print("scp commands:")
for _, _, label in SWEEP_CONFIGS:
    name = f"crn_bilstm_{label}.onnx"
    print(f"  scp {name} bachkukkik@192.168.55.1:~/ANC_Hardware_Test/simulation/")
print()
print("Then on Xavier:")
print("  bash ~/AI_builders/build_trt_engines.sh")
