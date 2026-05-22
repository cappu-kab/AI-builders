#!/usr/bin/env python3
"""
ANC Phase 2 — Future-Sample Predictor (LPC + Lightweight Causal TCN)
=====================================================================
Predicts the next 80 samples (5 ms @ 16 kHz) of low-frequency engine noise so
the ANC pipeline can compensate for the CRN's processing latency.

Two backends:
  1. LPC  — classic autoregressive predictor.  Very fast, CPU-only, no training.
  2. TCN  — tiny dilated causal convnet (~5 k params, ~21 kB).  Jetson-friendly.

Usage:
    python predictor.py            # train TCN + evaluate both backends + plot

Exports (consumed by predictor_phase_eval.py):
    SAMPLE_RATE, INPUT_LEN, PREDICT_LEN, LPC_ORDER, DEVICE
    generate_engine_noise, make_training_pairs
    predict_lpc, LightweightTCN, train_tcn, predict_tcn
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# Time-based constants — sample-rate-independent physical horizons.
#
# The ANC pipeline is defined in MILLISECONDS, not sample counts, so the model
# stays physically correct regardless of the mic's native rate.  If you ever
# switch TARGET_SR (e.g. to 8 kHz on a tighter MCU), the four MS knobs below
# are the only things that need to change — CHUNK_SIZE / INPUT_LEN /
# PREDICT_LEN follow automatically.
# ══════════════════════════════════════════════════════════════════════════════
TARGET_SR    = 16_000        # Internal pipeline rate.  Optimal for <200 Hz ANC
                             # (Nyquist 8 kHz; any noise source above is filtered
                             # by the CRN upstream anyway).
CHUNK_MS     = 16.0          # Hardware I2S block size — one PortAudio callback.
CONTEXT_MS   = 64.0          # Model receptive-field horizon (= 4 hardware chunks).
PREDICT_MS   =  5.0          # Look-ahead — how far the model forecasts.


def _ms_to_samples(ms: float, sr: int = TARGET_SR) -> int:
    """Round-to-nearest conversion from milliseconds to integer sample count."""
    return int(round(ms * sr / 1000.0))


CHUNK_SIZE   = _ms_to_samples(CHUNK_MS)      # 256 smp @ 16 kHz   (hardware chunk)
INPUT_LEN    = _ms_to_samples(CONTEXT_MS)    # 1024 smp @ 16 kHz  (model context)
PREDICT_LEN  = _ms_to_samples(PREDICT_MS)    # 80  smp @ 16 kHz   (look-ahead)

# Back-compat aliases — keep existing code and predictor_phase_eval.py working.
SAMPLE_RATE     = TARGET_SR
STREAMING_CHUNK = CHUNK_SIZE

LPC_ORDER       = 24
TCN_CHANNELS    = 32
# Dilations chosen so the receptive field matches INPUT_LEN almost exactly.
# At CONTEXT_MS=64 ms / TARGET_SR=16 kHz that's dilations 1..256 → RF=1023 smp.
# If you change TARGET_SR or CONTEXT_MS materially, re-tune this tuple so RF≈INPUT_LEN.
TCN_DILATIONS   = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED            = 42


def tcn_receptive_field(dilations: Tuple[int, ...], kernel_size: int = 3) -> int:
    """RF for a stack of dilated causal convs (no residual)."""
    return 1 + (kernel_size - 1) * sum(dilations)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data — synthetic engine noise (swap with sf.read('your.wav') for real data)
# ══════════════════════════════════════════════════════════════════════════════

# def generate_engine_noise(duration_s: float = 30.0,
#                           sr: int = SAMPLE_RATE,
#                           seed: int = SEED) -> np.ndarray:
#     """
#     Multi-component low-frequency engine noise, matching file_simulator.py.
#     Replace with a WAV load if you want to train on real recordings:
#         import soundfile as sf
#         x, sr_in = sf.read('clean_lowfreq.wav')
#     """
#     rng = np.random.default_rng(seed)
#     n   = int(duration_s * sr)
#     t   = np.arange(n) / sr
#     x = (
#         0.38 * np.sin(2 * np.pi *  75 * t) +   # fundamental
#         0.22 * np.sin(2 * np.pi * 150 * t) +   # 2nd harmonic
#         0.14 * np.sin(2 * np.pi * 100 * t) +   # secondary rumble
#         0.10 * np.sin(2 * np.pi *  60 * t) +   # drivetrain hum
#         0.12 * np.sin(2 * np.pi * 320 * t) +   # intake harmonic
#         0.08 * np.sin(2 * np.pi * 480 * t) +   # body resonance
#         0.032 * rng.standard_normal(n)
#     ).astype(np.float32)
#     return (x / 1.1 * 0.85).astype(np.float32)
import numpy as np
import soundfile as sf
import librosa
import os
from scipy.signal import cheby2, sosfilt, sosfilt_zi

# =======================================================
# 1. ฟังก์ชันโหลดไฟล์เสียง + Auto-Resample เป็น TARGET_SR
# =======================================================
def generate_engine_noise(duration_s: float | None = None,
                          sr: int = TARGET_SR,
                          seed: int = 42,
                          file_path: str = '../AC_noise.wav',
                          skip_seconds: float = 5.0) -> np.ndarray:
    """
    โหลด noise_sound.wav, แปลงเป็น mono, AUTO-RESAMPLE ไปที่ `sr` (default = TARGET_SR),
    ข้าม `skip_seconds` วินาทีแรก (silence preamble), แล้วคืนค่าทั้งหมดที่เหลือ.

    การ resample ใช้ librosa ดังนั้น native SR ใดๆ ก็ได้ (16 / 22.05 / 44.1 / 48 kHz, ฯลฯ)
    — โมเดลภายในยังเห็น TARGET_SR เสมอ.  Physical time-horizon (CHUNK_MS, CONTEXT_MS,
    PREDICT_MS) จะคงที่เพราะ derive จาก ms ไม่ใช่ sample count.

    Args:
        duration_s    None  → ใช้ทุก sample หลังจากข้าม skip_seconds (default: เต็ม)
                      float → cap ที่ความยาวนี้
        sr            target sample rate (default = TARGET_SR = 16 kHz)
        skip_seconds  เวลาเงียบที่ต้นคลิป (default 5.0 s)
        file_path     path ไปยังไฟล์ WAV (default = '../noise_sound.wav')
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ไม่พบไฟล์: {file_path}")

    # ── โหลดที่ native SR ก่อน (sr=None) แล้วเช็คว่าต้อง resample ไหม ──────
    x, native_sr = librosa.load(file_path, sr=None, mono=True)
    if native_sr != sr:
        print(f"  [resample]  native = {native_sr} Hz  →  target = {sr} Hz "
              f"(librosa kaiser_best)")
        x = librosa.resample(x, orig_sr=native_sr, target_sr=sr)
    else:
        print(f"  [SR ok]     native = {native_sr} Hz (= TARGET_SR — no resample)")

    # ── ข้ามวินาทีแรก (silence preamble) — fixed offset ───────────────────
    skip_smp = int(skip_seconds * sr)
    if len(x) > skip_smp:
        x = x[skip_smp:]
    else:
        raise ValueError(f"ไฟล์สั้นกว่า {skip_seconds} วินาที — ตัดทิ้งหมดเลย")

    # ── cap ความยาวเฉพาะเมื่อ caller ระบุ duration_s (default = ใช้เต็ม) ──
    if duration_s is not None:
        target_len = int(duration_s * sr)
        if len(x) > target_len:
            x = x[:target_len]
        elif len(x) < target_len:
            repeats = (target_len // len(x)) + 1
            x = np.tile(x, repeats)[:target_len]

    return x.astype(np.float32)

# =======================================================
# 2. ฟังก์ชันหั่นข้อมูล (ตัวที่หายไปจน Error)
# =======================================================
def make_training_pairs(signal: np.ndarray,
                        hop: int = 64,
                        chunk_size: int = INPUT_LEN,
                        predict_size: int = PREDICT_LEN):
    """
    ฟังก์ชันเลื่อนหน้าต่าง (Sliding-window) เพื่อสร้าง Input และ Target.
    Defaults ติดตาม INPUT_LEN / PREDICT_LEN ที่ derive จาก ms ดังนั้นถ้าเปลี่ยน
    TARGET_SR หรือ CONTEXT_MS แล้ว defaults จะปรับให้อัตโนมัติ.
    """
    X, Y = [], []
    # ลูปหั่นข้อมูลทีละ hop
    for i in range(0, len(signal) - chunk_size - predict_size, hop):
        X.append(signal[i : i + chunk_size])                           # อดีต 256 samples
        Y.append(signal[i + chunk_size : i + chunk_size + predict_size]) # อนาคต 80 samples ที่ต้องทาย

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# =======================================================
# 3. ฟิลเตอร์โลว์พาส (เตรียม training data ให้สะอาด)
# =======================================================
def low_pass_filter(signal: np.ndarray,
                    cutoff_hz: float = 180.0,
                    sr: int = TARGET_SR,
                    order: int = 8,
                    stopband_db: float = 60.0) -> np.ndarray:
    """
    STRICTLY CAUSAL low-pass filter — Chebyshev Type II, 8th-order,
    single forward pass via `sosfilt`.  No future samples are ever
    consulted; the output at time t depends only on input at times ≤ t.

    Why causal-only (the user's "no cheating" directive)
    -----------------------------------------------------
    The previous version used `sosfiltfilt` (forward + backward) for zero
    phase shift, but that is NON-CAUSAL:

        filtered[t]  depends on  raw[t-N], ..., raw[t-1], raw[t], raw[t+1], ..., raw[t+N]
                                                                ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                                                                future samples — illegal in
                                                                real-time DSP

    Applied to TRAINING data this is silent data leakage in two ways:
      1. Each target sample Y[i][m] is computed from raw samples in the
         future of position i+1024+m — so the model is matching a signal
         that "smells of" the future it's supposed to predict.
      2. The backward pass smears information across the train/test
         boundary: filtered samples at the end of train_sig depend on
         raw samples at the start of test_sig, and vice versa.

    Net effect: training corr inflates to ~+0.999 even when the model
    has learned nothing useful, because it ends up implicitly learning
    the bidirectional filter's response rather than the noise dynamics.
    On hardware (where filtfilt cannot exist), accuracy collapses.

    What this causal version does instead
    --------------------------------------
    • One forward pass only — `sosfilt`.  Identical to what the ADC + DSP
      block on Jetson Nano would do in real time.
    • Initial state set to the steady-state response for the first input
      sample via `sosfilt_zi`, so the filter starts at its asymptote
      instead of ringing up from zero (otherwise the first ~order/cutoff
      samples are corrupted by a startup transient).
    • Introduces realistic group delay (~order/(2·cutoff_hz) seconds —
      tens of samples).  The TCN has to learn to compensate for this
      delay during training, just as it would on real hardware.
    • Cheby II + SOS form preserved: monotonic passband, equiripple
      −60 dB stopband, numerically stable at high orders.

    Args:
        cutoff_hz    −3 dB corner.  180 Hz hugs the sub-200 Hz ANC band.
        order        Filter order — single-pass roll-off ≈ 6·order dB/oct
                     (8th order → ~48 dB/octave).
        stopband_db  Minimum stopband attenuation (default 60 dB).

    Returns:
        Causally filtered signal, same shape as input, float32.
    """
    nyquist = sr / 2.0
    if not (0 < cutoff_hz < nyquist):
        raise ValueError(f"cutoff_hz must be in (0, {nyquist}); got {cutoff_hz}")
    sos = cheby2(order, stopband_db, cutoff_hz / nyquist,
                 btype='low', output='sos')

    # Warm-start initial conditions: steady-state response for an input
    # held at signal[0].  Without this, the first ~50–100 samples ramp up
    # from zero (a non-physical transient that the model would then have
    # to either ignore or — worse — over-fit).
    zi = sosfilt_zi(sos) * float(signal[0])
    filtered, _ = sosfilt(sos, signal, zi=zi)
    return filtered.astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# 2. LPC predictor — Levinson-Durbin + iterative AR extrapolation
# ══════════════════════════════════════════════════════════════════════════════

def _lpc_levinson(signal: np.ndarray, order: int) -> np.ndarray:
    """Autocorrelation method + Levinson-Durbin recursion.

    Returns coefficients a[0..p] such that  x[n] + Σ a[k]·x[n-k] ≈ 0  (k=1..p),
    i.e. the forward predictor is  x[n] = -Σ a[k]·x[n-k].
    """
    sig = signal - signal.mean()
    R   = np.array(
        [float(np.dot(sig[:len(sig) - k], sig[k:])) for k in range(order + 1)],
        dtype=np.float64,
    )
    a    = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    e    = R[0] + 1e-12
    for i in range(1, order + 1):
        k_i        = -(R[i] + float(np.dot(a[1:i], R[i - 1:0:-1]))) / e
        a_new      = a.copy()
        a_new[i]   = k_i
        a_new[1:i] = a[1:i] + k_i * a[i - 1:0:-1]
        a = a_new
        e *= (1.0 - k_i ** 2)
        if e <= 0:
            break
    return a


def predict_lpc(history: np.ndarray,
                n_predict: int = PREDICT_LEN,
                order: int = LPC_ORDER) -> np.ndarray:
    """Forward-extrapolate `n_predict` samples by iterating the LPC recursion."""
    a      = _lpc_levinson(history, order)
    coeffs = a[1:]
    buf    = np.concatenate([
        history[-order:].astype(np.float64),
        np.zeros(n_predict,  dtype=np.float64),
    ])
    for i in range(n_predict):
        past = buf[i : order + i][::-1]              # x[t-1], x[t-2], ..., x[t-p]
        buf[order + i] = -float(np.dot(coeffs, past))
    return buf[order:].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Lightweight causal TCN — ~5 k params, ~21 kB on disk
# ══════════════════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    """1-D convolution padded only on the LEFT — no look-ahead leakage."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class LightweightTCN(nn.Module):
    """
    Lightweight causal TCN with built-in z-score normalisation.

    Architecture (channels=32, dilations=(1,2,4,8,16,32,64)):
        7 dilated causal-conv layers (kernel=3)
        receptive field = 1 + (3-1)·(1+2+4+8+16+32+64) = 255 smp ≈ 16 ms
        ReLU activations, no BatchNorm (latency-sensitive)
        Last time-step feature  →  Linear(C→4C)  →  GELU  →  Linear(4C→predict_len)

    Normalisation:
        Input is standardised  (x - x_mean) / x_std  inside forward().
        Output is un-standardised back to original amplitude before returning.
        Stats are stored as buffers so they travel with state_dict.
        Call `model.fit_stats(X_train, Y_train)` BEFORE training.

        Why: real noise sits at ±0.2 (std≈0.05–0.08), which is ~14× smaller than
        what Kaiming init assumes, so gradients are ~14× too small and MSE
        plateaus at "predict the mean".  Z-scoring fixes the gradient scale.
    """
    def __init__(self,
                 input_len:   int = INPUT_LEN,
                 predict_len: int = PREDICT_LEN,
                 channels:    int = TCN_CHANNELS,
                 dilations: Tuple[int, ...] = TCN_DILATIONS,
                 head_expansion: int = 4,
                 pyramid_lags: Tuple[int, ...] = (1, 32, 256)):
        super().__init__()
        layers, in_ch = [], 1
        for d in dilations:
            layers += [
                CausalConv1d(in_ch, channels, kernel_size=3, dilation=d),
                nn.ReLU(inplace=True),
            ]
            in_ch = channels
        self.tcn  = nn.Sequential(*layers)

        # ── Temporal pyramid pooling for the head input ─────────────────────
        # Instead of taking only the last time-step feature (h[:,:,-1]) and
        # throwing away the other 1023, we concatenate the last sample with
        # mean-pooled features at MULTIPLE physical time scales:
        #   lag=1   →  instantaneous   (   62.5 µs)
        #   lag=32  →  sub-cycle       (    2.0 ms)
        #   lag=256 →  one fundamental (   16.0 ms — ~1 cycle at 60 Hz)
        # This gives the decoder explicit short/mid/long-range context for
        # ~8 k extra params; biggest single win once the data is clean.
        for lag in pyramid_lags:
            if lag < 1 or lag > input_len:
                raise ValueError(
                    f"pyramid_lags must be in [1, input_len={input_len}]; got {lag}"
                )
        self.pyramid_lags = tuple(pyramid_lags)

        hidden = channels * head_expansion
        head_in = channels * len(pyramid_lags)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, predict_len),
        )

        # Normalisation buffers — identity until fit_stats() is called.
        self.register_buffer("x_mean", torch.tensor(0.0))
        self.register_buffer("x_std",  torch.tensor(1.0))
        self.register_buffer("y_mean", torch.tensor(0.0))
        self.register_buffer("y_std",  torch.tensor(1.0))

        self.input_len   = input_len
        self.predict_len = predict_len

    @torch.no_grad()
    def fit_stats(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Compute global z-score stats from training data and store as buffers."""
        dev = next(self.parameters()).device
        self.x_mean = torch.tensor(float(X.mean()), device=dev)
        self.x_std  = torch.tensor(float(X.std() + 1e-8), device=dev)
        self.y_mean = torch.tensor(float(Y.mean()), device=dev)
        self.y_std  = torch.tensor(float(Y.std() + 1e-8), device=dev)

    def forward(self, x: torch.Tensor,
                return_normalised: bool = False) -> torch.Tensor:
        """
        x:  (B, T) or (B, 1, T) — RAW amplitude (any scale).
        Returns predicted future samples in RAW amplitude.
        When `return_normalised=True`, returns z-scored output instead — used
        by the training loop so the loss is computed in standardised space.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)                              # (B, 1, T)

        # ── Standardise ──────────────────────────────────────────────────────
        x = (x - self.x_mean) / self.x_std

        h = self.tcn(x)                                     # (B, C, T)

        # ── Temporal pyramid pooling ─────────────────────────────────────────
        pool_feats = []
        for lag in self.pyramid_lags:
            if lag == 1:
                pool_feats.append(h[:, :, -1])              # (B, C) instantaneous
            else:
                pool_feats.append(h[:, :, -lag:].mean(dim=-1))   # (B, C) avg
        combined = torch.cat(pool_feats, dim=-1)            # (B, C * n_lags)

        y_n = self.head(combined)                           # (B, predict_len), z-scored

        if return_normalised:
            return y_n
        return y_n * self.y_std + self.y_mean               # un-standardise


def _composite_loss(pred_n: torch.Tensor,
                    target_n: torch.Tensor,
                    delta_weight:  float = 2.0,
                    delta2_weight: float = 1.0) -> torch.Tensor:
    """
    Multi-scale temporal loss in STANDARDISED space (z-scored args).

        L =       MSE(pred,   target)            ← absolute level
            +  α · MSE(Δpred,  Δtarget)          ← slope (1st derivative)
            +  β · MSE(Δ²pred, Δ²target)         ← curvature (2nd derivative)

    Why three terms instead of two
    -------------------------------
    A trough is where the slope reverses (Δ→0) AND the curvature is large
    (Δ²<0 then Δ²>0).  A pure-slope penalty can't distinguish "predicted a
    flat valley" from "predicted a deep one" — both have low |Δ| there.
    The Δ² term forces the model to commit to the curvature of the
    extremum, which is exactly what pushes correlation past +0.90 on
    harmonic noise that keeps plunging into deep troughs.
    """
    mse_t   = F.mse_loss(pred_n, target_n)

    d_pred  = pred_n[:,   1:] - pred_n[:,   :-1]
    d_targ  = target_n[:, 1:] - target_n[:, :-1]
    mse_d1  = F.mse_loss(d_pred, d_targ)

    d2_pred = d_pred[:,  1:] - d_pred[:,  :-1]
    d2_targ = d_targ[:,  1:] - d_targ[:,  :-1]
    mse_d2  = F.mse_loss(d2_pred, d2_targ)

    return mse_t + delta_weight * mse_d1 + delta2_weight * mse_d2


def train_tcn(model: LightweightTCN,
              X: np.ndarray, Y: np.ndarray,
              epochs:        int   = 100,
              batch_size:    int   = 64,
              lr:            float = 5e-4,
              delta_weight:  float = 2.0,
              delta2_weight: float = 1.0,
              verbose:       bool  = True,
              fit_stats:     bool  = True) -> List[float]:
    """
    Train the TCN with:
      • Global z-score normalisation (input + target) — fixes plateau on
        quiet real-noise data (std ≈ 0.07 vs Kaiming-init assumption ≈ 1.0).
      • Composite loss = MSE + α·MSE(Δ) + β·MSE(Δ²) — level + slope + curvature,
        which is what catches the deep troughs of harmonic AC-compressor noise
        that pure MSE flattens through.
      • CosineAnnealingLR — smooth decay; avoids hand-tuning LR steps.

    Returns the per-epoch loss history (loss is in z-scored units, so the
    magnitude itself is no longer directly comparable to a raw-MSE plot).
    """
    model.to(DEVICE)

    if fit_stats:
        model.fit_stats(X, Y)
        if verbose:
            print(f"  z-score stats:  x_mean={float(model.x_mean):+.4f}  "
                  f"x_std={float(model.x_std):.4f}  "
                  f"y_mean={float(model.y_mean):+.4f}  "
                  f"y_std={float(model.y_std):.4f}")

    model.train()
    X_t = torch.from_numpy(X).to(DEVICE)
    Y_t = torch.from_numpy(Y).to(DEVICE)
    # Pre-standardise targets once so we don't redo it per batch.
    Y_n = (Y_t - model.y_mean) / model.y_std
    n   = X_t.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history: List[float] = []
    for ep in range(epochs):
        perm  = torch.randperm(n, device=DEVICE)
        total = 0.0
        for s in range(0, n, batch_size):
            idx     = perm[s : s + batch_size]
            yhat_n  = model(X_t[idx], return_normalised=True)
            loss    = _composite_loss(yhat_n, Y_n[idx],
                                      delta_weight=delta_weight,
                                      delta2_weight=delta2_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        sch.step()
        ep_loss = total / (n * model.predict_len)
        history.append(ep_loss)
        if verbose and ((ep + 1) % 5 == 0 or ep == 0):
            print(f"  epoch {ep + 1:3d}/{epochs}   loss = {ep_loss:.6f}   "
                  f"lr = {sch.get_last_lr()[0]:.2e}")
    model.eval()
    return history


def predict_tcn(model: LightweightTCN, history: np.ndarray) -> np.ndarray:
    """One-shot inference: (input_len,) → (predict_len,).

    Runs on whichever device the model already lives on, so callers stay free
    to mix .to(DEVICE) / .cpu() without re-engineering this helper.
    """
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(history.astype(np.float32))[None, :].to(dev)
        y = model(x).cpu().numpy()[0]
    return y.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Streaming ring buffer  (Jetson runtime topology)
# ══════════════════════════════════════════════════════════════════════════════

class StreamingRingBuffer:
    """
    Sliding history of `capacity` samples, fed by chunks of any size ≤ capacity.

    This is the bridge between the I2S mic (which delivers STREAMING_CHUNK
    samples every 16 ms) and the model (which wants to see a longer
    INPUT_LEN-sample context window).  The buffer lives in HOST memory; the
    mic callback only ever pushes 256 samples, while the model sees the
    rolling 1024-sample snapshot — there is NO increase in hardware-side
    latency from expanding the model context.

    Cost per push: one numpy memmove of `capacity − k` floats (≈ 4 kB at
    capacity=1024) plus a `k`-sample copy.  On Jetson Orin Nano this is
    well under 50 µs — negligible compared to the 16 ms chunk period.
    """
    def __init__(self, capacity: int):
        self.capacity   = capacity
        self._buf       = np.zeros(capacity, dtype=np.float32)
        self._n_filled  = 0

    def push(self, chunk: np.ndarray) -> None:
        """Append `chunk` to the right; oldest samples fall off the left."""
        k = len(chunk)
        if k >= self.capacity:
            self._buf[:]   = chunk[-self.capacity:]
            self._n_filled = self.capacity
            return
        # Shift left by k samples, then write the new chunk at the tail.
        self._buf[:-k] = self._buf[k:]
        self._buf[-k:] = chunk
        self._n_filled = min(self.capacity, self._n_filled + k)

    def snapshot(self) -> np.ndarray:
        """Contiguous copy of the current `capacity`-sample window."""
        return self._buf.copy()

    @property
    def is_full(self) -> bool:
        return self._n_filled >= self.capacity

    def reset(self) -> None:
        self._buf.fill(0.0)
        self._n_filled = 0


def streaming_inference_demo(model: LightweightTCN,
                             test_signal: np.ndarray,
                             n_show: int = 3) -> dict:
    """
    Simulate the Jetson runtime exactly as it will run on hardware:

        for every CHUNK_SIZE-sample block from the mic:
            ring.push(chunk)                  # ~µs   — host-side bookkeeping
            if ring.is_full:
                snapshot = ring.snapshot()    # INPUT_LEN smp = 64 ms context
                pred = model(snapshot)        # → next PREDICT_LEN smp (5 ms ahead)

    Reports inference-latency statistics over the WHOLE test signal and plots
    `n_show` representative (context, actual_future, prediction) triplets.

    Deployment note — wiring PortAudio / sounddevice on the Jetson Orin Nano
    ────────────────────────────────────────────────────────────────────────
    The I2S mic (INMP441) often captures at 48 kHz natively, but the model
    expects TARGET_SR.  IMPORTANT: open the audio stream at TARGET_SR — not
    the mic's native rate — so every value below stays in physical units:

        import sounddevice as sd

        sd.InputStream(
            device      = 'plughw:1,0',    # 'plug:' prefix lets ALSA resample
            samplerate  = TARGET_SR,       # 16 000 — model's expected rate
            blocksize   = CHUNK_SIZE,      # exactly one model chunk per callback
            channels    = 1,
            dtype       = 'float32',
            callback    = lambda data, *_: ring.push(data[:, 0]),
        )

    Why this matters:
      • If you open at 48 kHz and forget to resample, the model sees 3× more
        samples per `CHUNK_MS` window — physical time-horizon is wrong, the
        TCN's RF no longer covers a sub-200 Hz cycle, and prediction collapses.
      • Letting ALSA resample (via the `plug:` device wrapper) is essentially
        free on Jetson — the kernel does it in the driver path with no CPU
        cost visible to Python.
      • If you'd rather resample in software (more control, slightly higher
        CPU), open at native SR and call `librosa.resample` inside the
        callback before pushing into the ring buffer.
    """
    ring             = StreamingRingBuffer(INPUT_LEN)
    chunks_to_fill   = INPUT_LEN // STREAMING_CHUNK
    n_chunks         = (len(test_signal) - PREDICT_LEN) // STREAMING_CHUNK
    chunk_period_ms  = STREAMING_CHUNK / SAMPLE_RATE * 1000.0

    print(f"  hardware chunk : {STREAMING_CHUNK} smp "
          f"({chunk_period_ms:.1f} ms / arrival)")
    print(f"  model context  : {INPUT_LEN} smp "
          f"({INPUT_LEN / SAMPLE_RATE * 1000:.0f} ms = "
          f"{chunks_to_fill} chunks of history)")
    print(f"  prediction     : {PREDICT_LEN} smp "
          f"({PREDICT_LEN / SAMPLE_RATE * 1000:.1f} ms ahead)")
    print(f"  ring warmup    : {chunks_to_fill} chunks "
          f"({chunks_to_fill * chunk_period_ms:.0f} ms) before first valid prediction")

    latencies: List[float] = []
    captures:  List[tuple] = []     # (chunk_idx, snapshot, actual, pred, corr)

    for i in range(n_chunks):
        s     = i * STREAMING_CHUNK
        chunk = test_signal[s : s + STREAMING_CHUNK]
        ring.push(chunk)

        if not ring.is_full:
            continue                                      # still warming up

        actual_start = s + STREAMING_CHUNK                # immediately after this chunk
        if actual_start + PREDICT_LEN > len(test_signal):
            break
        actual = test_signal[actual_start : actual_start + PREDICT_LEN]

        snapshot = ring.snapshot()
        t0   = time.perf_counter()
        pred = predict_tcn(model, snapshot)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        corr = float(np.corrcoef(pred, actual)[0, 1]) if actual.std() > 0 else 0.0
        captures.append((i, snapshot, actual, pred, corr))

    lat = np.array(latencies)
    corrs = np.array([c[4] for c in captures])

    print(f"\n  predicted {len(captures)} blocks "
          f"(corr: mean={corrs.mean():+.3f}  median={np.median(corrs):+.3f}  "
          f"p10={np.percentile(corrs,10):+.3f}  p90={np.percentile(corrs,90):+.3f})")
    print(f"  inference latency: mean={lat.mean():.3f} ms  "
          f"p95={np.percentile(lat,95):.3f} ms  max={lat.max():.3f} ms  "
          f"(budget = {chunk_period_ms:.1f} ms)")
    if lat.max() > chunk_period_ms:
        print(f"  WARNING: peak latency exceeds the {chunk_period_ms:.1f} ms "
              f"chunk period — inference would back up on Jetson.")
    else:
        print(f"  OK: peak latency uses "
              f"{lat.max() / chunk_period_ms * 100:.1f}% of the chunk-period budget.")

    # Pick representative examples (worst / median / best correlation).
    if len(captures) >= n_show:
        order = np.argsort(corrs)
        if n_show == 3:
            picks = [order[0], order[len(order) // 2], order[-1]]
            tags  = ["WORST corr", "MEDIAN corr", "BEST corr"]
        else:
            idxs  = np.linspace(0, len(order) - 1, n_show).astype(int)
            picks = order[idxs].tolist()
            tags  = [f"#{p}" for p in picks]
        plot_streaming_examples([captures[p] for p in picks], tags)

    return dict(
        n_predictions = len(captures),
        latency_ms    = lat,
        corr          = corrs,
    )


def plot_streaming_examples(captures: List[tuple], tags: List[str]) -> None:
    """One row per capture: full 1024-sample context + prediction overlay."""
    n = len(captures)
    fig, axes = plt.subplots(n, 1, figsize=(12.0, 2.7 * n), sharex=False)
    if n == 1:
        axes = [axes]

    chunk_ms = STREAMING_CHUNK / SAMPLE_RATE * 1000.0

    for ax, (chunk_idx, snapshot, actual, pred, corr), tag in zip(axes, captures, tags):
        n_hist = len(snapshot)
        n_pred = len(actual)
        t_hist = np.arange(n_hist) / SAMPLE_RATE * 1000.0
        t_pred = (n_hist + np.arange(n_pred)) / SAMPLE_RATE * 1000.0

        # Shade the 4 hardware chunks so the 256-vs-1024 relationship is obvious.
        chunks_in_ctx = n_hist // STREAMING_CHUNK
        for k in range(chunks_in_ctx):
            ax.axvspan(k * chunk_ms, (k + 1) * chunk_ms,
                       color=('whitesmoke' if k % 2 == 0 else 'gainsboro'),
                       alpha=0.6, zorder=0)
            ax.text((k + 0.5) * chunk_ms, ax.get_ylim()[1],
                    f'chunk {k+1}', ha='center', va='bottom',
                    fontsize=7, color='gray', alpha=0.7)

        ax.plot(t_hist, snapshot, color='dimgray',  lw=0.9,            label='Context (ring buffer)')
        ax.plot(t_pred, actual,   color='black',    lw=1.8,            label='Actual future')
        ax.plot(t_pred, pred,     color='tab:red',  lw=1.3, ls='--',   label='TCN prediction')
        ax.axvline(t_pred[0], color='k', alpha=0.5, lw=0.8)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'[{tag}]  chunk #{chunk_idx}  '
                     f'(corr = {corr:+.3f},  '
                     f'context = {n_hist} smp = '
                     f'{chunks_in_ctx} × {STREAMING_CHUNK})',
                     fontsize=10)
        ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle('Streaming inference — Jetson runtime topology', fontsize=12)
    fig.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4b. Audible ANC ear-test — export 3 WAVs you can listen to side-by-side
# ══════════════════════════════════════════════════════════════════════════════

def export_audio_simulation(model: LightweightTCN,
                            test_signal: np.ndarray,
                            sr: int = TARGET_SR,
                            duration_s: float = 5.0,
                            out_dir: Path | None = None) -> dict:
    """
    Simulate the ANC pipeline over a held-out audio segment and export three
    16-bit PCM WAV files for an audible "ear test":

        sim_1_original_noise.wav     ground-truth target signal (what the mic hears)
        sim_2_anti_noise.wav         TCN prediction × -1         (what the speaker emits)
        sim_3_cancelled_residual.wav original + anti_noise       (what your ear hears)

    Pipeline
    --------
    1.  Walk the test signal with the same StreamingRingBuffer the Jetson uses.
    2.  Hop by PREDICT_LEN (not CHUNK_SIZE) so consecutive prediction windows
        TILE the timeline back-to-back with no gaps — this is what lets the
        three exported WAVs play back at normal real-time speed.
    3.  Strict time alignment: at step i, the input window is
        `test_signal[t-INPUT_LEN : t]` and the prediction MUST be compared
        against `test_signal[t : t+PREDICT_LEN]`, where t = INPUT_LEN + i·PREDICT_LEN.
        Past samples pushed back into the ring are the ACTUAL future
        (open-loop), so the model never gets to "hear" its own prediction —
        identical to live operation on Jetson where the mic always wins.
    4.  Anti-noise is built as  anti = -1.0 × pred.  Physical air mixing is
        approximated as  residual = actual + anti.

    Args:
        model        Trained LightweightTCN (must be in eval mode).
        test_signal  1-D float32 noise (≥ INPUT_LEN + PREDICT_LEN samples).
        sr           Sample rate of `test_signal` (and of the output WAVs).
                     Defaults to TARGET_SR = 16 000.
        duration_s   Length of audio to export (default 5 s; capped at len(test_signal)).
        out_dir      Where to write the WAVs (default: this script's directory).

    Returns:
        dict with file paths and RMS reduction in dB.

    Note on PCM format
    -------------------
    Saved as int16 (clipped to [-1, 1] then scaled by 32767) for maximum
    playback compatibility across OSes/players.  If the cancelled residual
    is so quiet it disappears into int16 quantisation noise, the RMS report
    will still tell you the true reduction.
    """
    from scipy.io import wavfile

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent
    out_dir = Path(out_dir)

    # Clip to requested duration (or all that's available).
    n_keep = min(int(duration_s * sr), len(test_signal))
    sig = test_signal[:n_keep].astype(np.float32)

    if len(sig) < INPUT_LEN + PREDICT_LEN:
        raise ValueError(
            f"test_signal too short for ear test: need ≥ "
            f"{INPUT_LEN + PREDICT_LEN} smp, got {len(sig)}"
        )

    # Number of back-to-back PREDICT_LEN windows we can tile.
    n_preds  = (len(sig) - INPUT_LEN) // PREDICT_LEN
    out_len  = n_preds * PREDICT_LEN
    out_secs = out_len / sr

    print(f"  ear-test sim   : {n_preds} predictions tiled back-to-back "
          f"-> {out_len} smp ({out_secs:.2f} s of audio)")
    print(f"  hop            : {PREDICT_LEN} smp ({PREDICT_LEN/sr*1000:.1f} ms) "
          f"— each prediction perfectly aligned with its actual future window")

    # Pre-allocate contiguous output buffers.
    original_out = np.empty(out_len, dtype=np.float32)
    anti_out     = np.empty(out_len, dtype=np.float32)

    # Ring buffer warm-up: first INPUT_LEN samples of true history.
    ring = StreamingRingBuffer(INPUT_LEN)
    ring.push(sig[:INPUT_LEN])

    for i in range(n_preds):
        # Prediction is for samples [t, t + PREDICT_LEN).
        t = INPUT_LEN + i * PREDICT_LEN

        snapshot = ring.snapshot()                   # last INPUT_LEN smp of true past
        pred     = predict_tcn(model, snapshot)       # → next PREDICT_LEN smp
        actual   = sig[t : t + PREDICT_LEN]           # the matching future window

        slc = slice(i * PREDICT_LEN, (i + 1) * PREDICT_LEN)
        original_out[slc] = actual
        anti_out[slc]     = -1.0 * pred

        # Feed the ACTUAL future (not the prediction) back into the buffer —
        # open-loop, mirroring live operation where the mic delivers truth.
        ring.push(actual)

    residual_out = original_out + anti_out

    # ── Reduction statistics ─────────────────────────────────────────────────
    rms_original = float(np.sqrt(np.mean(original_out ** 2)))
    rms_residual = float(np.sqrt(np.mean(residual_out ** 2)))
    rms_anti     = float(np.sqrt(np.mean(anti_out ** 2)))
    reduction_db = 20.0 * np.log10((rms_original + 1e-12) / (rms_residual + 1e-12))

    print(f"  RMS original   : {rms_original:.5f}")
    print(f"  RMS anti-noise : {rms_anti:.5f}")
    print(f"  RMS residual   : {rms_residual:.5f}")
    print(f"  cancellation   : {reduction_db:+.2f} dB  "
          f"(higher = more noise removed)")

    # ── Save as 16-bit PCM for universal playback ────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, audio in [
        ("sim_1_original_noise.wav",     original_out),
        ("sim_2_anti_noise.wav",         anti_out),
        ("sim_3_cancelled_residual.wav", residual_out),
    ]:
        path  = out_dir / name
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(str(path), sr, pcm16)
        files.append(path)
        print(f"  saved -> {path}")

    return dict(
        files        = [str(p) for p in files],
        n_predictions= n_preds,
        duration_s   = out_secs,
        rms_original = rms_original,
        rms_anti     = rms_anti,
        rms_residual = rms_residual,
        reduction_db = reduction_db,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    mse   = float(np.mean((pred - actual) ** 2))
    rmse  = float(np.sqrt(mse))
    sig   = float(np.std(actual))
    nrmse = rmse / (sig + 1e-9)
    corr  = float(np.corrcoef(pred, actual)[0, 1]) if sig > 0 else 0.0
    return dict(mse=mse, rmse=rmse, nrmse=nrmse, corr=corr)


def _time_predictor(fn, history: np.ndarray, n_iter: int = 200) -> float:
    for _ in range(5):              # warmup
        fn(history)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(history)
    return (time.perf_counter() - t0) / n_iter * 1000.0       # ms / chunk


def _summary_line(name: str, t_ms: float, m: dict) -> str:
    return (f"  {name:<18s} {t_ms:8.3f} ms   "
            f"rmse={m['rmse']:.4f}   nrmse={m['nrmse']:.3f}   "
            f"corr={m['corr']:+.3f}")


def _verdict(m: dict) -> str:
    if m['corr'] > 0.85 and m['nrmse'] < 0.4:
        return "EXCELLENT — usable for ANC"
    if m['corr'] > 0.5:
        return "OK — borderline for ANC"
    return "POOR — would harm ANC"


def plot_predictions(history:  np.ndarray,
                     actual:   np.ndarray,
                     pred_lpc: np.ndarray,
                     pred_tcn: np.ndarray,
                     save_path: Path | None = None) -> None:
    n_hist, n_pred = len(history), len(actual)
    t_hist = np.arange(n_hist) / SAMPLE_RATE * 1000.0
    t_pred = (np.arange(n_pred) + n_hist) / SAMPLE_RATE * 1000.0

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.plot(t_hist, history,  color='gray',     alpha=0.55, lw=1.0, label='History (input)')
    ax.plot(t_pred, actual,   color='black',    lw=2.0,                label='Actual future')
    ax.plot(t_pred, pred_lpc, color='tab:blue', lw=1.3, ls='--',       label='LPC prediction')
    ax.plot(t_pred, pred_tcn, color='tab:red',  lw=1.3, ls='-.',       label='TCN prediction')
    ax.axvline(t_pred[0], color='k', alpha=0.4, lw=0.8)
    ax.text(t_pred[0], ax.get_ylim()[1] * 0.92, ' prediction →',
            fontsize=9, color='k', alpha=0.7)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'ANC predictor — next {PREDICT_LEN} smp '
                 f'({PREDICT_LEN / SAMPLE_RATE * 1000:.0f} ms) ahead')
    ax.legend(loc='upper right'); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"  saved plot -> {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main demo
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("ANC Predictor  |  LPC vs Lightweight Causal TCN")
    print(f"SR={SAMPLE_RATE} Hz   context={INPUT_LEN} smp "
          f"({INPUT_LEN / SAMPLE_RATE * 1000:.0f} ms)   "
          f"hw_chunk={STREAMING_CHUNK} smp "
          f"({STREAMING_CHUNK / SAMPLE_RATE * 1000:.0f} ms)   "
          f"predict={PREDICT_LEN} smp ({PREDICT_LEN / SAMPLE_RATE * 1000:.1f} ms)")
    rf = tcn_receptive_field(TCN_DILATIONS, kernel_size=3)
    print(f"TCN receptive field = {rf} smp ({rf / SAMPLE_RATE * 1000:.1f} ms)"
          f"   |   device = {DEVICE}")
    print("=" * 72)

    # ── Data: load entire WAV, skip 5 s silence, low-pass, split off 5 s ───
    print("\n[1/5] Loading noise dataset ...")
    full_sig = generate_engine_noise()            # uses full file after 5 s skip

    # Steep CAUSAL low-pass (Cheby II, 8th-order, cutoff 180 Hz, stopband
    # −60 dB) — single forward pass via sosfilt, exactly mirroring what
    # the ADC + DSP block on Jetson Nano would do.  Group delay is real
    # and the TCN learns to compensate for it during training.
    LPF_CUTOFF_HZ = 180.0
    LPF_ORDER     = 8
    full_sig = low_pass_filter(full_sig, cutoff_hz=LPF_CUTOFF_HZ, order=LPF_ORDER)
    print(f"  low-pass         : {LPF_CUTOFF_HZ:.0f} Hz, Cheby II order {LPF_ORDER}, "
          f"strictly CAUSAL (sosfilt, no future leakage)")

    test_len = int(5.0 * SAMPLE_RATE)
    if len(full_sig) <= test_len + INPUT_LEN + PREDICT_LEN:
        raise RuntimeError(
            f"Loaded noise has {len(full_sig)} smp "
            f"({len(full_sig)/SAMPLE_RATE:.1f} s) — too short for the train/test "
            f"split + context window."
        )
    train_sig = full_sig[:-test_len]
    test_sig  = full_sig[-test_len:]
    print(f"  full (post-skip) : {len(full_sig)} smp "
          f"({len(full_sig) / SAMPLE_RATE:.1f} s)")
    print(f"  train            : {len(train_sig)} smp "
          f"({len(train_sig) / SAMPLE_RATE:.1f} s)")
    print(f"  test (held out)  : {len(test_sig)} smp "
          f"({len(test_sig) / SAMPLE_RATE:.1f} s)")

    # ── Training pairs ───────────────────────────────────────────────────────
    print("\n[2/5] Building (input, target) training pairs ...")
    X_train, Y_train = make_training_pairs(
        train_sig, hop=64, chunk_size=INPUT_LEN, predict_size=PREDICT_LEN,
    )
    print(f"  X={X_train.shape}   Y={Y_train.shape}   "
          f"(hop=64 → ~{INPUT_LEN/64:.0f}× window overlap)")

    # ── TCN training ─────────────────────────────────────────────────────────
    print("\n[3/5] Training Lightweight Causal TCN ...")
    torch.manual_seed(SEED)
    model    = LightweightTCN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params}   (~{n_params * 4 / 1024:.1f} kB at fp32)")
    train_tcn(model, X_train, Y_train, epochs=100)

    # ── Single-shot accuracy on held-out signal (LPC vs TCN side-by-side) ───
    print("\n[4/5] Single-shot accuracy on held-out signal ...")
    eval_start = INPUT_LEN
    history    = test_sig[eval_start - INPUT_LEN : eval_start]
    actual     = test_sig[eval_start : eval_start + PREDICT_LEN]

    pred_lpc = predict_lpc(history)
    pred_tcn = predict_tcn(model, history)

    t_lpc = _time_predictor(lambda h: predict_lpc(h),       history)
    t_tcn = _time_predictor(lambda h: predict_tcn(model, h), history)

    m_lpc = _metrics(pred_lpc, actual)
    m_tcn = _metrics(pred_tcn, actual)

    print("\nResults (per-chunk latency + prediction accuracy):")
    print("  " + "-" * 64)
    print(_summary_line(f"LPC (order={LPC_ORDER})", t_lpc, m_lpc))
    print(_summary_line("Causal TCN",               t_tcn, m_tcn))
    print("  " + "-" * 64)
    print("\nInterpretation:")
    print(f"  LPC : {_verdict(m_lpc)}")
    print(f"  TCN : {_verdict(m_tcn)}")
    plot_predictions(history, actual, pred_lpc, pred_tcn)

    # ── Streaming demo (Jetson runtime topology) ─────────────────────────────
    print("\n[5/5] Streaming inference demo (RingBuffer over the held-out signal) ...")
    streaming_inference_demo(model, test_sig, n_show=3)

    # ── Persist trained TCN for the phase-eval companion script ──────────────
    ckpt = Path(__file__).resolve().parent / "tcn_AC_predictor.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"\nSaved TCN checkpoint -> {ckpt}")

    # ── Audible ANC ear-test — export 3 WAVs you can listen to side-by-side ─
    print("\n[ear test] Exporting ANC simulation WAVs ...")
    export_audio_simulation(model, test_sig, sr=SAMPLE_RATE, duration_s=5.0)


if __name__ == "__main__":
    main()
