"""
denoise_core.py — CRN speech denoiser, offline.
Public API: process(in_wav_path, dry_mix=None) -> out_wav_path
Model loaded once at import time.
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).parent.resolve()          # AI_builders/
_CRN_DIR = _ROOT / "Run" / "CRN"
_CKPT    = _CRN_DIR / "checkpoints" / "best.pt"
_OUT_DIR = _ROOT / "outputs_tmp"
_OUT_DIR.mkdir(exist_ok=True)

# Insert _ROOT first, then _CRN_DIR last → _CRN_DIR gets index 0 (highest priority).
# Prevents loading stale AI_builders/inference.py instead of Run/CRN/inference.py.
for _p in (str(_ROOT), str(_CRN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference import load_model  # noqa: E402

# ── Post-processing constants ──────────────────────────────────────────────────
_SAMPLE_RATE  = 16_000
_MAX_SAMPLES  = 480_000   # 30 s cap
_PRE_PAD_S    = 8_192     # silence warmup for forward BiLSTM
_DRY_MIX      = 0.05      # default: blend 5% original back in (tuned 2026-06-05)
_TARGET_RMS   = 0.15      # output loudness target
_PEAK_GUARD   = 0.99      # hard ceiling after RMS normalisation

# ── Noise gate constants (ratio-based, ported from v6 ANC) ────────────────────
_GATE_FRAME_S        = 320    # 20 ms at 16 kHz — RMS analysis frame
_GATE_RATIO_LOW      = 0.10   # ratio ≤ LOW  → full attenuation (noise-only frame)
_GATE_RATIO_HIGH     = 0.20   # ratio ≥ HIGH → no attenuation (speech frame)
_GATE_ATTENUATION_DB = -15.0  # attenuation floor when fully gated (dB)
_GATE_ATTACK_MS      = 72.0   # ms — smoothed gate-close; release is instant

# ── AGC constants ─────────────────────────────────────────────────────────────
_AGC_TARGET_RMS = 0.15   # target local RMS; quiet sections below this get boosted
_AGC_MAX_BOOST  = 8.0    # max gain multiplier (~18 dB); lifts faint speech sections
_AGC_ATTACK_MS  = 50     # ms — gain-decrease time constant (loud section response)
_AGC_RELEASE_MS = 500    # ms — gain-increase time constant (section-level ramp-up)
_AGC_FRAME_MS   = 100    # ms — RMS analysis frame (longer = measures section level, not peaks)

# ── Compression constants ─────────────────────────────────────────────────────
_COMP_THRESHOLD_DB = -18.0  # dB — levels above this get compressed
_COMP_RATIO        = 3.0    # compression ratio above threshold (3:1)
_COMP_KNEE_DB      = 6.0    # soft-knee width; wider = more natural transition
_COMP_ATTACK_MS    = 30     # ms — how fast gain drops on loud sections
_COMP_RELEASE_MS   = 150    # ms — how fast gain recovers (slow prevents pumping)
_COMP_FRAME_MS     = 10     # ms — analysis frame

# OLA constants (mirror crn_pipeline.py)
_CRN_CTX_N    = 47_616
_SAFE_SINGLE  = 64_000
_OLA_HOP      = _CRN_CTX_N // 2


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_crn():
    import torch
    cuda = torch.device("cuda")
    cpu  = torch.device("cpu")

    if not torch.cuda.is_available():
        model, _ = load_model(_CKPT, cpu)
        print("[denoise_core] model ready on cpu")
        return model, cpu

    # Load to CPU → eval() → move to CUDA (avoids cuDNN flatten_parameters crash)
    try:
        model, _ = load_model(_CKPT, cpu)
        model.eval()
        model.to(cuda)
        print("[denoise_core] model ready on cuda")
        return model, cuda
    except RuntimeError as exc:
        print(f"[denoise_core] CUDA failed ({exc}); trying cuDNN-disabled")

    try:
        torch.backends.cudnn.enabled = False
        model, _ = load_model(_CKPT, cuda)
        print("[denoise_core] model ready on cuda (cuDNN disabled)")
        return model, cuda
    except RuntimeError as exc:
        print(f"[denoise_core] CUDA+no-cuDNN failed ({exc}); falling back to cpu")
        torch.backends.cudnn.enabled = True

    model, _ = load_model(_CKPT, cpu)
    print("[denoise_core] model ready on cpu (fallback)")
    return model, cpu


print(f"[denoise_core] loading checkpoint: {_CKPT}")
_MODEL, _DEVICE = _load_crn()


# ── Inference helpers ──────────────────────────────────────────────────────────

def _run_model(x_np: np.ndarray) -> np.ndarray:
    import torch
    def _infer(dev):
        with torch.inference_mode():
            x   = torch.from_numpy(x_np).unsqueeze(0).to(dev)
            out = _MODEL(x)
            return out.squeeze(0).cpu().numpy().astype(np.float32)

    if _DEVICE.type == "cuda":
        try:
            return _infer(_DEVICE)
        except Exception as exc:
            print(f"[denoise_core] CUDA inference failed ({exc}); retrying on cpu")
            _MODEL.to("cpu")
            return _infer(torch.device("cpu"))
    return _infer(_DEVICE)


def _enhance_chunked(pcm_f32: np.ndarray) -> np.ndarray:
    L = len(pcm_f32)
    if L == 0:
        return np.zeros(0, dtype=np.float32)
    if L <= _SAFE_SINGLE:
        enh = _run_model(pcm_f32)
        return enh[:L]

    W = _CRN_CTX_N
    H = _OLA_HOP
    win = np.hanning(W).astype(np.float32)
    n_win = max(1, int(np.ceil((L - W) / H)) + 1) if L > W else 1
    padded_len = (n_win - 1) * H + W
    pcm_pad = np.concatenate([pcm_f32, np.zeros(padded_len - L, dtype=np.float32)])
    output = np.zeros(padded_len, dtype=np.float32)
    norm   = np.zeros(padded_len, dtype=np.float32)
    for i in range(n_win):
        s   = i * H
        seg = pcm_pad[s : s + W]
        enh = _run_model(seg)
        if len(enh) > W:
            enh = enh[:W]
        elif len(enh) < W:
            enh = np.pad(enh, (0, W - len(enh)))
        output[s : s + W] += enh * win
        norm[s : s + W]   += win
    return (output / np.maximum(norm, 1e-8))[:L]


def _noise_gate(enh_f32: np.ndarray, orig_f32: np.ndarray) -> np.ndarray:
    """
    Ratio-based gate (logic ported from v6 ANC file_simulator).
    Detection: ratio = rms(output_frame) / rms(input_frame).
      Low ratio  → model suppressed the frame heavily → noise-only → attenuate.
      High ratio → model preserved amplitude         → speech     → pass through.
    This direction is correct: gate CLOSES on noise, OPENS on speech.
    Asymmetric: instant open (speech onset), smoothed close (_GATE_ATTACK_MS).
    """
    L = len(enh_f32)
    F = _GATE_FRAME_S
    n_frames = (L + F - 1) // F
    _floor = float(10.0 ** (_GATE_ATTENUATION_DB / 20.0))

    # Per-frame ratio: rms(output) / rms(input)
    ratios = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * F
        e = min(s + F, L)
        rms_in  = float(np.sqrt(np.mean(orig_f32[s:e] ** 2)))
        rms_out = float(np.sqrt(np.mean(enh_f32[s:e] ** 2)))
        ratios[i] = rms_out / (rms_in + 1e-9)

    n_gated = int((ratios < _GATE_RATIO_LOW).sum())
    print(f"[gate] ratio min={ratios.min():.3f} mean={ratios.mean():.3f} max={ratios.max():.3f}  "
          f"gated(ratio<{_GATE_RATIO_LOW})={n_gated}/{n_frames}")

    # Target gain per frame: linear interp in dB between floor and 0 dB
    target = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        r = float(ratios[i])
        if r <= _GATE_RATIO_LOW:
            target[i] = _floor
        elif r >= _GATE_RATIO_HIGH:
            target[i] = 1.0
        else:
            t = (r - _GATE_RATIO_LOW) / (_GATE_RATIO_HIGH - _GATE_RATIO_LOW)
            target[i] = float(10.0 ** (_GATE_ATTENUATION_DB * (1.0 - t) / 20.0))

    # Asymmetric IIR: instant open (gain up), smoothed close (gain down)
    frame_ms    = 1000.0 * F / _SAMPLE_RATE  # 20 ms
    alpha_close = float(1.0 - np.exp(-frame_ms / _GATE_ATTACK_MS))
    smooth = np.zeros(n_frames, dtype=np.float32)
    gain = 1.0
    for i in range(n_frames):
        tgt = float(target[i])
        if tgt >= gain:
            gain = tgt                       # instant open
        else:
            gain += alpha_close * (tgt - gain)  # smoothed close
        smooth[i] = gain

    sample_gain = np.repeat(smooth, F)[:L]
    return (enh_f32 * sample_gain).astype(np.float32)


def _compress(pcm_f32: np.ndarray) -> np.ndarray:
    """
    Gentle downward compressor. Reduces loud frames above _COMP_THRESHOLD_DB
    at _COMP_RATIO:1, soft knee, asymmetric attack/release.
    Only attenuates — never amplifies. Works with AGC to narrow dynamic range.
    """
    L = len(pcm_f32)
    F = max(1, int(_SAMPLE_RATE * _COMP_FRAME_MS / 1000))
    n_frames = (L + F - 1) // F

    # Per-frame RMS → dB
    frame_rms = np.array([
        float(np.sqrt(np.mean(pcm_f32[i * F : min((i + 1) * F, L)] ** 2)))
        for i in range(n_frames)
    ], dtype=np.float32)
    frame_db = (20.0 * np.log10(np.maximum(frame_rms, 1e-9))).astype(np.float32)

    above_thresh = int((frame_db > _COMP_THRESHOLD_DB).sum())
    print(f"[compress] signal dB: min={frame_db.min():.1f} mean={frame_db.mean():.1f} "
          f"max={frame_db.max():.1f}  threshold={_COMP_THRESHOLD_DB} dB  "
          f"frames above threshold: {above_thresh}/{n_frames}")

    # Soft-knee gain (dB) per frame
    half_knee = _COMP_KNEE_DB / 2.0
    gain_db = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        above = float(frame_db[i]) - _COMP_THRESHOLD_DB
        if above <= -half_knee:
            gain_db[i] = 0.0
        elif above >= half_knee:
            gain_db[i] = (1.0 / _COMP_RATIO - 1.0) * above
        else:
            gain_db[i] = (1.0 / _COMP_RATIO - 1.0) * (above + half_knee) ** 2 / (2.0 * _COMP_KNEE_DB)

    # Desired linear gain (≤ 1.0 — compression only reduces)
    desired = np.power(10.0, gain_db / 20.0).astype(np.float32)

    # Asymmetric IIR: fast attack (gain down), slow release (gain up)
    attack_coef  = float(1.0 - np.exp(-_COMP_FRAME_MS / _COMP_ATTACK_MS))
    release_coef = float(1.0 - np.exp(-_COMP_FRAME_MS / _COMP_RELEASE_MS))
    smooth = np.zeros(n_frames, dtype=np.float32)
    gain = 1.0
    for i in range(n_frames):
        d = float(desired[i])
        coef = attack_coef if d < gain else release_coef
        gain += coef * (d - gain)
        smooth[i] = gain

    centers     = np.arange(n_frames, dtype=np.float32) * F + F * 0.5
    sample_gain = np.interp(np.arange(L, dtype=np.float32), centers, smooth).astype(np.float32)
    return (pcm_f32 * sample_gain).astype(np.float32)


def _agc(pcm_f32: np.ndarray) -> np.ndarray:
    """
    Automatic gain control — boosts only (gain >= 1.0), never cuts.
    Loud sections (RMS >= target) are untouched. Quiet sections are lifted.
    Smooth attack/release over frames prevents pumping.
    """
    L = len(pcm_f32)
    F = max(1, int(_SAMPLE_RATE * _AGC_FRAME_MS / 1000))  # samples per frame
    n_frames = (L + F - 1) // F

    # Per-frame RMS
    frame_rms = np.array([
        float(np.sqrt(np.mean(pcm_f32[i * F : min((i + 1) * F, L)] ** 2)))
        for i in range(n_frames)
    ], dtype=np.float32)

    # Desired gain: boost only (floor 1.0), cap at max_boost
    desired = np.clip(
        _AGC_TARGET_RMS / np.maximum(frame_rms, 1e-6),
        1.0, _AGC_MAX_BOOST
    ).astype(np.float32)

    # Attack/release smoothing — loop over frames (~3000 iters for 30 s, fast)
    attack_coef  = float(1.0 - np.exp(-1.0 / max(_AGC_ATTACK_MS  / _AGC_FRAME_MS, 1e-9)))
    release_coef = float(1.0 - np.exp(-1.0 / max(_AGC_RELEASE_MS / _AGC_FRAME_MS, 1e-9)))
    smooth = np.zeros(n_frames, dtype=np.float32)
    gain = 1.0
    for i in range(n_frames):
        d = float(desired[i])
        gain += (attack_coef if d < gain else release_coef) * (d - gain)
        smooth[i] = gain

    # Interpolate frame gains to sample level and apply
    centers     = np.arange(n_frames, dtype=np.float32) * F + F * 0.5
    sample_gain = np.interp(np.arange(L, dtype=np.float32), centers, smooth).astype(np.float32)
    return (pcm_f32 * sample_gain).astype(np.float32)


# ── Public API ─────────────────────────────────────────────────────────────────

def process(in_wav_path: str, dry_mix: float | None = None,
            gate: bool = False, agc: bool = False,
            compress: bool = False) -> str:
    """
    Denoise a WAV file. Returns path to denoised WAV in outputs_tmp/.
    dry_mix:  override _DRY_MIX (0.0–1.0). None = use module default.
    gate:     ratio-based noise gate after dry-mix.
    agc:      automatic gain control after gate.
    compress: downward compression after AGC (threshold/ratio in constants).
    """
    import scipy.signal as ss

    mix = _DRY_MIX if dry_mix is None else float(dry_mix)

    # Read input
    with wave.open(in_wav_path, "rb") as wf:
        sr   = wf.getframerate()
        n_ch = wf.getnchannels()
        sw   = wf.getsampwidth()
        raw  = wf.readframes(wf.getnframes())

    dtype = np.int16 if sw == 2 else np.int32
    pcm   = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    pcm  /= (32768.0 if sw == 2 else 2147483648.0)

    # Stereo → mono
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)

    # Resample to 16 kHz if needed
    if sr != _SAMPLE_RATE:
        target_len = int(len(pcm) * _SAMPLE_RATE / sr)
        pcm = ss.resample(pcm, target_len).astype(np.float32)

    # 30 s cap
    if len(pcm) > _MAX_SAMPLES:
        pcm = pcm[:_MAX_SAMPLES]

    orig = pcm.copy()

    # Pre-pad warmup → enhance → trim
    padded     = np.concatenate([np.zeros(_PRE_PAD_S, dtype=np.float32), pcm])
    enh_padded = _enhance_chunked(padded)
    enh        = enh_padded[_PRE_PAD_S : _PRE_PAD_S + len(pcm)]

    # Dry mix
    enh = (1.0 - mix) * enh + mix * orig

    # Noise gate (optional) — uses orig (noisy input) as detection reference
    if gate:
        enh = _noise_gate(enh, orig)

    # AGC (optional) — prints per-2s section RMS before/after to prove leveling
    if agc:
        _sec = _SAMPLE_RATE * 2
        _n   = min(6, len(enh) // _sec)
        _b   = [20.0 * np.log10(max(float(np.sqrt(np.mean(enh[i*_sec:(i+1)*_sec]**2))), 1e-9))
                for i in range(_n)]
        enh  = _agc(enh)
        _a   = [20.0 * np.log10(max(float(np.sqrt(np.mean(enh[i*_sec:(i+1)*_sec]**2))), 1e-9))
                for i in range(_n)]
        print("[agc] per-2s section dB  before->after:")
        for i, (b, a) in enumerate(zip(_b, _a)):
            print(f"  s={i*2:2d}-{i*2+2:2d}s: {b:+.1f} -> {a:+.1f} dB  (d={a-b:+.1f})")

    # Compression (optional)
    if compress:
        enh = _compress(enh)

    # RMS normalisation
    rms = float(np.sqrt(np.mean(enh ** 2)))
    if rms > 1e-6:
        enh = enh * min(_TARGET_RMS / rms, 3.0)

    # Peak guard
    peak = float(np.abs(enh).max())
    if peak > _PEAK_GUARD:
        enh = enh * (_PEAK_GUARD / peak)

    # Write output
    stem     = Path(in_wav_path).stem
    out_path = str(_OUT_DIR / f"{stem}_denoised.wav")
    enh_i16  = np.clip(enh * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(enh_i16.tobytes())

    return out_path
