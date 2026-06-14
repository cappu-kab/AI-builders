"""
inference.py
============
Run a trained CRN denoiser on .wav or .npy inputs.

Examples
--------
    # single file
    python inference.py --ckpt checkpoints/best.pt --input noisy.wav --output clean.wav

    # whole directory
    python inference.py --ckpt checkpoints/best.pt --input ./noisy_dir --output ./out_dir

    # supplied data_npy/test split  (mixes clean+noise on-the-fly at SNR 5 dB)
    python inference.py --ckpt checkpoints/best.pt \
                        --data_root ./data_npy --split test \
                        --snr 5 --output ./out_dir
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch

from model import CRN

try:
    import soundfile as sf
    HAVE_SF = True
except Exception:
    HAVE_SF = False

try:
    import scipy.io.wavfile as wavfile
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
def read_audio(path: Path, sr: int) -> np.ndarray:
    """Return a mono float32 waveform.  Supports .wav and .npy."""
    if path.suffix.lower() == ".npy":
        x = np.load(path).astype(np.float32).squeeze()
        if x.ndim > 1:
            x = x.mean(axis=0)
        return x
    if HAVE_SF:
        x, file_sr = sf.read(str(path), always_2d=False, dtype="float32")
        if x.ndim > 1: x = x.mean(axis=1)
        if file_sr != sr:
            x = _resample_linear(x, file_sr, sr)
        return x.astype(np.float32)
    if HAVE_SCIPY:
        file_sr, x = wavfile.read(str(path))
        if x.dtype.kind in "iu":
            x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
        else:
            x = x.astype(np.float32)
        if x.ndim > 1: x = x.mean(axis=1)
        if file_sr != sr:
            x = _resample_linear(x, file_sr, sr)
        return x
    raise RuntimeError("Need either soundfile or scipy to read .wav files")


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    n_out = int(round(x.size * sr_out / sr_in))
    idx = np.linspace(0.0, x.size - 1, n_out)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, x.size - 1)
    fr = (idx - i0).astype(np.float32)
    return ((1.0 - fr) * x[i0] + fr * x[i1]).astype(np.float32)


def write_wav(path: Path, x: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    if HAVE_SF:
        sf.write(str(path), x, sr, subtype="PCM_16")
    elif HAVE_SCIPY:
        wavfile.write(str(path), sr, (x * 32767.0).astype(np.int16))
    else:
        # Fallback: save as .npy
        np.save(str(path.with_suffix(".npy")), x)


# ---------------------------------------------------------------------------
@torch.no_grad()
def enhance_waveform(model: CRN, wav: np.ndarray, device: torch.device,
                     chunk_seconds: float = 8.0, sr: int = 16_000) -> np.ndarray:
    """Run the model in fixed-size chunks with 50 ms overlap to avoid edges."""
    chunk = int(chunk_seconds * sr)
    overlap = int(0.05 * sr)                # 50 ms cross-fade
    if wav.size <= chunk:
        x = torch.from_numpy(wav).unsqueeze(0).to(device)
        y = model(x).squeeze(0).cpu().numpy()
        return y[: wav.size]

    out = np.zeros_like(wav)
    win = np.ones(chunk, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        win[:overlap]  = ramp
        win[-overlap:] = ramp[::-1]

    pos = 0
    norm = np.zeros_like(wav)
    while pos < wav.size:
        end = min(pos + chunk, wav.size)
        seg = wav[pos:end]
        if seg.size < chunk:
            seg = np.pad(seg, (0, chunk - seg.size))
        x = torch.from_numpy(seg).unsqueeze(0).to(device)
        y = model(x).squeeze(0).cpu().numpy()
        out_len = end - pos
        out[pos:end]  += y[:out_len] * win[:out_len]
        norm[pos:end] += win[:out_len]
        pos += chunk - overlap
    norm[norm < 1e-6] = 1.0
    return (out / norm).astype(np.float32)


# ---------------------------------------------------------------------------
def soft_noise_gate(wav: np.ndarray, sr: int,
                    threshold_db: float = -40.0,
                    knee_db:      float =   6.0,
                    attack_ms:    float =   5.0,
                    release_ms:   float =  80.0) -> np.ndarray:
    """Apply a soft noise gate with smooth attack/release envelope.

    Algorithm
    ---------
    1. Compute short-time RMS energy in 20 ms windows at a 10 ms hop.
    2. Map RMS to a linear gain [0, 1] using a soft cosine knee centred on
       `threshold_db` ± `knee_db/2`.  No hard 0/1 step — the gain ramps
       smoothly through the knee so the gate never snaps open or shut.
    3. Smooth the per-frame gain with a one-pole IIR filter that uses a
       *fast* attack coefficient (gate opens quickly when speech starts) and
       a *slow* release coefficient (gate stays open between short pauses).
    4. Linearly interpolate the smoothed frame-rate envelope back to sample
       rate before multiplying — guarantees no digital clicks regardless of
       hop size.

    Parameters
    ----------
    threshold_db : float
        RMS level (dBFS) below which the gate begins to close.
        Default -40 dB targets typical background haze while leaving
        quiet speech (usually above -30 dB) untouched.
    knee_db : float
        Width of the soft-knee transition zone centred on threshold.
        Default 6 dB means full attenuation below -43 dB, fully open
        above -37 dB, and a smooth cosine curve between.
    attack_ms : float
        Time (ms) for the gate to open when signal rises above threshold.
        5 ms is imperceptibly fast — avoids clipping word onsets.
    release_ms : float
        Time (ms) for the gate to close after signal drops below threshold.
        80 ms keeps the gate open through brief inter-word pauses and
        prevents the characteristic "pumping" of aggressive gating.
    """
    n = len(wav)
    if n == 0:
        return wav

    # 10 ms hop, 20 ms analysis window (perceptually appropriate; not too
    # short to misfire on single-cycle pitch periods, not too long to miss
    # brief pauses between syllables).
    hop = max(1, sr // 100)        # 10 ms
    win = max(hop, sr // 50)       # 20 ms

    # --- 1. Per-frame RMS in dBFS ---
    n_frames = max(1, (n + hop - 1) // hop)
    rms_db   = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s        = i * hop
        e        = min(s + win, n)
        rms_db[i] = 20.0 * np.log10(
            float(np.sqrt(np.mean(wav[s:e] ** 2))) + 1e-12
        )

    # --- 2. Soft-knee gain (0.0 = closed, 1.0 = open) ---
    half_knee = knee_db * 0.5
    lo, hi    = threshold_db - half_knee, threshold_db + half_knee
    gain = np.where(
        rms_db <= lo, 0.0,
        np.where(
            rms_db >= hi, 1.0,
            0.5 * (1.0 - np.cos(np.pi * (rms_db - lo) / knee_db))
        )
    ).astype(np.float32)

    # --- 3. One-pole IIR attack / release smoothing at frame rate ---
    fps = sr / hop                                        # frames per second
    a_c = np.exp(-1.0 / max(1.0, attack_ms  * fps / 1000.0))
    r_c = np.exp(-1.0 / max(1.0, release_ms * fps / 1000.0))
    g   = float(gain[0])
    for i in range(n_frames):
        target  = float(gain[i])
        coeff   = a_c if target > g else r_c   # rising → attack, falling → release
        g       = coeff * g + (1.0 - coeff) * target
        gain[i] = g

    # --- 4. Upsample envelope to sample rate (linear interp → zero clicks) ---
    frame_pos    = np.arange(n_frames, dtype=np.float64) * hop
    sample_pos   = np.arange(n,        dtype=np.float64)
    gain_samples = np.interp(sample_pos, frame_pos, gain).astype(np.float32)

    return (wav * gain_samples).astype(np.float32)


# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, device: torch.device) -> Tuple[CRN, dict]:
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck.get("args", {})
    model = CRN(
        n_fft=cfg.get("n_fft", 512),
        hop_length=cfg.get("hop_length", 128),
        rnn_hidden=cfg.get("rnn_hidden", 192),
        rnn_layers=cfg.get("rnn_layers", 1),
        rnn_type=cfg.get("rnn_type", "gru"),
        bottleneck_dim=cfg.get("bottleneck_dim", 384),
    ).to(device)
    # `strict=False` lets us load checkpoints from older config variants
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"[infer] load_state_dict — missing: {len(missing)}  unexpected: {len(unexpected)}")
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
def iter_inputs(args) -> Iterable[Tuple[str, np.ndarray]]:
    """Yield (name, noisy_waveform) tuples."""
    sr = args.sample_rate
    if args.data_root:
        # mix clean + noise from the requested split at the requested SNR
        from dataset import _read_labels, _crop_or_pad, mix_at_snr
        split_dir = Path(args.data_root) / args.split
        clean_dir = split_dir / "clean"
        noise_dir = split_dir / "noise"
        clean_files = sorted(clean_dir.glob("*.npy"))
        noise_files = sorted(noise_dir.glob("*.npy"))
        if not clean_files: raise RuntimeError(f"No clean files in {clean_dir}")
        if not noise_files: raise RuntimeError(f"No noise files in {noise_dir}")
        rng = np.random.default_rng(args.seed)
        for cf in clean_files:
            clean = np.load(cf).astype(np.float32).squeeze()
            if clean.ndim > 1: clean = clean.mean(axis=0)
            n_path = noise_files[int(rng.integers(0, len(noise_files)))]
            noise = np.load(n_path).astype(np.float32).squeeze()
            if noise.ndim > 1: noise = noise.mean(axis=0)
            noise = _crop_or_pad(noise, clean.size)
            noisy, _ = mix_at_snr(clean, noise, args.snr)
            yield cf.stem, noisy
        return

    src = Path(args.input)
    if src.is_dir():
        for f in sorted(src.iterdir()):
            if f.suffix.lower() in (".wav", ".flac", ".ogg", ".npy"):
                yield f.stem, read_audio(f, sr)
    else:
        yield src.stem, read_audio(src, sr)


# ---------------------------------------------------------------------------
def main():
    # Defaults are chosen so you can hit VS Code's Run button (no CLI args).
    # They assume the project layout:
    #   project/
    #     inference.py
    #     checkpoints/best.pt
    #     data_npy/{train,val,test}/{clean,noise}/
    #     outputs/
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",      default="./checkpoints/best.pt")
    ap.add_argument("--input",     default="")
    ap.add_argument("--output",    default="./outputs/enhanced")
    ap.add_argument("--data_root", default="./data_npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--snr", type=float, default=5.0,
                    help="When mixing on-the-fly from data_root, target SNR in dB.")
    ap.add_argument("--sample_rate", type=int, default=16_000)
    ap.add_argument("--chunk_seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise_gate_db", type=float, default=-40.0,
                    help="Soft noise gate threshold in dBFS. Segments whose RMS falls "
                         "below this level are smoothly attenuated toward zero. "
                         "Default -40 dB silences background haze without touching "
                         "quiet speech. Use -120 to disable effectively.")
    args = ap.parse_args()

    if not args.input and not args.data_root:
        print("Provide either --input or --data_root", file=sys.stderr); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _cfg = load_model(args.ckpt, device)
    print(f"[infer] loaded {args.ckpt} on {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    treat_output_as_dir = args.data_root or Path(args.input).is_dir()

    gate_db = args.noise_gate_db
    print(f"[infer] noise gate threshold: {gate_db:.1f} dBFS  "
          f"(knee=6 dB  attack=5 ms  release=80 ms)")

    n = 0
    for name, noisy in iter_inputs(args):
        enhanced = enhance_waveform(model, noisy, device,
                                    chunk_seconds=args.chunk_seconds,
                                    sr=args.sample_rate)
        enhanced = soft_noise_gate(enhanced, args.sample_rate,
                                   threshold_db=gate_db)
        if treat_output_as_dir:
            write_wav(out_dir / f"{name}_enhanced.wav", enhanced, args.sample_rate)
            write_wav(out_dir / f"{name}_noisy.wav",    noisy,    args.sample_rate)
        else:
            write_wav(Path(args.output), enhanced, args.sample_rate)
        n += 1
    print(f"[infer] wrote {n} file(s) to {args.output}")


if __name__ == "__main__":
    main()
