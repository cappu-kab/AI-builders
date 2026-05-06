"""
dataset.py
==========
Custom Dataset for on-the-fly speech-denoising training.

CRITICAL DESIGN RULE
--------------------
NO augmentation is performed during preprocessing.  Everything below
(noise mixing, alpha scaling, SNR variation, time stretch, pitch shift,
random gain, partial dropout, frequency masking) happens INSIDE
`__getitem__` so each epoch sees a different noisy realisation of every
clean utterance.

Expected on-disk layout (already preprocessed to .npy waveforms):

    DATA_ROOT/
        train/
            clean/*.npy          # mono float32 waveform, sr = SAMPLE_RATE
            noise/*.npy
            labels_npy.csv       # filename,transcript
        val/   (same)
        test/  (same)
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
SEGMENT_SECONDS = 4.0
SEGMENT_LEN = int(SAMPLE_RATE * SEGMENT_SECONDS)

EPS = 1e-8


# ---------------------------------------------------------------------------
# Augmentation primitives  (operate on 1-D float32 numpy arrays)
# ---------------------------------------------------------------------------
def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + EPS))


def sample_snr(snr_low: float, snr_high: float,
               hard_low: float = -5.0, hard_high: float = 5.0,
               p_hard: float = 0.5) -> float:
    """Biased SNR sampler — 50% of draws fall in the hard zone [hard_low, hard_high].

    Low-SNR samples are the hardest for the model and the most informative
    for gradient updates.  Uniform sampling underrepresents them when the
    full range is wide (e.g. -20 to +20 dB).
    """
    lo = max(snr_low,  hard_low)
    hi = min(snr_high, hard_high)
    if lo < hi and random.random() < p_hard:
        return random.uniform(lo, hi)
    return random.uniform(snr_low, snr_high)


def noise_offset(noise: np.ndarray) -> np.ndarray:
    """Circularly shift noise by a random amount before mixing.

    Prevents the model from exploiting any fixed temporal alignment between
    the noise segment and the speech crop (e.g. both starting at sample 0).
    """
    shift = random.randint(0, max(0, len(noise) - 1))
    return np.roll(noise, shift).astype(np.float32)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> Tuple[np.ndarray, float]:
    """Return (noisy, alpha) where  noisy = clean + alpha * noise  achieves snr_db."""
    clean_rms = _rms(clean)
    noise_rms = _rms(noise)
    alpha = clean_rms / (noise_rms * (10.0 ** (snr_db / 20.0)) + EPS)
    noisy = clean + alpha * noise
    return noisy.astype(np.float32), float(alpha)


def random_gain(x: np.ndarray, low_db: float = -6.0, high_db: float = 6.0) -> np.ndarray:
    g = 10.0 ** (random.uniform(low_db, high_db) / 20.0)
    return (x * g).astype(np.float32)


def partial_dropout(x: np.ndarray, p: float = 0.5,
                    min_frac: float = 0.02, max_frac: float = 0.08) -> np.ndarray:
    """Zero-out a contiguous chunk of the waveform (simulates packet loss / dropouts).

    DISABLED by default (p=0.0 at call site): zeroing 2-8% of raw samples creates
    broadband vertical transients in the STFT that have no natural phase structure.
    The STFT model cannot reconstruct these, so the spectral noise loss drives the
    mask toward suppressing the surrounding speech instead — exactly the target
    leakage pattern seen in spectrogram evaluations.
    """
    if random.random() > p or x.size < 32:
        return x
    n = x.size
    span = int(n * random.uniform(min_frac, max_frac))
    start = random.randint(0, n - span - 1)
    out = x.copy()
    out[start:start + span] = 0.0
    return out


def time_stretch(x: np.ndarray, p: float = 0.5,
                 low: float = 0.9, high: float = 1.1) -> np.ndarray:
    """Cheap (linear-interp) time stretch — avoids the librosa dependency.
    rate < 1  -> slower / longer, rate > 1 -> faster / shorter."""
    if random.random() > p:
        return x
    rate = random.uniform(low, high)
    n_out = max(1, int(round(x.size / rate)))
    idx = np.linspace(0.0, x.size - 1, n_out)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, x.size - 1)
    frac = (idx - i0).astype(np.float32)
    return ((1.0 - frac) * x[i0] + frac * x[i1]).astype(np.float32)


def pitch_shift(x: np.ndarray, p: float = 0.3,
                semitones_low: float = -2.0, semitones_high: float = 2.0) -> np.ndarray:
    """Pitch shift via resample-then-restore-length (rough but library-free)."""
    if random.random() > p:
        return x
    semitones = random.uniform(semitones_low, semitones_high)
    rate = 2.0 ** (semitones / 12.0)
    n1 = max(1, int(round(x.size / rate)))
    idx = np.linspace(0.0, x.size - 1, n1)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, x.size - 1)
    frac = (idx - i0).astype(np.float32)
    pitched = (1.0 - frac) * x[i0] + frac * x[i1]
    idx2 = np.linspace(0.0, pitched.size - 1, x.size)
    j0 = np.floor(idx2).astype(np.int64)
    j1 = np.clip(j0 + 1, 0, pitched.size - 1)
    fr2 = (idx2 - j0).astype(np.float32)
    return ((1.0 - fr2) * pitched[j0] + fr2 * pitched[j1]).astype(np.float32)


# Cache one Hann window per (n_fft,) so we don't re-allocate inside hot loops.
_HANN_CACHE: Dict[int, torch.Tensor] = {}


def _hann(n_fft: int) -> torch.Tensor:
    w = _HANN_CACHE.get(n_fft)
    if w is None:
        w = torch.hann_window(n_fft)
        _HANN_CACHE[n_fft] = w
    return w


def freq_mask_waveform(x: np.ndarray, p: float = 0.3,
                       n_fft: int = 512, hop: int = 128,
                       max_bins: int = 24) -> np.ndarray:
    """Apply SpecAugment-style frequency masking by round-tripping through STFT."""
    if random.random() > p:
        return x
    t = torch.from_numpy(x)
    win = _hann(n_fft)
    spec = torch.stft(t, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    F_bins = spec.shape[0]
    width = random.randint(1, max_bins)
    start = random.randint(0, max(0, F_bins - width))
    spec[start:start + width, :] = 0.0
    y = torch.istft(spec, n_fft=n_fft, hop_length=hop, window=win, length=x.size)
    return y.numpy().astype(np.float32)


def apply_rir(x: np.ndarray, p: float = 0.4,
              sample_rate: int = 16_000) -> np.ndarray:
    """Convolve with a synthetic exponentially-decaying room impulse response.

    Applied to the noisy mix (not the clean target) so the model learns to
    suppress both noise AND room echo.  RT60 ∈ [0.08, 0.55] s covers small
    rooms to medium halls.  Early reflections (2–50 ms delays) are added
    stochastically to make each RIR unique.
    """
    if random.random() > p or x.size < 64:
        return x
    rt60          = random.uniform(0.08, 0.55)
    decay_samples = max(1, int(rt60 * sample_rate))
    t             = np.arange(decay_samples, dtype=np.float32)
    rir = np.exp(-13.816 * t / decay_samples)
    n_ref = random.randint(2, 7)
    for _ in range(n_ref):
        delay = random.randint(int(0.002 * sample_rate), int(0.05 * sample_rate))
        if delay < decay_samples:
            amp = random.uniform(0.2, 0.75) * np.exp(-13.816 * delay / decay_samples)
            rir[delay] += amp
    rir /= (np.abs(rir).sum() + EPS)

    n_fft = len(x) + len(rir) - 1
    Y     = np.fft.rfft(x,   n=n_fft) * np.fft.rfft(rir, n=n_fft)
    y     = np.fft.irfft(Y)[:len(x)].astype(np.float32)

    rms_in  = float(np.sqrt(np.mean(x ** 2) + EPS))
    rms_out = float(np.sqrt(np.mean(y ** 2) + EPS))
    return (y * rms_in / rms_out).astype(np.float32)


def color_noise(x: np.ndarray, p: float = 0.5,
                alpha_low: float = -1.0, alpha_high: float = 0.0) -> np.ndarray:
    """Spectrally tilt noise from white toward pink (α=-0.5) or brown (α=-1).

    Multiplies FFT amplitudes by f^α, then re-normalises to preserve RMS.
    Applied to the raw noise segment before SNR mixing so every training
    sample gets a different noise spectral shape.

    alpha_low / alpha_high control the spectral-tilt range:
      alpha=0   → white noise  (flat spectrum)
      alpha=-0.5 → pink noise  (PSD ∝ 1/f)
      alpha=-1  → brown noise  (PSD ∝ 1/f²) — energy concentrated below 200 Hz
    """
    if random.random() > p or x.size < 4:
        return x
    alpha  = random.uniform(alpha_low, alpha_high)
    X      = np.fft.rfft(x)
    freqs  = np.fft.rfftfreq(len(x))
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    X     *= np.abs(freqs) ** alpha
    y      = np.fft.irfft(X, n=len(x)).astype(np.float32)
    rms_in  = float(np.sqrt(np.mean(x ** 2) + EPS))
    rms_out = float(np.sqrt(np.mean(y ** 2) + EPS))
    return (y * rms_in / rms_out).astype(np.float32)


def inject_low_freq_noise(noise: np.ndarray, p: float = 0.35) -> np.ndarray:
    """Additively blend a synthetic brown/pink noise component into the noise segment.

    Unlike color_noise (which tilts the existing spectrum), this generates a
    FRESH low-frequency noise signal and adds it on top, guaranteeing that
    sub-200 Hz energy is present in the mixture even when color_noise chose a
    white or pink tilt.  Supports the frequency-weighted STFT loss objective.

    The additive level is drawn from 30–80% of the existing noise RMS, so the
    low-freq component is meaningful but never overwhelms the primary noise.
    alpha ∈ [-1.0, -0.5]: brown-to-pink range — all have PSD peak below 200 Hz.
    """
    if random.random() > p or noise.size < 4:
        return noise
    alpha  = random.uniform(-1.0, -0.5)
    white  = np.random.randn(noise.size).astype(np.float32)
    X      = np.fft.rfft(white)
    freqs  = np.fft.rfftfreq(noise.size)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    X     *= np.abs(freqs) ** alpha
    lf     = np.fft.irfft(X, n=noise.size).astype(np.float32)
    rms_n  = float(np.sqrt(np.mean(noise ** 2) + EPS))
    rms_lf = float(np.sqrt(np.mean(lf   ** 2) + EPS))
    level  = random.uniform(0.3, 0.8)
    return (noise + lf * (rms_n * level / rms_lf)).astype(np.float32)


def random_eq(x: np.ndarray, p: float = 0.4,
              sample_rate: int = 16_000) -> np.ndarray:
    """FFT-domain random parametric EQ: 2-3 Gaussian-shaped bands, each ±3 dB.

    Simulates diverse microphone frequency responses without creating extreme
    frequency notches.  Phase-preserving (FFT multiply), RMS-normalised after.
    """
    if random.random() > p or x.size < 4:
        return x
    X      = np.fft.rfft(x)
    freqs  = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    gains  = np.ones(len(freqs), dtype=np.float32)
    for _ in range(random.randint(2, 3)):
        fc       = random.uniform(100.0, sample_rate / 2 - 100.0)
        gain_db  = random.uniform(-3.0, 3.0)
        bw_oct   = random.uniform(0.5, 2.0)
        gain_lin = 10.0 ** (gain_db / 20.0)
        sigma    = fc * (2.0 ** (bw_oct / 2.0) - 1.0)
        band     = np.exp(-0.5 * ((freqs - fc) / (sigma + 1e-6)) ** 2).astype(np.float32)
        gains   *= 1.0 + (gain_lin - 1.0) * band
    y       = np.fft.irfft(X * gains, n=len(x)).astype(np.float32)
    rms_in  = float(np.sqrt(np.mean(x ** 2) + EPS))
    rms_out = float(np.sqrt(np.mean(y ** 2) + EPS))
    return (y * rms_in / rms_out).astype(np.float32) if rms_out > EPS else x


def bandpass_filter(x: np.ndarray, p: float = 0.2,
                    sample_rate: int = 16_000) -> np.ndarray:
    """Zero out frequencies outside a random band to simulate microphone limits.

    Band edges are drawn from:  low ∈ [100, 500] Hz,  high ∈ [3000, 7500] Hz.
    This covers telephone (300–3400 Hz), laptop mics, earphones, etc.
    """
    if random.random() > p or x.size < 4:
        return x
    low_hz  = random.uniform(100.0,  500.0)
    high_hz = random.uniform(3000.0, min(7500.0, sample_rate / 2 - 1.0))
    X       = np.fft.rfft(x)
    freqs   = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    X[(freqs < low_hz) | (freqs > high_hz)] = 0.0
    y       = np.fft.irfft(X, n=len(x)).astype(np.float32)
    rms_in  = float(np.sqrt(np.mean(x ** 2) + EPS))
    rms_out = float(np.sqrt(np.mean(y ** 2) + EPS))
    return (y * rms_in / rms_out).astype(np.float32) if rms_out > EPS else x


def soft_clip(x: np.ndarray, p: float = 0.25) -> np.ndarray:
    """Soft saturation via tanh to simulate microphone overload or AD clipping."""
    if random.random() > p:
        return x
    drive     = random.uniform(1.5, 5.0)
    rms_in    = float(np.sqrt(np.mean(x ** 2) + EPS))
    y         = np.tanh(x * drive) / drive
    rms_out   = float(np.sqrt(np.mean(y ** 2) + EPS))
    return (y * rms_in / rms_out).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_FNAME_COLS = {"file_name", "filename", "file", "name"}
_TEXT_COLS  = {"transcription_cleaned", "transcription", "transcript", "text", "label"}


def _read_labels(csv_path: Path) -> Dict[str, str]:
    """Read filename→transcript mapping from CSV.

    Uses csv.DictReader so the header row is always skipped correctly.
    Column selection order:
      filename : file_name > filename > file > name
      text     : transcription_cleaned > transcription > transcript > text > label
    Falls back to positional col[0]→col[1] when no recognised header is found.
    """
    out: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

        fname_col = next(
            (f for f in fieldnames if f.lower().strip() in _FNAME_COLS), None
        )
        text_col = next(
            (f for f in fieldnames if f.lower().strip() in _TEXT_COLS), None
        )

        if fname_col is None:
            fh.seek(0)
            for i, row in enumerate(csv.reader(fh)):
                if not row or i == 0:
                    continue
                fname = row[0].strip()
                text  = row[1].strip() if len(row) > 1 else ""
                if fname:
                    out[fname] = text
            return out

        for row in reader:
            fname = row[fname_col].strip()
            text  = (row[text_col].strip() if text_col and row.get(text_col) else "")
            if fname:
                out[fname] = text

    return out


def _crop_or_pad(x: np.ndarray, length: int) -> np.ndarray:
    if x.size >= length:
        s = random.randint(0, x.size - length)
        return x[s:s + length]
    pad = length - x.size
    left = random.randint(0, pad)
    return np.pad(x, (left, pad - left), mode="constant").astype(np.float32)


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------
class SpeechDenoiseDataset(Dataset):
    """
    Returns (noisy, clean, transcript) per item.

    Parameters
    ----------
    data_root : str | Path        — root containing train/val/test subfolders
    split     : 'train' | 'val' | 'test'
    segment_len : int             — fixed sample length per item
    snr_range : (low, high)       — uniform-random SNR in dB
    augment   : bool              — turn the augmentation chain on/off (off for val/test)
    sample_rate : int             — assumed sample rate of stored .npy files
    """

    def __init__(self,
                 data_root: str | os.PathLike,
                 split: str = "train",
                 segment_len: int = SEGMENT_LEN,
                 snr_range: Tuple[float, float] = (-5.0, 15.0),
                 augment: Optional[bool] = None,
                 sample_rate: int = SAMPLE_RATE):
        self.root = Path(data_root) / split
        self.split = split
        self.segment_len = int(segment_len)
        self.snr_low, self.snr_high = snr_range
        self.sample_rate = sample_rate
        self.augment = (split == "train") if augment is None else augment

        self.clean_dir = self.root / "clean"
        self.noise_dir = self.root / "noise"
        if not self.clean_dir.is_dir():
            raise FileNotFoundError(f"Missing folder: {self.clean_dir}")
        if not self.noise_dir.is_dir():
            raise FileNotFoundError(f"Missing folder: {self.noise_dir}")

        self.clean_files: List[Path] = sorted(self.clean_dir.glob("*.npy"))
        self.noise_files: List[Path] = sorted(self.noise_dir.glob("*.npy"))
        if not self.clean_files:
            raise RuntimeError(f"No .npy files in {self.clean_dir}")
        if not self.noise_files:
            raise RuntimeError(f"No .npy files in {self.noise_dir}")

        labels_csv = self.root / "labels_npy.csv"
        self.labels: Dict[str, str] = _read_labels(labels_csv) if labels_csv.is_file() else {}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.clean_files)

    # ------------------------------------------------------------------
    def _load(self, path: Path) -> np.ndarray:
        x = np.load(path).astype(np.float32).squeeze()
        if x.ndim > 1:
            x = x.mean(axis=0)
        x = x - x.mean()
        return x

    # ------------------------------------------------------------------
    def _grab_noise(self, length: int) -> np.ndarray:
        """Dynamic mixing: blend 1-3 noise sources at random relative levels."""
        n_sources = random.randint(1, 3) if self.augment else 1
        combined = np.zeros(length, dtype=np.float32)
        for _ in range(n_sources):
            chunks: List[np.ndarray] = []
            gathered = 0
            while gathered < length:
                chunk = self._load(random.choice(self.noise_files))
                if chunk.size == 0:
                    continue
                chunks.append(chunk)
                gathered += chunk.size
            segment = _crop_or_pad(np.concatenate(chunks), length)
            weight = random.uniform(0.3, 1.0) if n_sources > 1 else 1.0
            combined += weight * segment

        rms = float(np.sqrt(np.mean(combined ** 2) + EPS))
        if rms > EPS:
            combined = combined / rms
        return combined.astype(np.float32)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        clean_path = self.clean_files[idx]
        clean = self._load(clean_path)

        # ---------- per-utterance augmentation BEFORE we fix length ----------
        if self.augment:
            clean = time_stretch(clean, p=0.7, low=0.85, high=1.15)
            clean = pitch_shift(clean, p=0.5, semitones_low=-3.0, semitones_high=3.0)

        clean = _crop_or_pad(clean, self.segment_len)
        noise = self._grab_noise(self.segment_len)

        # ---------- pre-mix noise shaping ----------
        if self.augment:
            noise = color_noise(noise, p=0.7)            # white → pink / brown
            noise = inject_low_freq_noise(noise, p=0.35) # additive brown/pink for <200 Hz focus
            noise = noise_offset(noise)                  # random temporal shift

        # ---------- noise mixing with biased SNR ----------
        snr_db = sample_snr(self.snr_low, self.snr_high) if self.augment \
                 else random.uniform(self.snr_low, self.snr_high)
        noisy, _ = mix_at_snr(clean, noise, snr_db)

        # ---------- post-mix augmentation (applied to the NOISY signal only) ----------
        if self.augment:
            noisy = apply_rir(noisy, p=0.4, sample_rate=self.sample_rate)
            noisy = random_gain(noisy, -6.0, 6.0)
            noisy = soft_clip(noisy, p=0.25)
            noisy = partial_dropout(noisy, p=0.0)   # disabled: creates STFT-breaking transients
            noisy = random_eq(noisy, p=0.4, sample_rate=self.sample_rate)
            noisy = bandpass_filter(noisy, p=0.2, sample_rate=self.sample_rate)
            noisy = freq_mask_waveform(noisy, p=0.15, max_bins=10)

        # ---------- safety: clip to avoid wraparound when saved as int wav ----------
        peak = max(np.max(np.abs(noisy)), np.max(np.abs(clean)), 1.0)
        if peak > 0.99:
            noisy = noisy / peak * 0.99
            clean = clean / peak * 0.99

        transcript = self.labels.get(clean_path.name, "")
        return (torch.from_numpy(noisy.astype(np.float32)),
                torch.from_numpy(clean.astype(np.float32)),
                transcript)


# ---------------------------------------------------------------------------
# Startup data-quality check
# ---------------------------------------------------------------------------
def validate_dataset(data_root: str | os.PathLike, split: str = "train",
                     n_check: int = 200) -> None:
    """Sample n_check files evenly and print a one-line quality report.

    Checks: silent files, NaN/Inf corruption, CSV label coverage.
    Call once at training startup — fast enough to not matter (~1-2 s).
    """
    root = Path(data_root) / split
    clean_dir = root / "clean"
    noise_dir = root / "noise"

    clean_files = sorted(clean_dir.glob("*.npy")) if clean_dir.is_dir() else []
    noise_files = sorted(noise_dir.glob("*.npy")) if noise_dir.is_dir() else []

    silent, corrupt, rms_vals = [], [], []
    step = max(1, len(clean_files) // n_check)
    for p in clean_files[::step][:n_check]:
        try:
            x = np.load(p).astype(np.float32).squeeze()
            if not np.isfinite(x).all():
                corrupt.append(p.name)
            elif float(np.abs(x).max()) < 1e-6:
                silent.append(p.name)
            else:
                rms_vals.append(float(np.sqrt(np.mean(x ** 2))))
        except Exception:
            corrupt.append(p.name)

    rms_med = float(np.median(rms_vals)) if rms_vals else 0.0
    n_bad = len(silent) + len(corrupt)
    status = "OK" if n_bad == 0 else f"WARNING {n_bad} bad"

    csv_path = root / "labels_npy.csv"
    labels = _read_labels(csv_path) if csv_path.is_file() else {}
    clean_names = {p.name for p in clean_files}
    n_matched = len(clean_names & set(labels.keys()))

    print(f"[data-check] {split:5s}: {len(clean_files):6d} clean  "
          f"{len(noise_files):5d} noise  "
          f"rms_med={rms_med:.4f}  labels={n_matched}/{len(clean_files)}  [{status}]")
    if silent:
        print(f"[data-check]   silent  : {silent[:5]}")
    if corrupt:
        print(f"[data-check]   corrupt : {corrupt[:5]}")


# ---------------------------------------------------------------------------
# Collate fn (handles transcript strings that default_collate refuses)
# ---------------------------------------------------------------------------
def denoise_collate(batch):
    noisy = torch.stack([b[0] for b in batch], dim=0)
    clean = torch.stack([b[1] for b in batch], dim=0)
    transcripts = [b[2] for b in batch]
    return noisy, clean, transcripts


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data_npy")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    ds = SpeechDenoiseDataset(args.data_root, split=args.split)
    print(f"{args.split}: {len(ds)} items")
    noisy, clean, txt = ds[0]
    print(" noisy:", noisy.shape, noisy.dtype, " clean:", clean.shape, " txt:", txt[:60])
