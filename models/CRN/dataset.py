"""
dataset.py
==========
On-the-fly speech-denoising dataset (speed-tuned).

Speed-tuned changes vs original:
- Module-level _HANN_CACHE + _RFFTFREQ_CACHE (eliminate per-call allocation).
- Noise pool pre-loaded into RAM at __init__ (no disk I/O in __getitem__).
- RIR pool pre-built once and sampled (no per-call RIR construction).
- Deterministic val/test path: same (clean_idx, noise, snr) every epoch.
- worker_init_fn for reproducible numpy/random per DataLoader worker.

CRITICAL DESIGN RULE: NO augmentation during preprocessing. Everything
happens in __getitem__ (per-epoch diversity).

Expected on-disk layout:
    DATA_ROOT/
        train/{clean,noise}/*.npy   labels_npy.csv
        val/   ...
        test/  ...
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
SAMPLE_RATE = 16_000
SEGMENT_SECONDS = 4.0
SEGMENT_LEN = int(SAMPLE_RATE * SEGMENT_SECONDS)
EPS = 1e-8

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
_HANN_CACHE: Dict[int, torch.Tensor] = {}


def _hann(n_fft: int) -> torch.Tensor:
    w = _HANN_CACHE.get(n_fft)
    if w is None:
        w = torch.hann_window(n_fft)
        _HANN_CACHE[n_fft] = w
    return w


_RFFTFREQ_CACHE: Dict[Tuple[int, float], np.ndarray] = {}


def _rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    key = (n, d)
    f = _RFFTFREQ_CACHE.get(key)
    if f is None:
        f = np.fft.rfftfreq(n, d=d).astype(np.float32)
        _RFFTFREQ_CACHE[key] = f
    return f


def worker_init_fn(worker_id: int) -> None:
    """Seed numpy + random per DataLoader worker (otherwise all workers share state)."""
    base = torch.initial_seed() & 0xFFFFFFFF
    np.random.seed((base + worker_id) & 0xFFFFFFFF)
    random.seed((base + worker_id) & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Augmentation primitives
# ---------------------------------------------------------------------------
def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + EPS))


def sample_snr(snr_low: float, snr_high: float,
               hard_low: float = -5.0, hard_high: float = 5.0,
               p_hard: float = 0.5) -> float:
    lo = max(snr_low, hard_low)
    hi = min(snr_high, hard_high)
    if lo < hi and random.random() < p_hard:
        return random.uniform(lo, hi)
    return random.uniform(snr_low, snr_high)


def noise_offset(noise: np.ndarray) -> np.ndarray:
    shift = random.randint(0, max(0, len(noise) - 1))
    return np.roll(noise, shift).astype(np.float32)


def noise_phase_jitter(x: np.ndarray, p: float = 0.5,
                        max_deg: float = 45.0) -> np.ndarray:
    """Random per-bin constant-magnitude phase rotation.

    Simulates microphone-to-noise-source distance fluctuations: the spectral
    envelope is unchanged but every STFT bin is rotated by an independent
    random angle in [-max_deg, +max_deg].  The BiLSTM/dilated-conv layers must
    become invariant to this rotation because the physical path length to any
    real noise source drifts over time.

    max_deg=45 covers up to λ/8 of path-length variation at any frequency,
    which is realistic for a source that moves ±a few centimetres.
    """
    if random.random() > p or x.size < 4:
        return x
    X = np.fft.rfft(x)
    max_rad = np.pi * max_deg / 180.0
    shift = np.random.uniform(-max_rad, max_rad, len(X))
    X = X * np.exp(1j * shift)
    return np.fft.irfft(X, n=len(x)).astype(np.float32)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> Tuple[np.ndarray, float]:
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


def freq_mask_waveform(x: np.ndarray, p: float = 0.3,
                       n_fft: int = 512, hop: int = 128,
                       max_bins: int = 24) -> np.ndarray:
    if random.random() > p:
        return x
    win = _hann(n_fft)
    t = torch.from_numpy(x)
    spec = torch.stft(t, n_fft=n_fft, hop_length=hop,
                      window=win, return_complex=True)
    F_bins = spec.shape[0]
    width = random.randint(1, max_bins)
    start = random.randint(0, max(0, F_bins - width))
    spec[start:start + width, :] = 0.0
    y = torch.istft(spec, n_fft=n_fft, hop_length=hop,
                    window=win, length=x.size)
    return y.numpy().astype(np.float32)


def _make_rir(sample_rate: int = 16_000) -> np.ndarray:
    """Generate one synthetic exponentially-decaying RIR with a few early reflections."""
    rt60 = random.uniform(0.08, 0.55)
    decay_samples = max(1, int(rt60 * sample_rate))
    t = np.arange(decay_samples, dtype=np.float32)
    rir = np.exp(-13.816 * t / decay_samples)
    n_ref = random.randint(2, 7)
    for _ in range(n_ref):
        delay = random.randint(int(0.002 * sample_rate), int(0.05 * sample_rate))
        if delay < decay_samples:
            amp = random.uniform(0.2, 0.75) * np.exp(-13.816 * delay / decay_samples)
            rir[delay] += amp
    rir /= (np.abs(rir).sum() + EPS)
    return rir.astype(np.float32)


def apply_rir(x: np.ndarray, p: float = 0.4,
              sample_rate: int = 16_000,
              rir_pool: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """Convolve with a synthetic RIR; pool-sampled if `rir_pool` is given."""
    if random.random() > p or x.size < 64:
        return x
    rir = random.choice(rir_pool) if rir_pool else _make_rir(sample_rate)

    n_fft = len(x) + len(rir) - 1
    Y = np.fft.rfft(x, n=n_fft) * np.fft.rfft(rir, n=n_fft)
    y = np.fft.irfft(Y)[:len(x)].astype(np.float32)

    rms_in = _rms(x)
    rms_out = _rms(y)
    return (y * rms_in / rms_out).astype(np.float32)


def color_noise(x: np.ndarray, p: float = 0.5,
                alpha_low: float = -1.0, alpha_high: float = 0.0) -> np.ndarray:
    if random.random() > p or x.size < 4:
        return x
    alpha = random.uniform(alpha_low, alpha_high)
    X = np.fft.rfft(x)
    freqs = _rfftfreq(len(x)).copy()
    if len(freqs) > 1:
        freqs[0] = freqs[1]
    else:
        freqs[0] = 1.0
    X = X * (np.abs(freqs) ** alpha)
    y = np.fft.irfft(X, n=len(x)).astype(np.float32)
    rms_in = _rms(x)
    rms_out = _rms(y)
    return (y * rms_in / rms_out).astype(np.float32)


def inject_low_freq_noise(noise: np.ndarray, p: float = 0.35) -> np.ndarray:
    if random.random() > p or noise.size < 4:
        return noise
    alpha = random.uniform(-1.0, -0.5)
    white = np.random.randn(noise.size).astype(np.float32)
    X = np.fft.rfft(white)
    freqs = _rfftfreq(noise.size).copy()
    if len(freqs) > 1:
        freqs[0] = freqs[1]
    else:
        freqs[0] = 1.0
    X = X * (np.abs(freqs) ** alpha)
    lf = np.fft.irfft(X, n=noise.size).astype(np.float32)
    rms_n = _rms(noise)
    rms_lf = _rms(lf)
    level = random.uniform(0.3, 0.8)
    return (noise + lf * (rms_n * level / rms_lf)).astype(np.float32)


def generate_lf_noise(length: int, sample_rate: int = 16_000) -> np.ndarray:
    """Synthetic low-frequency noise (< 200 Hz): HVAC rumble, fan, 50/60 Hz hum."""
    white = np.random.randn(length).astype(np.float32)
    X = np.fft.rfft(white)
    freqs = _rfftfreq(length, d=1.0 / sample_rate)
    freqs_safe = np.where(freqs < 1.0, 1.0, freqs).astype(np.float32)
    alpha = random.uniform(-1.8, -2.5)
    color = freqs_safe ** alpha
    lp = (1.0 / (1.0 + np.exp(np.clip((freqs - 200.0) / 25.0, -500.0, 500.0)))).astype(np.float32)
    X = X * color * lp
    if random.random() < 0.4:
        hum_hz = random.choice([50.0, 60.0])
        for k in range(1, 4):
            fk = hum_hz * k
            if fk >= sample_rate / 2:
                break
            bin_idx = int(fk * length / sample_rate)
            if 0 < bin_idx < len(X):
                amp = random.uniform(0.1, 0.4) / k
                phase = random.uniform(0.0, 2.0 * np.pi)
                X[bin_idx] += amp * np.exp(1j * phase)
    y = np.fft.irfft(X, n=length).astype(np.float32)
    rms = float(np.sqrt(np.mean(y ** 2) + EPS))
    return (y / rms).astype(np.float32)


def lowpass_filter(x: np.ndarray, cutoff_hz: float = 200.0,
                   sample_rate: int = 16_000, slope_hz: float = 80.0) -> np.ndarray:
    """FFT soft lowpass — passes energy below cutoff_hz."""
    X = np.fft.rfft(x)
    freqs = _rfftfreq(len(x), d=1.0 / sample_rate)
    gain = (1.0 / (1.0 + np.exp(np.clip((freqs - cutoff_hz) / (slope_hz / 6.0), -500.0, 500.0)))).astype(np.float32)
    y = np.fft.irfft(X * gain, n=len(x)).astype(np.float32)
    rms_in, rms_out = _rms(x), _rms(y)
    return (y * rms_in / (rms_out + EPS)).astype(np.float32)


def highpass_filter_signal(x: np.ndarray, cutoff_hz: float = 1000.0,
                            sample_rate: int = 16_000, slope_hz: float = 200.0) -> np.ndarray:
    """FFT soft highpass — passes energy above cutoff_hz."""
    X = np.fft.rfft(x)
    freqs = _rfftfreq(len(x), d=1.0 / sample_rate)
    gain = (1.0 / (1.0 + np.exp(np.clip(-(freqs - cutoff_hz) / (slope_hz / 6.0), -500.0, 500.0)))).astype(np.float32)
    y = np.fft.irfft(X * gain, n=len(x)).astype(np.float32)
    rms_in, rms_out = _rms(x), _rms(y)
    return (y * rms_in / (rms_out + EPS)).astype(np.float32)


def generate_click_noise(length: int, sample_rate: int = 16_000) -> np.ndarray:
    """Sudden transient clicks/pops — sparse impulses in time domain."""
    y = np.zeros(length, dtype=np.float32)
    n_clicks = random.randint(1, max(1, int(length / sample_rate * 8)))
    for _ in range(n_clicks):
        pos = random.randint(0, length - 1)
        click_len = random.randint(1, max(1, int(0.003 * sample_rate)))
        end = min(pos + click_len, length)
        amp = random.uniform(0.8, 3.0)
        y[pos:end] = np.random.randn(end - pos).astype(np.float32) * amp
    return y


def generate_hiss_noise(length: int, sample_rate: int = 16_000) -> np.ndarray:
    """High-frequency hiss: wideband noise above 1.5–4 kHz."""
    white = np.random.randn(length).astype(np.float32)
    X = np.fft.rfft(white)
    freqs = _rfftfreq(length, d=1.0 / sample_rate)
    cutoff = random.uniform(1500.0, 4000.0)
    gain = (1.0 / (1.0 + np.exp(np.clip(-(freqs - cutoff) / 250.0, -500.0, 500.0)))).astype(np.float32)
    y = np.fft.irfft(X * gain, n=length).astype(np.float32)
    rms = _rms(y)
    return (y / (rms + EPS)).astype(np.float32)


def generate_burst_noise(length: int, sample_rate: int = 16_000) -> np.ndarray:
    """Short sudden noise bursts — envelope-windowed wideband segments."""
    y = np.zeros(length, dtype=np.float32)
    for _ in range(random.randint(1, 5)):
        start = random.randint(0, max(0, length - 1))
        dur = random.randint(int(0.01 * sample_rate), int(0.10 * sample_rate))
        end = min(start + dur, length)
        burst_len = end - start
        if burst_len < 2:
            continue
        burst = np.random.randn(burst_len).astype(np.float32)
        env = np.hanning(burst_len).astype(np.float32)
        y[start:end] += burst * env * random.uniform(0.5, 2.5)
    return y


def random_eq(x: np.ndarray, p: float = 0.4,
              sample_rate: int = 16_000) -> np.ndarray:
    if random.random() > p or x.size < 4:
        return x
    X = np.fft.rfft(x)
    freqs = _rfftfreq(len(x), d=1.0 / sample_rate)
    gains = np.ones(len(freqs), dtype=np.float32)
    for _ in range(random.randint(2, 3)):
        fc = random.uniform(100.0, sample_rate / 2 - 100.0)
        gain_db = random.uniform(-3.0, 3.0)
        bw_oct = random.uniform(0.5, 2.0)
        gain_lin = 10.0 ** (gain_db / 20.0)
        sigma = fc * (2.0 ** (bw_oct / 2.0) - 1.0)
        band = np.exp(-0.5 * ((freqs - fc) / (sigma + 1e-6)) ** 2).astype(np.float32)
        gains *= 1.0 + (gain_lin - 1.0) * band
    y = np.fft.irfft(X * gains, n=len(x)).astype(np.float32)
    rms_in = _rms(x)
    rms_out = _rms(y)
    return (y * rms_in / rms_out).astype(np.float32) if rms_out > EPS else x


def bandpass_filter(x: np.ndarray, p: float = 0.2,
                    sample_rate: int = 16_000) -> np.ndarray:
    if random.random() > p or x.size < 4:
        return x
    low_hz = random.uniform(100.0, 500.0)
    high_hz = random.uniform(3000.0, min(7500.0, sample_rate / 2 - 1.0))
    X = np.fft.rfft(x).copy()
    freqs = _rfftfreq(len(x), d=1.0 / sample_rate)
    X[(freqs < low_hz) | (freqs > high_hz)] = 0.0
    y = np.fft.irfft(X, n=len(x)).astype(np.float32)
    rms_in = _rms(x)
    rms_out = _rms(y)
    return (y * rms_in / rms_out).astype(np.float32) if rms_out > EPS else x


def soft_clip(x: np.ndarray, p: float = 0.25) -> np.ndarray:
    if random.random() > p:
        return x
    drive = random.uniform(1.5, 5.0)
    rms_in = _rms(x)
    y = np.tanh(x * drive) / drive
    rms_out = _rms(y)
    return (y * rms_in / rms_out).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_FNAME_COLS = {"file_name", "filename", "file", "name"}
_TEXT_COLS = {"transcription_cleaned", "transcription", "transcript", "text", "label"}


def _read_labels(csv_path: Path) -> Dict[str, str]:
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
                text = row[1].strip() if len(row) > 1 else ""
                if fname:
                    out[fname] = text
            return out
        for row in reader:
            fname = row[fname_col].strip()
            text = (row[text_col].strip() if text_col and row.get(text_col) else "")
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


def _crop_or_pad_det(x: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    """Deterministic crop/pad using a provided Random instance (val/test)."""
    if x.size >= length:
        s = rng.randint(0, x.size - length)
        return x[s:s + length]
    pad = length - x.size
    left = rng.randint(0, pad)
    return np.pad(x, (left, pad - left), mode="constant").astype(np.float32)


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------
class SpeechDenoiseDataset(Dataset):
    """
    Returns (noisy, clean, transcript) per item.

    Speed-tuned: pre-loads noise pool + RIR pool, deterministic val/test path.
    """

    def __init__(self,
                 data_root: str | os.PathLike,
                 split: str = "train",
                 segment_len: int = SEGMENT_LEN,
                 snr_range: Tuple[float, float] = (-5.0, 15.0),
                 augment: Optional[bool] = None,
                 sample_rate: int = SAMPLE_RATE,
                 rir_pool_size: int = 256,
                 preload_noise: bool = True,
                 lf_noise_ratio: float = 0.70,
                 lf_cutoff_hz: float = 200.0):
        self.root = Path(data_root) / split
        self.split = split
        self.segment_len = int(segment_len)
        self.snr_low, self.snr_high = snr_range
        self.sample_rate = sample_rate
        self.augment = (split == "train") if augment is None else augment
        self.lf_noise_ratio = float(lf_noise_ratio)
        self.lf_cutoff_hz = float(lf_cutoff_hz)

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

        # Pre-load noise pool into RAM; reject nan/inf files that would corrupt training
        self.noise_pool: List[np.ndarray] = []
        _n_rejected = 0
        if preload_noise:
            for p in self.noise_files:
                try:
                    n = np.load(p).astype(np.float32).squeeze()
                    if n.ndim > 1:
                        n = n.mean(axis=0)
                    if n.size == 0 or not np.isfinite(n).all():
                        _n_rejected += 1
                        continue
                    n = n - n.mean()
                    rms = float(np.sqrt(np.mean(n ** 2) + EPS))
                    n = n / rms                   # unit-RMS at load time → stable mixing
                    self.noise_pool.append(n)
                except Exception:
                    _n_rejected += 1
            if not self.noise_pool:
                raise RuntimeError(f"All noise .npy files failed to load in {self.noise_dir}")
            if _n_rejected:
                print(f"[dataset] WARNING: rejected {_n_rejected} corrupted/non-finite noise files")

        # Categorise noise pool by LF energy fraction (< lf_cutoff_hz)
        self.real_lf_pool: List[np.ndarray] = []
        self.general_noise_pool: List[np.ndarray] = []
        _lf_thr = 0.40  # files with > 40% energy below lf_cutoff_hz → LF pool
        for _n in self.noise_pool:
            if _n.size < 4:
                self.general_noise_pool.append(_n)
                continue
            _fft_len = min(_n.size, 4096)
            _X = np.fft.rfft(_n[:_fft_len])
            _freqs_cat = np.fft.rfftfreq(_fft_len, d=1.0 / self.sample_rate)
            _energy = np.abs(_X) ** 2
            _lf_e = _energy[_freqs_cat < self.lf_cutoff_hz].sum()
            if _lf_e / (_energy.sum() + EPS) > _lf_thr:
                self.real_lf_pool.append(_n)
            else:
                self.general_noise_pool.append(_n)
        if not self.real_lf_pool:
            self.real_lf_pool = self.general_noise_pool
        if not self.general_noise_pool:
            self.general_noise_pool = self.real_lf_pool
        print(f"[dataset] noise pools — real_lf={len(self.real_lf_pool)}  "
              f"general={len(self.general_noise_pool)}")

        # Pre-build RIR pool (deterministic — same pool every run, for reproducibility)
        rng_state = random.getstate()
        random.seed(0)
        self.rir_pool: List[np.ndarray] = [_make_rir(sample_rate) for _ in range(rir_pool_size)]
        random.setstate(rng_state)

    def __len__(self) -> int:
        return len(self.clean_files)

    def _load(self, path: Path) -> np.ndarray:
        x = np.load(path).astype(np.float32).squeeze()
        if x.ndim > 1:
            x = x.mean(axis=0)
        x = x - x.mean()
        return x

    def _grab_noise(self, length: int, rng: Optional[random.Random] = None) -> np.ndarray:
        rng_o = rng if rng is not None else random
        n_sources = rng_o.randint(1, 3) if self.augment else 1
        combined = np.zeros(length, dtype=np.float32)

        if self.noise_pool:
            pool = self.noise_pool
            n_files = len(pool)
            for _ in range(n_sources):
                chunks: List[np.ndarray] = []
                gathered = 0
                while gathered < length:
                    chunk = pool[rng_o.randrange(n_files)]
                    if chunk.size == 0:
                        continue
                    chunks.append(chunk)
                    gathered += chunk.size
                merged = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
                if merged.size > length:
                    s = rng_o.randint(0, merged.size - length)
                    segment = merged[s:s + length].copy()
                else:
                    pad = length - merged.size
                    segment = np.pad(merged, (0, pad), mode="constant").astype(np.float32)
                weight = rng_o.uniform(0.3, 1.0) if n_sources > 1 else 1.0
                combined += weight * segment
        else:
            for _ in range(n_sources):
                chunks: List[np.ndarray] = []
                gathered = 0
                while gathered < length:
                    chunk = self._load(rng_o.choice(self.noise_files))
                    if chunk.size == 0:
                        continue
                    chunks.append(chunk)
                    gathered += chunk.size
                segment = _crop_or_pad(np.concatenate(chunks), length)
                weight = rng_o.uniform(0.3, 1.0) if n_sources > 1 else 1.0
                combined += weight * segment

        rms = _rms(combined)
        if rms > EPS:
            combined = combined / rms
        return combined.astype(np.float32)

    def _get_pool_noise(self, length: int, rng,
                        pool: Optional[List[np.ndarray]] = None) -> np.ndarray:
        pool = pool if pool is not None else self.noise_pool
        rng_o = rng if rng is not None else random
        chunks: List[np.ndarray] = []
        gathered = 0
        n_files = len(pool)
        while gathered < length:
            chunk = pool[rng_o.randrange(n_files)]
            if chunk.size == 0:
                continue
            chunks.append(chunk)
            gathered += chunk.size
        merged = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if merged.size > length:
            s = rng_o.randint(0, merged.size - length)
            return merged[s:s + length].copy().astype(np.float32)
        pad = length - merged.size
        return np.pad(merged, (0, pad), mode="constant").astype(np.float32)

    def _grab_lf_noise(self, length: int) -> np.ndarray:
        """Real LF-dominant noise (<lf_cutoff_hz), blended with synthetic LF."""
        synth = generate_lf_noise(length, self.sample_rate)
        if self.real_lf_pool:
            pool_seg = self._get_pool_noise(length, random, self.real_lf_pool)
            blend = random.uniform(0.3, 0.7)
            noise = blend * pool_seg + (1.0 - blend) * synth
        else:
            noise = synth
        rms = _rms(noise)
        return (noise / (rms + EPS)).astype(np.float32)

    def _grab_hf_noise(self, length: int) -> np.ndarray:
        """General/HF noise from real general pool, blended with synthetic HF archetypes."""
        archetype = random.choice(["click", "hiss", "burst"])
        if archetype == "click":
            synth = generate_click_noise(length, self.sample_rate)
        elif archetype == "hiss":
            synth = generate_hiss_noise(length, self.sample_rate)
        else:
            synth = generate_burst_noise(length, self.sample_rate)
        if self.general_noise_pool and random.random() < 0.6:
            pool_seg = self._get_pool_noise(length, random, self.general_noise_pool)
            blend = random.uniform(0.4, 0.8)
            noise = blend * pool_seg + (1.0 - blend) * synth
        else:
            noise = synth
        rms = _rms(noise)
        return (noise / (rms + EPS)).astype(np.float32)

    # ------------------------------------------------------------------
    # Deterministic (val/test) noise builders — accept a random.Random rng
    # so val noise is the same every epoch for a given idx.
    # Structure mirrors the train LF+HF blend so val measures the SAME
    # noise distribution the model is trained to suppress.
    # ------------------------------------------------------------------
    def _grab_lf_noise_det(self, length: int, rng: random.Random) -> np.ndarray:
        synth = generate_lf_noise(length, self.sample_rate)
        if self.real_lf_pool:
            pool_seg = self._get_pool_noise(length, rng, self.real_lf_pool)
            blend = rng.uniform(0.3, 0.7)
            noise = blend * pool_seg + (1.0 - blend) * synth
        else:
            noise = synth
        rms = _rms(noise)
        return (noise / (rms + EPS)).astype(np.float32)

    def _grab_hf_noise_det(self, length: int, rng: random.Random) -> np.ndarray:
        archetype = rng.choice(["click", "hiss", "burst"])
        if archetype == "click":
            synth = generate_click_noise(length, self.sample_rate)
        elif archetype == "hiss":
            synth = generate_hiss_noise(length, self.sample_rate)
        else:
            synth = generate_burst_noise(length, self.sample_rate)
        if self.general_noise_pool and rng.random() < 0.6:
            pool_seg = self._get_pool_noise(length, rng, self.general_noise_pool)
            blend = rng.uniform(0.4, 0.8)
            noise = blend * pool_seg + (1.0 - blend) * synth
        else:
            noise = synth
        rms = _rms(noise)
        return (noise / (rms + EPS)).astype(np.float32)

    def __getitem__(self, idx: int):
        clean_path = self.clean_files[idx]
        clean = self._load(clean_path)

        if self.augment:
            clean = time_stretch(clean, p=0.70, low=0.85, high=1.15)
            clean = pitch_shift(clean, p=0.50, semitones_low=-3.0, semitones_high=3.0)
            clean = _crop_or_pad(clean, self.segment_len)
            clean = random_gain(clean, -6.0, 6.0)

            # --- LF-focused noise synthesis -----------------------------------------
            lf_noise  = self._grab_lf_noise(self.segment_len)
            gen_noise = self._grab_hf_noise(self.segment_len)

            # LF-specific gain mutation: forces BiLSTM to handle the full range
            # from faint -8 dB hums to sudden +8 dB mechanical rumbles without
            # relying on a narrow gain band seen during training.
            lf_noise = random_gain(lf_noise, low_db=-8.0, high_db=8.0)

            # Phase jitter on the LF component only — simulates physical distance
            # variation between the noise source and the reference microphone.
            lf_noise = noise_phase_jitter(lf_noise, p=0.6, max_deg=45.0)

            # Independent temporal shifts per component so the BiLSTM never
            # pattern-matches a fixed phase-alignment between LF rumble and HF noise.
            lf_noise  = noise_offset(lf_noise)
            gen_noise = noise_offset(gen_noise)

            lf_w  = random.uniform(0.45, 0.75)
            noise = lf_w * lf_noise + (1.0 - lf_w) * gen_noise
            _rms_n = _rms(noise)
            if _rms_n > EPS:
                noise = noise / _rms_n
            noise = random_gain(noise, -4.0, 4.0)   # overall level randomisation
            snr_db = sample_snr(self.snr_low, self.snr_high)
            noisy, _ = mix_at_snr(clean, noise, snr_db)
            noisy = apply_rir(noisy, p=0.4, sample_rate=self.sample_rate, rir_pool=self.rir_pool)
            noisy = soft_clip(noisy, p=0.25)
            noisy = random_eq(noisy, p=0.4, sample_rate=self.sample_rate)
        else:
            # Deterministic val/test: same crop, noise pairing, snr per idx every epoch.
            # Use the SAME LF+HF blend structure as training so val measures the
            # distribution the model is trained on (not a random pool draw).
            rng = random.Random(idx * 1000003 + 7)
            clean = _crop_or_pad_det(clean, self.segment_len, rng)

            use_lf = rng.random() < self.lf_noise_ratio
            if use_lf:
                lf_n  = self._grab_lf_noise_det(self.segment_len, rng)
                hf_n  = self._grab_hf_noise_det(self.segment_len, rng)
                lf_w  = rng.uniform(0.45, 0.75)
                noise = lf_w * lf_n + (1.0 - lf_w) * hf_n
            else:
                noise = self._grab_hf_noise_det(self.segment_len, rng)
            rms_n = _rms(noise)
            if rms_n > EPS:
                noise = noise / rms_n

            snr_db = rng.uniform(self.snr_low, self.snr_high)
            noisy, _ = mix_at_snr(clean, noise, snr_db)

        # Safety: reject any batch that went NaN (corrupted noise / edge case)
        if not (np.isfinite(noisy).all() and np.isfinite(clean).all()):
            noisy = clean.copy()   # fallback: pass-through (SI-SDR = inf → tanh ≈ 1)

        peak = max(float(np.max(np.abs(noisy))), float(np.max(np.abs(clean))), 1.0)
        if peak > 0.99:
            noisy = noisy / peak * 0.99
            clean = clean / peak * 0.99

        transcript = self.labels.get(clean_path.name, "")
        return (torch.from_numpy(noisy.astype(np.float32)),
                torch.from_numpy(clean.astype(np.float32)),
                transcript)


# ---------------------------------------------------------------------------
def validate_dataset(data_root: str | os.PathLike, split: str = "train",
                     n_check: int = 200) -> None:
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


def denoise_collate(batch):
    noisy = torch.stack([b[0] for b in batch], dim=0)
    clean = torch.stack([b[1] for b in batch], dim=0)
    transcripts = [b[2] for b in batch]
    return noisy, clean, transcripts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data_npy")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    ds = SpeechDenoiseDataset(args.data_root, split=args.split)
    print(f"{args.split}: {len(ds)} items, noise_pool={len(ds.noise_pool)}, rir_pool={len(ds.rir_pool)}")
    noisy, clean, txt = ds[0]
    print(" noisy:", noisy.shape, noisy.dtype, " clean:", clean.shape, " txt:", txt[:60])
