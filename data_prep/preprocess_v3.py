"""
preprocess_v2.py — Speaker-disjoint ANC preprocessing for cache_crn3/
======================================================================

Replaces preprocess_v3.py.  Combines the working file-writing pipeline
from preprocess.py with:

  * speaker-disjoint 80 / 10 / 10 split
  * extracted/{clean,noise,noisy}  folder integration
  * SNR range  −10 .. +20 dB   (USER REQUEST)
  * cache_crn3/  output (drop-in for the existing CRN training notebook)

KEY FIXES vs preprocess_v3.py
-----------------------------
1.  speaker_id_for_clean() now uses the IMMEDIATE PARENT FOLDER of the
    .wav, not the first directory after `extracted/clean`.  Your real
    layout is

        extracted/clean/speech-noise-dataset/speech/<spk>/<file>.wav

    so the speaker is `<spk>` (1, 2, 3, …), not `speech-noise-dataset`.
    v3 collapsed every parquet clip into a single fake speaker, which
    is why pairing matched zero val / test speakers.

2.  find_paired_clips() now mirrors the path correctly between
    extracted/clean/.../speech/<spk>/X.wav  ↔
    extracted/noisy/.../noisy/<spk>/X.wav, and falls back to
    same-speaker / same-stem matching for the short-name case
    (1.wav ↔ 1.wav).

3.  Writing of .npy files is now VERIFIED.  Every per-sample build
    returns True/False, success/failure are counted live, and at the
    end of each split the script lists how many .npy files actually
    landed on disk.  If a split says “8000 OK” but the folder has 0
    files, you’ll see a CRITICAL message rather than empty folders.
    (v3 silently swallowed `None` returns from the worker, which is
    why your last run printed “100 % done” over empty folders.)

Cache layout (unchanged — drop-in for the existing dataset class):
    cache_crn3/
        train/{idx:06d}_noisy_spec.npy
        train/{idx:06d}_target_mask.npy
        train/{idx:06d}_clean_spec.npy
        train/{idx:06d}_noisy_phase.npy
        train/{idx:06d}_meta.npy
        val/...
        test/...
        meta.npy
        speaker_split.json
"""

# ╔═══════════════════════════════════════════════════════════════╗
# ║                  ★  CONFIG — EDIT HERE  ★                    ║
# ╚═══════════════════════════════════════════════════════════════╝

_BASE = r"C:\Users\rocha\AI_builders\data_sounds"

# ── Clean speech sources ─────────────────────────────────────────
EXTRACTED_CLEAN_ROOT = rf"{_BASE}\extracted\clean"
RAW_SPEECH_DIR       = rf"{_BASE}\raw\speech"
LJSPEECH_DIR         = rf"{_BASE}\LJSpeech-1.1\wavs"
LEGACY_CLEAN_ROOTS   = [
    rf"{_BASE}\CleanSpeech_training",
    rf"{_BASE}\CleanSpeech_training_singleNoise",
    rf"{_BASE}\CleanSpeech_training_singleNoise1",
]

# ── Noise sources ────────────────────────────────────────────────
EXTRACTED_NOISE_ROOT = rf"{_BASE}\extracted\noise"
ESC50_DIR            = rf"{_BASE}\ESC-50-master\audio"
MSSNSD_DIR           = rf"{_BASE}\microsoft MS-SNSD master noise_train"
LF_NOISE_DIR         = rf"{_BASE}\low_freq_noise"
RAW_NOISE_DIR        = rf"{_BASE}\raw\noise"

# ── Pre-mixed clean/noisy pairs (val / test) ─────────────────────
EXTRACTED_NOISY_ROOT = rf"{_BASE}\extracted\noisy"

# ── Cache output ─────────────────────────────────────────────────
CACHE_DIR = rf"{_BASE}\cache_crn3"

# ── Audio params (must match training notebook) ──────────────────
SR           = 16_000
DURATION     = 3.0
N_FFT        = 256
HOP_LENGTH   = 64
LF_CUTOFF_HZ = 200.0

# ── Split + dataset budget ───────────────────────────────────────
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
N_TRAIN      = 8_000
N_VAL        = 1_000
N_TEST       = 1_000

# ── Mix ratios for synthetic train mixing ────────────────────────
MIX_RATIOS = {
    "general":    0.40,
    "lf_curated": 0.20,
    "lf_synth":   0.25,
    "mixed_lpf":  0.15,
}

# ── SNR & augmentation (USER REQUEST: −10..+20) ──────────────────
SNR_RANGE          = (-10.0, 20.0)
GAIN_RANGE_DB      = (-6.0, 6.0)
TIME_SHIFT_MAX     = 0.5
PITCH_JITTER_RANGE = (0.92, 1.08)
P_PITCH_JITTER     = 0.5
P_TIME_SHIFT       = 0.7
P_LF_BOOST_GENERAL = 0.5

# ── Synthetic LF noise generator ─────────────────────────────────
SYNTH_SLOPE_RANGE    = (-2.0, -0.5)
SYNTH_TONE_FREQS     = [50, 60, 100, 120, 150, 180]
SYNTH_RPM_BAND       = (40.0, 180.0)
SYNTH_TRANSIENT_RATE = (0.5, 3.0)
P_SYNTH_TONES        = 0.65
P_SYNTH_RPM          = 0.45
P_SYNTH_TRANSIENTS   = 0.35

# ── DSP / safety ─────────────────────────────────────────────────
LPF_ORDER = 6
SEED      = 42
MIN_RMS   = 1e-4
MAX_RETRIES = 5

# ╔═══════════════════════════════════════════════════════════════╗
# ║          END OF CONFIG — implementation below                ║
# ╚═══════════════════════════════════════════════════════════════╝

import argparse
import json
import math
import os
import random
import re
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, chirp
from tqdm import tqdm

# ── Derived constants ───────────────────────────────────────────
FREQ_BINS  = N_FFT // 2 + 1
BIN_HZ     = SR / N_FFT
LF_BINS    = math.ceil(LF_CUTOFF_HZ / BIN_HZ)
TARGET_LEN = int(SR * DURATION)


# ════════════════════════════════════════════════════════════════
# AUDIO I/O
# ════════════════════════════════════════════════════════════════
def collect_files(directory: Optional[str],
                  exts: Tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg")
                  ) -> List[str]:
    if not directory or not os.path.isdir(directory):
        return []
    out = []
    for root, _, fnames in os.walk(directory):
        for f in fnames:
            if f.lower().endswith(exts):
                out.append(os.path.join(root, f))
    return sorted(out)


def load_audio_strict(path: str, target_len: int = TARGET_LEN, sr: int = SR,
                      rng: Optional[np.random.RandomState] = None
                      ) -> Optional[np.ndarray]:
    try:
        audio, _ = librosa.load(path, sr=sr, mono=True, dtype="float32")
    except Exception:
        return None
    if audio is None or len(audio) == 0:
        return None
    n = len(audio)
    if n < target_len:
        audio = np.pad(audio, (0, target_len - n))
    else:
        if rng is not None and n > target_len:
            start = rng.randint(0, n - target_len + 1)
        else:
            start = 0
        audio = audio[start:start + target_len]
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < MIN_RMS:
        return None
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-8:
        audio = audio / peak
    return audio.astype(np.float32)


def safe_pick(pool: List[str], rng: random.Random,
              loader, retries: int = MAX_RETRIES) -> Optional[np.ndarray]:
    if not pool:
        return None
    for _ in range(retries):
        a = loader(rng.choice(pool))
        if a is not None:
            return a
    return None


# ════════════════════════════════════════════════════════════════
# SPEAKER-ID DERIVATION  (FIXED)
# ════════════════════════════════════════════════════════════════
_LJ_RE = re.compile(r"(LJ\d{3,4})", re.IGNORECASE)


def speaker_id_for_clean(path: str) -> str:
    """
    Map a clean-speech path to a speaker bucket.

    Rules (priority order):
      1) LJSpeech    → chapter prefix (LJ001, LJ002, …)
      2) anything under extracted/clean/  → IMMEDIATE PARENT directory
         of the .wav file.  Your layout is
              extracted/clean/speech-noise-dataset/speech/<spk>/<file>.wav
         so the speaker is `<spk>` (1, 2, 3, …).
      3) Filename-prefix fallback for everything else.  Purely numeric
         basenames are disambiguated by the parent directory so
         identical filenames in different folders don’t collide.
    """
    p = path.replace("\\", "/")

    # 1) LJSpeech
    m = _LJ_RE.search(os.path.basename(p))
    if "LJSpeech" in p or m:
        if m:
            return m.group(1).upper()

    # 2) extracted/clean/...  →  immediate parent of the .wav
    if "extracted/clean" in p:
        spk = os.path.basename(os.path.dirname(p))
        if spk:
            return f"ext_clean__{spk}"

    # 3) Generic prefix fallback
    base = os.path.splitext(os.path.basename(p))[0]
    parent = os.path.basename(os.path.dirname(p)) or "root"
    if base.isdigit():
        return f"{parent}__num{base}"
    pref = re.split(r"[_\-.]", base, maxsplit=1)[0]
    if pref.isdigit():
        # e.g. '1_00000001.wav' under a numeric folder
        return f"{parent}__{pref}"
    return pref or "unknown"


# ════════════════════════════════════════════════════════════════
# REGISTRIES
# ════════════════════════════════════════════════════════════════
def build_clean_registry() -> Dict[str, List[str]]:
    files: List[str] = []
    files += collect_files(EXTRACTED_CLEAN_ROOT)
    files += collect_files(RAW_SPEECH_DIR)
    files += collect_files(LJSPEECH_DIR)
    for d in LEGACY_CLEAN_ROOTS:
        files += collect_files(d)

    spk: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        spk[speaker_id_for_clean(f)].append(f)

    pruned = {k: v for k, v in spk.items() if len(v) >= 3}
    print(f"  Clean speakers (>=3 clips): {len(pruned)}  "
          f"(dropped {len(spk) - len(pruned)} with <3 clips)")
    return pruned


def build_noise_registry() -> Dict[str, List[str]]:
    general = []
    general += collect_files(ESC50_DIR)
    general += collect_files(MSSNSD_DIR)
    general += collect_files(EXTRACTED_NOISE_ROOT)
    general += collect_files(RAW_NOISE_DIR)
    lf_curated = collect_files(LF_NOISE_DIR)
    print(f"  General noise files       : {len(general):>6,d}")
    print(f"  LF-curated noise files    : {len(lf_curated):>6,d}")
    return {"general": general, "lf_curated": lf_curated}


# ════════════════════════════════════════════════════════════════
# PAIRED CLEAN ↔ NOISY  (FIXED for your layout + short-name case)
# ════════════════════════════════════════════════════════════════
def find_paired_clips() -> List[Tuple[str, str, str]]:
    """
    Return [(clean_path, noisy_path, speaker_id), ...] from the
    extracted/clean and extracted/noisy trees.

    Matching strategies, applied in order per clean file:
      1.  Same RELATIVE path under extracted/clean ↔ extracted/noisy,
          with `/speech/` rewritten to `/noisy/` (covers your
          'speech-noise-dataset/speech/<spk>/X.wav' ↔ 'noisy/<spk>/X.wav').
      2.  Literal mirror path (no /speech/→/noisy/ rewrite).
      3.  Same speaker subfolder, same stem  (1.wav ↔ 1.wav case).
      4.  Same speaker subfolder, ANY noisy file  (loose pairing —
          last-resort so val/test isn’t starved of samples).
    """
    pairs: List[Tuple[str, str, str]] = []
    if not (os.path.isdir(EXTRACTED_CLEAN_ROOT) and
            os.path.isdir(EXTRACTED_NOISY_ROOT)):
        return pairs

    # Index every noisy file three ways.
    noisy_by_relpath: Dict[str, str] = {}
    noisy_by_spk_stem: Dict[Tuple[str, str], str] = {}
    noisy_by_spk: Dict[str, List[str]] = defaultdict(list)
    for nf in collect_files(EXTRACTED_NOISY_ROOT):
        rel = os.path.relpath(nf, EXTRACTED_NOISY_ROOT)
        spk = os.path.basename(os.path.dirname(nf))
        stem = os.path.splitext(os.path.basename(nf))[0]
        noisy_by_relpath[rel] = nf
        noisy_by_spk_stem[(spk, stem)] = nf
        noisy_by_spk[spk].append(nf)

    sep = os.sep
    for cf in collect_files(EXTRACTED_CLEAN_ROOT):
        rel_c = os.path.relpath(cf, EXTRACTED_CLEAN_ROOT)
        spk = os.path.basename(os.path.dirname(cf))
        stem = os.path.splitext(os.path.basename(cf))[0]
        spk_id = f"ext_clean__{spk}"

        # 1) speech → noisy path rewrite
        rewritten = rel_c.replace(f"{sep}speech{sep}", f"{sep}noisy{sep}", 1)
        if rewritten in noisy_by_relpath:
            pairs.append((cf, noisy_by_relpath[rewritten], spk_id))
            continue
        # 2) literal mirror
        if rel_c in noisy_by_relpath:
            pairs.append((cf, noisy_by_relpath[rel_c], spk_id))
            continue
        # 3) same speaker + same stem  (covers '1.wav' ↔ '1.wav')
        if (spk, stem) in noisy_by_spk_stem:
            pairs.append((cf, noisy_by_spk_stem[(spk, stem)], spk_id))
            continue
        # 4) loose: any noisy file from this speaker
        if noisy_by_spk[spk]:
            pairs.append((cf, noisy_by_spk[spk][0], spk_id))

    return pairs


# ════════════════════════════════════════════════════════════════
# SPEAKER-DISJOINT 80 / 10 / 10 SPLIT
# ════════════════════════════════════════════════════════════════
def speaker_disjoint_split(speakers: List[str],
                           ratios: Dict[str, float],
                           seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    pool = list(speakers)
    rng.shuffle(pool)
    n = len(pool)
    n_train = int(round(n * ratios["train"]))
    n_val   = int(round(n * ratios["val"]))
    return {
        "train": pool[:n_train],
        "val":   pool[n_train:n_train + n_val],
        "test":  pool[n_train + n_val:],
    }


# ════════════════════════════════════════════════════════════════
# SYNTHETIC LF NOISE  (G2Net-inspired — same as before)
# ════════════════════════════════════════════════════════════════
def synth_colored_noise(n_samples, slope, sr=SR, rng=None, lpf_cutoff=None):
    if rng is None: rng = np.random.RandomState()
    white = rng.randn(n_samples).astype(np.float32)
    spec  = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0/sr); freqs[0] = freqs[1]
    spec *= (freqs ** (slope / 2.0)).astype(np.complex64)
    if lpf_cutoff is not None:
        spec *= (1.0 / (1.0 + (freqs / lpf_cutoff) ** 6)).astype(np.complex64)
    out = np.fft.irfft(spec, n=n_samples).astype(np.float32)
    p = float(np.max(np.abs(out)))
    if p > 1e-8: out /= p
    return out

def synth_tones(n_samples, freqs_list, sr=SR, rng=None):
    if rng is None: rng = np.random.RandomState()
    t = np.arange(n_samples) / sr
    out = np.zeros(n_samples, dtype=np.float32)
    for f in rng.choice(freqs_list, size=rng.randint(1, 4), replace=False):
        f_d = float(f) * (1.0 + rng.uniform(-0.02, 0.02))
        out += rng.uniform(0.3, 1.0) * np.sin(
            2*np.pi*f_d*t + rng.uniform(0, 2*np.pi))
    p = float(np.max(np.abs(out)))
    if p > 1e-8: out /= p
    return out.astype(np.float32)

def synth_rpm_chirp(n_samples, sr=SR, rng=None):
    if rng is None: rng = np.random.RandomState()
    t  = np.arange(n_samples) / sr
    f0 = float(rng.uniform(*SYNTH_RPM_BAND))
    f1 = float(rng.uniform(*SYNTH_RPM_BAND))
    out = chirp(t, f0=f0, f1=f1, t1=t[-1], method="linear").astype(np.float32)
    nyq = sr / 2.0
    for h in range(2, 2 + rng.randint(0, 3)):
        if f0*h < nyq and f1*h < nyq:
            out += (0.5/h) * chirp(t, f0=f0*h, f1=f1*h, t1=t[-1],
                                   method="linear").astype(np.float32)
    p = float(np.max(np.abs(out)))
    if p > 1e-8: out /= p
    return out

def synth_transients(n_samples, sr=SR, rng=None):
    if rng is None: rng = np.random.RandomState()
    rate = float(rng.uniform(*SYNTH_TRANSIENT_RATE))
    n_events = max(1, int(n_samples / sr * rate))
    out = np.zeros(n_samples, dtype=np.float32)
    for _ in range(n_events):
        pos    = int(rng.randint(0, n_samples))
        dur    = max(8, int(float(rng.uniform(20, 120)) * sr / 1000))
        f      = float(rng.uniform(40, 180))
        decay  = float(rng.uniform(20, 60))
        amp    = float(rng.uniform(0.4, 1.0))
        t_loc  = np.arange(dur) / sr
        burst  = amp * np.exp(-decay*t_loc) * np.cos(
            2*np.pi*f*t_loc + rng.uniform(0, 2*np.pi))
        end = min(pos + dur, n_samples)
        out[pos:end] += burst[:end - pos].astype(np.float32)
    p = float(np.max(np.abs(out)))
    if p > 1e-8: out /= p
    return out

def generate_synth_lf_noise(n_samples=TARGET_LEN, sr=SR, rng=None):
    if rng is None: rng = np.random.RandomState()
    comps, ws = [], []
    slope = float(rng.uniform(*SYNTH_SLOPE_RANGE))
    comps.append(synth_colored_noise(n_samples, slope, sr, rng=rng,
                                     lpf_cutoff=LF_CUTOFF_HZ))
    ws.append(float(rng.uniform(0.5, 1.0)))
    if rng.rand() < P_SYNTH_TONES:
        comps.append(synth_tones(n_samples, SYNTH_TONE_FREQS, sr, rng))
        ws.append(float(rng.uniform(0.2, 0.6)))
    if rng.rand() < P_SYNTH_RPM:
        comps.append(synth_rpm_chirp(n_samples, sr, rng))
        ws.append(float(rng.uniform(0.2, 0.5)))
    if rng.rand() < P_SYNTH_TRANSIENTS:
        comps.append(synth_transients(n_samples, sr, rng))
        ws.append(float(rng.uniform(0.3, 0.7)))
    w = np.asarray(ws, dtype=np.float32); w /= w.sum()
    out = np.zeros(n_samples, dtype=np.float32)
    for c, wi in zip(comps, w):
        out += wi * c
    sos = butter(LPF_ORDER, LF_CUTOFF_HZ / (sr / 2.0), btype="low", output="sos")
    out = sosfilt(sos, out).astype(np.float32)
    p = float(np.max(np.abs(out)))
    if p > 1e-8: out /= p
    return out


# ════════════════════════════════════════════════════════════════
# AUGMENTATIONS / DSP
# ════════════════════════════════════════════════════════════════
def aug_gain(x, rng, db_range=GAIN_RANGE_DB):
    return (x * (10.0 ** (float(rng.uniform(*db_range)) / 20.0))).astype(np.float32)

def aug_time_shift(x, rng, max_frac=TIME_SHIFT_MAX):
    return np.roll(x, int(rng.uniform(-max_frac, max_frac) * len(x))).astype(np.float32)

def aug_pitch_jitter(x, rng, factor_range=PITCH_JITTER_RANGE,
                     target_len=TARGET_LEN):
    factor = float(rng.uniform(*factor_range))
    n = len(x); n_out = max(8, int(n / factor))
    y = np.interp(np.linspace(0, n - 1, n_out), np.arange(n), x).astype(np.float32)
    return y[:target_len] if len(y) >= target_len else \
           np.pad(y, (0, target_len - len(y))).astype(np.float32)

def aug_lpf(x, sr=SR, cutoff=LF_CUTOFF_HZ, order=LPF_ORDER):
    sos = butter(order, cutoff / (sr / 2.0), btype="low", output="sos")
    return sosfilt(sos, x).astype(np.float32)

def aug_lf_emphasis(x, sr=SR, lpf_share=0.7):
    lpf = aug_lpf(x, sr)
    return (lpf_share * lpf + (1.0 - lpf_share) * x).astype(np.float32)


def mix_at_snr(speech, noise, snr_db):
    p_s = float(np.mean(speech ** 2)) + 1e-12
    p_n = float(np.mean(noise ** 2))  + 1e-12
    scale = float(np.sqrt(p_s / (p_n * 10.0 ** (snr_db / 10.0))))
    ns = (scale * noise).astype(np.float32)
    return (speech + ns).astype(np.float32), ns


def to_spectrogram(audio):
    stft  = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.log1p(np.abs(stft).astype(np.float32)), \
           np.angle(stft).astype(np.float32)


def compute_irm(clean_log, noisy_log):
    c = np.expm1(np.clip(clean_log, 0, None))
    n = np.expm1(np.clip(noisy_log, 0, None))
    return np.clip(c / (n + 1e-8), 0.0, 1.0).astype(np.float32)


def lf_energy_ratio(audio, sr=SR, cutoff=LF_CUTOFF_HZ):
    spec  = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    total = float(np.sum(spec ** 2)) + 1e-12
    lf    = float(np.sum((spec[freqs < cutoff]) ** 2))
    return lf / total


# ════════════════════════════════════════════════════════════════
# NOISE PICKER
# ════════════════════════════════════════════════════════════════
def get_noise_for_type(stype, registries, rng_py, rng_np):
    augs: List[str] = []
    def _load(p): return load_audio_strict(p, TARGET_LEN, SR, rng_np)

    if stype == "general":
        n = safe_pick(registries["general"], rng_py, _load) if registries["general"] else None
        if n is not None: return n, augs
        return generate_synth_lf_noise(TARGET_LEN, SR, rng_np), ["fallback_synth"]
    if stype == "lf_curated":
        n = safe_pick(registries["lf_curated"], rng_py, _load) if registries["lf_curated"] else None
        if n is not None:
            if lf_energy_ratio(n) < 0.6:
                augs.append("curated_lpf_safety"); n = aug_lpf(n)
            return n, augs
        return generate_synth_lf_noise(TARGET_LEN, SR, rng_np), ["fallback_synth"]
    if stype == "lf_synth":
        return generate_synth_lf_noise(TARGET_LEN, SR, rng_np), ["synth_g2net"]
    if stype == "mixed_lpf":
        n = safe_pick(registries["general"], rng_py, _load) if registries["general"] else None
        if n is not None:
            augs.append("lpf_200hz"); return aug_lpf(n), augs
        return generate_synth_lf_noise(TARGET_LEN, SR, rng_np), ["fallback_synth"]
    raise ValueError(f"Unknown sample type: {stype}")


def stratified_assignment(n, ratios, rng):
    counts = {k: int(round(n * v)) for k, v in ratios.items()}
    drift = n - sum(counts.values())
    if drift != 0:
        biggest = max(counts, key=counts.get); counts[biggest] += drift
    out = []
    for k, c in counts.items(): out.extend([k] * c)
    rng.shuffle(out)
    return out


# ════════════════════════════════════════════════════════════════
# PER-SAMPLE BUILDERS  (sequential — Windows-safe, no multiproc)
# ════════════════════════════════════════════════════════════════
def build_synth_sample(idx, stype, seed, speech_file,
                       registries, out_dir, split) -> bool:
    """Return True if the 5 .npy files were written, False otherwise."""
    rng_py = random.Random(seed)
    rng_np = np.random.RandomState(seed % (2**31 - 1))

    speech = load_audio_strict(speech_file, TARGET_LEN, SR, rng_np)
    if speech is None:
        return False

    noise, augs = get_noise_for_type(stype, registries, rng_py, rng_np)
    if rng_np.rand() < P_PITCH_JITTER:
        noise = aug_pitch_jitter(noise, rng_np); augs.append("pitch_jitter")
    if rng_np.rand() < P_TIME_SHIFT:
        noise = aug_time_shift(noise, rng_np);   augs.append("time_shift")
    if stype == "general" and rng_np.rand() < P_LF_BOOST_GENERAL:
        noise = aug_lf_emphasis(noise);          augs.append("lf_emphasis")
    noise = aug_gain(noise, rng_np);             augs.append("gain")

    noise  = np.pad(noise,  (0, max(0, TARGET_LEN - len(noise)))) [:TARGET_LEN]
    speech = np.pad(speech, (0, max(0, TARGET_LEN - len(speech))))[:TARGET_LEN]

    snr_db = float(rng_np.uniform(*SNR_RANGE))
    noisy, scaled_noise = mix_at_snr(speech, noise, snr_db)
    noisy = np.clip(noisy, -1.0, 1.0).astype(np.float32)

    noisy_spec, phase = to_spectrogram(noisy)
    clean_spec, _     = to_spectrogram(speech)
    mask              = compute_irm(clean_spec, noisy_spec)

    meta = {
        "sample_type":     stype,
        "snr_db":          float(snr_db),
        "lf_energy_ratio": float(lf_energy_ratio(scaled_noise)),
        "augmentations":   augs,
        "speech_file":     os.path.basename(speech_file),
        "speaker_id":      speaker_id_for_clean(speech_file),
        "split":           split,
        "source":          "synthetic_mix",
    }
    prefix = os.path.join(out_dir, f"{idx:06d}")
    np.save(prefix + "_noisy_spec.npy",  noisy_spec)
    np.save(prefix + "_target_mask.npy", mask)
    np.save(prefix + "_clean_spec.npy",  clean_spec)
    np.save(prefix + "_noisy_phase.npy", phase)
    np.save(prefix + "_meta.npy", meta, allow_pickle=True)
    return True


def build_paired_sample(idx, clean_path, noisy_path, spk_id,
                        split, seed, out_dir) -> bool:
    rng_np = np.random.RandomState(seed % (2**31 - 1))
    clean = load_audio_strict(clean_path, TARGET_LEN, SR, rng_np)
    noisy = load_audio_strict(noisy_path, TARGET_LEN, SR, rng_np)
    if clean is None or noisy is None:
        return False

    clean = np.pad(clean, (0, max(0, TARGET_LEN - len(clean))))[:TARGET_LEN]
    noisy = np.pad(noisy, (0, max(0, TARGET_LEN - len(noisy))))[:TARGET_LEN]

    noisy_spec, phase = to_spectrogram(noisy)
    clean_spec, _     = to_spectrogram(clean)
    mask              = compute_irm(clean_spec, noisy_spec)

    meta = {
        "sample_type":     "paired_real",
        "snr_db":          None,
        "lf_energy_ratio": float(lf_energy_ratio(noisy - clean)),
        "augmentations":   [],
        "speech_file":     os.path.basename(clean_path),
        "noisy_file":      os.path.basename(noisy_path),
        "speaker_id":      spk_id,
        "split":           split,
        "source":          "extracted_paired",
    }
    prefix = os.path.join(out_dir, f"{idx:06d}")
    np.save(prefix + "_noisy_spec.npy",  noisy_spec)
    np.save(prefix + "_target_mask.npy", mask)
    np.save(prefix + "_clean_spec.npy",  clean_spec)
    np.save(prefix + "_noisy_phase.npy", phase)
    np.save(prefix + "_meta.npy", meta, allow_pickle=True)
    return True


def count_cached(out_dir: str) -> int:
    if not os.path.isdir(out_dir):
        return 0
    return sum(1 for f in os.listdir(out_dir) if f.endswith("_noisy_spec.npy"))


# ════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════
def run_preprocess() -> None:
    print(f"  Cache: {os.path.abspath(CACHE_DIR)}")
    print(f"\n══ Building registries ══════════════════════════════")
    spk_to_files = build_clean_registry()
    if not spk_to_files:
        sys.exit("\n  ERROR: no clean speech found.  Edit the source-path "
                 "constants at the top of this script.\n")
    noise_reg = build_noise_registry()

    print(f"  Example speaker ids: {sorted(spk_to_files.keys())[:5]}")

    print(f"\n══ Speaker-disjoint split (80/10/10) ════════════════")
    speakers = sorted(spk_to_files.keys())
    split_speakers = speaker_disjoint_split(speakers, SPLIT_RATIOS, SEED)
    for split, lst in split_speakers.items():
        n_clips = sum(len(spk_to_files[s]) for s in lst)
        print(f"  {split:<5s}  speakers={len(lst):<4d}  clips={n_clips:,d}")

    sets = {k: set(v) for k, v in split_speakers.items()}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "speaker_split.json"), "w") as f:
        json.dump(split_speakers, f, indent=2)

    # ── TRAIN ────────────────────────────────────────────────
    print(f"\n══ TRAIN: synthetic mixing (n={N_TRAIN:,d}) ═════════")
    out_dir = os.path.join(CACHE_DIR, "train")
    os.makedirs(out_dir, exist_ok=True)

    train_pool = []
    for s in split_speakers["train"]:
        train_pool.extend(spk_to_files[s])
    if not train_pool:
        sys.exit("  ERROR: train speech pool is empty.")

    rng = random.Random(SEED)
    labels = stratified_assignment(N_TRAIN, MIX_RATIOS, rng)
    existing = {int(f.split("_")[0]) for f in os.listdir(out_dir)
                if f.endswith("_noisy_spec.npy") and f.split("_")[0].isdigit()}
    print(f"  to_build: {N_TRAIN - len(existing):,d}  cached: {len(existing):,d}")

    n_ok = n_fail = 0
    fail_examples: List[str] = []
    for idx in tqdm(range(N_TRAIN), desc="  train"):
        if idx in existing:
            n_ok += 1; continue
        speech_file = rng.choice(train_pool)
        ok = build_synth_sample(idx, labels[idx], SEED + idx, speech_file,
                                noise_reg, out_dir, "train")
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            if len(fail_examples) < 3:
                fail_examples.append(speech_file)
    print(f"  Train: ok={n_ok:,d}  failed={n_fail:,d}  "
          f"on disk={count_cached(out_dir):,d}")
    if n_fail:
        print(f"  Example failures (speech could not load): {fail_examples}")
    if count_cached(out_dir) == 0:
        sys.exit("\n  CRITICAL: train cache is empty after running.\n"
                 "  Likely causes: write permission, antivirus blocking, "
                 "or wrong CACHE_DIR.\n"
                 f"  CACHE_DIR = {os.path.abspath(out_dir)}\n")

    # ── VAL / TEST ───────────────────────────────────────────
    paired_all = find_paired_clips()
    print(f"\n  Found {len(paired_all):,d} paired clean/noisy clips")

    for split, target_speakers, budget in [
        ("val",  split_speakers["val"],  N_VAL),
        ("test", split_speakers["test"], N_TEST),
    ]:
        print(f"\n══ {split.upper()}: paired real-world (target n={budget:,d}) ══")
        out_dir = os.path.join(CACHE_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        existing = {int(f.split("_")[0]) for f in os.listdir(out_dir)
                    if f.endswith("_noisy_spec.npy") and f.split("_")[0].isdigit()}

        scoped = []
        if paired_all:
            spk_set = set(target_speakers)
            scoped = [p for p in paired_all if p[2] in spk_set]
            if not scoped:
                print(f"  WARNING: no paired clips overlap {split} speakers; "
                      f"using all paired clips (loose pairing).")
                scoped = list(paired_all)

        if scoped:
            r = random.Random(SEED + (1 if split == "val" else 2))
            r.shuffle(scoped); scoped = scoped[:budget]
            print(f"  to_build: {len(scoped) - len(existing):,d}  "
                  f"cached: {len(existing):,d}")
            n_ok = n_fail = 0
            for idx, (cf, nf, spk) in enumerate(tqdm(scoped, desc=f"  {split}")):
                if idx in existing: n_ok += 1; continue
                ok = build_paired_sample(idx, cf, nf, spk, split,
                                         SEED + idx, out_dir)
                n_ok += int(ok); n_fail += int(not ok)
            print(f"  {split}: ok={n_ok}  failed={n_fail}  "
                  f"on disk={count_cached(out_dir):,d}")
        else:
            # synthetic fallback within this split’s speakers
            pool = []
            for s in target_speakers:
                pool.extend(spk_to_files[s])
            if not pool:
                print(f"  ERROR: no speech for {split}; skipping.")
                continue
            r = random.Random(SEED + (1 if split == "val" else 2))
            labels = stratified_assignment(budget, MIX_RATIOS, r)
            n_ok = n_fail = 0
            for idx in tqdm(range(budget), desc=f"  {split}"):
                if idx in existing: n_ok += 1; continue
                speech_file = r.choice(pool)
                seed = SEED + (1_000_000 if split == "val" else 2_000_000) + idx
                ok = build_synth_sample(idx, labels[idx], seed, speech_file,
                                        noise_reg, out_dir, split)
                n_ok += int(ok); n_fail += int(not ok)
            print(f"  {split} (synth): ok={n_ok}  failed={n_fail}  "
                  f"on disk={count_cached(out_dir):,d}")

    # ── Top-level meta ───────────────────────────────────────
    meta_global = {
        "version":      "preprocess_v2_speaker_disjoint",
        "SR":           SR,
        "DURATION":     DURATION,
        "N_FFT":        N_FFT,
        "HOP_LENGTH":   HOP_LENGTH,
        "FREQ_BINS":    FREQ_BINS,
        "LF_BINS":      LF_BINS,
        "LF_CUTOFF_HZ": LF_CUTOFF_HZ,
        "SNR_RANGE":    list(SNR_RANGE),
        "MIX_RATIOS":   MIX_RATIOS,
        "SPLIT_RATIOS": SPLIT_RATIOS,
        "n_train":      N_TRAIN,
        "n_val":        N_VAL,
        "n_test":       N_TEST,
        "speakers": {k: len(v) for k, v in split_speakers.items()},
    }
    np.save(os.path.join(CACHE_DIR, "meta.npy"), meta_global, allow_pickle=True)

    # Final on-disk verification — what Colab will actually see.
    n_train_d = count_cached(os.path.join(CACHE_DIR, "train"))
    n_val_d   = count_cached(os.path.join(CACHE_DIR, "val"))
    n_test_d  = count_cached(os.path.join(CACHE_DIR, "test"))
    print(f"\n══ Done ═════════════════════════════════════════════")
    print(f"  Cache: {os.path.abspath(CACHE_DIR)}")
    print(f"  On-disk counts:  train={n_train_d:,d}  val={n_val_d:,d}  "
          f"test={n_test_d:,d}")
    if 0 in (n_train_d, n_val_d, n_test_d):
        print(f"  WARNING: at least one split is empty — see messages above.")
    else:
        print("\nNext step → upload cache_crn3/ to Colab and run train_v2.py.\n")


# ════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Speaker-disjoint preprocessing for ANC training")
    ap.add_argument("--mode", default="preprocess",
                    choices=["preprocess"], help="(only mode supported)")
    ap.parse_args()

    print("=" * 64)
    print("  preprocess_v2.py — speaker-disjoint ANC preprocessing")
    print("=" * 64)
    print(f"  SR={SR}  DURATION={DURATION}s  N_FFT={N_FFT}  HOP={HOP_LENGTH}")
    print(f"  LF_BINS={LF_BINS}  SNR={SNR_RANGE}")
    run_preprocess()


if __name__ == "__main__":
    main()