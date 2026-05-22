r"""
clean_audio_dataset.py
======================

Cleans an audio dataset by keeping ONLY files whose dominant acoustic energy
sits below 200 Hz (e.g. infrasound, low rumbles, bass-heavy field recordings)
and removing any file that YAMNet classifies as human speech / vocal noise.

Pipeline
--------
1. Run YAMNet (TensorFlow Hub) on every audio file.
2. If the top-K predictions contain any "vocal / speech" class, SKIP the file
   (it stays in the source folder, untouched).
3. Otherwise, run a spectral check with librosa: confirm that the share of
   power below 200 Hz exceeds a configurable threshold (default 60 %).
4. Files that pass both checks are COPIED into ./cleaned_data/.

The source folder is treated as READ-ONLY — nothing is ever deleted or moved
from it. The cleaned dataset is built up as a copy in `cleaned_data/`.

------------------------------------------------------------------------
Environment setup
------------------------------------------------------------------------
Tested with Python 3.10 - 3.11.

    python -m venv .venv
    source .venv/bin/activate            # Windows: .venv\Scripts\activate
    pip install --upgrade pip
    pip install tensorflow==2.15.*       # CPU build is fine
    pip install tensorflow-hub
    pip install librosa soundfile numpy scipy resampy tqdm

YAMNet expects mono audio at 16 kHz. We resample on the fly with librosa,
so the source files can be any sample rate / channel count that librosa
can read (wav, flac, ogg, mp3 with the right backend, etc.).

------------------------------------------------------------------------
YAMNet class mapping
------------------------------------------------------------------------
YAMNet outputs probabilities over 521 AudioSet ontology classes. The model
ships a CSV mapping that lives next to the saved model on TF-Hub. We load
it via `model.class_map_path()` -> a CSV with columns:
    index, mid, display_name

We treat anything in VOCAL_CLASSES (case-insensitive substring match against
`display_name`) as "speech / vocal noise" and reject those files.
Edit VOCAL_CLASSES below to tune what counts as a vocal class for your data.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# YAMNet's required input sample rate.
YAMNET_SR = 16_000

# Substrings of YAMNet display_names we treat as "vocal / speech".
# Matching is case-insensitive; any prediction whose display_name CONTAINS
# one of these tokens is considered vocal.
VOCAL_CLASSES = {
    "speech",
    "conversation",
    "narration",
    "monologue",
    "whispering",
    "shout",
    "yell",
    "screaming",
    "child speech",
    "kid speaking",
    "male speech",
    "female speech",
    "singing",
    "chant",
    "rapping",
    "humming",
    "laughter",
    "crying",
    "babbling",
    "whistling",
}

# Audio extensions we will try to process.
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif"}

# Number of top YAMNet predictions to inspect per file.
TOP_K = 5

# Minimum confidence (mean across frames) required for a vocal class to
# trigger deletion. Prevents deleting on a single noisy frame.
VOCAL_PROB_THRESHOLD = 0.20

# Fraction of total power that must lie below LOW_FREQ_HZ for the file to
# be considered "low frequency dominant".
LOW_FREQ_HZ = 200.0
LOW_FREQ_POWER_RATIO = 0.60

# ----------------------------------------------------------------------
# YAMNet helpers
# ----------------------------------------------------------------------

def load_yamnet():
    """Download (cached) and return the YAMNet model + class names list."""
    print("Loading YAMNet from TF-Hub ...", file=sys.stderr)
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    # The model exposes a CSV path with the AudioSet class map.
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names: list[str] = []
    with tf.io.gfile.GFile(class_map_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row["display_name"])
    return model, class_names


def is_vocal_class(name: str) -> bool:
    """True if the YAMNet display_name matches any token in VOCAL_CLASSES."""
    lname = name.lower()
    return any(token in lname for token in VOCAL_CLASSES)


def classify_with_yamnet(model, class_names, waveform_16k: np.ndarray):
    """
    Run YAMNet on a mono 16 kHz waveform.

    Returns:
        ranked: list of (display_name, mean_probability) sorted desc, length TOP_K
        vocal_score: max mean_probability across all vocal-classed entries
    """
    # YAMNet expects float32 in [-1, 1].
    waveform_16k = waveform_16k.astype(np.float32, copy=False)
    scores, _embeddings, _spectrogram = model(waveform_16k)
    scores_np = scores.numpy()                     # shape: (frames, 521)
    mean_scores = scores_np.mean(axis=0)           # shape: (521,)

    top_idx = np.argsort(mean_scores)[::-1][:TOP_K]
    ranked = [(class_names[i], float(mean_scores[i])) for i in top_idx]

    vocal_score = 0.0
    for i, p in enumerate(mean_scores):
        if is_vocal_class(class_names[i]):
            vocal_score = max(vocal_score, float(p))
    return ranked, vocal_score


# ----------------------------------------------------------------------
# Spectral check
# ----------------------------------------------------------------------

def low_frequency_power_ratio(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = LOW_FREQ_HZ,
    n_fft: int = 4096,
) -> float:
    """
    Compute the fraction of total spectral power that lies at or below
    `cutoff_hz`. Uses an STFT power spectrogram averaged over time.

    A larger n_fft gives finer frequency resolution, which matters when
    cutoff_hz is small (200 Hz). At sr=22050 and n_fft=4096 the bin width
    is ~5.4 Hz, plenty to resolve the 0-200 Hz band cleanly.
    """
    # Magnitude STFT -> power spectrogram, then average across frames.
    stft = librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 4, window="hann")
    power = np.mean(np.abs(stft) ** 2, axis=1)             # shape: (n_fft/2+1,)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)    # shape: (n_fft/2+1,)

    total = float(power.sum())
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    low_mask = freqs <= cutoff_hz
    low = float(power[low_mask].sum())
    return low / total


# ----------------------------------------------------------------------
# Per-file processing
# ----------------------------------------------------------------------

def iter_audio_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def process_file(
    path: Path,
    model,
    class_names,
    cleaned_dir: Path,
    dry_run: bool,
) -> str:
    """
    Returns one of: 'kept', 'skipped_vocal', 'skipped_highfreq',
                     'load_error'.

    The source file is NEVER deleted or moved. Files that pass both checks
    are COPIED into `cleaned_dir`; everything else is left where it is.
    """
    try:
        # Load at native sr for the spectral check; we'll resample a copy
        # to 16 kHz for YAMNet.
        y_native, sr_native = librosa.load(path, sr=None, mono=True)
        if y_native.size == 0:
            return "load_error"
        y_16k = librosa.resample(y_native, orig_sr=sr_native, target_sr=YAMNET_SR)
    except Exception as exc:
        print(f"  ! could not load {path.name}: {exc}", file=sys.stderr)
        return "load_error"

    # Step 1: YAMNet classification ------------------------------------
    ranked, vocal_score = classify_with_yamnet(model, class_names, y_16k)
    top_str = ", ".join(f"{n} ({p:.2f})" for n, p in ranked[:3])

    if vocal_score >= VOCAL_PROB_THRESHOLD:
        print(f"  - {path.name}: VOCAL ({vocal_score:.2f}) [top: {top_str}]")
        return "skipped_vocal"

    # Step 2: spectral check at native sample rate ---------------------
    ratio = low_frequency_power_ratio(y_native, sr_native, cutoff_hz=LOW_FREQ_HZ)
    if ratio < LOW_FREQ_POWER_RATIO:
        print(
            f"  - {path.name}: high-freq dominant "
            f"(low-band ratio={ratio:.2f}) [top: {top_str}]"
        )
        return "skipped_highfreq"

    # Step 3: keep it (copy into cleaned_dir, source untouched) --------
    dest = cleaned_dir / path.name
    # Avoid overwriting on filename collisions.
    n = 1
    while dest.exists():
        dest = cleaned_dir / f"{path.stem}__{n}{path.suffix}"
        n += 1
    print(
        f"  + {path.name}: KEEP "
        f"(low-band ratio={ratio:.2f}) [top: {top_str}]"
    )
    if not dry_run:
        shutil.copy2(str(path), str(dest))
    return "kept"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> int:
    # `global` must appear before any reference to these names in this
    # function. argparse reads them as defaults below, so declare first.
    global LOW_FREQ_HZ, LOW_FREQ_POWER_RATIO, VOCAL_PROB_THRESHOLD

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "source",
        type=Path,
        help="Folder containing the raw audio dataset (searched recursively).",
    )
    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=None,
        help="Destination folder for verified low-frequency files. "
             "Defaults to ./cleaned_data next to the source folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without deleting or moving anything.",
    )
    parser.add_argument(
        "--low-cutoff-hz",
        type=float,
        default=LOW_FREQ_HZ,
        help=f"Upper bound of the 'low frequency' band (default {LOW_FREQ_HZ}).",
    )
    parser.add_argument(
        "--low-power-ratio",
        type=float,
        default=LOW_FREQ_POWER_RATIO,
        help=f"Required share of power below the cutoff (default {LOW_FREQ_POWER_RATIO}).",
    )
    parser.add_argument(
        "--vocal-threshold",
        type=float,
        default=VOCAL_PROB_THRESHOLD,
        help=f"YAMNet mean probability above which a vocal class triggers "
             f"deletion (default {VOCAL_PROB_THRESHOLD}).",
    )
    args = parser.parse_args()

    # Apply CLI overrides to module-level knobs the helpers reference.
    LOW_FREQ_HZ = args.low_cutoff_hz
    LOW_FREQ_POWER_RATIO = args.low_power_ratio
    VOCAL_PROB_THRESHOLD = args.vocal_threshold

    if not args.source.is_dir():
        print(f"Source folder not found: {args.source}", file=sys.stderr)
        return 2

    cleaned_dir = args.cleaned_dir or (args.source.parent / "cleaned_data")
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    print(f"Source : {args.source.resolve()}")
    print(f"Output : {cleaned_dir.resolve()}")
    print(f"Dry run: {args.dry_run}")

    model, class_names = load_yamnet()

    counts = {"kept": 0, "skipped_vocal": 0, "skipped_highfreq": 0, "load_error": 0}
    files = list(iter_audio_files(args.source))
    if not files:
        print("No audio files found.")
        return 0

    for path in tqdm(files, desc="Processing", unit="file"):
        result = process_file(path, model, class_names, cleaned_dir, args.dry_run)
        counts[result] += 1

    # Summary -----------------------------------------------------------
    print("\n=== Summary ===")
    print(f"  Kept (copied to cleaned_data): {counts['kept']}")
    print(f"  Skipped as vocal/speech      : {counts['skipped_vocal']}")
    print(f"  Skipped (high-freq dominant) : {counts['skipped_highfreq']}")
    print(f"  Could not load               : {counts['load_error']}")
    print(f"  Total                        : {sum(counts.values())}")
    print("  (source folder was not modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())