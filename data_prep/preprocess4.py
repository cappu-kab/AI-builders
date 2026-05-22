"""
preprocess.py
=============

Convert raw speech (.wav) and noise (.wav) files into .npy tensors for
fast training, while STRICTLY preserving the (filename -> label) mapping
defined in a CSV metadata file.

For clean / speech audio (the ones with labels):
    1. Read the metadata CSV (e.g. `file_name,transcription,...`).
    2. Iterate the rows IN ORDER (no shuffling, no reordering).
    3. For every row: load .wav -> resample 16 kHz -> peak-normalize ->
       pad/trim to fixed length -> save as `<stem>.npy`.
    4. Write a parallel CSV `labels_npy.csv` with the SAME rows in the
       SAME ORDER, only the filename column rewritten from `*.wav` to
       `*.npy`.  All other columns are copied verbatim, so labels cannot
       drift.

For noise audio (no labels needed):
    - Same audio pipeline, written to a separate `noise/` output folder.

NO augmentation is performed here (no speech+noise mixing, no SNR, no
masking).  Augmentation belongs in the training DataLoader.

Usage
-----
Whole dataset (auto-iterates train / val / test under one root):
    python preprocess.py --root <dataset_root> --out-root <out_root>

Single split:
    python preprocess.py \
        --speech-dir <dir> --noise-dir <dir> \
        --csv-path   <csv> --out-dir   <dir>
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "librosa is required.  Install it with:\n"
        "    pip install librosa soundfile numpy"
    ) from exc


# ---------------------------------------------------------------------------
# Audio configuration (overridable from the CLI in main()).
# ---------------------------------------------------------------------------
TARGET_SR: int = 16_000
CLIP_SECONDS: float = 4.0
TARGET_LEN: int = int(TARGET_SR * CLIP_SECONDS)
DTYPE = np.float32

# Robust column-name detection (case-insensitive).  Order matters: the
# first candidate found in the CSV header wins.
FILENAME_COL_CANDIDATES = (
    "file_name", "filename", "file", "wav", "wav_filename",
    "path", "audio", "audio_path", "audio_file",
)
LABEL_COL_CANDIDATES = (
    "transcription_cleaned", "transcription",
    "transcript", "label", "text", "sentence", "normalized_text",
)

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def load_wav_mono(path: Path) -> np.ndarray:
    """Load a wav file as float32 mono at TARGET_SR."""
    audio, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return audio.astype(DTYPE, copy=False)


def peak_normalize(audio: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Peak-normalize the waveform to ~[-1, 1].  Silence is left as-is."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < eps:
        return audio.astype(DTYPE, copy=False)
    return (audio / peak).astype(DTYPE, copy=False)


def fix_length(audio: np.ndarray) -> np.ndarray:
    """Pad with zeros (short) or trim (long) so length == TARGET_LEN."""
    n = int(audio.shape[0])
    if n == TARGET_LEN:
        return audio.astype(DTYPE, copy=False)
    if n > TARGET_LEN:
        return audio[:TARGET_LEN].astype(DTYPE, copy=False)
    pad = np.zeros(TARGET_LEN - n, dtype=DTYPE)
    return np.concatenate([audio, pad], axis=0)


def process_audio_file(path: Path) -> np.ndarray:
    """Full per-file pipeline: load -> normalize -> pad/trim."""
    audio = load_wav_mono(path)
    audio = peak_normalize(audio)
    audio = fix_length(audio)
    assert audio.shape == (TARGET_LEN,), (
        f"Bad output shape for {path}: {audio.shape} (expected ({TARGET_LEN},))"
    )
    return audio


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def detect_columns(fieldnames: Iterable[str]) -> Tuple[str, Optional[str]]:
    """Find (filename_col, label_col) from a CSV header (case-insensitive)."""
    fnames = list(fieldnames)
    lower_to_orig = {f.lower().strip(): f for f in fnames}

    fname_col = next(
        (lower_to_orig[c] for c in FILENAME_COL_CANDIDATES if c in lower_to_orig),
        None,
    )
    if fname_col is None:
        raise ValueError(
            f"No filename column found in CSV.  Looked for one of "
            f"{FILENAME_COL_CANDIDATES}; got headers {fnames}."
        )
    label_col = next(
        (lower_to_orig[c] for c in LABEL_COL_CANDIDATES if c in lower_to_orig),
        None,
    )
    return fname_col, label_col


# ---------------------------------------------------------------------------
# Per-split routines
# ---------------------------------------------------------------------------
def process_speech_split(
    csv_path: Path,
    speech_dir: Path,
    out_speech_dir: Path,
    out_csv_path: Path,
) -> None:
    """Convert speech WAVs to .npy and write a parallel labels CSV.

    Mapping integrity:
      - Each output row is written immediately after its .npy file is
        successfully saved, in the same order as the input CSV.
      - Only the filename column changes (`*.wav` -> `*.npy`); all other
        columns (transcription, source, ...) are copied verbatim.
      - Rows whose audio is missing or fails to decode are SKIPPED and
        omitted from the output CSV, so the CSV always points only at
        .npy files that actually exist on disk.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    if not speech_dir.is_dir():
        raise FileNotFoundError(f"Speech directory not found: {speech_dir}")

    out_speech_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_ok = n_missing = n_failed = n_dup = 0
    seen_basenames: set = set()

    # utf-8-sig handles a possible BOM in the metadata CSV gracefully.
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"Empty / header-less CSV: {csv_path}")
        fname_col, label_col = detect_columns(reader.fieldnames)
        log.info("CSV %s -> filename col='%s', label col='%s'",
                 csv_path.name, fname_col, label_col)

        out_fieldnames = list(reader.fieldnames)

        with open(out_csv_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
            writer.writeheader()

            for row in reader:
                n_total += 1
                raw_name = (row.get(fname_col) or "").strip()
                if not raw_name:
                    log.warning("Row %d has empty filename - skipped.", n_total)
                    continue

                # Try basename first (handles 'speech/foo.wav' style paths),
                # then fall back to using the raw value as a relative path.
                wav_basename = Path(raw_name).name
                wav_path = speech_dir / wav_basename
                if not wav_path.is_file():
                    alt = speech_dir / raw_name
                    if alt.is_file():
                        wav_path = alt
                    else:
                        log.warning("Missing WAV for row %d: %s - skipped.",
                                    n_total, wav_path)
                        n_missing += 1
                        continue

                stem = Path(wav_basename).stem
                npy_basename = f"{stem}.npy"

                if npy_basename in seen_basenames:
                    log.warning(
                        "Duplicate output stem '%s' (row %d) - skipping the "
                        "later occurrence to keep mapping unambiguous.",
                        stem, n_total,
                    )
                    n_dup += 1
                    continue

                try:
                    audio = process_audio_file(wav_path)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed to process %s (%s) - skipped.",
                                wav_path, exc)
                    n_failed += 1
                    continue

                np.save(out_speech_dir / npy_basename, audio,
                        allow_pickle=False)
                seen_basenames.add(npy_basename)

                # Write the matching CSV row right after the .npy is on disk.
                out_row = dict(row)
                out_row[fname_col] = npy_basename
                writer.writerow(out_row)
                n_ok += 1

    log.info(
        "Speech split done - saved %d/%d (missing=%d, failed=%d, duplicates=%d). "
        "Output CSV: %s",
        n_ok, n_total, n_missing, n_failed, n_dup, out_csv_path,
    )


def process_noise_split(noise_dir: Path, out_noise_dir: Path) -> None:
    """Convert every noise WAV (recursively) to .npy.  No CSV produced."""
    if not noise_dir.is_dir():
        log.warning("Noise directory not found: %s - skipping.", noise_dir)
        return
    out_noise_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(noise_dir.rglob("*.wav"))
    if not wavs:
        log.warning("No noise WAVs found under %s.", noise_dir)
        return

    n_ok = n_failed = n_dup = 0
    seen: set = set()
    for wav_path in wavs:
        out_name = wav_path.stem + ".npy"
        if out_name in seen:
            log.warning("Duplicate noise stem '%s' - skipping %s.",
                        wav_path.stem, wav_path)
            n_dup += 1
            continue
        try:
            audio = process_audio_file(wav_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to process noise %s (%s) - skipped.",
                        wav_path, exc)
            n_failed += 1
            continue
        np.save(out_noise_dir / out_name, audio, allow_pickle=False)
        seen.add(out_name)
        n_ok += 1

    log.info("Noise done - saved %d/%d (failed=%d, duplicates=%d). Output: %s",
             n_ok, len(wavs), n_failed, n_dup, out_noise_dir)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_single(
    speech_dir: Path,
    noise_dir: Optional[Path],
    csv_path: Path,
    out_dir: Path,
    out_csv_name: str = "labels_npy.csv",
) -> None:
    out_dir = Path(out_dir)
    process_speech_split(
        csv_path=csv_path,
        speech_dir=speech_dir,
        out_speech_dir=out_dir / "clean",
        out_csv_path=out_dir / out_csv_name,
    )
    if noise_dir is not None:
        process_noise_split(noise_dir, out_dir / "noise")


def run_multi_split(
    root: Path,
    out_root: Path,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> None:
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            log.warning("Split '%s' not found at %s - skipping.", split, split_dir)
            continue

        speech_dir = next(
            (split_dir / name for name in ("speech", "clean")
             if (split_dir / name).is_dir()),
            None,
        )
        if speech_dir is None:
            log.warning("No 'speech' or 'clean' dir under %s - skipping split.",
                        split_dir)
            continue

        noise_dir = split_dir / "noise"
        if not noise_dir.is_dir():
            noise_dir = None

        csv_path = next(
            (p for p in (split_dir / "metadata.csv",
                         split_dir / "labels.csv",
                         split_dir / f"{split}.csv")
             if p.is_file()),
            None,
        )
        if csv_path is None:
            log.warning("No metadata CSV for split %s - skipping.", split)
            continue

        log.info("=== Processing split: %s ===", split)
        run_single(
            speech_dir=speech_dir,
            noise_dir=noise_dir,
            csv_path=csv_path,
            out_dir=out_root / split,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess speech / noise WAVs into NPY with strict label alignment.",
    )
    p.add_argument("--speech-dir", type=Path, help="Directory of clean speech WAVs.")
    p.add_argument("--noise-dir", type=Path, help="(Optional) directory of noise WAVs.")
    p.add_argument("--csv-path", type=Path, help="Metadata CSV.")
    p.add_argument("--out-dir", type=Path, help="Output directory for this split.")
    p.add_argument("--root", type=Path, help="Dataset root containing train/val/test.")
    p.add_argument("--out-root", type=Path, help="Output root for all splits.")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   help="Subdirs of --root to process (default: train val test).")
    p.add_argument("--target-sr", type=int, default=TARGET_SR,
                   help=f"Target sample rate in Hz (default {TARGET_SR}).")
    p.add_argument("--clip-seconds", type=float, default=CLIP_SECONDS,
                   help=f"Fixed clip length in seconds (default {CLIP_SECONDS}).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    global TARGET_SR, CLIP_SECONDS, TARGET_LEN
    TARGET_SR = int(args.target_sr)
    CLIP_SECONDS = float(args.clip_seconds)
    TARGET_LEN = int(round(TARGET_SR * CLIP_SECONDS))
    log.info("Audio config: target_sr=%d Hz, clip_seconds=%.3f s, target_len=%d samples",
             TARGET_SR, CLIP_SECONDS, TARGET_LEN)

    if args.root is not None and args.out_root is not None:
        run_multi_split(args.root, args.out_root, splits=tuple(args.splits))
        return 0

    if args.speech_dir and args.csv_path and args.out_dir:
        run_single(
            speech_dir=args.speech_dir,
            noise_dir=args.noise_dir,
            csv_path=args.csv_path,
            out_dir=args.out_dir,
        )
        return 0

    print(
        "ERROR: please pass either\n"
        "  --root <dataset_root> --out-root <out_root>\n"
        "or\n"
        "  --speech-dir <dir> --csv-path <csv> --out-dir <dir> [--noise-dir <dir>]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())