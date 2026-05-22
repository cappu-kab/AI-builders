"""
filter_low_freq_noise.py
------------------------
Iterate through every audio file in folders ``fold1`` ... ``fold10`` of an
UrbanSound8K-style dataset, analyse the frequency content of each file, and
copy any file whose dominant frequency (or whose significant spectral energy)
sits below 200 Hz into a destination folder called ``low_freq_noise``.

Dependencies
------------
- librosa      (audio loading + STFT / spectral utilities)
- numpy
- shutil       (file copy)
- pathlib      (path handling)

Usage
-----
    python filter_low_freq_noise.py \\
        --dataset-root /path/to/UrbanSound8K/audio \\
        --dest-folder  /path/to/low_freq_noise \\
        --threshold-hz 200
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "ERROR: librosa is required. Install it with `pip install librosa`.\n"
    )
    raise


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AUDIO_EXTENSIONS: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a")
DEFAULT_THRESHOLD_HZ: float = 200.0
# Fraction of total spectral energy that must lie below the threshold for a
# file to qualify as "significant low-frequency energy".
LOW_BAND_ENERGY_RATIO: float = 0.5
# STFT window size — 2048 samples gives ~21.5 Hz resolution at 44.1 kHz, which
# is more than enough to pick out the sub-200 Hz region.
N_FFT: int = 2048
HOP_LENGTH: int = 512


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("filter_low_freq_noise")


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def analyse_frequency(
    file_path: Path,
    threshold_hz: float = DEFAULT_THRESHOLD_HZ,
    low_band_energy_ratio: float = LOW_BAND_ENERGY_RATIO,
) -> Optional[Tuple[float, float, bool]]:
    """
    Compute the dominant frequency of an audio file and the fraction of its
    spectral energy that lies below ``threshold_hz``.

    Returns
    -------
    (dominant_freq_hz, low_band_ratio, qualifies)
        ``qualifies`` is True when EITHER the dominant frequency is below the
        threshold OR the fraction of energy below the threshold is at least
        ``low_band_energy_ratio``.
    Returns ``None`` if the file could not be processed.
    """
    try:
        # librosa.load handles arbitrary sample rates and lengths; mono=True
        # collapses multi-channel audio to a single channel for analysis.
        y, sr = librosa.load(str(file_path), sr=None, mono=True)
    except Exception as exc:
        logger.warning("Could not load %s: %s", file_path, exc)
        return None

    if y is None or y.size == 0:
        logger.warning("Empty audio data in %s — skipping.", file_path)
        return None

    # Very short clips (< one FFT window) are zero-padded so STFT still works.
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size), mode="constant")

    try:
        # Magnitude spectrogram → average across time → power spectrum per bin.
        stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        spectrum = np.mean(stft ** 2, axis=1)  # power, averaged over frames
        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    except Exception as exc:
        logger.warning("STFT failed for %s: %s", file_path, exc)
        return None

    total_energy = float(np.sum(spectrum))
    if total_energy <= 0.0 or not np.isfinite(total_energy):
        logger.warning("No usable spectral energy in %s — skipping.", file_path)
        return None

    # Dominant frequency = bin with maximum mean power.
    dominant_freq = float(freqs[int(np.argmax(spectrum))])

    # Fraction of total energy below the threshold.
    low_band_mask = freqs < threshold_hz
    low_band_energy = float(np.sum(spectrum[low_band_mask]))
    low_band_ratio = low_band_energy / total_energy

    qualifies = (dominant_freq < threshold_hz) or (low_band_ratio >= low_band_energy_ratio)
    return dominant_freq, low_band_ratio, qualifies


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------
def iter_audio_files(dataset_root: Path) -> Iterable[Path]:
    """Yield every audio file inside ``fold1`` ... ``fold10`` under ``dataset_root``."""
    for fold_idx in range(1, 11):
        fold_dir = dataset_root / f"fold{fold_idx}"
        if not fold_dir.is_dir():
            logger.warning("Expected folder is missing: %s", fold_dir)
            continue
        for path in sorted(fold_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                yield path


def copy_with_unique_name(src: Path, dest_dir: Path) -> Path:
    """
    Copy ``src`` into ``dest_dir``. If a file with the same name already exists
    (different folds can share filenames), the parent fold name is prepended to
    keep things unique.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src.name
    if target.exists():
        target = dest_dir / f"{src.parent.name}_{src.name}"
        # As a final safety net, append an index if even that collides.
        idx = 1
        while target.exists():
            target = dest_dir / f"{src.parent.name}_{idx}_{src.name}"
            idx += 1
    shutil.copy2(src, target)
    return target


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def filter_low_frequency_files(
    dataset_root: Path,
    dest_folder: Path,
    threshold_hz: float = DEFAULT_THRESHOLD_HZ,
    low_band_energy_ratio: float = LOW_BAND_ENERGY_RATIO,
) -> None:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    dest_folder.mkdir(parents=True, exist_ok=True)

    total = matched = errors = 0
    for audio_path in iter_audio_files(dataset_root):
        total += 1
        try:
            result = analyse_frequency(
                audio_path,
                threshold_hz=threshold_hz,
                low_band_energy_ratio=low_band_energy_ratio,
            )
        except Exception as exc:  # last-resort guard — analyse_frequency already handles most.
            logger.error("Unexpected error on %s: %s", audio_path, exc)
            errors += 1
            continue

        if result is None:
            errors += 1
            continue

        dominant_freq, low_ratio, qualifies = result
        if qualifies:
            try:
                target = copy_with_unique_name(audio_path, dest_folder)
                matched += 1
                logger.info(
                    "MATCH  %s  (dominant=%.1f Hz, low-band energy=%.0f%%) -> %s",
                    audio_path.name, dominant_freq, low_ratio * 100, target.name,
                )
            except Exception as exc:
                logger.error("Failed to copy %s -> %s: %s", audio_path, dest_folder, exc)
                errors += 1
        else:
            logger.debug(
                "skip   %s  (dominant=%.1f Hz, low-band energy=%.0f%%)",
                audio_path.name, dominant_freq, low_ratio * 100,
            )

    logger.info(
        "Done. Scanned %d file(s); copied %d low-frequency file(s); %d error(s).",
        total, matched, errors,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan UrbanSound8K folds and copy files whose dominant frequency "
            "(or significant energy) is below a threshold into a destination folder."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the directory containing fold1 ... fold10.",
    )
    parser.add_argument(
        "--dest-folder",
        type=Path,
        default=Path("low_freq_noise"),
        help="Destination folder for copied files (default: ./low_freq_noise).",
    )
    parser.add_argument(
        "--threshold-hz",
        type=float,
        default=DEFAULT_THRESHOLD_HZ,
        help=f"Frequency cutoff in Hz (default: {DEFAULT_THRESHOLD_HZ}).",
    )
    parser.add_argument(
        "--low-band-energy-ratio",
        type=float,
        default=LOW_BAND_ENERGY_RATIO,
        help=(
            "Minimum fraction of total energy that must lie below the threshold "
            f"to count as 'significant' (default: {LOW_BAND_ENERGY_RATIO})."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging (prints skipped files too).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    try:
        filter_low_frequency_files(
            dataset_root=args.dataset_root,
            dest_folder=args.dest_folder,
            threshold_hz=args.threshold_hz,
            low_band_energy_ratio=args.low_band_energy_ratio,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())