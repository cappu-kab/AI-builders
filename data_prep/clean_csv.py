"""
clean_csv.py

Cleans the `transcription` column of speech-dataset metadata files
(metadata.csv) located in dataset root directories.

Subfolder layouts vary by dataset — some have `train/`, `test/`, and
`validation/`, while others (e.g. thai_elderly_local, thai_isan_dialect_local)
only have a `train/` split. The script auto-discovers every immediate
subfolder of each root and processes any `metadata.csv` it finds, so it
adapts to whatever layout is on disk without printing spurious "missing
folder" warnings.

The cleaned transcription keeps only:
    * Thai characters         (U+0E00 - U+0E7F)
    * Arabic numerals         (0-9)
    * Thai numerals           (U+0E50 - U+0E59, already covered by the Thai range)
    * Single space characters

Everything else (English letters, punctuation, symbols, emoji, etc.)
is stripped. Multiple consecutive spaces are collapsed into one and
leading/trailing whitespace is removed.

The result is written next to the original file as `metadata_cleaned.csv`
using `utf-8-sig` encoding so Thai text renders correctly in Excel.

Row-level integrity is preserved: NO rows are dropped, even if the
cleaned transcription becomes empty.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ROOTS: List[str] = [
    "common_voice_local",
    "fleurs_local",
    "thai_elderly_local",
    "thai_isan_dialect_local",
]

# Preferred split names. If any of these exist as subfolders of a dataset
# root they are processed in this order; any *other* subfolders containing
# a metadata.csv are picked up afterwards. Datasets that only have a
# `train/` split (like thai_elderly_local) just work — no [miss] noise for
# the splits they don't have.
PREFERRED_SUBFOLDERS: List[str] = ["train", "test", "validation"]

# Anything that is NOT a Thai character (U+0E00-U+0E7F), an ASCII digit,
# or a whitespace character is considered "noise" and gets removed.
# Note: the Thai Unicode block already includes Thai digits (U+0E50-U+0E59).
NOISE_PATTERN = r"[^฀-๿0-9\s]"

# Detection patterns used purely for reporting (not for cleaning).
ENGLISH_PATTERN = r"[A-Za-z]"
SPECIAL_PATTERN = r"[^฀-๿0-9A-Za-z\s]"

# Multiple-whitespace collapser.
MULTI_SPACE_PATTERN = r"\s+"


# ---------------------------------------------------------------------------
# Core cleaning routine
# ---------------------------------------------------------------------------

def _clean_series(series: pd.Series) -> pd.Series:
    """Apply the vectorized cleaning pipeline to a Pandas Series."""
    s = series.fillna("").astype(str)
    s = s.str.replace(NOISE_PATTERN, "", regex=True)
    s = s.str.replace(MULTI_SPACE_PATTERN, " ", regex=True)
    s = s.str.strip()
    return s


def _process_metadata_file(metadata_path: Path) -> None:
    """Clean a single metadata.csv and write metadata_cleaned.csv next to it."""
    df = pd.read_csv(metadata_path, encoding="utf-8")

    if "transcription" not in df.columns or "file_name" not in df.columns:
        print(
            f"  [skip] {metadata_path} is missing required columns "
            f"(found: {list(df.columns)})"
        )
        return

    original = df["transcription"].fillna("").astype(str)

    has_english = original.str.contains(ENGLISH_PATTERN, regex=True, na=False)
    has_special = original.str.contains(SPECIAL_PATTERN, regex=True, na=False)
    flagged_mask = has_english | has_special

    cleaned = _clean_series(original)

    out = pd.DataFrame(
        {
            "file_name": df["file_name"],
            "transcription": original,
            "transcription_cleaned": cleaned,
        }
    )

    output_path = metadata_path.with_name("metadata_cleaned.csv")
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    total_rows = len(df)
    flagged_rows = int(flagged_mask.sum())
    print(
        f"  [ok]   {metadata_path.parent.name:<12} "
        f"rows={total_rows:>6} | flagged(en/special)={flagged_rows:>6} "
        f"| -> {output_path}"
    )


# ---------------------------------------------------------------------------
# Auto-discovery of split folders
# ---------------------------------------------------------------------------

def _discover_split_dirs(
    root_path: Path, preferred: Iterable[str]
) -> List[Path]:
    """Return immediate subdirectories of ``root_path`` that contain a
    ``metadata.csv``.

    Preferred names (train / test / validation) are listed first in the
    given order; any additional metadata-bearing subfolders follow in
    alphabetical order. Subfolders without a metadata.csv are silently
    skipped, so datasets that only ship a `train/` split don't generate
    misleading "missing" messages.
    """
    if not root_path.is_dir():
        return []

    existing = {p.name: p for p in root_path.iterdir() if p.is_dir()}
    ordered: List[Path] = []
    seen = set()

    for name in preferred:
        candidate = existing.get(name)
        if candidate is not None and (candidate / "metadata.csv").is_file():
            ordered.append(candidate)
            seen.add(name)

    for name in sorted(existing):
        if name in seen:
            continue
        if (existing[name] / "metadata.csv").is_file():
            ordered.append(existing[name])

    return ordered


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def clean_dataset_inplace(
    dataset_roots: Iterable[str] = DATASET_ROOTS,
    preferred_subfolders: Iterable[str] = PREFERRED_SUBFOLDERS,
    base_dir: Optional[Union[str, "os.PathLike[str]"]] = None,
) -> None:
    """Iterate through dataset roots, cleaning every ``metadata.csv`` found.

    Parameters
    ----------
    dataset_roots:
        Names (or paths) of dataset root directories to scan.
    preferred_subfolders:
        Split-folder names processed first when present (typically
        train / test / validation). Any other subfolders that contain a
        ``metadata.csv`` are processed afterwards. Splits that simply
        don't exist for a given dataset are silently skipped.
    base_dir:
        Optional parent directory. If provided, each entry of
        ``dataset_roots`` is resolved relative to this directory.
    """
    base = Path(base_dir) if base_dir else Path.cwd()

    for root_name in dataset_roots:
        root_path = (base / root_name).resolve()
        print(f"\n=== Dataset: {root_name} ({root_path}) ===")

        if not root_path.is_dir():
            print(f"  [warn] root directory not found, skipping: {root_path}")
            continue

        split_dirs = _discover_split_dirs(root_path, preferred_subfolders)

        if not split_dirs:
            print("  [warn] no subfolders with metadata.csv were found")
            continue

        for sub_path in split_dirs:
            metadata_path = sub_path / "metadata.csv"
            try:
                _process_metadata_file(metadata_path)
            except Exception as exc:  # noqa: BLE001 - broad catch in a CLI tool
                print(f"  [err]  failed on {metadata_path}: {exc!r}")


# ---------------------------------------------------------------------------
# CLI hook
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    clean_dataset_inplace()