"""
integrate_speech_to_final_dataset.py

Integrate speech data from 4 sources into an existing
`final_dataset/{train,val,test}/` structure that already contains
`noise/` subfolders.

Sources
-------
1. CommonVoice  (pre-split: train / validation / test)
2. FLEURS       (pre-split: train / validation / test)
3. Thai Elderly (unsplit: single folder + single metadata.csv)
4. Thai Isan    (unsplit: single folder + single metadata.csv)

RAM-friendly design
-------------------
The script never accumulates rows from all sources in memory.
For every (source x target_split) chunk it:
    1. Copies audio files one-by-one (with a progress counter).
    2. Rewrites the file_name column to `speech/<prefix>_<basename>`.
    3. Writes the chunk *immediately* to
       `final_dataset/<split>/metadata.csv` in append mode
       (`mode='w'` for the first writer of that split, `mode='a'` after).
       UTF-8-sig encoding preserves Thai characters.

Existing `noise/` folders are NOT touched.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIGURATION  --  edit the paths below to match your layout
# ============================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Target dataset root (already contains train/val/test/<noise>/)
FINAL_DATASET_DIR = Path(r"C:\Users\rocha\AI_builders\final_dataset")

# Final dataset split folder names
TARGET_SPLITS = ["train", "val", "test"]

# Split ratios for unsplit sources
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# CSV encoding (utf-8-sig keeps Thai characters intact in Excel/Windows)
CSV_ENCODING = "utf-8-sig"

# Print a heartbeat every N file copies
PROGRESS_EVERY = 500

# Each source declares either:
#   type == "pre-split":
#       root         -> source root containing one folder per split
#       split_map    -> {source_split_name: target_split_name}
#       audio_subdir -> name of the audio subfolder within each split
#       metadata     -> metadata.csv name within each split
#   type == "unsplit":
#       audio_dir    -> folder with all audio files
#       metadata     -> single metadata.csv file
#
# Folder layout assumed (edit the `root` / `audio_dir` / `metadata` paths
# below to match where these folders actually live on your machine):
#
#   common_voice_local/
#       train/      audio/  metadata_cleaned.csv
#       validation/ audio/  metadata_cleaned.csv
#       test/       audio/  metadata_cleaned.csv
#
#   fleurs_local/
#       train/      audio/  metadata_cleaned.csv
#       validation/ audio/  metadata_cleaned.csv
#       test/       audio/  metadata_cleaned.csv
#
#   thai_elderly_local/train/         audio/  metadata_cleaned.csv
#   thai_isan_dialect_local/train/    audio/  metadata_cleaned.csv
#
SOURCES = {
    "commonvoice": {
        "type": "pre-split",
        "root": Path(r"C:\Users\rocha\AI_builders\common_voice_local"),
        "split_map": {
            "train": "train",
            "validation": "val",
            "test": "test",
        },
        "audio_subdir": "audio",
        "metadata": "metadata_cleaned.csv",
    },
    "fleurs": {
        "type": "pre-split",
        "root": Path(r"C:\Users\rocha\AI_builders\fleurs_local"),
        "split_map": {
            "train": "train",
            "validation": "val",
            "test": "test",
        },
        "audio_subdir": "audio",
        "metadata": "metadata_cleaned.csv",
    },
    "thai_elderly": {
        "type": "unsplit",
        "audio_dir": Path(r"C:\Users\rocha\AI_builders\thai_elderly_local\train\audio"),
        "metadata": Path(r"C:\Users\rocha\AI_builders\thai_elderly_local\train\metadata_cleaned.csv"),
    },
    "thai_isan": {
        "type": "unsplit",
        "audio_dir": Path(r"C:\Users\rocha\AI_builders\thai_isan_dialect_local\train\audio"),
        "metadata": Path(r"C:\Users\rocha\AI_builders\thai_isan_dialect_local\train\metadata_cleaned.csv"),
    },
}

# ============================================================
# HELPERS
# ============================================================
def ensure_speech_dirs() -> None:
    """Create `final_dataset/<split>/speech/` for every split."""
    for s in TARGET_SPLITS:
        (FINAL_DATASET_DIR / s / "speech").mkdir(parents=True, exist_ok=True)


def reset_master_csvs() -> None:
    """Delete any existing master metadata.csv so append mode starts clean."""
    for s in TARGET_SPLITS:
        out_csv = FINAL_DATASET_DIR / s / "metadata.csv"
        if out_csv.exists():
            out_csv.unlink()


def normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a `file_name` column exists; rename common alternatives."""
    if "file_name" in df.columns:
        return df.copy()
    for alt in ("filename", "path", "audio_path", "audio", "file"):
        if alt in df.columns:
            return df.rename(columns={alt: "file_name"}).copy()
    raise ValueError(
        f"metadata is missing a recognizable filename column. "
        f"Columns found: {list(df.columns)}"
    )


def basename_only(name: str) -> str:
    """Strip any directory component (some CSVs store relative paths)."""
    return os.path.basename(str(name))


def split_indices(n: int) -> dict[str, list[int]]:
    """Reproducible 70/15/15 split using random.seed(42)."""
    idx = list(range(n))
    random.shuffle(idx)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    return {
        "train": idx[:n_train],
        "val":   idx[n_train:n_train + n_val],
        "test":  idx[n_train + n_val:],
    }


# Module-level counter so progress is global across all chunks.
_files_copied_total = 0


def _copy_one(src_audio_dir: Path, original_name: str,
              target_split: str, source_prefix: str) -> str:
    """
    Copy one audio file to:
        final_dataset/<target_split>/speech/<source_prefix>_<basename>
    Return the relative path string for the metadata.csv `file_name` column.
    Emits a progress line every PROGRESS_EVERY copies.
    """
    global _files_copied_total

    original_basename = basename_only(original_name)
    new_filename = f"{source_prefix}_{original_basename}"
    src = src_audio_dir / original_basename
    dst = FINAL_DATASET_DIR / target_split / "speech" / new_filename

    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {src}")

    shutil.copy2(src, dst)

    _files_copied_total += 1
    if _files_copied_total % PROGRESS_EVERY == 0:
        print(f"    ... {_files_copied_total} files copied so far", flush=True)

    return f"speech/{new_filename}"


# ============================================================
# CORE: copy + stream-write
# ============================================================
def process_and_copy(
    source_prefix: str,
    audio_dir: Path,
    df: pd.DataFrame,
    target_split: str,
) -> pd.DataFrame:
    """
    Copy every audio file referenced by `df` into the target split's
    speech/ folder using the source prefix, and return a NEW DataFrame
    with `file_name` rewritten to the relative `speech/<prefixed>` form
    and a `source` column added.

    File copies emit a heartbeat every PROGRESS_EVERY files via _copy_one.
    """
    new_paths: list[str] = []
    for original in df["file_name"].tolist():
        new_paths.append(
            _copy_one(audio_dir, original, target_split, source_prefix)
        )

    out = df.copy()
    out["file_name"] = new_paths
    out["source"] = source_prefix
    return out


# Track header state per split so the first writer creates the file
# with a header and subsequent writers append without one.
_first_write_per_split: dict[str, bool] = {s: True for s in TARGET_SPLITS}


def append_chunk_to_master(chunk: pd.DataFrame, target_split: str) -> None:
    """
    Append a processed chunk to final_dataset/<target_split>/metadata.csv.
    First call per split: mode='w', header=True.
    Subsequent calls   : mode='a', header=False.
    Memory footprint   : O(chunk), not O(total).
    """
    if chunk.empty:
        return

    out_csv = FINAL_DATASET_DIR / target_split / "metadata.csv"
    is_first = _first_write_per_split[target_split]

    chunk.to_csv(
        out_csv,
        mode="w" if is_first else "a",
        header=is_first,
        index=False,
        encoding=CSV_ENCODING,
    )
    _first_write_per_split[target_split] = False


# ============================================================
# ORCHESTRATION
# ============================================================
def integrate_data() -> None:
    """
    Walk every source, copy audio + stream-write its rows to the master
    metadata.csv for the right split. Nothing is held in memory across
    sources -- each chunk is flushed to disk before the next is loaded.
    """
    print(f"[INFO] random.seed       : {RANDOM_SEED}")
    print(f"[INFO] final_dataset dir : {FINAL_DATASET_DIR.resolve()}")
    print(f"[INFO] csv encoding      : {CSV_ENCODING}")

    ensure_speech_dirs()
    reset_master_csvs()

    rows_per_split = {s: 0 for s in TARGET_SPLITS}

    for prefix, cfg in SOURCES.items():
        print(f"\n[INFO] Processing source: {prefix} ({cfg['type']})")

        if cfg["type"] == "pre-split":
            for src_split, tgt_split in cfg["split_map"].items():
                split_dir = cfg["root"] / src_split
                meta_path = split_dir / cfg["metadata"]
                audio_dir = split_dir / cfg["audio_subdir"]

                if not meta_path.exists():
                    print(f"  [WARN] missing {meta_path} -- skipping")
                    continue

                df = normalize_metadata(
                    pd.read_csv(meta_path, encoding=CSV_ENCODING)
                )
                print(f"  {prefix} -> {tgt_split}: {len(df)} rows  "
                      f"(audio: {audio_dir})")

                processed = process_and_copy(prefix, audio_dir, df, tgt_split)
                append_chunk_to_master(processed, tgt_split)
                rows_per_split[tgt_split] += len(processed)

                # Drop references so the GC can reclaim memory immediately.
                del df, processed

        elif cfg["type"] == "unsplit":
            df = normalize_metadata(
                pd.read_csv(cfg["metadata"], encoding=CSV_ENCODING)
            )
            splits = split_indices(len(df))
            print(f"  {prefix} totals -> "
                  + ", ".join(f"{k}:{len(v)}" for k, v in splits.items()))

            for tgt_split, indices in splits.items():
                chunk = df.iloc[indices].reset_index(drop=True)
                processed = process_and_copy(
                    prefix, cfg["audio_dir"], chunk, tgt_split
                )
                append_chunk_to_master(processed, tgt_split)
                rows_per_split[tgt_split] += len(processed)

                del chunk, processed

            del df

        else:
            raise ValueError(f"Unknown source type: {cfg['type']}")

    # ----- summary -----
    print(f"\n[INFO] Total audio files copied: {_files_copied_total}")
    for s in TARGET_SPLITS:
        out_csv = FINAL_DATASET_DIR / s / "metadata.csv"
        print(f"  {s:<5}: {rows_per_split[s]:>6} rows -> {out_csv}")
    print("\n[OK] Integration complete. Existing noise/ folders untouched.")


if __name__ == "__main__":
    integrate_data()