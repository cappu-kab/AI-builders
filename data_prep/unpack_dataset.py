"""
unpack_dataset.py — HuggingFace Parquet → .wav extractor + class router
========================================================================

Step 1 of the refactored ANC pipeline.  Reads the 12 parquet files in
`speech-noise-dataset/data/`, decodes each row's HF-style audio struct,
and writes 16 kHz mono .wav files into THREE buckets:

    <OUT_ROOT>/clean/<source_tag>/speech/<speaker_id>/<basename>.wav
    <OUT_ROOT>/noise/<source_tag>/noise/<basename>.wav
    <OUT_ROOT>/noisy/<source_tag>/noisy/<speaker_id>/<basename>.wav

The 'noisy' bucket holds pre-mixed (clean+noise) samples, kept WITH a
speaker dim so preprocess_v2.py can pair each noisy clip back to its
clean counterpart (by shared numeric index in the filename) and use
those pairs as the held-out val/test set.

Speaker / category folders are derived from the label column AND any
free-text fields (filename, metadata) so downstream preprocess_v2 can
do speaker-disjoint splitting deterministically.

USAGE
-----
1) Probe one parquet to see its schema (dry-run, no files written):

    python unpack_dataset.py --probe \
        "C:/Users/rocha/AI_builders/data_sounds/speech-noise-dataset/data/train-00000-of-00012.parquet"

   Prints columns, dtypes, label distribution, sample row, and a
   suggested CLASS_MAP.  Use this once to confirm routing rules before
   the full run.

2) Full extraction (12 files, multiprocessing):

    python unpack_dataset.py \
        --src "C:/Users/rocha/AI_builders/data_sounds/speech-noise-dataset/data" \
        --out "C:/Users/rocha/AI_builders/data_sounds/extracted" \
        --workers 4

3) Resume mode is implicit: any .wav already on disk is skipped.

ASSUMPTIONS (HuggingFace audio dataset shape)
---------------------------------------------
Each parquet row is expected to have at minimum:
    audio  : struct  { bytes: binary,  path: string }
    label  : int64    (class index, 0-based)

Optional but used when present:
    speaker_id, speaker, client_id, voice_id   → speaker grouping
    filename, file, path                       → fallback speaker grouping
    transcription, sentence, text              → ignored (kept in side-car JSON)
    category, class_name                       → noise sub-bucket

If the actual schema differs, --probe prints what it found and you can
edit the CLASS_MAP block at the top of this file.
"""

# ╔═══════════════════════════════════════════════════════════════╗
# ║                  ★  YOUR CONFIG — EDIT HERE  ★               ║
# ╚═══════════════════════════════════════════════════════════════╝

# Default source/destination roots (overridable via CLI flags)
DEFAULT_SRC = r"C:\Users\rocha\AI_builders\data_sounds\speech-noise-dataset\data"
DEFAULT_OUT = r"C:\Users\rocha\AI_builders\data_sounds\extracted"

# Target audio params
SR             = 16_000
TARGET_DTYPE   = "float32"  # written as 16-bit PCM by soundfile

# Class routing.
#   - Keys are label values as they appear in the 'label' column.
#     This dataset uses STRING labels (confirmed via --probe):
#       'clean_speech' / 'noise_only' / 'noisy_speech'
#   - Values are ('<bucket>', '<sub_folder>') where <bucket> is one of
#     'clean' | 'noise' | 'noisy'.
#   - 'clean'  → clean studio speech                    (speaker dim kept)
#   - 'noise'  → background noise only                  (speaker dim collapsed)
#   - 'noisy'  → pre-mixed clean+noise (val/test source) (speaker dim kept)
# Rows with labels not in this map are routed to 'unknown/<label>' and
# reported at the end.  Use --probe to verify the label distribution.
CLASS_MAP = {
    "clean_speech": ("clean", "speech"),
    "noise_only":   ("noise", "noise"),
    "noisy_speech": ("noisy", "noisy"),
    # Legacy int keys kept as a fallback in case other shards still encode
    # the label as an int (some HF builds use ClassLabel int features).
    0: ("clean", "speech"),
    1: ("noise", "noise"),
    2: ("noisy", "noisy"),
}

# Fields we'll search for a usable speaker ID (in priority order)
SPEAKER_FIELDS = (
    "speaker_id", "speaker", "client_id", "voice_id",
    "speaker_uid", "spkr", "spk_id",
)

# Fields we'll search for an original filename (used as basename + as
# a last-resort speaker grouping when no real speaker column is present)
FILENAME_FIELDS = ("filename", "file", "path", "audio_path", "name")

# Skip rows whose decoded audio is shorter than this many seconds
MIN_DURATION_S = 0.5

# ╔═══════════════════════════════════════════════════════════════╗
# ║          END OF CONFIG — implementation below                ║
# ╚═══════════════════════════════════════════════════════════════╝

import argparse
import io
import json
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# LAZY IMPORTS  (heavy deps only loaded when actually needed)
# ════════════════════════════════════════════════════════════════

def _import_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa
        import pyarrow as pa          # noqa
        return pq, pa
    except ImportError:
        sys.exit(
            "  ERROR: pyarrow is required.\n"
            "    pip install pyarrow\n")


def _import_audio():
    try:
        import soundfile as sf
        import librosa
        return sf, librosa
    except ImportError:
        sys.exit(
            "  ERROR: soundfile and librosa are required.\n"
            "    pip install soundfile librosa\n")


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

_NON_FS_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(s: str, fallback: str = "x") -> str:
    """Slugify any string into a filesystem-safe basename."""
    if s is None:
        return fallback
    s = str(s).strip().replace(os.sep, "_").replace("/", "_")
    s = _NON_FS_RE.sub("_", s)
    s = s.strip("._")
    return s or fallback


def get_first(row: Dict[str, Any], keys) -> Optional[Any]:
    """Return the first non-None value among the given keys in a row dict."""
    for k in keys:
        if k in row and row[k] is not None and row[k] != "":
            return row[k]
    return None


def derive_speaker_id(row: Dict[str, Any], fallback_idx: int) -> str:
    """
    Pull a speaker grouping out of the row.  Priority:
       1) explicit speaker columns
       2) filename prefix up to first underscore/dash
       3) deterministic 'spkr_<idx>' (last resort — disables speaker-disjoint)
    """
    spk = get_first(row, SPEAKER_FIELDS)
    if spk is not None:
        return safe_name(spk, fallback="spkr")

    fname = get_first(row, FILENAME_FIELDS)
    if fname:
        base = os.path.basename(str(fname))
        base = re.split(r"[_\-.]", base, maxsplit=1)[0]
        if base:
            return safe_name(base, fallback="spkr")

    return f"spkr_{fallback_idx:06d}"


def decode_audio_struct(audio_field: Any, sr_target: int = SR
                        ) -> Optional[Tuple[np.ndarray, str]]:
    """
    Decode HF audio cell -> (mono float32 @ sr_target, original_basename).
    Returns None on failure.
    """
    sf, librosa = _import_audio()

    raw_bytes = None
    orig_path = None
    if isinstance(audio_field, dict):
        raw_bytes = audio_field.get("bytes")
        orig_path = audio_field.get("path")
    elif isinstance(audio_field, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(audio_field)
    else:
        # Some HF builds store it as a list/struct of two columns
        try:
            raw_bytes = audio_field["bytes"]
            orig_path = audio_field.get("path") if hasattr(audio_field, "get") else None
        except Exception:
            return None

    if not raw_bytes:
        return None

    try:
        with io.BytesIO(raw_bytes) as buf:
            audio, sr = sf.read(buf, dtype=TARGET_DTYPE, always_2d=False)
    except Exception:
        return None

    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    if sr != sr_target:
        try:
            audio = librosa.resample(audio.astype(np.float32),
                                     orig_sr=sr, target_sr=sr_target)
        except Exception:
            return None

    if len(audio) < int(MIN_DURATION_S * sr_target):
        return None

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-8:
        audio = audio / peak

    base = "audio"
    if orig_path:
        base = os.path.splitext(os.path.basename(str(orig_path)))[0] or base

    return audio.astype(np.float32), safe_name(base, fallback="audio")


# ════════════════════════════════════════════════════════════════
# PROBE MODE  (schema dry-run)
# ════════════════════════════════════════════════════════════════

def probe_parquet(path: str, n_rows: int = 5) -> None:
    pq, pa = _import_pyarrow()
    print(f"\n══ Probing {path}")
    if not os.path.exists(path):
        sys.exit(f"  ERROR: not found: {path}")

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    print(f"\n  Total rows : {pf.metadata.num_rows:,d}")
    print(f"  Columns    :")
    for f in schema:
        print(f"    {f.name:<20s}  {f.type}")

    # Pull first batch only (tiny — for sample inspection)
    batch_iter = pf.iter_batches(batch_size=max(n_rows, 16))
    first = next(batch_iter)
    rows = first.to_pylist()[:n_rows]

    # Label distribution from this single batch (probabilistic but useful)
    if "label" in [f.name for f in schema]:
        labels = [r.get("label") for r in first.to_pylist()]
        dist = Counter(labels)
        print(f"\n  Label dist (first batch only):")
        for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
            mapped = CLASS_MAP.get(k, ("UNKNOWN", "—"))
            print(f"    label={k!s:<8s}  count={v:<6d}  → {mapped[0]}/{mapped[1]}")

    print(f"\n  Sample rows (first {len(rows)}):")
    for i, r in enumerate(rows):
        # Strip audio bytes from the printed view
        view = {}
        for k, v in r.items():
            if isinstance(v, dict) and "bytes" in v:
                nb = len(v["bytes"]) if v["bytes"] else 0
                view[k] = f"<audio: {nb} bytes, path={v.get('path')!r}>"
            elif isinstance(v, (bytes, bytearray)):
                view[k] = f"<{len(v)} bytes>"
            else:
                view[k] = v
        print(f"    [{i}] {view}")

    print(f"\n  Suggested CLASS_MAP (verify against your dataset card):")
    print(f"    CLASS_MAP = {{")
    if "label" in [f.name for f in schema]:
        observed = sorted(
            set(r.get("label") for r in first.to_pylist() if r.get("label") is not None),
            key=lambda x: str(x),
        )
        for k in observed:
            if k in CLASS_MAP:
                current = CLASS_MAP[k]
            else:
                # Heuristic guess from the label string itself
                ks = str(k).lower()
                if "clean" in ks:
                    current = ("clean", "speech")
                elif "noisy" in ks:
                    current = ("noisy", "noisy")
                elif "noise" in ks:
                    current = ("noise", "noise")
                elif k == 0:
                    current = ("clean", "speech")
                elif k == 1:
                    current = ("noise", "noise")
                elif k == 2:
                    current = ("noisy", "noisy")
                else:
                    current = ("unknown", str(k))
            print(f"        {k!r}: {current!r},")
    print(f"    }}")
    print(f"\n  Buckets: 'clean' (keeps speaker), 'noise' (no speaker), "
          f"'noisy' (keeps speaker, used as paired val/test source).")
    print(f"  If labels above match your intent, run without --probe.\n")


# ════════════════════════════════════════════════════════════════
# EXTRACTION
# ════════════════════════════════════════════════════════════════

def extract_one_parquet(parquet_path: str, out_root: str,
                        source_tag: str,
                        global_counter: Dict[str, int]) -> Dict[str, int]:
    """
    Stream one parquet file row-by-row, write .wav files, return per-class counts.
    """
    pq, pa = _import_pyarrow()
    sf, _ = _import_audio()

    pf = pf_total = pq.ParquetFile(parquet_path)
    n_rows = pf.metadata.num_rows
    print(f"\n  [{source_tag}]  {os.path.basename(parquet_path)}  ({n_rows:,d} rows)")

    counts = Counter()
    skipped = Counter()

    # We chunk in batches to bound memory
    batch_size = 128
    cursor = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        rows = batch.to_pylist()
        for row in rows:
            cursor += 1
            audio_field = row.get("audio")
            if audio_field is None:
                skipped["no_audio_field"] += 1
                continue

            decoded = decode_audio_struct(audio_field, SR)
            if decoded is None:
                skipped["decode_fail"] += 1
                continue
            audio, orig_base = decoded

            label = row.get("label")
            if label is None or label not in CLASS_MAP:
                bucket, sub = "unknown", f"label_{label}"
            else:
                bucket, sub = CLASS_MAP[label]

            spk = derive_speaker_id(row, fallback_idx=cursor)

            # Folder layout: <out>/<bucket>/<source_tag>/<sub>/...
            #   - 'clean' keeps speaker dim → .../<spk>/<base>.wav
            #   - 'noisy' keeps speaker dim → .../<spk>/<base>.wav
            #     (so preprocess_v2 can pair noisy↔clean by shared index)
            #   - 'noise' collapses speaker dim → .../<base>.wav
            #   - 'unknown' collapses speaker dim too
            if bucket in ("clean", "noisy"):
                out_dir = os.path.join(out_root, bucket, source_tag, sub, spk)
            else:
                out_dir = os.path.join(out_root, bucket, source_tag, sub)
            os.makedirs(out_dir, exist_ok=True)

            # Make basename unique per source+row (handles collisions across speakers).
            # We KEEP the original basename as a prefix so 'clean' and 'noisy' rows
            # that came from the same source index (e.g. clean/1.wav ↔ noisy/1.wav)
            # can still be matched up by their leading numeric token in preprocess_v2.
            global_counter["n"] += 1
            uid = global_counter["n"]
            base = f"{orig_base}_{uid:08d}.wav"
            out_path = os.path.join(out_dir, base)

            # Resume: skip if exists
            if os.path.exists(out_path):
                counts[f"{bucket}/{sub}"] += 1
                continue

            try:
                sf.write(out_path, audio, SR, subtype="PCM_16")
                counts[f"{bucket}/{sub}"] += 1
            except Exception as e:
                skipped[f"write_fail:{type(e).__name__}"] += 1

    if skipped:
        print(f"    skipped: {dict(skipped)}")
    print(f"    written: {dict(counts)}")
    return dict(counts)


def run_extract(src_dir: str, out_root: str, workers: int = 1) -> None:
    """Find all .parquet files under src_dir and extract them."""
    parquet_files = sorted(str(p) for p in Path(src_dir).rglob("*.parquet"))
    if not parquet_files:
        sys.exit(f"  ERROR: no .parquet files in {src_dir}")

    print(f"\n══ Found {len(parquet_files)} parquet file(s) ══")
    for p in parquet_files:
        print(f"    {os.path.basename(p)}")

    os.makedirs(out_root, exist_ok=True)

    # Source-tag is the parent dir's basename (handles multi-source datasets)
    source_tag = safe_name(os.path.basename(os.path.dirname(src_dir.rstrip(os.sep + "/"))),
                            fallback="speech_noise")

    # We use a serial loop with an internal counter to keep filenames unique.
    # Multiprocessing here would race on the counter; the per-file work is
    # already streamed batch-by-batch which keeps RAM bounded.  If you want
    # to parallelise, run separate shells over disjoint parquet shards.
    if workers and workers > 1:
        print(f"\n  NOTE: --workers ignored in current implementation "
              f"(single-process avoids filename races).  "
              f"To parallelise, run multiple shells on disjoint shards.\n")

    grand_counts = Counter()
    counter = {"n": 0}
    for pf_path in parquet_files:
        c = extract_one_parquet(pf_path, out_root, source_tag, counter)
        grand_counts.update(c)

    # Write a summary side-car
    summary = {
        "src":          src_dir,
        "out":          out_root,
        "source_tag":   source_tag,
        "sr":           SR,
        "class_map":    {str(k): v for k, v in CLASS_MAP.items()},
        "n_written":    counter["n"],
        "by_class":     dict(grand_counts),
    }
    summary_path = os.path.join(out_root, f"unpack_summary_{source_tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n══ Done ════════════════════════════════════════════════")
    for k, v in sorted(grand_counts.items()):
        print(f"  {k:<24s}  {v:>8,d}")
    print(f"\n  Total .wav written: {counter['n']:,d}")
    print(f"  Summary           : {summary_path}")
    print(f"\nNext step → preprocess_v2.py")
    print(f"  EXTRACTED_ROOT       = {out_root!r}")
    print(f"  EXTRACTED_NOISE_ROOT = {os.path.join(out_root, 'noise')!r}")
    print(f"  Add to NOISY_PAIRED_ROOTS:  {os.path.join(out_root, 'noisy')!r}")
    print(f"  (the 'noisy' bucket is your paired val/test source — "
          f"point preprocess_v2's NOISY_PAIRED_ROOTS at it)\n")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unpack HF parquet audio dataset into .wav files "
                    "routed to clean_speech/ and noise/.")
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="Folder containing .parquet files "
                         f"(default: {DEFAULT_SRC})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="Output root for clean/ and noise/ folders "
                         f"(default: {DEFAULT_OUT})")
    ap.add_argument("--probe", default=None, metavar="PARQUET",
                    help="Probe ONE parquet file: print schema, label distribution, "
                         "and a suggested CLASS_MAP.  Use before the full run.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Currently 1 (see code comment).  Reserved for future use.")
    args = ap.parse_args()

    print("=" * 64)
    print("  unpack_dataset.py — HF parquet → .wav extractor")
    print("=" * 64)

    if args.probe:
        probe_parquet(args.probe)
        return

    print(f"  src     : {args.src}")
    print(f"  out     : {args.out}")
    print(f"  sr      : {SR}")
    print(f"  classes : {CLASS_MAP}")
    run_extract(args.src, args.out, workers=args.workers)


if __name__ == "__main__":
    main()