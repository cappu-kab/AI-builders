"""
preprocess.py
=============
Convert raw audio (wav/flac/mp3) into the .npy layout expected by both
CRN/ and speech_denoise/ trainers.

Input layout (any of these works):
    IN_ROOT/clean/*.wav
    IN_ROOT/noise/*.wav
    IN_ROOT/{train,val,test}/{clean,noise}/*.wav   # already split

Output layout:
    OUT_ROOT/{train,val,test}/{clean,noise}/*.npy   # mono float32 @ sample_rate
    OUT_ROOT/{train,val,test}/labels_npy.csv         # passes through if input dir has it

Operations per file (in order):
    1. Load (soundfile)
    2. Stereo → mono (mean across channels)
    3. Resample to target sample_rate (linear interp; or scipy if available)
    4. Trim leading/trailing silence below --silence_db threshold
    5. Loudness normalise to target LUFS (pyloudnorm if installed; else RMS-based)
    6. Drop NaN/Inf, clip peak to 0.99
    7. Save as float32 .npy

Usage
-----
    # Already-split input
    python preprocess.py --in_root ./raw_audio --out_root ./data_npy

    # Single-folder input → auto split 80/10/10
    python preprocess.py --in_root ./raw_audio --out_root ./data_npy --auto_split

    # Run on a single split only
    python preprocess.py --in_root ./raw_audio --out_root ./data_npy --split train
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("[preprocess] FATAL: soundfile not installed. pip install soundfile", file=sys.stderr)
    raise

try:
    import pyloudnorm as pyln
    _HAVE_LOUDNORM = True
except ImportError:
    _HAVE_LOUDNORM = False

try:
    from scipy.signal import resample_poly
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    if _HAVE_SCIPY:
        from math import gcd
        g = gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    # crude linear interp fallback
    n_out = int(round(x.size * sr_out / sr_in))
    idx = np.linspace(0.0, x.size - 1, n_out)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, x.size - 1)
    frac = (idx - i0).astype(np.float32)
    return ((1.0 - frac) * x[i0] + frac * x[i1]).astype(np.float32)


def _trim_silence(x: np.ndarray, sr: int, threshold_db: float = -40.0,
                  frame_ms: int = 20) -> np.ndarray:
    """Trim leading/trailing silence based on per-frame RMS."""
    frame = max(1, int(sr * frame_ms / 1000))
    if x.size < frame * 2:
        return x
    n_frames = x.size // frame
    rms = np.sqrt(np.mean(x[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    voiced = db > threshold_db
    if not voiced.any():
        return x  # all-silent file: keep as-is so caller can flag it
    first = int(np.argmax(voiced)) * frame
    last = (n_frames - int(np.argmax(voiced[::-1]))) * frame
    return x[first:last]


def _loudnorm(x: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    if _HAVE_LOUDNORM:
        try:
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(x.astype(np.float64))
            if np.isfinite(loudness):
                gain_db = target_lufs - loudness
                return (x * (10.0 ** (gain_db / 20.0))).astype(np.float32)
        except Exception:
            pass
    # RMS-based fallback (rough but unitless)
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    target_rms = 10.0 ** (target_lufs / 20.0)  # treat target_lufs as dBFS
    if rms > 1e-9:
        return (x * (target_rms / rms)).astype(np.float32)
    return x


def _process_one(args_tuple: Tuple[Path, Path, int, float, float, bool]) -> Tuple[str, str]:
    src, dst, sr_out, target_lufs, silence_db, do_trim = args_tuple
    try:
        x, sr_in = sf.read(str(src), always_2d=False)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = _resample(x, sr_in, sr_out)
        if do_trim:
            x = _trim_silence(x, sr_out, threshold_db=silence_db)
        if np.abs(x).max() < 1e-6:
            return ("silent", str(src))
        x = _loudnorm(x, sr_out, target_lufs)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.abs(x).max())
        if peak > 0.99:
            x = (x / peak * 0.99).astype(np.float32)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, x.astype(np.float32))
        return ("ok", str(dst))
    except Exception as e:
        return ("error", f"{src}: {e}")


def _list_audio(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def _job_list(in_split: Path, out_split: Path, sr_out: int,
              target_lufs: float, silence_db: float, do_trim: bool) -> List[tuple]:
    jobs: List[tuple] = []
    for kind in ("clean", "noise"):
        files = _list_audio(in_split / kind)
        for src in files:
            dst = out_split / kind / (src.stem + ".npy")
            jobs.append((src, dst, sr_out, target_lufs, silence_db, do_trim))
    return jobs


def _passthrough_labels(in_split: Path, out_split: Path) -> None:
    src = in_split / "labels_npy.csv"
    if not src.is_file():
        # also accept labels.csv → renamed
        alt = in_split / "labels.csv"
        if alt.is_file():
            src = alt
        else:
            return
    out_split.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_split / "labels_npy.csv")


def _auto_split(jobs: List[tuple], ratios=(0.8, 0.1, 0.1), seed: int = 42) -> dict:
    """Group flat-input jobs (no train/val/test) into splits."""
    rng = random.Random(seed)
    # group by kind so split fractions are equal across clean/noise
    by_kind: dict = {"clean": [], "noise": []}
    for j in jobs:
        kind = j[1].parent.name  # dst is OUT/<kind>/<file>.npy at this point
        if kind in by_kind:
            by_kind[kind].append(j)
    out = {"train": [], "val": [], "test": []}
    for kind, lst in by_kind.items():
        rng.shuffle(lst)
        n = len(lst)
        if n == 0:
            continue
        if n < 3:
            # too few to split: put everything in train; user can fix manually
            out["train"].extend(lst)
            continue
        n_tr = max(1, int(round(n * ratios[0])))
        n_va = max(1, int(round(n * ratios[1])))
        # ensure test gets at least 1
        if n - n_tr - n_va < 1:
            n_tr = max(1, n - n_va - 1)
        out["train"].extend(lst[:n_tr])
        out["val"].extend(lst[n_tr:n_tr + n_va])
        out["test"].extend(lst[n_tr + n_va:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--sample_rate", type=int, default=16_000)
    ap.add_argument("--target_lufs", type=float, default=-23.0,
                    help="Target loudness in LUFS (or dBFS if pyloudnorm not installed).")
    ap.add_argument("--silence_db", type=float, default=-40.0,
                    help="Silence threshold for trim (dB below full scale).")
    ap.add_argument("--no_trim", action="store_true",
                    help="Skip silence trimming (useful if your data is already clean).")
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    ap.add_argument("--auto_split", action="store_true",
                    help="Input has flat clean/ noise/ folders (no train/val/test); split 80/10/10.")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    in_root: Path = args.in_root
    out_root: Path = args.out_root
    if not in_root.is_dir():
        print(f"[preprocess] FATAL: {in_root} not found"); sys.exit(1)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[preprocess] in={in_root}  out={out_root}  sr={args.sample_rate}  "
          f"lufs={args.target_lufs}  scipy={_HAVE_SCIPY}  pyloudnorm={_HAVE_LOUDNORM}")

    if args.auto_split:
        # Flat layout under in_root: clean/, noise/
        flat_jobs = _job_list(in_root, out_root, args.sample_rate,
                              args.target_lufs, args.silence_db, not args.no_trim)
        # Re-target dst paths into temp location, then split
        for i, (src, dst, *rest) in enumerate(flat_jobs):
            kind = dst.parent.name
            flat_jobs[i] = (src, out_root / kind / dst.name, *rest)
        groups = _auto_split(flat_jobs)
        all_jobs = []
        for split, lst in groups.items():
            for src, flat_dst, *rest in lst:
                kind = flat_dst.parent.name
                new_dst = out_root / split / kind / flat_dst.name
                all_jobs.append((src, new_dst, *rest))
        jobs_by_split = groups
        for split in ("train", "val", "test"):
            (out_root / split).mkdir(parents=True, exist_ok=True)
    else:
        splits = ["train", "val", "test"] if args.split == "all" else [args.split]
        all_jobs: List[tuple] = []
        jobs_by_split: dict = {}
        for split in splits:
            in_split = in_root / split
            out_split = out_root / split
            j = _job_list(in_split, out_split, args.sample_rate,
                          args.target_lufs, args.silence_db, not args.no_trim)
            jobs_by_split[split] = j
            all_jobs.extend(j)
            _passthrough_labels(in_split, out_split)

    if not all_jobs:
        print("[preprocess] no audio files found"); sys.exit(1)
    print(f"[preprocess] total files: {len(all_jobs)}")

    n_ok = n_silent = n_err = 0
    if args.workers <= 1:
        for j in all_jobs:
            status, info = _process_one(j)
            if status == "ok": n_ok += 1
            elif status == "silent": n_silent += 1; print(f"  silent: {info}")
            else: n_err += 1; print(f"  error: {info}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_process_one, j) for j in all_jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                status, info = fut.result()
                if status == "ok": n_ok += 1
                elif status == "silent": n_silent += 1
                else: n_err += 1
                if i % 100 == 0:
                    print(f"  [{i}/{len(all_jobs)}] ok={n_ok} silent={n_silent} err={n_err}")

    print(f"[preprocess] done: ok={n_ok}  silent={n_silent}  err={n_err}")
    print(f"[preprocess] output tree: {out_root}")


if __name__ == "__main__":
    main()
