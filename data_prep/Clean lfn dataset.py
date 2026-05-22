"""
LFN Best-of-One Extractor
=========================
Scan each source WAV for the best 5-second window of pure low-frequency
noise, extract that exact segment, standardize it (16 kHz mono, -25 dBFS
RMS, -1 dBFS hard limit), and save one output file per source.

Window must satisfy ALL three criteria to be a candidate:
    1. LF dominant : energy <200 Hz  >= 80% of total
    2. No speech/HF: energy 200-4000 Hz <= 20% of total
    3. Consistent  : window's mean frame-RMS within 40 dB of file's peak

Selection: the candidate with the *highest* LF ratio wins. Ties broken by
earliest offset.

If a file has zero passing windows, it's skipped — no output, logged as
"Fail / no_valid_window" in the report.

Output layout:
    <output>/<category>_<source_stem>.wav         # one per passing source
    <output>/extraction_report.csv                # always written

Dependencies: librosa, numpy, scipy.

Usage:
    python extract_best_lfn.py --source ./raw_audio
    python extract_best_lfn.py --source ./dataset --output ./best_lfn --stride 0.5
"""
from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import librosa
from scipy.io import wavfile

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

CONFIG = {
    "TARGET_SR": 16000,
    "CHUNK_LEN_SAMPLES": 80_000,            # 5 s @ 16 kHz
    "CANDIDATE_STRIDE_SAMPLES": 16_000,     # 1 s between candidate window starts

    # Stricter "pure noise" thresholds
    "LFN_CUTOFF_HZ": 200.0,
    "LFN_MIN_RATIO": 0.80,                  # > 80% energy below 200 Hz
    "SPEECH_BAND_HZ": (200.0, 4000.0),
    "SPEECH_BAND_MAX_RATIO": 0.20,          # < 20% energy in 200-4000 Hz
    "SILENCE_TOP_DB": 40.0,                 # frames > 40 dB below file peak = silent
    "MIN_NON_SILENT_RATIO": 0.80,           # >= 80% of frames in window must be non-silent

    # DSP
    "TARGET_RMS_DB": -25.0,
    "PEAK_CEILING_DB": -1.0,

    # STFT
    "N_FFT": 2048,
    "HOP_LENGTH": 512,

    # I/O
    "OUTPUT_DIR": Path("./best_lfn"),
}


# -----------------------------------------------------------------------------
# Result row
# -----------------------------------------------------------------------------

@dataclass
class FileResult:
    rel_path: str
    status: str                   # 'Pass' | 'Fail'
    output_filename: str = ""
    offset_sec: float = 0.0
    lf_ratio: float = 0.0
    speech_ratio: float = 0.0
    candidates_total: int = 0
    candidates_passed: int = 0
    reason: str = ""              # empty when Pass


# -----------------------------------------------------------------------------
# Categorize from first-level subfolder under source root
# -----------------------------------------------------------------------------

def categorize(file_path: Path, source_root: Path) -> str:
    try:
        rel = file_path.relative_to(source_root)
    except ValueError:
        return "noise"
    parts = rel.parts
    if len(parts) > 1 and parts[0]:
        return parts[0].lower().replace(" ", "_")
    return "noise"


# -----------------------------------------------------------------------------
# DSP — RMS normalize then hard limit
# -----------------------------------------------------------------------------

def normalize_and_limit(y: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y * y))) + 1e-12
    target_rms = 10.0 ** (CONFIG["TARGET_RMS_DB"] / 20.0)
    y = y * (target_rms / rms)
    ceiling = 10.0 ** (CONFIG["PEAK_CEILING_DB"] / 20.0)
    y = np.clip(y, -ceiling, ceiling)
    return y.astype(np.float32, copy=False)


def write_pcm16_wav(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, y16)


# -----------------------------------------------------------------------------
# Per-file scan
# -----------------------------------------------------------------------------

@dataclass
class BestPick:
    start_sample: int
    lf_ratio: float
    speech_ratio: float


def scan_for_best(y: np.ndarray, sr: int) -> Tuple[Optional[BestPick], int, int, int, int, int]:
    """Slide a 5-s window across the file and return the highest-LF passing window.

    Returns (best_or_None, candidates_total, passed, fail_silence, fail_speech, fail_lf).
    """
    n_fft = CONFIG["N_FFT"]
    hop = CONFIG["HOP_LENGTH"]
    chunk_len = CONFIG["CHUNK_LEN_SAMPLES"]
    stride = CONFIG["CANDIDATE_STRIDE_SAMPLES"]

    if len(y) < chunk_len:
        return None, 0, 0, 0, 0, 0

    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))   # (n_freq, n_frames)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]

    if len(rms) == 0 or stft.shape[1] == 0:
        return None, 0, 0, 0, 0, 0

    file_peak = float(rms.max())
    if file_peak <= 1e-10:
        return None, 0, 0, 0, 0, 0
    silence_threshold = file_peak * (10.0 ** (-CONFIG["SILENCE_TOP_DB"] / 20.0))

    lf_mask = freqs < CONFIG["LFN_CUTOFF_HZ"]
    sp_lo, sp_hi = CONFIG["SPEECH_BAND_HZ"]
    sp_mask = (freqs >= sp_lo) & (freqs < sp_hi)

    candidates_total = 0
    passed = 0
    fail_silence = fail_speech = fail_lf = 0
    best: Optional[BestPick] = None

    cursor = 0
    while cursor + chunk_len <= len(y):
        candidates_total += 1
        f0 = cursor // hop
        f1 = (cursor + chunk_len) // hop

        win_rms = rms[f0:f1]
        win_stft = stft[:, f0:f1]

        # 1. Silence consistency
        if len(win_rms) == 0:
            fail_silence += 1
            cursor += stride
            continue
        non_silent = float((win_rms > silence_threshold).sum()) / float(len(win_rms))
        if non_silent < CONFIG["MIN_NON_SILENT_RATIO"]:
            fail_silence += 1
            cursor += stride
            continue

        total = float(win_stft.sum())
        if total <= 0.0:
            fail_silence += 1
            cursor += stride
            continue

        sp_ratio = float(win_stft[sp_mask].sum() / total)
        lf_ratio = float(win_stft[lf_mask].sum() / total)

        # 2. Speech / HF cap
        if sp_ratio > CONFIG["SPEECH_BAND_MAX_RATIO"]:
            fail_speech += 1
            cursor += stride
            continue

        # 3. LF dominance
        if lf_ratio < CONFIG["LFN_MIN_RATIO"]:
            fail_lf += 1
            cursor += stride
            continue

        # Passed all gates — track the best (highest LF ratio).
        passed += 1
        if best is None or lf_ratio > best.lf_ratio:
            best = BestPick(
                start_sample=cursor,
                lf_ratio=lf_ratio,
                speech_ratio=sp_ratio,
            )

        cursor += stride

    return best, candidates_total, passed, fail_silence, fail_speech, fail_lf


# -----------------------------------------------------------------------------
# Per-file processing
# -----------------------------------------------------------------------------

def process_file(path: Path,
                 source_root: Path,
                 output_dir: Path,
                 results: List[FileResult],
                 log: logging.Logger) -> None:
    try:
        rel = path.relative_to(source_root)
    except ValueError:
        rel = Path(path.name)
    rel_str = str(rel)
    category = categorize(path, source_root)

    try:
        y, _ = librosa.load(str(path), sr=CONFIG["TARGET_SR"], mono=True)
    except Exception as e:
        results.append(FileResult(rel_str, "Fail", reason=f"unreadable: {e}"))
        log.warning(f"unreadable {rel_str}: {e}")
        return

    if len(y) < CONFIG["CHUNK_LEN_SAMPLES"]:
        results.append(FileResult(rel_str, "Fail", reason="too_short"))
        log.info(f"FAIL {rel_str}: too_short")
        return

    best, total, passed, fs, fp, fl = scan_for_best(y, CONFIG["TARGET_SR"])

    if best is None:
        reason = (f"no_valid_window (silence={fs}, speech={fp}, lf={fl}, "
                  f"out of {total})")
        results.append(FileResult(
            rel_str, "Fail",
            candidates_total=total, candidates_passed=0,
            reason=reason,
        ))
        log.info(f"FAIL {rel_str}: {reason}")
        return

    # Extract, standardize, write
    chunk = y[best.start_sample:best.start_sample + CONFIG["CHUNK_LEN_SAMPLES"]].copy()
    chunk = normalize_and_limit(chunk)
    out_name = f"{category}_{path.stem}.wav"
    out_path = output_dir / out_name
    try:
        write_pcm16_wav(out_path, chunk, CONFIG["TARGET_SR"])
    except Exception as e:
        results.append(FileResult(
            rel_str, "Fail",
            candidates_total=total, candidates_passed=passed,
            lf_ratio=best.lf_ratio, speech_ratio=best.speech_ratio,
            offset_sec=best.start_sample / CONFIG["TARGET_SR"],
            reason=f"write_error: {e}",
        ))
        log.warning(f"write failed {out_name}: {e}")
        return

    results.append(FileResult(
        rel_str, "Pass",
        output_filename=out_name,
        offset_sec=best.start_sample / CONFIG["TARGET_SR"],
        lf_ratio=best.lf_ratio,
        speech_ratio=best.speech_ratio,
        candidates_total=total,
        candidates_passed=passed,
    ))
    log.info(f"PASS {rel_str} -> {out_name} "
             f"@ {best.start_sample / CONFIG['TARGET_SR']:.1f}s "
             f"(lf={best.lf_ratio:.3f}, sp={best.speech_ratio:.3f}, "
             f"{passed}/{total} passed)")


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def write_report(results: List[FileResult],
                 output_dir: Path,
                 log: logging.Logger) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "extraction_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "source", "status", "output_filename", "offset_sec",
            "lf_ratio", "speech_ratio",
            "candidates_total", "candidates_passed", "reason",
        ])
        for r in results:
            w.writerow([
                r.rel_path, r.status, r.output_filename,
                f"{r.offset_sec:.2f}",
                f"{r.lf_ratio:.4f}", f"{r.speech_ratio:.4f}",
                r.candidates_total, r.candidates_passed, r.reason,
            ])
    log.info(f"report written: {csv_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source", type=Path, required=True,
                    help="folder containing source WAVs (recursed)")
    ap.add_argument("--output", type=Path, default=CONFIG["OUTPUT_DIR"])
    ap.add_argument("--stride", type=float, default=1.0,
                    help="seconds between candidate window starts (default 1.0)")
    ap.add_argument("--lfn-min", type=float, default=CONFIG["LFN_MIN_RATIO"],
                    help="min fraction of energy below 200 Hz (default 0.80)")
    ap.add_argument("--speech-max", type=float, default=CONFIG["SPEECH_BAND_MAX_RATIO"],
                    help="max fraction of energy in 200-4000 Hz (default 0.20)")
    ap.add_argument("--silence-db", type=float, default=CONFIG["SILENCE_TOP_DB"])
    ap.add_argument("--non-silent-min", type=float, default=CONFIG["MIN_NON_SILENT_RATIO"])
    ap.add_argument("--rms-db", type=float, default=CONFIG["TARGET_RMS_DB"])
    ap.add_argument("--peak-db", type=float, default=CONFIG["PEAK_CEILING_DB"])
    args = ap.parse_args()

    CONFIG["CANDIDATE_STRIDE_SAMPLES"] = max(1, int(round(args.stride * CONFIG["TARGET_SR"])))
    CONFIG["LFN_MIN_RATIO"] = args.lfn_min
    CONFIG["SPEECH_BAND_MAX_RATIO"] = args.speech_max
    CONFIG["SILENCE_TOP_DB"] = args.silence_db
    CONFIG["MIN_NON_SILENT_RATIO"] = args.non_silent_min
    CONFIG["TARGET_RMS_DB"] = args.rms_db
    CONFIG["PEAK_CEILING_DB"] = args.peak_db
    output_dir: Path = args.output

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("extract")

    if not args.source.exists():
        log.error(f"source {args.source} not found")
        return 1

    results: List[FileResult] = []

    interrupted = {"flag": False}
    def handler(signum, _frame):
        if interrupted["flag"]:
            log.error("forced exit")
            sys.exit(130)
        interrupted["flag"] = True
        log.warning(f"caught signal {signum}; writing partial report and exiting")
        try:
            write_report(results, output_dir, log)
        except Exception as e:
            log.error(f"report write failed: {e}")
        sys.exit(130)
    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handler)

    files = sorted(args.source.rglob("*.wav")) + sorted(args.source.rglob("*.WAV"))
    log.info(f"found {len(files)} WAV files under {args.source}")
    log.info(f"thresholds: lfn>={CONFIG['LFN_MIN_RATIO']}, "
             f"speech<={CONFIG['SPEECH_BAND_MAX_RATIO']}, "
             f"non_silent>={CONFIG['MIN_NON_SILENT_RATIO']}")
    log.info(f"stride={CONFIG['CANDIDATE_STRIDE_SAMPLES']/CONFIG['TARGET_SR']:.2f}s, "
             f"rms={CONFIG['TARGET_RMS_DB']} dBFS, peak={CONFIG['PEAK_CEILING_DB']} dBFS")
    log.info(f"output -> {output_dir}")

    start = time.monotonic()
    for i, p in enumerate(files, 1):
        if i == 1 or i % 25 == 0 or i == len(files):
            log.info(f"[{i}/{len(files)}]")
        try:
            process_file(p, args.source, output_dir, results, log)
        except Exception as e:
            log.warning(f"error on {p.name}: {e}")

    elapsed = time.monotonic() - start
    n_pass = sum(1 for r in results if r.status == "Pass")
    n_fail = len(results) - n_pass
    log.info(f"done in {elapsed:.1f}s — {n_pass} extracted, {n_fail} skipped")
    write_report(results, output_dir, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())