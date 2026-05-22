"""
Low-Frequency Noise (LFN) Dataset Pipeline
==========================================
Builds a curated dataset of non-speech low-frequency noises (<200 Hz) from YouTube
for training LFN cancellation models.

Download strategy:
    - Audio-only: yt-dlp prefers `bestaudio/best` so visual streams are never fetched.
    - Short videos (duration < CLIP_SECONDS, default 5 min): download the whole audio.
    - Long videos (duration >= CLIP_SECONDS): split the timeline into SEGMENT_COUNT
      equal zones (default 3 -> beginning / middle / end) and pick one random
      CLIP_SECONDS-long start within each zone. Use yt-dlp's `download_ranges` to
      pull only those bytes from the network — never the full file.
    - Unknown duration: one conservative 5-min segment from offset 0.

Pipeline stages:
    1. Search + download as above (rate-limited, 16 kHz mono WAV).
    2. Tag each downloaded segment with YAMNet (positive + negative class filtering).
    3. Spectrally validate that each chunk's energy is concentrated below 200 Hz.
    4. Export 3–5 s WAV chunks to:
         /dataset/<class>/<video_id>_seg<NN>_<chunk_start>-<chunk_end>.wav
       and emit a CSV + JSON summary log.

Author: generated for Cappu
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np  # noqa: F401  # for static analyzers only

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

CONFIG = {
    # ---- Scraping ----
    "SEARCH_QUERIES": [
        "industrial fan hum",
        "AC compressor noise",
        "low frequency machinery drone",
        "heavy engine idle",
        # "industrial fan hum",
        # "AC compressor noise",
        #"low frequency machinery drone",
    ],
    "VIDEOS_PER_QUERY": 100,
    "MIN_DELAY_SEC": 5,
    "MAX_DELAY_SEC": 15,
    "MIN_DURATION_SEC": 5,
    "MAX_DURATION_SEC": 86400,      # sanity ceiling (24 h) — catches malformed metadata only

    # ---- Segmented download ----
    "CLIP_SECONDS": 300,            # length of each downloaded slice; also the short/long threshold
    "SEGMENT_COUNT": 3,             # number of random zones to sample for long videos
    "DOWNLOAD_TIMEOUT_SEC": 600,    # hard wall-clock cap on a single segment download
    "RESUME_FROM_WORKDIR": True,    # reuse existing WAVs in _lfn_work on restart

    # ---- Audio ----
    "SAMPLE_RATE": 16000,           # YAMNet expects 16 kHz mono
    "CHANNELS": 1,
    "CHUNK_MIN_SEC": 3.0,
    "CHUNK_MAX_SEC": 5.0,

    # ---- YAMNet classification ----
    "YAMNET_HANDLE": "https://tfhub.dev/google/yamnet/1",
    "POSITIVE_CLASSES": [
        "Heavy engine (low frequency)",
        "Vehicle",
        "Engine",
        "Fan",
        "Air conditioning",
    ],
    "NEGATIVE_CLASSES": [
        "Speech",
        "Music",
        "Bird",
        "Bird vocalization, bird call, bird song",
    ],
    "POSITIVE_MIN_CONF": 0.30,
    "NEGATIVE_MAX_CONF": 0.10,

    # ---- Spectral validation ----
    "LFN_CUTOFF_HZ": 200,
    "LFN_MIN_ENERGY_RATIO": 0.60,
    "N_FFT": 2048,
    "HOP_LENGTH": 512,

    # ---- Output ----
    "DATASET_ROOT": Path("./dataset"),
    "WORK_DIR": Path("./_lfn_work"),
    "SUMMARY_CSV": Path("./dataset/summary.csv"),
    "SUMMARY_JSON": Path("./dataset/summary.json"),
    "LOG_FILE": Path("./dataset/pipeline.log"),

    # ---- Mapping from YAMNet label -> folder name ----
    "CLASS_FOLDER_MAP": {
        "Heavy engine (low frequency)": "heavy_engine",
        "Engine": "engine",
        "Vehicle": "vehicle",
        "Fan": "fan",
        "Air conditioning": "air_conditioning",
    },
}

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lfn_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# -----------------------------------------------------------------------------
# LAZY IMPORTS
# -----------------------------------------------------------------------------

def _lazy_imports():
    import numpy as np
    import librosa
    import tensorflow as tf
    import tensorflow_hub as hub
    from pydub import AudioSegment
    import yt_dlp

    return {
        "np": np,
        "librosa": librosa,
        "tf": tf,
        "hub": hub,
        "AudioSegment": AudioSegment,
        "yt_dlp": yt_dlp,
    }


# -----------------------------------------------------------------------------
# DATA CLASSES
# -----------------------------------------------------------------------------

@dataclass
class DownloadedSegment:
    """One downloaded chunk of a video.

    For short videos this is the full audio (segment_idx = 0, is_full_video = True).
    For long videos there are SEGMENT_COUNT of these per video, anchored by random
    offsets distributed across the timeline.
    """
    segment_idx: int
    offset_sec: float
    is_full_video: bool
    wav_path: Path


@dataclass
class VideoRecord:
    video_id: str
    title: str
    url: str
    duration: float
    query: str
    segments: List[DownloadedSegment] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ChunkResult:
    video_id: str
    segment_idx: int
    source_offset_sec: float
    start_sec: float
    end_sec: float
    label: str
    confidence: float
    lfn_ratio: float
    out_path: Path


@dataclass
class Summary:
    per_class_duration: Dict[str, float] = field(default_factory=dict)
    per_class_count: Dict[str, int] = field(default_factory=dict)
    total_videos_scraped: int = 0
    total_videos_failed: int = 0
    total_segments_downloaded: int = 0
    total_chunks_kept: int = 0
    total_chunks_rejected: int = 0


# =============================================================================
# STAGE 1 — SEARCH + DOWNLOAD
# =============================================================================

def search_video_urls(query: str, n: int, logger: logging.Logger) -> List[Dict]:
    deps = _lazy_imports()
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "default_search": f"ytsearch{n}",
        "noplaylist": True,
    }
    try:
        with deps["yt_dlp"].YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
        entries = info.get("entries", []) if info else []
        logger.info(f"[search] '{query}' -> {len(entries)} candidates")
        return entries
    except Exception as e:
        logger.warning(f"[search] query '{query}' failed: {e}")
        return []


def compute_random_offsets(duration: float,
                           clip_sec: float,
                           n_segments: int) -> List[float]:
    """Pick `n_segments` random start offsets, one per equal-sized zone of the video.

    Each zone is [i*duration/n, (i+1)*duration/n]. Within each zone we pick a
    uniform-random start, clamped so the segment ends before `duration`. For
    sources between CLIP_SECONDS and n_segments * CLIP_SECONDS, segments will
    overlap — that's accepted as the cost of always returning n_segments.
    """
    if duration <= 0 or duration <= clip_sec:
        return []  # caller treats empty list as "use full-audio download"

    max_start = max(0.0, duration - clip_sec)
    zone_size = duration / float(n_segments)

    offsets: List[float] = []
    for i in range(n_segments):
        lo = i * zone_size
        hi = min((i + 1) * zone_size, max_start)
        if hi <= lo:
            offsets.append(min(lo, max_start))
        else:
            offsets.append(random.uniform(lo, hi))

    offsets.sort()
    return offsets


def _common_ydl_opts(out_template: str) -> Dict:
    """yt-dlp options shared by both full and segmented downloads."""
    return {
        "quiet": True,
        "no_warnings": True,
        "outtmpl": out_template,
        # Strict audio-only: prefer audio-only formats, fall back to muxed only
        # if no audio-only stream is offered. yt-dlp's ffmpeg post-processor
        # then strips video before encoding to WAV, so visual frames are never
        # written to disk in either branch.
        "format": "bestaudio/best",
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",
        }],
        "postprocessor_args": [
            "-ar", str(CONFIG["SAMPLE_RATE"]),
            "-ac", str(CONFIG["CHANNELS"]),
            # -vn forces ffmpeg to drop any video stream, belt-and-suspenders
            # alongside the audio-only format selector.
            "-vn",
        ],
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": 20,
    }


def _cleanup_non_wav(work_dir: Path, base: str) -> None:
    """Remove any non-wav artifacts (source containers etc.) for this base name."""
    for leftover in work_dir.glob(f"{base}.*"):
        if leftover.suffix.lower() != ".wav":
            try:
                leftover.unlink()
            except OSError:
                pass


def _cleanup_partials(work_dir: Path, base: str) -> None:
    """Remove leftover *.part / *.ytdl files for this base (from a hung download)."""
    for leftover in list(work_dir.glob(f"{base}.*")):
        if leftover.suffix.lower() in (".part", ".ytdl") or ".part" in leftover.name:
            try:
                leftover.unlink()
            except OSError:
                pass


class _DownloadTimeout(Exception):
    """Raised when a single yt-dlp download exceeds DOWNLOAD_TIMEOUT_SEC."""


def _alarm_handler(signum, frame):
    raise _DownloadTimeout("yt-dlp exceeded DOWNLOAD_TIMEOUT_SEC")


def _run_with_timeout(timeout_sec: int, func, *args, **kwargs):
    """Run a callable with a hard wall-clock timeout (Unix/WSL only).

    Raises _DownloadTimeout if the call doesn't return within timeout_sec.
    Falls back to a plain call on platforms without SIGALRM (e.g. native
    Windows), where you'd need a thread-based timeout instead.
    """
    if not hasattr(signal, "SIGALRM"):
        return func(*args, **kwargs)
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(timeout_sec))
    try:
        return func(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _download_full_audio(video_id: str,
                         url: str,
                         work_dir: Path,
                         logger: logging.Logger) -> Optional[Path]:
    """Download the entire audio stream as a single WAV (for short videos)."""
    deps = _lazy_imports()
    base = f"{video_id}_seg00"
    wav_path = work_dir / f"{base}.wav"

    # Resume: skip if we already have this segment from a prior run.
    if CONFIG.get("RESUME_FROM_WORKDIR") and wav_path.exists() and wav_path.stat().st_size > 0:
        logger.info(f"[download-full] {video_id}: reusing existing wav")
        return wav_path

    # Wipe any stale .part files from a previous hang on this base name.
    _cleanup_partials(work_dir, base)

    out_template = str(work_dir / f"{base}.%(ext)s")
    ydl_opts = _common_ydl_opts(out_template)
    # No download_ranges: take the whole stream.

    def _do_download():
        with deps["yt_dlp"].YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    try:
        _run_with_timeout(CONFIG["DOWNLOAD_TIMEOUT_SEC"], _do_download)
    except _DownloadTimeout:
        logger.warning(f"[download-full] {video_id}: timed out after "
                       f"{CONFIG['DOWNLOAD_TIMEOUT_SEC']}s, abandoning")
        _cleanup_partials(work_dir, base)
        return None
    except Exception as e:
        logger.warning(f"[download-full] {video_id} failed: {e}")
        return None

    if not wav_path.exists():
        logger.warning(f"[download-full] {video_id}: wav not produced")
        return None

    _cleanup_non_wav(work_dir, base)
    return wav_path


def _download_segment(video_id: str,
                      url: str,
                      segment_idx: int,
                      offset_sec: float,
                      clip_sec: float,
                      work_dir: Path,
                      logger: logging.Logger) -> Optional[Path]:
    """Download one [offset, offset+clip_sec] slice as WAV using download_ranges."""
    deps = _lazy_imports()
    base = f"{video_id}_seg{segment_idx:02d}"
    wav_path = work_dir / f"{base}.wav"

    # Resume: skip if we already have this segment from a prior run.
    if CONFIG.get("RESUME_FROM_WORKDIR") and wav_path.exists() and wav_path.stat().st_size > 0:
        logger.info(f"[segment] {base}: reusing existing wav")
        return wav_path

    # Wipe any stale .part files from a previous hang on this base name.
    _cleanup_partials(work_dir, base)

    out_template = str(work_dir / f"{base}.%(ext)s")
    start_t = float(offset_sec)
    end_t = start_t + float(clip_sec)

    ydl_opts = _common_ydl_opts(out_template)
    ydl_opts["download_ranges"] = (
        lambda info_dict, _ydl: [{
            "start_time": start_t,
            "end_time": end_t,
        }]
    )
    ydl_opts["force_keyframes_at_cuts"] = True

    def _do_download():
        with deps["yt_dlp"].YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    try:
        _run_with_timeout(CONFIG["DOWNLOAD_TIMEOUT_SEC"], _do_download)
    except _DownloadTimeout:
        logger.warning(
            f"[segment] {base}@{int(offset_sec)}s: timed out after "
            f"{CONFIG['DOWNLOAD_TIMEOUT_SEC']}s, abandoning"
        )
        _cleanup_partials(work_dir, base)
        return None
    except Exception as e:
        logger.warning(f"[segment] {base}@{int(offset_sec)}s failed: {e}")
        return None

    if not wav_path.exists():
        logger.warning(f"[segment] {base}: wav not produced")
        return None

    _cleanup_non_wav(work_dir, base)
    return wav_path


def download_audio(entry: Dict,
                   query: str,
                   work_dir: Path,
                   logger: logging.Logger) -> VideoRecord:
    """Decide short vs long, dispatch to the appropriate downloader."""
    video_id = entry.get("id") or entry.get("url", "unknown")
    url = entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
    title = entry.get("title") or ""
    duration = float(entry.get("duration") or 0.0)

    record = VideoRecord(video_id=video_id, title=title, url=url,
                         duration=duration, query=query)

    # Sanity-only bounds. The 10-min upper cap is gone; only catch obviously
    # broken metadata (negative / >24 h streams) and very short clips.
    if duration and duration < CONFIG["MIN_DURATION_SEC"]:
        record.error = f"duration {duration:.0f}s shorter than MIN_DURATION_SEC"
        return record
    if duration and duration > CONFIG["MAX_DURATION_SEC"]:
        record.error = f"duration {duration:.0f}s exceeds MAX_DURATION_SEC sanity cap"
        return record

    work_dir.mkdir(parents=True, exist_ok=True)
    clip_sec = float(CONFIG["CLIP_SECONDS"])

    # --- Branch A: short video -> full audio download
    if 0 < duration < clip_sec:
        wav = _download_full_audio(video_id, url, work_dir, logger)
        if wav is not None:
            record.segments.append(DownloadedSegment(
                segment_idx=0,
                offset_sec=0.0,
                is_full_video=True,
                wav_path=wav,
            ))

    # --- Branch B: long video (or unknown duration) -> segmented download
    else:
        if duration > 0:
            offsets = compute_random_offsets(duration, clip_sec, CONFIG["SEGMENT_COUNT"])
        else:
            # Unknown duration — be conservative: pull one 5-min segment from start.
            offsets = [0.0]

        logger.info(
            f"[plan] {video_id}: {duration:.0f}s, "
            f"{len(offsets)} segment(s) @ "
            f"{[int(o) for o in offsets]}"
        )

        for idx, offset in enumerate(offsets):
            wav = _download_segment(
                video_id, url, idx, offset, clip_sec, work_dir, logger
            )
            if wav is not None:
                record.segments.append(DownloadedSegment(
                    segment_idx=idx,
                    offset_sec=offset,
                    is_full_video=False,
                    wav_path=wav,
                ))
            # Polite short pause between consecutive segments of the same video.
            if idx < len(offsets) - 1:
                time.sleep(random.uniform(
                    CONFIG["MIN_DELAY_SEC"] / 2.0,
                    CONFIG["MAX_DELAY_SEC"] / 2.0,
                ))

    if not record.segments:
        record.error = record.error or "no segments downloaded"
    return record


def scrape_all(logger: logging.Logger) -> List[VideoRecord]:
    records: List[VideoRecord] = []
    work_dir = CONFIG["WORK_DIR"]

    for query in CONFIG["SEARCH_QUERIES"]:
        entries = search_video_urls(query, CONFIG["VIDEOS_PER_QUERY"], logger)
        for entry in entries:
            rec = download_audio(entry, query, work_dir, logger)
            records.append(rec)
            if rec.error:
                logger.warning(f"[scrape] {rec.video_id}: {rec.error}")
            else:
                logger.info(
                    f"[scrape] {rec.video_id} OK "
                    f"({rec.duration:.0f}s, {len(rec.segments)} segment(s))"
                )

            # Polite delay between videos, even on failure.
            delay = random.uniform(CONFIG["MIN_DELAY_SEC"], CONFIG["MAX_DELAY_SEC"])
            time.sleep(delay)

    return records


# =============================================================================
# STAGE 2 — YAMNET CLASSIFICATION
# =============================================================================

class YamnetTagger:
    def __init__(self, logger: logging.Logger):
        deps = _lazy_imports()
        self.np = deps["np"]
        self.tf = deps["tf"]
        logger.info(f"[yamnet] loading model: {CONFIG['YAMNET_HANDLE']}")
        self.model = deps["hub"].load(CONFIG["YAMNET_HANDLE"])
        self.class_names = self._load_class_names()
        logger.info(f"[yamnet] {len(self.class_names)} classes loaded")

    def _load_class_names(self) -> List[str]:
        class_map_path = self.model.class_map_path().numpy().decode("utf-8")
        names: List[str] = []
        with open(class_map_path) as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    names.append(row[2])
        return names

    def tag(self, waveform) -> "np.ndarray":
        scores, _embeddings, _spec = self.model(waveform)
        return scores.numpy()

    def segment_label(self, scores) -> Tuple[Optional[str], float, Dict[str, float]]:
        np = self.np
        mean_scores = scores.mean(axis=0)

        tracked: Dict[str, float] = {}
        for name in CONFIG["POSITIVE_CLASSES"] + CONFIG["NEGATIVE_CLASSES"]:
            if name in self.class_names:
                idx = self.class_names.index(name)
                tracked[name] = float(mean_scores[idx])

        for neg in CONFIG["NEGATIVE_CLASSES"]:
            if tracked.get(neg, 0.0) > CONFIG["NEGATIVE_MAX_CONF"]:
                return None, 0.0, tracked

        best_label, best_conf = None, 0.0
        for pos in CONFIG["POSITIVE_CLASSES"]:
            conf = tracked.get(pos, 0.0)
            if conf > best_conf:
                best_label, best_conf = pos, conf

        if best_label is None or best_conf < CONFIG["POSITIVE_MIN_CONF"]:
            return None, best_conf, tracked

        return best_label, best_conf, tracked


# =============================================================================
# STAGE 3 — SPECTRAL VALIDATION
# =============================================================================

def lfn_energy_ratio(y, sr: int) -> float:
    deps = _lazy_imports()
    np = deps["np"]
    librosa = deps["librosa"]

    if len(y) < CONFIG["N_FFT"]:
        pad = CONFIG["N_FFT"] - len(y)
        y = np.pad(y, (0, pad), mode="constant")

    stft = np.abs(librosa.stft(y, n_fft=CONFIG["N_FFT"],
                               hop_length=CONFIG["HOP_LENGTH"]))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=CONFIG["N_FFT"])
    total = stft.sum()
    if total <= 0:
        return 0.0

    low_mask = freqs < CONFIG["LFN_CUTOFF_HZ"]
    low_energy = stft[low_mask, :].sum()
    return float(low_energy / total)


def passes_spectral_filter(y, sr: int) -> Tuple[bool, float]:
    ratio = lfn_energy_ratio(y, sr)
    return ratio >= CONFIG["LFN_MIN_ENERGY_RATIO"], ratio


# =============================================================================
# STAGE 4 — SEGMENTATION, EXPORT
# =============================================================================

def iter_chunks(total_sec: float,
                chunk_min: float,
                chunk_max: float) -> Iterable[Tuple[float, float]]:
    cursor = 0.0
    while cursor + chunk_min <= total_sec:
        length = random.uniform(chunk_min, chunk_max)
        end = min(cursor + length, total_sec)
        if end - cursor >= chunk_min:
            yield cursor, end
        cursor = end


def export_wav(audio_segment, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_segment.export(out_path, format="wav")


def _process_segment(record: VideoRecord,
                     segment: DownloadedSegment,
                     tagger: YamnetTagger,
                     summary: Summary,
                     logger: logging.Logger) -> List[ChunkResult]:
    """Classify, validate and export 3–5 s chunks from one downloaded segment."""
    deps = _lazy_imports()
    librosa = deps["librosa"]
    np = deps["np"]
    AudioSegment = deps["AudioSegment"]

    results: List[ChunkResult] = []
    wav_path = segment.wav_path
    seg_idx = segment.segment_idx

    if not wav_path.exists():
        return results

    try:
        full_audio = AudioSegment.from_wav(wav_path)
    except Exception as e:
        logger.warning(f"[process] {record.video_id} seg{seg_idx:02d} unreadable: {e}")
        return results

    full_sec = len(full_audio) / 1000.0

    for start, end in iter_chunks(full_sec,
                                  CONFIG["CHUNK_MIN_SEC"],
                                  CONFIG["CHUNK_MAX_SEC"]):
        chunk = full_audio[int(start * 1000): int(end * 1000)]

        samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
        if chunk.sample_width > 0:
            samples /= float(1 << (8 * chunk.sample_width - 1))
        if chunk.channels > 1:
            samples = samples.reshape(-1, chunk.channels).mean(axis=1)

        sr = chunk.frame_rate
        if sr != CONFIG["SAMPLE_RATE"]:
            samples = librosa.resample(samples, orig_sr=sr,
                                       target_sr=CONFIG["SAMPLE_RATE"])
            sr = CONFIG["SAMPLE_RATE"]

        try:
            scores = tagger.tag(samples)
        except Exception as e:
            logger.warning(
                f"[yamnet] {record.video_id} seg{seg_idx:02d}@{start:.1f}s failed: {e}"
            )
            summary.total_chunks_rejected += 1
            continue

        label, conf, _ = tagger.segment_label(scores)
        if label is None:
            summary.total_chunks_rejected += 1
            continue

        ok, ratio = passes_spectral_filter(samples, sr)
        if not ok:
            summary.total_chunks_rejected += 1
            continue

        folder = CONFIG["CLASS_FOLDER_MAP"].get(label, "other")
        ts_tag = f"{int(start):05d}-{int(end):05d}"
        out_path = (
            CONFIG["DATASET_ROOT"] / folder
            / f"{record.video_id}_seg{seg_idx:02d}_{ts_tag}.wav"
        )
        try:
            export_wav(
                chunk.set_frame_rate(CONFIG["SAMPLE_RATE"]).set_channels(CONFIG["CHANNELS"]),
                out_path,
            )
        except Exception as e:
            logger.warning(f"[export] {out_path} failed: {e}")
            summary.total_chunks_rejected += 1
            continue

        summary.total_chunks_kept += 1
        summary.per_class_count[folder] = summary.per_class_count.get(folder, 0) + 1
        summary.per_class_duration[folder] = (
            summary.per_class_duration.get(folder, 0.0) + (end - start)
        )

        results.append(ChunkResult(
            video_id=record.video_id,
            segment_idx=seg_idx,
            source_offset_sec=segment.offset_sec,
            start_sec=start,
            end_sec=end,
            label=label,
            confidence=conf,
            lfn_ratio=ratio,
            out_path=out_path,
        ))

    return results


def process_video(record: VideoRecord,
                  tagger: YamnetTagger,
                  summary: Summary,
                  logger: logging.Logger) -> List[ChunkResult]:
    results: List[ChunkResult] = []
    for segment in record.segments:
        try:
            results.extend(_process_segment(record, segment, tagger, summary, logger))
        except Exception as e:
            logger.warning(
                f"[process] {record.video_id} seg{segment.segment_idx:02d} crashed: {e}"
            )
    return results


# =============================================================================
# SUMMARY / CLEANUP
# =============================================================================

def write_summary(summary: Summary, logger: logging.Logger) -> None:
    CONFIG["SUMMARY_CSV"].parent.mkdir(parents=True, exist_ok=True)

    with open(CONFIG["SUMMARY_CSV"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "num_chunks", "total_seconds", "total_minutes"])
        for cls in sorted(summary.per_class_count.keys()):
            sec = summary.per_class_duration.get(cls, 0.0)
            writer.writerow([cls,
                             summary.per_class_count[cls],
                             f"{sec:.1f}",
                             f"{sec/60:.2f}"])

    with open(CONFIG["SUMMARY_JSON"], "w", encoding="utf-8") as f:
        json.dump({
            "per_class_duration_sec": summary.per_class_duration,
            "per_class_count": summary.per_class_count,
            "total_videos_scraped": summary.total_videos_scraped,
            "total_videos_failed": summary.total_videos_failed,
            "total_segments_downloaded": summary.total_segments_downloaded,
            "total_chunks_kept": summary.total_chunks_kept,
            "total_chunks_rejected": summary.total_chunks_rejected,
            "config": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in CONFIG.items()},
        }, f, indent=2)

    logger.info(f"[summary] wrote {CONFIG['SUMMARY_CSV']} and {CONFIG['SUMMARY_JSON']}")


def cleanup_workdir(logger: logging.Logger) -> None:
    try:
        shutil.rmtree(CONFIG["WORK_DIR"], ignore_errors=True)
        logger.info("[cleanup] work dir removed")
    except Exception as e:
        logger.warning(f"[cleanup] could not remove work dir: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    CONFIG["DATASET_ROOT"].mkdir(parents=True, exist_ok=True)
    logger = setup_logging(CONFIG["LOG_FILE"])
    logger.info("=== LFN dataset pipeline started ===")

    summary = Summary()

    try:
        records = scrape_all(logger)
    except KeyboardInterrupt:
        logger.warning("interrupted during scraping")
        return 130
    except Exception as e:
        logger.error(f"fatal scraping error: {e}\n{traceback.format_exc()}")
        return 1

    summary.total_videos_scraped = sum(1 for r in records if r.segments)
    summary.total_videos_failed = sum(1 for r in records if r.error)
    summary.total_segments_downloaded = sum(len(r.segments) for r in records)

    try:
        tagger = YamnetTagger(logger)
    except Exception as e:
        logger.error(f"could not initialize YAMNet: {e}")
        return 2

    for rec in records:
        if not rec.segments:
            continue
        try:
            process_video(rec, tagger, summary, logger)
        except Exception as e:
            logger.warning(f"[process] {rec.video_id} crashed: {e}")
        finally:
            for segment in rec.segments:
                try:
                    if segment.wav_path and segment.wav_path.exists():
                        segment.wav_path.unlink()
                except OSError:
                    pass

    write_summary(summary, logger)
    # Only wipe the work dir on a clean finish AND when resume isn't desired.
    # When RESUME_FROM_WORKDIR is on, leftover WAVs are intentional — they
    # let the next run skip already-downloaded segments.
    if not CONFIG.get("RESUME_FROM_WORKDIR"):
        cleanup_workdir(logger)
    else:
        logger.info(
            f"[cleanup] work dir preserved at {CONFIG['WORK_DIR']} "
            f"(RESUME_FROM_WORKDIR is on)"
        )

    logger.info(
        f"=== done. kept {summary.total_chunks_kept} chunks, "
        f"rejected {summary.total_chunks_rejected}, "
        f"videos ok/failed: {summary.total_videos_scraped}/{summary.total_videos_failed}, "
        f"segments downloaded: {summary.total_segments_downloaded} ==="
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())