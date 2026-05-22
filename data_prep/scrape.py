"""
Low-Frequency Noise (LFN) Dataset Pipeline
==========================================
Builds a curated dataset of non-speech low-frequency noises (<200 Hz) from YouTube
for training LFN cancellation models.

Pipeline stages:
    1. Scrape YouTube with yt-dlp (rate-limited, audio-only, 16 kHz mono).
       - Short videos (<35,000 s): download the first CLIP_SECONDS.
       - Long videos (>=35,000 s): download a CLIP_SECONDS slice at every hour.
    2. Tag audio segments with YAMNet (positive + negative class filtering).
    3. Spectrally validate that energy is concentrated below 200 Hz.
    4. Export 3-5 s WAV chunks to:
         /dataset/<class>/<video_id>_offset_<offset_sec>_<chunk_start>-<chunk_end>.wav
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
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
    # Imported only for static analyzers (Pylance / mypy); no runtime cost.
    import numpy as np  # noqa: F401

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

CONFIG = {
    # ---- Scraping ----
    "SEARCH_QUERIES": [
        "heavy engine idle",
        "industrial fan hum",
        "AC compressor noise",
        "low frequency machinery drone",
        "low frequency noise",
        "industrial power transformer hum",
        "diesel engine idling vibration",
        "ship engine room ambient",
        "server room low frequency drone",
        "chiller plant noise",
        "ventilation shaft humming",
        "large hydraulic pump operation",
    ],

    "VIDEOS_PER_QUERY": 100,        # ~12 queries x 100 candidates
    "MIN_DELAY_SEC": 5,
    "MAX_DELAY_SEC": 15,
    "MAX_DURATION_SEC": 86400,      # allow up to 24-hour streams (download is capped)
    "MIN_DURATION_SEC": 15,

    # ---- Temporal sampling ----
    "CLIP_SECONDS": 300,                    # length of each downloaded slice
    "LONG_VIDEO_THRESHOLD_SEC": 37000,      # >= this -> multi-segment sampling
    "HOURLY_SAMPLE_INTERVAL_SEC": 3600,     # sample cadence for long videos

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
    "NEGATIVE_CLASSES": ["Speech", "Music", "Bird", "Bird vocalization, bird call, bird song"],
    "POSITIVE_MIN_CONF": 0.30,
    "NEGATIVE_MAX_CONF": 0.10,      # strict: discard if any neg class > 0.1

    # ---- Spectral validation ----
    "LFN_CUTOFF_HZ": 200,
    "LFN_MIN_ENERGY_RATIO": 0.60,   # >=60% of STFT magnitude below 200 Hz
    "N_FFT": 2048,
    "HOP_LENGTH": 512,

    # ---- Output ----
    "DATASET_ROOT": Path("./dataset"),
    "WORK_DIR": Path("./_lfn_work"),
    "SUMMARY_CSV": Path("./dataset/summary.csv"),
    "SUMMARY_JSON": Path("./dataset/summary.json"),
    "MANIFEST_JSONL": Path("./dataset/manifest.jsonl"),  # append-only crash-safe log
    "LOG_FILE": Path("./dataset/pipeline.log"),
    "CHECKPOINT_EVERY_VIDEO": True,  # rewrite summary.csv/json after each video

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
# LAZY IMPORTS (so the config block is readable without heavy deps loaded)
# -----------------------------------------------------------------------------

def _lazy_imports():
    """Import heavy deps once and return them as a dict."""
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
class AudioSlice:
    """One downloaded segment of a video, anchored by its offset into the source."""
    offset_sec: float
    wav_path: Path


@dataclass
class VideoRecord:
    video_id: str
    title: str
    url: str
    duration: float
    query: str
    segments: List[AudioSlice] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ChunkResult:
    video_id: str
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
# STAGE 1 — SCRAPING (with multi-segment sampling)
# =============================================================================

def search_video_urls(query: str, n: int, logger: logging.Logger) -> List[Dict]:
    """Use yt-dlp in search mode to collect up to n video metadata entries."""
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


def compute_offsets(duration: float) -> List[float]:
    """Return the list of time offsets (in seconds) to sample from the video.

    - Short videos (<LONG_VIDEO_THRESHOLD_SEC): [0.0] only.
    - Long videos: one sample at t=0, then every HOURLY_SAMPLE_INTERVAL_SEC,
      stopping at the last offset that still leaves CLIP_SECONDS of content.
    """
    clip = float(CONFIG["CLIP_SECONDS"])
    threshold = float(CONFIG["LONG_VIDEO_THRESHOLD_SEC"])
    interval = float(CONFIG["HOURLY_SAMPLE_INTERVAL_SEC"])

    if duration < threshold:
        return [0.0]

    offsets: List[float] = []
    t = 0.0
    while t + clip <= duration:
        offsets.append(t)
        t += interval
    return offsets if offsets else [0.0]


def _download_single_slice(video_id: str,
                           url: str,
                           offset_sec: float,
                           work_dir: Path,
                           logger: logging.Logger) -> Optional[Path]:
    """Download a single [offset, offset+CLIP_SECONDS] slice of a video as WAV."""
    deps = _lazy_imports()
    out_base = f"{video_id}_offset_{int(offset_sec):06d}"
    out_template = str(work_dir / f"{out_base}.%(ext)s")

    start_t = float(offset_sec)
    end_t = start_t + float(CONFIG["CLIP_SECONDS"])

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "outtmpl": out_template,
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
        ],
        # Only download this specific 5-minute slice. yt-dlp passes the
        # range to ffmpeg so the server transmits only the requested bytes.
        "download_ranges": (
            lambda info_dict, _ydl: [{
                "start_time": start_t,
                "end_time": end_t,
            }]
        ),
        "force_keyframes_at_cuts": True,
        # Extra politeness towards YouTube:
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": 20,
    }

    try:
        with deps["yt_dlp"].YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.warning(f"[slice] {video_id}@{int(offset_sec)}s failed: {e}")
        return None

    wav_path = work_dir / f"{out_base}.wav"
    if not wav_path.exists():
        logger.warning(f"[slice] {video_id}@{int(offset_sec)}s: wav not produced")
        return None

    # Remove any leftover non-wav files for this slice (source container, etc.)
    for leftover in work_dir.glob(f"{out_base}.*"):
        if leftover.suffix.lower() != ".wav":
            try:
                leftover.unlink()
            except OSError:
                pass

    return wav_path


def download_audio(entry: Dict,
                   query: str,
                   work_dir: Path,
                   logger: logging.Logger) -> VideoRecord:
    """Download one or more slices of a video depending on its length."""
    video_id = entry.get("id") or entry.get("url", "unknown")
    url = entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
    title = entry.get("title") or ""
    duration = float(entry.get("duration") or 0.0)

    record = VideoRecord(video_id=video_id, title=title, url=url,
                         duration=duration, query=query)

    # Reject pathological durations (e.g., corrupt metadata or >24 h streams).
    if duration and (duration > CONFIG["MAX_DURATION_SEC"]
                     or duration < CONFIG["MIN_DURATION_SEC"]):
        record.error = f"duration {duration:.0f}s out of bounds"
        return record

    work_dir.mkdir(parents=True, exist_ok=True)
    offsets = compute_offsets(duration)

    mode = "standard" if len(offsets) == 1 else "long-form"
    logger.info(
        f"[plan] {video_id}: {duration:.0f}s, mode={mode}, "
        f"{len(offsets)} slice(s)"
    )

    for i, offset in enumerate(offsets):
        wav_path = _download_single_slice(video_id, url, offset, work_dir, logger)
        if wav_path is not None:
            record.segments.append(AudioSlice(offset_sec=offset, wav_path=wav_path))

        # Polite short delay between consecutive slice downloads of the same
        # long video. Full inter-video delay happens in scrape_all().
        if len(offsets) > 1 and i < len(offsets) - 1:
            time.sleep(random.uniform(CONFIG["MIN_DELAY_SEC"] / 2.0,
                                      CONFIG["MAX_DELAY_SEC"] / 2.0))

    if not record.segments:
        record.error = "no segments downloaded"
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
                    f"({rec.duration:.0f}s, {len(rec.segments)} slice(s))"
                )

            # Rate-limit politely even on failure.
            delay = random.uniform(CONFIG["MIN_DELAY_SEC"], CONFIG["MAX_DELAY_SEC"])
            time.sleep(delay)

    return records


# =============================================================================
# STAGE 2 — YAMNET CLASSIFICATION
# =============================================================================

class YamnetTagger:
    """Thin wrapper around the YAMNet TF-Hub model."""

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
            next(reader, None)   # header
            for row in reader:
                if len(row) >= 3:
                    names.append(row[2])
        return names

    def tag(self, waveform) -> "np.ndarray":
        """Return per-frame scores of shape (frames, classes)."""
        scores, _embeddings, _spec = self.model(waveform)
        return scores.numpy()

    def segment_label(self, scores) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Reduce frame scores to a single label + confidence for the segment.

        Returns (label_or_None, confidence, raw_mean_scores_for_tracked_classes).
        label is None when the segment should be rejected.
        """
        np = self.np
        mean_scores = scores.mean(axis=0)

        tracked = {}
        for name in CONFIG["POSITIVE_CLASSES"] + CONFIG["NEGATIVE_CLASSES"]:
            if name in self.class_names:
                idx = self.class_names.index(name)
                tracked[name] = float(mean_scores[idx])

        # Negative filter: hard reject if any forbidden class fires.
        for neg in CONFIG["NEGATIVE_CLASSES"]:
            if tracked.get(neg, 0.0) > CONFIG["NEGATIVE_MAX_CONF"]:
                return None, 0.0, tracked

        # Positive: take the highest-scoring tracked positive class.
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
    """Fraction of STFT magnitude energy located below CONFIG['LFN_CUTOFF_HZ']."""
    deps = _lazy_imports()
    np = deps["np"]
    librosa = deps["librosa"]

    if len(y) < CONFIG["N_FFT"]:
        # Pad short clips so STFT still works.
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
# STAGE 4 — SEGMENTATION, EXPORT, LOGGING
# =============================================================================

def iter_chunks(total_sec: float,
                chunk_min: float,
                chunk_max: float) -> Iterable[Tuple[float, float]]:
    """Yield (start, end) windows covering the audio with random chunk lengths."""
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


def append_manifest(entry: Dict) -> None:
    """Append one JSON line describing a successfully exported chunk.

    The manifest is crash-safe: flush() + fsync() guarantees each line hits
    disk before returning. If the pipeline is killed mid-run, the manifest
    still reflects every chunk that was fully written.
    """
    path = CONFIG["MANIFEST_JSONL"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _process_slice(record: VideoRecord,
                   slice_: AudioSlice,
                   tagger: YamnetTagger,
                   summary: Summary,
                   logger: logging.Logger) -> List[ChunkResult]:
    """Classify, validate and export chunks from a single downloaded slice."""
    deps = _lazy_imports()
    librosa = deps["librosa"]
    np = deps["np"]
    AudioSegment = deps["AudioSegment"]

    results: List[ChunkResult] = []
    wav_path = slice_.wav_path
    offset_sec = slice_.offset_sec

    if not wav_path.exists():
        return results

    try:
        full_audio = AudioSegment.from_wav(wav_path)
    except Exception as e:
        logger.warning(
            f"[process] {record.video_id}@{int(offset_sec)}s unreadable: {e}"
        )
        return results

    full_sec = len(full_audio) / 1000.0

    for start, end in iter_chunks(full_sec,
                                  CONFIG["CHUNK_MIN_SEC"],
                                  CONFIG["CHUNK_MAX_SEC"]):
        chunk = full_audio[int(start * 1000): int(end * 1000)]

        # pydub -> numpy float32 in [-1, 1]
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

        # ---- YAMNet tagging ----
        try:
            scores = tagger.tag(samples)
        except Exception as e:
            logger.warning(
                f"[yamnet] {record.video_id}@{int(offset_sec)}s+{start:.1f}s failed: {e}"
            )
            summary.total_chunks_rejected += 1
            continue

        label, conf, _tracked = tagger.segment_label(scores)
        if label is None:
            summary.total_chunks_rejected += 1
            continue

        # ---- Spectral validation ----
        ok, ratio = passes_spectral_filter(samples, sr)
        if not ok:
            summary.total_chunks_rejected += 1
            continue

        # ---- Export ----
        folder = CONFIG["CLASS_FOLDER_MAP"].get(label, "other")
        ts_tag = f"{int(start):05d}-{int(end):05d}"
        out_path = (
            CONFIG["DATASET_ROOT"] / folder
            / f"{record.video_id}_offset_{int(offset_sec):06d}_{ts_tag}.wav"
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

        append_manifest({
            "video_id": record.video_id,
            "url": record.url,
            "query": record.query,
            "source_offset_sec": offset_sec,
            "chunk_start_sec": start,
            "chunk_end_sec": end,
            "label": label,
            "confidence": conf,
            "lfn_ratio": ratio,
            "class_folder": folder,
            "out_path": str(out_path),
        })

        results.append(ChunkResult(
            video_id=record.video_id,
            source_offset_sec=offset_sec,
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
    """Process every downloaded slice of a video."""
    results: List[ChunkResult] = []
    for slice_ in record.segments:
        try:
            results.extend(_process_slice(record, slice_, tagger, summary, logger))
        except Exception as e:
            logger.warning(
                f"[process] {record.video_id}@{int(slice_.offset_sec)}s crashed: {e}"
            )
    return results


# =============================================================================
# SUMMARY / CLEANUP
# =============================================================================

def write_summary(summary: Summary, logger: logging.Logger) -> None:
    CONFIG["SUMMARY_CSV"].parent.mkdir(parents=True, exist_ok=True)

    # CSV: per-class breakdown
    with open(CONFIG["SUMMARY_CSV"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "num_chunks", "total_seconds", "total_minutes"])
        for cls in sorted(summary.per_class_count.keys()):
            sec = summary.per_class_duration.get(cls, 0.0)
            writer.writerow([cls,
                             summary.per_class_count[cls],
                             f"{sec:.1f}",
                             f"{sec/60:.2f}"])

    # JSON: full summary
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
# MAIN ORCHESTRATION
# =============================================================================

def main() -> int:
    CONFIG["DATASET_ROOT"].mkdir(parents=True, exist_ok=True)
    logger = setup_logging(CONFIG["LOG_FILE"])
    logger.info("=== LFN dataset pipeline started ===")

    summary = Summary()

    # 1. Scrape
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

    # 2+3+4. Tag, filter, export
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
            # Delete intermediate slice WAVs aggressively to save disk.
            for seg in rec.segments:
                try:
                    if seg.wav_path and seg.wav_path.exists():
                        seg.wav_path.unlink()
                except OSError:
                    pass
            # Checkpoint after every video so an interrupted run still has
            # an accurate summary on disk.
            if CONFIG.get("CHECKPOINT_EVERY_VIDEO"):
                try:
                    write_summary(summary, logger)
                except Exception as e:
                    logger.warning(f"[checkpoint] summary write failed: {e}")

    # 5. Summary + cleanup
    write_summary(summary, logger)
    cleanup_workdir(logger)

    logger.info(
        f"=== done. kept {summary.total_chunks_kept} chunks, "
        f"rejected {summary.total_chunks_rejected}, "
        f"videos ok/failed: {summary.total_videos_scraped}/{summary.total_videos_failed}, "
        f"slices downloaded: {summary.total_segments_downloaded} ==="
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())