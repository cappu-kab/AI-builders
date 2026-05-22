#!/usr/bin/env python3
"""
ANC Phase 1 — Real-time Noise Cancellation + Visualization
===========================================================
Target hardware : Jetson Nano
Microphone      : INMP441  (I2S MEMS digital mic  → ALSA capture device)
Speaker         : MAX98357A (I2S DAC/Amp           → ALSA playback device)

Thread architecture
-------------------

  [PortAudio / ALSA hardware]
         │                  ▲
   _input_callback    _output_callback
         │                  │
         ▼                  │
      _input_q          _output_q          ← bounded queues (QUEUE_MAXSIZE)
         │                  ▲
         └──────────────────┘
           ANC-Inference thread
                   │
                   │  _push_vis()          ← non-blocking put_nowait — NEVER stalls
                   ▼
                _vis_q                     ← shallow queue (VIS_QUEUE_MAXSIZE = 3)
                   │
            Qt main thread
        QTimer @ VIS_FPS (30 Hz)
         → drain vis_q (keep newest)
         → .setData() on 4 pyqtgraph curves

Key guarantee: the audio callbacks and inference thread NEVER wait on the
vis_q.  If the queue is full, the oldest frame is evicted and the fresh
frame is inserted.  The GUI may skip frames; audio never does.

Usage:
    python main_hardware.py                 # GUI + mock model  (default / safest)
    python main_hardware.py --no-mock       # GUI + real ONNX checkpoint
    python main_hardware.py --no-gui        # headless audio only (SSH / no display)
    python main_hardware.py --list-devices  # print ALSA indices and exit
"""
from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from configs.config import (
    INPUT_DEVICE_INDEX,
    OUTPUT_DEVICE_INDEX,
    SAMPLE_RATE,
    CHUNK_SIZE,
    NUM_CHANNELS,
    DTYPE,
    PHASE_INVERSION_FACTOR,
    MAX_CONSECUTIVE_ERRORS,
    OVERFLOW_LOG_THROTTLE_S,
    QUEUE_MAXSIZE,
    HEALTH_CHECK_INTERVAL_S,
    VIS_QUEUE_MAXSIZE,
    WARMUP_CHUNKS,
    # ── Residual VAD noise-gate + alignment (mirror file_simulator.py v6d) ────
    NOISE_GATE_ENABLE,
    NOISE_GATE_RATIO_LOW,
    NOISE_GATE_RATIO_HIGH,
    NOISE_GATE_ATTENUATION_DB,
    NOISE_GATE_ATTACK_MS,
    NOISE_GATE_RATIO_SMOOTH_MS,
)
from model_runtime import ModelRuntime
from utils.audio_utils import clip_output
from utils.logger_setup import get_logger

log = get_logger("anc_main")

# ── Module-level shared state ─────────────────────────────────────────────────
# Written/read from designated threads only (see architecture diagram above).
_stop_event = threading.Event()
_input_q:  queue.Queue[np.ndarray] = queue.Queue(maxsize=QUEUE_MAXSIZE)
_output_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=QUEUE_MAXSIZE)

_consecutive_errors   = 0     # written only by inference thread
_last_overflow_log_ts = 0.0   # written only by input callback


# ══════════════════════════════════════════════════════════════════════════════
# Hardware-Alignment & Noise-Gate Derived Constants
# Mirror the values computed at startup in file_simulator.py so the hardware
# pipeline applies the EXACT same per-chunk DSP as the validated v6d simulator
# (CHUNK=256, 72 ms attack, -15 dB floor, 30/45 % ratio band).
# ══════════════════════════════════════════════════════════════════════════════

# Fixed acoustic propagation delay — INMP441 / MAX98357A on a solid chassis.
# Represents the speaker→mic air-gap at 343 m/s: 1 smp / 16 000 Hz ≈ 21 mm.
FIXED_DELAY = 1  # samples — physical chassis geometry

# 1-chunk look-ahead offset.  model_runtime._infer_crn_stateful extracts
# enhanced[-2*CHUNK:-CHUNK] instead of enhanced[-CHUNK:], because the
# rightmost CHUNK samples of the iSTFT reconstruction are distorted by
# torch.stft's center=True reflect-pad.  The returned noise_est therefore
# corresponds to the chunk submitted on the PREVIOUS call, so the raw
# reference path AND the anti-noise speaker path must each be delayed by
# CHUNK_SIZE + 1 samples to stay time-aligned with noise_est on each call.
_LOOKAHEAD_OFFSET  = CHUNK_SIZE
_TOTAL_ALIGN_DELAY = _LOOKAHEAD_OFFSET + FIXED_DELAY   # 257 samples

# Noise-gate floor gain (10**(dB/20) precomputed once).
_GATE_FLOOR_GAIN = float(10.0 ** (NOISE_GATE_ATTENUATION_DB / 20.0))
assert NOISE_GATE_RATIO_LOW < NOISE_GATE_RATIO_HIGH, (
    "NOISE_GATE_RATIO_LOW must be < NOISE_GATE_RATIO_HIGH"
)
assert NOISE_GATE_ATTACK_MS       > 0.0, "NOISE_GATE_ATTACK_MS must be > 0"
assert NOISE_GATE_RATIO_SMOOTH_MS > 0.0, "NOISE_GATE_RATIO_SMOOTH_MS must be > 0"

# Per-chunk IIR smoothing coefficients derived from wall-clock time constants:
#     α = 1 − exp(−chunk_period_ms / τ_ms)
# Both the gate's attack (closing) and the ratio low-pass therefore behave the
# same in wall-clock time regardless of CHUNK_SIZE.
_CHUNK_PERIOD_MS         = 1000.0 * CHUNK_SIZE / SAMPLE_RATE
_GATE_ATTACK_ALPHA       = float(1.0 - np.exp(-_CHUNK_PERIOD_MS / NOISE_GATE_ATTACK_MS))
_GATE_RATIO_SMOOTH_ALPHA = float(1.0 - np.exp(-_CHUNK_PERIOD_MS / NOISE_GATE_RATIO_SMOOTH_MS))


def _noise_gate_target_gain_from_ratio(ratio: float) -> float:
    """Map a (smoothed) clean/raw RMS ratio to a gate gain in [_GATE_FLOOR_GAIN, 1.0].

    ratio <= NOISE_GATE_RATIO_LOW  -> full attenuation (gate closed)
    ratio >= NOISE_GATE_RATIO_HIGH -> 1.0 (gate open, signal untouched)
    in between                      -> linear interpolation in dB.
    Returns 1.0 when the gate is disabled (no-op).
    """
    if not NOISE_GATE_ENABLE:
        return 1.0
    if ratio <= NOISE_GATE_RATIO_LOW:
        return _GATE_FLOOR_GAIN
    if ratio >= NOISE_GATE_RATIO_HIGH:
        return 1.0
    t  = (ratio - NOISE_GATE_RATIO_LOW) / (NOISE_GATE_RATIO_HIGH - NOISE_GATE_RATIO_LOW)
    db = NOISE_GATE_ATTENUATION_DB * (1.0 - t)
    return float(10.0 ** (db / 20.0))


# ══════════════════════════════════════════════════════════════════════════════
# PortAudio Callbacks  (run in PortAudio's high-priority audio thread)
# Rule: no blocking calls, no Python exceptions, minimal work.
# ══════════════════════════════════════════════════════════════════════════════

def _input_callback(
    indata:  np.ndarray,   # shape (CHUNK_SIZE, NUM_CHANNELS) — PortAudio-owned buffer
    frames:  int,
    time_info,
    status:  sd.CallbackFlags,
) -> None:
    """Copy one mic block into _input_q.  Evict oldest frame on overflow."""
    global _last_overflow_log_ts

    if status.input_overflow:
        now = time.monotonic()
        if now - _last_overflow_log_ts >= OVERFLOW_LOG_THROTTLE_S:
            log.warning(
                "INPUT OVERFLOW — inference is too slow for CHUNK_SIZE=%d. "
                "Consider raising CHUNK_SIZE in config.py.", CHUNK_SIZE
            )
            _last_overflow_log_ts = now

    chunk = indata[:, 0].copy()   # must copy: PortAudio recycles indata immediately

    if _input_q.full():
        try:
            _input_q.get_nowait()
        except queue.Empty:
            pass

    try:
        _input_q.put_nowait(chunk)
    except queue.Full:
        pass   # race so rare it is safe to drop


def _output_callback(
    outdata: np.ndarray,   # shape (CHUNK_SIZE, NUM_CHANNELS) — must be filled each call
    frames:  int,
    time_info,
    status:  sd.CallbackFlags,
) -> None:
    """Write one anti-noise block to the speaker.  Output silence on underflow."""
    if status.output_underflow:
        log.debug("OUTPUT UNDERFLOW — writing silence to protect speaker.")
        outdata[:] = 0.0
        return

    try:
        outdata[:, 0] = _output_q.get_nowait()
    except queue.Empty:
        outdata[:] = 0.0   # inference not ready yet — silent gap


# ══════════════════════════════════════════════════════════════════════════════
# Visualization Push Helper
# ══════════════════════════════════════════════════════════════════════════════

def _push_vis(vis_q: queue.Queue, raw, noise, anti, clean) -> None:
    """
    Push a VisFrame snapshot to the GUI queue without ever blocking.

    If the queue is full (Qt is rendering slower than inference), evict the
    oldest (stale) frame and insert the new one.  Dropping a display frame
    is always preferable to stalling the inference thread even for 1 µs.
    """
    from visualizer import VisFrame   # deferred import — not needed in headless mode

    frame = VisFrame(
        raw=raw.copy(),
        noise=noise.copy(),
        anti=anti.copy(),
        clean=clean.copy(),
    )
    try:
        vis_q.put_nowait(frame)
    except queue.Full:
        try:
            vis_q.get_nowait()        # evict oldest stale frame
        except queue.Empty:
            pass
        try:
            vis_q.put_nowait(frame)
        except queue.Full:
            pass   # extremely rare double-race; accept the loss


# ══════════════════════════════════════════════════════════════════════════════
# Inference Worker Thread
# ══════════════════════════════════════════════════════════════════════════════

def _inference_loop(model: ModelRuntime, vis_q: Optional[queue.Queue] = None) -> None:
    """
    Core DSP loop — runs in ANC-Inference thread.

    Hardware port of file_simulator.py's _sim_streamer_worker, adapted for the
    streaming I2S path:  PortAudio → _input_q → here → _output_q → PortAudio.

    Per-chunk pipeline (identical DSP to the validated v6d simulator):
        1.  Block-fetch one chunk from _input_q (200 ms timeout).
        2.  Model inference → noise_estimate  (always — keeps RNN/STFT contexts
            populated so the warmup mute can be released cleanly).
        3.  Warmup mute — zero noise_estimate for the first WARMUP_CHUNKS so the
            speaker stays silent while the model context buffer fills.
        4.  Raw FIFO (_TOTAL_ALIGN_DELAY = 257 smp) → delayed_raw — the raw
            sample that lines up in time with noise_estimate after the CRN's
            1-chunk look-ahead extraction.
        5.  clean_proxy = clip(delayed_raw − noise_estimate, ±1).
        6.  Residual VAD noise gate — ratio = rms(clean) / rms(delayed_raw),
            IIR-smoothed, mapped to a [-15 dB, 0 dB] gain envelope (instant
            release, 72 ms wall-clock smoothed attack).
        7.  anti_noise = clip(noise_estimate × PHASE_INVERSION_FACTOR × gate_gain).
        8.  Anti-noise FIFO (_TOTAL_ALIGN_DELAY = 257 smp) → delayed_anti, so
            the MAX98357A emits each cancellation block at the same physical
            instant the model intended.
        9.  Enqueue delayed_anti to _output_q (PortAudio output callback).
       10.  Push visualization snapshot (raw, noise_est, −noise_est, clean_gated).

    State between chunks: _raw_delay_buf, _anti_delay_buf, _ratio_smooth,
    _gate_gain_smooth, _warmup_counter.
    """
    global _consecutive_errors
    log.info("Inference thread started  |  backend='%s'", model.backend)

    # ── Alignment FIFOs (mirror file_simulator._delay_buf) ────────────────────
    # Raw FIFO  : provides time-aligned raw for the gate's RMS ratio detector.
    # Anti FIFO : delays the speaker output to match the model's look-ahead, so
    #             the anti-noise block emitted during chunk N lines up in time
    #             with the noise that noise_estimate actually corresponds to.
    _raw_delay_buf  = np.zeros(_TOTAL_ALIGN_DELAY, dtype=np.float32)
    _anti_delay_buf = np.zeros(_TOTAL_ALIGN_DELAY, dtype=np.float32)

    # Warmup counter — speaker muted while the model's context buffer fills.
    _warmup_counter: int = 0

    # Noise-gate envelope state.  Both start "open" (1.0) so the warmup-period
    # passthrough isn't accidentally suppressed.
    _ratio_smooth     = 1.0
    _gate_gain_smooth = 1.0

    if NOISE_GATE_ENABLE:
        log.info(
            "Noise gate: on  | ratio [%.2f, %.2f]  | floor %.1f dB"
            "  | attack %.0f ms (α=%.3f)  | ratio_smooth %.0f ms (α=%.3f)",
            NOISE_GATE_RATIO_LOW, NOISE_GATE_RATIO_HIGH,
            NOISE_GATE_ATTENUATION_DB,
            NOISE_GATE_ATTACK_MS,       _GATE_ATTACK_ALPHA,
            NOISE_GATE_RATIO_SMOOTH_MS, _GATE_RATIO_SMOOTH_ALPHA,
        )
    else:
        log.info("Noise gate: off")
    log.info(
        "Alignment FIFO: %d smp (lookahead %d + phys %d)  | warmup %d chunks",
        _TOTAL_ALIGN_DELAY, _LOOKAHEAD_OFFSET, FIXED_DELAY, WARMUP_CHUNKS,
    )

    while not _stop_event.is_set():

        # ── 1. Fetch ──────────────────────────────────────────────────────────
        try:
            chunk: np.ndarray = _input_q.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            raw = chunk.astype(np.float32, copy=False)

            # ── 2. Model inference ────────────────────────────────────────────
            noise_estimate = model.predict(raw)
            if noise_estimate.shape != (CHUNK_SIZE,) or noise_estimate.dtype != np.float32:
                raise RuntimeError(
                    f"model.predict returned shape={noise_estimate.shape} "
                    f"dtype={noise_estimate.dtype}; expected ({CHUNK_SIZE},) float32"
                )

            _warmup_counter += 1

            # ── 3. Warmup mute ────────────────────────────────────────────────
            if _warmup_counter <= WARMUP_CHUNKS:
                noise_estimate = np.zeros(CHUNK_SIZE, dtype=np.float32)

            # ── 4. Raw FIFO → delayed_raw ─────────────────────────────────────
            # delayed_raw[n] corresponds to the same physical instant as
            # noise_estimate[n] (CRN look-ahead 256 smp + 1 smp prop delay).
            _ext_raw       = np.concatenate([_raw_delay_buf, raw])
            delayed_raw    = _ext_raw[:CHUNK_SIZE].copy()
            _raw_delay_buf = _ext_raw[CHUNK_SIZE:].copy()      # always 257 smp

            # ── 5. Clean proxy (mirror simulator's delayed_raw - noise_est) ──
            clean_proxy = np.clip(
                delayed_raw - noise_estimate, -1.0, 1.0
            ).astype(np.float32)

            # ── 6. Residual VAD noise gate ────────────────────────────────────
            # Detection: ratio of clean to raw RMS.  Small ratio ⇒ model
            # cancelled almost everything ⇒ noise-only chunk ⇒ gate closes.
            # Envelope: instant release (open immediately on speech onset),
            # smoothed attack (close gradually so brief noise lulls don't pump).
            # Skipped during warmup because clean_proxy == delayed_raw and the
            # ratio would falsely register as 1.0.
            if NOISE_GATE_ENABLE and _warmup_counter > WARMUP_CHUNKS:
                _rms_clean    = float(np.sqrt(np.mean(clean_proxy.astype(np.float64) ** 2)))
                _rms_raw      = float(np.sqrt(np.mean(delayed_raw.astype(np.float64) ** 2)))
                _ratio_raw    = _rms_clean / (_rms_raw + 1e-9)
                _ratio_smooth = (
                    _GATE_RATIO_SMOOTH_ALPHA * _ratio_raw
                    + (1.0 - _GATE_RATIO_SMOOTH_ALPHA) * _ratio_smooth
                )
                _gain_tgt = _noise_gate_target_gain_from_ratio(_ratio_smooth)
                if _gain_tgt >= _gate_gain_smooth:
                    _gate_gain_smooth = _gain_tgt              # instant release
                else:
                    _gate_gain_smooth = (                      # smoothed attack
                        _GATE_ATTACK_ALPHA * _gain_tgt
                        + (1.0 - _GATE_ATTACK_ALPHA) * _gate_gain_smooth
                    )
            else:
                _gate_gain_smooth = 1.0

            # ── 7. Anti-noise (phase invert → gate → DAC-safe clip) ───────────
            anti_noise = clip_output(
                (noise_estimate * PHASE_INVERSION_FACTOR * _gate_gain_smooth)
                .astype(np.float32)
            )

            # ── 8. Anti-noise FIFO → delayed_anti ─────────────────────────────
            # Mirror of the simulator's raw FIFO, applied to the speaker path
            # so the MAX98357A emits each cancellation block at the same
            # physical instant the model intended.  Physical mic↔speaker
            # acoustic delay is calibrated separately (chassis geometry).
            _ext_anti       = np.concatenate([_anti_delay_buf, anti_noise])
            delayed_anti    = _ext_anti[:CHUNK_SIZE].copy()
            _anti_delay_buf = _ext_anti[CHUNK_SIZE:].copy()    # always 257 smp

            # ── 9. Enqueue for speaker ────────────────────────────────────────
            _enqueue_output(delayed_anti)

            # ── 10. Visualization (fire-and-forget) ───────────────────────────
            # Match file_simulator's frame composition:
            #   Plot 1: raw          (positive noise reference)
            #   Plot 2: noise_est    (model output, same polarity as raw)
            #   Plot 3: -noise_est   (180° inverted — what the speaker emits)
            #   Plot 4: clean        (cancellation residual after gate)
            if vis_q is not None:
                clean_gated = (clean_proxy * _gate_gain_smooth).astype(np.float32)
                _push_vis(
                    vis_q,
                    raw, noise_estimate, -noise_estimate, clean_gated,
                )

            _consecutive_errors = 0

        except Exception as exc:
            _consecutive_errors += 1
            log.error(
                "Inference error #%d/%d: %s",
                _consecutive_errors, MAX_CONSECUTIVE_ERRORS, exc,
                exc_info=True,
            )
            _enqueue_output(np.zeros(CHUNK_SIZE, dtype=np.float32))

            if _consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.critical(
                    "Exceeded %d consecutive errors — requesting shutdown.",
                    MAX_CONSECUTIVE_ERRORS,
                )
                _stop_event.set()

    log.info("Inference thread exiting.")


def _enqueue_output(chunk: np.ndarray) -> None:
    """Deliver anti-noise to the output callback queue, evicting stale data if full."""
    if _output_q.full():
        try:
            _output_q.get_nowait()
        except queue.Empty:
            pass
    try:
        _output_q.put_nowait(chunk)
    except queue.Full:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Device Validation
# ══════════════════════════════════════════════════════════════════════════════

def _validate_devices() -> None:
    """
    Verify configured device indices exist and have the required channel count.
    Raises RuntimeError with a diagnostic message on any mismatch.
    """
    try:
        all_devs = sd.query_devices()
    except sd.PortAudioError as exc:
        raise RuntimeError(f"PortAudio initialisation failed: {exc}") from exc

    def _check(idx, direction: str, min_ch: int) -> dict:
        if isinstance(idx, int):
            if not (0 <= idx < len(all_devs)):
                raise RuntimeError(
                    f"{direction} device index {idx} is out of range "
                    f"(valid: 0–{len(all_devs) - 1}).  "
                    f"Run 'python main_hardware.py --list-devices' to see options."
                )
            dev = all_devs[idx]
        else:
            dev = sd.query_devices(idx)   # ALSA string name

        ch_key = f"max_{direction.lower()}_channels"
        if dev[ch_key] < min_ch:
            raise RuntimeError(
                f"{direction} device [{idx}] '{dev['name']}' has "
                f"{dev[ch_key]} {direction.lower()} channel(s) — need {min_ch}."
            )
        return dev

    in_dev  = _check(INPUT_DEVICE_INDEX,  "Input",  NUM_CHANNELS)
    out_dev = _check(OUTPUT_DEVICE_INDEX, "Output", NUM_CHANNELS)
    log.info("Input  device [%s]: %s", INPUT_DEVICE_INDEX,  in_dev["name"])
    log.info("Output device [%s]: %s", OUTPUT_DEVICE_INDEX, out_dev["name"])


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Worker  (runs in ANC-Pipeline background thread)
# ══════════════════════════════════════════════════════════════════════════════

def _pipeline_worker(model: ModelRuntime, vis_q: Optional[queue.Queue] = None) -> None:
    """
    Opens the ALSA input/output streams and manages the inference thread.
    Designed to run in a daemon background thread so the Qt event loop can
    own the main thread.

    On any fatal error, sets _stop_event so the GUI closes cleanly.
    """
    try:
        _validate_devices()
    except RuntimeError as exc:
        log.critical("Device validation failed: %s", exc)
        _stop_event.set()
        return

    stream_cfg = dict(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        channels=NUM_CHANNELS,
        dtype=DTYPE,
        latency="low",
    )

    inf_thread = threading.Thread(
        target=_inference_loop,
        args=(model, vis_q),
        name="ANC-Inference",
        daemon=True,
    )

    log.info(
        "Opening audio streams — SR=%d Hz | chunk=%d samples (%.1f ms)",
        SAMPLE_RATE, CHUNK_SIZE, CHUNK_SIZE / SAMPLE_RATE * 1000,
    )

    try:
        with (
            sd.InputStream(
                device=INPUT_DEVICE_INDEX,
                callback=_input_callback,
                **stream_cfg,
            ) as in_stream,
            sd.OutputStream(
                device=OUTPUT_DEVICE_INDEX,
                callback=_output_callback,
                **stream_cfg,
            ) as out_stream,
        ):
            log.info("=" * 60)
            log.info("ANC PIPELINE RUNNING.  Close the dashboard window to stop.")
            log.info("=" * 60)

            inf_thread.start()

            while not _stop_event.is_set():
                time.sleep(HEALTH_CHECK_INTERVAL_S)
                if not in_stream.active:
                    raise RuntimeError(
                        "Input stream (INMP441) became inactive — "
                        "check I2S wiring and ALSA device tree."
                    )
                if not out_stream.active:
                    raise RuntimeError(
                        "Output stream (MAX98357A) became inactive — "
                        "check I2S wiring and ALSA device tree."
                    )

    except sd.PortAudioError as exc:
        log.critical("Fatal PortAudio error: %s", exc, exc_info=True)
    except RuntimeError as exc:
        log.critical("Fatal stream error: %s", exc)
    except Exception as exc:
        log.critical("Unexpected pipeline error: %s", exc, exc_info=True)
    finally:
        _stop_event.set()
        if inf_thread.is_alive():
            inf_thread.join(timeout=2.0)
        log.info("Pipeline worker exiting.")


# ══════════════════════════════════════════════════════════════════════════════
# Headless Mode  (--no-gui or PyQtGraph unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def _run_headless(model: ModelRuntime) -> None:
    """
    Run the audio pipeline in the main thread with no Qt dependency.
    Useful over SSH or when the Jetson has no display attached.
    """
    log.info("Running in HEADLESS mode — no visualization.")

    def _handler(signum, frame):
        log.info("Signal %d received — shutting down.", signum)
        _stop_event.set()

    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)

    _pipeline_worker(model, vis_q=None)   # blocks here until _stop_event


# ══════════════════════════════════════════════════════════════════════════════
# GUI Mode  (default)
# ══════════════════════════════════════════════════════════════════════════════

def _run_gui(model: ModelRuntime) -> int:
    """
    Initialize the Qt application, show the dashboard, and run the event loop.

    Thread ownership:
        Main thread  →  Qt event loop + QTimer (plot updates)
        ANC-Pipeline →  sounddevice streams + health monitor
        ANC-Inference → AI model inference

    Returns the Qt exit code (passed to sys.exit).
    """
    from pyqtgraph.Qt import QtWidgets
    from visualizer import ANCDashboard

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ANC Phase 1 Dashboard")

    vis_q = queue.Queue(maxsize=VIS_QUEUE_MAXSIZE)

    dashboard = ANCDashboard(vis_q=vis_q, stop_event=_stop_event)
    dashboard.show()

    pipeline_thread = threading.Thread(
        target=_pipeline_worker,
        args=(model, vis_q),
        name="ANC-Pipeline",
        daemon=True,
    )
    pipeline_thread.start()

    # OS signal handlers — must be registered in the main thread.
    # They call app.quit() which exits exec_() cleanly.
    def _qt_signal_handler(signum, frame):
        log.info("Signal %d received — closing application.", signum)
        _stop_event.set()
        app.quit()

    signal.signal(signal.SIGINT,  _qt_signal_handler)
    signal.signal(signal.SIGTERM, _qt_signal_handler)

    log.info("Qt event loop started.")
    exit_code = app.exec()   # exec() works on both PyQt5 and PyQt6 (exec_() removed in PyQt6)

    # ── Graceful teardown ──────────────────────────────────────────────────────
    _stop_event.set()
    dashboard.stop_timers()

    log.info("Qt event loop exited (code=%d) — waiting for pipeline thread...", exit_code)
    pipeline_thread.join(timeout=3.0)
    if pipeline_thread.is_alive():
        log.warning("Pipeline thread did not exit within 3 s — proceeding anyway.")

    log.info("Shutdown complete.")
    return exit_code


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _list_devices() -> None:
    """Print all PortAudio/ALSA devices with their numeric indices."""
    print("\n" + "=" * 70)
    print("Available audio devices (use these indices in configs/config.py):")
    print("=" * 70)
    print(sd.query_devices())
    print("=" * 70)
    try:
        print(f"\nDefault input  device: [{sd.default.device[0]}] "
              f"{sd.query_devices(sd.default.device[0])['name']}")
        print(f"Default output device: [{sd.default.device[1]}] "
              f"{sd.query_devices(sd.default.device[1])['name']}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ANC Phase 1 — Real-time noise cancellation on Jetson Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-mock",
        action="store_true",
        help="Load real ONNX/TorchScript checkpoint (MODEL_PATH must exist).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Headless mode — disable visualization (useful over SSH).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print detected ALSA/PortAudio device indices and exit.",
    )
    args = parser.parse_args()

    if args.list_devices:
        _list_devices()
        sys.exit(0)

    if args.no_mock:
        import configs.config as _cfg
        _cfg.USE_MOCK_MODEL = False

    log.info("=" * 60)
    log.info("ANC Phase 1 — Jetson Nano Real-time Noise Cancellation")
    log.info("SR: %d Hz  |  Chunk: %d samples (%.1f ms)  |  Band: <200 Hz",
             SAMPLE_RATE, CHUNK_SIZE, CHUNK_SIZE / SAMPLE_RATE * 1000)
    log.info("=" * 60)

    model = ModelRuntime()
    log.info("Model runtime ready  |  backend='%s'", model.backend)

    if args.no_gui:
        _run_headless(model)
        sys.exit(0)

    # Default: GUI mode — fall back to headless if PyQtGraph is absent
    try:
        exit_code = _run_gui(model)
        sys.exit(exit_code)
    except ImportError as exc:
        log.warning("PyQtGraph/PyQt5 not found (%s) — falling back to headless.", exc)
        log.warning("Install with:  pip install pyqtgraph PyQt6   (Windows)")
        log.warning("           or: pip install pyqtgraph && sudo apt-get install python3-pyqt5  (Linux/Jetson)")
        _run_headless(model)
        sys.exit(0)


if __name__ == "__main__":
    main()
