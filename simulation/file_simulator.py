#!/usr/bin/env python3
"""
ANC Phase 1 — Hardware-Accurate Streaming Simulator
=====================================================
Strict real-time streaming replica of the Jetson Nano deployment.
Audio is processed one CHUNK_SIZE block at a time; no full-file array is
ever held in memory.

Key hardware-alignment properties:
  CHUNK_SIZE = 256 smp (16 ms @ 16 kHz)   — Jetson low-latency target
  BiLSTM look-ahead N = 1 chunk            — no multi-chunk output delay
  WARMUP_CHUNKS mute                       — speaker silenced while context fills
  Propagation-delay jitter                 — random [MIN, MAX] sample offset per chunk
  Acoustic feedback                        — fraction of anti-noise re-injected into mic
  Real-time summation                      — clean = raw + anti at the current time index
  Streaming WAV output                     — written incrementally, never accumulated

Thread map
----------
  Main thread   Qt event loop + QTimer @ VIS_FPS → 4-panel dashboard
  SimStreamer   Chunk I/O + jitter/feedback + incremental WAV writer
  ANC-Inference model.predict() pipeline → _output_q

Usage
-----
  python file_simulator.py                          # synthetic 10 s, mock model
  python file_simulator.py dirty.wav               # one-pass WAV, mock model
  python file_simulator.py dirty.wav --loop        # loop until window closed
  python file_simulator.py dirty.wav -m crn        # CRN model
  python file_simulator.py dirty.wav -m unet       # AdvancedUNetSE
  python file_simulator.py dirty.wav -m resemble   # Resemble Enhance
  python file_simulator.py dirty.wav -m crn -o out.wav
"""
from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

# ── Ensure project root is importable regardless of working directory ─────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Project imports ───────────────────────────────────────────────────────────
from configs.config import (
    SAMPLE_RATE,
    CHUNK_SIZE,
    VIS_QUEUE_MAXSIZE,
    WARMUP_CHUNKS,
    SIM_PROP_DELAY_MIN,
    SIM_PROP_DELAY_MAX,
    SIM_FEEDBACK_SCALE,
    ANC_XFADE_SAMPLES,
    NOISE_GATE_ENABLE,
    NOISE_GATE_RATIO_LOW,
    NOISE_GATE_RATIO_HIGH,
    NOISE_GATE_ATTENUATION_DB,
    NOISE_GATE_ATTACK_MS,
    NOISE_GATE_RATIO_SMOOTH_MS,
)
from model_runtime import ModelRuntime, _StationaryResampler
from utils.logger_setup import get_logger

import main_hardware as mh

log = get_logger("file_sim")

# ── Constants ─────────────────────────────────────────────────────────────────
_OUTPUT_WAV_PATTERN    = "simulated_output_clean_{model}.wav"
TEST_SIGNAL_DURATION_S = 10.0
_CHUNK_DURATION_S      = CHUNK_SIZE / SAMPLE_RATE

_ALL_MODELS = ("mock", "crn", "unet", "resemble", "resemble_ft", "mossformer_ft")

# Conservative amplitude ceiling for the synthetic engine signal.
# Sum of component amplitudes = 0.38+0.22+0.14+0.10+0.12+0.08 = 1.04;
# noise floor adds ~0.032 ⇒ peak ≤ ~1.07.  Using 1.1 guarantees no clipping.
_SYNTH_PEAK = 1.1

# Fixed acoustic propagation delay — INMP441 / MAX98357A on a solid chassis.
# Represents the speaker→mic air-gap at 343 m/s: 16 smp / 16 000 Hz ≈ 1 mm.
# This is the ONLY delay constant used in the signal path.
FIXED_DELAY = 1  # samples — physical chassis geometry (INMP441/MAX98357A at 343 m/s)

# 1-chunk look-ahead offset.
# model_runtime._infer_crn_stateful extracts enhanced[-2*CHUNK:-CHUNK] instead
# of enhanced[-CHUNK:], because the rightmost CHUNK samples of the iSTFT
# reconstruction are distorted by torch.stft's center=True reflect-pad.  The
# returned noise_est therefore corresponds to the chunk submitted on the
# PREVIOUS call, so the raw reference path must be delayed by CHUNK_SIZE
# samples (plus the physical mic↔speaker propagation delay) to stay
# time-aligned with noise_est on each iteration.
_LOOKAHEAD_OFFSET  = CHUNK_SIZE
_TOTAL_ALIGN_DELAY = _LOOKAHEAD_OFFSET + FIXED_DELAY   # CHUNK_SIZE + 1 samples

# Cross-fade length at 16 ms block boundaries.
# Ramps noise_est's leading edge from the previous block's exit level to the
# current block's natural value, eliminating any hard phase discontinuity at
# the chunk boundary.  With look-ahead extraction (model_runtime stateful CRN)
# the extracted segments are already continuous, so ANC_XFADE_SAMPLES=0 is the
# default and the cross-fade step is skipped entirely.  Set 1..CHUNK_SIZE to
# re-enable a leading-edge ramp.
_XFADE_LEN = int(ANC_XFADE_SAMPLES)
assert 0 <= _XFADE_LEN <= CHUNK_SIZE, (
    f"ANC_XFADE_SAMPLES={_XFADE_LEN} must satisfy 0 <= N <= CHUNK_SIZE={CHUNK_SIZE}"
)
_xfade_ramp = (
    np.linspace(0.0, 1.0, _XFADE_LEN, dtype=np.float32) if _XFADE_LEN > 0 else None
)

# Noise-gate floor gain (10**(dB/20) precomputed once).
_GATE_FLOOR_GAIN = float(10.0 ** (NOISE_GATE_ATTENUATION_DB / 20.0))
assert NOISE_GATE_RATIO_LOW < NOISE_GATE_RATIO_HIGH, (
    "NOISE_GATE_RATIO_LOW must be < NOISE_GATE_RATIO_HIGH"
)
assert NOISE_GATE_ATTACK_MS       > 0.0, "NOISE_GATE_ATTACK_MS must be > 0"
assert NOISE_GATE_RATIO_SMOOTH_MS > 0.0, "NOISE_GATE_RATIO_SMOOTH_MS must be > 0"

# Derive the per-chunk smoothing coefficients from the wall-clock time
# constants.  For the IIR low-pass y = α·target + (1−α)·y_prev driven by a
# unit step, the (1−1/e) settling time is τ = −chunk_period / ln(1−α);
# inverted: α = 1 − exp(−chunk_period_ms / τ_ms).  Both the gate's attack
# (closing) and the ratio low-pass therefore behave the same in wall-clock
# time regardless of CHUNK_SIZE.
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
    # Linear interpolation in dB between floor and 0 dB.
    t  = (ratio - NOISE_GATE_RATIO_LOW) / (NOISE_GATE_RATIO_HIGH - NOISE_GATE_RATIO_LOW)
    db = NOISE_GATE_ATTENUATION_DB * (1.0 - t)
    return float(10.0 ** (db / 20.0))


# ══════════════════════════════════════════════════════════════════════════════
# Streaming Input Sources
# ══════════════════════════════════════════════════════════════════════════════

def _wav_chunk_stream(path: Path) -> Iterator[np.ndarray]:
    """
    Yield CHUNK_SIZE mono float32 chunks from a WAV file one at a time.
    Never loads the full file into memory.

    Files at non-native sample rates are converted on-the-fly via a stateful
    polyphase resampler (_StationaryResampler) so the accum buffer never grows
    beyond 2 × CHUNK_SIZE samples.
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for streaming WAV input: pip install soundfile"
        )

    resampler: Optional[_StationaryResampler] = None
    accum = np.zeros(0, dtype=np.float32)

    with sf.SoundFile(str(path)) as f:
        src_sr = f.samplerate
        nch    = f.channels

        if src_sr != SAMPLE_RATE:
            resampler = _StationaryResampler(src_sr, SAMPLE_RATE)
            # Read enough source samples to produce ≥ 2 × CHUNK_SIZE output samples
            read_size = max(CHUNK_SIZE, int(CHUNK_SIZE * 2 * src_sr / SAMPLE_RATE) + 128)
            log.info(
                "Streaming '%s' — src_sr=%d Hz → %d Hz (streaming resample), ch=%d",
                path.name, src_sr, SAMPLE_RATE, nch,
            )
        else:
            read_size = CHUNK_SIZE
            log.info("Streaming '%s' — SR=%d Hz, ch=%d", path.name, src_sr, nch)

        while True:
            block = f.read(read_size, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            mono = block.mean(axis=1).astype(np.float32)
            if resampler is not None:
                mono = resampler.process(mono)
            accum = np.concatenate([accum, mono])

            while len(accum) >= CHUNK_SIZE:
                yield accum[:CHUNK_SIZE].copy()
                accum = accum[CHUNK_SIZE:]


def _engine_noise_stream(duration_s: float = TEST_SIGNAL_DURATION_S) -> Iterator[np.ndarray]:
    """
    Yield CHUNK_SIZE chunks of synthetic multi-component engine noise.
    Sample values are computed analytically per chunk — no full-signal array.
    """
    rng      = np.random.default_rng(seed=42)
    n_chunks = int(duration_s * SAMPLE_RATE) // CHUNK_SIZE

    for i in range(n_chunks):
        t = (i * CHUNK_SIZE + np.arange(CHUNK_SIZE, dtype=np.float64)) / SAMPLE_RATE
        chunk = (
            0.38 * np.sin(2 * np.pi *  75 * t) +   # engine fundamental
            0.22 * np.sin(2 * np.pi * 150 * t) +   # 2nd harmonic
            0.14 * np.sin(2 * np.pi * 100 * t) +   # secondary rumble
            0.10 * np.sin(2 * np.pi *  60 * t) +   # drivetrain hum
            0.12 * np.sin(2 * np.pi * 320 * t) +   # intake harmonic (above ANC band)
            0.08 * np.sin(2 * np.pi * 480 * t) +   # body resonance (above ANC band)
            0.032 * rng.standard_normal(CHUNK_SIZE)
        ).astype(np.float32)
        yield (chunk / _SYNTH_PEAK * 0.85).astype(np.float32)


def _looping(make_iter):
    """Restart an iterator factory indefinitely when exhausted."""
    while True:
        yield from make_iter()


# ══════════════════════════════════════════════════════════════════════════════
# Streaming Worker: SimStreamer
# ══════════════════════════════════════════════════════════════════════════════

def _sim_streamer_worker(
    chunk_iter: Iterator[np.ndarray],
    model: ModelRuntime,
    output_path: Path,
    vis_q: Optional[queue.Queue],
) -> None:
    """
    Hardware-equivalent streaming loop with STFT group-delay compensation
    and block-boundary cross-fade.

    Pipeline per chunk:
        1.  Real-time pace via deadline sleep.
        2.  model.predict(raw) → noise_est (CRN reflect-pad inverts polarity;
            + in step 5 gives destructive cancellation).
        3.  Cross-fade noise_est's leading _XFADE_LEN samples with the
            previous block's tail to eliminate hard phase discontinuities
            at 16 ms block boundaries.
        4.  FIFO-delay raw by _TOTAL_ALIGN_DELAY = STFT_GROUP_DELAY + FIXED_DELAY
            so that delayed_raw[n] is time-aligned with noise_est[n].
        5.  clean_chunk = clip(delayed_raw + noise_est).
        6.  Write WAV + push visualization.

    State between chunks: _delay_buf (FIFO tail), _prev_noise_tail (xfade),
    _warmup_counter.  No adaptive filters, no feedback, no cross-correlation.
    """
    n_chunks        = 0
    _warmup_counter = 0
    deadline        = time.monotonic()

    # Reference delay FIFO — _TOTAL_ALIGN_DELAY = STFT group delay + physical prop.
    _delay_buf = np.zeros(_TOTAL_ALIGN_DELAY, dtype=np.float32)

    # Cross-fade: trailing samples of previous noise_est, used to smooth
    # the hard amplitude/phase jump at each 16 ms block boundary.
    _prev_noise_tail = np.zeros(_XFADE_LEN, dtype=np.float32)

    # Noise gate envelope: track the smoothed detection ratio AND the
    # smoothed gate gain across chunks.  Both start "open" (1.0) so the
    # warmup-period passthrough isn't accidentally suppressed.
    _ratio_smooth     = 1.0
    _gate_gain_smooth = 1.0

    writer = None
    _fallback: List[np.ndarray] = []
    try:
        import soundfile as sf
        writer = sf.SoundFile(
            str(output_path), mode="w",
            samplerate=SAMPLE_RATE, channels=1, subtype="FLOAT",
        )
        log.info("Streaming output: '%s'", output_path)
    except ImportError:
        log.warning(
            "soundfile not available — output will accumulate in RAM and be "
            "written at end via scipy.  pip install soundfile for true streaming."
        )

    if NOISE_GATE_ENABLE:
        _gate_str = (
            "gate=on (ratio [%.2f, %.2f], floor=%.1f dB,"
            " ratio_smooth=%.0f ms->α=%.3f, attack=%.0f ms->α=%.3f)" % (
                NOISE_GATE_RATIO_LOW, NOISE_GATE_RATIO_HIGH,
                NOISE_GATE_ATTENUATION_DB,
                NOISE_GATE_RATIO_SMOOTH_MS, _GATE_RATIO_SMOOTH_ALPHA,
                NOISE_GATE_ATTACK_MS,        _GATE_ATTACK_ALPHA,
            )
        )
    else:
        _gate_str = "gate=off"
    log.info(
        "SimStreamer started — align_delay=%d smp (lookahead=%d + phys=%d)"
        " | xfade=%d smp | warmup=%d chunks | %s",
        _TOTAL_ALIGN_DELAY, _LOOKAHEAD_OFFSET, FIXED_DELAY,
        _XFADE_LEN, WARMUP_CHUNKS, _gate_str,
    )

    try:
        for raw in chunk_iter:
            if mh._stop_event.is_set():
                break

            raw = raw.astype(np.float32)

            # ── 1. Real-time pacing ───────────────────────────────────────
            deadline += _CHUNK_DURATION_S
            wait_s    = deadline - time.monotonic()
            if wait_s > 0:
                time.sleep(wait_s)

            # ── 2. Model inference → noise estimate ───────────────────────
            noise_est = model.predict(raw)
            if noise_est.shape != (CHUNK_SIZE,) or noise_est.dtype != np.float32:
                raise RuntimeError(
                    f"model.predict returned shape={noise_est.shape} "
                    f"dtype={noise_est.dtype}; expected ({CHUNK_SIZE},) float32"
                )

            _warmup_counter += 1
            if _warmup_counter <= WARMUP_CHUNKS:
                noise_est = np.zeros(CHUNK_SIZE, dtype=np.float32)

            # ── 3. Block-boundary cross-fade (optional) ───────────────────
            # Skipped entirely when _XFADE_LEN == 0 (default with look-ahead
            # extraction, where extracted segments are already continuous).
            if _XFADE_LEN > 0:
                _new_tail              = noise_est[-_XFADE_LEN:].copy()
                noise_est              = noise_est.copy()
                noise_est[:_XFADE_LEN] = (
                    _prev_noise_tail * (1.0 - _xfade_ramp)
                    + noise_est[:_XFADE_LEN] * _xfade_ramp
                )
                _prev_noise_tail = _new_tail

            # ── 4. Phase-aligned reference delay ─────────────────────────
            # Prepend the _TOTAL_ALIGN_DELAY-sample FIFO tail so that
            # delayed_raw[n] corresponds to the same physical instant as
            # noise_est[n] after CRN STFT group delay + physical propagation.
            _extended   = np.concatenate([_delay_buf, raw])
            delayed_raw = _extended[:CHUNK_SIZE].copy()
            _delay_buf  = _extended[CHUNK_SIZE:].copy()   # always _TOTAL_ALIGN_DELAY smp

            # ── 5. Destructive subtraction ────────────────────────────────
            # With look-ahead extraction in _infer_crn_stateful, noise_est is
            # the model's clean residual on the interior of the iSTFT output
            # (no reflect-pad distortion), so noise_est has the natural
            # +true_noise sign and cancellation is subtraction:
            #     clean = raw_at_(t-256) - noise_at_(t-256).
            # The 257-sample delay buffer in _delay_buf aligns delayed_raw
            # with the previous chunk that noise_est corresponds to (256 smp
            # look-ahead + 1 smp physical propagation = 257 smp total).
            clean_chunk = np.clip(
                delayed_raw - noise_est, -1.0, 1.0
            ).astype(np.float32)

            # ── 5b. Residual noise gate ───────────────────────────────────
            # Suppress what's left of the residual when the chunk is noise-only.
            # Detection: ratio of clean to raw RMS (small => model cancelled
            # almost everything => noise-only chunk).  Envelope: instant
            # release (open the gate immediately on speech onset), smoothed
            # attack (close gradually so brief noise lulls don't pump).
            # Skip while warmup is muting noise_est, since clean_chunk would
            # equal delayed_raw and the ratio would falsely register as 1.0.
            if NOISE_GATE_ENABLE and _warmup_counter > WARMUP_CHUNKS:
                _rms_clean   = float(np.sqrt(np.mean(clean_chunk.astype(np.float64) ** 2)))
                _rms_raw     = float(np.sqrt(np.mean(delayed_raw.astype(np.float64) ** 2)))
                _ratio_raw   = _rms_clean / (_rms_raw + 1e-9)
                # Wall-clock IIR low-pass on the detection signal — keeps the
                # threshold-crossing behaviour chunk-size-invariant.
                _ratio_smooth = (
                    _GATE_RATIO_SMOOTH_ALPHA * _ratio_raw
                    + (1.0 - _GATE_RATIO_SMOOTH_ALPHA) * _ratio_smooth
                )
                _gain_tgt = _noise_gate_target_gain_from_ratio(_ratio_smooth)
                if _gain_tgt >= _gate_gain_smooth:
                    _gate_gain_smooth = _gain_tgt           # instant release
                else:
                    _gate_gain_smooth = (                   # smoothed attack
                        _GATE_ATTACK_ALPHA * _gain_tgt
                        + (1.0 - _GATE_ATTACK_ALPHA) * _gate_gain_smooth
                    )
                clean_chunk = (clean_chunk * _gate_gain_smooth).astype(np.float32)

            n_chunks += 1

            # ── 6. WAV write + visualization ──────────────────────────────
            if writer is not None:
                writer.write(clean_chunk)
            else:
                _fallback.append(clean_chunk.copy())

            if vis_q is not None:
                # Plot 1: raw input  (positive noise)
                # Plot 2: noise_est  (positive, same polarity as Plot 1)
                # Plot 3: -noise_est (180° inverted — what the speaker emits)
                # Plot 4: clean_chunk (cancellation residual)
                mh._push_vis(vis_q, raw, noise_est, -noise_est, clean_chunk)

    finally:
        if writer is not None:
            writer.close()
        elif _fallback:
            from scipy.io import wavfile as _wv
            arr = np.concatenate(_fallback)
            _wv.write(
                str(output_path), SAMPLE_RATE,
                np.clip(arr, -1.0, 1.0).astype(np.float32),
            )
            log.info("Fallback output written via scipy.")

        mh._stop_event.set()
        log.info(
            "SimStreamer exiting — %d chunks (%.2f s) → '%s'",
            n_chunks, n_chunks * _CHUNK_DURATION_S, output_path,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Simulation Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def _run_simulation(
    chunk_iter: Iterator[np.ndarray],
    model: ModelRuntime,
    args,
) -> int:
    """
    Wire SimStreamer thread, show ANCDashboard, block on Qt event loop.

    SimStreamer calls model.predict() directly — no separate inference thread.
    """
    from pyqtgraph.Qt import QtWidgets
    from visualizer import ANCDashboard

    model.reset()
    mh._stop_event.clear()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ANC Phase 1 — HW Sim")

    vis_q     = queue.Queue(maxsize=VIS_QUEUE_MAXSIZE)
    dashboard = ANCDashboard(vis_q=vis_q, stop_event=mh._stop_event)

    source_str = Path(args.input).name if args.input else "Synthetic Engine Noise"
    dashboard.setWindowTitle(
        f"ANC HW-Sim  |  {source_str}"
        f"  |  model={model.model_name}"
        f"  |  chunk={CHUNK_SIZE} smp ({CHUNK_SIZE / SAMPLE_RATE * 1000:.0f} ms)"
        f"  |  delay={FIXED_DELAY} smp  |  warmup={WARMUP_CHUNKS} chunks"
    )
    dashboard.show()

    worker_threads = [
        threading.Thread(
            target=_sim_streamer_worker,
            args=(chunk_iter, model, Path(args.output), vis_q),
            name="SimStreamer",
            daemon=True,
        ),
    ]
    for t in worker_threads:
        t.start()

    log.info(
        "HW-Sim: chunk=%d smp (%.0f ms) | fixed_delay=%d smp"
        " | warmup=%d chunks (%.1f s) | model=%s",
        CHUNK_SIZE, CHUNK_SIZE / SAMPLE_RATE * 1000,
        FIXED_DELAY, WARMUP_CHUNKS, WARMUP_CHUNKS * _CHUNK_DURATION_S,
        model.model_name,
    )
    log.info("Close the dashboard window or press Ctrl+C to stop.")

    def _on_signal(signum, _frame):
        log.info("Signal %d received — stopping.", signum)
        mh._stop_event.set()
        app.quit()

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    exit_code = app.exec()

    log.info("Qt event loop exited (code=%d) — tearing down ...", exit_code)
    mh._stop_event.set()
    dashboard.stop_timers()

    for t in worker_threads:
        t.join(timeout=3.0)
        if t.is_alive():
            log.warning("Thread '%s' did not exit within 3 s.", t.name)

    log.info("Simulation complete. Output: '%s'", args.output)
    return exit_code


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def _print_banner(source_label: str, model: ModelRuntime) -> None:
    from model_runtime import _RESAMP_TAIL
    chunk_ms = CHUNK_SIZE / SAMPLE_RATE * 1000
    hist_smp = model._ctx_buf._hist if model._ctx_buf is not None else 0
    ctx_ms   = hist_smp / SAMPLE_RATE * 1000
    log.info("=" * 68)
    log.info("ANC Phase 1 — Hardware-Accurate Streaming Simulator")
    log.info("Source    : %s", source_label)
    log.info("SR        : %d Hz  |  Chunk: %d smp (%.1f ms)", SAMPLE_RATE, CHUNK_SIZE, chunk_ms)
    log.info(
        "Warmup    : %d chunks (%.1f s) — speaker muted during context fill",
        WARMUP_CHUNKS, WARMUP_CHUNKS * _CHUNK_DURATION_S,
    )
    log.info(
        "Jitter    : [%d, %d] smp  |  Feedback: %.4f",
        SIM_PROP_DELAY_MIN, SIM_PROP_DELAY_MAX, SIM_FEEDBACK_SCALE,
    )
    log.info(
        "Model     : %-18s |  Target SR : %d Hz",
        model.model_name, model.target_sample_rate,
    )
    log.info(
        "Context   : %d smp (%.0f ms history + %d smp chunk)",
        hist_smp + CHUNK_SIZE, ctx_ms, CHUNK_SIZE,
    )
    if model._up_resampler is not None:
        log.info(
            "Resamplers: %d Hz -> %d Hz -> %d Hz  (tail=%d smp)",
            SAMPLE_RATE, model.target_sample_rate, SAMPLE_RATE, _RESAMP_TAIL,
        )
    else:
        log.info("Resamplers: none (model native SR matches pipeline)")
    log.info("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ANC Phase 1 — Hardware-Accurate Streaming Simulator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python file_simulator.py                          # synthetic 10 s, mock model
  python file_simulator.py dirty.wav               # one-pass WAV, mock model
  python file_simulator.py dirty.wav --loop        # loop until window closed
  python file_simulator.py dirty.wav -m crn        # CRN model
  python file_simulator.py dirty.wav -m unet       # AdvancedUNetSE
  python file_simulator.py dirty.wav -m resemble   # Resemble Enhance
  python file_simulator.py dirty.wav -m crn -o out.wav

Default output: {_OUTPUT_WAV_PATTERN}
        """,
    )
    parser.add_argument(
        "input", nargs="?", default=None, metavar="INPUT_WAV",
        help="Path to a dirty-noise WAV file.  Omit to stream synthetic engine noise.",
    )
    parser.add_argument(
        "--model", "-m", default="mock", choices=list(_ALL_MODELS), metavar="MODEL",
        help=f"Model backend. Choices: {', '.join(_ALL_MODELS)}  (default: mock)",
    )
    parser.add_argument(
        "--output", "-o", default=None, metavar="OUTPUT_WAV",
        help=f"Output WAV path. Default: {_OUTPUT_WAV_PATTERN}",
    )
    parser.add_argument(
        "--loop", "-l", action="store_true",
        help="Loop the input until the dashboard window is closed.",
    )
    parser.add_argument(
        "--duration", type=float, default=TEST_SIGNAL_DURATION_S, metavar="SECONDS",
        help=f"Synthetic signal duration in seconds (default: {TEST_SIGNAL_DURATION_S:.0f}).",
    )
    parser.add_argument(
        "--ckpt", default=None, metavar="CHECKPOINT_PT",
        help="Override the default checkpoint path for --model crn or unet.",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = _OUTPUT_WAV_PATTERN.format(model=args.model)

    # ── Build streaming chunk iterator — no full-file pre-load ─────────────────
    if args.input is not None:
        source_label = f"WAV file: {args.input}"
        if args.loop:
            chunk_iter = _looping(lambda: _wav_chunk_stream(Path(args.input)))
        else:
            chunk_iter = _wav_chunk_stream(Path(args.input))
    else:
        source_label = f"Synthetic engine noise ({args.duration:.0f} s)"
        log.info("No input file — streaming synthetic engine noise (%.0f s).", args.duration)
        if args.loop:
            chunk_iter = _looping(lambda: _engine_noise_stream(args.duration))
        else:
            chunk_iter = _engine_noise_stream(args.duration)

    # BiLSTM look-ahead is fixed at N=1 for real-time hardware alignment.
    # The backward LSTM sees exactly 1 chunk of future context — enough to
    # avoid catastrophic boundary artefacts without introducing multi-chunk
    # output delays that are unacceptable on Jetson Nano.
    model = ModelRuntime(args.model, bilstm_lookahead=1, ckpt_path=args.ckpt)

    _print_banner(source_label, model)

    try:
        exit_code = _run_simulation(chunk_iter, model, args)
    except ImportError as exc:
        log.error("Missing dependency: %s", exc)
        log.error("Install: pip install pyqtgraph PyQt6   (Windows)")
        log.error("     or: pip install pyqtgraph && sudo apt-get install python3-pyqt5  (Linux)")
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
