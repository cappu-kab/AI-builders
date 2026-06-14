#!/usr/bin/env python3
"""
anc_latency.py — ANC pipeline latency measurement tool.

Stage 1: ESP32 I2S chunk cadence, UART stream quality, Jetson polling overhead.
         Requires ESP32 firmware compiled with #define LATENCY_DEBUG true.

Stage 2: CRN BiLSTM inference time across chunk sizes (no UART needed).

Usage (on Jetson, anc_env active):
    python3 ~/AI_builders/anc_latency.py --stage 1 [--port /dev/ttyTHS0] [--packets 100]
    python3 ~/AI_builders/anc_latency.py --stage 2
    python3 ~/AI_builders/anc_latency.py --stage all

Extended packet format when LATENCY_DEBUG=true in firmware:
    [0xCC][0xDD][len_lo][len_hi][t0][t1][t2][t3][PCM...]
    len = PCM bytes only; t0..t3 = uint32 LE inter-read interval in microseconds.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
CRN_ROOT    = Path.home() / "AI_builders" / "Run" / "CRN"
REPORT_PATH = Path.home() / "anc_latency_report.txt"

for _d in [str(CRN_ROOT), str(Path(__file__).resolve().parent), str(Path.home())]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

SAMPLE_RATE = 16_000
MIC_H0      = 0xCC
MIC_H1      = 0xDD


# ─────────────────────────────────────────────────────────────────────────────
# Low-level UART helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_n(port, n: int, timeout: float = 2.0) -> Tuple[bytes, Optional[float], float]:
    """
    Read exactly n bytes.  Returns (data, t_first_byte, t_last_byte).
    t_first_byte is None only if n == 0.
    """
    if n == 0:
        return b"", None, time.perf_counter()
    buf       = bytearray()
    t_first: Optional[float] = None
    t_last    = time.perf_counter()
    deadline  = time.perf_counter() + timeout
    while len(buf) < n:
        if time.perf_counter() > deadline:
            raise TimeoutError(f"Timeout reading {n} bytes; got {len(buf)}")
        avail = port.in_waiting
        chunk = port.read(min(avail, n - len(buf)) if avail else 1)
        if chunk:
            if t_first is None:
                t_first = time.perf_counter()
            buf.extend(chunk)
            t_last = time.perf_counter()
    return bytes(buf), t_first, t_last


def _scan_header(port, timeout: float = 5.0) -> float:
    """
    Scan the byte stream for 0xCC 0xDD.
    Returns perf_counter() timestamp captured when 0xCC was seen.
    Handles false-sync (0xCC not followed by 0xDD) correctly.
    """
    deadline   = time.perf_counter() + timeout
    last_val: Optional[int]  = None
    t_cc: Optional[float]    = None

    while time.perf_counter() < deadline:
        b = port.read(1)
        if not b:
            continue
        val = b[0]
        if val == MIC_H1 and last_val == MIC_H0 and t_cc is not None:
            return t_cc
        if val == MIC_H0:
            t_cc = time.perf_counter()
        else:
            t_cc = None
        last_val = val

    raise TimeoutError("No 0xCC 0xDD mic-packet header found within timeout")


def _wait_ok(port, token: bytes, timeout: float = 5.0) -> None:
    """Block until `token` appears in the incoming stream."""
    deadline = time.perf_counter() + timeout
    buf      = bytearray()
    while time.perf_counter() < deadline:
        d = port.read(max(1, port.in_waiting))
        if d:
            buf.extend(d)
            if token in buf:
                return
    raise TimeoutError(f"Timed out waiting for {token!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — UART + I2S latency
# ─────────────────────────────────────────────────────────────────────────────

def _measure_stage1_packets(port, n_packets: int, baud: int) -> Tuple[
        List[float], List[float], List[float]]:
    """
    Collect n_packets extended packets (LATENCY_DEBUG=true firmware).

    Returns three lists (all in milliseconds):
        i2s_ms   — 1a: ESP32 inter-read-completion interval (chunk cadence)
        uart_ms  — 1b: UART stream-stall per packet
        idle_ms  — 1c: idle gap between packets minus expected chunk duration
    """
    i2s_ms:  List[float] = []
    uart_ms: List[float] = []
    idle_ms: List[float] = []

    last_rx_end:       Optional[float] = None
    last_pcm_samples:  int             = 0

    for pkt_idx in range(n_packets):
        # ── Find 0xCC 0xDD ────────────────────────────────────────────────────
        t_rx_start = _scan_header(port)

        # ── Read [len_lo][len_hi][t0][t1][t2][t3] ────────────────────────────
        hdr_rest, _, _ = _read_n(port, 6)   # 2 B length + 4 B timestamp
        pcm_len        = struct.unpack_from("<H", hdr_rest, 0)[0]
        t_interval_us  = struct.unpack_from("<I", hdr_rest, 2)[0]

        # ── Read PCM payload ──────────────────────────────────────────────────
        _, _, t_rx_end = _read_n(port, pcm_len)

        n_samples = pcm_len // 2   # int16 samples

        # 1a — I2S inter-read interval from ESP32.
        # Skip pkt 0: ESP32 sends 0 because there is no previous timestamp yet.
        if pkt_idx > 0 and t_interval_us > 0:
            i2s_ms.append(t_interval_us / 1000.0)

        # 1b — UART stream stall.
        # "actual rx window" = time Jetson spent receiving this packet.
        # Compared against the theoretical serialization time at `baud` baud.
        # Excess = pauses inside the byte stream (e.g., UART FIFO drain, OS jitter).
        total_bytes       = 2 + 2 + 4 + pcm_len   # 0xCC 0xDD + len(2) + ts(4) + pcm
        expected_serial_s = total_bytes * 10 / baud
        stall_ms = (t_rx_end - t_rx_start - expected_serial_s) * 1000.0
        uart_ms.append(max(0.0, stall_ms))

        # 1c — Jetson idle gap between packets minus expected chunk duration.
        # Idle gap = how long Jetson waited before the next 0xCC arrived.
        # Expected = how long that chunk's worth of audio takes (~32 ms @ 16 kHz).
        if last_rx_end is not None and last_pcm_samples > 0:
            idle_gap_ms  = (t_rx_start - last_rx_end) * 1000.0
            expected_gap = last_pcm_samples / SAMPLE_RATE * 1000.0
            idle_ms.append(idle_gap_ms - expected_gap)

        last_rx_end      = t_rx_end
        last_pcm_samples = n_samples

        if (pkt_idx + 1) % 25 == 0 or pkt_idx == 0:
            print(f"    ... {pkt_idx + 1}/{n_packets}", flush=True)

    return i2s_ms, uart_ms, idle_ms


def run_stage1(args) -> List[str]:
    import serial  # guard: only needed for Stage 1

    port = serial.Serial(
        port     = args.port,
        baudrate = args.baud,
        bytesize = serial.EIGHTBITS,
        parity   = serial.PARITY_NONE,
        stopbits = serial.STOPBITS_ONE,
        timeout  = 0.01,    # 10 ms — short for responsive byte scanning
    )
    port.reset_input_buffer()
    print(f"  Opened {args.port} @ {args.baud} baud.", flush=True)

    try:
        port.write(b"startmic\n")
        port.flush()
        _wait_ok(port, b"OK:mic_start")
        print(f"  Mic started.  Collecting {args.packets} packets...", flush=True)

        i2s_ms, uart_ms, idle_ms = _measure_stage1_packets(
            port, args.packets, args.baud)

        port.write(b"stopmic\n")
        port.flush()
        try:
            _wait_ok(port, b"OK:mic_stop", timeout=2.0)
        except TimeoutError:
            pass   # non-fatal; measurements already captured

    finally:
        port.close()

    def _stats(data: List[float]) -> Tuple[float, float, float, int]:
        if not data:
            return 0.0, 0.0, 0.0, 0
        a = np.array(data, dtype=np.float64)
        return float(a.mean()), float(a.min()), float(a.max()), len(a)

    i2s_avg, i2s_min, i2s_max, i2s_n = _stats(i2s_ms)
    urt_avg, urt_min, urt_max, urt_n = _stats(uart_ms)
    idl_avg, idl_min, idl_max, idl_n = _stats(idle_ms)

    # Expected chunk duration for 512 samples @ 16 kHz
    expected_chunk_ms = 512 / SAMPLE_RATE * 1000   # = 32.0 ms

    lines = [
        "",
        "=== Stage 1 Latency Breakdown ===",
        f"1a  Mic → ESP32 I2S buffer:   {i2s_avg:6.1f} ms  "
        f"(avg over {i2s_n} pkts | min {i2s_min:.1f}  max {i2s_max:.1f}  "
        f"expected ≈{expected_chunk_ms:.0f} ms)",
        f"1b  ESP32 → Jetson UART:      {urt_avg:6.1f} ms  "
        f"(avg over {urt_n} pkts | min {urt_min:.1f}  max {urt_max:.1f}  "
        f"stream-stall indicator, ideal=0)",
        f"1c  Jetson polling overhead: {idl_avg:6.1f} ms  "
        f"(avg over {idl_n} pkts | min {idl_min:.1f}  max {idl_max:.1f}  "
        f"idle gap − chunk duration)",
        "─" * 51,
        f"    Total Stage 1:          {i2s_avg + urt_avg + idl_avg:6.1f} ms",
        "",
        "  Notes:",
        "    1a = inter-read-completion interval on ESP32 (chunk cadence).",
        f"         Expected ≈{expected_chunk_ms:.0f} ms for {512} samples @ {SAMPLE_RATE} Hz.",
        "    1b = (actual Jetson rx window) − (total bytes \xd7 10 / baud).",
        "         Values near 0 = continuous byte stream; large = FIFO stall.",
        "    1c = (idle time between packets) − (expected chunk duration).",
        "         Values near 0 = Python polling keeps up with the mic.",
        "",
    ]
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — CRN inference benchmark
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SPECS: List[Tuple[int, int]] = [
    (   20,   320),
    (   50,   800),
    (  100,  1600),
    (  200,  3200),
    (  500,  8000),
    ( 3000, 48000),
]
_N_WARMUP = 3
_N_RUNS   = 10

LOOKAHEAD_MS_LIST: List[int] = [0, 50, 100, 200, 500]


def _bench_one(model, device, n_samples: int, run_fn) -> dict:
    """
    Benchmark a single chunk size.

    GPU path: torch.cuda.synchronize() before and after each timed forward to
    ensure kernel execution is complete before reading perf_counter().
    Without sync, CUDA kernels are async and timings will be implausibly small.

    Warmup: run _N_WARMUP passes first so that lazy cuDNN init / kernel
    compilation cost is paid before the timed loop.
    """
    import torch

    x = np.random.randn(n_samples).astype(np.float32)
    is_cuda = (device.type == "cuda")

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    for _ in range(_N_WARMUP):
        _sync()
        run_fn(model, x, device)
        _sync()

    times_ms: List[float] = []
    for _ in range(_N_RUNS):
        _sync()
        t0 = time.perf_counter()
        run_fn(model, x, device)
        _sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return {
        "mean": float(np.mean(times_ms)),
        "min":  float(np.min(times_ms)),
        "max":  float(np.max(times_ms)),
    }


def run_stage2(args) -> List[str]:
    from crn_pipeline import load_crn, _run_model

    print(f"  Loading CRN from {CRN_ROOT} ...", end=" ", flush=True)
    model, device = load_crn()
    dev_label = "GPU" if device.type == "cuda" else "CPU"
    print(f"OK  ({dev_label})", flush=True)

    gpu_name = ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = f" — {torch.cuda.get_device_name(0)}"
    except Exception:
        pass

    hdr = (f"{'Chunk':>8}  {'Samples':>8}  {'Duration':>10}  "
           f"{'Mean inf.':>10}  {'Min':>8}  {'Max':>8}  RT-capable?")
    lines = [
        "",
        "=== Stage 2 — CRN Inference Time vs Chunk Size ===",
        f"  Device : {dev_label}{gpu_name}",
        f"  Warmup : {_N_WARMUP} runs   Timed : {_N_RUNS} runs per chunk",
        "",
        "  " + hdr,
        "  " + "─" * len(hdr),
    ]

    for ms_dur, n_samples in CHUNK_SPECS:
        print(f"  Benchmarking {ms_dur:5d} ms / {n_samples:6d} samples ...",
              end=" ", flush=True)
        stats = _bench_one(model, device, n_samples, _run_model)
        rt_ok  = stats["mean"] < ms_dur
        rt_str = "✅" if rt_ok else "❌"
        row = (
            f"{ms_dur:>5} ms  {n_samples:>8}  {float(ms_dur):>8.1f} ms"
            f"  {stats['mean']:>8.1f} ms  {stats['min']:>6.1f} ms"
            f"  {stats['max']:>6.1f} ms  {rt_str}"
        )
        lines.append("  " + row)
        print(f"mean={stats['mean']:.1f} ms  {rt_str}", flush=True)

    lines.append("")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Pseudo-streaming simulation
# ─────────────────────────────────────────────────────────────────────────────

def _stream_sim(
    model,
    device,
    signal: np.ndarray,
    hop_s: int,
    lookahead_s: int,
    window_size: int,
    run_fn,
) -> Tuple[np.ndarray, List[float]]:
    """
    Pseudo-streaming simulation: slide a fixed-size window over `signal` in
    steps of `hop_s`, extracting `hop_s` denoised samples per step.

    Window layout at hop k
    ──────────────────────────────────────────────────────────────────────────
    ← history (window_size − hop_s − lookahead_s) → | curr hop (hop_s) | LA (lookahead_s) |
    ←────────────────────── window_size samples (= _CRN_CTX_N) ──────────────────────────→

    The model always receives a full window_size array — the same shape it saw
    during training.  Only output[history_len : history_len+hop_s] is emitted:
    those positions correspond to the denoised estimate for signal[k*H:(k+1)*H],
    having seen real past context to the left and real lookahead context to the
    right.  Earlier and later output positions are discarded each step because
    they are available — with better context — from adjacent hops.

    Algorithmic latency: hop-k output can only be emitted after the lookahead
    region (signal[k*H+H : k*H+H+L]) has arrived, so:
        algo_latency = lookahead_ms + mean_inference_ms

    Returns
        streaming_out : shape (N,) float32 — reconstructed signal
        times_ms      : per-hop inference time in milliseconds
    """
    import torch

    W            = window_size
    H            = hop_s
    L            = lookahead_s
    history_len  = W - H - L
    assert history_len >= 0, (
        f"hop ({H}) + lookahead ({L}) = {H + L} exceeds window size {W}"
    )

    N       = len(signal)
    n_hops  = max(1, (N + H - 1) // H)
    is_cuda = (device.type == "cuda")

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    # One warmup forward pass so lazy CUDA/cuDNN init is paid before timed hops
    _sync()
    run_fn(model, np.zeros(W, dtype=np.float32), device)
    _sync()

    output_parts: List[np.ndarray] = []
    times_ms:     List[float]      = []

    for k in range(n_hops):
        # ── Build padded window ───────────────────────────────────────────────
        # Signal range for this window (may extend outside [0, N) — zero fill).
        sig_win_start = k * H - history_len   # negative early on → zero history
        sig_win_end   = k * H + H + L         # may exceed N → zero lookahead

        window    = np.zeros(W, dtype=np.float32)
        src_start = max(0, sig_win_start)
        src_end   = min(N, sig_win_end)
        if src_end > src_start:
            dst_start = src_start - sig_win_start
            dst_end   = dst_start + (src_end - src_start)
            window[dst_start:dst_end] = signal[src_start:src_end]

        # ── Timed inference ───────────────────────────────────────────────────
        _sync()
        t0         = time.perf_counter()
        out_window = run_fn(model, window, device)
        _sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

        # ── Extract current-hop output ────────────────────────────────────────
        # out_window[history_len : history_len+H] is the best-quality estimate
        # for signal[k*H : (k+1)*H]: the model had real past samples on the
        # left and real lookahead samples on the right at those output positions.
        hop_out    = out_window[history_len : history_len + H]
        actual_len = min(H, N - k * H)   # trim the final hop to signal boundary
        output_parts.append(hop_out[:actual_len].copy())

    streaming_out = (
        np.concatenate(output_parts) if output_parts
        else np.zeros(0, dtype=np.float32)
    )
    return streaming_out, times_ms


def run_stage3(args) -> List[str]:
    """
    Stage 3 entry point.  Loads CRN, generates/reads input signal, runs
    _stream_sim for every lookahead in LOOKAHEAD_MS_LIST, and compares
    each streaming output to the offline enhance_waveform_chunked reference.
    """
    import wave as _wave

    from crn_pipeline import (
        load_crn, _run_model,
        enhance_waveform_chunked,
        _CRN_CTX_N,
    )

    hop_ms = args.hop
    hop_s  = int(round(hop_ms * SAMPLE_RATE / 1000))
    W      = _CRN_CTX_N   # 47616 samples ≈ 2976 ms — the training window

    # Guard: largest lookahead + hop must fit inside the training window
    max_la_s = int(round(max(LOOKAHEAD_MS_LIST) * SAMPLE_RATE / 1000))
    if hop_s + max_la_s >= W:
        raise ValueError(
            f"hop ({hop_ms} ms = {hop_s}) + max lookahead "
            f"({max(LOOKAHEAD_MS_LIST)} ms = {max_la_s}) = {hop_s + max_la_s}"
            f" must be < window {W}.  Reduce --hop or lookahead list."
        )

    # ── Input signal ─────────────────────────────────────────────────────────
    if args.wav:
        with _wave.open(args.wav, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        print(f"  Loaded WAV : {args.wav}  "
              f"({len(signal) / SAMPLE_RATE:.2f} s  {len(signal)} samples)", flush=True)
    else:
        dur_s  = 10.0
        n_sig  = int(dur_s * SAMPLE_RATE)
        np.random.seed(42)
        signal = (np.random.randn(n_sig) * 0.1).astype(np.float32)
        print(f"  Synthetic  : {dur_s:.0f} s white noise  ({n_sig} samples, seed=42)",
              flush=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"  Loading CRN ...", end=" ", flush=True)
    model, device = load_crn()
    dev_label = "GPU" if device.type == "cuda" else "CPU"
    print(f"OK  ({dev_label})", flush=True)

    # ── Offline reference ─────────────────────────────────────────────────────
    # enhance_waveform_chunked uses Hann-windowed OLA with 50% overlap at the
    # full training window size — the highest-quality non-causal baseline.
    print(f"  Running offline reference (enhance_waveform_chunked) ...",
          end=" ", flush=True)
    t0_off      = time.perf_counter()
    offline_out = enhance_waveform_chunked(model, signal, device)
    t1_off      = time.perf_counter()
    print(f"done  ({(t1_off - t0_off) * 1000:.0f} ms)", flush=True)

    # ── Table ─────────────────────────────────────────────────────────────────
    win_ms = W / SAMPLE_RATE * 1000
    n_hops = max(1, (len(signal) + hop_s - 1) // hop_s)
    hdr = (
        f"{'Lookahead':>10}  {'Inf/window':>11}  {'Algo latency':>13}  "
        f"{'RT-capable?':>11}  Quality vs offline"
    )
    lines: List[str] = [
        "",
        f"=== Stage 3 — Pseudo-streaming "
        f"(hop={hop_ms} ms, window={win_ms:.0f} ms, {n_hops} hops) ===",
        f"  Device   : {dev_label}",
        f"  Signal   : {len(signal)} samples = {len(signal) / SAMPLE_RATE:.2f} s",
        f"  Hop      : {hop_ms} ms = {hop_s} samples",
        f"  Window   : {win_ms:.0f} ms = {W} samples  (= _CRN_CTX_N, model unchanged)",
        "",
        "  " + hdr,
        "  " + "─" * len(hdr),
    ]

    for la_ms in LOOKAHEAD_MS_LIST:
        la_s = int(round(la_ms * SAMPLE_RATE / 1000))

        print(f"  Lookahead {la_ms:4d} ms ({la_s:5d} samples) ...", end=" ", flush=True)

        stream_out, times_ms = _stream_sim(
            model, device, signal, hop_s, la_s, W, _run_model,
        )

        mean_inf = float(np.mean(times_ms))
        algo_lat = la_ms + mean_inf
        rt_ok    = mean_inf < hop_ms
        rt_str   = "✅" if rt_ok else "❌"

        # Pearson correlation vs offline output (same-length prefix)
        n_cmp = min(len(stream_out), len(offline_out))
        so = stream_out[:n_cmp]
        oo = offline_out[:n_cmp]
        if n_cmp > 16 and np.std(so) > 1e-9 and np.std(oo) > 1e-9:
            corr     = float(np.corrcoef(so, oo)[0, 1])
            qual_str = f"corr={corr:+.4f}"
        else:
            qual_str = "corr=N/A  (near-zero variance)"

        row = (
            f"{la_ms:>7} ms  "
            f"{mean_inf:>9.1f} ms  "
            f"{algo_lat:>11.1f} ms  "
            f"{rt_str:>11}  "
            f"{qual_str}"
        )
        lines.append("  " + row)
        print(
            f"mean={mean_inf:.1f} ms  algo={algo_lat:.1f} ms  {rt_str}  {qual_str}",
            flush=True,
        )

    lines += [
        "",
        "  Notes:",
        f"    Algo latency = lookahead_ms + mean_inference_ms per window.",
        f"    RT-capable   = mean_inference < hop_duration ({hop_ms} ms).",
        "    corr         = Pearson r vs offline enhance_waveform_chunked output.",
        "                   Larger lookahead → more future context → closer to offline.",
        "",
    ]
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "ANC pipeline latency measurement.\n"
            "  Stage 1: ESP32 I2S + UART (needs LATENCY_DEBUG=true firmware)\n"
            "  Stage 2: CRN BiLSTM inference time vs chunk size\n"
            "  Stage 3: Pseudo-streaming simulation with look-ahead buffer"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--stage",   choices=["1", "2", "3", "all"], default="all",
                    help="Stage(s) to run (default: all)")
    # Stage 1 args
    ap.add_argument("--port",    default="/dev/ttyTHS0",
                    help="UART device for Stage 1 (default: /dev/ttyTHS0)")
    ap.add_argument("--baud",    type=int, default=2_000_000,
                    help="Baud rate for Stage 1 (default: 2000000)")
    ap.add_argument("--packets", type=int, default=100,
                    help="Packets to collect for Stage 1 (default: 100)")
    # Stage 3 args
    ap.add_argument("--hop",     type=int, default=100,
                    help="Streaming hop size in ms for Stage 3 (default: 100)")
    ap.add_argument("--wav",     default=None, metavar="PATH",
                    help="WAV file for Stage 3 input (default: 10 s synthetic noise)")
    args = ap.parse_args()

    ts        = time.strftime("%Y-%m-%d %H:%M:%S")
    all_lines = [f"ANC Latency Report — {ts}", "=" * 51]

    if args.stage in ("1", "all"):
        print("\n=== Stage 1: UART + I2S Latency ===")
        s1 = run_stage1(args)
        all_lines.extend(s1)
        for ln in s1:
            print(ln)

    if args.stage in ("2", "all"):
        print("\n=== Stage 2: CRN Inference Benchmark ===")
        s2 = run_stage2(args)
        all_lines.extend(s2)
        for ln in s2:
            print(ln)

    if args.stage in ("3", "all"):
        print("\n=== Stage 3: Pseudo-streaming Simulation ===")
        s3 = run_stage3(args)
        all_lines.extend(s3)
        for ln in s3:
            print(ln)

    report = "\n".join(all_lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
