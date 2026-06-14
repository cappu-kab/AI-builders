#!/usr/bin/env python3
"""
benchmark_crn_onnx.py
=====================
Measure CRN ONNX Runtime latency at multiple context-window sizes.

The ONNX model (crn_crn2_opset15.onnx) was exported with a dynamic T axis,
so a single file covers every context size.  No re-export needed per sweep.

STFT shape for context prefix N (CHUNK_SIZE=256, HOP=128, n_fft=512, center=True):
    T_wave  = N + 256
    T_stft  = T_wave // 128 + 1         (center=True adds n_fft//2 padding each side)
    Input   = (1, 2, 257, T_stft)  float32

Sweep defaults:  --ctx 3200 8000 16000 47616
    N=3200   -> T_stft=28    (200 ms context,  real-time target)
    N=8000   -> T_stft=65    (500 ms context)
    N=16000  -> T_stft=128   (1 s context)
    N=47616  -> T_stft=375   (3 s context, model training length — baseline)

Usage on Jetson (anc_env must be active):
    source ~/anc_env/bin/activate
    python3 ~/benchmark_crn_onnx.py
    python3 ~/benchmark_crn_onnx.py --onnx ~/AI_builders/Run/CRN/crn_crn2.onnx
    python3 ~/benchmark_crn_onnx.py --ctx 3200 8000 16000 47616
    python3 ~/benchmark_crn_onnx.py --ctx 3200 --warmup 20 --runs 100
    python3 ~/benchmark_crn_onnx.py --cpu
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

CHUNK_SIZE      = 256
HOP             = 128
F_BINS          = 257          # n_fft=512 -> 257 bins
CHUNK_BUDGET_MS = 16.0         # 256/16000*1000

_DEFAULT_CTX    = [3200, 8000, 16000, 47616]


def stft_frames(ctx_n: int) -> int:
    """T_stft = (ctx_n + CHUNK_SIZE) // HOP + 1  (center=True padding)."""
    return (ctx_n + CHUNK_SIZE) // HOP + 1


def parse_args():
    p = argparse.ArgumentParser(description="CRN ONNX Runtime latency sweep")
    p.add_argument("--onnx",   default=None,
                   help="Path to crn_crn2.onnx "
                        "(default: ~/AI_builders/Run/CRN/crn_crn2.onnx)")
    p.add_argument("--ctx",    type=int, nargs="+", default=_DEFAULT_CTX,
                   metavar="N",
                   help="Context prefix size(s) in samples "
                        "(default: 3200 8000 16000 47616)")
    p.add_argument("--warmup", type=int, default=10,
                   help="Warm-up calls before timing (default: 10)")
    p.add_argument("--runs",   type=int, default=50,
                   help="Timed calls (default: 50)")
    p.add_argument("--cpu",    action="store_true",
                   help="Force CPUExecutionProvider (baseline comparison)")
    return p.parse_args()


def main():
    args = parse_args()

    onnx_path = (Path(args.onnx).expanduser() if args.onnx
                 else Path(__file__).parent.parent / "Run" / "CRN" / "crn_crn2.onnx")

    print("=" * 66)
    print("  CRN ONNX Runtime Latency Sweep")
    print("=" * 66)
    print(f"  ONNX file : {onnx_path}")
    print(f"  Contexts  : {args.ctx}  (samples)")
    print(f"  Warm-up   : {args.warmup}  Timed runs : {args.runs}")
    print()

    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}")
        print("  scp command from Windows:")
        print("    scp <windows-ip>:/Users/rocha/AI_builders/crn_crn2_opset15.onnx "
              "~/AI_builders/Run/CRN/crn_crn2.onnx")
        sys.exit(1)

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed.")
        print("  pip install ~/anc_wheels/onnxruntime-*.whl")
        sys.exit(1)

    print(f"[1] ORT version          : {ort.__version__}")
    print(f"    Available providers  : {ort.get_available_providers()}")
    print()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = (["CPUExecutionProvider"] if args.cpu
                 else ["CUDAExecutionProvider", "CPUExecutionProvider"])

    print(f"[2] Loading session (requested: {providers[0]}) ...")
    t0 = time.perf_counter()
    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                    providers=providers)
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)
    load_ms = (time.perf_counter() - t0) * 1000

    active = sess.get_providers()
    using_gpu = any("CUDA" in p or "TensorRT" in p for p in active)
    backend   = "GPU (CUDA)" if using_gpu else "CPU"

    print(f"  Active providers : {active}")
    print(f"  Session loaded   : {load_ms:.0f} ms")
    if not args.cpu and not using_gpu:
        print()
        print("  WARNING: CUDA requested but NOT active — fell back to CPU.")
        print("    Check: onnxruntime-gpu installed? LD_LIBRARY_PATH set?")
    print()

    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print(f"[3] Model I/O : input='{in_name}' / output='{out_name}'")
    print()

    # ── per-context sweep ─────────────────────────────────────────────────────
    results = []

    for ctx_n in args.ctx:
        t_frames = stft_frames(ctx_n)
        dummy    = np.zeros((1, 2, F_BINS, t_frames), dtype=np.float32)

        print(f"[ctx={ctx_n:>6} smp | T={t_frames:>4} frames | "
              f"input shape (1,2,{F_BINS},{t_frames})]")

        # warm-up
        for _ in range(args.warmup):
            sess.run([out_name], {in_name: dummy})

        # timed runs
        lats = []
        for _ in range(args.runs):
            t_s = time.perf_counter()
            sess.run([out_name], {in_name: dummy})
            lats.append((time.perf_counter() - t_s) * 1000.0)

        arr    = np.array(lats)
        avg_ms = float(arr.mean())
        min_ms = float(arr.min())
        p50_ms = float(np.percentile(arr, 50))
        p95_ms = float(np.percentile(arr, 95))
        p99_ms = float(arr.max())
        ratio  = avg_ms / CHUNK_BUDGET_MS
        status = "OK " if ratio <= 1.0 else f"x{ratio:.1f}"

        print(f"  avg={avg_ms:7.2f} ms  min={min_ms:6.2f}  "
              f"p50={p50_ms:7.2f}  p95={p95_ms:7.2f}  p99={p99_ms:7.2f}  "
              f"=> {status}")
        print()

        results.append((ctx_n, t_frames, avg_ms, p50_ms, p95_ms, ratio))

    # ── summary table ─────────────────────────────────────────────────────────
    print("=" * 66)
    print(f"  CONTEXT SWEEP SUMMARY  [{backend}]")
    print("=" * 66)
    print(f"  {'ctx_N':>7}  {'T_stft':>6}  {'ctx_ms':>7}  "
          f"{'avg_ms':>8}  {'p50_ms':>8}  {'p95_ms':>8}  {'ratio':>6}")
    print("  " + "-" * 63)
    for ctx_n, t_frames, avg_ms, p50_ms, p95_ms, ratio in results:
        ctx_duration_ms = ctx_n / 16_000 * 1000
        flag = " <-- RT OK" if ratio <= 1.0 else (
               " <-- marginal" if ratio <= 2.0 else "")
        print(f"  {ctx_n:>7}  {t_frames:>6}  {ctx_duration_ms:>7.0f}  "
              f"{avg_ms:>8.2f}  {p50_ms:>8.2f}  {p95_ms:>8.2f}  "
              f"{ratio:>5.2f}x{flag}")
    print("=" * 66)
    print()

    # ── recommendation ────────────────────────────────────────────────────────
    rt_ok = [r for r in results if r[5] <= 1.0]
    marginal = [r for r in results if 1.0 < r[5] <= 6.25]  # <=100 ms for phase corr

    if rt_ok:
        best = max(rt_ok, key=lambda r: r[0])
        print(f"  Largest real-time context: N={best[0]} ({best[0]/16:.0f} ms) "
              f"avg={best[2]:.1f} ms")
    if marginal:
        best_m = max(marginal, key=lambda r: r[0])
        print(f"  Largest phase-correction context (<100 ms): "
              f"N={best_m[0]} ({best_m[0]/16:.0f} ms) avg={best_m[2]:.1f} ms")

    if not rt_ok and not marginal:
        print("  All contexts exceed 100 ms — try TensorRT export (Step 4).")
    elif not using_gpu and not args.cpu:
        print()
        print("  Re-run without --cpu to use GPU path.")


if __name__ == "__main__":
    main()
