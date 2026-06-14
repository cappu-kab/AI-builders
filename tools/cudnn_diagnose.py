#!/usr/bin/env python3
"""
cudnn_diagnose.py — Jetson cuDNN diagnostic + fix tester.

Run on Jetson (single pass, no interactive steps):
    source ~/anc_env/bin/activate
    python3 ~/AI_builders/cudnn_diagnose.py \
        --ckpt ~/AI_builders/Run/CRN/checkpoints_crn2/best.pt

Paste the full output back to the Claude Code session.

Phases
------
1  Environment — torch/CUDA/cuDNN versions, LD_LIBRARY_PATH, libcudnn locations,
               whether torch/lib's bundled .so is real or stripped.
2  Model load  — loads checkpoint on CUDA.
3A Fix attempt — flatten_parameters() + cuDNN enabled.
3B Fix attempt — flatten_parameters() + benchmark=True.
3C Full trace  — raw traceback so the real cuDNN error string is visible.
3D Baseline    — generic CUDA (cuDNN disabled) to establish before/after numbers.
4  Summary     — clear verdict + next-step recommendations.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


# ──────────────────────────────────────────────────────────────
def banner(msg: str) -> None:
    print(f"\n{'='*64}\n  {msg}\n{'='*64}")


def shell(cmd: str, timeout: int = 15) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as exc:
        return f"[cmd error: {exc}]"


def _time_inference(model, dummy, n: int = 5):
    """Warmup 2 runs, then return mean of n runs in ms."""
    import torch
    with torch.inference_mode():
        for _ in range(2):
            model(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model(dummy)
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def _flatten_all(model) -> None:
    import torch
    for m in model.modules():
        if isinstance(m, (torch.nn.LSTM, torch.nn.GRU)):
            m.flatten_parameters()
            print(f"    flatten_parameters() → {type(m).__name__}"
                  f"  layers={m.num_layers}"
                  f"  bidir={getattr(m, 'bidirectional', False)}")


# ──────────────────────────────────────────────────────────────
# PHASE 1: Environment
# ──────────────────────────────────────────────────────────────
banner("PHASE 1: Environment")

import torch  # noqa: E402 — intentionally after banner

print(f"torch:          {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("FATAL: CUDA not available — cannot test cuDNN")
    sys.exit(1)

print(f"torch.version.cuda: {torch.version.cuda}")
print(f"GPU:            {torch.cuda.get_device_name(0)}")

cudnn_rt = torch.backends.cudnn.version()
if cudnn_rt == 8401:
    cudnn_note = "← SYSTEM 8.4.1 loaded (MISMATCH: torch built against 8.6.0)"
elif cudnn_rt == 8600:
    cudnn_note = "← BUNDLED 8.6.0 loaded (correct)"
else:
    cudnn_note = f"← version {cudnn_rt}"
print(f"cudnn runtime:  {cudnn_rt}  {cudnn_note}")
print(f"cudnn.enabled:  {torch.backends.cudnn.enabled}")

print("\nLD_LIBRARY_PATH entries:")
for p in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    marker = " ← torch/lib" if "torch" in p else ""
    print(f"  {p}{marker}")

# Jetson board info
print("\nJetson board:")
print(shell("cat /etc/nv_tegra_release 2>/dev/null | head -3 || echo '(not found)'"))
print(shell("nvcc --version 2>/dev/null | head -4 || echo '(nvcc not found)'"))

# libcudnn locations
banner("libcudnn.so locations (all)")
locs = shell(
    "find /usr /opt /lib -name 'libcudnn*.so*' 2>/dev/null "
    "| grep -v __pycache__ | sort -u"
)
print(locs or "(none found in /usr /opt /lib)")

# torch/lib specifically
torch_lib = Path(torch.__file__).parent / "lib"
print(f"\ntorch/lib path: {torch_lib}")
bundled = list(torch_lib.glob("libcudnn*.so*"))
if bundled:
    for f in bundled:
        real_f = f.resolve()
        try:
            sz = real_f.stat().st_size
            stripped = sz < 200_000  # real libcudnn.so.8 is hundreds of MB
            tag = "  *** STRIPPED (placeholder only) ***" if stripped else f"  {sz/1e6:.1f} MB — looks real"
        except OSError:
            tag = "  (dangling symlink?)"
        print(f"  {f.name} → {real_f.name}{tag}")
else:
    print("  (no libcudnn* in torch/lib)")

# ldd on torch's _C.so to see which cuDNN it links at runtime
torchC = list(torch_lib.glob("_C*.so")) or list(torch_lib.glob("libtorch_cuda*.so"))
if torchC:
    print(f"\nldd {torchC[0].name} | grep cudnn:")
    print(shell(f"ldd {torchC[0]} 2>/dev/null | grep -i cudnn || echo '(none)'"))

# ──────────────────────────────────────────────────────────────
# PHASE 2: Load model
# ──────────────────────────────────────────────────────────────
banner("PHASE 2: Load CRN checkpoint")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    default=str(Path(__file__).parent.parent / "Run" / "CRN" / "checkpoints" / "best.pt"),
)
parser.add_argument("--ctx", type=int, default=47616,
                    help="Context samples (default=47616 = 3s @ 16 kHz)")
args = parser.parse_args()
ckpt_path = Path(args.ckpt)

if not ckpt_path.exists():
    print(f"ERROR: checkpoint not found: {ckpt_path}")
    sys.exit(1)

crn_dir = Path(__file__).parent.parent / "Run" / "CRN"
if str(crn_dir) not in sys.path:
    sys.path.insert(0, str(crn_dir))

from inference import load_model  # noqa: E402

device = torch.device("cuda")
print(f"Loading {ckpt_path} → cuda ...")
t0 = time.time()
model, ckpt = load_model(str(ckpt_path), device)
print(f"Loaded in {time.time()-t0:.2f}s")
print(f"rnn type:  {type(model.rnn).__name__}")
print(f"bidir:     {getattr(model.rnn, 'bidirectional', False)}")
print(f"layers:    {model.rnn.num_layers}")
print(f"hidden:    {model.rnn.hidden_size}")

dummy = torch.randn(1, args.ctx, device=device)
print(f"Input shape: {dummy.shape}  ({args.ctx/16000:.2f}s @ 16 kHz)")

# ──────────────────────────────────────────────────────────────
# PHASE 3A: flatten_parameters + cuDNN enabled
# ──────────────────────────────────────────────────────────────
banner("PHASE 3A: flatten_parameters() + cudnn.enabled=True  (benchmark=False)")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
_flatten_all(model)

cudnn_ok = False
ms_cudnn = None
try:
    ms_cudnn = _time_inference(model, dummy)
    print(f"\nSUCCESS — cuDNN path working.  Avg: {ms_cudnn:.1f} ms/window")
    cudnn_ok = True
except RuntimeError as exc:
    print(f"\nFAILED: {exc}")

# ──────────────────────────────────────────────────────────────
# PHASE 3B: flatten_parameters + benchmark=True
# ──────────────────────────────────────────────────────────────
if not cudnn_ok:
    banner("PHASE 3B: flatten_parameters() + cudnn.benchmark=True")
    torch.backends.cudnn.benchmark = True
    _flatten_all(model)
    try:
        ms_cudnn = _time_inference(model, dummy)
        print(f"\nSUCCESS — cuDNN benchmark path working.  Avg: {ms_cudnn:.1f} ms/window")
        cudnn_ok = True
    except RuntimeError as exc:
        print(f"\nFAILED: {exc}")
    finally:
        torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────────────────────
# PHASE 3C: full traceback (so the real cuDNN error is visible)
# ──────────────────────────────────────────────────────────────
if not cudnn_ok:
    banner("PHASE 3C: Full traceback with cudnn.enabled=True")
    torch.backends.cudnn.enabled = True
    _flatten_all(model)
    try:
        with torch.inference_mode():
            model(dummy)
            torch.cuda.synchronize()
    except Exception:
        traceback.print_exc()

# ──────────────────────────────────────────────────────────────
# PHASE 3D: generic CUDA baseline
# ──────────────────────────────────────────────────────────────
banner("PHASE 3D: Generic CUDA baseline (cudnn.enabled=False)")
torch.backends.cudnn.enabled = False
_flatten_all(model)
ms_generic = None
try:
    ms_generic = _time_inference(model, dummy)
    print(f"Generic CUDA avg: {ms_generic:.1f} ms/window")
except Exception as exc:
    print(f"Generic CUDA also failed: {exc}")

# restore
torch.backends.cudnn.enabled = True

# ──────────────────────────────────────────────────────────────
# PHASE 4: Summary & recommendation
# ──────────────────────────────────────────────────────────────
banner("PHASE 4: Summary")

if cudnn_ok:
    print("VERDICT: cuDNN LSTM is WORKING after flatten_parameters().")
    print()
    print(f"  cuDNN path:   {ms_cudnn:.1f} ms / {args.ctx}-sample window")
    if ms_generic:
        print(f"  Generic CUDA: {ms_generic:.1f} ms / {args.ctx}-sample window")
        print(f"  Speedup:      {ms_generic/ms_cudnn:.2f}×")
    print()
    print("NEXT STEPS:")
    print("  1. inference.py: flatten_parameters() is now added — scp updated file.")
    print("  2. In load_crn() (crn_pipeline.py): add _flatten_lstm(model) call")
    print("     after model.to(device) + model.eval(), BEFORE the first cuDNN test.")
    print("  3. Ensure torch.backends.cudnn.enabled = True is NOT overwritten to")
    print("     False in the load_crn() fallback chain AFTER a successful 3A pass.")
else:
    print("VERDICT: cuDNN LSTM is NOT WORKING on this board/torch combination.")
    print()
    if cudnn_rt == 8401:
        print("ROOT CAUSE: version mismatch.")
        print(f"  PyTorch 2.0.0+nv23.05 was compiled against cuDNN 8.6.0.")
        print(f"  Runtime is loading system cuDNN 8.4.1 (via LD_LIBRARY_PATH).")
        print()
        print("CHECK: does torch/lib contain a real (non-stripped) libcudnn.so.8?")
        print("  If YES → run ~/fix_cudnn.sh to prepend torch/lib to LD_LIBRARY_PATH.")
        print("  If NO  → the bundled cuDNN is stripped; options:")
        print("    A) scp cuDNN 8.6.x .deb from Windows and dpkg --extract to ~/cudnn86/")
        print("       then prepend ~/cudnn86/usr/lib/aarch64-linux-gnu to LD_LIBRARY_PATH.")
        print("    B) Accept generic CUDA path.")
        if ms_generic:
            print(f"       Generic CUDA: {ms_generic:.1f} ms — compare to TRT 11ms path.")
    else:
        print(f"cuDNN runtime version: {cudnn_rt}")
        print("See 3C traceback above for the root error string.")
    print()
    print("Paste this full output back to Claude Code to get the targeted fix.")
