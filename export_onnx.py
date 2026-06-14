"""
export_onnx.py
==============
Export the trained Causal-TCN predictor (`tcn_predictor.pt`) to ONNX so the
Jetson Nano can run it with TensorRT / ONNX-Runtime instead of native PyTorch.

What gets exported
------------------
A single ONNX graph with one dynamic axis on the time dimension, so the same
file covers BOTH inference modes the pipeline uses:

    chunk-based  (T = 256)    one hardware I2S block        — debug / smoke
    context-based (T = 1024)  full TCN receptive field (64 ms) — production

Input  : tensor `noise_context`  shape (B, T)   float32  RAW amplitude
                                 dynamic axes : batch (0), time (1)
Output : tensor `prediction`     shape (B, 80)  float32  RAW amplitude
                                 dynamic axes : batch (0)

The model bakes its z-score buffers (x_mean / x_std / y_mean / y_std) into the
graph as ONNX constants, so the exported file is fully self-contained.

Verification
------------
The script also reloads the exported file with `onnxruntime`, runs it at both
T=256 and T=1024, and prints the max absolute deviation versus the PyTorch
reference output.  A clean export keeps this delta well under 1e-5.

Usage
-----
    python export_onnx.py                                    # defaults
    python export_onnx.py --ckpt tcn_predictor.pt \
                          --out  tcn_predictor.onnx \
                          --opset 17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ANC_DIR    = SCRIPT_DIR / "ANC_Hardware_Test"
if str(ANC_DIR) not in sys.path:
    sys.path.insert(0, str(ANC_DIR))

from predictor import LightweightTCN, INPUT_LEN, PREDICT_LEN   # noqa: E402

# Hardware-chunk size — match `predictor.CHUNK_SIZE`.  Imported lazily so a stale
# checkpoint without that symbol still works.
try:
    from predictor import CHUNK_SIZE                            # noqa: E402
except ImportError:
    CHUNK_SIZE = 256


class TCNExportWrapper(torch.nn.Module):
    """Strip the `return_normalised` kwarg so the ONNX graph has a single input.

    `torch.onnx.export` will otherwise either trace through the conditional or
    refuse altogether — wrapping is cleaner and faster than passing kwargs.
    """

    def __init__(self, tcn: LightweightTCN) -> None:
        super().__init__()
        self.tcn = tcn

    def forward(self, noise_context: torch.Tensor) -> torch.Tensor:
        return self.tcn(noise_context)              # always un-standardised output


def load_tcn(ckpt: Path, device: torch.device) -> LightweightTCN:
    model = LightweightTCN().to(device)
    state = torch.load(str(ckpt), map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        # strict=True would have raised, but be loud if we ever loosen it.
        print(f"  [warn] missing={missing}  unexpected={unexpected}")
    model.eval()
    return model


def export(model: LightweightTCN, out_path: Path, opset: int,
           device: torch.device) -> None:
    wrapper = TCNExportWrapper(model).to(device).eval()

    # Trace at the production context length so all dilations + pyramid pools
    # see their natural footprint.  The dynamic time axis lets the same graph
    # accept T = CHUNK_SIZE (256) or any other length at runtime.
    dummy = torch.zeros(1, INPUT_LEN, dtype=torch.float32, device=device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names  = ["noise_context"],
        output_names = ["prediction"],
        dynamic_axes = {
            "noise_context": {0: "batch", 1: "time"},
            "prediction":    {0: "batch"},
        },
        opset_version = opset,
        do_constant_folding = True,
    )
    print(f"  wrote -> {out_path}  ({out_path.stat().st_size / 1024:.1f} kB)")


def verify(model: LightweightTCN, out_path: Path, device: torch.device) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [verify] onnxruntime not installed; skipping numeric check.")
        return

    sess = ort.InferenceSession(str(out_path),
                                providers=["CPUExecutionProvider"])

    for tag, T in [("chunk", CHUNK_SIZE), ("context", INPUT_LEN)]:
        rng = np.random.default_rng(1234)                       # deterministic
        x_np = rng.standard_normal((2, T)).astype(np.float32) * 0.05

        # ONNX side
        y_ort = sess.run(["prediction"], {"noise_context": x_np})[0]

        # PyTorch reference
        with torch.no_grad():
            y_pt = model(torch.from_numpy(x_np).to(device)).cpu().numpy()

        delta = float(np.max(np.abs(y_ort - y_pt)))
        verdict = "OK" if delta < 1e-4 else "WARN — large drift"
        print(f"  [verify] {tag:7s}  T={T:>4d}  out={y_ort.shape}  "
              f"max|ort-pt| = {delta:.2e}   {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export tcn_predictor.pt → ONNX.")
    ap.add_argument("--ckpt", type=Path,
                    default=SCRIPT_DIR / "tcn_predictor.pt",
                    help="LightweightTCN state_dict (default: tcn_predictor.pt).")
    ap.add_argument("--out",  type=Path,
                    default=SCRIPT_DIR / "tcn_predictor.onnx",
                    help="Output ONNX file (default: tcn_predictor.onnx).")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset version (17 covers GELU + Conv1d cleanly).")
    args = ap.parse_args()

    # ONNX export is friendlier and more reproducible on CPU; the runtime still
    # runs identically on GPU/Jetson because the graph is device-agnostic.
    device = torch.device("cpu")

    print(f"[load]  {args.ckpt}")
    model = load_tcn(args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"        LightweightTCN  params = {n_params}  "
          f"input_len={INPUT_LEN}  predict_len={PREDICT_LEN}  chunk={CHUNK_SIZE}")

    print(f"\n[export] opset={args.opset}  dynamic axes: batch + time")
    export(model, args.out, args.opset, device)

    print("\n[verify] running both inference modes through onnxruntime ...")
    verify(model, args.out, device)


if __name__ == "__main__":
    main()
