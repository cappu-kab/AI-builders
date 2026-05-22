"""
eval.py
=======
Quick single-model evaluation on the test split.

Usage (from Run/):
    python eval.py crn
    python eval.py unet

    # custom checkpoint or data:
    python eval.py crn  --ckpt CRN/checkpoints_crn/best.pt --max_files 50
    python eval.py unet --ckpt U_net/checkpoints/best.pt   --snr 0

    # summary only (no per-file rows):
    python eval.py crn --summary_only
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
def _load_module(path: Path, name: str):
    parent = str(path.parent)
    added  = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        if added:
            sys.path.remove(parent)
    return mod


def si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha  = float(np.dot(est, ref) / (np.dot(ref, ref) + eps))
    target = alpha * ref
    noise  = est - target
    return float(10.0 * np.log10((np.sum(target ** 2) + eps) /
                                  (np.sum(noise  ** 2) + eps)))


def _mix(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    eps = 1e-8
    cr = float(np.sqrt(np.mean(clean ** 2) + eps))
    nr = float(np.sqrt(np.mean(noise ** 2) + eps))
    a  = cr / (nr * (10.0 ** (snr_db / 20.0)) + eps)
    return (clean + a * noise).astype(np.float32)


def _crop_or_pad(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if x.size >= n else np.pad(x, (0, n - x.size)).astype(np.float32)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model",        choices=["crn", "unet"],
                    help="Which model to evaluate")
    ap.add_argument("--data_root",  default="../data_npy")
    ap.add_argument("--split",      default="test")
    ap.add_argument("--ckpt",       default=None,
                    help="Override checkpoint path")
    ap.add_argument("--max_files",  type=int,   default=200)
    ap.add_argument("--snr",        type=float, default=5.0)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--sample_rate",type=int,   default=16_000)
    ap.add_argument("--summary_only", action="store_true",
                    help="Skip per-file rows, only print summary")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- checkpoint & module paths ----
    if args.model == "crn":
        model_dir    = ROOT / "CRN"
        default_ckpt = ROOT / "checkpoints_crn" / "best.pt"
    else:
        model_dir    = ROOT / "U_net"
        default_ckpt = ROOT / "U_net" / "checkpoints" / "best.pt"

    ckpt_path = Path(args.ckpt) if args.ckpt else default_ckpt
    if not ckpt_path.is_file():
        print(f"[eval] ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    inf_mod       = _load_module(model_dir / "inference.py", f"{args.model}_inf")
    model, ckpt_d = inf_mod.load_model(str(ckpt_path), device)
    epoch = ckpt_d.get("epoch", "?") if isinstance(ckpt_d, dict) else "?"
    print(f"[eval] model={args.model.upper()}  epoch={epoch}  device={device}")
    print(f"[eval] ckpt : {ckpt_path}")

    # ---- test data ----
    split_dir   = Path(args.data_root) / args.split
    clean_files = sorted((split_dir / "clean").glob("*.npy"))
    noise_files = sorted((split_dir / "noise").glob("*.npy"))

    if not clean_files:
        print(f"[eval] ERROR: no .npy files found in {split_dir / 'clean'}")
        sys.exit(1)

    random.seed(args.seed)
    if len(clean_files) > args.max_files:
        clean_files = random.sample(clean_files, args.max_files)

    rng = np.random.default_rng(args.seed)
    print(f"[eval] split={args.split}  snr={args.snr} dB  "
          f"files={len(clean_files)}  noise={len(noise_files)}\n")

    # ---- header ----
    if not args.summary_only:
        print(f"{'#':>4}  {'File':<38}  {'Noisy':>7}  {'Enh':>7}  {'Delta':>7}")
        print("-" * 68)

    noisy_sum = 0.0
    enh_sum   = 0.0
    errors    = 0
    deltas    = []

    for i, cf in enumerate(clean_files):
        clean = np.load(cf).astype(np.float32).squeeze()
        if clean.ndim > 1:
            clean = clean.mean(axis=0)

        if noise_files:
            nf    = noise_files[int(rng.integers(0, len(noise_files)))]
            noise = np.load(nf).astype(np.float32).squeeze()
            if noise.ndim > 1:
                noise = noise.mean(axis=0)
            noise = _crop_or_pad(noise, clean.size)
        else:
            noise = rng.standard_normal(clean.size).astype(np.float32)

        noisy = _mix(clean, noise, args.snr)

        try:
            enh = inf_mod.enhance_waveform(model, noisy, device,
                                            sr=args.sample_rate)
        except Exception as e:
            print(f"  [!] error on {cf.name}: {e}")
            errors += 1
            continue

        enh       = _crop_or_pad(enh, clean.size)
        noisy_s   = si_sdr(clean, noisy)
        enh_s     = si_sdr(clean, enh)
        delta     = enh_s - noisy_s

        noisy_sum += noisy_s
        enh_sum   += enh_s
        deltas.append(delta)

        if not args.summary_only:
            print(f"{i+1:>4}  {cf.name:<38}  {noisy_s:>+7.2f}  "
                  f"{enh_s:>+7.2f}  {delta:>+7.2f}")

    # ---- summary ----
    n = len(deltas)
    if n == 0:
        print("[eval] No files processed successfully.")
        return

    if not args.summary_only:
        print("-" * 68)

    print(f"\n{'='*40}")
    print(f" Model       : {args.model.upper()}")
    print(f" Checkpoint  : {ckpt_path.name}")
    print(f" Files       : {n}  (errors={errors})")
    print(f" Avg Noisy   : {noisy_sum/n:>+7.2f} dB SI-SNR")
    print(f" Avg Enhanced: {enh_sum/n:>+7.2f} dB SI-SNR")
    print(f" Avg Delta   : {(enh_sum-noisy_sum)/n:>+7.2f} dB")
    print(f" Best file   : {max(deltas):>+7.2f} dB")
    print(f" Worst file  : {min(deltas):>+7.2f} dB")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
