"""
run.py
======
Single entry-point to drive preprocessing, training, and evaluation for
either denoiser project (CRN or speech_denoise/AdvancedUNetSE).

Examples
--------
    # Preprocess raw audio → .npy
    python run.py preprocess --in_root ./raw_audio --out_root ./data_npy

    # Smoke-test (1 epoch, tiny budget) — verify the pipeline runs end-to-end
    python run.py smoke --project crn
    python run.py smoke --project speech_denoise

    # Full training (sane defaults for limited-resource GPUs)
    python run.py train --project crn          --data_root ./CRN/data_npy
    python run.py train --project speech_denoise --data_root ./speech_denoise/data_npy

    # Finetune from existing checkpoint
    python run.py train --project speech_denoise --finetune \\
        --resume ./speech_denoise/checkpoints/best.pt

    # Evaluate
    python run.py evaluate --project crn --ckpt ./CRN/checkpoints/best.pt
    python run.py evaluate --project speech_denoise --ckpt ./speech_denoise/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PROJECT_DIRS = {
    "crn":            REPO / "CRN",
    "speech_denoise": REPO / "speech_denoise",
}


def _run(cmd: list, cwd: Path) -> int:
    print(f"\n$ (cd {cwd.name} && {' '.join(cmd)})\n", flush=True)
    return subprocess.call(cmd, cwd=str(cwd))


def cmd_preprocess(args, extra: list) -> int:
    # Canonical preprocessor: data_prep/preprocess4.py — strict CSV→label
    # row-aligned, fixed-length pad/trim, peak-normalized. See PREDICTOR.md /
    # the deep-audit notes for why the older preprocess.py / preprocess_v3.py
    # variants were removed.
    preproc = REPO.parent / "data_prep" / "preprocess4.py"
    cmd = [sys.executable, str(preproc)] + extra
    return _run(cmd, REPO)


def cmd_smoke(args, extra: list) -> int:
    project_dir = PROJECT_DIRS[args.project]
    data_root = args.data_root or "./data_npy"
    cmd = [
        sys.executable, "train.py",
        "--data_root", data_root,
        "--epochs", "1",
        "--steps_per_epoch", "5",
        "--val_max_batches", "2",
        "--num_workers", "2",
        "--rir_pool_size", "16",
        "--debug_every", "0",
    ] + extra
    return _run(cmd, project_dir)


def cmd_train(args, extra: list) -> int:
    project_dir = PROJECT_DIRS[args.project]
    cmd = [sys.executable, "train.py"]
    if args.data_root:
        cmd += ["--data_root", args.data_root]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size:
        cmd += ["--batch_size", str(args.batch_size)]
    if args.resume:
        cmd += ["--resume", args.resume]
    if args.finetune:
        cmd += ["--finetune"]
    if args.no_preload_noise:
        cmd += ["--no_preload_noise"]
    if args.amp_dtype:
        cmd += ["--amp_dtype", args.amp_dtype]
    cmd += extra
    return _run(cmd, project_dir)


def cmd_evaluate(args, extra: list) -> int:
    project_dir = PROJECT_DIRS[args.project]
    eval_path = project_dir / "evaluate.py"
    if not eval_path.is_file():
        print(f"[run] {eval_path} not found"); return 1
    cmd = [sys.executable, "evaluate.py"]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]
    if args.data_root:
        cmd += ["--data_root", args.data_root]
    if args.max_files:
        cmd += ["--max_files", str(args.max_files)]
    cmd += extra
    return _run(cmd, project_dir)


def main():
    ap = argparse.ArgumentParser(description="Speech-denoiser pipeline runner.")
    sub = ap.add_subparsers(dest="stage", required=True)

    p_pre = sub.add_parser("preprocess", help="Run preprocess.py (raw audio → .npy)")
    p_pre.set_defaults(fn=cmd_preprocess)

    p_smoke = sub.add_parser("smoke", help="1-epoch smoke test (5 steps, 2 val batches)")
    p_smoke.add_argument("--project", choices=list(PROJECT_DIRS), required=True)
    p_smoke.add_argument("--data_root", default=None)
    p_smoke.set_defaults(fn=cmd_smoke)

    p_tr = sub.add_parser("train", help="Full training")
    p_tr.add_argument("--project", choices=list(PROJECT_DIRS), required=True)
    p_tr.add_argument("--data_root", default=None)
    p_tr.add_argument("--epochs", type=int, default=None)
    p_tr.add_argument("--batch_size", type=int, default=None)
    p_tr.add_argument("--resume", default=None)
    p_tr.add_argument("--finetune", action="store_true")
    p_tr.add_argument("--no_preload_noise", action="store_true")
    p_tr.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default=None)
    p_tr.set_defaults(fn=cmd_train)

    p_ev = sub.add_parser("evaluate", help="Run evaluate.py")
    p_ev.add_argument("--project", choices=list(PROJECT_DIRS), required=True)
    p_ev.add_argument("--ckpt", default=None)
    p_ev.add_argument("--data_root", default=None)
    p_ev.add_argument("--max_files", type=int, default=None)
    p_ev.set_defaults(fn=cmd_evaluate)

    args, extra = ap.parse_known_args()
    rc = args.fn(args, extra)
    sys.exit(rc)


if __name__ == "__main__":
    main()
