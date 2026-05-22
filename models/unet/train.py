"""
train.py
========
Train the AdvancedUNetSE speech-denoiser.

Architecture: pure U-Net with residual noise prediction.
    enhanced = noisy_wave − iSTFT( mask ⊙ STFT(noisy_wave) )

The model accepts raw waveforms (B, T) and returns enhanced waveforms (B, T).
STFT/iSTFT live inside the model; no STFTFront module needed here.

Training infrastructure:
- On-the-fly augmentation (brown/pink noise, biased SNR, RIR, etc.)
- 10-term DenoiseLoss with 5x low-frequency boost < 200 Hz
- Full hard SNR range [-5, 15] dB from epoch 1 (no curriculum)
- ReduceLROnPlateau scheduler (halves LR after 3 epochs no improvement)
- SI-SDR metric-driven checkpointing
- Early stopping (--patience)
- Spectrogram debug images every N epochs

Local usage
-----------
    python train.py --data_root ./data_npy --epochs 50 --batch_size 8

Colab usage
-----------
    !python train.py --data_root /content/data_npy --epochs 50 --batch_size 16
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):
        return it

from dataset import SpeechDenoiseDataset, denoise_collate, validate_dataset, worker_init_fn
from losses import DenoiseLoss
from model_unet_advanced import AdvancedUNetSE

import json
import glob


def _rotate_topk(ckpt_dir: str, score: float, payload: dict, k: int = 3) -> None:
    pattern = os.path.join(ckpt_dir, "best_top*.pt")
    existing = []
    for p in sorted(glob.glob(pattern)):
        sp = p + ".score"
        if os.path.isfile(sp):
            try:
                existing.append((float(open(sp).read().strip()), p, sp))
            except Exception:
                pass
    tmps = []
    for sc, ck_p, sc_p in existing:
        ck_t = ck_p + ".rotate"
        sc_t = sc_p + ".rotate"
        try:
            os.replace(ck_p, ck_t)
            os.replace(sc_p, sc_t)
            tmps.append((sc, ck_t, sc_t))
        except Exception:
            pass
    cand = tmps + [(score, None, None)]
    cand.sort(key=lambda t: t[0])
    keep = cand[:k]
    drop = cand[k:]
    for rank, (sc, ck_src, sc_src) in enumerate(keep, start=1):
        ck_dst = os.path.join(ckpt_dir, f"best_top{rank}.pt")
        sc_dst = ck_dst + ".score"
        if ck_src is None:
            torch.save(payload, ck_dst)
            with open(sc_dst, "w") as fh:
                fh.write(f"{sc:.6f}")
        else:
            os.replace(ck_src, ck_dst)
            os.replace(sc_src, sc_dst)
    for sc, ck_src, sc_src in drop:
        if ck_src and os.path.isfile(ck_src):
            try: os.remove(ck_src)
            except Exception: pass
        if sc_src and os.path.isfile(sc_src):
            try: os.remove(sc_src)
            except Exception: pass


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data_npy",
                   help="Path to data_npy/. Default points one level up from Colab_Ready_UNet "
                        "which is where the dataset lives locally. "
                        "Colab override: /content/data_local/data_npy")
    p.add_argument("--ckpt_dir",  default="./checkpoints_unet",
                   help="Where to save checkpoints. Default matches the Colab Drive folder name "
                        "(checkpoints_unet) so the same --resume path works on both platforms.")
    p.add_argument("--epochs", type=int, default=50)
    # U-Net holds all 5 encoder skips in VRAM simultaneously, so per-sample
    # memory is higher than the CRN (sequential).  batch=16 keeps peak VRAM
    # under 8 GB; steps_per_epoch is raised to preserve the same clips/epoch
    # as the CRN (270 × 16 = 180 × 24 = 4 320 clips).
    p.add_argument("--batch_size", type=int, default=16)
    _default_workers = min(max(os.cpu_count() or 1, 1) - 1, 8)
    p.add_argument("--num_workers", type=int, default=_default_workers,
                   help="DataLoader workers. Auto = cpu_count-1 capped at 8. "
                        "Use 0 to debug hangs.")
    p.add_argument("--warmup_steps", type=int, default=500,
                   help="Linear LR warmup steps before main schedule (0 = off).")
    p.add_argument("--no_preload_noise", action="store_true",
                   help="Skip pre-loading noise pool into RAM.")
    p.add_argument("--noise_pool_max", type=int, default=4000,
                   help="Cap noise pool to N files (0 = all ~8 GB). "
                        "4000 files ≈ 1 GB; full pool ≈ 8 GB.")
    p.add_argument("--preload_clean", action="store_true", default=True,
                   help="Pre-load clean speech files into RAM — eliminates disk I/O for "
                        "clean files, which is the main bottleneck on Windows. "
                        "6000 files ≈ 1.1 GB extra RAM.")
    p.add_argument("--no_preload_clean", dest="preload_clean", action="store_false",
                   help="Disable clean-file preloading.")
    p.add_argument("--clean_pool_max", type=int, default=6000,
                   help="Cap clean pool to N files (0 = all files). "
                        "6000 ≈ 1.1 GB RAM; higher = more diversity but more RAM. "
                        "Only active when --preload_clean is set.")
    p.add_argument("--rir_pool_size", type=int, default=256,
                   help="Number of synthetic RIRs to pre-build at __init__.")
    p.add_argument("--channels_last", action="store_true", default=False,
                   help="Use channels_last memory format (Conv2d speedup on Ampere+). "
                        "Disabled by default on Windows (causes overhead). "
                        "Enable with --channels_last on Linux with Ampere+ GPU.")
    p.add_argument("--no_channels_last", dest="channels_last", action="store_false",
                   help="Disable channels_last memory format (default on Windows).")
    p.add_argument("--finetune", action="store_true",
                   help="Finetune mode: load model weights only from --resume; "
                        "fresh optim/sched; default lr 2e-5; curriculum off.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--snr_low", type=float, default=-5.0)
    p.add_argument("--snr_high", type=float, default=15.0)
    p.add_argument("--lf_noise_ratio", type=float, default=0.70,
                   help="Fraction of training samples that receive LF noise (<lf_cutoff_hz). "
                        "0.7 = 70%% LF, 30%% sudden HF (clicks/hiss/bursts).")
    p.add_argument("--lf_cutoff_hz", type=float, default=200.0,
                   help="Lowpass cutoff for the LF noise path (Hz). Default 200 Hz.")
    p.add_argument("--segment_seconds", type=float, default=3.0)
    p.add_argument("--sample_rate", type=int, default=16_000)
    # STFT params — must match the model
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--hop_length", type=int, default=128)
    p.add_argument("--win_length", type=int, default=512)
    # Model size knobs
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--mask_bound", type=float, default=1.2,
                   help="Magnitude bound on complex noise mask (tanh-scaled). "
                        "1.2 allows 20%% overshoot vs. noisy magnitude — prevents "
                        "mask inversion where the model predicts speech instead of noise.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default="auto",
                   help="Auto picks bf16 on Blackwell/Ampere+, otherwise fp16.")
    p.add_argument("--grad_clip", type=float, default=3.0,
                   help="Max gradient norm. 3.0 stabilises U-Net layers; 1.0 was too tight for deep skips.")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.")
    p.add_argument("--no_wide_bottleneck", action="store_true",
                   help="Disable extra dilation sweep (1,2,4,8,16). Required when "
                        "resuming old checkpoints trained without wide bottleneck.")
    p.add_argument("--prefetch_factor", type=int, default=4)
    # ---- training-budget knobs -------------------------------------------
    # 270 steps × 16 batch = 4 320 clips/epoch = CRN's 180 × 24 (same budget).
    p.add_argument("--steps_per_epoch", type=int, default=270,
                   help="Cap training steps per epoch (0 = full epoch). "
                        "270 × 16 = 4 320 clips/epoch = CRN's 180 × 24.")
    p.add_argument("--val_every", type=int, default=1,
                   help="Run validation every N epochs (1 = every epoch).")
    p.add_argument("--val_max_batches", type=int, default=20,
                   help="Cap validation to N batches (0 = full val set). "
                        "20 × 16 = 320 clips — enough for stable, low-variance val metrics.")
    # ---- loss weights ---------------------------------------------------
    p.add_argument("--w_l1",           type=float, default=1.0)
    p.add_argument("--w_sc",           type=float, default=0.1)
    p.add_argument("--w_mag",          type=float, default=0.1)
    p.add_argument("--w_sisnr",        type=float, default=1.0)
    p.add_argument("--w_noise",        type=float, default=1.0)
    p.add_argument("--w_sil",          type=float, default=0.1)
    p.add_argument("--w_noise_stft",   type=float, default=0.5)
    p.add_argument("--w_anticollapse", type=float, default=0.03)
    p.add_argument("--w_speech_floor", type=float, default=0.2)
    p.add_argument("--w_perceptual",   type=float, default=0.0,
                   help="Weight for ASR perceptual loss (frozen Wav2Vec2 CNN). "
                        "0=off (default). Disabled by default because loading and "
                        "running Wav2Vec2 on every step is the dominant training bottleneck "
                        "for a 2M-param model. Enable with --w_perceptual 0.05 for "
                        "fine-tuning runs once the model has converged.")
    p.add_argument("--low_freq_boost", type=float, default=5.0,
                   help="STFT loss multiplier for frequency bins ≤ 187.5 Hz (< 200 Hz). "
                        "Applied to both spectral-convergence and log-magnitude L1 terms.")
    # ---- LR scheduler ---------------------------------------------------
    p.add_argument("--scheduler", default="plateau",
                   choices=["cosine", "onecycle", "plateau"],
                   help="plateau = halve LR after 3 epochs no improvement (recommended); "
                        "onecycle = faster early convergence; cosine = stable long runs.")
    p.add_argument("--stft_sched_epoch", type=int, default=20,
                   help="Epoch at which STFT weights are halved (0 = no scheduling). "
                        "Early epochs focus on STFT structure; later epochs let L1 dominate.")
    p.add_argument("--patience", type=int, default=12,
                   help="Early stopping: halt after N epochs with no val improvement (0 = off).")
    p.add_argument("--debug_every", type=int, default=5,
                   help="Save spectrogram debug images every N epochs (0 = off).")
    # ---- curriculum learning ------------------------------------------------
    p.add_argument("--curriculum", action="store_true", default=True,
                   help="Gradually increase difficulty: ramp SNR from --snr_start to --snr_low "
                        "over --curriculum_epochs. Disabled by --finetune.")
    p.add_argument("--no_curriculum", dest="curriculum", action="store_false",
                   help="Disable curriculum (full hard SNR range from epoch 1).")
    p.add_argument("--curriculum_epochs", type=int, default=15,
                   help="Epochs over which SNR ramps from --snr_start to --snr_low.")
    p.add_argument("--snr_start", type=float, default=5.0,
                   help="Initial easy SNR lower bound. Ramped linearly to --snr_low.")
    return p.parse_args()


# ---------------------------------------------------------------------------
def set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
def build_loaders(args):
    seg_len = int(args.segment_seconds * args.sample_rate)

    ds_kw = dict(
        segment_len=seg_len,
        snr_range=(args.snr_low, args.snr_high),
        sample_rate=args.sample_rate,
        rir_pool_size=args.rir_pool_size,
        preload_noise=not args.no_preload_noise,
        noise_pool_max=args.noise_pool_max,
        preload_clean=args.preload_clean,
        clean_pool_max=args.clean_pool_max,
        lf_noise_ratio=args.lf_noise_ratio,
        lf_cutoff_hz=args.lf_cutoff_hz,
    )
    train_ds = SpeechDenoiseDataset(args.data_root, split="train", augment=True,  **ds_kw)
    val_ds   = SpeechDenoiseDataset(args.data_root, split="val",   augment=False, **ds_kw)
    pin = torch.cuda.is_available()

    # On Windows the default spawn context has high per-batch IPC overhead.
    # 'spawn' is still required (no fork on Windows), but naming it explicitly
    # prevents PyTorch from silently choosing a slower fallback on some builds.
    mp_ctx = "spawn" if platform.system() == "Windows" and args.num_workers > 0 else None

    common = dict(
        batch_size=args.batch_size,
        pin_memory=pin,
        pin_memory_device="cuda" if pin else "",
        collate_fn=denoise_collate,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=mp_ctx,
    )
    if args.num_workers > 0:
        common["prefetch_factor"] = args.prefetch_factor
    train_dl = DataLoader(train_ds, shuffle=True,  drop_last=True,  **common)
    val_common = dict(common); val_common["num_workers"] = max(1, args.num_workers // 2)
    val_dl   = DataLoader(val_ds,   shuffle=False, drop_last=False, **val_common)
    return train_dl, val_dl


# ---------------------------------------------------------------------------
def run_epoch(model, loader, loss_fn, device, optim=None, scaler=None,
              amp=True, amp_dtype=torch.float16,
              max_batches: int = 0, grad_accum: int = 1,
              sched_per_batch=None, grad_clip: float = 1.0):
    """One epoch of training or validation.

    model accepts (B, T) waveform and returns (B, T) enhanced waveform.
    loss_fn signature: forward(enhanced, clean, noisy).
    """
    is_train = optim is not None
    model.train(is_train)

    sums = {"loss": 0.0, "l1": 0.0, "sc": 0.0, "mag": 0.0,
            "sisnr": 0.0, "sil": 0.0, "noise_stft": 0.0, "npe": 0.0,
            "speech_floor": 0.0, "perceptual": 0.0}
    n = 0
    if is_train:
        optim.zero_grad(set_to_none=True)

    desc = "train" if is_train else "val  "
    # Use the actual number of steps that will run, not len(loader), so the bar
    # shows 0->100% over the capped epoch rather than 0->3% of the full dataset.
    _total = min(max_batches, len(loader)) if max_batches else len(loader)
    pbar = tqdm(loader, desc=desc, total=_total, dynamic_ncols=True, leave=False, unit="batch")

    for batch_idx, (noisy, clean, _) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            if amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = model(noisy)
                    loss, parts = loss_fn(pred, clean, noisy)
            else:
                pred = model(noisy)
                loss, parts = loss_fn(pred, clean, noisy)

        if is_train:
            loss_acc = loss / grad_accum
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_acc).backward()
            else:
                loss_acc.backward()

            is_last = (max_batches and batch_idx + 1 >= max_batches) or \
                      (batch_idx + 1 >= len(loader))
            if (batch_idx + 1) % grad_accum == 0 or is_last:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optim)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optim.step()
                optim.zero_grad(set_to_none=True)
                if sched_per_batch is not None:
                    sched_per_batch.step()

        b = noisy.size(0)
        loss_val = loss.detach().item()
        sums["loss"]         += loss_val                        * b
        sums["l1"]           += parts["l1"].item()              * b
        sums["sc"]           += parts["sc"].item()              * b
        sums["mag"]          += parts["mag"].item()             * b
        sums["sisnr"]        += parts["sisnr"].item()           * b
        sums["sil"]          += parts["sil"].item()             * b
        sums["noise_stft"]   += parts["noise_stft"].item()      * b
        sums["npe"]          += parts["npe"].item()             * b
        sums["speech_floor"] += parts["speech_floor"].item()    * b
        sums["perceptual"]   += parts["perceptual"].item()      * b
        n += b

        pbar.set_postfix(loss=f"{sums['loss'] / max(1, n):.4f}")

    return {k: v / max(1, n) for k, v in sums.items()}


# ---------------------------------------------------------------------------
def save_debug_spectrograms(model, val_dl, device, epoch, ckpt_dir,
                             n_fft: int = 512, hop_length: int = 128):
    """Save a 2×2 spectrogram grid: noisy / enhanced / clean / predicted noise.

    Verifies visually that noise IS being subtracted.  If 'enhanced ≈ noisy'
    the model is still collapsing; 'predicted noise ≈ 0' confirms it.
    Silently skipped when matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    was_training = model.training
    model.eval()
    with torch.no_grad():
        noisy_b, clean_b, _ = next(iter(val_dl))
        noisy_b = noisy_b[:1].to(device)
        clean_b = clean_b[:1].to(device)
        enhanced_b   = model(noisy_b)
        noise_pred_b = noisy_b - enhanced_b
    if was_training:
        model.train()

    def log_spec(x: torch.Tensor) -> "np.ndarray":
        w = x[0].cpu().float()
        spec = torch.stft(w, n_fft=n_fft, hop_length=hop_length,
                          window=torch.hann_window(n_fft), return_complex=True,
                          center=True)
        return torch.log1p(spec.abs()).numpy()

    panels = [
        ("Noisy Input",      noisy_b),
        ("Enhanced Output",  enhanced_b),
        ("Clean Target",     clean_b),
        ("Predicted Noise",  noise_pred_b),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (title, sig) in zip(axes.flat, panels):
        img = ax.imshow(log_spec(sig), aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"{title}  [epoch {epoch}]", fontsize=10)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Freq bin")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_dir = Path(ckpt_dir) / "spectrograms"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"epoch_{epoch:03d}.png"
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"[debug] spectrogram -> {out_path}")


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.finetune:
        if args.lr == 2e-4:
            args.lr = 2e-5
    set_seed(args.seed)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    try:
        with open(os.path.join(args.ckpt_dir, "args.json"), "w") as fh:
            json.dump(vars(args), fh, indent=2, default=str)
    except Exception as e:
        print(f"[train] could not write args.json: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"[train] device={device}  data_root={args.data_root}")
    print(f"[train] STFT: n_fft={args.n_fft} hop={args.hop_length} win={args.win_length}")

    validate_dataset(args.data_root, "train")
    validate_dataset(args.data_root, "val")

    train_dl, val_dl = build_loaders(args)
    print(f"[train] train batches/epoch: {len(train_dl)}   val: {len(val_dl)}")

    # ----- model -----
    model = AdvancedUNetSE(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        base_channels=args.base_channels,
        depth=args.depth,
        mask_bound=args.mask_bound,
        wide_bottleneck=not args.no_wide_bottleneck,
        sample_rate=args.sample_rate,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] AdvancedUNetSE params: {n_params/1e6:.2f} M  "
          f"mask_bound={args.mask_bound}  (residual noise-prediction mode)")

    # ----- loss -----
    loss_fn = DenoiseLoss(
        w_l1=args.w_l1, w_sc=args.w_sc, w_mag=args.w_mag,
        w_sisnr=args.w_sisnr, w_noise=args.w_noise, w_sil=args.w_sil,
        w_noise_stft=args.w_noise_stft, w_anticollapse=args.w_anticollapse,
        w_speech_floor=args.w_speech_floor, w_perceptual=args.w_perceptual,
        low_freq_boost=args.low_freq_boost, sample_rate=args.sample_rate,
    ).to(device)
    print(f"[train] loss  l1={args.w_l1}  sc={args.w_sc}  mag={args.w_mag}  "
          f"sisnr={args.w_sisnr}  noise={args.w_noise}  sil={args.w_sil}  "
          f"noise_stft={args.w_noise_stft}  anticollapse={args.w_anticollapse}  "
          f"sfloor={args.w_speech_floor}  perc={args.w_perceptual}  "
          f"low_freq_boost={args.low_freq_boost}")

    # ----- channels_last -----
    if args.channels_last and device.type == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)
            print("[train] channels_last memory format enabled")
        except Exception as e:
            print(f"[train] channels_last unavailable ({e})")

    # ----- optimizer -----
    # Fused AdamW collapses the per-param elementwise kernels into one CUDA
    # call — typically 5-10% step-time win on small models.  Falls back to
    # the standard implementation on CPU or older PyTorch (no `fused` kwarg).
    fused_kw = {"fused": True} if device.type == "cuda" else {}
    try:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999),fused=True if device.type == "cuda" else False)
        if fused_kw:
            print("[train] AdamW(fused=True) enabled")
    except TypeError:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999))
        print("[train] fused AdamW unavailable; using default AdamW")

    # ----- scheduler -----
    steps_per_epoch = args.steps_per_epoch or len(train_dl)
    if args.scheduler == "onecycle":
        total_opt_steps = args.epochs * (steps_per_epoch // max(1, args.grad_accum))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=args.lr,
            total_steps=total_opt_steps,
            pct_start=0.3, anneal_strategy="cos",
        )
        print(f"[train] scheduler=OneCycleLR  total_opt_steps={total_opt_steps}")
    elif args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
        print("[train] scheduler=CosineAnnealingLR")
    else:  # plateau
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.5, patience=3, min_lr=1e-6,
        )
        print("[train] scheduler=ReduceLROnPlateau  (factor=0.5  patience=3)")
    _curr_note = f"curriculum: {args.snr_start:.0f}→{args.snr_low:.0f} dB over {args.curriculum_epochs} epochs" \
                 if args.curriculum and not args.finetune else "no curriculum"
    print(f"[train] grad_accum={args.grad_accum}  "
          f"effective batch={args.batch_size * args.grad_accum}  "
          f"snr=[{args.snr_low:.0f},{args.snr_high:.0f}]dB  {_curr_note}")

    # ----- AMP setup -----
    use_amp = (not args.no_amp) and device.type == "cuda"
    if args.amp_dtype == "auto":
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    else:
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler_enabled = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(device="cuda", enabled=scaler_enabled)
    print(f"[train] AMP: {use_amp}  dtype: {amp_dtype}  GradScaler: {scaler_enabled}")

    # ----- resume -----
    start_epoch = 1
    best_sisnr  = float("inf")   # sisnr loss is negated, so lower = better SI-SDR
    epochs_no_improve = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        sd = ck["model"]
        bare = model._orig_mod if hasattr(model, "_orig_mod") else model
        missing, unexpected = bare.load_state_dict(sd, strict=False)
        if missing:
            print(f"[train] resume: {len(missing)} new params initialised from scratch")
        if unexpected:
            print(f"[train] resume: {len(unexpected)} stale keys ignored")
        if not args.finetune:
            try:
                optim.load_state_dict(ck["optim"])
                sched.load_state_dict(ck["sched"])
                if scaler is not None and "scaler" in ck and ck["scaler"] is not None:
                    scaler.load_state_dict(ck["scaler"])
                start_epoch = ck.get("epoch", 0) + 1
                best_sisnr  = ck.get("best_sisnr", ck.get("best_val", best_sisnr))
                print(f"[train] resumed from {args.resume} @ epoch {start_epoch-1}  "
                      f"best_sisnr={best_sisnr:.4f}")
            except Exception as e:
                print(f"[train] resume: optim/sched restore failed ({e}); starting fresh.")
        else:
            print(f"[train] finetune: weights loaded from {args.resume}; "
                  f"fresh optim/sched, lr={args.lr}")
    elif args.resume:
        print(f"[train] WARNING: resume path not found: {args.resume} (starting fresh)")

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, "tb"))
    except Exception:
        tb = None

    sched_per_batch = sched if args.scheduler == "onecycle" else None

    class WarmupWrap:
        def __init__(self, optim, base_lrs, warmup_steps, inner=None):
            self.optim = optim
            self.base_lrs = list(base_lrs)
            self.warmup = max(0, int(warmup_steps))
            self.inner = inner
            self.step_n = 0

        def step(self):
            self.step_n += 1
            if self.warmup > 0 and self.step_n <= self.warmup:
                scale = self.step_n / self.warmup
                for g, lr in zip(self.optim.param_groups, self.base_lrs):
                    g["lr"] = lr * scale
            elif self.inner is not None:
                self.inner.step()

    base_lrs = [g["lr"] for g in optim.param_groups]
    # OneCycleLR manages its own warmup via pct_start=0.3; an external WarmupWrap
    # causes the LR to crash to 8e-6 at step warmup_steps+1 when OneCycleLR's
    # internal first step fires.  Suppress external warmup for onecycle only.
    _warmup = 0 if args.scheduler == "onecycle" else args.warmup_steps
    sched_per_batch = WarmupWrap(optim, base_lrs, _warmup, sched_per_batch) \
        if _warmup > 0 or sched_per_batch is not None else None

    # Clear history.json on a fresh run so plot lines don't overlap with a prior run
    if start_epoch == 1:
        _hist_path = os.path.join(args.ckpt_dir, "history.json")
        if os.path.isfile(_hist_path):
            import shutil as _sh
            _sh.copy2(_hist_path, _hist_path + ".bak")
        with open(_hist_path, "w") as _fh:
            json.dump([], _fh)

    for epoch in range(start_epoch, args.epochs + 1):
        # Curriculum: ramp SNR lower-bound from easy → hard over curriculum_epochs.
        if args.curriculum and not args.finetune:
            if epoch <= args.curriculum_epochs:
                progress    = epoch / args.curriculum_epochs
                curr_snr_lo = args.snr_start + (args.snr_low - args.snr_start) * progress
            else:
                curr_snr_lo = args.snr_low
            train_dl.dataset.snr_range = (curr_snr_lo, args.snr_high)

        # STFT weight schedule: focus structure early, accuracy late
        if args.stft_sched_epoch > 0 and epoch >= args.stft_sched_epoch:
            stft_scale = 0.5
        else:
            stft_scale = 1.0
        loss_fn.w_sc  = args.w_sc  * stft_scale
        loss_fn.w_mag = args.w_mag * stft_scale

        t0 = time.time()
        tr = run_epoch(model, train_dl, loss_fn, device,
                       optim=optim, scaler=scaler, amp=use_amp, amp_dtype=amp_dtype,
                       max_batches=args.steps_per_epoch,
                       grad_accum=args.grad_accum,
                       sched_per_batch=sched_per_batch,
                       grad_clip=args.grad_clip)

        do_val = (args.val_every <= 1) or (epoch % args.val_every == 0) \
                 or (epoch == args.epochs)
        if do_val:
            vl = run_epoch(model, val_dl, loss_fn, device, amp=use_amp, amp_dtype=amp_dtype,
                           max_batches=args.val_max_batches)
        else:
            vl = {k: float("nan") for k in tr}

        if args.scheduler == "cosine":
            sched.step()
        elif args.scheduler == "plateau" and do_val:
            sched.step(vl["loss"])

        dt = time.time() - t0
        current_lr = optim.param_groups[0]["lr"]
        neg_note = "  [SI-SDR>0 ok]" if tr["loss"] < 0 else ""

        print(f"epoch {epoch:03d}/{args.epochs}  "
              f"train={tr['loss']:.4f}  val={vl['loss']:.4f}  "
              f"tr_sisnr={tr['sisnr']:.4f}  tr_npe={tr['npe']:.4f}  "
              f"l1={vl['l1']:.4f}  sc={vl['sc']:.4f}  "
              f"mag={vl['mag']:.4f}  sisnr={vl['sisnr']:.4f}  "
              f"sil={vl['sil']:.4f}  noise_stft={vl['noise_stft']:.4f}  "
              f"npe={vl['npe']:.4f}  sfloor={vl['speech_floor']:.4f}  "
              f"perc={vl['perceptual']:.4f}  "
              f"stft_scale={stft_scale:.1f}  "
              f"lr={current_lr:.2e}  ({dt:.1f}s){neg_note}")

        if tb is not None:
            for k, v in tr.items(): tb.add_scalar(f"train/{k}", v, epoch)
            if do_val:
                for k, v in vl.items(): tb.add_scalar(f"val/{k}", v, epoch)
            tb.add_scalar("lr", current_lr, epoch)
            tb.add_scalar("stft_scale", stft_scale, epoch)

        bare_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        ck = {
            "model":      bare_model.state_dict(),
            "optim":      optim.state_dict(),
            "sched":      sched.state_dict(),
            "scaler":     scaler.state_dict() if scaler_enabled else None,
            "epoch":      epoch,
            "best_sisnr": best_sisnr,
            "args":       vars(args),
            "arch":       "AdvancedUNetSE",
        }
        torch.save(ck, os.path.join(args.ckpt_dir, "last.pt"))

        if do_val and vl["sisnr"] < best_sisnr:
            best_sisnr = vl["sisnr"]
            epochs_no_improve = 0
            ck["best_sisnr"] = best_sisnr
            torch.save(ck, os.path.join(args.ckpt_dir, "best.pt"))
            torch.save({"model": bare_model.state_dict(), "args": vars(args),
                        "epoch": epoch, "arch": "AdvancedUNetSE"},
                       os.path.join(args.ckpt_dir, "best_model_only.pt"))
            _rotate_topk(args.ckpt_dir, best_sisnr, ck, k=3)
            print(f"  >> saved new best (val_sisnr={best_sisnr:.4f}  "
                  f"approx SI-SDR {-20*best_sisnr:.1f} dB)")
        elif do_val:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"[train] early stop at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

        if args.debug_every > 0 and epoch % args.debug_every == 0:
            save_debug_spectrograms(model, val_dl, device, epoch, args.ckpt_dir,
                                    n_fft=args.n_fft, hop_length=args.hop_length)

        # --- Per-epoch history logging (history.json + metrics.csv) ---
        try:
            import math as _math, csv as _csv
            _vl_s = float(vl["sisnr"])
            _clamped = max(-0.9999, min(0.9999, -_vl_s))
            _sisnr_db = 20.0 * _math.atanh(_clamped) if not _math.isnan(_vl_s) else float("nan")
        except Exception:
            _sisnr_db = float("nan")
        _entry = {
            "epoch":          epoch,
            "train_loss":     round(float(tr["loss"]), 6),
            "val_loss":       round(float(vl["loss"]), 6),
            "val_sisnr_loss": round(float(vl["sisnr"]), 6),
            "val_sisnr_db":   round(_sisnr_db, 4) if _sisnr_db == _sisnr_db else None,
        }
        _hist_path = os.path.join(args.ckpt_dir, "history.json")
        try:
            _hist = json.load(open(_hist_path)) if os.path.isfile(_hist_path) else []
            _hist.append(_entry)
            with open(_hist_path, "w") as _fh:
                json.dump(_hist, _fh, indent=2)
        except Exception:
            pass
        _csv_path = os.path.join(args.ckpt_dir, "metrics.csv")
        try:
            _write_hdr = not os.path.isfile(_csv_path)
            with open(_csv_path, "a", newline="") as _fh:
                _wr = _csv.DictWriter(_fh, fieldnames=list(_entry.keys()))
                if _write_hdr:
                    _wr.writeheader()
                _wr.writerow(_entry)
        except Exception:
            pass

    print("[train] done.")


if __name__ == "__main__":
    main()
