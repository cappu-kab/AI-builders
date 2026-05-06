"""
train.py
========
Train the AdvancedUNetSE speech-denoiser.

Architecture: pure U-Net with residual noise prediction.
    enhanced = noisy_wave − iSTFT( mask ⊙ STFT(noisy_wave) )

The model accepts raw waveforms (B, T) and returns enhanced waveforms (B, T).
STFT/iSTFT live inside the model; no STFTFront module needed here.

Training infrastructure is synchronized with the CRN project:
- On-the-fly augmentation (brown/pink noise, biased SNR, RIR, etc.)
- 10-term DenoiseLoss with 5x low-frequency boost < 200 Hz
- 3-phase SNR curriculum learning
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

from dataset import SpeechDenoiseDataset, denoise_collate, validate_dataset
from losses import DenoiseLoss
from model_unet_advanced import AdvancedUNetSE


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data_npy")
    p.add_argument("--ckpt_dir",  default="./checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    # U-Net holds all 5 encoder skips in VRAM simultaneously, so per-sample
    # memory is higher than the CRN (sequential).  batch=16 keeps peak VRAM
    # under 8 GB; steps_per_epoch is raised to preserve the same clips/epoch
    # as the CRN (270 × 16 = 180 × 24 = 4 320 clips).
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--snr_low", type=float, default=-5.0)
    p.add_argument("--snr_high", type=float, default=15.0)
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
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Max gradient norm. 1.0 is safe with AMP; use 2.0 for slower convergence.")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable torch.compile (use if you hit Triton/Inductor errors).")
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
    p.add_argument("--val_max_batches", type=int, default=29,
                   help="Cap validation to N batches (0 = full val set). "
                        "29 × 16 = 464 clips (~10%% of train budget, same as CRN's 19 × 24).")
    # ---- curriculum learning --------------------------------------------
    p.add_argument("--curriculum_epochs", type=int, default=12,
                   help="3-phase SNR difficulty ramp over N epochs (0 = off). "
                        "Phase 1: [≥5,high]  Phase 2: [≥0,≤10]  Phase 3: [low,high].")
    # ---- loss weights ---------------------------------------------------
    p.add_argument("--w_l1",           type=float, default=1.0)
    p.add_argument("--w_sc",           type=float, default=0.1)
    p.add_argument("--w_mag",          type=float, default=0.1)
    p.add_argument("--w_sisnr",        type=float, default=0.1)
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
    return p.parse_args()


# ---------------------------------------------------------------------------
def set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
def _curriculum_snr_range(epoch: int, snr_low: float, snr_high: float,
                           curriculum_epochs: int) -> "tuple[float, float]":
    """3-phase SNR curriculum — slides both bounds from easy to hard.

    Phase 1 (first 1/3 of curriculum_epochs): clamp low to ≥ +5 dB
    Phase 2 (middle 1/3): clamp low to ≥ 0 dB, high to ≤ 10 dB
    Phase 3 (final 1/3 and epoch > curriculum_epochs): full range [snr_low, snr_high]

    Example with defaults snr_low=-5, snr_high=15, curriculum_epochs=12:
      Epochs  1-4  →  [5, 15]  easy
      Epochs  5-8  →  [0, 10]  medium
      Epochs  9+   →  [-5, 15] full difficulty
    """
    if curriculum_epochs <= 0 or epoch > curriculum_epochs:
        return snr_low, snr_high
    t = epoch / curriculum_epochs
    if t <= 1 / 3:
        return max(snr_low, 5.0), snr_high
    elif t <= 2 / 3:
        return max(snr_low, 0.0), min(snr_high, 10.0)
    else:
        return snr_low, snr_high


# ---------------------------------------------------------------------------
def build_loaders(args):
    seg_len = int(args.segment_seconds * args.sample_rate)

    train_ds = SpeechDenoiseDataset(
        args.data_root, split="train",
        segment_len=seg_len,
        snr_range=(args.snr_low, args.snr_high),
        augment=True,
        sample_rate=args.sample_rate,
    )
    val_ds = SpeechDenoiseDataset(
        args.data_root, split="val",
        segment_len=seg_len,
        snr_range=(args.snr_low, args.snr_high),
        augment=False,
        sample_rate=args.sample_rate,
    )
    pin = torch.cuda.is_available()
    common = dict(
        batch_size=args.batch_size, pin_memory=pin,
        collate_fn=denoise_collate,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
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
    print(f"[debug] spectrogram → {out_path}")


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

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

    # ----- torch.compile -----
    if not args.no_compile and device.type == "cuda":
        try:
            # model = torch.compile(model, mode="default", dynamic=False)
            print("[train] torch.compile enabled (mode=default)")
        except Exception as e:
            print(f"[train] torch.compile failed ({e}); running uncompiled")

    # ----- optimizer -----
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, betas=(0.9, 0.999))

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
    print(f"[train] grad_accum={args.grad_accum}  "
          f"effective batch={args.batch_size * args.grad_accum}  "
          f"curriculum_epochs={args.curriculum_epochs}")

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
        optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        if scaler is not None and "scaler" in ck and ck["scaler"] is not None:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck.get("epoch", 0) + 1
        best_sisnr  = ck.get("best_sisnr", ck.get("best_val", best_sisnr))
        print(f"[train] resumed from {args.resume} @ epoch {start_epoch-1}  "
              f"best_sisnr={best_sisnr:.4f}")
    elif args.resume:
        print(f"[train] WARNING: resume path not found: {args.resume} (starting fresh)")

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, "tb"))
    except Exception:
        tb = None

    sched_per_batch = sched if args.scheduler == "onecycle" else None

    for epoch in range(start_epoch, args.epochs + 1):
        # Curriculum: update dataset SNR range each epoch
        cur_snr_low, cur_snr_high = _curriculum_snr_range(
            epoch, args.snr_low, args.snr_high, args.curriculum_epochs)
        train_dl.dataset.snr_low  = cur_snr_low
        train_dl.dataset.snr_high = cur_snr_high

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
        neg_warn = "  *** NEGATIVE LOSS ***" if tr["loss"] < 0 else ""

        print(f"epoch {epoch:03d}/{args.epochs}  "
              f"train={tr['loss']:.4f}  val={vl['loss']:.4f}  "
              f"tr_sisnr={tr['sisnr']:.4f}  tr_npe={tr['npe']:.4f}  "
              f"l1={vl['l1']:.4f}  sc={vl['sc']:.4f}  "
              f"mag={vl['mag']:.4f}  sisnr={vl['sisnr']:.4f}  "
              f"sil={vl['sil']:.4f}  noise_stft={vl['noise_stft']:.4f}  "
              f"npe={vl['npe']:.4f}  sfloor={vl['speech_floor']:.4f}  "
              f"perc={vl['perceptual']:.4f}  "
              f"stft_scale={stft_scale:.1f}  "
              f"snr=[{cur_snr_low:.0f},{cur_snr_high:.0f}]dB  "
              f"lr={current_lr:.2e}  ({dt:.1f}s){neg_warn}")

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
            print(f"  ↳ saved new best (val_sisnr={best_sisnr:.4f}  "
                  f"≈ SI-SDR {-20*best_sisnr:.1f} dB)")
        elif do_val:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"[train] early stop at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

        if args.debug_every > 0 and epoch % args.debug_every == 0:
            save_debug_spectrograms(model, val_dl, device, epoch, args.ckpt_dir,
                                    n_fft=args.n_fft, hop_length=args.hop_length)

    print("[train] done.")


if __name__ == "__main__":
    main()
