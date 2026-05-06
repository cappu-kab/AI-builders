"""
train.py
========
Train the CRN speech-denoiser.

Highlights
----------
* Custom Dataset that does ALL augmentation on-the-fly (see dataset.py).
* Loss = L1(wave) + multi-resolution STFT (spectral-convergence + log-mag).
* Mixed-precision (`torch.cuda.amp`) on GPU.
* Best-on-val and last-epoch checkpoints saved to ./checkpoints.
* TensorBoard scalars (optional — silently skipped if tensorboard absent).

Local usage
-----------
    python train.py --data_root ./data_npy --epochs 30 --batch_size 8

Colab usage
-----------
    !python train.py --data_root /content/data_npy --epochs 30 --batch_size 16
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SpeechDenoiseDataset, denoise_collate, validate_dataset
from losses import DenoiseLoss
from model import CRN


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # Defaults match the Colab fast preset so you can hit VS Code's Run button
    # with no CLI flags.  Project layout assumed:
    #   project/{train,inference,evaluate}.py
    #   project/data_npy/{train,val,test}/{clean,noise}/
    #   project/checkpoints/   <-- created automatically
    p.add_argument("--data_root", default="./data_npy")
    p.add_argument("--ckpt_dir",  default="./checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--snr_low", type=float, default=-5.0,
                   help="Lower SNR bound for training mixtures (dB). "
                        "Narrower range than -20/+20 reduces extreme mixtures "
                        "that destabilise training and cause mask inversion.")
    p.add_argument("--snr_high", type=float, default=15.0,
                   help="Upper SNR bound for training mixtures (dB).")
    p.add_argument("--segment_seconds", type=float, default=3.0)
    p.add_argument("--sample_rate", type=int, default=16_000)
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--hop_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--no_amp", action="store_true")
    # ---- training-budget knobs --------------------------------------------
    # 180 steps x 24 batch = 4,320 clips per epoch (~20% more than original 150).
    p.add_argument("--steps_per_epoch", type=int, default=180,
                   help="Cap training steps per epoch (0 = full epoch).")
    p.add_argument("--val_every", type=int, default=1,
                   help="Run validation every N epochs (1 = every epoch).")
    p.add_argument("--val_max_batches", type=int, default=19,
                   help="Cap validation to N batches (0 = full val set). "
                        "19 x 24 = 456 clips, ~10%% of train budget.")
    p.add_argument("--rnn_hidden", type=int, default=256,
                   help="Hidden size per direction. BiLSTM output = rnn_hidden × 2.")
    p.add_argument("--rnn_layers", type=int, default=2)
    p.add_argument("--rnn_type", default="bilstm", choices=["gru", "lstm", "bilstm"],
                   help="Recurrent core. bilstm = 2-layer bidirectional LSTM (default, "
                        "most powerful). Projections proj_in/proj_out are always kept.")
    p.add_argument("--bottleneck_dim", type=int, default=384)
    p.add_argument("--compile", action="store_true",
                   help="Try torch.compile on the model (PyTorch 2.x).")
    p.add_argument("--no_se", action="store_true",
                   help="Disable SE attention (needed to resume old pre-SE checkpoints).")
    # ---- curriculum learning ------------------------------------------------
    p.add_argument("--curriculum_epochs", type=int, default=12,
                   help="4-phase SNR difficulty ramp over N epochs (0 = off). "
                        "Phase 1: [high,high]  Phase 2: [mid,high]  "
                        "Phase 3: [0,high]  Phase 4+: [low,high].")
    # ---- loss weights -------------------------------------------------------
    # STFT weights default to 0.1 so they don't dominate waveform L1.
    p.add_argument("--w_l1",    type=float, default=1.0,
                   help="Weight for Huber(enhanced, clean). 1:1 with w_noise balances "
                        "speech-quality and noise-removal gradients symmetrically.")
    p.add_argument("--w_sc",    type=float, default=0.1)
    p.add_argument("--w_mag",   type=float, default=0.1)
    p.add_argument("--w_sisnr", type=float, default=0.1,
                   help="Weight for SI-SDR loss — primary guardian of speech integrity.")
    p.add_argument("--w_noise", type=float, default=1.0,
                   help="Weight for noise-prediction Huber loss. Equal to w_l1 for 1:1 "
                        "speech/noise balance. Since Huber(noise_pred, noise_tgt) == "
                        "Huber(enhanced, clean) numerically, combined effective weight is "
                        "w_l1 + w_noise = 2.0.")
    p.add_argument("--w_sil",   type=float, default=0.1,
                   help="Weight for silent-region penalty (suppress residual noise in pauses).")
    p.add_argument("--w_noise_stft", type=float, default=0.5,
                   help="Weight for frequency-weighted spectral noise L1. Halved from 1.0 "
                        "— at 1.0 it caused over-suppression by dominating the gradient.")
    p.add_argument("--w_anticollapse", type=float, default=0.03,
                   help="Weight for anti-collapse penalty. Reduced from 0.1 (which caused "
                        "over-suppression) — 0.03 still prevents identity collapse without "
                        "forcing the model to mute speech.")
    p.add_argument("--w_speech_floor", type=float, default=0.2,
                   help="Weight for speech-floor penalty (fires when enhanced energy < clean "
                        "energy per sample; prevents over-suppression in later epochs).")
    p.add_argument("--w_perceptual", type=float, default=0.05,
                   help="Weight for ASR perceptual loss (frozen Wav2Vec2 CNN feature-matching). "
                        "0 disables the loss entirely. Requires torchaudio.")
    p.add_argument("--low_freq_boost", type=float, default=5.0,
                   help="Frequency-weight multiplier for STFT loss bins ≤ 187.5 Hz (< 200 Hz). "
                        "Applied as a step function to both spectral-convergence and "
                        "log-magnitude L1 sub-terms in MultiResolutionSTFTLoss, and to the "
                        "noise-STFT loss weight tensor. "
                        "Exact cutoff bins at 16 kHz: n_fft=256→bin3, 512→bin6, 1024→bin12.")
    p.add_argument("--mask_bound", type=float, default=1.2,
                   help="Magnitude bound on complex mask (tanh-scaled). "
                        "Bounds |mask| so noise_pred energy <= mask_bound * noisy energy, "
                        "preventing mask inversion where the model learns to predict speech.")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Max gradient norm for clip_grad_norm_. "
                        "1.0 is safe with AMP (float16); use 2.0 for slower convergence. "
                        "Was hardcoded to 5.0 which allowed gradient spikes under AMP.")
    p.add_argument("--dropout_p", type=float, default=0.1,
                   help="Dropout in CRN bottleneck to reduce identity-collapse bias.")
    p.add_argument("--debug_every", type=int, default=5,
                   help="Save spectrogram debug images every N epochs (0 = off).")
    # ---- training efficiency ------------------------------------------------
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.")
    p.add_argument("--scheduler", default="plateau",
                   choices=["cosine", "onecycle", "plateau"],
                   help="LR scheduler. plateau = halve LR after 3 epochs no val improvement "
                        "(best for escaping identity-map local minima); "
                        "onecycle = faster early convergence; cosine = stable long runs.")
    p.add_argument("--stft_sched_epoch", type=int, default=20,
                   help="Epoch at which STFT weights are halved (0 = no scheduling). "
                        "Early epochs focus on STFT structure; later epochs let L1 dominate. "
                        "Raised to 20 (was 10) to keep spectral constraints through the "
                        "early-degradation window observed around epoch 15.")
    p.add_argument("--patience", type=int, default=12,
                   help="Early stopping: halt after N consecutive epochs with no val improvement "
                        "(0 = disabled). Best checkpoint is always saved regardless.")
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
      → model only sees high-SNR mixtures [≥5, snr_high]: learns basic structure
    Phase 2 (middle 1/3): clamp low to ≥ 0 dB, high to ≤ 10 dB
      → medium difficulty [≥0, ≤10]: speech is still dominant
    Phase 3 (final 1/3 and epoch > curriculum_epochs): full range [snr_low, snr_high]
      → hardest mixtures; model already knows what speech looks like

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
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=pin,
                          drop_last=True, collate_fn=denoise_collate,
                          persistent_workers=args.num_workers > 0)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=max(1, args.num_workers // 2), pin_memory=pin,
                          drop_last=False, collate_fn=denoise_collate,
                          persistent_workers=args.num_workers > 0)
    return train_dl, val_dl


# ---------------------------------------------------------------------------
def run_epoch(model, loader, loss_fn, device, optim=None, scaler=None,
              amp=True, max_batches: int = 0, grad_accum: int = 1,
              sched_per_batch=None, grad_clip: float = 1.0):
    is_train = optim is not None
    model.train(is_train)
    sums = {"loss": 0.0, "l1": 0.0, "sc": 0.0, "mag": 0.0,
            "sisnr": 0.0, "sil": 0.0, "noise_stft": 0.0, "npe": 0.0,
            "speech_floor": 0.0, "perceptual": 0.0}
    n = 0
    if is_train:
        optim.zero_grad(set_to_none=True)

    for batch_idx, (noisy, clean, _) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            if amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    pred = model(noisy)
                    loss, parts = loss_fn(pred, clean, noisy)
            else:
                pred = model(noisy)
                loss, parts = loss_fn(pred, clean, noisy)

        if is_train:
            # scale loss for accumulation so effective gradient == full batch
            loss_acc = loss / grad_accum
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_acc).backward()
            else:
                loss_acc.backward()

            # optimizer step every grad_accum batches (or at the final batch)
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
        sums["loss"]  += float(loss)            * b
        sums["l1"]    += float(parts["l1"])     * b
        sums["sc"]    += float(parts["sc"])     * b
        sums["mag"]   += float(parts["mag"])    * b
        sums["sisnr"]      += float(parts["sisnr"])      * b
        sums["sil"]        += float(parts["sil"])        * b
        sums["noise_stft"]   += float(parts["noise_stft"])   * b
        sums["npe"]          += float(parts["npe"])           * b
        sums["speech_floor"] += float(parts["speech_floor"]) * b
        sums["perceptual"]   += float(parts["perceptual"])   * b
        n += b

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
        torch.backends.cudnn.benchmark = True   # autotune conv kernels
    print(f"[train] device={device}  data_root={args.data_root}")
    validate_dataset(args.data_root, "train")
    validate_dataset(args.data_root, "val")

    train_dl, val_dl = build_loaders(args)
    print(f"[train] train batches/epoch: {len(train_dl)}   val: {len(val_dl)}")

    model = CRN(n_fft=args.n_fft, hop_length=args.hop_length,
                rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers,
                rnn_type=args.rnn_type, bottleneck_dim=args.bottleneck_dim,
                mask_bound=args.mask_bound, dropout_p=args.dropout_p,
                use_se=not args.no_se).to(device)
    if args.compile:
        try:
            model = torch.compile(model)
            print("[train] torch.compile enabled")
        except Exception as e:
            print(f"[train] torch.compile unavailable ({e})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.2f} M")
    loss_fn = DenoiseLoss(
        w_l1=args.w_l1, w_sc=args.w_sc,
        w_mag=args.w_mag, w_sisnr=args.w_sisnr,
        w_noise=args.w_noise, w_sil=args.w_sil,
        w_noise_stft=args.w_noise_stft, w_anticollapse=args.w_anticollapse,
        w_speech_floor=args.w_speech_floor, w_perceptual=args.w_perceptual,
        low_freq_boost=args.low_freq_boost, sample_rate=args.sample_rate,
    ).to(device)
    print(f"[train] loss weights  l1={args.w_l1}  sc={args.w_sc}  "
          f"mag={args.w_mag}  sisnr={args.w_sisnr}  "
          f"noise={args.w_noise}  sil={args.w_sil}  "
          f"noise_stft={args.w_noise_stft}  anticollapse={args.w_anticollapse}  "
          f"speech_floor={args.w_speech_floor}  perceptual={args.w_perceptual}  "
          f"low_freq_boost={args.low_freq_boost}")
    print(f"[train] mask_bound={args.mask_bound}  dropout_p={args.dropout_p}  "
          f"(residual noise-prediction mode)")
    print(f"[train] grad_accum={args.grad_accum}  scheduler={args.scheduler}")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, betas=(0.9, 0.999))

    steps_per_epoch = args.steps_per_epoch or len(train_dl)
    if args.scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=args.lr,
            total_steps=args.epochs * (steps_per_epoch // args.grad_accum),
            pct_start=0.3, anneal_strategy="cos",
        )
    elif args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    else:  # plateau
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.5, patience=3, min_lr=1e-6,
        )

    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_sisnr = float("inf")   # sisnr loss is negated, so lower = better SI-SDR
    epochs_no_improve = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if missing:
            print(f"[train] resume: {len(missing)} new params initialised from scratch "
                  f"(e.g. SE blocks) — normal when adding SE to an old checkpoint.")
        if unexpected:
            print(f"[train] resume: {len(unexpected)} stale keys ignored.")
        optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck.get("epoch", 0) + 1
        best_sisnr = ck.get("best_sisnr", ck.get("best_val", best_sisnr))
        print(f"[train] resumed from {args.resume} @ epoch {start_epoch-1}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, "tb"))
    except Exception:
        tb = None

    sched_per_batch = sched if args.scheduler == "onecycle" else None

    for epoch in range(start_epoch, args.epochs + 1):
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
                       optim=optim, scaler=scaler, amp=use_amp,
                       max_batches=args.steps_per_epoch,
                       grad_accum=args.grad_accum,
                       sched_per_batch=sched_per_batch,
                       grad_clip=args.grad_clip)
        # validate every N epochs (and on the last) to save GPU time
        do_val = (args.val_every <= 1) or (epoch % args.val_every == 0) \
                 or (epoch == args.epochs)
        if do_val:
            vl = run_epoch(model, val_dl, loss_fn, device, amp=use_amp,
                           max_batches=args.val_max_batches)
        else:
            vl = {"loss": float("nan"), "l1": 0.0, "sc": 0.0, "mag": 0.0,
                  "sisnr": 0.0, "sil": 0.0, "noise_stft": 0.0, "npe": 0.0,
                  "speech_floor": 0.0, "perceptual": 0.0}
        if args.scheduler == "cosine":
            sched.step()
        elif args.scheduler == "plateau" and do_val:
            sched.step(vl["loss"])
        dt = time.time() - t0

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
              f"({dt:.1f}s){neg_warn}")

        if tb is not None:
            for k, v in tr.items(): tb.add_scalar(f"train/{k}", v, epoch)
            if do_val:
                for k, v in vl.items(): tb.add_scalar(f"val/{k}", v, epoch)
            current_lr = sched.get_last_lr()[0] if hasattr(sched, "get_last_lr") \
                         else optim.param_groups[0]["lr"]
            tb.add_scalar("lr", current_lr, epoch)
            tb.add_scalar("stft_scale", stft_scale, epoch)

        # state_dict needs to come from the un-compiled module
        sd = (model._orig_mod.state_dict()
              if hasattr(model, "_orig_mod") else model.state_dict())
        ck = {
            "model":      sd,
            "optim":      optim.state_dict(),
            "sched":      sched.state_dict(),
            "epoch":      epoch,
            "best_sisnr": best_sisnr,
            "args":       vars(args),
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
