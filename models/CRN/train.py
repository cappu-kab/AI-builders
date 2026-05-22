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

from dataset import SpeechDenoiseDataset, denoise_collate, validate_dataset, worker_init_fn
from losses import DenoiseLoss
from model import CRN

import json
import glob


def _rotate_topk(ckpt_dir: str, score: float, payload: dict, k: int = 3) -> None:
    """Keep top-K best checkpoints by score (lower is better; matches sisnr-loss).

    Files are written as best_topN.pt where N=1 is the absolute best.
    Score is encoded in companion `.score` files for next-run discovery.

    Two-phase rename to avoid overwriting (e.g. when new best displaces top1):
      1) Move existing best_top*.pt to .rotate temp paths.
      2) Place keepers at final ranks; remove the dropped one.
    """
    pattern = os.path.join(ckpt_dir, "best_top*.pt")
    existing = []  # (score, ckpt_path, score_path)
    for p in sorted(glob.glob(pattern)):
        sp = p + ".score"
        if os.path.isfile(sp):
            try:
                existing.append((float(open(sp).read().strip()), p, sp))
            except Exception:
                pass
    # Phase 1: move all existing files to .rotate temps
    tmps = []  # (score, ckpt_tmp, score_tmp)
    for sc, ck_p, sc_p in existing:
        ck_t = ck_p + ".rotate"
        sc_t = sc_p + ".rotate"
        try:
            os.replace(ck_p, ck_t)
            os.replace(sc_p, sc_t)
            tmps.append((sc, ck_t, sc_t))
        except Exception:
            pass
    # Phase 2: rank and place
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
    # Defaults match the Colab fast preset so you can hit VS Code's Run button
    # with no CLI flags.  Project layout assumed:
    #   project/{train,inference,evaluate}.py
    #   project/data_npy/{train,val,test}/{clean,noise}/
    #   project/checkpoints/   <-- created automatically
    p.add_argument("--data_root", default="./data_npy")
    p.add_argument("--ckpt_dir",  default="./checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="DataLoader prefetch_factor per worker.")
    p.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default="auto",
                   help="AMP dtype. auto = bf16 on Ampere+/Blackwell, fp16 elsewhere.")
    p.add_argument("--warmup_steps", type=int, default=500,
                   help="Linear LR warmup steps before main schedule (0 = off).")
    p.add_argument("--no_preload_noise", action="store_true",
                   help="Skip pre-loading noise pool into RAM (saves memory but "
                        "re-reads .npy from disk every __getitem__).")
    p.add_argument("--rir_pool_size", type=int, default=256,
                   help="Number of synthetic RIRs to pre-build at __init__.")
    p.add_argument("--channels_last", action="store_true", default=False,
                   help="Use channels_last memory format (helps Conv2d on Ampere+).")
    p.add_argument("--finetune", action="store_true",
                   help="Finetune mode: load model weights only from --resume, "
                        "fresh optimizer/scheduler, lower default lr (2e-5), "
                        "disable curriculum.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--snr_low", type=float, default=-5.0,
                   help="Lower SNR bound for training mixtures (dB). "
                        "Used for both train and val to ensure consistency.")
    p.add_argument("--snr_high", type=float, default=15.0,
                   help="Upper SNR bound for training mixtures (dB).")
    p.add_argument("--lf_noise_ratio", type=float, default=0.80,
                   help="Fraction of val batches that receive the LF+HF blend "
                        "(HVAC, fan, 50/60 Hz hum). Train always blends LF+HF; "
                        "0.8 = 80%% of val samples get the LF-heavy distribution.")
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
    p.add_argument("--val_max_batches", type=int, default=20,
                   help="Cap validation to N batches (0 = full val set). "
                        "20 x 24 = 480 clips — enough for stable, low-variance val metrics.")
    p.add_argument("--rnn_hidden", type=int, default=256,
                   help="Hidden size per direction. BiLSTM output = rnn_hidden × 2.")
    p.add_argument("--rnn_layers", type=int, default=2)
    p.add_argument("--rnn_type", default="bilstm", choices=["gru", "lstm", "bilstm"],
                   help="Recurrent core. bilstm = 2-layer bidirectional LSTM (default, "
                        "most powerful). Projections proj_in/proj_out are always kept.")
    p.add_argument("--bottleneck_dim", type=int, default=512)
    p.add_argument("--base_channels", type=int, default=32,
                   help="Base channel count for CRN encoder. 5 blocks double it: "
                        "base*1 → base*2 → ... → base*16 at bottleneck. "
                        "32 gives [32,64,128,256,512] channels vs legacy [16,32,64,128,256].")
    p.add_argument("--compile", action="store_true",
                   help="Try torch.compile on the model (PyTorch 2.x).")
    p.add_argument("--no_se", action="store_true",
                   help="Disable SE attention (needed to resume old pre-SE checkpoints).")
    # ---- loss weights -------------------------------------------------------
    p.add_argument("--w_sisnr", type=float, default=1.0,
                   help="Weight for tanh-smoothed SI-SDR loss.")
    p.add_argument("--w_mag",   type=float, default=0.5,
                   help="Weight for multi-resolution log-freq compressed MagMSE.")
    p.add_argument("--w_aux",   type=float, default=0.3,
                   help="Weight for auxiliary multi-resolution STFT loss (spectral convergence + log-mag L1). "
                        "Raised to 0.3 to increase speech-preservation pressure alongside the new LF noise term.")
    p.add_argument("--w_lf_noise_stft", type=float, default=0.5,
                   help="Weight for LF-weighted noise prediction MagL1 loss (Term 4). "
                        "Directly supervises the BiLSTM noise estimate below 200 Hz so LF leakage "
                        "produces large gradients before any other frequency band.")
    p.add_argument("--w_speech_floor", type=float, default=0.2,
                   help="Weight for LF speech floor penalty (Term 5): relu(E_lf[clean]-E_lf[enhanced]) "
                        "in the 80-300 Hz voiced fundamental band. Prevents the model from "
                        "over-suppressing speech vowels while aggressively hunting for hum/rumble.")
    p.add_argument("--lf_peak", type=float, default=6.0,
                   help="LF weight multiplier: W(f)=lf_peak below lf_cutoff, cosine taper to 1x. "
                        "6.0x forces heavy gradient pressure on sub-200 Hz noise in both "
                        "the MagMSE (speech path) and LFWeightedNoiseMSE (noise path) terms.")
    p.add_argument("--lf_cutoff", type=float, default=200.0,
                   help="Frequency (Hz) where LF weighting begins its cosine taper to 1×.")
    p.add_argument("--lf_transition", type=float, default=100.0,
                   help="Width (Hz) of the cosine taper band (lf_cutoff → lf_cutoff+lf_transition).")
    p.add_argument("--mask_bound", type=float, default=1.2,
                   help="Magnitude bound on complex mask (tanh-scaled). "
                        "Bounds |mask| so noise_pred energy <= mask_bound * noisy energy, "
                        "preventing mask inversion where the model learns to predict speech.")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Max gradient norm for clip_grad_norm_. "
                        "1.0 prevents runaway gradients in BiLSTM layers and keeps the loss surface stable.")
    p.add_argument("--dropout_p", type=float, default=0.2,
                   help="Dropout in CRN bottleneck to reduce identity-collapse bias. "
                        "0.2 is appropriate for 2-layer BiLSTM (512 dim) — 0.1 under-regularises.")
    p.add_argument("--debug_every", type=int, default=5,
                   help="Save spectrogram debug images every N epochs (0 = off).")
    # ---- training efficiency ------------------------------------------------
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.")
    p.add_argument("--scheduler", default="onecycle",
                   choices=["cosine", "onecycle", "plateau", "warmrestarts"],
                   help="LR scheduler. onecycle = starts low, peaks, decays — best for escaping "
                        "local minima after stabilisation; warmrestarts = periodic restarts; "
                        "cosine = smooth single-cycle decay; plateau = halve on stagnation.")
    p.add_argument("--cosine_T0", type=int, default=15,
                   help="Restart period in epochs for CosineAnnealingWarmRestarts. "
                        "E.g. 15 restarts at epoch 15, 30, 45 … — jolts LR back to max "
                        "so the model can escape local minima at each restart.")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping: halt after N consecutive epochs with no val improvement "
                        "(0 = disabled). Best checkpoint is always saved regardless.")
    # ---- curriculum learning ------------------------------------------------
    p.add_argument("--curriculum", action="store_true", default=True,
                   help="Gradually increase difficulty: start from --snr_start and ramp to "
                        "--snr_low over --curriculum_epochs. Disabled by --finetune.")
    p.add_argument("--no_curriculum", dest="curriculum", action="store_false",
                   help="Disable curriculum (full hard SNR range from epoch 1).")
    p.add_argument("--curriculum_epochs", type=int, default=15,
                   help="Number of epochs over which SNR ramps from --snr_start to --snr_low.")
    p.add_argument("--snr_start", type=float, default=5.0,
                   help="Initial (easy) SNR lower bound for curriculum. "
                        "Ramped linearly to --snr_low over --curriculum_epochs.")
    p.add_argument("--audio_every", type=int, default=5,
                   help="Save noisy+enhanced audio sample pairs every N epochs to "
                        "--results_dir/audio/. 0 = disabled.")
    p.add_argument("--results_dir", default="./results",
                   help="Output directory for audio samples and loss-history plots.")
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
        lf_noise_ratio=args.lf_noise_ratio,
    )
    train_ds = SpeechDenoiseDataset(args.data_root, split="train", augment=True,  **ds_kw)
    val_ds   = SpeechDenoiseDataset(args.data_root, split="val",   augment=False, **ds_kw)
    pin = torch.cuda.is_available()
    common = dict(
        pin_memory=pin,
        collate_fn=denoise_collate,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    if args.num_workers > 0:
        common["prefetch_factor"] = args.prefetch_factor
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True, **common)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=max(1, args.num_workers // 2),
                          drop_last=False, **common)
    return train_dl, val_dl


# ---------------------------------------------------------------------------
def run_epoch(model, loader, loss_fn, device, optim=None, scaler=None,
              amp=True, amp_dtype: torch.dtype = torch.float16,
              max_batches: int = 0, grad_accum: int = 1,
              sched_per_batch=None, grad_clip: float = 1.0,
              channels_last: bool = False):
    is_train = optim is not None
    model.train(is_train)
    sums = {"loss": 0.0, "sisnr": 0.0, "sisnr_db": 0.0,
            "mag": 0.0, "aux_stft": 0.0,
            "lf_noise": 0.0, "speech_floor": 0.0}
    n = 0
    if is_train:
        optim.zero_grad(set_to_none=True)

    for batch_idx, (noisy, clean, _) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        if channels_last and noisy.dim() == 4:
            noisy = noisy.contiguous(memory_format=torch.channels_last)

        with torch.set_grad_enabled(is_train):
            if amp and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=amp_dtype):
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
        sums["loss"]         += float(loss)                     * b
        sums["sisnr"]        += float(parts["sisnr"])           * b
        sums["sisnr_db"]     += float(parts["sisnr_db"])        * b
        sums["mag"]          += float(parts["mag"])             * b
        sums["aux_stft"]     += float(parts["aux_stft"])        * b
        sums["lf_noise"]     += float(parts["lf_noise"])        * b
        sums["speech_floor"] += float(parts["speech_floor"])    * b
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
def save_audio_samples(model, val_dl, device, epoch: int, results_dir: str,
                       sample_rate: int = 16_000, n_samples: int = 2) -> None:
    """Save noisy-input vs enhanced-output wav pairs for subjective listening.

    Tries audio backends in order: soundfile → torchaudio → scipy.
    Writes to  results_dir/audio/ep{epoch}_s{i}_noisy.wav
               results_dir/audio/ep{epoch}_s{i}_enhanced.wav

    Silently skipped if no audio backend is available.
    """
    try:
        import soundfile as _sf
        def _write(w, p): _sf.write(str(p), w.astype(np.float32), sample_rate)
    except ImportError:
        try:
            import torchaudio as _ta
            def _write(w, p): _ta.save(str(p), torch.from_numpy(w.astype(np.float32)).unsqueeze(0), sample_rate)
        except ImportError:
            try:
                from scipy.io import wavfile as _wv
                def _write(w, p):
                    pcm = np.clip(w * 32767, -32768, 32767).astype(np.int16)
                    _wv.write(str(p), sample_rate, pcm)
            except ImportError:
                print("[audio] no audio backend (soundfile / torchaudio / scipy); skipping.")
                return

    out_dir = Path(results_dir) / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        noisy_b, clean_b, _ = next(iter(val_dl))
        noisy_b    = noisy_b[:n_samples].to(device)
        enhanced_b = model(noisy_b)
    if was_training:
        model.train()

    saved = 0
    for i in range(min(n_samples, noisy_b.shape[0])):
        _write(noisy_b[i].cpu().float().numpy(),
               out_dir / f"ep{epoch:03d}_s{i+1}_noisy.wav")
        _write(enhanced_b[i].cpu().float().numpy(),
               out_dir / f"ep{epoch:03d}_s{i+1}_enhanced.wav")
        saved += 1
    print(f"[audio] {saved} pairs saved → {out_dir}")


# ---------------------------------------------------------------------------
def plot_training_history(ckpt_dir: str, epoch: int) -> None:
    """Save a 2-panel Train/Val loss + SI-SDR history plot to ckpt_dir/plots/.

    Reads history.json which is appended every epoch by the training loop.
    Silently skipped when matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    hist_path = os.path.join(ckpt_dir, "history.json")
    if not os.path.isfile(hist_path):
        return
    try:
        with open(hist_path) as fh:
            hist = json.load(fh)
    except Exception:
        return
    if not hist:
        return

    epochs     = [e["epoch"]                        for e in hist]
    train_loss = [e["train_loss"]                    for e in hist]
    val_loss   = [e.get("val_loss", float("nan"))    for e in hist]
    sisnr_db   = [e.get("val_sisnr_db")              for e in hist]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train loss",  linewidth=1.5)
    ax.plot(epochs, val_loss,   label="Val loss",    linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch");  ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss")
    ax.legend();  ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    valid = [(e, s) for e, s in zip(epochs, sisnr_db) if s is not None]
    if valid:
        ep_s, si_s = zip(*valid)
        ax2.plot(ep_s, si_s, color="green", linewidth=1.5, label="Val SI-SDR (dB)")
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Epoch");  ax2.set_ylabel("SI-SDR (dB)")
    ax2.set_title("Validation SI-SDR")
    ax2.legend();  ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Training History — epoch {epoch}", fontsize=12)
    fig.tight_layout()

    plots_dir = os.path.join(ckpt_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"loss_history_ep{epoch:03d}.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


# ---------------------------------------------------------------------------
def _resolve_amp_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    # auto: prefer bf16 on Ampere+ / Blackwell
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main():
    args = parse_args()
    if args.finetune:
        if args.lr == 2e-4:
            args.lr = 2e-5
    set_seed(args.seed)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Snapshot args for reproducibility
    try:
        with open(os.path.join(args.ckpt_dir, "args.json"), "w") as fh:
            json.dump(vars(args), fh, indent=2, default=str)
    except Exception as e:
        print(f"[train] could not write args.json: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # autotune conv kernels
    amp_dtype = _resolve_amp_dtype(args.amp_dtype)
    print(f"[train] device={device}  data_root={args.data_root}  "
          f"amp_dtype={str(amp_dtype).split('.')[-1]}")
    validate_dataset(args.data_root, "train")
    validate_dataset(args.data_root, "val")

    train_dl, val_dl = build_loaders(args)
    print(f"[train] train batches/epoch: {len(train_dl)}   val: {len(val_dl)}")

    model = CRN(n_fft=args.n_fft, hop_length=args.hop_length,
                base_channels=args.base_channels,
                rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers,
                rnn_type=args.rnn_type, bottleneck_dim=args.bottleneck_dim,
                mask_bound=args.mask_bound, dropout_p=args.dropout_p,
                use_se=not args.no_se,
                sample_rate=args.sample_rate).to(device)
    if args.channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
            print("[train] channels_last memory format enabled")
        except Exception as e:
            print(f"[train] channels_last unavailable ({e})")
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[train] torch.compile(mode=reduce-overhead) enabled")
        except Exception as e:
            print(f"[train] torch.compile unavailable ({e})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.2f} M")
    loss_fn = DenoiseLoss(
        w_sisnr=args.w_sisnr, w_mag=args.w_mag, w_aux=args.w_aux,
        w_lf_noise_stft=args.w_lf_noise_stft,
        w_speech_floor=args.w_speech_floor,
        lf_peak=args.lf_peak, lf_cutoff_hz=args.lf_cutoff,
        transition_hz=args.lf_transition, sample_rate=args.sample_rate,
    ).to(device)
    print(f"[train] loss weights  sisnr={args.w_sisnr}  mag={args.w_mag}  aux={args.w_aux}  "
          f"lf_noise_stft={args.w_lf_noise_stft}  speech_floor={args.w_speech_floor}  "
          f"lf_peak={args.lf_peak}x  lf_cutoff={args.lf_cutoff}Hz  "
          f"lf_transition={args.lf_transition}Hz")
    print(f"[train] mask_bound={args.mask_bound}  dropout_p={args.dropout_p}  "
          f"(residual noise-prediction mode)")
    print(f"[train] grad_accum={args.grad_accum}  scheduler={args.scheduler}")
    # Split param groups: LSTM layers get 0.5× LR (multiplicative gating is
    # ~2× more sensitive to weight changes than Conv layers at the same LR).
    _rnn_ids = {id(p) for p in model.rnn.parameters()}
    _rnn_params  = [p for p in model.parameters() if id(p) in _rnn_ids]
    _conv_params = [p for p in model.parameters() if id(p) not in _rnn_ids]
    optim = torch.optim.AdamW(
        [
            {"params": _conv_params, "lr": args.lr},
            {"params": _rnn_params,  "lr": args.lr * 0.5, "weight_decay": args.weight_decay * 2},
        ],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98),
    )

    steps_per_epoch = args.steps_per_epoch or len(train_dl)
    if args.scheduler == "onecycle":
        # Pass per-group max_lr so the LSTM group peaks at 0.5× conv group LR.
        _max_lrs = [g["lr"] for g in optim.param_groups]
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=_max_lrs,
            total_steps=args.epochs * (steps_per_epoch // args.grad_accum),
            pct_start=0.3, anneal_strategy="cos",
        )
    elif args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    elif args.scheduler == "warmrestarts":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=args.cosine_T0, T_mult=1, eta_min=1e-6,
        )
    else:  # plateau
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.5, patience=3, min_lr=1e-6,
        )

    use_amp = (not args.no_amp) and device.type == "cuda"
    # GradScaler only needed for fp16. bf16 has fp32-equivalent dynamic range.
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    start_epoch = 1
    best_sisnr = float("-inf")   # now tracking SI-SDR dB directly; higher = better
    epochs_no_improve = 0

    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        # un-compiled model receives state
        target = model._orig_mod if hasattr(model, "_orig_mod") else model
        missing, unexpected = target.load_state_dict(ck["model"], strict=False)
        if missing:
            print(f"[train] resume: {len(missing)} new params initialised from scratch.")
        if unexpected:
            print(f"[train] resume: {len(unexpected)} stale keys ignored.")
        if not args.finetune:
            try:
                optim.load_state_dict(ck["optim"])
                sched.load_state_dict(ck["sched"])
                start_epoch = ck.get("epoch", 0) + 1
                _ck_best = ck.get("best_sisnr", ck.get("best_val", None))
                if _ck_best is not None:
                    # checkpoints saved before this change stored loss (negative dB-ish);
                    # if the stored value is negative and small, it's the old convention.
                    best_sisnr = float(_ck_best) if float(_ck_best) > -50 else float("-inf")
                print(f"[train] resumed from {args.resume} @ epoch {start_epoch-1}")
            except Exception as e:
                print(f"[train] resume: optim/sched restore failed ({e}); starting fresh.")
        else:
            print(f"[train] finetune: model weights only loaded from {args.resume}; "
                  f"fresh optim/sched, lr={args.lr}, snr_range=[{args.snr_low},{args.snr_high}]dB")

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, "tb"))
    except Exception:
        tb = None

    sched_per_batch = sched if args.scheduler == "onecycle" else None

    # Clear history.json on a fresh run so plots don't overlap with a prior run
    if start_epoch == 1:
        _hist_path = os.path.join(args.ckpt_dir, "history.json")
        if os.path.isfile(_hist_path):
            import shutil as _sh
            _sh.copy2(_hist_path, _hist_path + ".bak")
        with open(_hist_path, "w") as _fh:
            json.dump([], _fh)

    # Linear warmup wrapper. Step every optimizer.step(); pass through to inner sched if any.
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
    # OneCycleLR has its own built-in warmup (pct_start=0.3).
    # An external warmup wrapper causes a double-warmup: at the transition point the
    # WarmupWrap stops setting LR manually and begins calling sched.step(), whose
    # first call resets LR to initial_lr (max_lr/25 ≈ 8e-6) — a sudden crash from
    # the just-completed warmup peak.  Suppress external warmup for onecycle.
    _warmup = 0 if args.scheduler == "onecycle" else args.warmup_steps
    sched_per_batch = WarmupWrap(optim, base_lrs, _warmup, sched_per_batch) \
        if _warmup > 0 or sched_per_batch is not None else None

    for epoch in range(start_epoch, args.epochs + 1):
        # Curriculum: ramp SNR lower-bound from easy → hard over curriculum_epochs.
        # Assumes SpeechDenoiseDataset reads self.snr_range each __getitem__ call.
        if args.curriculum and not args.finetune:
            if epoch <= args.curriculum_epochs:
                progress    = epoch / args.curriculum_epochs
                curr_snr_lo = args.snr_start + (args.snr_low - args.snr_start) * progress
            else:
                curr_snr_lo = args.snr_low
            train_dl.dataset.snr_range = (curr_snr_lo, args.snr_high)

        t0 = time.time()
        tr = run_epoch(model, train_dl, loss_fn, device,
                       optim=optim, scaler=scaler, amp=use_amp,
                       amp_dtype=amp_dtype,
                       max_batches=args.steps_per_epoch,
                       grad_accum=args.grad_accum,
                       sched_per_batch=sched_per_batch,
                       grad_clip=args.grad_clip,
                       channels_last=args.channels_last)
        # validate every N epochs (and on the last) to save GPU time
        do_val = (args.val_every <= 1) or (epoch % args.val_every == 0) \
                 or (epoch == args.epochs)
        if do_val:
            vl = run_epoch(model, val_dl, loss_fn, device, amp=use_amp,
                           amp_dtype=amp_dtype,
                           max_batches=args.val_max_batches,
                           channels_last=args.channels_last)
        else:
            vl = {"loss": float("nan"), "sisnr": 0.0, "sisnr_db": 0.0, "mag": 0.0, "aux_stft": 0.0}
        if args.scheduler in ("cosine", "warmrestarts"):
            sched.step()
        elif args.scheduler == "plateau" and do_val:
            sched.step(vl["loss"])
        dt = time.time() - t0

        neg_note = "  [SI-SDR>0 ok]" if tr["loss"] < 0 else ""
        print(f"epoch {epoch:03d}/{args.epochs}  "
              f"train={tr['loss']:.4f}  val={vl['loss']:.4f}  "
              f"tr_sisnr={tr['sisnr_db']:.2f}dB  val_sisnr={vl['sisnr_db']:.2f}dB  "
              f"tr_mag={tr['mag']:.4f}  val_mag={vl['mag']:.4f}  "
              f"val_aux={vl['aux_stft']:.4f}  "
              f"val_lf_noise={vl['lf_noise']:.4f}  val_sfloor={vl['speech_floor']:.4f}  "
              f"snr=[{getattr(train_dl.dataset, 'snr_range', (args.snr_low, args.snr_high))[0]:.1f},"
              f"{args.snr_high:.0f}]dB  "
              f"({dt:.1f}s){neg_note}")

        if tb is not None:
            for k, v in tr.items(): tb.add_scalar(f"train/{k}", v, epoch)
            if do_val:
                for k, v in vl.items(): tb.add_scalar(f"val/{k}", v, epoch)
            current_lr = sched.get_last_lr()[0] if hasattr(sched, "get_last_lr") \
                         else optim.param_groups[0]["lr"]
            tb.add_scalar("lr", current_lr, epoch)

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
        # Save best checkpoint on highest val SI-SDR in dB (higher = better).
        # Using sisnr_db (true dB) avoids the tanh-compression bias of the loss value.
        _val_db = float(vl["sisnr_db"]) if do_val else float("-inf")
        if do_val and _val_db > best_sisnr:
            best_sisnr = _val_db
            epochs_no_improve = 0
            ck["best_sisnr"] = best_sisnr
            torch.save(ck, os.path.join(args.ckpt_dir, "best.pt"))
            torch.save({"model": sd, "args": vars(args), "epoch": epoch},
                       os.path.join(args.ckpt_dir, "best_model_only.pt"))
            _rotate_topk(args.ckpt_dir, -best_sisnr, ck, k=3)   # rotate uses lower=better
            print(f"  >> new best  val_sisnr={best_sisnr:.2f} dB")
        elif do_val:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"[train] early stop at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

        if args.debug_every > 0 and epoch % args.debug_every == 0:
            save_debug_spectrograms(model, val_dl, device, epoch, args.ckpt_dir,
                                    n_fft=args.n_fft, hop_length=args.hop_length)
            plot_training_history(args.ckpt_dir, epoch)

        if args.audio_every > 0 and epoch % args.audio_every == 0:
            save_audio_samples(model, val_dl, device, epoch, args.results_dir,
                               sample_rate=args.sample_rate)

        # --- Per-epoch history logging (history.json + metrics.csv) ---
        import csv as _csv
        _sisnr_db_val   = float(vl["sisnr_db"])   if do_val else float("nan")
        _sisnr_db_train = float(tr["sisnr_db"])
        _entry = {
            "epoch":           epoch,
            "train_loss":      round(float(tr["loss"]),      6),
            "val_loss":        round(float(vl["loss"]),      6),
            "train_sisnr_db":  round(_sisnr_db_train, 4),
            "val_sisnr_db":    round(_sisnr_db_val,   4) if _sisnr_db_val == _sisnr_db_val else None,
            "val_aux_stft":    round(float(vl["aux_stft"]), 6) if do_val else None,
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

    # Final loss-history plot always saved after training ends
    plot_training_history(args.ckpt_dir, epoch if "epoch" in dir() else args.epochs)
    print("[train] done.")


if __name__ == "__main__":
    main()
