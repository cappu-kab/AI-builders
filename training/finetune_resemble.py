"""
finetune_resemble.py
====================
Fine-tune Resemble Enhance's denoiser on the project's 70/30 LF/HF noise dataset.

The Resemble Enhance enhancer has two stages:
  1. Denoiser  — a U-Net that suppresses noise (the part we fine-tune)
  2. Enhancer  — a CFM-based super-resolution stage (frozen / untouched)

We fine-tune ONLY the denoiser on our (noisy, clean) pairs using:
  * Huber waveform loss  (direct signal regression)
  * Multi-scale STFT loss (frequency-weighted, 5x boost < 200 Hz)

The fine-tuned weights are saved to --out_dir/denoiser_ft.pt and can be
passed to benchmark.py via --resemble_ft_ckpt.

Usage
-----
    cd C:\\Users\\rocha\\AI_builders\\Run
    python finetune_resemble.py \\
        --data_root ./data_npy \\
        --resemble_dir C:\\Users\\rocha\\resemble-enhance \\
        --epochs 10 --batch_size 4 --lr 1e-4

    # then benchmark:
    python benchmark.py --resemble_ft_ckpt ./benchmark_outputs/resemble_ft/denoiser_ft.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torchaudio.transforms as T


# ── Dual-layout path resolution ─────────────────────────────────────────────
# (A) Author's local machine: this file sits in  C:\...\AI_builders\Run\
#     beside  Run/U_net/  as siblings → ROOT / "U_net" exists.
# (B) Public Git repo (AI_Builders_ANC_Pipeline): this file sits in
#     <repo>/training/  and the U-Net code lives at <repo>/models/unet/.
# Path.is_dir() picks whichever tree the user actually has on disk.
ROOT        = Path(__file__).parent
_LOCAL_UNET = ROOT / "U_net"                        # layout (A)
_REPO_UNET  = ROOT.parent / "models" / "unet"       # layout (B)
UNET_DIR    = _LOCAL_UNET if _LOCAL_UNET.is_dir() else _REPO_UNET

if str(UNET_DIR) not in sys.path:
    sys.path.insert(0, str(UNET_DIR))

from dataset import SpeechDenoiseDataset, denoise_collate, worker_init_fn


# ---------------------------------------------------------------------------
# Multi-scale STFT loss (matches losses.py structure, no external dep)
# ---------------------------------------------------------------------------

def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int,
              window: torch.Tensor) -> torch.Tensor:
    spec = torch.stft(x.float(), n_fft=n_fft, hop_length=hop, win_length=win,
                      window=window, return_complex=True, center=True)
    return torch.abs(spec) + 1e-7


class _STFTLoss(nn.Module):
    def __init__(self, n_fft, hop, win, sr=16_000, lf_boost=5.0):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        # Sigmoid taper: lf_boost at DC, smoothly decays to 1.0 above 200 Hz.
        # Hard-step cutoff creates spectral ringing; sigmoid avoids that.
        cutoff     = int(200 * n_fft / sr)
        transition = max(int(80 * n_fft / sr), 4)
        bins = torch.arange(n_fft // 2 + 1, dtype=torch.float32)
        w = 1.0 + (lf_boost - 1.0) * torch.sigmoid(-8.0 / transition * (bins - cutoff))
        self.register_buffer("fw", w.view(-1, 1), persistent=False)

    def forward(self, pred, tgt):
        mp = _stft_mag(pred, self.n_fft, self.hop, self.win, self.window)
        mt = _stft_mag(tgt,  self.n_fft, self.hop, self.win, self.window)
        sc = torch.norm((mt - mp) * self.fw, p="fro") / (torch.norm(mt * self.fw, p="fro") + 1e-7)
        ll = (torch.abs(torch.log(mp + 1e-7) - torch.log(mt + 1e-7)) * self.fw).mean()
        return sc + ll


class FTLoss(nn.Module):
    def __init__(self, sr: int = 16_000, lf_boost: float = 5.0):
        super().__init__()
        self.stft_losses = nn.ModuleList([
            _STFTLoss(256,  64,  240, sr, lf_boost),
            _STFTLoss(512,  128, 480, sr, lf_boost),
            _STFTLoss(1024, 256, 960, sr, lf_boost),
        ])

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor):
        huber = F.huber_loss(enhanced, clean, delta=0.5)
        stft  = sum(L(enhanced, clean) for L in self.stft_losses) / len(self.stft_losses)
        # SI-SDR loss: directly optimises the evaluation metric and prevents
        # STFT from settling in spectral local minima with poor SNR.
        ref     = clean    - clean.mean(-1, keepdim=True)
        est     = enhanced - enhanced.mean(-1, keepdim=True)
        eps     = 1e-8
        alpha   = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
        proj    = alpha * ref
        noise_  = est - proj
        si_loss = -(10 * torch.log10((proj.pow(2).sum(-1) + eps) /
                                     (noise_.pow(2).sum(-1) + eps))).mean()
        si_loss = si_loss / 30.0   # normalise to ~same scale as Huber
        return huber + 0.2 * stft + 0.3 * si_loss, {
            "huber": huber.detach(), "stft": stft.detach(), "si_loss": si_loss.detach()
        }


# ---------------------------------------------------------------------------
# Resemble denoiser wrapper
# ---------------------------------------------------------------------------

def _load_resemble_denoiser(resemble_dir: Path, device: torch.device):
    """Load the pretrained Resemble denoiser and return (denoiser, run_dir)."""
    if str(resemble_dir) not in sys.path:
        sys.path.insert(0, str(resemble_dir))

    from resemble_enhance.enhancer.download import download
    from resemble_enhance.denoiser.denoiser import Denoiser
    from resemble_enhance.denoiser.hparams import HParams as DenoiserHParams

    # ------------------------------------------------------------------
    # Patch HParams.from_yaml on the BASE class before any .load() call.
    #
    # The shared hparams.yaml contains enhancer-only keys such as
    # 'cfm_solver_method' that are not declared on the denoiser's HParams
    # dataclass.  OmegaConf.merge is strict and raises ConfigKeyError for
    # any unknown key.  We filter the raw YAML to known fields first.
    # ------------------------------------------------------------------
    import dataclasses
    from omegaconf import OmegaConf
    import resemble_enhance.hparams as _re_hp_mod

    def _safe_from_yaml_impl(cls, path):
        raw     = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        known   = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            print(f"[resemble patch] dropping {len(unknown)} unknown YAML "
                  f"key(s) not in {cls.__name__}: {sorted(unknown)}")
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**dict(OmegaConf.merge(cls(), OmegaConf.create(filtered))))

    _re_hp_mod.HParams.from_yaml = classmethod(_safe_from_yaml_impl)
    # ------------------------------------------------------------------

    # download() puts models in ~/.cache/resemble-enhance/
    run_dir = None
    try:
        run_dir = download()
    except Exception as e:
        print(f"[finetune] auto-download failed ({e}); trying without run_dir")

    # ------------------------------------------------------------------
    # Manual denoiser load — bypasses load_denoiser() to fix key mismatch.
    #
    # The combined enhancer checkpoint (mp_rank_00_model_states.pt) stores
    # ALL sub-models under a single "module" dict with prefixed keys:
    #   denoiser.net.*   ← denoiser weights (what we want)
    #   lcfm.*           ← CFM enhancer (ignore)
    #   vocoder.*        ← vocoder (ignore)
    #   normalizer.*     ← normalizer (ignore)
    #
    # Denoiser.load_state_dict expects bare "net.*" keys.
    # We filter and strip the "denoiser." prefix before loading.
    # ------------------------------------------------------------------
    if run_dir is None:
        print("[finetune] No run_dir — building denoiser with default HParams (random weights)")
        hp = DenoiserHParams()
    else:
        hp = DenoiserHParams.load(run_dir)

    denoiser = Denoiser(hp).to(device)

    if run_dir is not None:
        ckpt_path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
        if ckpt_path.is_file():
            raw_sd = torch.load(str(ckpt_path), map_location="cpu")["module"]
            prefix = "denoiser."
            filtered_sd = {
                k[len(prefix):]: v
                for k, v in raw_sd.items()
                if k.startswith(prefix)
            }
            if filtered_sd:
                missing, unexpected = denoiser.load_state_dict(filtered_sd, strict=False)
                if missing:
                    print(f"[finetune] WARNING — {len(missing)} missing key(s) in denoiser ckpt "
                          f"(first 5: {missing[:5]})")
                if unexpected:
                    print(f"[finetune] WARNING — {len(unexpected)} unexpected key(s) ignored "
                          f"(first 5: {unexpected[:5]})")
                print(f"[finetune] Loaded {len(filtered_sd)} denoiser weights from {ckpt_path}")
            else:
                print(f"[finetune] WARNING — no 'denoiser.*' keys found in checkpoint; "
                      f"proceeding with random weights")
        else:
            print(f"[finetune] WARNING — checkpoint not found at {ckpt_path}; "
                  f"proceeding with random weights")

    denoiser.train()
    print(f"[finetune] Resemble denoiser ready  "
          f"({sum(p.numel() for p in denoiser.parameters())/1e6:.1f} M params)")
    return denoiser, run_dir


def _denoiser_forward(denoiser, noisy: torch.Tensor, sr: int) -> torch.Tensor:
    """Run the denoiser and return the enhanced waveform (same length).

    Resemble Denoiser.forward(x, y=None) — y is an optional clean target for
    computing L1 training loss internally; sr is not a parameter.
    The model operates at its own internal sample rate (set in HParams); our
    data is already at 16 kHz which matches the downloaded checkpoint.
    """
    L = noisy.shape[-1]
    with torch.amp.autocast(device_type=noisy.device.type,
                            enabled=(noisy.device.type == "cuda")):
        out = denoiser(noisy)   # no sr arg — Denoiser.forward(x, y=None)
    if isinstance(out, tuple):
        out = out[0]
    if out.shape[-1] > L:
        out = out[..., :L]
    elif out.shape[-1] < L:
        out = F.pad(out, (0, L - out.shape[-1]))
    return out

def plot_visuals(clean, noisy, enhanced, history, epoch, out_dir):
    """สร้างภาพเปรียบเทียบ Spectrogram และกราฟ Loss"""
    # ย้ายข้อมูลไป CPU และดึงมาแค่ตัวอย่างแรกใน Batch (Index 0)
    # ตัดมิติ Batch ออกเพื่อให้เหลือแค่ [Channels, Time]
    c = clean[0].cpu()
    n = noisy[0].cpu()
    e = enhanced[0].cpu()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Plot Loss History
    if history:
        epochs_list = [h['epoch'] for h in history]
        tr_loss = [h['train'] for h in history]
        vl_loss = [h['val'] for h in history]
        axes[0, 0].plot(epochs_list, tr_loss, label='Train')
        axes[0, 0].plot(epochs_list, vl_loss, label='Val')
        axes[0, 0].set_title(f"Loss Curve (Epoch {epoch})")
        axes[0, 0].legend()

    # ฟังก์ชันแปลง Waveform เป็น Spectrogram (dB)
    spec_transform = T.Spectrogram(n_fft=512, hop_length=128)
    
    def get_spec_db(wav):
        # บังคับให้เป็น 2D [Freq, Time]
        spec = spec_transform(wav)
        if spec.ndim > 2: spec = spec[0] 
        return 10 * torch.log10(spec + 1e-9)

    # 2. Spectrogram: Clean
    axes[0, 1].imshow(get_spec_db(c), origin='lower', aspect='auto', cmap='magma')
    axes[0, 1].set_title("Clean Spectrogram")

    # 3. Spectrogram: Noisy
    axes[1, 0].imshow(get_spec_db(n), origin='lower', aspect='auto', cmap='magma')
    axes[1, 0].set_title("Noisy Spectrogram")

    # 4. Spectrogram: Enhanced
    axes[1, 1].imshow(get_spec_db(e), origin='lower', aspect='auto', cmap='magma')
    axes[1, 1].set_title("Enhanced Spectrogram")

    plt.tight_layout()
    plt.savefig(out_dir / f"visual_epoch_{epoch:03d}.png")
    plt.close()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[finetune] device={device}")

    resemble_dir = Path(args.resemble_dir)
    if not resemble_dir.is_dir():
        raise FileNotFoundError(
            f"resemble-enhance dir not found: {resemble_dir}\n"
            f"Clone with: git clone https://github.com/resemble-ai/resemble-enhance {resemble_dir}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- dataset ----
    seg_len = int(args.segment_seconds * args.sample_rate)
    ds_kw = dict(
        segment_len=seg_len,
        snr_range=(args.snr_low, args.snr_high),
        sample_rate=args.sample_rate,
        lf_noise_ratio=args.lf_noise_ratio,
        lf_cutoff_hz=200.0,
        noise_pool_max=args.noise_pool_max,
    )
    train_ds = SpeechDenoiseDataset(args.data_root, split="train", augment=True,  **ds_kw)
    val_ds   = SpeechDenoiseDataset(args.data_root, split="val",   augment=False, **ds_kw)

    pin = device.type == "cuda"
    loader_kw = dict(batch_size=args.batch_size, pin_memory=pin,
                     collate_fn=denoise_collate, worker_init_fn=worker_init_fn,
                     persistent_workers=args.num_workers > 0)
    if args.num_workers > 0:
        loader_kw["num_workers"]     = args.num_workers
        loader_kw["prefetch_factor"] = 4
    train_dl = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kw)
    val_dl   = DataLoader(val_ds,   shuffle=False, drop_last=False, **{
        **loader_kw, "num_workers": max(1, args.num_workers // 2)})
    print(f"[finetune] train batches: {len(train_dl)}  val: {len(val_dl)}")

    # ---- model ----
    denoiser, run_dir = _load_resemble_denoiser(resemble_dir, device)

    # Freeze encoder if requested — only fine-tune decoder
    if args.freeze_encoder:
        frozen = 0
        for name, p in denoiser.named_parameters():
            if "encoder" in name or "down" in name:
                p.requires_grad_(False)
                frozen += p.numel()
        print(f"[finetune] encoder frozen ({frozen/1e6:.1f} M params locked)")

    # ---- EWC pretrained snapshot ----
    # Clone initial weights for EWC L2 penalty; prevents catastrophic forgetting
    # on broadband noise while adapting to LF noise.
    pretrained_snap = {n: p.detach().clone()
                       for n, p in denoiser.named_parameters() if p.requires_grad}
    print(f"[finetune] EWC snapshot: {len(pretrained_snap)} param tensors  λ={args.ewc_lambda}")

    # ---- loss + optim ----
    loss_fn = FTLoss(sr=args.sample_rate, lf_boost=args.lf_boost).to(device)
    optim   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, denoiser.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    scaler  = torch.amp.GradScaler(device="cuda", enabled=pin)

    best_val   = float("inf")
    history    = []
    WARMUP_EPOCHS = 2   # ramp LR from lr/10 → lr over first 2 epochs

    for epoch in range(1, args.epochs + 1):
        # Linear warmup: keeps early gradients small so pretrained features survive
        if epoch <= WARMUP_EPOCHS:
            for pg in optim.param_groups:
                pg["lr"] = args.lr * (epoch / WARMUP_EPOCHS)

        # ---- train ----
        denoiser.train()
        t0      = time.time()
        tr_loss = 0.0
        n_tr    = 0
        optim.zero_grad(set_to_none=True)
        for step, (noisy, clean, _) in enumerate(train_dl):
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=pin):
                enhanced     = _denoiser_forward(denoiser, noisy, args.sample_rate)
                loss, _parts = loss_fn(enhanced, clean)

            # EWC: computed in float32 outside autocast to avoid precision loss
            if args.ewc_lambda > 0:
                ewc = sum((p.float() - pretrained_snap[n]).pow(2).mean()
                          for n, p in denoiser.named_parameters()
                          if n in pretrained_snap)
                loss = loss + args.ewc_lambda * ewc / len(pretrained_snap)

            if not torch.isfinite(loss):
                optim.zero_grad(set_to_none=True)
                continue

            if pin:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                optim.step()
            optim.zero_grad(set_to_none=True)
            tr_loss += loss.item() * noisy.size(0)
            n_tr    += noisy.size(0)

        tr_avg = tr_loss / max(1, n_tr)

        # ---- val ----
        denoiser.eval()
        vl_loss = 0.0
        n_vl    = 0
        with torch.no_grad():
            for step, (noisy, clean, _) in enumerate(val_dl):
                if args.val_batches and step >= args.val_batches:
                    break
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                enhanced     = _denoiser_forward(denoiser, noisy, args.sample_rate)
                loss, _parts = loss_fn(enhanced, clean)
                vl_loss += loss.item() * noisy.size(0)
                n_vl    += noisy.size(0)
        vl_avg = vl_loss / max(1, n_vl)
        sched.step(vl_avg)

        dt  = time.time() - t0
        lr  = optim.param_groups[0]["lr"]
        msg = (f"epoch {epoch:03d}/{args.epochs}  "
               f"train={tr_avg:.4f}  val={vl_avg:.4f}  "
               f"lr={lr:.2e}  ({dt:.1f}s)")
        print(msg)
        history.append({"epoch": epoch, "train": tr_avg, "val": vl_avg, "lr": lr})
        # ---- เซฟรูปทุก 5 Epoch ----
        if epoch % 5 == 0 or epoch == 1:
            # ใช้ข้อมูล noisy, clean, enhanced จาก loop validation ล่าสุด
            plot_visuals(clean, noisy, enhanced, history, epoch, out_dir)
            print(f"  ↳ [Visuals] Saved spectrograms to {out_dir}")

        if vl_avg < best_val:
            best_val = vl_avg
            ck = {"denoiser": denoiser.state_dict(),
                  "epoch":    epoch,
                  "val_loss": vl_avg,
                  "args":     vars(args),
                  "run_dir":  str(run_dir) if run_dir else ""}
            torch.save(ck, out_dir / "denoiser_ft.pt")
            print(f"  ↳ saved best (val={best_val:.4f}) → {out_dir / 'denoiser_ft.pt'}")

    # ---- save training history ----
    with open(out_dir / "ft_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"[finetune] done.  Best val loss: {best_val:.4f}")
    print(f"[finetune] fine-tuned checkpoint: {out_dir / 'denoiser_ft.pt'}")
    print(f"\nNext step — run benchmark with fine-tuned model:")
    print(f"  python benchmark.py --resemble_ft_ckpt {out_dir / 'denoiser_ft.pt'}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",        default="../data_npy")
    ap.add_argument("--resemble_dir",     default=str(Path.home() / "resemble-enhance"))
    ap.add_argument("--out_dir",          default="./benchmark_outputs/resemble_ft")
    ap.add_argument("--epochs",           type=int,   default=50)
    ap.add_argument("--batch_size",       type=int,   default=4,
                    help="Keep small (4-8) — Resemble denoiser is larger than CRN/U-Net.")
    ap.add_argument("--num_workers",      type=int,   default=2)
    ap.add_argument("--lr",               type=float, default=3e-5,
                    help="Peak learning rate. Warmed up from lr/10 over 2 epochs. "
                         "3e-5 is safer than 1e-4 for fine-tuning a large pretrained model.")
    ap.add_argument("--ewc_lambda",       type=float, default=1e-3,
                    help="EWC regularisation strength. Penalises deviation from pretrained weights. "
                         "0 = disabled. 1e-3 balances adaptation vs. forgetting.")
    ap.add_argument("--snr_low",          type=float, default=-5.0)
    ap.add_argument("--snr_high",         type=float, default=15.0)
    ap.add_argument("--lf_noise_ratio",   type=float, default=0.70)
    ap.add_argument("--lf_boost",         type=float, default=5.0,
                    help="STFT loss weight multiplier for bins < 200 Hz.")
    ap.add_argument("--sample_rate",      type=int,   default=16_000)
    ap.add_argument("--segment_seconds",  type=float, default=3.0)
    ap.add_argument("--noise_pool_max",   type=int,   default=4000,
                    help="Cap noise pool (default 4000 ≈ 1 GB) to save RAM.")
    ap.add_argument("--steps_per_epoch",  type=int,   default=100,
                    help="Cap steps per epoch (0 = full). Default 100 for quick runs.")
    ap.add_argument("--val_batches",      type=int,   default=20)
    ap.add_argument("--freeze_encoder",   action="store_true", default=True,
                    help="Freeze denoiser encoder; only fine-tune decoder (faster, less forgetting).")
    ap.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())


