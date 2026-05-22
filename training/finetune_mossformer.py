"""
finetune_mossformer.py
======================
Fine-tune a pretrained SpeechBrain speech enhancement model on the
project's 70/30 LF/HF noise dataset.

Model priority (first one that loads wins):
  1. speechbrain/MossFormer2-SE-16kHz  — best; needs HuggingFace login for gated access.
     Log in first with:  huggingface-cli login
  2. speechbrain/metricgan-plus-voicebank — works without auth; solid baseline SE model.

Same structure as finetune_resemble.py:
  * FTLoss — Huber waveform + multi-scale STFT (5× boost <200 Hz)
  * SpeechDenoiseDataset with 70/30 LF/HF split
  * AMP, grad-clip, ReduceLROnPlateau, best-val checkpoint

Usage
-----
    cd C:\\Users\\rocha\\AI_builders\\Run
    python finetune_mossformer.py ^
        --data_root ../data_npy ^
        --epochs 10 --batch_size 4 --lr 2e-5

    # then benchmark:
    python benchmark.py --mossformer_ft_ckpt ./benchmark_outputs/mossformer_ft/mossformer_ft.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# torch._dynamo is lazy-imported the first time torch.optim.AdamW is called.
# That import chain runs inspect.getmodule on every sys.modules entry, which
# triggers SpeechBrain's k2_fsa LazyModule → ImportError (k2 not installed).
# Importing _dynamo here (before SpeechBrain is ever imported) means AdamW
# finds it cached in sys.modules and the inspect chain never re-runs.
try:
    import torch._dynamo  # noqa: F401
except Exception:
    pass

# ── Dual-layout path resolution ─────────────────────────────────────────────
# Two valid layouts are supported:
#   (A) Author's local machine — this file sits in  C:\...\AI_builders\Run\
#       alongside  Run/CRN/  and  Run/U_net/  as siblings.  ROOT / "U_net"
#       resolves to the real folder; this path wins by Path.is_dir() check.
#   (B) Public Git repo (AI_Builders_ANC_Pipeline) — this file sits in
#       <repo>/training/  while the U-Net code lives at <repo>/models/unet/.
#       The local sibling does NOT exist, so we fall back to the repo layout.
# Comparison is by Path.is_dir() rather than os.environ / hostname so the
# logic stays correct if either tree is copied/renamed.
ROOT        = Path(__file__).parent
_LOCAL_UNET = ROOT / "U_net"                        # layout (A)
_REPO_UNET  = ROOT.parent / "models" / "unet"       # layout (B)
UNET_DIR    = _LOCAL_UNET if _LOCAL_UNET.is_dir() else _REPO_UNET

if str(UNET_DIR) not in sys.path:
    sys.path.insert(0, str(UNET_DIR))

from dataset import SpeechDenoiseDataset, denoise_collate, worker_init_fn


# ---------------------------------------------------------------------------
# Multi-scale STFT loss  (identical to finetune_resemble.py)
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
        # Hard step (w[:cutoff] = boost) creates a spectral discontinuity that
        # causes ringing and musical noise artifacts in the enhanced output.
        cutoff     = int(200 * n_fft / sr)
        transition = max(int(80 * n_fft / sr), 4)   # 80 Hz crossfade band
        bins = torch.arange(n_fft // 2 + 1, dtype=torch.float32)
        w = 1.0 + (lf_boost - 1.0) * torch.sigmoid(-8.0 / transition * (bins - cutoff))
        self.register_buffer("fw", w.view(-1, 1), persistent=False)

    def forward(self, pred, tgt):
        mp = _stft_mag(pred, self.n_fft, self.hop, self.win, self.window)
        mt = _stft_mag(tgt,  self.n_fft, self.hop, self.win, self.window)
        sc = torch.norm((mt - mp) * self.fw, p="fro") / (torch.norm(mt * self.fw, p="fro") + 1e-7)
        ll = (torch.abs(torch.log(mp) - torch.log(mt)) * self.fw).mean()
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
        # SI-SNR loss: directly optimises the evaluation metric, preventing STFT from
        # finding spectral local minima with poor SNR (residual noise bursts, smearing).
        ref  = clean    - clean.mean(-1, keepdim=True)
        est  = enhanced - enhanced.mean(-1, keepdim=True)
        eps  = 1e-8
        alpha   = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
        proj    = alpha * ref
        noise_  = est - proj
        si_loss = -(10 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise_.pow(2).sum(-1) + eps))).mean()
        si_loss = si_loss / 30.0   # normalise to ~same scale as Huber
        return huber + 0.2 * stft + 0.3 * si_loss, {
            "huber": huber.detach(), "stft": stft.detach(), "si_loss": si_loss.detach()
        }


# ---------------------------------------------------------------------------
# SI-SNR metric (scale-invariant, dB)
# ---------------------------------------------------------------------------

def _si_snr(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batch SI-SNR in dB. Both tensors: (B, T)."""
    ref = ref - ref.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    proj  = alpha * ref
    noise = est - proj
    return 10 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))


# ---------------------------------------------------------------------------
# SpeechBrain SE model loader
# ---------------------------------------------------------------------------

_CANDIDATES = [
    ("speechbrain/MossFormer2-SE-16kHz",    "MossFormer2-SE"),
    ("speechbrain/metricgan-plus-voicebank", "MetricGAN+"),
]


def _load_se_model(savedir: Path, device: torch.device):
    """Load the best available SpeechBrain SE model; return (sb_model, label)."""
    try:
        # Pre-import speechbrain.nnet sub-modules BEFORE from_hparams runs.
        # SpeechBrain 1.1 lazy-loads nnet via k2_fsa integration; accessing
        # sb.nnet.RNN inside a HyperPyYAML model file triggers a failed lazy
        # import chain.  Explicitly importing first bypasses the lazy loader.
        import speechbrain.nnet.RNN      # noqa: F401
        import speechbrain.nnet.linear   # noqa: F401
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
    except ImportError:
        raise ImportError(
            "speechbrain not installed.\n"
            "  pip install speechbrain"
        )

    # SpeechBrain 1.1 registers a LazyModule for k2_fsa (Kaldi FSA library).
    # When torch.optim.AdamW init triggers torch._dynamo, Python's inspect
    # machinery calls hasattr(module, '__file__') on every sys.modules entry.
    # For the k2_fsa LazyModule this fires __getattr__ → ensure_module →
    # importlib.import_module('speechbrain.integrations.k2_fsa') → ImportError
    # (k2 not installed).  Replace the lazy module with a real dummy that has
    # __file__ defined so hasattr() short-circuits without triggering the import.
    import sys, types as _types
    _k2_key = "speechbrain.integrations.k2_fsa"
    if _k2_key in sys.modules and not isinstance(sys.modules[_k2_key], _types.ModuleType):
        _dummy_k2 = _types.ModuleType(_k2_key)
        _dummy_k2.__file__ = "<k2 not installed — dummy>"
        sys.modules[_k2_key] = _dummy_k2

    # SpeechBrain parses "cuda" as "cuda:N"; pass "cuda:0" explicitly.
    sb_device = f"{device.type}:{device.index or 0}" if device.type == "cuda" else "cpu"

    savedir.mkdir(parents=True, exist_ok=True)
    model = None
    label = None

    for source, name in _CANDIDATES:
        slug = name.replace(" ", "_").replace("+", "plus")
        try:
            model = SpectralMaskEnhancement.from_hparams(
                source=source,
                savedir=str(savedir / slug),
                run_opts={"device": sb_device},
            )
            label = name
            print(f"[finetune] Loaded '{name}' from {source}")
            break
        except Exception as e:
            print(f"[finetune] {name} skipped: {e}")

    if model is None:
        raise RuntimeError(
            "No SE model loaded. Check internet / HuggingFace login.\n"
            "  huggingface-cli login"
        )

    # SpeechBrain loads weights for inference with requires_grad=False.
    # Re-enable gradients so we can fine-tune.
    n_total = 0
    for p in model.mods.parameters():
        p.requires_grad_(True)
        n_total += p.numel()

    print(f"[finetune] {label}: {n_total / 1e6:.1f} M trainable params")
    return model, label


# ---------------------------------------------------------------------------
# Forward wrapper — calls enhance_batch in train mode
# ---------------------------------------------------------------------------

def _se_forward(model, noisy: torch.Tensor) -> torch.Tensor:
    """Run SpeechBrain SE model in training mode; return (B, T)."""
    L       = noisy.shape[-1]
    device  = noisy.device
    lengths = torch.ones(noisy.shape[0], device=device)

    # keep sub-modules in train mode (enhance_batch may flip to eval)
    model.mods.train()

    try:
        out = model.enhance_batch(noisy, lengths=lengths)
    except TypeError:
        out = model.enhance_batch(noisy)

    if out.dim() == 3:
        out = out.squeeze(1)
    elif out.dim() == 1:
        out = out.unsqueeze(0)

    if out.shape[-1] > L:
        out = out[..., :L]
    elif out.shape[-1] < L:
        out = F.pad(out, (0, L - out.shape[-1]))
    return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[finetune] device={device}")

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
        loader_kw["prefetch_factor"] = 2
    train_dl = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kw)
    val_dl   = DataLoader(val_ds,   shuffle=False, drop_last=False, **{
        **loader_kw, "num_workers": max(1, args.num_workers // 2)})
    print(f"[finetune] train batches={len(train_dl)}  val={len(val_dl)}")

    # ---- model ----
    savedir = Path(args.savedir)
    model, label_used = _load_se_model(savedir, device)

    # Optionally freeze encoder-like submodules
    if args.freeze_encoder:
        frozen = 0
        _enc_keys = {"encoder", "enc", "embed", "feature_extractor", "input_proj"}
        for name, p in model.mods.named_parameters():
            tok = name.split(".")[0].lower()
            if tok in _enc_keys or "encoder" in name.lower():
                p.requires_grad_(False)
                frozen += p.numel()
        if frozen:
            print(f"[finetune] encoder frozen ({frozen / 1e6:.1f} M params locked)")
        else:
            print("[finetune] freeze_encoder: no 'encoder' submodule found — all params trainable")

    trainable_params = [p for p in model.mods.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found after freeze step.")

    # ---- EWC pretrained snapshot ----
    # Clone initial weights for EWC L2 penalty; prevents catastrophic forgetting
    # on general noise while adapting to LF noise.
    pretrained_snap = {n: p.detach().clone()
                       for n, p in model.mods.named_parameters() if p.requires_grad}
    print(f"[finetune] EWC snapshot: {len(pretrained_snap)} param tensors  λ={args.ewc_lambda}")

    # ---- loss + optimiser ----
    loss_fn = FTLoss(sr=args.sample_rate, lf_boost=args.lf_boost).to(device)
    optim   = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    # mode="max": scheduler tracks val SI-SNR (higher = better).
    # patience=5: avoids collapsing LR on measurement noise (~0.3 dB jitter).
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=5, min_lr=2e-7)
    scaler  = torch.amp.GradScaler(device="cuda", enabled=pin)

    best_sisnr  = float("-inf")
    history     = []
    WARMUP_EPOCHS = 2     # ramp LR from lr/10 → lr over first 2 epochs
    curr_ewc    = args.ewc_lambda   # mutable EWC lambda (decays over training)

    for epoch in range(1, args.epochs + 1):
        # Linear warmup: keeps early gradients small so pre-trained features survive
        if epoch <= WARMUP_EPOCHS:
            for pg in optim.param_groups:
                pg["lr"] = args.lr * (epoch / WARMUP_EPOCHS)

        # Curriculum: ramp SNR lower bound from easy → hard
        if args.curriculum:
            if epoch <= args.curriculum_epochs:
                progress    = epoch / args.curriculum_epochs
                curr_snr_lo = args.snr_start + (args.snr_low - args.snr_start) * progress
            else:
                curr_snr_lo = args.snr_low
            train_dl.dataset.snr_range = (curr_snr_lo, args.snr_high)

        # Progressive unfreezing: thaw encoder after unfreeze_epoch so decoder stabilises first
        if args.freeze_encoder and args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch + 1:
            newly_unfrozen = 0
            _enc_keys = {"encoder", "enc", "embed", "feature_extractor", "input_proj"}
            for name, p in model.mods.named_parameters():
                tok = name.split(".")[0].lower()
                if not p.requires_grad and (tok in _enc_keys or "encoder" in name.lower()):
                    p.requires_grad_(True)
                    newly_unfrozen += p.numel()
            if newly_unfrozen:
                # Re-create optimizer to pick up the newly trainable params
                trainable_params = [p for p in model.mods.parameters() if p.requires_grad]
                for pg in optim.param_groups:
                    pg["params"] = trainable_params
                print(f"  [progressive unfreeze] encoder unfrozen at epoch {epoch}: "
                      f"{newly_unfrozen/1e6:.1f} M params added  (total trainable: "
                      f"{sum(p.numel() for p in trainable_params)/1e6:.1f} M)")

        # EWC lambda decay: loosen regularisation as model adapts
        if args.ewc_decay_epochs > 0 and epoch > WARMUP_EPOCHS and \
                (epoch - WARMUP_EPOCHS) % args.ewc_decay_epochs == 0:
            curr_ewc *= 0.5
            print(f"  [EWC] lambda decayed → {curr_ewc:.2e}")

        # ---- train ----
        model.mods.train()
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
                enhanced     = _se_forward(model, noisy)
                loss, _parts = loss_fn(enhanced, clean)

            # EWC computed in float32 outside autocast (bf16/fp16 loses precision)
            if curr_ewc > 0:
                ewc = sum((p.float() - pretrained_snap[n]).pow(2).mean()
                          for n, p in model.mods.named_parameters()
                          if n in pretrained_snap)
                loss = loss + curr_ewc * ewc / len(pretrained_snap)

            if not torch.isfinite(loss):
                optim.zero_grad(set_to_none=True)
                continue

            if pin:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.mods.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.mods.parameters(), 1.0)
                optim.step()

            optim.zero_grad(set_to_none=True)
            tr_loss += loss.item() * noisy.size(0)
            n_tr    += noisy.size(0)

        tr_avg = tr_loss / max(1, n_tr)

        # ---- val ----
        model.mods.eval()
        vl_loss  = 0.0
        vl_sisnr = 0.0
        n_vl     = 0
        with torch.no_grad():
            for step, (noisy, clean, _) in enumerate(val_dl):
                if args.val_batches and step >= args.val_batches:
                    break
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                enhanced     = _se_forward(model, noisy)
                loss, _parts = loss_fn(enhanced, clean)
                min_len = min(enhanced.shape[-1], clean.shape[-1])
                sisnr   = _si_snr(clean[..., :min_len].cpu(),
                                  enhanced[..., :min_len].cpu()).mean().item()
                vl_loss  += loss.item() * noisy.size(0)
                vl_sisnr += sisnr       * noisy.size(0)
                n_vl     += noisy.size(0)

        vl_avg   = vl_loss  / max(1, n_vl)
        vl_sisnr = vl_sisnr / max(1, n_vl)
        sched.step(vl_sisnr)   # scheduler tracks SI-SNR (higher = better)

        dt  = time.time() - t0
        lr  = optim.param_groups[0]["lr"]
        msg = (f"epoch {epoch:03d}/{args.epochs}  "
               f"train={tr_avg:.4f}  val={vl_avg:.4f}  "
               f"SI-SNR={vl_sisnr:+.2f} dB  lr={lr:.2e}  ({dt:.1f}s)")
        print(msg)
        history.append({"epoch": epoch, "train": tr_avg, "val": vl_avg,
                         "val_sisnr": vl_sisnr, "lr": lr})

        if vl_sisnr > best_sisnr:
            best_sisnr = vl_sisnr
            ck = {
                "mods_state_dict": model.mods.state_dict(),
                "epoch":           epoch,
                "val_sisnr":       vl_sisnr,
                "val_loss":        vl_avg,
                "args":            vars(args),
                "model_src":       label_used,
            }
            torch.save(ck, out_dir / "mossformer_ft.pt")
            print(f"  >> best SI-SNR={best_sisnr:+.2f} dB -> {out_dir / 'mossformer_ft.pt'}")

    with open(out_dir / "ft_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"[finetune] done. Best val SI-SNR: {best_sisnr:+.2f} dB")
    print(f"[finetune] checkpoint: {out_dir / 'mossformer_ft.pt'}")
    print(f"\nNext: python benchmark.py --mossformer_ft_ckpt {out_dir / 'mossformer_ft.pt'}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",        default="../data_npy")
    ap.add_argument("--savedir",          default="./pretrained_models/mossformer2",
                    help="Where SpeechBrain caches downloaded weights.")
    ap.add_argument("--out_dir",          default="./benchmark_outputs/mossformer_ft")
    ap.add_argument("--epochs",           type=int,   default=50)
    ap.add_argument("--batch_size",       type=int,   default=4)
    ap.add_argument("--num_workers",      type=int,   default=2)
    ap.add_argument("--lr",               type=float, default=2e-5)
    ap.add_argument("--snr_low",          type=float, default=-5.0)
    ap.add_argument("--snr_high",         type=float, default=15.0)
    ap.add_argument("--lf_noise_ratio",   type=float, default=0.70)
    ap.add_argument("--lf_boost",         type=float, default=5.0)
    ap.add_argument("--ewc_lambda",       type=float, default=1e-3,
                    help="EWC regularisation strength (0 = disabled).")
    ap.add_argument("--sample_rate",      type=int,   default=16_000)
    ap.add_argument("--segment_seconds",  type=float, default=3.0)
    ap.add_argument("--noise_pool_max",   type=int,   default=4000)
    ap.add_argument("--steps_per_epoch",  type=int,   default=100)
    ap.add_argument("--val_batches",      type=int,   default=20)
    ap.add_argument("--freeze_encoder",   action="store_true", default=True)
    ap.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false")
    ap.add_argument("--unfreeze_epoch",   type=int, default=10,
                    help="Progressively unfreeze the encoder after this epoch. "
                         "0 = never unfreeze (encoder stays frozen). "
                         "Useful when freeze_encoder=True: encoder thaws after the "
                         "decoder has stabilised, letting the full model fine-tune.")
    ap.add_argument("--ewc_decay_epochs", type=int, default=10,
                    help="Halve ewc_lambda every N epochs so regularisation loosens "
                         "as the model adapts. 0 = constant lambda.")
    ap.add_argument("--curriculum", action="store_true", default=True,
                    help="Ramp SNR lower bound from --snr_start to --snr_low over "
                         "--curriculum_epochs epochs.")
    ap.add_argument("--no_curriculum", dest="curriculum", action="store_false")
    ap.add_argument("--curriculum_epochs", type=int, default=10,
                    help="Epochs to ramp SNR from --snr_start to --snr_low.")
    ap.add_argument("--snr_start", type=float, default=5.0,
                    help="Initial easy SNR lower bound for curriculum.")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
