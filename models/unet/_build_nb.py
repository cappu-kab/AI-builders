"""Build the Colab training notebook by embedding existing .py files."""
import json, pathlib

HERE = pathlib.Path("/sessions/quirky-lucid-newton/mnt/outputs")
UPLD = pathlib.Path("/sessions/quirky-lucid-newton/mnt/uploads")

MODEL_SRC   = (HERE / "model_unet_advanced.py").read_text()
DATASET_SRC = (UPLD / "a9179137-81c7-4b80-b355-61f01699a9cb-1777729847489_dataset.py").read_text()
LOSSES_SRC  = (UPLD / "05d8c7ca-52b8-4f46-8687-5ecec601eb9d-1777729885415_losses.py").read_text()


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md(
    "# Speech Denoising — AdvancedUNetSE training (Colab)\n"
    "\n"
    "Single-GPU training notebook for an Advanced U-Net with SE attention,\n"
    "dilated bottleneck and gated skips. Drop-in replacement for a CRN.\n"
    "\n"
    "**Before you run:** set Runtime → Change runtime type → GPU (T4 is enough).\n"
    "\n"
    "This notebook is self-contained — it writes `dataset.py`, `losses.py`,\n"
    "and `model_unet_advanced.py` to `/content` and trains inline so you can\n"
    "watch progress live and interrupt at any point.\n"
    "\n"
    "**Required:** your data folder laid out as:\n"
    "\n"
    "```\n"
    "DATA_ROOT/\n"
    "  train/clean/*.npy\n"
    "  train/noise/*.npy\n"
    "  train/labels_npy.csv\n"
    "  val/  (same)\n"
    "  test/ (same)\n"
    "```\n"
))

cells.append(md("## 1. GPU sanity check\n"))
cells.append(code(
    "!nvidia-smi -L\n"
    "import torch\n"
    "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\n"
))

cells.append(md(
    "## 2. Mount Google Drive (optional, recommended)\n"
    "\n"
    "Saving checkpoints to Drive means you can resume after Colab disconnects.\n"
    "Skip this cell if your data already lives in `/content`.\n"
))
cells.append(code(
    "from google.colab import drive\n"
    "drive.mount('/content/drive')\n"
    "\n"
    "# EDIT THESE PATHS to match your Drive layout:\n"
    "DATA_ROOT = '/content/drive/MyDrive/speech_denoise/data_npy'\n"
    "CKPT_DIR  = '/content/drive/MyDrive/speech_denoise/checkpoints_unet'\n"
    "\n"
    "import os\n"
    "os.makedirs(CKPT_DIR, exist_ok=True)\n"
    "print('data:', DATA_ROOT)\n"
    "print('ckpt:', CKPT_DIR)\n"
))

cells.append(md("## 3. Write project files to /content\n"))
cells.append(code("%%writefile /content/dataset.py\n" + DATASET_SRC))
cells.append(code("%%writefile /content/losses.py\n"  + LOSSES_SRC))
cells.append(code("%%writefile /content/model_unet_advanced.py\n" + MODEL_SRC))
cells.append(code(
    "import sys\n"
    "if '/content' not in sys.path:\n"
    "    sys.path.insert(0, '/content')\n"
    "# Force a fresh import in case you re-run after editing\n"
    "for m in ('dataset', 'losses', 'model_unet_advanced'):\n"
    "    sys.modules.pop(m, None)\n"
    "import dataset, losses, model_unet_advanced\n"
    "print('OK:', dataset.__file__, losses.__file__, model_unet_advanced.__file__)\n"
))

cells.append(md(
    "## 4. Training config\n"
    "\n"
    "Tweak these knobs and re-run the training cell. They mirror the CLI\n"
    "args of `train.py`.\n"
))
cells.append(code(
    "from types import SimpleNamespace\n"
    "\n"
    "args = SimpleNamespace(\n"
    "    data_root       = DATA_ROOT,\n"
    "    ckpt_dir        = CKPT_DIR,\n"
    "    epochs          = 30,\n"
    "    batch_size      = 16,           # drop to 8 if you OOM on T4\n"
    "    num_workers     = 2,\n"
    "    lr              = 2e-4,\n"
    "    weight_decay    = 0.0,\n"
    "    snr_low         = -20.0,\n"
    "    snr_high        = 20.0,\n"
    "    segment_seconds = 4.0,\n"
    "    sample_rate     = 16_000,\n"
    "    n_fft           = 512,\n"
    "    hop_length      = 128,\n"
    "    win_length      = 512,\n"
    "    base_channels   = 32,\n"
    "    depth           = 5,\n"
    "    mask_scale      = 10.0,\n"
    "    seed            = 1234,\n"
    "    resume          = '',           # e.g. f'{CKPT_DIR}/last.pt' to resume\n"
    "    no_amp          = False,\n"
    "    grad_clip       = 5.0,\n"
    ")\n"
))

cells.append(md("## 5. Build dataloaders, model, loss, optimizer\n"))
cells.append(code(
    "import os, time, random\n"
    "import numpy as np\n"
    "import torch\n"
    "from pathlib import Path\n"
    "from torch.utils.data import DataLoader\n"
    "\n"
    "from dataset import SpeechDenoiseDataset, denoise_collate\n"
    "from losses import DenoiseLoss\n"
    "from model_unet_advanced import AdvancedUNetSE\n"
    "\n"
    "# --- seed ---\n"
    "random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)\n"
    "torch.cuda.manual_seed_all(args.seed)\n"
    "\n"
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "print('device:', device)\n"
    "\n"
    "Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "seg_len = int(args.segment_seconds * args.sample_rate)\n"
    "\n"
    "train_ds = SpeechDenoiseDataset(args.data_root, split='train',\n"
    "                                segment_len=seg_len,\n"
    "                                snr_range=(args.snr_low, args.snr_high),\n"
    "                                augment=True, sample_rate=args.sample_rate)\n"
    "val_ds   = SpeechDenoiseDataset(args.data_root, split='val',\n"
    "                                segment_len=seg_len,\n"
    "                                snr_range=(args.snr_low, args.snr_high),\n"
    "                                augment=False, sample_rate=args.sample_rate)\n"
    "\n"
    "pin = torch.cuda.is_available()\n"
    "train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,\n"
    "                      num_workers=args.num_workers, pin_memory=pin,\n"
    "                      drop_last=True, collate_fn=denoise_collate,\n"
    "                      persistent_workers=args.num_workers > 0)\n"
    "val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,\n"
    "                      num_workers=max(1, args.num_workers // 2), pin_memory=pin,\n"
    "                      drop_last=False, collate_fn=denoise_collate,\n"
    "                      persistent_workers=args.num_workers > 0)\n"
    "print(f'train batches: {len(train_dl)}    val batches: {len(val_dl)}')\n"
    "\n"
    "# --- model ---\n"
    "model = AdvancedUNetSE(\n"
    "    n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length,\n"
    "    base_channels=args.base_channels, depth=args.depth, mask_scale=args.mask_scale,\n"
    ").to(device)\n"
    "n_params = sum(p.numel() for p in model.parameters())\n"
    "print(f'AdvancedUNetSE params: {n_params/1e6:.2f} M')\n"
    "\n"
    "# --- STFT front-end (waveform -> (B,2,F,T) in fp32, AMP-safe) ---\n"
    "class STFTFront(torch.nn.Module):\n"
    "    def __init__(self, n_fft, hop_length, win_length):\n"
    "        super().__init__()\n"
    "        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length\n"
    "        self.register_buffer('window', torch.hann_window(win_length), persistent=False)\n"
    "    def forward(self, wav):\n"
    "        with torch.cuda.amp.autocast(enabled=False):\n"
    "            wav = wav.float()\n"
    "            spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length,\n"
    "                              win_length=self.win_length, window=self.window,\n"
    "                              center=True, return_complex=True)\n"
    "            spec = torch.stack([spec.real, spec.imag], dim=1)\n"
    "        return spec\n"
    "\n"
    "stft_front = STFTFront(args.n_fft, args.hop_length, args.win_length).to(device)\n"
    "loss_fn    = DenoiseLoss().to(device)\n"
    "optim      = torch.optim.AdamW(model.parameters(), lr=args.lr,\n"
    "                               weight_decay=args.weight_decay, betas=(0.9, 0.999))\n"
    "sched      = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)\n"
    "\n"
    "use_amp = (not args.no_amp) and device.type == 'cuda'\n"
    "scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)\n"
    "print('AMP:', use_amp)\n"
))

cells.append(md("## 6. (Optional) Resume from a previous checkpoint\n"))
cells.append(code(
    "def safe_load_resume(model, optim, sched, scaler, ckpt_path, device):\n"
    "    ck = torch.load(ckpt_path, map_location=device)\n"
    "    sd = ck['model']\n"
    "    own = model.state_dict()\n"
    "    if set(sd.keys()) != set(own.keys()):\n"
    "        missing    = sorted(set(own.keys()) - set(sd.keys()))[:5]\n"
    "        unexpected = sorted(set(sd.keys()) - set(own.keys()))[:5]\n"
    "        raise RuntimeError(f'Arch mismatch.\\n missing: {missing}\\n unexpected: {unexpected}')\n"
    "    model.load_state_dict(sd, strict=True)\n"
    "    optim.load_state_dict(ck['optim'])\n"
    "    sched.load_state_dict(ck['sched'])\n"
    "    if scaler is not None and ck.get('scaler') is not None:\n"
    "        scaler.load_state_dict(ck['scaler'])\n"
    "    return ck.get('epoch', 0) + 1, ck.get('best_val', float('inf'))\n"
    "\n"
    "start_epoch, best_val = 1, float('inf')\n"
    "if args.resume and os.path.isfile(args.resume):\n"
    "    start_epoch, best_val = safe_load_resume(model, optim, sched, scaler, args.resume, device)\n"
    "    print(f'resumed from {args.resume} @ epoch {start_epoch-1}, best_val={best_val:.4f}')\n"
    "else:\n"
    "    print('starting fresh')\n"
))

cells.append(md("## 7. Training loop\n"))
cells.append(code(
    "def run_epoch(loader, optim=None):\n"
    "    is_train = optim is not None\n"
    "    model.train(is_train)\n"
    "    sums = {'loss': 0.0, 'l1': 0.0, 'sc': 0.0, 'mag': 0.0}\n"
    "    n = 0\n"
    "    for noisy, clean, _ in loader:\n"
    "        noisy = noisy.to(device, non_blocking=True)\n"
    "        clean = clean.to(device, non_blocking=True)\n"
    "        if noisy.dim() == 1:\n"
    "            noisy = noisy.unsqueeze(0); clean = clean.unsqueeze(0)\n"
    "        L = noisy.shape[-1]\n"
    "        spec = stft_front(noisy)                   # (B, 2, F, T)\n"
    "\n"
    "        with torch.set_grad_enabled(is_train):\n"
    "            if use_amp:\n"
    "                with torch.cuda.amp.autocast():\n"
    "                    pred = model(spec, length=L)\n"
    "                    loss, parts = loss_fn(pred, clean)\n"
    "            else:\n"
    "                pred = model(spec, length=L)\n"
    "                loss, parts = loss_fn(pred, clean)\n"
    "\n"
    "        if is_train:\n"
    "            optim.zero_grad(set_to_none=True)\n"
    "            if scaler.is_enabled():\n"
    "                scaler.scale(loss).backward()\n"
    "                scaler.unscale_(optim)\n"
    "                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)\n"
    "                scaler.step(optim); scaler.update()\n"
    "            else:\n"
    "                loss.backward()\n"
    "                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)\n"
    "                optim.step()\n"
    "\n"
    "        b = noisy.size(0)\n"
    "        sums['loss'] += float(loss) * b\n"
    "        sums['l1']   += float(parts['l1'])  * b\n"
    "        sums['sc']   += float(parts['sc'])  * b\n"
    "        sums['mag']  += float(parts['mag']) * b\n"
    "        n += b\n"
    "    return {k: v / max(1, n) for k, v in sums.items()}\n"
    "\n"
    "\n"
    "for epoch in range(start_epoch, args.epochs + 1):\n"
    "    t0 = time.time()\n"
    "    tr = run_epoch(train_dl, optim=optim)\n"
    "    vl = run_epoch(val_dl)\n"
    "    sched.step()\n"
    "    dt = time.time() - t0\n"
    "    print(f'epoch {epoch:03d}/{args.epochs}  '\n"
    "          f'train={tr[\"loss\"]:.4f}  val={vl[\"loss\"]:.4f}  '\n"
    "          f'l1={vl[\"l1\"]:.4f} sc={vl[\"sc\"]:.4f} mag={vl[\"mag\"]:.4f}  '\n"
    "          f'({dt:.1f}s)')\n"
    "\n"
    "    ck = {\n"
    "        'model':  model.state_dict(),\n"
    "        'optim':  optim.state_dict(),\n"
    "        'sched':  sched.state_dict(),\n"
    "        'scaler': scaler.state_dict() if use_amp else None,\n"
    "        'epoch':  epoch,\n"
    "        'best_val': best_val,\n"
    "        'args': vars(args),\n"
    "        'arch': 'AdvancedUNetSE',\n"
    "    }\n"
    "    torch.save(ck, os.path.join(args.ckpt_dir, 'last.pt'))\n"
    "    if vl['loss'] < best_val:\n"
    "        best_val = vl['loss']\n"
    "        ck['best_val'] = best_val\n"
    "        torch.save(ck, os.path.join(args.ckpt_dir, 'best.pt'))\n"
    "        print(f'  saved new best ({best_val:.4f})')\n"
    "print('done.')\n"
))

cells.append(md(
    "## 8. Quick inference / listening test\n"
    "\n"
    "Load `best.pt` and denoise one item from the val set. Saves WAV files\n"
    "you can play right in the notebook.\n"
))
cells.append(code(
    "import torch, soundfile as sf, numpy as np\n"
    "from IPython.display import Audio, display\n"
    "\n"
    "# Build a fresh model and load best\n"
    "infer = AdvancedUNetSE(\n"
    "    n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length,\n"
    "    base_channels=args.base_channels, depth=args.depth, mask_scale=args.mask_scale,\n"
    ").to(device).eval()\n"
    "ck = torch.load(os.path.join(args.ckpt_dir, 'best.pt'), map_location=device)\n"
    "infer.load_state_dict(ck['model']); print('loaded best @ epoch', ck['epoch'])\n"
    "\n"
    "# One val sample\n"
    "noisy, clean, _ = val_ds[0]\n"
    "noisy_b = noisy.unsqueeze(0).to(device)\n"
    "with torch.no_grad():\n"
    "    spec = stft_front(noisy_b)\n"
    "    enh  = infer(spec, length=noisy_b.shape[-1]).squeeze(0).cpu().numpy()\n"
    "\n"
    "sr = args.sample_rate\n"
    "sf.write('/content/noisy.wav', noisy.numpy(), sr)\n"
    "sf.write('/content/clean.wav', clean.numpy(), sr)\n"
    "sf.write('/content/enhanced.wav', enh, sr)\n"
    "\n"
    "print('noisy:');    display(Audio('/content/noisy.wav'))\n"
    "print('enhanced:'); display(Audio('/content/enhanced.wav'))\n"
    "print('clean:');    display(Audio('/content/clean.wav'))\n"
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = HERE / "train_unet_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "size", out.stat().st_size, "bytes")
print("cells:", len(cells))
