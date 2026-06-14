# Thai Speech Denoiser & Transcription Pipeline
## ระบบลดเสียงรบกวนและถอดความเสียงพูดภาษาไทย

> **AI Builders 2026** — End-to-end pipeline: data collection → model training → web demo → Jetson hardware prototype

[![HuggingFace Demo](https://img.shields.io/badge/🤗%20Demo-Thai%20Speech%20Denoiser-blue)](https://huggingface.co/spaces/cappuuuuu1234/thai-speech-denoiser)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Problem Statement

In real Thai environments — classrooms, meeting rooms, homes — **HVAC and fan noise is the dominant noise source**. These produce continuous low-frequency (LF) noise below 200 Hz, which directly overlaps with Thai vowel formants (F1 at 250–900 Hz) and causes standard ASR systems to fail.

**Why off-the-shelf models fail on Thai LF noise:**
- Whisper small/medium WER increases >30% when SNR < 5 dB
- MetricGAN+ **reduces** SNR by −3.51 dB on our test set
- Resemble Enhance (pretrained) over-suppresses the 300–4000 Hz speech band by 5× vs CRN

**Our solution:** Train CRN BiLSTM from scratch on a custom Thai noise dataset, then connect to Whisper large-v3 + Typhoon LLM for accurate Thai transcription.

---

## Benchmark Results

Measured on 16,438 test clips (SNR baseline = +5 dB):

| Model | SI-SDR Improvement | CER | Notes |
|-------|--------------------|-----|-------|
| Raw (no denoising) | +0.00 dB | 1.620 | Baseline |
| MetricGAN+ | **−3.51 dB** ❌ | 1.980 | Worse than baseline |
| **CRN BiLSTM (ours)** | **+7.79 dB** ✅ | 1.574 | Default model |
| UNet SE (ours) | +5.93 dB ✅ | 1.642 | Lightweight |
| Resemble pretrained | +9.40 dB | 1.600 | Over-suppresses speech |
| Resemble-FT | +4.58 dB | 1.643 | FT reduced generalization |
| MossFormer2-FT | +4.14 dB | 1.880 | Fine-tune didn't help |

> **Why CRN beats Resemble in practice:** Resemble over-suppresses 300–4000 Hz (speech formants) by 5× more than CRN, degrading transcription accuracy despite higher raw SNR numbers.

---

## 📥 Pre-trained Model Checkpoints

> **The model checkpoint is NOT included in this repository.** You must download it before running the local app or benchmark.

### Required file

| File | Size | Destination in repo |
|------|------|---------------------|
| `best.pt` | ~190 MB | `checkpoints/best.pt` |

### Download

[**→ Download from GitHub Releases**](https://github.com/cappu-kab/AI-builders/releases)
*(link active once the release is published)*

### Place the file

```bash
# From the repo root
mkdir -p checkpoints
# Move the downloaded file here: checkpoints/best.pt
```

The Gradio app (`app.py`) and the inference CLI (`models/crn/inference.py`) both default to `checkpoints/best.pt`. Pass `--ckpt <path>` to override.

---

## Repository Structure

```
AI-builders/
│
│  ── Web Application ──
├── app.py                          <- Entry point: launches Gradio UI (Denoise + Transcribe tabs)
├── app/
│   ├── denoise_core.py             <- CRN engine: noise gate + AGC + compressor
│   ├── model_selector.py           <- UNet / Resemble-FT lazy-load wrapper
│   └── transcript_tab.py           <- Whisper large-v3 + Typhoon LLM Thai transcription tab
│
│  ── CRN Model (primary, ~15.88M params) ──
├── models/crn/
│   ├── model.py                    <- Architecture: 5× Encoder, BiLSTM, LF-Bypass
│   ├── train.py                    <- Training with 10-term multi-objective loss
│   ├── inference.py                <- CLI batch inference
│   ├── dataset.py                  <- On-the-fly noise mixing (70% LF + 30% HF)
│   ├── evaluate.py                 <- Per-file SNR / SI-SDR / CER metrics
│   └── losses.py                   <- Huber + Multi-Res STFT + LF Boost (5.0×)
│
│  ── UNet Baseline (2.14M params) ──
├── models/unet/
│   ├── model_unet_advanced.py      <- Depthwise-sep ResidualBlocks + GatedSkip
│   ├── train.py                    <- UNet training script
│   ├── dataset.py                  <- Shared augmentation pipeline (same as CRN)
│   ├── losses.py                   <- Identical loss functions
│   ├── inference.py                <- UNet inference
│   ├── evaluate.py                 <- UNet evaluation
│   └── Unet_Colab_Master.ipynb    <- Colab training notebook
│
│  ── Data Pipeline ──
├── data_prep/
│   ├── preprocess.py               <- Raw audio → 16 kHz mono float32 .npy
│   ├── scrape.py                   <- YouTube Thai noise scraper
│   ├── Filter low freq noise.py    <- LF noise pool curation
│   ├── Clean lfn dataset.py        <- Dataset cleaner
│   └── Lfn dataset pipeline.py    <- Noise-augmented dataset builder
│
│  ── Training Orchestrator ──
├── training/
│   ├── run.py                      <- Master CLI: preprocess / smoke / train / evaluate
│   ├── finetune_resemble.py        <- Resemble Enhance fine-tuning
│   └── finetune_mossformer.py      <- MossFormer2 fine-tuning
│
│  ── Evaluation ──
├── evaluation/
│   ├── benchmark.py                <- 7-model unified benchmark
│   └── eval.py                     <- Per-file SNR, SI-SDR, CER metrics
│
│  ── ANC Simulation ──
├── simulation/
│   ├── file_simulator.py           <- File-based ANC simulator (no hardware needed)
│   ├── predictor.py                <- Causal TCN look-ahead predictor (5 ms)
│   ├── model_runtime.py            <- CRN / UNet streaming runtime
│   └── visualizer.py              <- Cancellation visualization
│
│  ── Jetson Hardware Pipeline ──
├── hardware/
│   ├── main_jetson_pipeline.py     <- Entry point: Jetson Orin Nano ANC pipeline
│   ├── crn_pipeline.py             <- Jetson + ESP32-S3 UART pipeline logic
│   └── anc_latency.py              <- Round-trip latency measurement
│
│  ── Scripts & Utilities ──
├── scripts/
│   ├── export_onnx.py              <- CRN → ONNX (opset 15, fixed-shape)
│   └── test_fusion_pipeline*.py    <- CRN + TCN streaming fusion tests
├── build_trt_engines.sh            <- TensorRT FP16 engine build (run on Jetson)
│
│  ── Utility & Analysis Tools ──
├── tools/
│   ├── _smoke_test.py              <- Pipeline smoke test
│   ├── baselines.py                <- Baseline comparisons
│   ├── band_eval.py                <- Per-frequency-band evaluation
│   ├── benchmark_crn_onnx.py       <- CRN ONNX runtime benchmark
│   ├── make_examples.py            <- Generate Gradio example audio clips
│   ├── export_crn_onnx.py          <- Standalone CRN ONNX export (200 ms context)
│   └── tune_compare.py             <- AGC tuning utility
│
│  ── HuggingFace Deployment ──
├── hf_deploy/                      <- Snapshot deployed to HF Spaces
│
│  ── Documentation ──
└── docs/
    ├── PROGRESS.md                 <- Day-by-day development log
    ├── HANDOFF.md                  <- Technical handoff document
    ├── FINAL_PROJECT_SUMMARY.md    <- Full architecture spec with hyperparameter tables
    ├── PREDICTOR.md                <- TCN predictor design notes
    ├── Simulation_README.md        <- ANC simulation guide
    ├── crn_pipeline_experiment_log.md <- CRN hardware iteration log
    └── model_narrative.md          <- Model design narrative
```

---

## Architecture: CRN BiLSTM

```
Noisy Waveform -> STFT(n_fft=512, hop=128) -> (B, 2, 257, T)
    |
    5x Encoder Blocks -- Conv2d(5x3) + BatchNorm + ELU + SE Attention
    channels: 2->32->64->128->256->512, stride-2 in freq
    |
    Bottleneck SE (global channel attention, separates LF vs HF)
    |
    BiLSTM: 2 layers, hidden=256, bidirectional (output=512)
    |
    5x Decoder Blocks -- ConvTranspose2d + skip connections
    |
    LF-Bypass Branch <- zero-init gradient path for bins 0-6 (0-187.5 Hz)
    |
    Magnitude-Bounded Mask (tanh x 1.2)
    |
    enhanced = noisy - iSTFT(mask * noisy_spec)
```

**Post-processing chain** (tuned for Thai LF noise, implemented in `app/denoise_core.py`):
1. **Ratio noise gate** — detects noise-only frames via `rms(output)/rms(input)` ratio; instant open, smoothed close
2. **Long-window AGC** (100 ms frames) — boosts quiet speech sections, never attenuates
3. **Soft-knee compressor** (3:1 ratio, -18 dBFS threshold) — narrows dynamic range
4. **Dry-mix** = 0.05 — blends 5% original back to prevent over-suppression

---

## Dataset

> **Training data is NOT included in this repository.** Download the source datasets independently before running training.

| Source | Train clips | Where to get it |
|--------|-------------|-----------------|
| Mozilla Common Voice 17 (Thai) | 32,785 | [commonvoice.mozilla.org](https://commonvoice.mozilla.org/en/datasets) |
| Thai Elderly Speech | 13,440 | Contact dataset authors |
| Thai Isan Dialect | 6,990 | Contact dataset authors |
| Google FLEURS (Thai) | 2,602 | HuggingFace `datasets`: `google/fleurs` (split: `th_th`) |
| **Total** | **55,817 train / 15,855 val / 16,438 test** | — |

**Noise mixing:** 70% LF (HVAC/fan, 50/60 Hz harmonics) + 30% HF (click/hiss), SNR -5 to +15 dB, curriculum learning (easy → hard in first 15 epochs).

---

## Setup

### Web Demo (no install needed)

[https://huggingface.co/spaces/cappuuuuu1234/thai-speech-denoiser](https://huggingface.co/spaces/cappuuuuu1234/thai-speech-denoiser)

### Local Gradio App

```bash
git clone https://github.com/cappu-kab/AI-builders
cd AI-builders

# Create virtual environment
python -m venv .venv

# Activate — choose your OS:
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (Command Prompt)
.venv\Scripts\Activate.ps1       # Windows (PowerShell)

pip install torch torchaudio gradio scipy librosa soundfile huggingface_hub matplotlib pillow

# Download checkpoint first (see "Pre-trained Model Checkpoints" above)
# Place it at: checkpoints/best.pt

python app.py
```

The **Denoise** tab works immediately. The **Transcribe** tab (Whisper large-v3 + Typhoon) requires `HF_TOKEN` — see below.

### HF_TOKEN (required for Transcription tab only)

The transcription tab pulls gated models from HuggingFace Hub and requires an access token.

**Get your token:**
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → select **Read** role → copy the token string

**Set the token before running `app.py`:**

```bash
# macOS / Linux
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# Windows — Command Prompt
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# Windows — PowerShell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxx"
```

Or create a `.env` file in the repo root (add `.env` to `.gitignore` — never commit tokens):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

### Train CRN from Scratch

> **Requires the 88K-clip Thai dataset** (not included — download Common Voice 17 Thai and FLEURS first, then run preprocessing).

```bash
# Step 1 — preprocess raw audio -> .npy
python training/run.py preprocess --in_root ./raw_audio --out_root ./data_npy

# Step 2 — smoke test (1 epoch, 5 steps — verify pipeline before full run)
python training/run.py smoke --project crn

# Step 3 — full training
cd models/crn
python train.py \
  --data_root ../../data_npy \
  --epochs 60 --batch_size 16 \
  --base_channels 32 --bottleneck_dim 512 \
  --rnn_hidden 256 --rnn_layers 2 --rnn_type bilstm \
  --scheduler warmrestarts --cosine_T0 15 \
  --curriculum_epochs 10 --low_freq_boost 8.0 \
  --lf_noise_ratio 0.70
```

Checkpoint saved to `checkpoints/best.pt` (highest SI-SDR on validation set).

### Run Benchmark

```bash
python evaluation/benchmark.py --max_files 200
# Results -> evaluation/benchmark_outputs/summary_table.csv
```

### ANC File Simulator (no hardware needed)

```bash
python simulation/file_simulator.py dirty.wav -m crn -o cancelled.wav
```

### Jetson Hardware Pipeline

> **Note:** The hardware pipeline script (`hardware/main_jetson_pipeline.py`) is currently a structural stub for integration testing. Actual microphone and UART integration are pending. All hardware I/O functions (Spresense serial, USB sound card, CRN inference) return dummy zeros until physical wiring is complete.

```bash
# Run on the Jetson Orin Nano:

# Step 1 — build TRT FP16 engine
bash build_trt_engines.sh

# Step 2 — run pipeline skeleton (hardware I/O stubs — not yet wired)
python hardware/main_jetson_pipeline.py
```

---

## Hardware Prototype

> **Note:** The hardware pipeline script (`hardware/main_jetson_pipeline.py`) is currently a structural stub for integration testing. Actual microphone and UART integration are pending.

**Stack:** Sony Spresense / ESP32-S3 (mic capture) → UART → Jetson Orin Nano (CRN inference) → speaker

| Stage | Latency | Budget | Status |
|-------|---------|--------|--------|
| ORT baseline (3 s context, CPU) | 193.64 ms | 16 ms | Failed |
| Context reduction (200 ms) | ~17 ms | 16 ms | Marginal |
| **TRT FP16 engine** | **11.02 ms** | 16 ms | Pass |
| Python ctypes wrapper | 17.04 ms | 16 ms | Marginal (1.2×) |

**Key constraint:** BiLSTM requires ~3 s context for accurate masking, making true real-time ANC (< 20 ms) an architectural trade-off. Project pivoted to offline speech enhancement + web demo using the same model — decision backed by benchmark data.

---

## Critical Bugs Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| `lf_bins=16` instead of 7 | SI-SDR plateau at ~8 dB for 3 weeks | `round(200 / hz_per_bin) + 1 = 7` |
| Bandpass filter on noisy only | Impossible training targets | Removed entirely |
| `bf16` SI-SDR overflow | Wrong val metrics | Cast to float32 before computing |
| Val noise != train distribution | Val flatline | Val uses same 70/30 LF+HF as train |
| WarmupWrap + OneCycleLR double warmup | LR crash at step 501 | `_warmup=0` when `scheduler=onecycle` |
| NaN in noise pool | Training divergence | Filter non-finite values on load |

---

## Not Included in This Repo

- Model checkpoints (`*.pt`, `*.pth`, `*.onnx`) — distribute via [Releases](https://github.com/cappu-kab/AI-builders/releases) / HuggingFace
- Raw datasets and audio files (`*.wav`, `data_npy/`) — gitignored
- Large dataset directories (Common Voice, FLEURS, UrbanSound8K, MS-SNSD)

---

## Documentation

Full technical documentation lives in [`docs/`](docs/):

| Document | Description |
|----------|-------------|
| [`docs/FINAL_PROJECT_SUMMARY.md`](docs/FINAL_PROJECT_SUMMARY.md) | Full architecture spec with hyperparameter tables |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | Day-by-day development log |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | Technical handoff document |
| [`docs/crn_pipeline_experiment_log.md`](docs/crn_pipeline_experiment_log.md) | CRN hardware iteration log |
| [`docs/Simulation_README.md`](docs/Simulation_README.md) | ANC simulation guide |
| [`docs/PREDICTOR.md`](docs/PREDICTOR.md) | TCN predictor design notes |

---

## Project Info

**AI Builders 2026** — Thai AI development program
**Repo:** [github.com/cappu-kab/AI-builders](https://github.com/cappu-kab/AI-builders)
**HuggingFace:** [cappuuuuu1234/thai-speech-denoiser](https://huggingface.co/spaces/cappuuuuu1234/thai-speech-denoiser)
