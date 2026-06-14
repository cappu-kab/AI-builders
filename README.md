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

## Repository Structure

```
AI-builders/
│
│  ── Root: Local web app (Gradio) ──
├── app.py                       <- Multi-tab Gradio UI (Denoise + Transcribe)
├── denoise_core.py              <- CRN engine: noise gate + AGC + compressor
├── model_selector.py            <- UNet / Resemble-FT lazy-load wrapper
├── transcript_tab.py            <- Whisper large-v3 + Typhoon LLM Thai tab
├── inference.py                 <- CRN CLI batch inference
├── tune_compare.py              <- AGC tuning utility
│
│  ── Data pipeline ──
├── data_prep/
│   ├── preprocess.py            <- Raw audio -> 16kHz mono float32 .npy
│   ├── scrape.py                <- YouTube Thai noise scraper
│   ├── Filter low freq noise.py <- LF noise pool curation
│   └── Lfn dataset pipeline.py  <- Noise-augmented dataset builder
│
│  ── Models ──
├── models/crn/                  <- CRN BiLSTM (15.88M params, default)
│   ├── model.py                 <- Architecture: 5x Encoder, BiLSTM, LF-Bypass
│   ├── train.py                 <- Training with 10-term multi-objective loss
│   ├── inference.py             <- Model loader
│   ├── dataset.py               <- On-the-fly noise mixing (70% LF + 30% HF)
│   └── losses.py                <- Huber + Multi-Res STFT + LF Boost (8.0x)
├── models/unet/                 <- AdvancedUNetSE (2.15M params, lightweight)
│   ├── model_unet_advanced.py   <- Depthwise-sep ResidualBlocks + GatedSkip
│   └── Unet_Colab_Master.ipynb  <- Colab training notebook
│
│  ── Training orchestrator ──
├── training/
│   ├── run.py                   <- Master: preprocess / smoke / train / evaluate
│   ├── finetune_resemble.py     <- Resemble Enhance fine-tuning
│   └── finetune_mossformer.py   <- MossFormer2 fine-tuning
│
│  ── Evaluation ──
├── evaluation/
│   ├── benchmark.py             <- 7-model unified benchmark
│   └── eval.py                  <- Per-file SNR, SI-SDR, CER metrics
├── baselines.py                 <- Baseline comparisons
├── band_eval.py                 <- Per-frequency-band evaluation
├── benchmark_crn_onnx.py        <- CRN ONNX runtime benchmark
│
│  ── Simulation & Hardware ──
├── simulation/
│   ├── file_simulator.py        <- File-based ANC simulator
│   ├── predictor.py             <- Causal TCN look-ahead predictor (5ms)
│   ├── model_runtime.py         <- CRN/UNet streaming runtime
│   └── visualizer.py            <- Cancellation visualization
├── crn_pipeline.py              <- Jetson Orin Nano + ESP32-S3 UART pipeline
├── main_jetson_pipeline.py      <- Jetson ANC hardware entry point
├── anc_latency.py               <- Round-trip latency measurement
│
│  ── Scripts & Utilities ──
├── scripts/
│   ├── export_onnx.py           <- CRN -> ONNX (opset 15, fixed-shape)
│   └── test_fusion_pipeline*.py <- CRN+TCN streaming fusion tests
├── export_crn_onnx.py           <- Standalone CRN ONNX export (200ms context)
├── simulate_anc_math.py         <- ANC algorithm math simulation
├── make_examples.py             <- Generate Gradio example audio clips
├── _smoke_test.py               <- Pipeline smoke test
│
│  ── Docs ──
├── docs/                        <- PREDICTOR.md, simulation history
├── PROGRESS.md                  <- Day-by-day development log
├── FINAL_PROJECT_SUMMARY.md     <- Full project summary
├── HANDOFF.md                   <- Technical handoff document
└── crn_pipeline_experiment_log.md <- CRN hardware iteration log
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

**Post-processing chain** (tuned for Thai LF noise, implemented in `denoise_core.py`):
1. **Ratio noise gate** — detects noise-only frames via `rms(output)/rms(input)` ratio; instant open, smoothed close
2. **Long-window AGC** (100ms frames) — boosts quiet speech sections, never attenuates
3. **Soft-knee compressor** (3:1 ratio, -18 dBFS threshold) — narrows dynamic range
4. **Dry-mix** = 0.05 — blends 5% original back to prevent over-suppression

---

## Dataset

| Source | Train clips | Notes |
|--------|-------------|-------|
| Mozilla Common Voice 17 (Thai) | 32,785 | Crowdsourced read speech |
| Thai Elderly Speech | 13,440 | Elderly speakers, varied prosody |
| Thai Isan Dialect | 6,990 | Regional dialect — high ASR failure rate |
| Google FLEURS (Thai) | 2,602 | Broadcast quality |
| **Total** | **55,817 train / 15,855 val / 16,438 test** | 88,110 clips |

**Noise mixing:** 70% LF (HVAC/fan, 50/60 Hz harmonics) + 30% HF (click/hiss), SNR -5 to +15 dB, curriculum learning (easy to hard in first 15 epochs).

---

## Setup

### Web Demo (no install needed)

[https://huggingface.co/spaces/cappuuuuu1234/thai-speech-denoiser](https://huggingface.co/spaces/cappuuuuu1234/thai-speech-denoiser)

### Local Gradio App

```bash
git clone https://github.com/cappu-kab/AI-builders
cd AI-builders

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install torch gradio scipy librosa soundfile huggingface_hub matplotlib pillow

# Requires Run/CRN/checkpoints/best.pt (190MB — see Releases)
python app.py
```

Set `HF_TOKEN` env var to enable Whisper + Typhoon transcription tab.

### Train CRN from Scratch

```bash
# Preprocess audio -> .npy
python training/run.py preprocess --in_root ./raw_audio --out_root ./data_npy

# Smoke test (1 epoch, 5 steps)
python training/run.py smoke --project crn

# Full training
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

### Run Benchmark

```bash
python evaluation/benchmark.py --max_files 200
# Results -> evaluation/benchmark_outputs/summary_table.csv
```

### ANC File Simulator

```bash
python simulation/file_simulator.py dirty.wav -m crn -o cancelled.wav
```

---

## Hardware Prototype

**Stack:** Sony Spresense / ESP32-S3 (mic capture) -> UART -> Jetson Orin Nano (CRN inference) -> speaker

| Stage | Latency | Budget | Status |
|-------|---------|--------|--------|
| ORT baseline (3s context, CPU) | 193.64 ms | 16 ms | Failed |
| Context reduction (200ms) | ~17 ms | 16 ms | Marginal |
| **TRT FP16 engine** | **11.02 ms** | 16 ms | Pass |
| Python ctypes wrapper | 17.04 ms | 16 ms | Marginal (1.2x) |

**Key constraint:** BiLSTM requires ~3s context for accurate masking, making true real-time ANC (< 20ms) an architectural trade-off. Project pivoted to offline speech enhancement + web demo using the same model — decision backed by benchmark data.

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

- Model checkpoints (`*.pt`, `*.pth`, `*.onnx`) — distribute via Releases / HuggingFace
- Raw datasets and audio files (`*.wav`, `data_npy/`) — gitignored
- Large dataset directories (Common Voice, FLEURS, UrbanSound8K, MS-SNSD)

---

## Project Info

**AI Builders 2026** — Thai AI development program  
**Repo:** [github.com/cappu-kab/AI-builders](https://github.com/cappu-kab/AI-builders)  
**HuggingFace:** [cappuuuuu1234/thai-speech-denoiser](https://huggingface.co/spaces/cappuuuuu1234/thai-speech-denoiser)
