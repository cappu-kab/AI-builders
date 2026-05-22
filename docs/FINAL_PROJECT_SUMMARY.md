# Speech Denoising Project — Technical Summary

---

## 1. Project Overview & Core Principle

This project implements a real-time-capable **Thai speech enhancement system** that removes background noise from a recorded speech signal while preserving the speaker's voice with high intelligibility and naturalness.

The project contains **two complete model implementations** trained on the same data and evaluated with the same pipeline, enabling a direct architectural comparison:

| System | Location | Architecture | Parameters |
|---|---|---|---|
| **CRN** (primary) | root folder | Convolutional Recurrent Network + BiLSTM | ~4–6 M |
| **U-Net** (baseline) | `speech_denoise/` | Advanced U-Net + dilated bottleneck | ~2.1 M |

Both models share the same **residual noise prediction** paradigm: rather than attempting to directly predict what clean speech looks like, the model learns to predict the *noise component* of the mixture. The clean output is recovered by subtraction:

```
enhanced = noisy_waveform − predicted_noise
```

At initialisation the noise mask is near zero, so the model's output defaults to passing the input through unchanged — a far better starting point than predicting silence, and the key reason training is stable from epoch one.

**What both models process:**
- **Input:** A mono 16 kHz waveform containing speech corrupted by additive noise (traffic, HVAC, crowd, electronic hum, etc.)
- **Output:** A mono 16 kHz waveform with the noise component suppressed and the speech preserved

**Primary design goals, in priority order:**
1. Suppress continuous background noise, especially low-frequency hum below 200 Hz
2. Preserve phonetic structure, formant boundaries, and consonant clarity
3. Avoid over-suppression — muting quiet speech is treated as a failure mode

---

## 2. Primary Model: CRN Architecture (`model.py`)

### Overview

The primary model is a **Complex-Spectrogram Convolutional Recurrent Network (CRN)** with U-Net-style skip connections. It operates on the full complex STFT representation — both magnitude and phase — which is essential for reconstructing perceptually correct audio from an inverse STFT.

```
Noisy waveform (B, T)
       |
    STFT -> complex (B, F, T') -> stack real+imag -> (B, 2, F, T')
       |
  +----+----------------------------------------------------------+
  |  CNN Encoder x5                                              |
  |  Conv2d(kernel 5x3, stride 2 along freq) + BN + ELU         |
  |  + SE channel attention after each block                     |
  |  Channels: 2 -> 16 -> 32 -> 64 -> 128 -> 256                |
  |  Freq bins: 257 -> 129 -> 65 -> 33 -> 17 -> 9 (bottleneck)  |
  +----+----------------------------------------------------------+
       |  (B, 256, 9, T') -- skip connections cached at each level
  proj_in: Linear(2304 -> 384)   <- compress before recurrent
       |
  +----+-----------------+
  |  2-Layer BiLSTM      |  hidden=256, bidirectional
  |  output dim=512      |  (forward + backward concatenated)
  +----+-----------------+
  proj_out: Linear(512 -> 2304)  <- restore to encoder feature dim
       |
  +----+----------------------------------------------------------+
  |  CNN Decoder x5                                              |
  |  ConvTranspose2d + skip-concat from encoder                  |
  |  Channels: 512 -> 256 -> 128 -> 64 -> 32 -> 2 (re/im mask)  |
  +----+----------------------------------------------------------+
       |
  Magnitude-bounded complex mask (mask_bound=1.2)
       |  |mask| <= 1.2 x |noisy|  -- prevents mask inversion
       |
  noise_spec = mask x noisy_complex   (complex multiply)
  noise_wave = iSTFT(noise_spec)
       |
  enhanced = noisy_waveform - noise_wave
```

### Key Architectural Decisions

**Squeeze-and-Excitation (SE) attention** is applied after each encoder block. It learns per-channel importance weights from global average pooling at negligible parameter cost (~22K total), allowing the encoder to focus on the most informative frequency-channel combinations for each input.

**U-Net skip connections** feed the full-resolution encoder activations directly into the corresponding decoder level via concatenation. This gives the decoder direct access to the input's fine-grained spectral structure without having to reconstruct it from the compressed bottleneck — essential for preserving sibilants and fricatives that are easily lost in the bottleneck.

**Magnitude-bounded complex mask** (`mask_bound=1.2`): the mask magnitude is normalised and clamped via `tanh(|mask|) x 1.2`. This bounds how much energy the model can "predict as noise" to 1.2x the noisy input energy per bin, making it physically impossible for the model to invert and start treating clean speech as noise.

### The Bottleneck Upgrade: BiLSTM + Projections

The original CRN bottleneck used a single unidirectional GRU. The final architecture replaces it with a **2-layer Bidirectional LSTM** (hidden size 256 per direction, 512-dimensional output).

**Why BiLSTM over GRU:** Bidirectional processing means every time step has access to the *full sequence context* in both directions. This is critical for distinguishing sustained low-frequency noise (present throughout) from voiced speech that starts and ends — a distinction a unidirectional model can only approximate.

**Why the projection layers are non-negotiable:** Without `proj_in`, the LSTM input would be 2304-dimensional (256 channels x 9 frequency bins). A two-layer bidirectional LSTM at 2304-dim input would require approximately 38M parameters for the first layer alone. `proj_in: Linear(2304 -> 384)` compresses the feature dimension by 6x before the recurrent layer; `proj_out: Linear(512 -> 2304)` restores it after. The BiLSTM itself operates at the manageable 384 -> 512 scale.

---

## 3. Baseline Model: Ultra-Lightweight U-Net (`speech_denoise/`)

### Motivation

The U-Net baseline exists to answer a direct question: *how much of the CRN's performance comes from the BiLSTM temporal modelling, and how much comes from the shared architecture primitives (residual noise prediction, complex masking, multi-objective loss)?* By training both models on identical data with identical loss functions and evaluation protocols, the comparison isolates the value of the recurrent bottleneck.

### Architecture Overview

The U-Net baseline (`model_unet_advanced.py`) is a **pure convolutional encoder-decoder** with the same residual noise-prediction interface as the CRN. It accepts raw waveforms `(B, T)` and returns enhanced waveforms `(B, T)`; STFT and iSTFT live inside the model.

```
Noisy waveform (B, T)
       |
    Internal STFT -> (B, 2, F, T)  [fp32, AMP-safe]
       |
  +----+----------------------------------------------------------+
  |  Encoder x5  (Depthwise-Separable ResidualConvBlock)         |
  |  Freq-only downsampling: stride (2,1) -> time preserved       |
  |  Channels: 2 -> 32 -> 64 -> 128 -> 256 -> 256               |
  |  Each skip saved BEFORE downsampling (full resolution)        |
  +----+----------------------------------------------------------+
       |
  +----+----------------------------------------------------------+
  |  Dilated Bottleneck  (already DS convolutions)               |
  |  Sweep 1: dilations (1,2,4,8)   + SE attention               |
  |  Sweep 2: dilations (1,2,4,8,16) + SE attention              |
  +----+----------------------------------------------------------+
       |
  +----+----------------------------------------------------------+
  |  Decoder x5  (Gated skip merge + DS ResidualConvBlock)       |
  |  ConvTranspose2d frequency upsample                           |
  |  Gated skip: sigmoid gate controls encoder info admission     |
  +----+----------------------------------------------------------+
       |
  Magnitude-bounded complex mask (mask_bound=1.2)  [identical logic to CRN]
       |
  noise_spec = mask x noisy_spec
  noise_wav  = iSTFT(noise_spec)
       |
  enhanced = noisy_waveform - noise_wav
```

### Extreme Channel Thinning via Depthwise Separable Convolutions

The primary parameter-reduction technique is replacing every standard `Conv2d(5x3)` in the encoder and decoder `ResidualConvBlock` with a **Depthwise-Separable (DS)** equivalent. The bottleneck already used DS convolutions; this change extends it throughout the entire network.

Each standard `Conv2d(in, out, 5x3)` block is replaced with a PW+DW pair:

```
Standard:  in x out x 5 x 3 params per conv
DS conv1:  PW(in->out, 1x1)  +  DW(out, 5x3, groups=out)
DS conv2:  DW(out, 5x3, groups=out)  +  PW(out->out, 1x1)
```

**Parameter savings at the largest stage (out_ch=256):**

| Layer | Standard Conv2d | DS Equivalent | Reduction |
|---|---|---|---|
| conv1 (128->256, 5x3) | 491,520 | 36,608 | **13x** |
| conv2 (256->256, 5x3) | 983,040 | 69,376 | **14x** |

With `base_channels=32` and `depth=5`, the total network reaches **2.14M parameters** — compared to the CRN's ~4–6M — while maintaining the same 256-channel bottleneck capacity for a fair architectural comparison.

### Gated Skip Connections

Unlike the CRN's simple concatenation, the U-Net decoder uses a learnable **sigmoid gate** at each skip connection:

```
g = sigmoid( Conv1x1( cat([decoder_feat, encoder_skip]) ) )
output = decoder_feat + g * Conv1x1(encoder_skip)
```

The gate learns to suppress encoder information that leakage noise rather than useful structure — particularly important in later training epochs when the decoder has learned to reconstruct fine spectral detail from the bottleneck alone.

### Memory-Aware Training Configuration

A U-Net holds all 5 encoder skip tensors in VRAM simultaneously throughout the forward and backward pass. This structural property means per-sample memory cost is higher than the sequential CRN at the same batch size. The training defaults are therefore adjusted to maintain an equivalent training budget while staying within 8 GB VRAM:

| Setting | CRN | U-Net | Equivalence |
|---|---|---|---|
| `batch_size` | 24 | 16 | — |
| `steps_per_epoch` | 180 | 270 | 180x24 = 270x16 = **4,320 clips/epoch** |
| `val_max_batches` | 19 | 29 | 19x24 ≈ 29x16 = **~456 clips** |
| `segment_seconds` | 3.0 | 3.0 | identical |

The clip-per-epoch and validation budgets are mathematically identical; only the batch size and step count differ. This ensures training throughput and data exposure are equated, not just wall-clock time.

### Experiment Isolation (`--exp_name`)

The U-Net training script accepts a `--exp_name` argument that appends a name to the checkpoint directory (e.g., `checkpoints/unet_base/`). This prevents checkpoint overwriting when running multiple experiments with different hyperparameters (depth, base_channels, loss weights) and makes it straightforward to resume any specific run.

---

## 4. Data Pipeline & Augmentation (`dataset.py`)

The same dataset and augmentation pipeline is shared by both models. No pre-mixed audio is stored on disk. Every training item is generated fresh at `__getitem__` time, meaning the model sees a different noisy realisation of every clean utterance each epoch.

**Per-item pipeline:**

1. Load a clean speech `.npy` file
2. Apply **time stretching** (p=0.7, rate in [0.85, 1.15]) and **pitch shifting** (p=0.5, +-3 semitones) to the clean signal
3. Crop or pad to the fixed segment length (3 seconds)
4. Grab 1-3 noise sources from the noise pool, blend them at random relative levels
5. Apply noise shaping (see below)
6. Mix clean + noise at a randomly sampled SNR using the biased sampler
7. Apply post-mix augmentations to the noisy signal only: room impulse response, random gain, soft-clipping, random EQ (+-3 dB, 2-3 bands), bandpass filtering, frequency masking

### Noise Shaping for Low-Frequency Focus

**`color_noise`** (p=0.7): spectrally tilts the noise by multiplying FFT amplitudes by `|f|^alpha`, where alpha in [-1.0, 0.0]. This smoothly interpolates between white noise (alpha=0), pink noise (alpha=-0.5), and **brown noise (alpha=-1.0, PSD proportional to 1/f^2)**. Brown noise concentrates almost all its energy below 500 Hz, making it an ideal training signal for the low-frequency suppression objective.

**`inject_low_freq_noise`** (p=0.35): generates a fresh brown-to-pink noise signal (alpha in [-1.0, -0.5]) and adds it additively to the existing noise at 30-80% of the current noise RMS. Unlike `color_noise` (which reshapes the existing spectrum), this function guarantees that sub-200 Hz energy is present in a training mixture regardless of the primary noise colour.

### Disabled: Partial Dropout

`partial_dropout` is set to `p=0.0` (disabled). Zeroing out contiguous chunks of raw samples creates **broadband vertical transients** in the STFT that have no natural phase structure. The noise prediction loss penalises these transients, but because they span all frequencies simultaneously, the gradient pushes the noise mask toward suppressing surrounding speech energy — the exact target-leakage pattern that caused mask inversion in early training runs.

### Biased SNR Sampling

Rather than sampling SNR uniformly over [-5, +15] dB, the sampler draws 50% of batches from the "hard zone" [-5, +5] dB — the range where noise is loud enough to seriously degrade speech quality and where gradient signal is most informative.

---

## 5. Training Strategy & Loss Functions (`train.py`, `losses.py`)

Both models share the same loss function, scheduler, and checkpoint strategy. The description below applies equally to both.

### Multi-Objective Composite Loss (`DenoiseLoss`)

| Term | Weight | Purpose |
|---|---|---|
| **Huber(enhanced, clean)** | 1.0 | Core waveform speech quality (delta=0.5) |
| **Huber(noise_pred, noise_tgt)** | 1.0 | Noise removal quality; combined weight 2.0 strongly anchors residual learning |
| **MRSTFT spectral convergence** | 0.1 | STFT structural accuracy at all resolutions |
| **MRSTFT log-magnitude L1** | 0.1 | Spectral envelope matching |
| **SI-SDR** | 0.1 | Phase-aware, scale-invariant speech quality |
| **Silence penalty** | 0.1 | Suppresses residual noise in segments where clean target is silent (< -40 dBFS) |
| **Noise STFT L1** | 0.5 | Frequency-weighted spectral supervision on the noise estimate |
| **Anti-collapse penalty** | 0.03 | -mean|noise_pred| penalises near-zero predictions |
| **Speech floor penalty** | 0.2 | `relu(E[clean^2] - E[enhanced^2])` fires only when enhanced energy < clean energy |
| **ASR Perceptual Loss** | 0.05 | Feature-matching on frozen Wav2Vec2 CNN encoder activations |

### Frequency-Weighted STFT Loss (< 200 Hz Focus)

The Multi-Resolution STFT Loss is applied at three FFT sizes (256, 512, 1024 points). All three resolutions carry a **5x penalty multiplier for bins below 200 Hz**:

| FFT size | Bin width | 200 Hz cutoff bin | Bins boosted |
|---|---|---|---|
| 256-pt | 62.5 Hz/bin | bin 3 | 4 of 129 |
| 512-pt | 31.25 Hz/bin | bin 6 | 7 of 257 |
| 1024-pt | 15.625 Hz/bin | bin 12 | 13 of 513 |

The boost is applied via two separate weight tensors: `_sc_weight` (spectral convergence term) and `_mag_weight` (log-magnitude L1 term), ensuring both sub-terms penalise low-frequency errors independently.

### Wav2Vec2 Perceptual Loss

The `ASRPerceptualLoss` module loads only the CNN feature extractor from `WAV2VEC2_BASE` (7 Conv1d layers, ~350K parameters, stride 320). The transformer encoder (94M parameters) is discarded. The extractor runs frozen in `eval()` mode; L1 is computed on its activations for enhanced vs. clean signals.

### SNR Curriculum Learning

| Phase | Epochs (default) | SNR range | Rationale |
|---|---|---|---|
| **Easy** | 1-4 | [+5, +15] dB | Basic separation at high SNR before hard cases |
| **Medium** | 5-8 | [0, +10] dB | Speech still dominant |
| **Hard** | 9+ | [-5, +15] dB | Full range including near-equal-energy mixtures |

### Scheduler & Checkpointing

**ReduceLROnPlateau**: LR starts at 2e-4, halved every 3 epochs with no validation improvement, floor 1e-6.

**SI-SDR checkpointing**: `best.pt` is saved based on the lowest `val_sisnr` loss (highest SI-SDR in dB), not lowest composite loss. The composite loss includes auxiliary terms that do not directly reflect perceived audio quality. SI-SDR is phase-aware and scale-invariant. The `_si_sdr` implementation uses `-tanh(SI-SDR/20)` for smooth gradients across the full dB range.

---

## 6. Thai-Optimised Evaluation Pipeline (`evaluate.py`)

The evaluation script was completely rewritten to be specific to Thai speech enhancement. It evaluates three systems — raw noisy, Wiener filter (classical baseline), and the trained CRN — across four complementary metrics, with automatic generation of qualitative outputs for reporting.

### 6.1 Balanced Dataset Sampling

**The problem:** Test files sorted alphabetically are dominated by whichever source dataset names come first — in this case `commonvoice` files, leaving three source datasets entirely unrepresented in the evaluation.

**The solution:** Files are grouped by source dataset before any sampling occurs:

```
all_clean/*.npy
      |
  _detect_source(filename)  -- keyword matching
      |
  +----------------+--------+----------+------------+
  | commonvoice    | fleurs | thai_isan | thai_elder |
  +----------------+--------+----------+------------+
         |
  random.seed(42)  -- fixed seed for reproducibility
         |
  sample N = max_files // 4  from each group
         |
  random.shuffle(combined)  -- mix sources evenly through evaluation loop
```

The seed `42` is hardcoded for the sampling step so that every run evaluates the **exact same balanced subset**. The `--seed` argument (default 0) remains available for controlling the noise-file selection within the evaluation loop and is independent of the sampling seed. If a source has fewer than `N` files, all available files are used without error.

With the default `--max_files 456`, each source contributes exactly **114 files**, giving a perfectly balanced 456-file evaluation set that is representative of all four dataset sources.

### 6.2 Thai-Aware ASR Evaluation

Standard ASR evaluation on Thai text requires special handling at two stages: transcription and scoring.

**Transcription:** The `WERScorer` class loads `openai/whisper-small` (244M parameters, fully multilingual) via the HuggingFace `transformers` pipeline. The decoder is forced to Thai with:

```python
generate_kwargs = {"language": "thai", "task": "transcribe"}
```

Without language forcing, Whisper auto-detects the language at the start of each chunk and can default to English for Thai audio with low-energy consonant frames — producing romanised transliteration rather than Thai script and making WER artificially high regardless of model quality.

**Thai word tokenisation:** Thai text has no spaces between words. Splitting on whitespace produces one token per sentence, making Word Error Rate meaningless. The pipeline uses `pythainlp.word_tokenize(engine="newmm")` to segment text into morphological units before WER calculation:

```python
def _tok_thai(text):
    tokens = word_tokenize(text.strip(), engine="newmm", keep_whitespace=False)
    return [t for t in tokens if t.strip()]

def _score_wer_cer(ref, hyp):
    ref_str = " ".join(_tok_thai(ref))   # space-join for jiwer
    hyp_str = " ".join(_tok_thai(hyp))
    wer = jiwer.wer(ref_str, hyp_str)   # word-level edit distance
    cer = jiwer.cer(ref.strip(), hyp.strip())  # character-level (raw strings)
    return wer, cer
```

**Character Error Rate (CER)** via `jiwer.cer` is computed on the raw un-tokenised strings. CER is often a more reliable metric for Thai because a morpheme segmentation error by the tokeniser inflates WER, while the underlying character sequence may be nearly correct.

**Graceful fallback chain:** If `transformers` is unavailable, the scorer falls back to `openai-whisper` pip package. If neither is available, WER/CER columns are omitted from the output rather than crashing.

### 6.3 Metrics Summary

The evaluation pipeline reports four metrics per system across all 456 files:

| Metric | Description | Notes |
|---|---|---|
| **MOS** | Perceptual quality score | DNSMOS (no reference) if `speechmos` installed; PESQ-WB (reference-based) if `pesq` installed; SNR-proxy fallback |
| **SI-SDR (dB)** | Scale-invariant signal-to-distortion ratio | Computed in numpy; positive is better |
| **WER** | Word Error Rate (Thai-tokenised) | Lower is better; requires reference labels |
| **CER** | Character Error Rate | Lower is better; more robust than WER for Thai |

Outputs are written to:
- `outputs/eval_per_utt.csv` — one row per file with all four metrics for all three systems
- `outputs/eval_summary.csv` — dataset-level averages with metadata (ASR model, language, pythainlp availability)

### 6.4 Qualitative Output Generation

After the main evaluation loop completes, the script automatically produces human-readable outputs for reporting and qualitative analysis.

**Text example block (console):**

One file is captured per source dataset during the main loop (the first file from each source that appears in the shuffled evaluation order). After all metrics are computed, the script prints:

```
========================================================================
  QUALITATIVE TEXT EXAMPLES  (one per dataset source)
========================================================================

  Dataset Source : commonvoice
  File           : commonvoice_cv-corpus-13_train_00001.npy
  Ground Truth   : [Thai ground truth text]
  CRN Enhanced   : [Whisper transcription of CRN output]
  Noisy (raw)    : [Whisper transcription of noisy input]

  Dataset Source : fleurs
  ...
```

This directly shows, for each source dataset, whether the CRN's output is more intelligible to Whisper than the raw noisy input — and whether the transcription error pattern differs between sources.

**Comparative spectrograms (`outputs/spectrogram_examples/`):**

For each of the four captured example files, a 3-panel PNG is saved showing the log-magnitude spectrogram of the noisy input, CRN enhanced output, and ground truth clean signal:

```
[Top]    Noisy Input          -- shows the noise floor and masking artefacts
[Middle] CRN Enhanced         -- shows what the model actually suppressed
[Bottom] Ground Truth (Clean) -- reference for ideal output
```

Vertical alignment between panels allows direct visual inspection of which frequency bands the model suppressed, whether low-frequency hum was removed, and whether any speech harmonics were over-suppressed.

---

## 7. Inference & Post-Processing (`inference.py`)

### Chunk-Based Inference

The trained model processes audio in **8-second chunks with 50 ms cross-fades** between adjacent chunks. Raw audio of any length is handled transparently by `enhance_waveform`, which:

1. Pads each chunk to exactly 8 seconds before passing to the model
2. Applies a Hann-like fade-in/fade-out ramp (50 ms) at the chunk boundaries
3. Accumulates the windowed outputs in an overlap-add buffer and divides by the sum-of-windows normalisation

### Soft Noise Gate (Post-Processing)

A **soft noise gate** is applied as a final signal-processing step after the model output. It targets the low-level continuous residual that the CRN cannot fully suppress in regions spectrally similar to the speaker's voice.

**Algorithm (4 stages):**

1. **Short-time RMS analysis**: 20 ms windows, 10 ms hop (100 frames/second)
2. **Soft-knee gain**: cosine curve across 6 dB knee centred on `--noise_gate_db`; below threshold-3 dB = 0.0, above threshold+3 dB = 1.0
3. **One-pole IIR smoothing**: attack 5 ms (opens quickly at word onset), release 80 ms (closes slowly, prevents pumping)
4. **Upsample**: `np.interp` to full sample rate, band-limited and click-free

```bash
# Default threshold -40 dBFS
python inference.py --ckpt checkpoints/best.pt --input noisy.wav --output clean.wav

# More aggressive gate
python inference.py --ckpt checkpoints/best.pt --input noisy.wav --output clean.wav \
                    --noise_gate_db -35

# Disable gate entirely
python inference.py --ckpt checkpoints/best.pt --input noisy.wav --output clean.wav \
                    --noise_gate_db -120
```

---

## Appendix A: CRN Hyperparameters

| Parameter | Value |
|---|---|
| n_fft | 512 |
| hop_length | 128 |
| Sample rate | 16,000 Hz |
| Segment length | 3.0 seconds |
| Encoder channels | 2->16->32->64->128->256 |
| SE reduction | 8 |
| Bottleneck proj | 2304->384->512->2304 |
| BiLSTM hidden | 256 per direction |
| BiLSTM layers | 2 |
| mask_bound | 1.2 |
| SNR range | -5 to +15 dB |
| Batch size | 24 |
| Steps/epoch | 180 (= 4,320 clips/epoch) |
| val_max_batches | 19 (= 456 clips) |
| Optimizer | AdamW (beta=0.9, 0.999) |
| Learning rate | 2e-4 |
| Weight decay | 1e-4 |
| Grad clip | 1.0 |
| Dropout (bottleneck) | 0.1 |
| Curriculum epochs | 12 |
| LR patience | 3 epochs |
| LR factor | 0.5 |
| low_freq_boost | 5.0x |
| high_freq_boost | 2.0x |
| Noise gate threshold | -40 dBFS |
| Noise gate knee | 6 dB |
| Noise gate attack | 5 ms |
| Noise gate release | 80 ms |

---

## Appendix B: U-Net Baseline Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| n_fft | 512 | identical to CRN |
| hop_length | 128 | identical to CRN |
| win_length | 512 | |
| Sample rate | 16,000 Hz | identical to CRN |
| Segment length | 3.0 seconds | identical to CRN |
| base_channels | 32 | channel schedule: 32->64->128->256->256 |
| depth | 5 | encoder/decoder stages |
| Bottleneck dilations | (1,2,4,8) + (1,2,4,8,16) | two sweeps + SE each |
| mask_bound | 1.2 | identical to CRN |
| Total parameters | ~2.14 M | |
| SNR range | -5 to +15 dB | identical to CRN |
| Batch size | 16 | reduced for skip-connection VRAM |
| Steps/epoch | 270 (= 4,320 clips/epoch) | equated to CRN budget |
| val_max_batches | 29 (~464 clips) | equated to CRN budget |
| Optimizer | AdamW (beta=0.9, 0.999) | identical to CRN |
| Learning rate | 2e-4 | identical to CRN |
| Weight decay | 1e-4 | identical to CRN |
| Grad clip | 1.0 | identical to CRN |
| Curriculum epochs | 12 | identical to CRN |
| LR patience | 3 epochs | identical to CRN |
| LR factor | 0.5 | identical to CRN |
| low_freq_boost | 5.0x | identical to CRN |

---

## Appendix C: Evaluation Pipeline Parameters

| Parameter | Default | Description |
|---|---|---|
| --ckpt | ./checkpoints/best.pt | CRN checkpoint to evaluate |
| --data_root | ./data_npy | Root of the .npy dataset |
| --split | test | Dataset split directory |
| --snr | 5.0 dB | Fixed SNR for evaluation mixtures |
| --max_files | 456 | Total files evaluated (114 per source) |
| --whisper_model | openai/whisper-small | ASR model for WER/CER |
| --language | thai | Forced Whisper decoder language |
| --seed | 0 | Controls noise-file selection within evaluation loop |
| --out_csv | ./outputs/eval_per_utt.csv | Per-utterance metrics |
| --out_summary | ./outputs/eval_summary.csv | Dataset-level averages |
| --spec_dir | ./outputs/spectrogram_examples/ | 3-panel spectrogram PNGs |
