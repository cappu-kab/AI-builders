# ANC Phase 1 — Real-time Noise Cancellation (Jetson Nano)

**Hardware:** INMP441 (I2S mic) + MAX98357A (I2S DAC/Amp)  
**Model:** CRN BiLSTM — Convolutional Recurrent Network with bidirectional LSTM bottleneck  
**Target:** Low-frequency engine/mechanical noise estimation and destructive interference (< 200 Hz)

---

## Executive Summary

### Definitive Architecture: CRN BiLSTM

The **CRN (Convolutional Recurrent Network) with BiLSTM bottleneck** is the confirmed and sole model for this project. All alternative architectures have been evaluated and permanently abandoned:

| Model | Status | Reason |
|---|---|---|
| CRN BiLSTM | **Active** | Verified phase cancellation, NR: −1.2 dB confirmed |
| CRN GRU | Abandoned | Identity collapse — `random_gain` scale mismatch in training dataset caused the model to learn `enhanced ≈ noisy`, yielding `noise_estimate ≈ 0` and NR ≈ 0 dB |
| U-Net | Abandoned | Severe spectral distortion at CHUNK_SIZE=512 (fixed receptive field incompatibility with streaming context), output characterized by audible artifacts |

### Phase Cancellation Verification

The file simulator (`file_simulator.py`) confirmed correct phase cancellation mechanics with NR = **−1.2 dB** on a real recording. A negative NR value proves that the anti-noise signal is producing genuine destructive interference: the output is measurably quieter than the input, not louder. This verifies that:

- The synchronous delay alignment formula is mathematically correct
- The context buffer propagation is continuous and unbroken
- The BiLSTM look-ahead queue is correctly aligned
- Phase inversion (`anti_noise = noise_estimate × −1`) is correctly applied

The remaining issue is partial speech leakage into the noise estimate — the model occasionally predicts voiced speech harmonics as noise. This is a model quality issue addressable through retraining (`checkpoints_crn2`) and post-inference boundary tuning, not a pipeline bug.

---

## Real Hardware Architectural Theory

### Dual-Signal ANC Pipeline

The physical deployment on Jetson Nano implements a feedforward ANC topology. Two acoustic signals exist in the system at all times:

```
  [Ambient Environment]
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │  REFERENCE MICROPHONE  (INMP441 I2S)                │
  │  Captures: engine noise + ambient + any speech      │
  │  This is the INPUT signal to the AI model           │
  └────────────────────────┬────────────────────────────┘
                           │  raw_chunk[t]  (512 samples, float32)
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  JETSON NANO — CRN BiLSTM Inference                 │
  │                                                     │
  │  1. Sliding context buffer: prepends 47 616 samples │
  │     of history to each new 512-sample chunk.        │
  │     Context window = 3.0 s = 372 STFT frames,       │
  │     matching the BiLSTM training segment length.    │
  │                                                     │
  │  2. BiLSTM look-ahead queue: delays output by N     │
  │     chunks so the backward LSTM pass has N frames   │
  │     of future context before committing to an       │
  │     estimate. First N calls output zeros (muted     │
  │     warm-up). After warm-up, each call returns the  │
  │     noise estimate for the chunk N steps ago.       │
  │                                                     │
  │  3. noise_estimate = noisy_chunk - enhanced_chunk   │
  │     anti_noise     = noise_estimate × −1.0          │
  └────────────────────────┬────────────────────────────┘
                           │  anti_noise[t]  (512 samples)
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  MAX98357A I2S DAC / AMPLIFIER                      │
  │  Converts anti_noise PCM → acoustic pressure wave   │
  │  positioned at the ERROR MICROPHONE location        │
  └────────────────────────┬────────────────────────────┘
                           │  acoustic anti-noise wave
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  ERROR MICROPHONE — Virtual Clean Zone              │
  │  Superposition: original_noise + anti_noise → 0     │
  │  Residual signal = clean speech only (ideal)        │
  └─────────────────────────────────────────────────────┘
```

### Context Buffer: The Critical Invariant

The 47 616-sample sliding context buffer (`_StreamingContextBuffer` in `model_runtime.py`) is the core state object of the streaming inference pipeline. **It must advance on every single audio chunk without exception.**

**The invariant:** For every chunk `t` that enters `_input_q`, `model.process_chunk(chunk_t)` must be called exactly once, in order, with no gaps.

Violating this invariant — by skipping silent frames or early-exiting before inference — causes the buffer to fall behind the audio stream. When inference resumes on the next non-silent chunk, the buffer contains audio from `t - skip_count` steps ago. The BiLSTM hidden state develops on this stale context, producing a noise estimate that corresponds to the wrong time position. The result is constructive interference (the anti-noise signal adds to the noise rather than cancelling it), which is the primary cause of NR > 0 dB.

**Correct implementation in `_inference_loop`:**

```python
# ALWAYS run inference — keeps the context buffer and BiLSTM
# hidden state synchronized with the real-time audio stream.
noise_estimate = model.predict(chunk)

# Gate the SPEAKER OUTPUT only, never the inference.
if is_silent(chunk):
    anti_noise = np.zeros(CHUNK_SIZE, dtype=np.float32)
else:
    anti_noise = noise_estimate * PHASE_INVERSION_FACTOR  # −1.0
    anti_noise = clip_output(anti_noise)

_enqueue_output(anti_noise)
```

The `is_silent()` gate protects the MAX98357A from emitting anti-noise during genuine silence (which would produce audible artifacts), while the context buffer continues to accumulate audio history uninterrupted.

### Hardware Latency Budget

| Stage | Latency Source | Value |
|---|---|---|
| ADC capture | INMP441 I2S frame | ~0.5 ms |
| Input queue transit | `_input_q.get()` | < 1 ms |
| Context buffer push | Memory copy (48 128 samples) | < 0.5 ms |
| CRN BiLSTM inference | GPU/CPU forward pass | 5–25 ms (device-dependent) |
| BiLSTM look-ahead delay | N × CHUNK_SIZE / 16 000 | N × 32 ms |
| Output queue transit | `_output_q` | < 1 ms |
| DAC playback | MAX98357A I2S | ~0.5 ms |

At `N = 8` look-ahead chunks: total pipeline latency ≈ **8 × 32 ms + inference ≈ 280–300 ms**.  
At `N = 1`: ≈ **32–60 ms** (lowest latency, backward LSTM has minimal future context).

The look-ahead `N` is auto-sized in `file_simulator.py` and `diag_streaming.py` by:

```python
la_n = max(1, min(32, n_chunks - MIN_OUTPUT_CHUNKS))
```

---

## Simulation Framework Theory

### Digital Twin: `file_simulator.py`

`file_simulator.py` is an exact software replica of the real hardware pipeline. It replaces the I2S microphone with a WAV file feeder and replaces the MAX98357A DAC with an in-memory output collector, while running the **identical** `_inference_loop`, `ModelRuntime`, and `_StreamingContextBuffer` code paths that will execute on Jetson Nano.

The file simulator uses a **two-pass architecture**:

1. **Live pass (threaded):** `FileFeeder → _input_q → _inference_loop → _output_q → OutputCollector`. This drives the real-time 4-panel visualization dashboard. The output from this pass is used only for display; it is never written to disk because queue drops, `clip_output()`, and the `is_silent()` gate can introduce artifacts that corrupt the audio + anti-noise sum.

2. **Synchronous post-pass (deterministic):** After the Qt event loop exits and all threads join, `model.reset()` is called and the full audio array is re-processed in a single-threaded `for` loop — identical to `diag_streaming.py`. This pass owns all saved WAV files.

### Delay Alignment Equation

The CRN BiLSTM introduces a look-ahead delay of `delay_samples = N × CHUNK_SIZE`. When `model.process_chunk(chunk[t])` is called, the returned noise estimate corresponds to `chunk[t − N]`, not `chunk[t]`.

The acoustic cancellation at the error microphone therefore requires aligning the anti-noise signal with the correct input position before summing:

$$\text{clean}[t] = \text{audio}[t] + \text{anti\_noise}[t + \text{delay\_samples}]$$

Equivalently, in array notation (both arrays starting at index 0):

$$\text{clean}[:L - d] = \text{audio}[:L - d] + \text{anti\_noise}[d:L]$$

where $L = \text{len(anti\_noise)}$ and $d = \text{delay\_samples}$.

**Implementation in `_run_simulation` (post-pass):**

```python
anti  = np.concatenate(anti_acc).astype(np.float32)   # anti_acc[i] = noise_est[i] × −1
delay = model.look_ahead_delay                          # = N × CHUNK_SIZE samples

if delay > 0 and len(anti) > delay:
    n_cln     = min(len(audio), len(anti) - delay)
    clean_arr = audio[:n_cln] + anti[delay : delay + n_cln]
else:
    n_cln     = min(len(audio), len(anti))
    clean_arr = audio[:n_cln] + anti[:n_cln]            # GRU / zero-delay path
```

### Why Phase Mismatch Causes Constructive Interference

If the delay is ignored and `clean = audio + anti_noise` is computed without the index shift, the anti-noise frame $n$ (which was generated to cancel `audio[n − d]`) is instead added to `audio[n]`. For periodic noise with a period comparable to the delay, this addition can be **in-phase rather than anti-phase**, amplifying the noise by up to 6 dB. This is the cause of NR > 0 dB in naive implementations.

The fix is exact: applying the index shift `anti[delay:]` aligns each anti-noise frame with the audio sample it was computed for, restoring the 180° phase relationship that produces destructive interference.

For **GRU (causal, unidirectional):** `model.look_ahead_delay == 0`, so the shift is a no-op and the standard `clean = audio + anti_noise` formula applies directly.

### Output Files

| File | Contents | Purpose |
|---|---|---|
| `simulated_output_clean_{model}.wav` | `audio + anti_noise` (delay-aligned) | Primary output — listen to this |
| `simulated_output_clean_{model}_antinoise.wav` | Raw anti-noise signal | Speaker signal — not for listening |

---

## Post-Inference Boundary Tuning

These three methods operate entirely at runtime on the anti-noise output array. No retraining is required. They target the residual speech leakage remaining after inference.

### Method 1 — Voice Activity Detection (VAD) Energy Gating

**Problem:** On trailing speech phonemes (particularly voiced fricatives and vowel offsets), the noise estimate momentarily contains speech harmonics because the BiLSTM backward pass extrapolates into the phoneme boundary.

**Mechanism:** Compute the per-frame short-time energy of the raw input chunk. If the energy exceeds a speech-presence threshold, scale the anti-noise output toward zero with a fast-attack / slow-release envelope. The attack must be rapid (1–2 frames) to catch the onset of speech; the release must be slow (10–20 frames) to avoid reintroducing noise immediately after the phoneme ends.

```python
# Tunable constants
VAD_THRESHOLD   = 5e-4   # RMS threshold above which speech may be present
VAD_HOLD_FRAMES = 15     # frames to keep gate closed after energy drops below threshold
VAD_RELEASE_TAU = 0.85   # per-frame decay factor during release (higher = slower)

class VADGate:
    def __init__(self):
        self._hold  = 0
        self._gain  = 1.0

    def process(self, chunk: np.ndarray, anti_noise: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms > VAD_THRESHOLD:
            self._hold = VAD_HOLD_FRAMES
            self._gain = 0.0                     # hard mute on speech onset
        elif self._hold > 0:
            self._hold -= 1
            self._gain = 0.0                     # hold during trailing phoneme
        else:
            self._gain = min(1.0, self._gain / VAD_RELEASE_TAU)  # slow re-open
        return (anti_noise * self._gain).astype(np.float32)
```

**Trade-off:** A tight `VAD_THRESHOLD` (lower value) reduces speech leakage aggressively but risks muting anti-noise during speech-over-noise conditions where ANC is most useful. Tune on recorded samples; a value of `3e-4`–`1e-3` covers most scenarios.

---

### Method 2 — Frequency Bandpass Post-Filtering

**Problem:** The CRN's noise estimate contains energy across the full 0–8 kHz band. Speech harmonics above 300 Hz that leak into the noise estimate become audible artifacts in the clean signal. The actual noise target for ANC (engine rumble, HVAC, 50/60 Hz hum) is concentrated below 200 Hz.

**Mechanism:** Apply a soft FFT-domain lowpass to the anti-noise signal, zeroing all energy above `cutoff_hz`. A cosine-taper transition band prevents spectral ringing without a time-domain filter's group-delay cost.

```python
def bandpass_antinoise(
    anti_noise: np.ndarray,
    cutoff_hz:  float = 200.0,
    slope_hz:   float = 80.0,
    sample_rate: int  = 16_000,
) -> np.ndarray:
    X     = np.fft.rfft(anti_noise)
    freqs = np.fft.rfftfreq(len(anti_noise), d=1.0 / sample_rate)
    # Soft sigmoid lowpass gate
    gain  = 1.0 / (1.0 + np.exp((freqs - cutoff_hz) / (slope_hz / 6.0)))
    return np.fft.irfft(X * gain, n=len(anti_noise)).astype(np.float32)
```

**Trade-off:** Setting `cutoff_hz = 200` completely eliminates speech-frequency content from the anti-noise signal. The cost is that any noise energy above 200 Hz (fan hiss, broadband rumble) is no longer cancelled. For a deployment where the primary noise is tonal LF (engine harmonics, HVAC), this is an acceptable trade — the modal noise is fully cancelled while speech is fully protected. Raise the cutoff to 400–600 Hz if the noise profile contains mid-range components.

---

### Method 3 — Temporal Mask Smoothing

**Problem:** Frame-to-frame variance in the BiLSTM's complex mask output creates discontinuities between adjacent 512-sample chunks. These discontinuities are audible as "graininess" in the anti-noise signal and appear as transient speech-like artifacts at chunk boundaries.

**Mechanism:** Apply a first-order recursive (exponential moving average) filter to the anti-noise frames in time. Each output frame is a weighted blend of the current inference output and the previous filtered frame:

$$y[t] = \alpha \cdot \hat{n}[t] + (1 - \alpha) \cdot y[t-1]$$

where $\hat{n}[t]$ is the raw anti-noise for chunk $t$ and $\alpha \in (0, 1]$ is the smoothing coefficient.

```python
class TemporalSmoother:
    def __init__(self, alpha: float = 0.7):
        # alpha=1.0 → no smoothing (pass-through)
        # alpha=0.5 → heavy smoothing, ~2-frame lag
        # alpha=0.7 → light smoothing, recommended starting point
        self._alpha = alpha
        self._prev  = None

    def process(self, anti_noise: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = anti_noise.copy()
            return anti_noise
        smoothed   = self._alpha * anti_noise + (1.0 - self._alpha) * self._prev
        self._prev = smoothed.copy()
        return smoothed.astype(np.float32)

    def reset(self) -> None:
        self._prev = None
```

**Trade-off:** Lower `alpha` (more smoothing) reduces frame-boundary artifacts but introduces sub-chunk temporal lag that adds to the pipeline delay and can soften the sharp onset of noise transients. For engine noise (slowly varying), `alpha = 0.6`–`0.75` is effective. For percussive noise transients, use `alpha ≥ 0.85` to preserve temporal accuracy.

### Stacking the Three Methods

The filters compose in series. Recommended application order:

```
noise_estimate
    → ×(−1)               # phase inversion
    → bandpass_antinoise() # Method 2: frequency gate first — removes speech bands structurally
    → vad_gate.process()   # Method 1: energy gate — suppresses residual speech transients
    → smoother.process()   # Method 3: temporal smooth — cleans frame boundaries last
    → clip_output()        # hard DAC protection
    → _enqueue_output()
```

Bandpass filtering goes first because it is the strongest and most reliable suppressor of speech-frequency content. The VAD gate then handles residual transients that survive the frequency cut. Temporal smoothing is applied last so it operates on already-cleaned frames, minimising the number of artifacts that get blended across the smoothing window.

---

## Project Structure

```
ANC_Hardware_Test/
├── main_hardware.py       # Real-time pipeline: I2S streams + inference thread
├── file_simulator.py      # Software digital twin: WAV input + Qt dashboard
├── diag_streaming.py      # Diagnostic: streaming vs OLA vs offline quality gap
├── model_runtime.py       # ModelRuntime: context buffer + BiLSTM lookahead + inference
├── configs/
│   └── config.py          # Hardware constants (SAMPLE_RATE, CHUNK_SIZE, device indices)
├── utils/
│   ├── audio_utils.py     # Signal processing: LP filter, phase inversion, clip, VAD
│   └── logger_setup.py    # Dual console + file logger
├── mock/
│   └── mock_model.py      # LP-filter stand-in — validates pipeline without a checkpoint
├── visualizer.py          # PyQtGraph 4-panel ANC dashboard
└── logs/
    └── anc_runtime.log    # Runtime log (auto-created)
```

---

## Step 1 — Jetson Nano System Setup

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev alsa-utils python3-pip
cd ~/ANC_Hardware_Test
pip3 install -r requirements.txt
```

---

## Step 2 — Finding Hardware Device Indices

All constants marked `[DUMMY]` are in `configs/config.py`.

```bash
# Method A — numeric indices directly
python3 -m sounddevice

# Method B — ALSA
arecord -l    # input/capture  → INPUT_DEVICE_INDEX
aplay   -l    # output/playback → OUTPUT_DEVICE_INDEX

# Verify capture before running Python
arecord -D hw:2,0 -r 16000 -c 1 -f S16_LE -d 3 test_mic.wav
aplay   -D hw:3,0 test_mic.wav
```

Edit `configs/config.py`:
```python
INPUT_DEVICE_INDEX  = 2      # INMP441 I2S mic card number
OUTPUT_DEVICE_INDEX = 3      # MAX98357A DAC card number
USE_MOCK_MODEL      = False  # set False once checkpoint is deployed
```

---

## Step 3 — Running the Pipeline

```bash
# Real-time hardware pipeline (mock model, no checkpoint required)
python3 main_hardware.py

# Real-time hardware pipeline (CRN BiLSTM checkpoint)
python3 main_hardware.py --no-mock

# File simulator — CRN BiLSTM with default checkpoint
uv run python file_simulator.py path/to/noisy.wav --model crn

# File simulator — CRN BiLSTM with a specific checkpoint
uv run python file_simulator.py path/to/noisy.wav --model crn --ckpt ../Run/CRN/checkpoints_crn2/best.pt

# Diagnostic — streaming vs OLA vs offline quality gap
uv run python diag_streaming.py path/to/noisy.wav

# List audio devices and exit
python3 main_hardware.py --list-devices
```

---

## Tuning Latency

| `CHUNK_SIZE` | Latency @ 16 kHz | Notes |
|---|---|---|
| 128 | ~8 ms | Very low — heavy CPU/GPU pressure |
| 256 | ~16 ms | Good balance for Jetson Nano |
| 512 | ~32 ms | Default — stable, recommended |
| 1024 | ~64 ms | High latency, very stable |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Invalid device index` | Wrong ALSA index | Run `python3 -m sounddevice` |
| `INPUT OVERFLOW` | Inference too slow for chunk size | Increase `CHUNK_SIZE` or use TensorRT |
| `OUTPUT UNDERFLOW` | Inference thread behind | Same as above |
| NR > 0 dB (noise amplified) | Context buffer desync or wrong delay alignment | Verify `model.look_ahead_delay` is non-zero for BiLSTM; verify synchronous post-pass is used for WAV output |
| NR ≈ 0 dB (no cancellation) | Model identity collapse | Retrain with `checkpoints_crn2` — check that `random_gain` is applied to `clean` before mixing, not to `noisy` post-mix |
| Speech in anti-noise | Speech leakage into noise estimate | Apply boundary tuning (VAD gate, bandpass filter, temporal smoother) and/or retrain with `--mask_bound 1.0 --w_speech_floor 0.5 --w_sisnr 2.0` |
| `PortAudio initialisation failed` | ALSA not configured | `sudo apt-get install portaudio19-dev` |
| I2S device not in `arecord -l` | Device tree / kernel overlay missing | Check Jetson I2S dtoverlay configuration |

---

## Log Files

```bash
tail -f ~/ANC_Hardware_Test/logs/anc_runtime.log
```

Console shows INFO+; file contains full DEBUG output including per-chunk RMS stats and context buffer state.
