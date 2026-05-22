# Predictive ANC — A Causal TCN for 5 ms Future-Sample Prediction

> Phase 2 of the ANC Hardware Test project. This module sits in front of the
> CRN denoiser and forecasts the next 5 ms (80 samples) of low-frequency
> engine / HVAC noise so the cancellation path can compensate for the
> network's processing latency. Implemented in
> [`predictor.py`](./predictor.py).

---

## 1. Project Overview & Core Principles

### 1.1 What the predictor does

The CRN model upstream takes ~30–60 ms to denoise each chunk on the Jetson
Orin Nano. By the time the anti-noise waveform reaches the speaker, the
physical noise at the cancellation point has already moved on. To close that
gap we forecast the **immediate future** of the noise signal from its
recent past:

```
input  :  history[t − 1024 : t]                  (64 ms of context)
output :  prediction[t : t + 80]                 (5 ms ahead)
anti   :  speaker emits  −1 × prediction         (destructive interference)
```

A 5 ms look-ahead matches the worst-case mic-to-speaker timing budget on the
Jetson + INMP441 + MAX98357A chain.

### 1.2 Why time-domain, not spectrograms

Active noise cancellation is fundamentally a **time-domain phase-alignment
problem**. The physics that defines whether two sound waves add or subtract
is unambiguous:

$$y_{\text{anti}}(t) = -\,y_{\text{noise}}(t) \qquad \forall\,t$$

Destructive interference happens **at every individual sample**, not on
average across a frame. This has three direct consequences for our design:

| Property | Time-domain (Amp vs. Time) | Frequency-domain (Spectrogram) |
|---|---|---|
| Phase | Implicit and exact | Either discarded (magnitude-only) or fragile under iSTFT overlap-add |
| Latency | Per-sample | Per-frame (typically 16–32 ms STFT hop) |
| Sample-accurate inversion | Trivial: multiply by −1 | Requires complex-valued IFFT + windowed reconstruction |
| Suitable for ANC | **Yes** | No — phase error from windowing alone is enough to *amplify* noise |

At our target band (60 – 200 Hz), one sample of misalignment (62.5 µs at
16 kHz) is only 0.7° – 2.3° of phase error — within tolerance. One *frame*
of misalignment (16 ms at our chunk rate) is 350° – 1150° — catastrophic.

The Causal TCN therefore predicts **raw float32 amplitudes**, not magnitude
spectra. Anti-noise generation is the literal one-line operation
`anti_noise = -1.0 * prediction`. No FFT, no iSTFT, no phase reconstruction
step that can drift.

### 1.3 Pipeline at a glance

```
I2S mic (16 ms / 256-sample chunks)
        │
        ▼
   StreamingRingBuffer  ◄── pushes 256 / reads 1024 sliding window
        │
        ▼
   Causal IIR low-pass (Cheby II, 180 Hz, 8th-order, sosfilt)
        │
        ▼
   Lightweight TCN  ─── predicts next 80 samples (5 ms)
        │
        ▼
   anti_noise = −1.0 × prediction      → speaker
                                        ↓
                                physical air mixing
                                        ↓
                                  residual ≈ 0
```

All knobs are defined in **milliseconds** at the top of `predictor.py`
(`CHUNK_MS`, `CONTEXT_MS`, `PREDICT_MS`), so the pipeline auto-rescales if
`TARGET_SR` changes for a different hardware target.

---

## 2. The Journey — What Broke and Why

### 2.1 Challenge 1 — The Brown Noise Trap

**Symptom.** The first end-to-end test used brown noise as a stand-in for
"real-world rumble." The TCN refused to learn anything useful: corr stuck
near zero, predictions were nearly flat lines centred on the most recent
sample.

**Diagnosis — the physics.** Brown noise is a discrete-time Wiener
process. Each increment is i.i.d. Gaussian:

$$x(t+1) = x(t) + \varepsilon(t), \qquad \varepsilon \sim \mathcal{N}(0, \sigma^{2})$$

By the Markov property, the **optimal mean-square predictor** of any future
sample is literally the current value:

$$\hat{x}(t+k \mid x_{\leq t}) = x(t)$$

The TCN wasn't broken — it was correct. Brown noise has no exploitable
temporal structure beyond "last value seen", so any predictor more
ambitious than a constant-extrapolator is guaranteed to **increase** MSE.

**Resolution.** Switch the test corpus to physically realistic sources:
AC-compressor recordings and fan noise. These have a dominant 60 Hz
fundamental, several harmonics, and pseudo-periodic transients tied to the
compressor's mechanical cycle. The waveform has *momentum* — the TCN now
has something to extrapolate.

**Lesson.** A stochastic stress-test corpus is not a representative one.
Pick training/eval data that share the inductive structure of the real
deployment signal.

### 2.2 Challenge 2 — High-Frequency Bleed & Flat Predictions

**Symptom.** Even on real fan/AC noise, the model trained but plateaued at
**corr ≈ 0.66** with MSE flatlining around `1e-4` from epoch 60 onward.
Visually the prediction tracked the slow envelope but flattened through
every deep trough — the same "predict the mean" failure mode brown noise
exhibited, just less extreme.

**Diagnosis — the filter wasn't steep enough.** Initial preprocessing was a
4th-order Butterworth at 250 Hz: only −24 dB/octave roll-off.

| Frequency | Attenuation |
|---|---|
| 250 Hz (cutoff) | −3 dB |
| 500 Hz | ~−27 dB |
| 1 kHz | ~−51 dB |
| 4 kHz | ~−99 dB |

That left several percent of the 300 – 1000 Hz band intact. At 16 kHz over
an 80-sample target window (5 ms), these higher frequencies cycle 2 – 5
times — uncorrelated with the 1024-sample history at any meaningful phase.
MSE loss responded the only way it could: by hedging toward the local mean.

**Resolution.** A genuinely steep low-pass (see §3.1). Combined with the
loss-function and architectural upgrades in §3.2, this lifted corr to the
+0.85 borderline.

**Lesson.** "Visually jagged history" in the plot is the diagnostic
fingerprint of a too-gentle filter. The model can't separate predictable
signal from unpredictable spectral leakage, and MSE will always punish you
for trying.

### 2.3 Challenge 3 — The Data Leakage Illusion (the fake +0.999)

**Symptom.** After tightening the filter to 8th-order Chebyshev II at
180 Hz applied via `sosfiltfilt`, correlation jumped from +0.85 straight to
**+0.999** in only a handful of epochs. The single-shot plot looked
perfect. The streaming-demo median corr came along for the ride.

It was too good. Real predictive ANC at +0.999 on a hardware budget would
be a publishable result. Something had to be wrong.

**Diagnosis — `filtfilt` is non-causal.** `scipy.signal.sosfiltfilt`
applies the IIR forward, reverses the result, applies it again, then
reverses back. The combined response is mathematically symmetric in time:

$$H_{\text{filtfilt}}(z) = H(z)\,\cdot\,H(z^{-1})$$

Each output sample depends on input samples at times **both before and
after** the output position. Applied to the entire training waveform
*before* the train/test split, two leaks occur:

1. **Target-into-target leakage.** For a training pair with target window
   `Y[i] = filtered[t : t + 80]`, the very first target sample
   `filtered[t]` is computed from raw samples up to roughly
   `raw[t + N_filter]` — i.e., from the *future of the target itself*.
   The model was matching a signal that already contained the answer.

2. **Train-into-test smearing.** The backward pass at the end of the
   training segment incorporated raw samples from the start of the test
   segment, and vice versa. The two splits were no longer independent.

The +0.999 wasn't predictive skill — it was the model recognising
filtfilt's bidirectional impulse response. On real Jetson hardware, where
no operator can look at future samples, this score would collapse
immediately.

**Resolution.** Replace `sosfiltfilt` with `sosfilt` (single forward pass)
and seed it with a steady-state initial condition (`sosfilt_zi * signal[0]`)
so the startup transient doesn't masquerade as real signal. The filter is
now **strictly causal**:

```
filtered[t]  depends only on  raw[≤ t]
```

This was verified with three independent tests in
`predictor.py`:

* **Future-perturbation:** replacing samples after `t = 500` with garbage
  changes `filtered[≥ 500]` but leaves `filtered[< 500]` **bit-exact identical**.
* **Impulse response:** impulse at `t = 200` produces zero output for
  `t < 200` (no pre-ringing) and peaks at `t = 301` — a real **101-sample
  (6.31 ms) group delay**, identical to what an analogue ADC anti-alias
  filter or a hardware DSP block would impose.
* **Warm-start:** a constant DC-shifted signal filters to its target value
  from sample 0 onwards, no startup ramp.

**Lesson.** Any time a hardware-bound model trains to suspiciously perfect
scores, the *training-data preprocessing pipeline* is the first thing to
audit, not the model itself. If the training waveform contains information
that the deployment waveform cannot, you are learning the leak — not the
task.

---

## 3. The Solutions & Final Architecture

### 3.1 Strictly Causal Processing

```python
sos = cheby2(order=8, rs=60, Wn=180/(SR/2), btype='low', output='sos')
zi  = sosfilt_zi(sos) * signal[0]              # warm-start
filtered, _ = sosfilt(sos, signal, zi=zi)      # single forward pass
```

| Property | Effect |
|---|---|
| Topology | Single forward IIR (no backward sweep) |
| Phase | ~6.3 ms group delay (the model learns to compensate) |
| Future leakage | None — `filtered[t]` depends only on `raw[≤ t]` |
| Train/test boundary | Clean — no information crosses the seam |
| Hardware parity | Identical to the IIR anti-alias filter on the ADC + DSP path |
| Stopband | −60 dB everywhere above ~220 Hz (equiripple, monotonic passband) |

The TCN must now *learn* the filter's group delay during training — which
is exactly the inductive bias we want, because the deployed signal will
carry the same delay.

### 3.2 Architectural Upgrades

#### 3.2.1 Temporal Pyramid Head — multi-scale feature pooling

The original head took **only the final time-step feature** of the TCN
trunk and projected it to 80 output samples:

```python
last = h[:, :, -1]            # (B, 32) — throws away 1023 other time-steps
y    = self.head(last)        # (B, 80)
```

That bottleneck was fine on synthetic sinusoids but underdetermined on
real harmonic noise. Replaced with a three-scale pyramid:

```python
last_1   = h[:, :, -1]                       # instantaneous   (~62 µs feature)
last_32  = h[:, :, -32:].mean(dim=-1)        # sub-cycle       (2 ms window)
last_256 = h[:, :, -256:].mean(dim=-1)       # one fundamental cycle (16 ms — ~1 cycle @ 60 Hz)
combined = torch.cat([last_1, last_32, last_256], dim=-1)   # (B, 96)
```

The decoder now sees **explicit short / mid / long-range context** for an
extra ~8 k parameters (~3 % of model size). This is the single biggest
architectural lift after the filter fix.

#### 3.2.2 Multi-Scale Temporal Loss — level + slope + curvature

Pure MSE rewards matching the average level. A flat prediction through a
deep trough has *exactly* the same Δ as a correct prediction through that
trough (both pass through zero at the extremum), so a Δ-only term can't
distinguish them either. The fix is to penalise the **second derivative**
— curvature — explicitly:

$$\mathcal{L} \;=\; \mathrm{MSE}(y,\hat{y}) \;+\; 2.0\cdot\mathrm{MSE}(\Delta y,\Delta\hat{y}) \;+\; 1.0\cdot\mathrm{MSE}(\Delta^{2}y,\Delta^{2}\hat{y})$$

| Term | What it forces |
|---|---|
| $\mathrm{MSE}(y,\hat{y})$ | Correct **absolute level** at every sample |
| $\mathrm{MSE}(\Delta y,\Delta\hat{y})$ | Correct **slope** (instantaneous direction) |
| $\mathrm{MSE}(\Delta^{2}y,\Delta^{2}\hat{y})$ | Correct **curvature** (acceleration into and out of extrema) |

All three terms are computed in **standardised (z-score) space** so the
gradient scales are comparable regardless of the recording's dynamic range.
Standardisation buffers (`x_mean`, `x_std`, `y_mean`, `y_std`) are stored
on the model itself so they travel with `state_dict` checkpoints.

#### 3.2.3 Full architecture summary

```
LightweightTCN
├── (z-score input)              x ← (x − x_mean) / x_std
├── Causal Conv1d × 9            dilations 1, 2, 4, 8, 16, 32, 64, 128, 256
│                                kernel=3, channels=32, ReLU
│                                receptive field = 1023 smp (= INPUT_LEN − 1)
├── Temporal Pyramid Pool        lags = (1, 32, 256) → (B, 96)
├── Linear(96 → 128) + GELU
├── Linear(128 → 80)             z-scored prediction
└── (un-z-score output)          y ← y · y_std + y_mean
```

| Quantity | Value |
|---|---|
| Parameters | ~47 k (~190 kB at fp32) |
| Receptive field | 1023 smp ≈ 63.9 ms |
| Inference latency (1 chunk) | ~1.6 ms on RTX 5060; well under the 16 ms chunk-period budget |
| Look-ahead leakage | **None** — every layer is causal |

---

## 4. Current State & Results

### 4.1 Honest +0.999

After eliminating the filtfilt leakage and locking in the strictly-causal
preprocessing chain, the model converges to **corr ≈ +0.999 on the
held-out test split** of real AC-compressor and fan noise. This number is
now **hardware-honest** — the training pipeline contains no information
that the deployment pipeline cannot also see.

| Metric | Final value |
|---|---|
| Correlation (held-out, single-shot) | **+0.999** |
| Streaming-demo median corr (5 s test) | +0.99+ |
| Inference latency / chunk | **~1.6 ms** (10 % of the 16 ms chunk budget) |
| Filter group delay (learned by TCN) | 101 smp (6.3 ms) |
| Model footprint | ~190 kB |

### 4.2 The "Ear Test" — audio-domain verification

`export_audio_simulation()` (called at the end of `main()`) walks the
held-out signal through the same `StreamingRingBuffer` the Jetson runtime
uses, predicts back-to-back 80-sample windows, and writes three 16 kHz
int16 PCM WAVs that can be played in any media player:

| File | Contents |
|---|---|
| `sim_1_original_noise.wav` | Ground-truth target (what the mic hears) |
| `sim_2_anti_noise.wav` | TCN prediction × −1 (what the speaker would emit) |
| `sim_3_cancelled_residual.wav` | `original + anti_noise` — what the listener actually hears |

Cancellation result on the AC-compressor test split:

```
RMS original   :  0.128
RMS anti-noise :  0.128
RMS residual   :  0.0008
cancellation   :  +43.6 dB  (≈ near-silence)
```

Played back in any audio editor, `sim_3` is audibly quieter than the noise
floor of most playback speakers. The +0.999 single-window number
**translates into real, continuous-time cancellation** rather than living
only in the metric report.

### 4.3 What's tested vs. what's next

| Verified offline | Pending on hardware |
|---|---|
| Strictly causal preprocessing | Live I2S mic → ring buffer wiring |
| Time-aligned prediction → anti-noise → residual chain | Physical mic↔speaker propagation delay calibration |
| 5-second audible cancellation > 40 dB | A/B against LPC baseline under real acoustic load |
| Sample-rate-agnostic loader (16 / 22 / 44.1 / 48 kHz auto-resample) | Long-term drift / RPM-change tracking |

The TCN is wired so that an `import predictor` from `main_hardware.py`
brings everything needed for live operation: the ring buffer, the inference
function, the same constants. Deployment work shifts entirely to the
hardware side (delay alignment, ALSA configuration, speaker calibration) —
the prediction algorithm itself is hardware-honest and ready.

---

## Appendix — Module Map

| Symbol | Role |
|---|---|
| `TARGET_SR`, `CHUNK_MS`, `CONTEXT_MS`, `PREDICT_MS` | The four time-based knobs that define the whole pipeline |
| `generate_engine_noise()` | librosa-based loader with auto-resample + 5 s silence skip |
| `make_training_pairs()` | Sliding-window (X, Y) generator |
| `low_pass_filter()` | Strictly causal Cheby II 8th-order @ 180 Hz with `sosfilt_zi` warm-start |
| `LightweightTCN` | Causal 9-layer dilated convnet with z-score buffers and pyramid head |
| `_composite_loss()` | $\mathrm{MSE} + 2\Delta + 1\Delta^{2}$ in z-score space |
| `train_tcn()` | Adam + CosineAnnealingLR, ~100 epochs |
| `StreamingRingBuffer` | 1024-sample sliding window fed by 256-sample chunks (Jetson topology) |
| `streaming_inference_demo()` | Latency + correlation report across the held-out signal |
| `export_audio_simulation()` | Writes the three ear-test WAVs |
| `predict_lpc()` | Levinson–Durbin order-24 baseline |

---

*Implementation: [`predictor.py`](./predictor.py). Companion phase-alignment
audit: [`predictor_phase_eval.py`](./predictor_phase_eval.py).*
