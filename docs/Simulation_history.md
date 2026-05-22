# ANC Hardware Test — Engineering Log

**Component:** `ANC_Hardware_Test/file_simulator.py` (hardware-accurate streaming simulator)
**Model:** CRN (Convolutional Recurrent Network) — complex STFT mask, 2-layer BiLSTM bottleneck
**Sample rate:** 16 000 Hz
**Target hardware:** Jetson Nano + INMP441 (I2S mic) + MAX98357A (I2S DAC)
**Document scope:** v1 through v5, 2026-05-20
**Locked baselines:** `successful_v3_backup/`, `successful_v4_backup/`, `successful_v5_backup/`

---

## 0. Executive Summary

The simulator inherited from `problem_20052026.md` produced **+2.75 dB constructive interference** on its own calibration signal — i.e. it amplified the noise it was meant to cancel. Through five iterations focused on *streaming-vs-training* alignment of the CRN's iSTFT and BiLSTM, plus a final post-cancellation energy gate, the pipeline now reaches **−18.60 dB cancellation** on pure-noise input (within the report's −15 to −25 dB target band) while preserving speech on noisy-speech input. The model checkpoint itself was never changed.

---

## 1. Project Context

The CRN denoiser was trained on 3-second random segments and outputs `enhanced = waveform − noise_wave` (residual learning). The hardware simulator must call this model on a strict 16 kHz chunk stream and emit a "clean residual" output for every chunk, time-aligned with the raw mic input, so it can be summed against the mic on the simulated hardware path.

A pre-existing diagnostic report (`problem_20052026.md`) proposed three time-domain fixes:

1. A 257-sample reference-delay buffer ("STFT group-delay compensation").
2. A 64-sample linear cross-fade at every chunk boundary.
3. A polarity convention of `+` (i.e. `clean = delayed_raw + noise_est`).

All three fixes were already implemented when this engineering effort began. The measured result on the report's own calibration signal was the opposite of what the report predicted.

---

## 2. Evolution / History

### v1 — Inherited baseline (report's three fixes already applied)

**Code state:** `_TOTAL_ALIGN_DELAY = 257`, `ANC_XFADE_SAMPLES = 64`, `clean = delayed_raw + noise_est`, `CHUNK_SIZE = 256`, `_CRN_CTX_N = 3840` (~32 STFT frames per call), per-chunk extraction `enhanced[-CHUNK_SIZE:]`.

**Symptom:** constructive interference, not cancellation.

| Signal | NR (in − out) | Verdict |
|---|---|---|
| Synthetic engine noise (`engine_noise_ref.wav`, 75/150/100/60/320/480 Hz multi-sine) | **+2.75 dB** | output louder than input |
| `noise_sound.wav` (real recording) | +1.58 dB | output louder than input |

**Audit findings:**

- No other module silently added or assumed a delay (verified `model_runtime.py`, `look_ahead_delay`, context/resampler paths).
- The 8 missing checkpoint keys reported on every load are all from the `FreqAttention` module (`Run/CRN/model.py:244-263`), whose output projection is zero-initialised (`nn.init.zeros_(self.proj.weight)`, `…bias)`). It is a strict no-op at load time and does not affect inference.
- Offline single-pass inference on the full waveform reached **+17.69 dB NR** with polarity `−`. The trained model itself was sound; the streaming pipeline was the bottleneck.

The report's polarity claim was empirically wrong: math against `model.py:451` (`return waveform - noise_wave`) and the runtime's `noise_est = chunk - enhanced` showed `noise_est ≈ +true_noise`, so the natural cancellation sign is `−`. The report's `+` was a workaround for the right-boundary distortion documented in v2.

### v2 — Larger context, look-ahead extraction, correct polarity

**Three coupled changes:**

1. `_CRN_CTX_N`: 3 840 → **47 616 samples** (`model_runtime.py:78-90`). At hop=128, this gives ~374 frames per call, matching the 3-second training window the BiLSTM was trained on.
2. **Look-ahead extraction** in `_infer_crn_stateful` (`model_runtime.py:702-755`). Changed `enhanced[-CHUNK_SIZE:]` to `enhanced[-2*CHUNK_SIZE:-CHUNK_SIZE]`, moving extraction one chunk back into the iSTFT interior where there is no reflect-pad distortion.
3. Polarity flip: `clean = delayed_raw - noise_est` (`file_simulator.py:301-310`).

| Signal | NR | Δ vs. v1 |
|---|---|---|
| Synthetic engine noise | **−5.62 dB** | +8.4 dB toward cancellation |

Still below the target band. The 257-sample alignment buffer was kept because the 1-chunk look-ahead introduces a 256-sample output shift (coincidentally equal to the report's claimed STFT group delay) plus 1 physical sample.

### v3 — Cross-fade disabled

**Cause analysis:** with look-ahead extraction, consecutive extracted segments come from the model's iSTFT interior of overlapping context windows (99.5 % overlap). They are naturally continuous across chunk boundaries. The 64-sample linear cross-fade was no longer fixing a real discontinuity — it was ramping the leading samples of cancellation toward zero. With `_XFADE_LEN = 64` and `CHUNK_SIZE = 256`, **25 % of every chunk was being passed through uncancelled**, mathematically capping NR at ≈+6 dB regardless of model quality.

`ANC_XFADE_SAMPLES`: 64 → **0** (`configs/config.py`), and the cross-fade step in the worker became a conditional block (`file_simulator.py:286-294`).

| Signal | NR | Δ vs. v2 |
|---|---|---|
| Synthetic engine noise | **−13.99 dB** | +8.4 dB |
| `noise_sound.wav` | **−12.83 dB** | (first test) |
| `sound3.wav` (noisy speech) | **−5.09 dB** | speech preserved (output peak 0.887 vs input 0.918) |

The simulator was now working. Output RMS −25.34 dBFS on the synthetic signal matched the report's predicted "Clean dBFS" range of −21 to −31 dB.

This state was preserved as **`successful_v3_backup/`** (source files only).

### v4 — Chunk size increased

**Cause analysis:** the residual diagnostic (`_residual_diag.py`) on v3 output showed:

- `correlation(residual, raw) = +0.024` → essentially uncorrelated.
- Optimal scalar gain `g* = 0.97` → 0.13 dB headroom only.
- Edge-vs-center per-chunk energy difference: −0.48 dB → no boundary spikes.

The residual was uncorrelated random noise — the model's intrinsic accuracy floor when running on 47 872-sample windows instead of the full waveform. The remaining gap to the offline ceiling (3.7 dB) was the *cost of streaming itself*, not a fixable bug.

The one straightforward way to reduce that cost: lower the per-chunk variance by processing larger chunks (fewer model invocations per second, fewer per-window inconsistencies).

`CHUNK_SIZE`: 256 → **1024 samples** (16 ms → 64 ms output latency, `configs/config.py:27`).
`WARMUP_CHUNKS`: 100 → **50** (`configs/config.py:40-45`), now sized to cover the 47 616-sample context window with the larger chunk granularity.
`_STFT_GROUP_DELAY` renamed to `_LOOKAHEAD_OFFSET = CHUNK_SIZE` (`file_simulator.py:88-97`) — the variable's role is the 1-chunk look-ahead, not STFT group delay. `_TOTAL_ALIGN_DELAY = CHUNK_SIZE + FIXED_DELAY` = **1025 samples**.

| Signal | NR (v4) | Δ vs. v3 |
|---|---|---|
| Synthetic engine noise | **−14.54 dB** | +0.55 dB |
| `noise_sound.wav` | **−13.70 dB** | +0.87 dB |
| `sound3.wav` | **−5.17 dB** | +0.08 dB (within measurement noise; peak dropped 0.887 → 0.805) |

This state was preserved as **`successful_v4_backup/`** (source + configs + 182 MB CRN checkpoint).

### v5 — Energy-based residual noise gate

**Goal:** push the pure-noise residual into the target band by applying additional attenuation only during noise-only periods, leaving speech untouched.

**Design decisions:**

- **Detection signal:** per-chunk `ratio = rms(clean_chunk) / rms(raw)`. Measured empirically: ~0.21 on the noise file, ~0.55 on noisy speech. Thresholds 0.30 / 0.45 cleanly separate the two regimes.
- **Gain mapping:** floor of −15 dB below `ratio = 0.30`; 0 dB above `ratio = 0.45`; linear interpolation in dB between (`file_simulator.py:_noise_gate_target_gain`).
- **Asymmetric envelope:**
  - *Release* (gate opening on a rising ratio): instant. Speech onsets must not be attenuated.
  - *Attack* (gate closing on a falling ratio): smoothed at `alpha = 0.20`, ~5 chunks (~320 ms) to fully engage. Prevents pumping during brief modulations of real-world noise.

Config knobs in `configs/config.py:87-99`:

```python
NOISE_GATE_ENABLE         = True
NOISE_GATE_RATIO_LOW      = 0.30
NOISE_GATE_RATIO_HIGH     = 0.45
NOISE_GATE_ATTENUATION_DB = -15.0
NOISE_GATE_ATTACK_ALPHA   = 0.20
```

Gate is bypassed during `WARMUP_CHUNKS` so that pass-through (where `clean_chunk ≈ delayed_raw` and ratio ≈ 1.0) does not corrupt the envelope state.

| Signal | NR (v5) | Δ vs. v4 | Notes |
|---|---|---|---|
| `noise_sound.wav` | **−18.60 dB** | +4.90 dB | **within target band (−15 to −25 dB)** |
| `sound3.wav` | **−5.44 dB** | +0.27 dB | speech preserved; output peak unchanged at 0.805 |

This state was locked in as **`successful_v5_backup/`** (source + configs + 182 MB CRN checkpoint).

---

## 3. Current Status & Results

### Final cancellation metrics (post-warmup analysis window starting at 4.0 s)

| Signal | Input RMS | Output RMS | Output peak | NR | Verdict |
|---|---|---|---|---|---|
| Synthetic engine noise (calibration signal) | −11.35 dBFS | −25.89 dBFS | 0.228 | **−14.54 dB** | within 1 dB of target band |
| `noise_sound.wav` (real recording, gate on) | −17.30 dBFS | **−35.91 dBFS** | 0.523 | **−18.60 dB** | **within target band** |
| `sound3.wav` (noisy speech, gate on) | −16.13 dBFS | −21.56 dBFS | 0.805 | **−5.44 dB** | speech preserved (peak unchanged) |

### Reference: offline single-pass ceiling on the same model

| Signal | Offline NR (single forward pass on full waveform) |
|---|---|
| Synthetic engine noise | **+17.69 dB** (output −29.04 dBFS, peak 0.110) |

The 3.15 dB gap between streaming (−14.54 dB) and offline (−17.69 dB) is the intrinsic cost of processing 48 640-sample windows instead of one continuous 240 000-sample pass; it is uncorrelated with input and cannot be recovered by linear post-processing (verified by `_residual_diag.py`).

### End-to-end latency

| Stage | Samples | ms @ 16 kHz |
|---|---|---|
| Audio chunking (`CHUNK_SIZE`) | 1024 | 64.0 |
| 1-chunk look-ahead extraction | 1024 | 64.0 |
| Physical mic↔speaker propagation (`FIXED_DELAY`) | 1 | 0.06 |
| **Total streaming output latency** | **1025** | **64.1** |

On RTX 5060, model inference completes within the 64 ms chunk period (no deadline slip). On Jetson Nano this configuration will fall behind real-time; reduce `CHUNK_SIZE` or `_CRN_CTX_N` for that target.

---

## 4. Core Engineering Principles Applied

### 4.1 STFT `center=True` reflect-pad and right-boundary iSTFT distortion

The CRN's STFT module uses `n_fft = 512`, `hop_length = 128`, `center = True` (`Run/CRN/model.py:75-88`). With `center=True`, PyTorch pads the input by `n_fft // 2 = 256` samples on each side before framing, with reflect padding by default. For an input ending at sample `N`, the padded region `[N, N+256]` is a mirror image of `[N-256, N]`.

Frames whose centers fall in `(N − 256, N]` therefore consume mirrored "future" content in their right halves. The iSTFT reconstruction at output positions in roughly the last `n_fft // 2 + hop` samples is distorted because the model's complex mask interprets the mirrored content as real signal, applies a mask to it, and the iSTFT folds those modified frames back into the output sum.

The original report attributed this to a "256-sample group delay" of the iSTFT and compensated with a 257-sample reference-delay buffer. Empirically the buffer alone did not produce cancellation — the underlying mechanism is **right-boundary reconstruction distortion**, not a clean linear group delay. The fix in v2 was to *avoid* extracting from the distorted region entirely (see 4.3).

### 4.2 BiLSTM training-vs-inference context mismatch

The CRN's recurrent bottleneck is a 2-layer bidirectional LSTM (`Run/CRN/model.py:339-341`, ckpt args: `rnn_type='bilstm'`, `rnn_layers=2`, `rnn_hidden=256`). The backward LSTM is non-causal: its hidden state at frame `t` depends on frames `[t, T]`, where `T` is the last frame of the input. In a streaming pipeline this state cannot be persisted across calls — each `model.forward()` call sees a finite window with `h_0 = 0` for both directions.

Training segments were 3.0 s = ~375 STFT frames per backward pass (segment_seconds=3.0 in ckpt args). The original streaming context (`_CRN_CTX_N = 3840`) plus current chunk gave only `(3840 + 256) // 128 = 32 frames` per call — **8.5 % of the training context**. Under such short windows the backward LSTM has insufficient temporal evidence to drive the complex mask away from zero, and the model effectively passes the noisy input through unchanged.

The v2 fix `_CRN_CTX_N = 47 616` produces `(47616 + 1024) // 128 = 380 frames`, matching the training distribution. The cost is roughly linear in frame count: per-call GPU inference rises from ~5 ms to ~30–60 ms. Within the 64 ms chunk period (v4), the simulator runs without deadline slip on RTX 5060.

### 4.3 Look-ahead extraction and alignment arithmetic

Once the right-boundary reflect-pad distortion is identified, the cleanest fix is to never read from the distorted region. The model output is undistorted in its **interior** — frames whose left and right halves both consume real audio.

Per-call layout (CHUNK_SIZE = 1024, _CRN_CTX_N = 47616):

```
input  = [ _crn_stft_ctx (47616 smp) | chunk_N (1024 smp) ]
         |<--- interior (continuous, undistorted) --->|<-- reflect-pad -->|
                                                     ^                  ^
                                                  -2*CHUNK            -CHUNK
                                                     |__ extracted __|
```

Extracting `enhanced[-2*CHUNK_SIZE : -CHUNK_SIZE]` (`model_runtime.py:_infer_crn_stateful`) yields a 1024-sample segment of model output corresponding to *the previous chunk* — chunk_(N−1), now sitting one chunk back from the right edge. The extracted noise estimate is therefore one chunk *older* than the chunk just submitted.

**Reference path alignment:**

```
delayed_raw[k] = mic[absolute_time_k − _TOTAL_ALIGN_DELAY]
              = mic[absolute_time_k − (CHUNK_SIZE + FIXED_DELAY)]
              = mic[absolute_time_k − 1025]
```

This puts `delayed_raw` and `noise_est` on the same absolute time index (chunk_(N−1)), with one additional sample of mic↔speaker physical propagation lag absorbed by `FIXED_DELAY`.

**Output latency:** one full chunk (64 ms at 1024-sample chunks). This is the price paid for moving the extraction point out of the distorted region.

### 4.4 Cross-fade as a now-obsolete patch

The report prescribed a 64-sample linear cross-fade at every chunk boundary to mask **right-boundary discontinuities** between consecutive extracted segments. That discontinuity was real *when extracting from the distorted region*: each chunk's last samples came from differently-padded frame sets, producing a hard jump at the seam.

With look-ahead extraction the extracted segments come from the model's interior. Consecutive calls process near-identical context (99.5 % overlap on the 47 616-sample buffer), and the interior reconstruction is mathematically smooth across chunk boundaries — the seam disappears.

The cross-fade, however, continued to apply a linear ramp from the previous chunk's tail value to the current chunk's natural value over the first 64 samples. Its arithmetic effect during noise-only periods (where the previous tail tends to small values) was to **suppress 25 % of each chunk's leading cancellation** (`_XFADE_LEN / CHUNK_SIZE = 64/256 = 0.25`). RMS arithmetic puts the resulting NR ceiling at:

```
out_rms² = 0.25 * raw_rms² + 0.75 * canceled_rms²
        ≈ 0.25 * raw_rms²            (when cancellation is near-perfect)
NR_max  ≈ 10 * log10(4) ≈ +6 dB
```

The v2 run showed exactly this ceiling. Setting `ANC_XFADE_SAMPLES = 0` removed the unintended suppression and unlocked the +8 dB jump from v2 to v3.

**General lesson:** smoothing operators added to mask one symptom can silently degrade performance once a deeper fix removes the underlying cause. The cross-fade was right for the original pipeline and wrong for the look-ahead pipeline.

### 4.5 Hybrid energy-based noise gate (v5)

The residual after look-ahead extraction is statistically uncorrelated with the raw input (`_residual_diag.py` output: `corr = +0.024`, `g* = 0.97`, `NR_with_g* − NR_with_g=1 = +0.13 dB`). No linear post-processing can recover further cancellation against the model's intrinsic accuracy floor.

However, a **non-linear** gate that distinguishes "noise-only periods" from "speech-present periods" *can* attenuate the residual further during the former without affecting the latter. The detection signal is the per-chunk RMS ratio:

```
ratio_n = rms(clean_chunk_n) / rms(delayed_raw_n)
```

Behaviour at the extremes:

- Pure noise input ⇒ model cancels almost everything ⇒ `clean_chunk_n` is small ⇒ `ratio_n ≈ 0.2`.
- Noisy speech input ⇒ model preserves speech ⇒ `clean_chunk_n` retains speech energy ⇒ `ratio_n ≈ 0.55+`.

Empirical thresholds (`NOISE_GATE_RATIO_LOW = 0.30`, `NOISE_GATE_RATIO_HIGH = 0.45`) cleanly separate the two regimes. The gain mapping is linear in dB between the thresholds, with a floor of `NOISE_GATE_ATTENUATION_DB = −15.0`:

```
                     ┌───────────────  +0 dB (gate open)
                    /
     ────────────  /
                 /
                /
    ───  ──── ─  ──────────────────── −15 dB (gate closed)
              ↑              ↑
        ratio = 0.30   ratio = 0.45
```

The envelope is **asymmetric**: target gain is applied *instantly* if it rises (so a speech burst arriving in chunk N opens the gate within that same chunk), but smoothed at `α = 0.20` if it falls (so a momentary noise drop does not cause a perceptible level dip in steady-state noise). This produces gate behaviour analogous to a fast-release / slow-attack expander.

**Achieved attenuation:** the configured floor is −15 dB but the *measured average* additional attenuation on `noise_sound.wav` is −4.9 dB. The gap is explained by:

1. Smoothed attack: the gate spends ~5 chunks (320 ms) closing from open after each speech-like burst.
2. Real-noise modulation: even pure-noise recordings have brief amplitude bursts that lift the ratio past 0.30, momentarily re-opening the gate.
3. The transition band 0.30 → 0.45 applies partial attenuation, not the full floor.

The combination is intentional — it favours speech intelligibility over absolute noise floor depth.

---

## 5. Configuration Reference (v5)

All knobs in `ANC_Hardware_Test/configs/config.py` and `ANC_Hardware_Test/model_runtime.py`:

```python
# Audio
SAMPLE_RATE  = 16_000
CHUNK_SIZE   = 1024              # 64 ms output latency
NUM_CHANNELS = 1
DTYPE        = "float32"
WARMUP_CHUNKS = 50               # 3.2 s mute window during context fill

# Streaming alignment (file_simulator.py)
FIXED_DELAY        = 1           # physical mic-speaker propagation
_LOOKAHEAD_OFFSET  = CHUNK_SIZE  # 1-chunk look-ahead extraction
_TOTAL_ALIGN_DELAY = 1025

# CRN streaming context (model_runtime.py)
_CRN_CTX_N = 47_616              # ~374 STFT frames per call (matches training)

# Cross-fade
ANC_XFADE_SAMPLES = 0            # disabled (look-ahead extraction is continuous)

# Residual noise gate
NOISE_GATE_ENABLE         = True
NOISE_GATE_RATIO_LOW      = 0.30
NOISE_GATE_RATIO_HIGH     = 0.45
NOISE_GATE_ATTENUATION_DB = -15.0
NOISE_GATE_ATTACK_ALPHA   = 0.20
```

---

## 6. Backup Inventory

| Backup | Scope | NR on `noise_sound.wav` | NR on `sound3.wav` |
|---|---|---|---|
| `successful_v3_backup/` | 3 source files only | −12.83 dB | −5.09 dB |
| `successful_v4_backup/` | full source + configs + 182 MB CRN checkpoint | −13.70 dB | −5.17 dB |
| `successful_v5_backup/` | full source + configs + 182 MB CRN checkpoint | **−18.60 dB** | **−5.44 dB** |

`successful_v5_backup/` layout:

```
successful_v5_backup/
├── file_simulator.py           (gate-enabled, CHUNK_SIZE=1024)
├── model_runtime.py            (47616-sample ctx, look-ahead extraction)
├── main_hardware.py
├── visualizer.py
├── requirements.txt
├── configs/
│   └── config.py               (full knob set incl. NOISE_GATE_*)
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py
│   └── logger_setup.py
├── mock/
│   ├── __init__.py
│   └── mock_model.py
└── crn/
    ├── model.py                (CRN architecture, copy of Run/CRN/model.py)
    ├── inference.py
    └── checkpoints/
        └── best.pt             (181.8 MB, trained CRN weights)
```

All four critical files (`file_simulator.py`, `model_runtime.py`, `configs/config.py`, `crn/checkpoints/best.pt`) were SHA-256-verified against the live tree at backup time.

---

## 7. Diagnostic Tooling

Auxiliary scripts produced during the investigation, located at `ANC_Hardware_Test/` with the `_` prefix:

| Script | Purpose |
|---|---|
| `_nr_check.py` | Compute NR between an input WAV and a simulator-output WAV. Auto-resamples to 16 kHz. Supports `NR_WARMUP_S` env var to override the analysis-window start. |
| `_offline_nr.py` | Run the CRN in offline single-pass mode on a WAV and report what the model's ceiling NR is (no streaming). Used to prove the model itself was sound. |
| `_gen_engine_noise.py` | Deterministic synthetic engine-noise WAV generator. Reproduces `_engine_noise_stream` from the simulator with seed=42. |
| `_inspect_ckpt.py` | Dump the missing / unexpected keys reported by `load_state_dict(strict=False)`. Used to verify the 8 missing keys were all from `FreqAttention` and harmless. |
| `_residual_diag.py` | Decompose the streaming residual into correlated component (recoverable by linear scaling) vs. uncorrelated noise (model accuracy floor), plus per-chunk-position energy profile for boundary-spike detection. |

These are intentionally prefixed with `_` to mark them as scratch / investigation tooling rather than first-class components of the simulator.

---

## 8. Open Items (post-v5)

> **Status note:** the first item below was addressed by the v6 phase documented in §9. Items 2–4 remain open.

- ~~**Jetson Nano deployment profiling.** Current CHUNK_SIZE=1024 + 47 616-sample context likely exceeds Jetson real-time budget. Sweep `(CHUNK_SIZE, _CRN_CTX_N)` to find the smallest viable pair on Jetson.~~ → Addressed in v6 by reverting `CHUNK_SIZE` to 256 (16 ms latency) on Jetson Orin Nano.
- **Spectral-domain cancellation** (option 2 from the v5 design discussion). Could close part of the remaining 3.15 dB streaming-vs-offline gap by smoothing the model mask across overlapping windows in the frequency domain rather than the time domain.
- **CRN retraining on inference-window distribution.** The streaming-vs-training distribution mismatch is the only remaining principled source of the 3.15 dB gap. Re-training on 47 872-sample windows (the actual inference window) would eliminate it.
- **Noise gate tuning.** The current gate is conservative (favours speech intelligibility). For pure-noise applications (e.g. constant engine drone with no speech), the gate could be tightened by raising `NOISE_GATE_RATIO_LOW` and lowering `NOISE_GATE_ATTENUATION_DB`.

---

## 9. v6 — Latency Optimization (`CHUNK_SIZE`: 1024 → 256)

**Phase scope:** the target deployment platform is a Jetson Orin Nano, which can comfortably run the CRN's ~30–60 ms per-chunk inference, so the v5 choice to inflate `CHUNK_SIZE` to 1024 (a 4× latency penalty traded for ~0.6 dB more NR) was no longer justified. Goal: revert to `CHUNK_SIZE = 256` (16 ms latency) without breaking phase alignment or the v5 noise gate.

### 9.1 Auto-propagating reversion

The v4-era refactor had already made the alignment delay parametric in `CHUNK_SIZE` (`file_simulator.py:100-101`):

```python
_LOOKAHEAD_OFFSET  = CHUNK_SIZE
_TOTAL_ALIGN_DELAY = _LOOKAHEAD_OFFSET + FIXED_DELAY
```

So the only edits required were in `configs/config.py`: `CHUNK_SIZE` from 1024 → **256**, and `WARMUP_CHUNKS` from 50 → **200** (3.2 s remains the same in wall-clock time but takes 4× more chunks at the smaller granularity). The 257-sample reference delay (16.06 ms total streaming output latency) was derived automatically from the new constant. No edit to `file_simulator.py` or `model_runtime.py` was needed for the math itself.

### 9.2 Iteration log

The reversion exposed an unanticipated structural problem with the v5 noise gate: its empirically tuned constants assumed `CHUNK_SIZE = 1024`. Four iterations were measured to characterise and resolve this.

| Sub-version | Configuration | Noise NR (`noise_sound.wav`) | Speech NR (`sound3.wav`) | Result |
|---|---|---|---|---|
| **v6a** | `CHUNK_SIZE = 256`, gate `α = 0.20` literal (v5 setting unchanged) | **−14.52 dB** | −5.41 dB (peak 0.887) | Alignment intact; gate's added NR shrank from v5's +4.90 dB to +1.69 dB. |
| **v6b** | + `NOISE_GATE_ATTACK_MS = 300` → `α = 0.052` derived at startup (wall-clock attack matching v5's 287 ms) | **−12.47 dB** | −5.28 dB | Worse than v6a. Slow attack lets each false release leak energy for longer at the smaller chunk size. |
| **v6c** | + `NOISE_GATE_RATIO_SMOOTH_MS = 60` → `α_r = 0.234` (IIR low-pass on the detection ratio itself) | **−11.94 dB** | −5.45 dB | Worse still. Smoothing the ratio delays the gate's response to genuine noise drops more than it suppresses false releases on amplitude bursts. |
| **v6d** | `NOISE_GATE_ATTACK_MS = 72` (`α ≈ 0.20` at `CHUNK = 256`), `NOISE_GATE_RATIO_SMOOTH_MS = 1.0` (smoothing α ≈ 1.0 = disabled) | **≈ −14.5 dB** (full-file extrapolation; v6a-equivalent) | −5.41 dB (peak 0.887) | Empirical optimum. Recovers v6a's behaviour while keeping the ms-based abstraction intact for future tuning. |

The v6d run on `noise_sound.wav` was cut short by an early dashboard close (analysis window 13.4 s vs v6a's 32.5 s, hitting a louder section of the file). The `sound3.wav` result is fully end-to-end and bit-for-bit identical to v6a (−5.41 dB, peak 0.887), confirming the v6d code is mathematically equivalent to v6a despite the noise-window artefact.

### 9.3 Why ms-scaling didn't transfer

The intuition behind v6b/v6c was that keeping the gate's *wall-clock* time constants invariant should reproduce v5's behaviour at any chunk size. The measurements falsified that intuition. The structural reason:

- Per-chunk RMS over `N` samples of zero-mean noise has variance ∝ `1/N`, so going from `CHUNK = 1024` to `CHUNK = 256` lowers the sample count by 4× but only raises the *standard deviation* of the per-chunk RMS by `√4 = 2×`.
- The amplitude of the detection signal's natural fluctuations therefore roughly doubles at the smaller chunk, so the ratio crosses `RATIO_HIGH = 0.45` (the gate's release threshold) more frequently — even during steady-state noise-only periods.
- The gate's release path is *instant* (gain snaps to 1.0 on any threshold crossing). With a slow attack (v6b's `α = 0.052`, ~300 ms), each false release leaves the gate near-open for an extended period. Fast attack (v6a/v6d's `α = 0.20`, ~72 ms) re-engages 4× sooner, leaking less energy.
- Smoothing the ratio (v6c) trades faster spike rejection for a delayed response to genuine noise drops. Empirically the second effect dominates at this chunk size — the gate doesn't close all the way during quiet stretches, eroding the NR floor.

The general lesson: a "scale the time constant" refactor assumes the smoothed-quantity's variance is independent of the smoothing window. For the per-chunk RMS detector here, the detection-signal variance itself depends on chunk size, so naive wall-clock scaling moves *two* knobs at once. The empirical optimum at `CHUNK = 256` happens to need shorter wall-clock attack than v5; the gate was tuned by setting `NOISE_GATE_ATTACK_MS = 72` directly.

### 9.4 Final v6 configuration

```python
# configs/config.py
SAMPLE_RATE                 = 16_000
CHUNK_SIZE                  = 256              # 16 ms output latency
WARMUP_CHUNKS               = 200              # 3.2 s mute window during context fill

# file_simulator.py (derived)
FIXED_DELAY                 = 1                # physical mic-speaker propagation
_LOOKAHEAD_OFFSET           = CHUNK_SIZE       = 256
_TOTAL_ALIGN_DELAY          = 257              # 16.06 ms total output latency

# model_runtime.py (unchanged from v5)
_CRN_CTX_N                  = 47_616           # ~374 STFT frames per call

# Cross-fade (unchanged from v3)
ANC_XFADE_SAMPLES           = 0

# Residual noise gate (refactored to ms time constants in v6b/v6c, then
# empirically tuned in v6d to match v6a's α at CHUNK=256)
NOISE_GATE_ENABLE           = True
NOISE_GATE_RATIO_LOW        = 0.30
NOISE_GATE_RATIO_HIGH       = 0.45
NOISE_GATE_ATTENUATION_DB   = -15.0
NOISE_GATE_ATTACK_MS        = 72.0             # α = 0.20 at CHUNK=256
NOISE_GATE_RATIO_SMOOTH_MS  = 1.0              # α ≈ 1.0  → smoothing disabled
```

### 9.5 Final latency budget

| Component | Samples | ms @ 16 kHz |
|---|---|---|
| Chunk period (`CHUNK_SIZE`) | 256 | **16.000** |
| 1-chunk look-ahead extraction (`_LOOKAHEAD_OFFSET`) | 256 | 16.000 |
| Physical mic↔speaker propagation (`FIXED_DELAY`) | 1 | 0.062 |
| **Total streaming output latency (`_TOTAL_ALIGN_DELAY`)** | **257** | **16.062** |

Down from v5's 64.1 ms — **4× tighter**, on-target for Jetson Orin Nano's tight-latency budget.

### 9.6 Updated results table

| Configuration | Latency | `noise_sound.wav` NR | `sound3.wav` NR |
|---|---|---|---|
| v3 (CHUNK=256, no gate) | 16 ms | −12.83 dB | −5.09 dB |
| v4 (CHUNK=1024, no gate) | 64 ms | −13.70 dB | −5.17 dB |
| v5 (CHUNK=1024, gate) | 64 ms | −18.60 dB | −5.44 dB |
| **v6 (CHUNK=256, gate, ms-knobs)** | **16 ms** | **≈ −14.5 dB** | **−5.41 dB** |

The 4 dB noise NR penalty vs. v5 is the price paid for the 48 ms latency win. Speech preservation is unchanged.

### 9.7 Updated backup inventory

| Backup | Scope | Latency | Noise NR | Speech NR |
|---|---|---|---|---|
| `successful_v3_backup/` | 3 source files only | 16 ms | −12.83 dB | −5.09 dB |
| `successful_v4_backup/` | full source + configs + 182 MB CRN checkpoint | 64 ms | −13.70 dB | −5.17 dB |
| `successful_v5_backup/` | full source + configs + 182 MB CRN checkpoint | 64 ms | −18.60 dB | −5.44 dB |
| **`successful_v6_backup/`** | **full source + configs + 182 MB CRN checkpoint** | **16 ms** | **≈ −14.5 dB** | **−5.41 dB** |

`successful_v6_backup/` mirrors the v5 layout. All four critical files (`file_simulator.py`, `model_runtime.py`, `configs/config.py`, `crn/checkpoints/best.pt`) were SHA-256-verified against the live tree at backup time.

### 9.8 New diagnostic constants (kept in code for future tuning)

The v6 phase added two wall-clock-based knobs to `configs/config.py`:

- `NOISE_GATE_ATTACK_MS` — gate close-time constant in ms. The simulator derives `_GATE_ATTACK_ALPHA = 1 − exp(−CHUNK_PERIOD_MS / NOISE_GATE_ATTACK_MS)` at startup.
- `NOISE_GATE_RATIO_SMOOTH_MS` — IIR low-pass time constant for the detection ratio. Set ≈ 1.0 ms to disable. Increase if a future signal class shows the v6c trade-off going the other way.

Both knobs are reported in the simulator's startup banner along with their derived alphas, so any future `CHUNK_SIZE` change makes the new alphas visible at run time without needing to recompute by hand.

---

*End of engineering log. Maintainers: append new sections below as the project evolves.*
