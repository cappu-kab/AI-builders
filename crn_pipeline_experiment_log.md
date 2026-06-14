# CRN Pipeline — Experiment Log
**Date:** 2026-06-11  
**Researcher:** bachkukkik  
**Goal:** Record Thai speech via floating INMP441 mic → denoise with CRN BiLSTM → play back through MAX98357A speaker on Jetson Orin Nano

---

## Hardware Setup

```
INMP441 mic (floating — SD pin not grounded) → I2S1 → ESP32-S3
Jetson Orin Nano → UART (ttyTHS0, 2 Mbaud) → ESP32-S3 → I2S0 → MAX98357A → Speaker
```

- MCU: ESP32-S3 (`work_sound_v5_1.ino`)
- UART baud: 2,000,000
- Audio: 16 kHz mono, int16 PCM
- CRN model: BiLSTM, checkpoint at `AI_builders/Run/CRN/checkpoints/best.pt`
- Training: 70% LF noise + 30% HF transients, SNR −5 to +15 dB, curriculum over 15 epochs

---

## Known Hardware Constraint

INMP441 SD pin is floating → I2S data line picks up electrical interference → produces:
- **Periodic impulse spikes** ("kik") at near-full-scale amplitude
- **Background electrical hum** below 200 Hz
- **Low SNR** input (floating mic noise ≠ training noise distribution)

Software-only fix is partial. Full fix requires 100 kΩ pull-down resistor on SD pin.

---

## Experiment Versions

### v1 — Baseline (original code)
**Changes:** none — first run  
**Problem found:** Garbled serial output from ESP32  
**Root cause:** `LATENCY_DEBUG true` in Arduino → 8-byte mic header, Python expected 4-byte  
**Fix:** Set `LATENCY_DEBUG false` in `work_sound_v5_1.ino`, reflash ESP32

---

### v2 — Working baseline
**Key settings:**
```python
SEND_RATE  = 1.10 × BYTES_PER_SEC   # 110% real-time
_DRY_MIX   = 0.20
_TARGET_RMS = 0.15
# volume: RMS-normalize then multiply directly
```
**Result:** Audio plays. Volume good at --volume 2.5.  
**Problem:** "kik" periodic, noise at tail, kik gets very loud at --volume 3+  
**Finding:** At 110% send rate, UART RX buffer (16384 bytes) overflows at 5.12 s → byte drop → click

---

### v3 — UART pacing fix
**Changed:** `SEND_RATE 1.10 → 1.02`  
**Math:** overflow now at 16384 / (32000 × 0.02) = 25.6 s — safe for 10 s recordings  
**Result:** Periodic 5 s kik eliminated. Residual kik still present (hardware origin)

---

### v4 — Clip-based spike gate
**Changed:** Added `np.clip(pcm_f32, ±5σ)` before CRN  
**Result:** Kik slightly reduced. Tail noise still audible.  
**Problem found:** Clipping leaves flat top → still sounds like softer click  
**Finding:** Clipping is wrong approach for impulse removal

---

### v5 — Ctrl+C recovery + peak normalization
**Changed:**
- Added `stopmic` flush before `startmic` → recovers from interrupted sessions
- Added `if peak > 1.0: enh_f32 /= peak` to prevent clipping at high volume
- Fade-out extended to 0.25 s
**Problem found:** Peak normalization killed volume — `--volume 10` same loudness as `--volume 3`  
**Finding:** Normalizing to peak=1.0 makes volume parameter useless

---

### v6 — Interpolation de-spike
**Changed:** Replaced `np.clip` spike gate with `np.interp` over spike samples  
**Rationale:** Interpolation produces smooth waveform at spike location; clip leaves discontinuity  
**Result:** Kik noticeably reduced  
**Changed also:** `_DRY_MIX 0.20 → 0.08`, fade-out 0.25 → 0.50 s

---

### v7 — Noise gate (failed)
**Changed:** Added frame-based soft noise gate (thresh=0.015 RMS)  
**Hypothesis:** Gate quiet frames to suppress tail noise  
**Result:** Speech almost inaudible, kik unchanged  
**Root cause:** Speech RMS from floating mic < gate threshold → speech gated out. Kik amplitude > threshold → passes through. Gate was backwards for this use case.  
**Conclusion:** Noise gate not suitable when SNR is negative (noise louder than speech after CRN)

---

### v8 — Tighter de-spike
**Changed:** De-spike threshold `σ × 3.0 → σ × 2.5`  
**Result:** Kik slightly lower. No speech degradation.

---

### v9 — Musical noise suppression + lower dry mix
**Changed:**
- Added `_suppress_musical_noise()` call after CRN (was defined but never called!)
- `_DRY_MIX 0.08 → 0.03`
**Result:** ~90% quality. Noise background noticeably reduced.  
**Finding:** `_suppress_musical_noise()` was a significant unused feature — zero-phase spectral smoothing removes tonal warble left by CRN mask. Score: **90%**

---

### v10 — Wider spectral smoothing (failed)
**Changed:** Musical noise suppression `size=3 → size=5`  
**Result:** Speech sounds echoey and unnatural  
**Root cause:** 5-frame = 20 ms window — approaches phoneme duration, smears speech  
**Conclusion:** `size=3` (12 ms) is safe upper limit for zero-phase spectral smoothing

---

### v11 — Combined best settings
**Changed:** Reverted smoothing to size=3. Kept 2.5σ de-spike.  
**Result:** Slightly fewer kiks vs v9. Background noise still present.

---

### v12 — High-pass filter 200 Hz ✓ BEST
**Changed:** Added Butterworth high-pass at 200 Hz, order=4, after dry mix  
**Rationale:** CRN residual LF noise below 200 Hz. Second-layer HP removes it. Male voice fundamentals (80–150 Hz) partly cut but harmonics (200 Hz+) preserve intelligibility.  
**Result:** Cleanest output. Score: **95%**

```python
# v12 final settings
_DRY_MIX     = 0.03
_TARGET_RMS  = 0.15
SEND_RATE    = 1.02 × BYTES_PER_SEC
de-spike     = σ × 2.5 (interpolation, not clip)
smoothing    = size=3 zero-phase (12 ms window)
highpass     = 200 Hz Butterworth order=4
fade_out     = 0.50 s
```

---

### v13 — Bandpass 200–7000 Hz (worse than v12)
**Changed:** HP → bandpass 200–7000 Hz, `_DRY_MIX 0.03 → 0.02`  
**Hypothesis:** Cut HF harshness above 7 kHz for smoother speech  
**Result:** Speech sounds thinner, loses presence. Reverted to v12.  
**Finding:** CRN output above 7 kHz is not harsh — cutting it removes air from speech.

---

## Summary Table

| Technique | Result |
|---|---|
| UART rate 110% | Buffer overflow → kik every 5.12 s — BAD |
| UART rate 102% | Safe for 10 s recordings — GOOD |
| Clip spike gate (5σ) | Flat-top remnant → soft click — MEDIOCRE |
| Interp de-spike (2.5σ) | Smooth waveform at spike location — GOOD |
| Noise gate | Kills speech when SNR negative — BAD |
| Musical noise suppression size=3 | Removes CRN warble — GOOD |
| Musical noise suppression size=5 | Echoey — BAD |
| Dry mix 0.20 | Too much raw noise bleed — BAD |
| Dry mix 0.03 | Good balance — GOOD |
| Peak normalization | Volume parameter useless — BAD |
| High-pass 200 Hz | Kills LF hum effectively — GOOD |
| Bandpass 200–7000 Hz | Too thin — BAD |
| Fade-out 0.50 s | Hides tail noise cleanly — GOOD |

---

## Remaining Limitations

1. **Kik (low level)** — hardware root cause: floating INMP441 SD pin picks up EMI. Software de-spike reduces but cannot fully eliminate. **Fix: 100 kΩ pull-down on SD pin to GND.**
2. **Background noise residual** — floating mic noise is not in CRN training distribution (training: HVAC/hum; mic produces: electrical interference). **Fix: retrain CRN with electrical impulse noise added to noise cocktail.**
3. **LF speech cut** — HP at 200 Hz removes male voice fundamentals. Trade-off accepted. If voice sounds thin, try cutoff at 150 Hz.

---

## Recommended Next Steps (Priority Order)

1. **Hardware fix** — solder 100 kΩ pull-down on INMP441 SD pin → eliminates kik source entirely
2. **Retrain CRN** — add impulse noise to `dataset.py` noise cocktail → fine-tune 5–10 epochs from `best.pt`
3. **ONNX on Jetson** — deploy `hf_deploy/model.onnx` with onnxruntime → no cuDNN warning, faster inference, no torch dependency
