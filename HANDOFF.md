# ANC Pipeline — Session Handoff
**Written:** 2026-05-27  
**Purpose:** Resume tomorrow without re-deriving anything. All findings are from code inspection or measured benchmarks, not assumptions.

---

## 1. PROJECT GOAL

Feedforward ANC on a **Jetson Orin Nano**.

- **Mic 1 (reference):** placed as close to the noise source as possible. Captures noise FIRST.
- **Mic 2 (error):** in the quiet zone to be protected.
- **2 speakers** straddling the error mic, creating a ~30 cm quiet zone (not a single null point).
- **Target band:** <200 Hz only. Deliberately limited to keep wavelengths long and avoid spatial aliasing.
- **Latency policy:** NOT hard real-time. Bounded, constant latency is acceptable. The strategy is: measure the total end-to-end system delay once at deployment, then phase-shift the anti-noise to align it with the incoming noise wave at the error mic.
- **Quality constraint:** Keep the BiLSTM CRN. GRU was tested on real recordings and its quality was rejected. Do not propose GRU again.

---

## 2. ENVIRONMENT FIXES ALREADY DONE

> **Always activate before running anything:** `source ~/anc_env/bin/activate`

### 2a. LD_LIBRARY_PATH fix (permanent, in activate script)

PyTorch 2.0.0+nv23.05 requires three extra shared-library directories. These were appended to `~/anc_env/bin/activate` by `fix_jetson_anc.sh`:

```bash
# ANC_LDPATH_BEGIN — shared libs required by PyTorch (torch 2.0.0+nv23.05)
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/extra:$VIRTUAL_ENV/lib/python3.8/site-packages/numpy.libs:$VIRTUAL_ENV/lib/python3.8/site-packages/scipy.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# ANC_LDPATH_END
```

This fixes the openblas/gfortran dependency. Do not remove or reorder these lines.

### 2b. model_runtime.py path fix

`_AI_DIR` in `~/ANC_Hardware_Test/simulation/model_runtime.py` was wrong — it pointed to `~/ANC_Hardware_Test` instead of `~/AI_builders`. Fixed by `fix_jetson_anc.sh`:

```python
# Before (wrong):
_AI_DIR = _ANC_DIR.parent

# After (correct):
_AI_DIR = Path.home() / "AI_builders"   # Jetson: ~/AI_builders
```

**Backup preserved at:** `~/ANC_Hardware_Test/simulation/model_runtime.py.bak` — do not delete.

### 2c. Correct checkpoint to use

```
~/AI_builders/Run/CRN/checkpoints_crn2/best.pt
```

This is the BiLSTM checkpoint (rnn_type=bilstm, confirmed in `checkpoints_crn2/args.json`). It was trained with `lf_noise_ratio=0.8` and `lf_peak=6.0` — explicitly optimized for LF noise cancellation. Not the default `checkpoints/best.pt` (also BiLSTM, slightly lower val_sisnr).

### 2d. Jetson has no internet

Cannot `pip install`, `apt-get`, or `curl` from the board. All dependencies must already be installed or transferred from Windows via scp.

### 2e. Benchmark command that confirmed CRN loads correctly

```bash
source ~/anc_env/bin/activate
cd ~/ANC_Hardware_Test/simulation
# CPU-only run (confirmed backend='crn', NOT mock):
CUDA_VISIBLE_DEVICES="" python3 benchmark_crn.py \
    --ckpt ~/AI_builders/Run/CRN/checkpoints_crn2/best.pt \
    --warmup 10 --runs 50
```

**Measured CPU result:** Avg 3225 ms / p95 3980 ms / p99 4368 ms — 201× over the 16 ms chunk budget. Expected, because the BiLSTM reprocesses ~3 s of audio per chunk.

---

## 3. KEY FINDINGS

### 3a. CRN architecture — BiLSTM confirmed

**Exact RNN constructor, `model.py` lines 346–348:**
```python
self.rnn = nn.LSTM(input_size=bottleneck_dim, hidden_size=rnn_hidden,
                   num_layers=rnn_layers, batch_first=True,
                   bidirectional=True)
```
`bidirectional=True` confirmed. `checkpoints_crn2/args.json`: `rnn_hidden=256, rnn_layers=2, bottleneck_dim=512`.

### 3b. Per-call cost is permanent, not a startup cost

`_infer_crn_stateful()` (`model_runtime.py:741`):
```python
inp = np.concatenate([self._crn_stft_ctx, chunk])  # always 47,872 samples
out = self._torch_mdl(x)                            # full forward pass every time
```

The BiLSTM backward pass requires seeing the complete window to produce output at any time step. Every 256-sample chunk triggers a full 47,872-sample (375 STFT frames) LSTM forward pass. This cost cannot be amortized.

### 3c. warm_up() and _crn_rnn_hidden are dead code for CRN

`warm_up()` (`model_runtime.py:351`): returns 0 immediately for CRN because `self._ctx_buf is None`. Does nothing.

`_crn_rnn_hidden`: declared, initialized to `None`, reset in `reset()`, but **never read or written in `_infer_crn_stateful`**. The field exists but is never wired to the model.

`model.py:410`: `seq, _ = self.rnn(seq)` — hidden state is explicitly discarded with `_`. The stale comment in `_init_streaming` describing "persistent hidden state" is aspirational and was never implemented.

### 3d. CRITICAL: LFBypass branch handles the target band directly

`model.py:373`:
```python
hz_per_bin = sample_rate / n_fft   # = 31.25 Hz for n_fft=512, sr=16000
self.lf_bins = max(4, round(200.0 / hz_per_bin) + 1)   # = 7 bins → covers 0–187.5 Hz
```

`model.py:424–426`:
```python
lf_corr = self.lf_bypass(x[:, :, :self.lf_bins, :])   # operates on raw STFT bins 0–6
h = torch.cat([h[:, :, :self.lf_bins, :] + lf_corr,
               h[:, :, self.lf_bins:, :]], dim=-2)
```

The LFBypass reads STFT bins 0–6 (covering **0–187.5 Hz — almost exactly our target band**) directly from the raw input, bypassing the BiLSTM bottleneck entirely. It has its own dilated-conv stack (kernel=5, dilations=(1,2,4)):

```
Temporal receptive field = (5-1)×(1+2+4)+1 = 29 STFT frames × 8 ms/frame = 232 ms
```

**Implication for the context sweep:** the <200 Hz cancellation quality depends primarily on LFBypass (232 ms RF) and only secondarily on the BiLSTM (which contributes via the CNN decoder skip connections). LFBypass is fully saturated at any context window ≥ 232 ms. This is why shrinking `_CRN_CTX_N` from 47,616 samples (3 s) to ~8,000 samples (500 ms) should preserve most of the LF quality — the LFBypass stays fully fed.

### 3e. Phase-shift approach: why it works for periodic noise

For HVAC noise (a harmonic series of a fundamental f₀ at 50 or 60 Hz):
- Adding a fixed total system delay Δt accumulates phase `φ(f) = 2π × f × Δt` at each frequency.
- For all harmonics of f₀: correcting the fundamental's phase automatically aligns all harmonics, because they are integer multiples and the inter-harmonic phase relationship is preserved.
- One scalar sample offset corrects the entire harmonic series simultaneously.
- **Frequency drift matters:** at Δt = 100 ms (GPU target), a ±0.1 Hz drift at 200 Hz accumulates ±3.6° phase error → negligible. At Δt = 3,225 ms (CPU), the same drift accumulates ±116° → cancellation collapses at the high LF end. **CPU latency is a blocker specifically for 150–200 Hz.**

### 3f. GPU jitter matters at the top of the target band

At 200 Hz, each 1 ms of timing jitter = `2π × 200 × 0.001 = 1.26 rad = 72°` of phase uncertainty. GPU cuDNN jitter is typically ±1–2 ms → ±72–144° at 200 Hz. This is marginal. TensorRT with `DETERMINISTIC_KERNELS` reduces jitter to ±0.1–0.2 ms → ±14°, which is comfortable. For the 50–100 Hz core HVAC band, ±2 ms jitter = ±36° — acceptable without TensorRT.

### 3g. Checkpoint comparison (for reference)

| Checkpoint | Architecture | Date | Best val_SI-SNR |
|---|---|---|---|
| `checkpoints/best.pt` ← code default | BiLSTM | 5/10/2026 | **15.11 dB** (best score) |
| `checkpoints_crn2/best.pt` ← **use this** | BiLSTM, LF-optimized | 5/18/2026 | 13.55 dB |
| `checkpoints_gru_fixed/best.pt` | GRU | 5/18/2026 | — (quality rejected) |
| `checkpoints_gru/best.pt` | GRU | 5/18/2026 | — (quality rejected) |

`checkpoints_crn2` has lower SI-SNR on the general benchmark but was trained with `lf_noise_ratio=0.8, lf_peak=6.0, lf_cutoff=200.0` — deliberately optimized for LF noise. Use it for this ANC application.

---

## 4. CURRENT BLOCKER

### cuDNN version conflict

PyTorch 2.0.0+nv23.05 was compiled against cuDNN 8.6.0. The system has cuDNN 8.4.1 (from JetPack) visible on `LD_LIBRARY_PATH`. Loading the model on CUDA triggers:

```
cuDNN version incompatibility: PyTorch compiled against (8,6,0) but found runtime version (8,4,1).
```

### Two scripts already on the Jetson board:

- `~/diagnose_cudnn.sh` — prints all `libcudnn*` locations, LD_LIBRARY_PATH contents, and PyTorch's `torch/lib` path. **Has NOT been run yet.**
- `~/fix_cudnn.sh` — patches `~/anc_env/bin/activate` to put `torch/lib` first in LD_LIBRARY_PATH (so PyTorch's bundled cuDNN 8.6.0 shadows the system 8.4.1), then runs the GPU benchmark.

### To unblock tomorrow:

```bash
source ~/anc_env/bin/activate
bash ~/diagnose_cudnn.sh        # step 1: print diagnostic
bash ~/fix_cudnn.sh             # step 2: fix + run GPU benchmark
```

If `fix_cudnn.sh` still shows a version mismatch after running, paste the diagnostic output to a new Claude Code session — the fix path depends on exactly where the conflicting libcudnn lives.

**GPU latency is still UNMEASURED.** Expected range based on architecture analysis: 20–60 ms for full 3 s context on Orin Nano FP32.

---

## 5. THE PLAN (ordered, do not skip steps)

### Step 1 — Fix cuDNN → measure GPU baseline (immediate, ~1 hour)

```bash
source ~/anc_env/bin/activate
bash ~/fix_cudnn.sh
```

Expected output: `backend='crn'`, `device=cuda`, Avg latency **20–60 ms**.  
If latency is ≤ 100 ms: the phase-shift approach is viable across the full <200 Hz band.  
If latency is > 150 ms: the 150–200 Hz edge is marginal; proceed to Step 2 first.

---

### Step 2 — Context window sweep (1 day, no training required)

**What to change:** In `~/ANC_Hardware_Test/simulation/model_runtime.py`, find and change the constant:
```python
_CRN_CTX_N = 47616   # current: 3s context
```

**Try in this order:**

| Value | Context | STFT frames | LFBypass coverage | Try first? |
|---|---|---|---|---|
| 47,616 | 2,976 ms | 375 | Full | Baseline (measured: +25.26 dB LF) |
| **8,000** | **500 ms** | **65** | **Full (65 >> 29 needed)** | **← START HERE** |
| 16,000 | 1,000 ms | 128 | Full | If 500ms degrades |
| 3,200 | 200 ms | 28 | At edge | For curiosity only |

**What to measure:**
```bash
cd ~/AI_builders
source .venv/Scripts/activate   # on Windows — or anc_env on Jetson
python test_fusion_pipeline_crn_streaming.py \
    --mix noisy_speech.wav \
    --tcn_ckpt ANC_Hardware_Test/tcn_predictor_ft.pt \
    --gate_db -28 \
    --smooth_alpha 0.4 \
    --out_dir fusion_outputs_sweep_500ms
```

Record: LF cancellation dB (same metric used to track +25.26 dB) and RMS residual.  
**Accept if within 1–2 dB of the 3 s baseline.**

**Why 500 ms is the theoretically motivated first try:**
- LFBypass RF = 232 ms → fully saturated at 65 frames (500 ms)
- BiLSTM at 65 frames still sees ~10+ periods of 50 Hz (which needs only ~200 ms)
- GPU inference drops to ~3–10 ms → digital overhead (~65 ms) becomes the dominant latency

---

### Step 3 — One-time hardware delay calibration

Run once per physical deployment site. Requires the hardware setup to be in its final physical configuration.

**What you need:**
- A test speaker or way to play audio near the reference mic position
- Both mics recording simultaneously
- A <200 Hz chirp signal

**Procedure:**
1. Play `calibration_chirp.wav` (see Appendix A, generate once) from near the reference mic position.
2. Record simultaneously: `ref_mic_recording.wav` (reference mic) and `error_mic_recording.wav` (error mic), **with the ANC system running** (emitting anti-noise).
3. Run `measure_system_delay()` (see Appendix A) → get `total_delay_samples`.
4. Run `compute_phase_shift_samples()` (see Appendix A) with the measured HVAC fundamental frequency (50 or 60 Hz depending on local mains) → get `shift_samples`.
5. Apply `shift_samples` as a fixed sample offset in the output buffer.

**One open question:** What is the HVAC fundamental at this deployment site? Measure it by recording a few seconds of noise and computing the FFT. The dominant peak below 100 Hz is f₀.

---

### Step 4 — TensorRT export (optional polish, ~2–3 days)

Primary purpose: reduce timing jitter from ±2 ms to ±0.1–0.2 ms to stabilize the 150–200 Hz band.  
Secondary: further latency reduction.

**Only needed if:**
- GPU latency without TRT is >100 ms, OR
- Measured cancellation at 150–200 Hz is visibly worse than at 50–100 Hz after calibration

```bash
# On Jetson (after GPU is working):
python3 -c "
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path.home() / 'AI_builders/Run/CRN'))
from inference import load_model
ckpt = Path.home() / 'AI_builders/Run/CRN/checkpoints_crn2/best.pt'
model, _ = load_model(ckpt, torch.device('cuda'))
model.eval()
dummy = torch.zeros(1, 47872, device='cuda')  # or 8000 if using 500ms context
torch.onnx.export(model, dummy, 'crn_bilstm.onnx',
                  input_names=['waveform'], output_names=['enhanced'],
                  dynamic_axes={'waveform': {1: 'time'}, 'enhanced': {1: 'time'}},
                  opset_version=17)
print('ONNX exported')
"
# Then: trtexec --onnx=crn_bilstm.onnx --saveEngine=crn_bilstm.trt --int8 --calib...
```

Full TensorRT calibration procedure: bring this up in the next session.

---

## 6. CONSTRAINTS — DO NOT VIOLATE

| Constraint | Reason |
|---|---|
| Do NOT touch `predictor.py` or any `tcn_*.pt` file | TCN is separately owned; changes not sanctioned |
| Do NOT switch to GRU | Quality was measured on real recordings and rejected |
| Do NOT retrain or change model architecture | Quality constraint requires keeping current BiLSTM weights |
| Keep `model_runtime.py.bak` intact | Rollback point if any edit breaks something |
| Do NOT change `CHUNK_SIZE`, `_CRN_CTX_N` context, or sample rate | Context sweep is a deliberate experiment, not a permanent change until quality is confirmed |

---

## 7. OPEN QUESTIONS FOR TOMORROW

1. **What is the GPU latency?** (measure in Step 1 — the gating question for everything else)
2. **Does 500 ms context preserve LF cancellation within 1–2 dB?** (Step 2)
3. **What is the HVAC fundamental frequency at the deployment site?** (needed for phase correction)
4. **What is the reference-to-error mic physical distance?** (sets the acoustic advance budget, needed to compute net delay for calibration)
5. **Does the cuDNN fix via `torch/lib` path ordering work, or is the system cuDNN hard-linked via `/etc/ld.so.conf.d/`?** (if the latter, alternative approach needed)

---

## Appendix A — Calibration Code

Save as `~/ANC_Hardware_Test/simulation/calibrate_delay.py` or add to the pipeline.

```python
"""
ANC Feedforward Delay Calibration
==================================
One-time calibration per deployment site.

Usage:
    python3 calibrate_delay.py --ref ref_mic_recording.wav --err error_mic_recording.wav
    python3 calibrate_delay.py --generate-chirp  # write calibration_chirp.wav
"""
import argparse
import numpy as np
from pathlib import Path
from scipy.signal import chirp, correlate, butter, sosfilt

SAMPLE_RATE = 16_000


def generate_calibration_chirp(duration_s: float = 2.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Swept-sine 10–200 Hz, length duration_s seconds."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    sig = chirp(t, f0=10, f1=200, t1=duration_s, method='linear').astype(np.float32)
    # Fade in/out to avoid clicks
    fade = int(0.05 * sr)
    sig[:fade]  *= np.linspace(0, 1, fade)
    sig[-fade:] *= np.linspace(1, 0, fade)
    return sig


def measure_system_delay(ref_signal: np.ndarray,
                         error_signal: np.ndarray,
                         sr: int = SAMPLE_RATE) -> tuple[int, float]:
    """
    Cross-correlate reference mic signal with error mic signal (both recorded
    while system is running) to find total system latency.

    Parameters
    ----------
    ref_signal   : what was recorded at the REFERENCE mic (system input)
    error_signal : what was recorded at the ERROR mic during the calibration
                   play-back (includes both incoming noise and anti-noise)
    sr           : sample rate

    Returns
    -------
    delay_samples : integer sample lag (positive = anti-noise arrives LATE)
    delay_s       : same value in seconds
    """
    # Low-pass to <200 Hz for clean LF correlation
    sos = butter(4, 200 / (sr / 2), 'low', output='sos')
    ref_lp   = sosfilt(sos, ref_signal.astype(np.float64))
    error_lp = sosfilt(sos, error_signal.astype(np.float64))

    corr = correlate(error_lp, ref_lp, mode='full')
    lags = np.arange(-(len(ref_lp) - 1), len(error_lp))
    delay_samples = int(lags[np.argmax(np.abs(corr))])
    return delay_samples, delay_samples / sr


def compute_phase_shift_samples(delay_samples: int,
                                f0_hz: float,
                                sr: int = SAMPLE_RATE) -> int:
    """
    Given a measured total system delay, compute the additional integer
    sample offset needed in the output buffer so the anti-noise arrives
    in anti-phase with the noise at the error mic.

    For HVAC noise (fundamental f0 + harmonics): correcting f0 automatically
    aligns all harmonics because they are integer multiples.

    Parameters
    ----------
    delay_samples : output of measure_system_delay()
    f0_hz         : HVAC fundamental (50 Hz or 60 Hz depending on site mains)
    sr            : sample rate

    Returns
    -------
    correction_samples : positive = advance the anti-noise earlier in time
                         negative = add more delay
    """
    period_samples = sr / f0_hz              # e.g. 320.0 for 50 Hz at 16 kHz
    # Current phase at error mic (as fraction of one period, 0–1):
    fractional = (delay_samples % period_samples) / period_samples
    current_phase_rad = fractional * 2 * np.pi
    # Target: π (anti-phase). Correction needed:
    correction_rad = np.pi - current_phase_rad
    correction_samples = int(round(correction_rad / (2 * np.pi) * period_samples))
    return correction_samples


def estimate_hvac_fundamental(noise_wav: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Given a short recording of background noise (a few seconds), estimate
    the HVAC fundamental by finding the dominant peak below 100 Hz.

    Returns dominant frequency in Hz.
    """
    # Use a Hann-windowed FFT
    win = np.hanning(len(noise_wav))
    spectrum = np.abs(np.fft.rfft(noise_wav * win))
    freqs    = np.fft.rfftfreq(len(noise_wav), d=1.0 / sr)
    # Search only 10–100 Hz
    mask = (freqs >= 10) & (freqs <= 100)
    peak_idx = np.argmax(spectrum[mask])
    return float(freqs[mask][peak_idx])


if __name__ == "__main__":
    import soundfile as sf

    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-chirp", action="store_true")
    parser.add_argument("--ref",  help="Reference mic WAV recording")
    parser.add_argument("--err",  help="Error mic WAV recording")
    parser.add_argument("--f0",   type=float, default=50.0,
                        help="HVAC fundamental Hz (default 50)")
    args = parser.parse_args()

    if args.generate_chirp:
        sig = generate_calibration_chirp()
        sf.write("calibration_chirp.wav", sig, SAMPLE_RATE)
        print(f"Written calibration_chirp.wav ({len(sig)/SAMPLE_RATE:.1f}s, 10-200Hz sweep)")

    elif args.ref and args.err:
        ref,  sr1 = sf.read(args.ref,  dtype='float32', always_2d=False)
        err,  sr2 = sf.read(args.err,  dtype='float32', always_2d=False)
        assert sr1 == sr2 == SAMPLE_RATE, "Both WAVs must be 16 kHz"

        delay_smp, delay_s = measure_system_delay(ref, err)
        shift = compute_phase_shift_samples(delay_smp, args.f0)

        print(f"Total system delay : {delay_smp} samples = {delay_s*1000:.1f} ms")
        print(f"HVAC fundamental   : {args.f0} Hz")
        print(f"Phase correction   : {shift:+d} samples = {shift/SAMPLE_RATE*1000:.2f} ms")
        print()
        print("Apply by inserting a circular buffer of abs(shift) samples between")
        print("TCN output and DAC output. Positive = advance (emit earlier).")
        print()
        print(f"# In file_simulator.py or main_hardware.py, set:")
        print(f"PHASE_CORRECTION_SAMPLES = {shift}")
```

---

## Appendix B — Latency Budget Reference

| Component | Value | Notes |
|---|---|---|
| ADC + I2S DMA | 2–4 ms | Hardware fixed |
| Input queue | 16–32 ms | 1–2 chunks typical |
| CRN inference (CPU, 3 s ctx) | 3,225 ms | Measured — unusable |
| CRN inference (GPU est., 3 s ctx) | 20–60 ms | Target for Step 1 |
| CRN inference (GPU est., 500 ms ctx) | 3–10 ms | Target for Step 2 |
| Look-ahead extraction offset | 16 ms | Baked into `_infer_crn_stateful` |
| LPF + TCN | 2–5 ms | Lightweight, not the bottleneck |
| Output queue + DAC | 16–32 ms | Symmetric with input |
| Speaker → error mic | 0.87 ms | 30 cm / 343 m/s |
| **Total (GPU, 3 s ctx)** | **~85–130 ms** | Phase correction viable at 50–150 Hz |
| **Total (GPU, 500 ms ctx)** | **~56–80 ms** | Phase correction viable full <200 Hz |

Phase error accumulation for ±0.1 Hz frequency drift:

| Total delay | 200 Hz | 100 Hz | 50 Hz |
|---|---|---|---|
| 3,225 ms (CPU) | ±116° — broken | ±58° — marginal | ±11° — OK |
| 100 ms (GPU) | ±3.6° — excellent | ±1.8° | ±0.9° |
| 70 ms (GPU+sweep) | ±2.5° | ±1.3° | ±0.6° |

---

## Appendix C — File Locations Quick Reference

```
~/ANC_Hardware_Test/
├── simulation/
│   ├── model_runtime.py            ← _AI_DIR patched; _CRN_CTX_N to change for sweep
│   ├── model_runtime.py.bak        ← original backup — keep
│   ├── benchmark_crn.py            ← latency benchmark script
│   ├── file_simulator.py           ← main streaming simulator
│   └── configs/config.py           ← CHUNK_SIZE=256, SAMPLE_RATE=16000
├── diagnose_cudnn.sh               ← run first to identify libcudnn conflict
└── fix_cudnn.sh                    ← run second to fix LD_LIBRARY_PATH ordering

~/AI_builders/
├── Run/CRN/
│   ├── model.py                    ← CRN architecture (BiLSTM, LFBypass)
│   ├── inference.py                ← load_model()
│   ├── checkpoints/best.pt         ← default (15.11 dB valSNR, NOT LF-optimized)
│   └── checkpoints_crn2/best.pt    ← USE THIS (13.55 dB but lf_noise_ratio=0.8)
└── ANC_Hardware_Test/
    ├── tcn_predictor_ft.pt         ← fine-tuned TCN — DO NOT TOUCH
    └── tcn_predictor.pt            ← old TCN — DO NOT TOUCH

~/anc_env/bin/activate              ← has ANC_LDPATH_BEGIN block (openblas fix)
~/diagnose_cudnn.sh
~/fix_cudnn.sh
~/fix_jetson_anc.sh                 ← already ran; created the path fix + .bak
```

---

*End of handoff. If picking this up in a fresh session: start at Section 4 (Current Blocker) and work through Section 5 in order.*
