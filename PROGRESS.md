# Speech Denoiser — Progress Log

## Day 1 — 2026-06-04 ✓ DONE

### What works
Local Gradio app: upload or record noisy WAV → CRN denoises and enhances → hear Original vs Denoised side-by-side in browser.
- Runs on GPU (CUDA), ~5.5s for 30s audio
- Handles any input sample rate (resampled to 16kHz) and stereo (converted to mono)
- 30s input cap
- RMS normalized to 0.15, peak guarded at 0.99

### Key files created today
| File | Purpose |
|------|---------|
| `denoise_core.py` | Denoiser engine — loads model once, exposes `process(in_path) -> out_path` |
| `app.py` | Gradio UI — upload/mic input, Denoise button, Original + Denoised outputs |
| `outputs_tmp/` | Temp dir for denoised WAV files |

### Checkpoint
`Run/CRN/checkpoints/best.pt` — 190MB, BiLSTM, trained 2026-05-10.
**Needs Git LFS for HuggingFace deploy (Day 3).**
Do NOT use `checkpoints_crn2/best.pt` — over-suppresses speech.

### Tuning knobs (Day 2)
In `denoise_core.py`:
- `_DRY_MIX = 0.20` — fraction of original blended back. Lower = cleaner output; higher = safer for speech (less over-suppression). Try 0.10–0.30.
- `_TARGET_RMS = 0.15` — output loudness target.
- `_suppress_musical_noise()` exists in `crn_pipeline.py` but is NOT currently called. Add after `_enhance_chunked` if tonal warble is heard.

### Listen-test result (2026-06-04)
~90% clean on `noisy_speech.wav` (44.1kHz stereo, fan/AC noise). No clipping. Occasional slight speech over-suppression on softer frames — dry mix helps, may need tuning.

---

## Day 2 — NOT STARTED
- Before/after spectrogram display in Gradio
- Dry-mix tuning based on listen tests
- (optional) `_suppress_musical_noise` toggle

## Day 3 — NOT STARTED
- Deploy to HuggingFace Spaces (CPU-only, free tier)
- Git LFS for 190MB checkpoint
- Confirm CPU inference time is acceptable
