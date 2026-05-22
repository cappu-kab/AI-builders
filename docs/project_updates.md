# Project Updates — Thai Speech Denoising Benchmark

_Date: 2026-05-10 | Goal: Push SI-SDR above 9.0 dB_

---

## 1. Smart Data Mining & Real Multi-Noise Mixing (`dataset.py`)

### Real Noise Pool Categorization
Both `Run/CRN/dataset.py` and `Colab_Ready_UNet/dataset.py` now scan the loaded
noise pool at `__init__` time and split it into two sub-pools:

| Pool | Criterion | Typical content |
|---|---|---|
| `real_lf_pool` | > 40% energy below 200 Hz | HVAC, fan hum, rumble files |
| `general_noise_pool` | ≤ 40% energy below 200 Hz | traffic, crowd, wideband noise |

If either pool is empty after categorisation the other pool is shared as a fallback.

### 2-Noise Mixing Logic
Every training sample is now built from **2 simultaneous real noise sources** (1 LF + 1 general)
instead of the previous 70/30 synthetic/real split:

```python
lf_noise  = self._grab_lf_noise(segment_len)   # from real_lf_pool + synthetic LF blend
gen_noise = self._grab_hf_noise(segment_len)   # from general_noise_pool + HF archetype blend
lf_w      = random.uniform(0.45, 0.75)
noise     = lf_w * lf_noise + (1 - lf_w) * gen_noise  # combined, then normalised
```

SNR range remains −5 to +15 dB; existing `sample_snr()` hard-bias toward −5…+5 dB is preserved.

### Math Overflow Fix
All sigmoid activations that convert frequency arrays (up to 8000 Hz) now clip the
exponent argument to avoid `RuntimeWarning: overflow encountered in exp`:

```python
# Before (overflows at high freq bins)
lp = 1.0 / (1.0 + np.exp((freqs - 200.0) / 25.0))

# After — safe
lp = 1.0 / (1.0 + np.exp(np.clip((freqs - 200.0) / 25.0, -500.0, 500.0)))
```

Fixed in: `generate_lf_noise`, `lowpass_filter`, `highpass_filter_signal` (all in both repos).

New synthetic noise generators added to CRN (previously only in Colab):
`generate_click_noise`, `generate_hiss_noise`, `generate_burst_noise`,
`lowpass_filter`, `highpass_filter_signal`.

---

## 2. Metric-Driven Stability & Logging (`train.py` — both repos)

### SI-SDR Weight Increased
`--w_sisnr` default raised **0.1 → 1.0** in both CRN and Colab_Ready_UNet train scripts.
`best.pt` was already saved based on minimum `val_sisnr_loss` (= maximum SI-SDR); the higher
weight makes SI-SDR the dominant gradient signal rather than an auxiliary term.

### Gradient Clipping Tightened
`--grad_clip` default changed **1.0 → 3.0**.  
1.0 was occasionally too tight and starved gradients in deep BiLSTM/U-Net skip layers;
3.0 stabilises training while still preventing gradient spikes under AMP.

### Per-Epoch History Logging
`train.py` (both repos) now appends to two files at each epoch:

| File | Format | Contents |
|---|---|---|
| `checkpoints/history.json` | JSON array | epoch, train_loss, val_loss, val_sisnr_loss, val_sisnr_db |
| `checkpoints/metrics.csv` | CSV | same fields, one row per epoch |

`val_sisnr_db` is derived as `20 × atanh(−val_sisnr_loss)` (exact inverse of the tanh-scaled
loss), giving an approximate SI-SDR in dB that is easy to plot and monitor.

---

## 3. Colab Speed, Visualisation & Audio Demo (`Unet_Colab_Master.ipynb`)

### Speed Settings (Cell 6 — Train)

| Setting | Old | New | Reason |
|---|---|---|---|
| `--epochs` | 60 | **30** | Faster iteration; early stop still available |
| `--num_workers` | 4 | **2** | Avoids Colab IPC overhead |
| `--steps_per_epoch` | 400 | **250** | 250 × 16 = 4 000 clips/epoch (≈ CRN budget) |
| `--w_sisnr` | _(not set)_ | **1.0** | Direct SI-SDR focus |
| `--grad_clip` | _(not set)_ | **3.0** | U-Net skip connection stability |

### New Cell 9 — Training Curves & SI-SDR Progress

Reads `history.json` from the checkpoint folder and produces a side-by-side matplotlib figure:
- **Left panel**: Training Loss vs Validation Loss over epochs
- **Right panel**: Validation SI-SDR (dB) with a dashed target line at 9.0 dB and an annotation for the best epoch

The figure is also saved to `{CKPT_DIR}/training_curves.png`.

### Updated Cell 8 — Audio Demo (2 Samples)

The inference demo now:
1. Processes **2 test files** (index 0 and 1 from `test/clean/`)
2. Synthesises realistic noisy input: LF pink noise (−5 dB SNR) + HF white noise (+10 dB SNR)
3. Displays `IPython.display.Audio` players for **Noisy Input / Enhanced Output / Clean Reference**
   for each sample so you can compare before/after immediately after training

---

## 4. CRN Training — PowerShell One-Liner

```powershell
cd "C:\Users\rocha\AI_builders\Run\CRN"; python train.py `
  --data_root  ..\..\data_npy `
  --ckpt_dir   .\checkpoints `
  --epochs     50 `
  --batch_size 16 `
  --num_workers 4 `
  --steps_per_epoch 180 `
  --w_sisnr    1.0 `
  --grad_clip  3.0 `
  --low_freq_boost 8.0 `
  --lf_noise_ratio 0.70 `
  --scheduler  warmrestarts `
  --cosine_T0  15 `
  --curriculum_epochs 12 `
  --patience   20 `
  --debug_every 5
```

Key flags:
- `--w_sisnr 1.0` — SI-SDR is now the dominant loss signal
- `--grad_clip 3.0` — stabilised BiLSTM gradients
- `--low_freq_boost 8.0` — 8× emphasis on sub-200 Hz bins
- Checkpoints saved to `./checkpoints/` with `best.pt`, `history.json`, `metrics.csv`

---

## Summary of Changed Files

| File | Changes |
|---|---|
| `Run/CRN/dataset.py` | Overflow fix; new LP/HP/click/hiss/burst generators; noise pool categorisation; 2-noise mixing |
| `Run/CRN/train.py` | `--w_sisnr` 0.1→1.0; `--grad_clip` 1.0→3.0; per-epoch `history.json` + `metrics.csv` |
| `Colab_Ready_UNet/dataset.py` | Overflow fix; noise pool categorisation; updated `_get_pool_noise`/`_grab_lf_noise`/`_grab_hf_noise`; 2-noise mixing |
| `Colab_Ready_UNet/train.py` | `--w_sisnr` 0.1→1.0; `--grad_clip` 1.0→3.0; per-epoch history logging |
| `Colab_Ready_UNet/Unet_Colab_Master.ipynb` | Cell 6: speed settings; Cell 8: 2-sample audio demo; Cell 9 (new): training curve plots |
