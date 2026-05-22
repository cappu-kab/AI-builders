# สรุปโปรเจค: Thai Speech Denoising

> **อัปเดตล่าสุด:** พฤษภาคม 2026  
> **เป้าหมาย:** กำจัดเสียงรบกวน (โดยเฉพาะความถี่ต่ำ เช่น HVAC, พัดลม, เสียงฮัม) ออกจากเสียงพูดภาษาไทย ที่อัตราตัวอย่าง 16 kHz

---

## 1. ภาพรวมโปรเจค

โปรเจคนี้พัฒนาระบบ **Speech Enhancement** สำหรับเสียงพูดภาษาไทย โดยฝึกโมเดล Deep Learning 2 ตัวหลัก ได้แก่ **CRN** (Convolutional Recurrent Network) และ **AdvancedUNetSE** (U-Net with Squeeze-and-Excitation) ทั้งคู่ทำงานบน Complex STFT Spectrogram และใช้ Residual Learning (ทำนายเสียงรบกวน แล้วลบออกจากสัญญาณต้นฉบับ)

โครงสร้างโฟลเดอร์หลัก:

```
AI_builders/Run/
├── preprocess.py          ← แปลงไฟล์เสียงดิบ → .npy
├── CRN/                   ← โมเดล CRN (ขนาดใหญ่, ~15.9M params)
├── speech_denoise/        ← โมเดล U-Net (ขนาดเล็ก, ~2.2M params)
├── benchmark.py           ← เปรียบเทียบโมเดลทั้งหมด
├── finetune_resemble.py   ← Fine-tune Resemble Enhance
└── finetune_mossformer.py ← Fine-tune MossFormer2
```

---

## 2. การเตรียมข้อมูล (Data Preparation)

### 2.1 โครงสร้างไดเรกทอรีข้อมูล

ข้อมูลถูกจัดเก็บในรูปแบบ `.npy` (NumPy float32 array) แยกตาม split:

```
data_npy/
├── train/
│   ├── clean/    (55,817 ไฟล์)
│   └── noise/    ( 8,108 ไฟล์)
├── val/
│   ├── clean/    (15,855 ไฟล์)
│   └── noise/    ( 1,168 ไฟล์)
└── test/
    ├── clean/    (16,438 ไฟล์)
    └── noise/    ( 1,164 ไฟล์)
```

| Split | Clean Clips | Noise Clips | รวม |
|-------|-------------|-------------|------|
| Train | 55,817 | 8,108 | 63,925 |
| Val   | 15,855 | 1,168 | 17,023 |
| Test  | 16,438 | 1,164 | 17,602 |
| **รวมทั้งหมด** | **88,110** | **10,440** | **98,550** |

- รูปแบบ: **Mono float32 @ 16,000 Hz**
- ข้อมูลดิบ (raw audio) เก็บไว้ใน `final_dataset/` แยกตาม train/val/test/speech พร้อม `metadata.csv`

---

### 2.2 กระบวนการ Preprocessing (`preprocess.py`)

ไฟล์เสียงดิบทุกไฟล์ (`.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`) ผ่านกระบวนการดังนี้:

```
ไฟล์เสียงดิบ
    │
    ▼
1. โหลดด้วย soundfile
    │
    ▼
2. Stereo → Mono  (เฉลี่ยทุก channel)
    │
    ▼
3. Resample → 16,000 Hz
   (ใช้ scipy.signal.resample_poly ถ้ามี, ไม่งั้นใช้ linear interpolation)
    │
    ▼
4. ตัดความเงียบ (Silence Trim)
   - threshold: -40 dBFS, frame: 20 ms
    │
    ▼
5. Loudness Normalization → -23 LUFS
   (ใช้ pyloudnorm ถ้ามี, ไม่งั้นใช้ RMS-based)
    │
    ▼
6. ลบ NaN/Inf, clip peak ≤ 0.99
    │
    ▼
7. บันทึกเป็น float32 .npy
```

**การแบ่ง Split:** หากข้อมูลดิบอยู่ในโฟลเดอร์เดียว (flat layout) สคริปต์จะแบ่งอัตโนมัติ **80% / 10% / 10%** (Train / Val / Test) โดยใช้ random seed=42 เพื่อความ reproducible  
**Parallel Processing:** รองรับ multi-process ผ่าน `ProcessPoolExecutor`

---

### 2.3 Data Augmentation ระหว่าง Training

**Dataset ของ CRN (`CRN/dataset.py`)**  
Augmentation ทำแบบ On-the-fly ทุก epoch:

| Augmentation | รายละเอียด | ความน่าจะเป็น |
|---|---|---|
| 🔊 LF Noise (Synthetic) | สร้างเสียง HVAC/พัดลม/ฮัม < 200 Hz (power-law spectrum + 50/60 Hz harmonics) | 70% ของ batch |
| 🔊 Real Noise Pool | ดึงจาก noise pool จริง + color filter | 30% ของ batch |
| 📏 Time Stretch | ยืด/หดสัญญาณ ±15% | p=0.7 |
| 🎵 Pitch Shift | เลื่อน pitch ±2 semitones | p=0.5 |
| 🏠 RIR Convolution | จำลองเสียงสะท้อนในห้อง (256 synthetic RIRs) | p=0.4 |
| 📢 Random Gain | ปรับระดับเสียง ±6 dB | p=1.0 |
| 📡 Soft Clipping | บิดเบือนแบบ tanh เล็กน้อย | p=0.25 |
| 🎛️ Random EQ | ปรับ EQ 2-3 Gaussian bands ±3 dB | p=0.4 |
| 🔉 Bandpass Filter | กรองความถี่ 100–7,500 Hz แบบสุ่ม | p=0.2 |
| 🎭 Frequency Masking | ปิดกั้น 1–24 bins ต่อเนื่อง | p=0.15 |
| 🎚️ SNR Mixing | ผสมสัญญาณที่ SNR: −5 ถึง +15 dB | ทุก sample |

**Curriculum Learning:** ช่วง epoch แรก (10 epoch) ฝึกด้วย SNR สูง (ง่าย) ก่อน แล้วค่อยๆ ลด SNR จนถึงช่วงยาก [-5, +15] dB

**Dataset ของ U-Net (`speech_denoise/dataset.py`)**  
มีโครงสร้างเหมือนกัน แต่ split เสียงรบกวน HF (30%) เป็น 3 archetype:
- **Click noise:** impulse สั้นๆ 1–8 ครั้ง/วินาที
- **Hiss noise:** เน้นความถี่ > 1.5–4 kHz
- **Burst noise:** wideband transient ยาว 10–100 ms

---

## 3. สถาปัตยกรรมโมเดล (Model Architecture)

### 3.1 CRN — Convolutional Recurrent Network

**ไฟล์:** `CRN/model.py` | **ขนาด:** ~15.88M parameters (BiLSTM variant)

```
Noisy Waveform (B, T)
        │
    [ STFT ]  n_fft=512, hop=128
        │
(B, 2, 257, T')   ← Real + Imaginary แยก channel
        │
┌───────────────────────┐
│   5× Encoder Blocks   │  Stride-2 (freq), channels: 2→32→64→128→256→512
│   Conv2d(5×3) + BN    │  ทุก block มี SE (Squeeze-and-Excitation) attention
│   + ELU + SEBlock     │
└──────────┬────────────┘
           │  skips บันทึกทุก level สำหรับ U-Net connections
           │
    [ Bottleneck SE ]   ← Global channel attention (B, 512, 9, T) ก่อนส่งเข้า RNN
           │
    [ proj_in: 4608 → 512 ]
           │
    [ BiLSTM: hidden=256, 2 layers, bidirectional ]  output dim=512
           │
    [ proj_out: 512 → 4608 ]
           │
┌───────────────────────┐
│   5× Decoder Blocks   │  ConvTranspose2d + skip-concat จาก Encoder
└──────────┬────────────┘
           │
    [ LF-Bypass Branch ]  ← เส้นทางตรงสำหรับ 16 bins ต่ำสุด (< ~500 Hz)
           │                 เพื่อแก้ปัญหา LF หาย ที่ bottleneck 9-bin
           │
    [ Magnitude-Bounded Mask ]  tanh × 1.2
           │
    noise_spec = mask ⊗ noisy_spec
    enhanced  = noisy − iSTFT(noise_spec)
```

**จุดเด่น:**
- ใช้ **Complex STFT masking** (real + imaginary แยกกัน)
- **Bottleneck SE** ใหม่: ช่วย weight channel ที่เกี่ยวกับ LF vs HF ก่อนส่งเข้า BiLSTM
- **LF-Bypass** แก้ปัญหา gradient หายสำหรับความถี่ < 200 Hz ที่ถูก compress เหลือ 9 bins
- **Residual Learning:** ทำนายเสียงรบกวน (ไม่ใช่เสียงสะอาด) ทำให้ initial output ≈ noisy input (เสถียรกว่า)

---

### 3.2 AdvancedUNetSE — U-Net with SE Attention

**ไฟล์:** `speech_denoise/model_unet_advanced.py` | **ขนาด:** ~2.15M parameters

```
Noisy Waveform (B, T)
        │
    [ STFT ]  n_fft=512, hop=128
        │
(B, 2, 257, T')
        │
┌────────────────────────────┐
│  5× Encoder Stages         │  ResidualConvBlock (Depthwise-Separable Conv)
│  (บันทึก skip ก่อน stride) │  + stride-2 downsampling
│  GroupNorm + ELU           │  ประหยัด params ~13-14×
└──────────┬─────────────────┘
           │  skips ถูกบันทึก ก่อน downsampling (full resolution)
           │
┌───────────────────────────────────────┐
│  DilatedBottleneck                    │
│  Dilations: 1,2,4,8 → 1,2,4,8,16     │  Receptive field กว้าง
│  + SE Attention (Squeeze-Excitation)  │
└──────────┬────────────────────────────┘
           │
┌────────────────────────────┐
│  5× Decoder Stages         │  ConvTranspose2d + GatedSkip
│  GatedSkip: sigmoid gate   │  gate เรียนรู้ว่าให้ skip ผ่านมากแค่ไหน
│  2 stages สุดท้าย: LF-     │  lf_alpha scalar (init=0) สำหรับ
│  direct path               │  16 bins ต่ำสุด ไม่ผ่าน sigmoid
└──────────┬─────────────────┘
           │
    [ LF-Bypass Branch ]  temporal-only dilated convs (ไม่มี freq downsampling)
           │
    [ Mask Head ]  → (B, 2, 257, T')
           │
    [ Magnitude-Bounded Mask ]  tanh × mask_bound
           │
    enhanced = noisy − iSTFT(mask ⊗ noisy_spec)
```

**จุดเด่น:**
- **Depthwise-Separable Convolution** ใน ResidualConvBlock: ประหยัด parameters มาก
- **GatedSkip Connections:** gate เรียนรู้ว่า skip feature สำคัญแค่ไหนสำหรับแต่ละ level
- **LF-direct path** ใน 2 decoder stages สุดท้าย: ช่วยให้ LF bins ไม่ถูก gate บล็อก
- **DilatedBottleneck** ขยาย temporal receptive field โดยไม่เพิ่ม params มากนัก
- เหมาะสำหรับ Training บน **Google Colab** (ขนาดเล็ก, ใช้ VRAM น้อย)

---

## 4. Loss Functions

ทั้ง 2 โมเดลใช้ Loss หลายตัวรวมกัน:

| Loss Term | สูตร | หน้าที่ |
|---|---|---|
| **Huber L1** (w=1.0) | Huber(enhanced, clean, δ=0.5) | คุณภาพเสียงพูดโดยรวม |
| **Noise Huber** (w=1.0) | Huber(noise\_pred, noise\_target) | ความแม่นยำการกำจัดเสียงรบกวน |
| **Multi-Res STFT** | SC + LogMag L1 × 3 resolutions (256/512/1024 FFT) | โครงสร้าง Spectrogram |
| **LF Boost** (8×) | เพิ่ม weight ที่ bin < 200 Hz | บังคับให้เน้น LF noise |
| **SI-SDR** (w=0.1) | −tanh(SI-SDR/20) | Phase-aware quality |
| **Formant Band** (dynamic) | STFT L1 ที่ 200–4000 Hz | ป้องกัน speech distortion |
| **Speech Floor** (w=0.2) | relu(E[clean²] − E[enhanced²]) | ป้องกัน over-suppression |
| **Silence Penalty** (w=0.1) | \|enhanced\| ในช่วง silence | กำจัด residual noise ในช่วงหยุด |
| **Perceptual** (w=0.05) | Wav2Vec2 CNN feature L1 | Phonetic feature matching |
| **Anti-collapse** (w=0.03) | −mean\|noise\_pred\| | ป้องกัน mask collapse เป็น 0 |

**Dynamic Formant Weight:** น้ำหนัก formant loss เริ่มต้นต่ำ (0.05) แล้วค่อยๆ เพิ่มขึ้นถึง 0.25 ใน epoch สุดท้าย เพื่อให้ LF boost ครอบงำช่วงต้นโดยไม่ขัดแย้งกัน

---

## 5. Optimizer และ Scheduler

| ตัวแปร | ค่าที่ใช้ |
|---|---|
| Optimizer | **AdamW** (lr=2×10⁻⁴, weight_decay=1×10⁻⁴, β=(0.9, 0.999)) |
| Scheduler | **CosineAnnealingWarmRestarts** (T₀=15 epochs) — restart ทุก 15 epoch เพื่อหลบ local minima |
| Warmup | 500 steps (linear warmup ก่อน main schedule) |
| AMP | bf16 (Bfloat16) ป้องกัน overflow บน LF signals |
| Grad Clip | 1.0 |
| Early Stopping | patience=20 epochs |

---

## 6. สถานะโปรเจคปัจจุบัน

### โมเดลที่ฝึกแล้ว

| โมเดล | สถานะ | Checkpoint | หมายเหตุ |
|---|---|---|---|
| **CRN** | 🔄 กำลังฝึกรอบใหม่ | `CRN/checkpoints/best.pt` (190.6 MB) | Plateau ที่ epoch 24–40 → อัปเกรดใหม่ |
| **AdvancedUNetSE** | ✅ ฝึกแล้ว | `speech_denoise/checkpoints/best.pt` (25.9 MB) | พร้อม deploy Colab |

### การอัปเกรด CRN รอบล่าสุด

เพื่อทลาย Plateau ที่ epoch 24 ได้ทำการอัปเกรด:

1. **เพิ่ม Channel Width** — base_channels `16 → 32` (encoder channels: [2,16,...,256] → [2,32,...,512])
2. **เพิ่ม Bottleneck SE** — Global channel attention ก่อน BiLSTM ช่วยแยก LF vs HF
3. **เปลี่ยน Scheduler** — CosineAnnealingWarmRestarts (T₀=15) แทน Plateau แบบเดิม
4. **Dynamic Formant Weight** — ramp 0.05 → 0.25 ป้องกัน LF boost ขัด formant loss
5. **เพิ่ม bottleneck_dim** — 384 → 512 ลด information loss ก่อน BiLSTM

**คำสั่งเริ่ม Training CRN ใหม่จาก Epoch 1:**
```bash
python train.py --data_root ../../data_npy --ckpt_dir ./checkpoints \
  --epochs 60 --batch_size 16 --num_workers 4 --segment_seconds 3.0 \
  --amp_dtype bf16 --scheduler warmrestarts --cosine_T0 15 \
  --lr 2e-4 --warmup_steps 500 --steps_per_epoch 180 \
  --curriculum_epochs 10 --low_freq_boost 8.0 --lf_noise_ratio 0.70 \
  --w_formant 0.05 --w_formant_final 0.25 --w_noise_stft 1.0 \
  --patience 20 --debug_every 5 \
  --base_channels 32 --bottleneck_dim 512 \
  --rnn_hidden 256 --rnn_layers 2 --rnn_type bilstm
```

### U-Net: พร้อม Colab Deployment

ไฟล์ใน `Colab_Ready_UNet/`:
- `model_unet_advanced.py`, `train.py`, `dataset.py`, `losses.py`, `evaluate.py`, `inference.py`
- **`Unet_Colab_Master.ipynb`** — Notebook 8 เซลล์: mount Drive, install deps, unzip data, auto-resume, train, evaluate, inference demo

### โมเดลอื่น (พร้อมใช้งาน)

| โมเดล | Script | สถานะ |
|---|---|---|
| **Resemble Enhance** | `finetune_resemble.py` | ✅ สคริปต์พร้อม |
| **MossFormer2** | `finetune_mossformer.py` | ✅ สคริปต์พร้อม |
| **Benchmark** | `benchmark.py` | ✅ เปรียบเทียบทุกโมเดลได้ |

---

## 7. ไฟล์สำคัญในโปรเจค

```
Run/
├── preprocess.py              ← แปลง raw audio → data_npy/
│
├── CRN/
│   ├── model.py               ← CRN + LFBypass + BottleneckSE (อัปเกรดล่าสุด)
│   ├── dataset.py             ← Augmentation on-the-fly, 70/30 LF/HF noise
│   ├── losses.py              ← Multi-res STFT + FormantBand + SI-SDR + ...
│   ├── train.py               ← AdamW + WarmRestarts + Dynamic formant weight
│   ├── evaluate.py            ← PESQ, STOI, SNR, Prefix WER (ภาษาไทย)
│   └── inference.py           ← รัน inference บนไฟล์เดี่ยว
│
└── speech_denoise/
    ├── model_unet_advanced.py ← AdvancedUNetSE + GatedSkip + LFBypass
    ├── dataset.py             ← 70% LF, 30% HF (click/hiss/burst)
    ├── losses.py              ← เหมือน CRN
    ├── train.py               ← รองรับ cosine/warmrestarts/plateau
    ├── evaluate.py            ← Prefix WER สำหรับ 4-second segments
    └── benchmark.py           ← เปรียบเทียบ U-Net vs CRN vs Resemble vs MossFormer2
```

---

*เอกสารนี้สร้างจากการวิเคราะห์โค้ดจริงใน `AI_builders/Run/` ทุกไฟล์*
