# AI Builders — Predictive ANC Pipeline

End-to-end Data, Training, and Simulation pipeline for **Predictive Active Noise Cancellation** built around a **CRN denoiser** and a **Causal TCN look-ahead predictor**. Physical hardware deployment scripts are intentionally excluded from this repository while they remain unvalidated.

## Pipeline stages

```
 raw audio ─► data_prep/ ─► .npy dataset ─► models/{crn,unet}/train.py ─► checkpoints
                                                                            │
 finetune/ (Mossformer, Resemble) ◄──────────────────────────────────────────┤
                                                                            ▼
                                          evaluation/ (benchmark, eval)
                                                                            │
                                                    simulation/ (TCN predictor
                                                    + CRN/UNet runtime + file
                                                       simulator + visualizer)
```

## Directory layout

| Folder         | Contents                                                                                     |
|----------------|----------------------------------------------------------------------------------------------|
| `data_prep/`   | Scraping (YouTube), low-frequency-noise filtering, dataset curation, audio → `.npy` convert. |
| `models/crn/`  | CRN architecture, dataset, losses, **train**, evaluate, inference (self-contained).          |
| `models/unet/` | Advanced U-Net SE architecture, train, evaluate, benchmark, Colab notebook.                  |
| `training/`    | `run.py` master orchestrator + Mossformer / Resemble Enhance fine-tuning scripts.            |
| `evaluation/`  | `benchmark.py`, `eval.py` — metrics + result tables.                                          |
| `simulation/`  | Hardware-accurate streaming simulator, Causal TCN predictor, model runtime, visualizer.      |
| `scripts/`     | `export_onnx.py`, `test_fusion_pipeline.py`, `inference.py` — utilities.                     |
| `docs/`        | `PREDICTOR.md`, project summaries, simulation history & README.                              |

## Quick start

```bash
# 1. Preprocess raw audio → .npy
python training/run.py preprocess --in_root ./raw_audio --out_root ./data_npy

# 2. Smoke-test the CRN pipeline (1 epoch, 5 steps)
python training/run.py smoke --project crn

# 3. Full CRN training
python training/run.py train --project crn --data_root ./data_npy

# 4. Evaluate
python training/run.py evaluate --project crn --ckpt ./checkpoints/best.pt

# 5. Run the file-based ANC simulator with the CRN denoiser
python simulation/file_simulator.py dirty.wav -m crn -o cancelled.wav
```

## Models

- **CRN** — Convolutional Recurrent Network, primary low-latency denoiser (~30–60 ms inference on Jetson Orin Nano).
- **Advanced U-Net SE** — second-stage / alternative denoiser used for benchmarking.
- **Causal TCN predictor** (`simulation/predictor.py`) — forecasts the next 5 ms (80 samples) of low-frequency engine / HVAC noise so the cancellation path can compensate for network processing latency. See [`docs/PREDICTOR.md`](docs/PREDICTOR.md).
- **Fine-tuned Mossformer2 / Resemble Enhance** — speech-quality baselines fine-tuned on the curated dataset.

## Excluded from this repository

- Physical-hardware deployment scripts (`main_hardware.py`, I²S device configs) — untested.
- Large model checkpoints (`*.pt`, `*.pth`, `*.onnx`) — distributed via Releases / Drive.
- Raw datasets and audio test artefacts (`*.wav`, `*.mp3`, `*.mp4`, `data_npy/`) — gitignored.

## Requirements

See `models/crn/requirements.txt`, `models/unet/requirements.txt`, and `simulation/requirements.txt` for stage-specific dependencies.

## Project status

This repository covers the **data, training, fine-tuning, evaluation, and simulation** layers of the predictive-ANC stack. Hardware deployment (real-time I²S streaming on Jetson Nano with INMP441 mic + MAX98357A amp) is being validated separately and will be merged once verified.
