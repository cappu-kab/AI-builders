"""
finetune_tcn.py
===============
Fine-tune the existing LightweightTCN checkpoint on the LF noise profile
extracted from the actual target recording (noisy_speech.wav).

Why fine-tune vs retrain from scratch
--------------------------------------
The base checkpoint was trained on AC_noise.wav — a controlled recording of
HVAC noise.  noisy_speech.wav has a real-room LF noise floor that may differ
in spectral shape, amplitude envelope, and harmonic content.  Fine-tuning
adapts the TCN's weights to this specific noise while keeping the temporal
structure it already learned.

Architecture stays identical → latency on Jetson Nano is unchanged.

Inputs
------
  base checkpoint : ANC_Hardware_Test/tcn_predictor.pt   (200 KB, the "good" arch)
  training signal : fusion_outputs_crn_streaming/noisy_speech/3b_extracted_noise_lpf.wav
                    (the Cheby II 180 Hz LPF-filtered CRN residual — exactly what
                    the TCN sees at inference time)

Output
------
  ANC_Hardware_Test/tcn_predictor_ft.pt   (same size, fine-tuned weights)

Run
---
  cd C:\\Users\\rocha\\AI_builders
  .venv\\Scripts\\python.exe finetune_tcn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "ANC_Hardware_Test"))

from predictor import (          # noqa: E402
    LightweightTCN,
    make_training_pairs,
    train_tcn,
    SAMPLE_RATE,
    INPUT_LEN,
    PREDICT_LEN,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LF_NOISE_PATH = (SCRIPT_DIR / "fusion_outputs_crn_streaming"
                 / "noisy_speech" / "3b_extracted_noise_lpf.wav")
CKPT_IN       = SCRIPT_DIR / "ANC_Hardware_Test" / "tcn_predictor.pt"
CKPT_OUT      = SCRIPT_DIR / "ANC_Hardware_Test" / "tcn_predictor_ft.pt"


def load_wav(path: Path) -> np.ndarray:
    import soundfile as sf
    import librosa
    try:
        x, sr = sf.read(str(path), dtype="float32")
    except Exception:
        x, sr = librosa.load(str(path), sr=None, mono=True)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE)
    return x.astype(np.float32)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # -- 1. Load the extracted LF noise ----------------------------------------
    print(f"\n[1/4] Loading LF noise from:\n      {LF_NOISE_PATH}")
    signal = load_wav(LF_NOISE_PATH)
    dur = len(signal) / SAMPLE_RATE
    print(f"      {len(signal)} smp  ({dur:.1f}s @ {SAMPLE_RATE} Hz)")

    # Hold out the last 15% as a test set for correlation reporting.
    split = int(len(signal) * 0.85)
    train_sig = signal[:split]
    test_sig  = signal[split:]
    print(f"      train: {len(train_sig)} smp ({len(train_sig)/SAMPLE_RATE:.1f}s)   "
          f"test: {len(test_sig)} smp ({len(test_sig)/SAMPLE_RATE:.1f}s)")

    # -- 2. Build training pairs ------------------------------------------------
    # Smaller hop (32 vs 64) = more pairs with higher overlap.
    # Good for fine-tuning on a short clip — more gradient steps per epoch,
    # each one sees highly correlated windows, which acts like data augmentation
    # rather than independent samples.  With ~43s total the model won't overfit
    # because the TCN only has ~5k parameters.
    print("\n[2/4] Building sliding-window training pairs  (hop=32) ...")
    X, Y = make_training_pairs(train_sig, hop=32,
                               chunk_size=INPUT_LEN, predict_size=PREDICT_LEN)
    print(f"      X={X.shape}   Y={Y.shape}")

    # -- 3. Load base checkpoint and fine-tune ----------------------------------
    print(f"\n[3/4] Loading base checkpoint: {CKPT_IN}")
    model = LightweightTCN().to(device)
    state = torch.load(str(CKPT_IN), map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    print(f"      z-score (before): x_mean={float(model.x_mean):+.5f}  "
          f"x_std={float(model.x_std):.5f}  "
          f"y_mean={float(model.y_mean):+.5f}  "
          f"y_std={float(model.y_std):.5f}")

    # Fine-tune: refit z-score stats to the new noise distribution, then train
    # with a 5× lower LR and the same composite loss (MSE + slope + curvature).
    # 60 epochs is enough to adapt without erasing the base weight structure.
    history = train_tcn(
        model, X, Y,
        epochs=60,
        batch_size=64,
        lr=1e-4,
        fit_stats=True,   # refit z-score to the real recording's noise level
        verbose=True,
    )
    print(f"      final loss: {history[-1]:.6f}")
    print(f"      z-score (after):  x_mean={float(model.x_mean):+.5f}  "
          f"x_std={float(model.x_std):.5f}  "
          f"y_mean={float(model.y_mean):+.5f}  "
          f"y_std={float(model.y_std):.5f}")

    # -- 4. Quick correlation check on the held-out test segment ---------------
    from predictor import predict_tcn, StreamingRingBuffer

    print("\n[4/4] Correlation check on held-out test segment ...")
    if len(test_sig) < INPUT_LEN + PREDICT_LEN:
        print("      test segment too short for evaluation — skipping.")
    else:
        ring = StreamingRingBuffer(INPUT_LEN)
        ring.push(test_sig[:INPUT_LEN])

        hop_eval = PREDICT_LEN // 2
        n_preds  = (len(test_sig) - INPUT_LEN - PREDICT_LEN) // hop_eval + 1
        corrs    = np.empty(n_preds, dtype=np.float32)

        win = np.hanning(PREDICT_LEN).astype(np.float32)
        out_len = (n_preds - 1) * hop_eval + PREDICT_LEN
        pred_acc   = np.zeros(out_len, dtype=np.float32)
        actual_acc = np.zeros(out_len, dtype=np.float32)
        norm_acc   = np.zeros(out_len, dtype=np.float32)

        for i in range(n_preds):
            t       = INPUT_LEN + i * hop_eval
            snap    = ring.snapshot()
            pred    = predict_tcn(model, snap)
            actual  = test_sig[t : t + PREDICT_LEN]

            slc = slice(i * hop_eval, i * hop_eval + PREDICT_LEN)
            pred_acc  [slc] += pred   * win
            actual_acc[slc] += actual * win
            norm_acc  [slc] += win

            corrs[i] = (float(np.corrcoef(pred, actual)[0, 1])
                        if actual.std() > 1e-9 else 0.0)
            ring.push(actual[:hop_eval])

        norm_acc[norm_acc < 1e-6] = 1.0
        pred_ola   = (pred_acc   / norm_acc).astype(np.float32)
        actual_ola = (actual_acc / norm_acc).astype(np.float32)
        residual   = actual_ola + (-pred_ola)

        rms_noise    = float(np.sqrt(np.mean(actual_ola.astype(np.float64) ** 2) + 1e-20))
        rms_residual = float(np.sqrt(np.mean(residual.astype(np.float64)   ** 2) + 1e-20))
        cancel_db    = 20.0 * np.log10((rms_noise + 1e-12) / (rms_residual + 1e-12))

        print(f"      test corr: mean={corrs.mean():+.3f}  "
              f"median={float(np.median(corrs)):+.3f}  "
              f"p10={float(np.percentile(corrs, 10)):+.3f}  "
              f"p90={float(np.percentile(corrs, 90)):+.3f}")
        print(f"      estimated cancellation on test segment: {cancel_db:+.2f} dB")

    # -- Save ------------------------------------------------------------------
    torch.save(model.state_dict(), CKPT_OUT)
    print(f"\nSaved fine-tuned checkpoint -> {CKPT_OUT}")
    print("\nNext step:")
    print("  python test_fusion_pipeline_crn_streaming.py "
          "--mix noisy_speech.wav "
          "--tcn_ckpt ANC_Hardware_Test\\tcn_predictor_ft.pt "
          "--gate_db -28 "
          "--out_dir fusion_outputs_ft")


if __name__ == "__main__":
    main()
