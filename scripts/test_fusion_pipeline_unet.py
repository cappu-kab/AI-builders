"""
test_fusion_pipeline_unet.py
============================
Fusion-pipeline simulation: AdvancedUNetSE noise extraction -> Causal TCN forecast.

Drop-in replacement for test_fusion_pipeline.py (CRN version) — but the
front-end denoiser is the time-domain Advanced U-Net SE.  Rationale:

  * The U-Net operates directly in waveform space (B, T) -> (B, T), with
    STFT/iSTFT internal and fp32-forced under AMP.  That avoids the
    chunk-boundary phase-reconstruction artefacts you can get when the CRN's
    iSTFT runs at the edges of a sliding context window.
  * The U-Net's bounded complex-ratio-mask head (|mask| = tanh(|raw|) *
    mask_bound) keeps the noise-prediction energy from over-shooting, so the
    `extracted_noise = mix - clean` residual has a more well-behaved
    amplitude distribution than the CRN's residual.

Pipeline
--------
                              +---------- U-Net --------+
       mix  ->   load_audio   |   clean = mix - noise   |
       (S+N)                  +-------------------------+
                                       |
                       extracted_noise = mix - clean
                                       |
                       low_pass_filter (180 Hz Cheby II, causal sosfilt)
                                       |  <- matches the TCN's training-time
                                       |     input distribution
                          +------- tile 1024-smp context -------+
                          |   TCN predicts next 80 smp at a time |
                          +--------------------------------------+
                                       |
              anti_noise = -prediction,  residual = noise_lpf + anti

Usage
-----
    # Pre-mixed speech+fan recording:
    python test_fusion_pipeline_unet.py --mix path/to/mix.wav

    # Or build the mix from a clean speech file + a fan-noise file:
    python test_fusion_pipeline_unet.py --speech speech.wav --noise fan_noise.wav --snr_db 0

    # Pick a different TCN checkpoint or U-Net checkpoint:
    python test_fusion_pipeline_unet.py --mix mix.wav \
        --tcn_ckpt tcn_predictor.pt \
        --unet_ckpt Run/U_net/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch

# U-Net checkpoints saved on Linux contain PosixPath objects; remap on Windows.
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

SCRIPT_DIR = Path(__file__).resolve().parent
ANC_DIR    = SCRIPT_DIR / "ANC_Hardware_Test"
UNET_DIR   = SCRIPT_DIR / "Run" / "U_net"

# Make the local predictor + U-Net packages importable without polluting cwd.
for p in (ANC_DIR, UNET_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from predictor import (                                        # noqa: E402
    LightweightTCN,
    StreamingRingBuffer,
    predict_tcn,
    low_pass_filter,                # 180 Hz Cheby II causal LPF — TCN training-time front end
    INPUT_LEN,
    PREDICT_LEN,
    SAMPLE_RATE,
)

SR = SAMPLE_RATE   # 16 kHz — single source of truth, matches the TCN's training rate


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    """Load any audio file -> mono float32 numpy array at `sr` Hz."""
    import soundfile as sf
    import librosa

    try:
        wav, sr_in = sf.read(str(path), always_2d=True, dtype="float32")
        wav = wav.mean(axis=1)
    except Exception:
        wav, sr_in = librosa.load(str(path), sr=None, mono=True, dtype=np.float32)

    if sr_in != sr:
        wav = librosa.resample(wav, orig_sr=sr_in, target_sr=sr)
    return wav.astype(np.float32)


def save_wav(wav: np.ndarray, path: Path, sr: int = SR) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(wav, -1.0, 1.0).astype(np.float32), sr)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-20))


def db(num: float, den: float) -> float:
    return 20.0 * np.log10((num + 1e-12) / (den + 1e-12))


# ---------------------------------------------------------------------------
# Mix builder (speech + fan @ target SNR) — only used when --mix is not given
# ---------------------------------------------------------------------------

def build_mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(speech):
        reps  = (len(speech) // len(noise)) + 1
        noise = np.tile(noise, reps)
    noise = noise[: len(speech)]

    p_s = np.mean(speech ** 2) + 1e-12
    p_n = np.mean(noise  ** 2) + 1e-12
    gain = np.sqrt(p_s / p_n) * (10.0 ** (-snr_db / 20.0))
    return np.clip(speech + gain * noise, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# U-Net loader
# ---------------------------------------------------------------------------

def load_unet(ckpt: Path, device: torch.device):
    """Return a function `enhance(noisy: np.ndarray) -> np.ndarray`.

    The U-Net consumes raw waveforms (B, T) and emits enhanced waveforms (B, T)
    — no STFT/iSTFT boundary handling needed on the caller side.
    """
    from inference import load_model, enhance_waveform   # noqa: E402  (local import — U_net's inference.py)
    model, _ = load_model(str(ckpt), device)
    def _enhance(noisy: np.ndarray) -> np.ndarray:
        return enhance_waveform(model, noisy, device, sr=SR)
    return _enhance


# ---------------------------------------------------------------------------
# Streaming TCN over the extracted-noise residual
# ---------------------------------------------------------------------------

def tcn_stream(model: LightweightTCN, noise_track: np.ndarray) -> dict:
    """Tile PREDICT_LEN-sample TCN forecasts back-to-back over `noise_track`.

    Open-loop / mic-wins: the ring buffer is always fed the ACTUAL past, never
    the model's own prediction.  Hop = PREDICT_LEN so consecutive predictions
    tile with no gaps and the exported WAV plays at real-time speed.
    """
    if len(noise_track) < INPUT_LEN + PREDICT_LEN:
        raise ValueError(
            f"Noise track too short for TCN streaming: need >= "
            f"{INPUT_LEN + PREDICT_LEN} smp, got {len(noise_track)}."
        )

    n_preds = (len(noise_track) - INPUT_LEN) // PREDICT_LEN
    out_len = n_preds * PREDICT_LEN

    pred_out   = np.empty(out_len, dtype=np.float32)
    actual_out = np.empty(out_len, dtype=np.float32)

    ring = StreamingRingBuffer(INPUT_LEN)
    ring.push(noise_track[:INPUT_LEN])              # warm-up with true history

    corrs = np.empty(n_preds, dtype=np.float32)
    for i in range(n_preds):
        t        = INPUT_LEN + i * PREDICT_LEN
        snapshot = ring.snapshot()
        pred     = predict_tcn(model, snapshot)
        actual   = noise_track[t : t + PREDICT_LEN]

        slc = slice(i * PREDICT_LEN, (i + 1) * PREDICT_LEN)
        pred_out[slc]   = pred
        actual_out[slc] = actual

        corrs[i] = (np.corrcoef(pred, actual)[0, 1]
                    if actual.std() > 1e-9 else 0.0)

        ring.push(actual)                            # open-loop, mic-wins

    return dict(tcn_pred=pred_out, actual_future=actual_out, corrs=corrs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fusion-pipeline simulation: U-Net noise extraction -> TCN forecast."
    )
    ap.add_argument("--mix",     type=Path, default=None,
                    help="Pre-mixed speech+noise wav. Skip --speech/--noise/--snr_db when set.")
    ap.add_argument("--speech",  type=Path, default=None, help="Clean speech wav.")
    ap.add_argument("--noise",   type=Path, default=SCRIPT_DIR / "fan_noise.wav",
                    help="Noise wav (default: fan_noise.wav).")
    ap.add_argument("--snr_db",  type=float, default=0.0,
                    help="Target speech-to-noise ratio in dB when mixing (default 0).")
    ap.add_argument("--tcn_ckpt", type=Path,
                    default=SCRIPT_DIR / "tcn_predictor.pt",
                    help="Trained LightweightTCN state_dict (default: tcn_predictor.pt).")
    ap.add_argument("--unet_ckpt", type=Path,
                    default=UNET_DIR / "checkpoints" / "best.pt",
                    help="U-Net checkpoint (default: Run/U_net/checkpoints/best.pt).")
    ap.add_argument("--out_dir", type=Path,
                    default=SCRIPT_DIR / "fusion_outputs_unet",
                    help="Output directory (default: ./fusion_outputs_unet/<stem>/).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # -- 1. Build / load the mix ----------------------------------------------
    if args.mix is not None:
        mix  = load_audio(args.mix)
        stem = args.mix.stem
        print(f"\n[mix] loaded {args.mix}  ({len(mix)/SR:.2f}s)")
    else:
        if args.speech is None:
            ap.error("Provide --mix, or --speech together with --noise.")
        speech = load_audio(args.speech)
        noise  = load_audio(args.noise)
        mix    = build_mix(speech, noise, args.snr_db)
        stem   = f"{args.speech.stem}_plus_{args.noise.stem}_{args.snr_db:+.0f}dB"
        print(f"\n[mix] built  speech={args.speech.name}  noise={args.noise.name}  "
              f"SNR={args.snr_db:+.1f} dB  ({len(mix)/SR:.2f}s)")

    out_dir = args.out_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    save_wav(mix, out_dir / "1_mix.wav")

    # -- 2. U-Net -> clean speech, then extract noise as the residual ---------
    print(f"\n[U-Net] loading {args.unet_ckpt}")
    unet_enhance = load_unet(args.unet_ckpt, device)
    clean = unet_enhance(mix).astype(np.float32)
    # Length safety: U-Net keeps T by construction, but be defensive about +/-1 smp.
    n = min(len(mix), len(clean))
    mix, clean = mix[:n], clean[:n]
    extracted_noise = (mix - clean).astype(np.float32)

    save_wav(clean,           out_dir / "2_unet_clean.wav")
    save_wav(extracted_noise, out_dir / "3_unet_extracted_noise.wav")
    print(f"      mix  RMS = {rms(mix):.5f}   "
          f"clean RMS = {rms(clean):.5f}   "
          f"extracted-noise RMS = {rms(extracted_noise):.5f}")

    # -- 2b. Match the TCN's training-time input distribution -----------------
    # The TCN was trained on signals that had passed through a strictly-causal
    # Cheby II 180 Hz low-pass (see predictor.low_pass_filter).  The U-Net
    # residual fed straight in is wideband and carries residual speech
    # leakage — outside the model's training distribution.  Applying the same
    # LPF here puts the input back into the band the TCN was trained to
    # forecast.  This was the single fix that took the CRN fusion from -14.92
    # dB to +10.75 dB; the U-Net residual needs the same treatment.
    tcn_input = low_pass_filter(extracted_noise)
    save_wav(tcn_input, out_dir / "3b_unet_extracted_noise_lpf.wav")
    print(f"      LPF (Cheby II 180 Hz, causal sosfilt)  "
          f"filtered-noise RMS = {rms(tcn_input):.5f}")

    # -- 3. Stream the residual through the TCN -------------------------------
    print(f"\n[TCN] loading {args.tcn_ckpt}")
    tcn = LightweightTCN().to(device)
    state = torch.load(str(args.tcn_ckpt), map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    tcn.load_state_dict(state)
    tcn.eval()
    print(f"      TCN z-score buffers: "
          f"x_mean={float(tcn.x_mean):+.5f}  x_std={float(tcn.x_std):.5f}  "
          f"y_mean={float(tcn.y_mean):+.5f}  y_std={float(tcn.y_std):.5f}")

    print(f"      streaming TCN over {len(tcn_input)} smp "
          f"(context={INPUT_LEN}, hop={PREDICT_LEN}) ...")
    stream = tcn_stream(tcn, tcn_input)
    tcn_pred = stream["tcn_pred"]
    actual   = stream["actual_future"]
    corrs    = stream["corrs"]

    anti     = (-tcn_pred).astype(np.float32)
    residual = (actual + anti).astype(np.float32)

    save_wav(tcn_pred, out_dir / "4_tcn_prediction.wav")
    save_wav(anti,     out_dir / "5_anti_noise.wav")
    save_wav(residual, out_dir / "6_residual.wav")

    # -- 4. Report -------------------------------------------------------------
    nr_db = db(rms(actual), rms(residual))
    print("\n" + "-" * 64)
    print(f"  windows               : {len(corrs)}  "
          f"(context={INPUT_LEN}, predict={PREDICT_LEN}, hop={PREDICT_LEN})")
    print(f"  TCN vs extracted-noise corr   "
          f"mean={corrs.mean():+.3f}   median={np.median(corrs):+.3f}   "
          f"p10={np.percentile(corrs,10):+.3f}   p90={np.percentile(corrs,90):+.3f}")
    print(f"  RMS extracted-noise   : {rms(actual):.5f}")
    print(f"  RMS anti-noise        : {rms(anti):.5f}")
    print(f"  RMS residual          : {rms(residual):.5f}")
    print(f"  cancellation          : {nr_db:+.2f} dB   (higher = more noise removed)")
    print("-" * 64)
    print(f"  WAVs written under    : {out_dir.resolve()}")
    for name in ("1_mix.wav", "2_unet_clean.wav", "3_unet_extracted_noise.wav",
                 "3b_unet_extracted_noise_lpf.wav",
                 "4_tcn_prediction.wav", "5_anti_noise.wav", "6_residual.wav"):
        print(f"    {name}")


if __name__ == "__main__":
    main()
