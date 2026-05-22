"""
test_fusion_pipeline_unet_streaming.py
======================================
U-Net + Causal-TCN fusion driven by **streaming chunked** U-Net inference.

Why this exists
---------------
`test_fusion_pipeline_unet.py` feeds the entire waveform into the U-Net in a
single forward pass.  For long files (>~20 s) this causes two pathologies:

  1. The U-Net's Squeeze-Excitation channel-attention blocks globally pool
     across the full time axis.  On a 43-second clip that's ~5400 STFT
     frames; the per-channel attention scalars get averaged toward the
     dataset mean and lose late-segment salience -> the model starts
     dropping speech that arrived "late" in the file.
  2. The internal STFT/iSTFT operates on the entire signal, so any tiny
     phase / DC inconsistency at one end can leak into reconstruction at
     the other end.

Both effects vanish once each chunk gets its own SE pooling and its own
iSTFT reconstruction.

Hardware mirroring
------------------
On Jetson the mic delivers audio in a stream — the model never sees the
whole utterance at once.  This script imitates that pattern with overlapping
fixed-length chunks (`--chunk_sec 2.0 --hop_sec 1.0` by default = 50 %
overlap).  Inside each chunk the U-Net sees a self-contained 2 s window;
between chunks a trapezoidal cosine cross-fade reconstructs a seamless
output via overlap-add.

Pipeline
--------
                  +-------- U-Net (chunked, OLA) --------+
       mix  ->   |   clean[k] = unet(mix[k])             |   <- 2 s windows,
       (43 s)    |   clean    = OLA-merge(clean[k])      |      1 s hop
                 +---------------------------------------+
                                  |
                  extracted_noise = mix - clean
                                  |
                  low_pass_filter (Cheby II 180 Hz, causal sosfilt)
                                  |  <- same training-time front end
                  TCN streams 80-sample forecasts in 1024-sample context
                                  |
       anti = -prediction,  residual = noise_lpf + anti

Run
---
    python test_fusion_pipeline_unet_streaming.py --mix recording.wav

    # Tighter latency (hardware-realistic; smaller window, more chunks):
    python test_fusion_pipeline_unet_streaming.py --mix recording.wav \
        --chunk_sec 1.0 --hop_sec 0.5

    # Pick a different TCN / U-Net checkpoint:
    python test_fusion_pipeline_unet_streaming.py --mix recording.wav \
        --tcn_ckpt ANC_Hardware_Test/tcn_predictor.pt \
        --unet_ckpt Run/U_net/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

# U-Net checkpoints saved on Linux contain PosixPath objects; remap on Windows.
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

SCRIPT_DIR = Path(__file__).resolve().parent
ANC_DIR    = SCRIPT_DIR / "ANC_Hardware_Test"
UNET_DIR   = SCRIPT_DIR / "Run" / "U_net"

for p in (ANC_DIR, UNET_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from predictor import (                                        # noqa: E402
    LightweightTCN,
    StreamingRingBuffer,
    predict_tcn,
    low_pass_filter,
    INPUT_LEN,
    PREDICT_LEN,
    SAMPLE_RATE,
)

SR = SAMPLE_RATE   # 16 kHz


# ---------------------------------------------------------------------------
# I/O helpers (identical to non-streaming version)
# ---------------------------------------------------------------------------

def load_audio(path: Path, sr: int = SR) -> np.ndarray:
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
# STREAMING U-NET INFERENCE  (the whole point of this script)
# ---------------------------------------------------------------------------

def _make_ola_window(chunk: int, overlap: int) -> np.ndarray:
    """Trapezoidal cross-fade window.

    Ones in the interior, linear ramp up over the first `overlap` samples,
    linear ramp down over the last `overlap` samples.  With hop = chunk -
    overlap and 50 % overlap (overlap == chunk/2) the window pair sums to
    1.0 across the entire interior of the output -- seamless OLA, no audible
    chunk boundaries.  We still divide by the per-sample window sum at the
    very edges (first chunk's leading half, last chunk's trailing half) so
    the output level stays flat even there.
    """
    win = np.ones(chunk, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        win[:overlap]  = ramp
        win[-overlap:] = ramp[::-1]
    return win


def enhance_waveform_streaming(model: torch.nn.Module,
                                noisy: np.ndarray,
                                device: torch.device,
                                sr: int = SR,
                                chunk_sec: float = 2.0,
                                hop_sec: float = 1.0,
                                verbose: bool = False) -> tuple:
    """Process the U-Net in fixed chunks with cosine overlap-add.

    Returns (enhanced_waveform, n_chunks, mean_chunk_latency_ms).
    """
    if noisy.ndim > 1:
        noisy = noisy.mean(axis=0)
    noisy = noisy.astype(np.float32)

    chunk = int(round(chunk_sec * sr))
    hop   = int(round(hop_sec   * sr))
    if hop <= 0 or hop > chunk:
        raise ValueError(f"hop_sec ({hop_sec}) must satisfy 0 < hop <= chunk")
    overlap = chunk - hop

    n = len(noisy)
    if n <= chunk:
        # File shorter than one chunk -- single inference, no OLA needed.
        with torch.inference_mode():
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            y = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return y[:n], 1, 0.0

    win = _make_ola_window(chunk, overlap)

    out  = np.zeros(n, dtype=np.float32)
    norm = np.zeros(n, dtype=np.float32)

    latencies_ms = []
    n_chunks = 0
    pos = 0
    while pos < n:
        end = min(pos + chunk, n)
        seg_len = end - pos
        seg = noisy[pos:end]
        if seg_len < chunk:
            seg = np.pad(seg, (0, chunk - seg_len))

        t0 = time.perf_counter()
        with torch.inference_mode():
            x = torch.from_numpy(seg).unsqueeze(0).to(device)
            y = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        # OLA: weighted accumulate, plus the matching window-sum normaliser
        # so the edges (where only one chunk contributes) stay level-flat.
        w = win[:seg_len]
        out [pos:pos + seg_len] += y[:seg_len] * w
        norm[pos:pos + seg_len] += w
        n_chunks += 1

        if verbose and (n_chunks % 10 == 0 or end >= n):
            print(f"      ... chunk {n_chunks:3d}  pos={pos/sr:5.2f}s  "
                  f"infer={latencies_ms[-1]:5.1f} ms")

        if end >= n:
            break
        pos += hop

    norm[norm < 1e-6] = 1.0
    enhanced = (out / norm).astype(np.float32)
    return enhanced, n_chunks, float(np.mean(latencies_ms))


# ---------------------------------------------------------------------------
# TCN streaming  (identical to non-streaming version)
# ---------------------------------------------------------------------------

def tcn_stream(model: LightweightTCN, noise_track: np.ndarray) -> dict:
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
    ring.push(noise_track[:INPUT_LEN])
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
        ring.push(actual)
    return dict(tcn_pred=pred_out, actual_future=actual_out, corrs=corrs)


# ---------------------------------------------------------------------------
# U-Net loader
# ---------------------------------------------------------------------------

def load_unet_raw(ckpt: Path, device: torch.device):
    """Return the raw torch model.  We need the bare module so we can drive
    it ourselves chunk-by-chunk -- the Run/U_net inference.py only exposes a
    whole-file enhance_waveform that we are explicitly NOT using here.
    """
    from inference import load_model        # noqa: E402  (Run/U_net/inference.py)
    model, _ = load_model(str(ckpt), device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Streaming U-Net + TCN fusion (chunked OLA inference, "
                    "hardware-mirroring)."
    )
    ap.add_argument("--mix",     type=Path, default=None,
                    help="Pre-mixed speech+noise wav.")
    ap.add_argument("--speech",  type=Path, default=None,
                    help="Clean speech wav (used with --noise and --snr_db).")
    ap.add_argument("--noise",   type=Path,
                    default=SCRIPT_DIR / "fan_noise.wav",
                    help="Noise wav (default: fan_noise.wav).")
    ap.add_argument("--snr_db",  type=float, default=0.0,
                    help="Target SNR when building the mix from --speech/--noise.")
    ap.add_argument("--tcn_ckpt", type=Path,
                    default=SCRIPT_DIR / "tcn_predictor.pt",
                    help="LightweightTCN state_dict.")
    ap.add_argument("--unet_ckpt", type=Path,
                    default=UNET_DIR / "checkpoints" / "best.pt",
                    help="U-Net checkpoint (default: Run/U_net/checkpoints/best.pt).")
    ap.add_argument("--chunk_sec", type=float, default=2.0,
                    help="U-Net inference chunk length in seconds (default 2.0).")
    ap.add_argument("--hop_sec",   type=float, default=1.0,
                    help="Hop between consecutive U-Net chunks "
                         "(default 1.0 -> 50%% overlap with chunk_sec=2).")
    ap.add_argument("--verbose",   action="store_true",
                    help="Print per-chunk progress + per-chunk latency.")
    ap.add_argument("--out_dir",   type=Path,
                    default=SCRIPT_DIR / "fusion_outputs_unet_streaming",
                    help="Output directory.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # -- 1. Build / load mix --
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

    # -- 2. STREAMING U-Net inference (chunked OLA) --
    print(f"\n[U-Net] loading {args.unet_ckpt}")
    unet = load_unet_raw(args.unet_ckpt, device)
    print(f"        chunked OLA: chunk={args.chunk_sec:.2f}s "
          f"hop={args.hop_sec:.2f}s  overlap={args.chunk_sec - args.hop_sec:.2f}s "
          f"({100*(1 - args.hop_sec/args.chunk_sec):.0f}%)")

    t0 = time.perf_counter()
    clean, n_chunks, mean_ms = enhance_waveform_streaming(
        unet, mix, device, sr=SR,
        chunk_sec=args.chunk_sec, hop_sec=args.hop_sec,
        verbose=args.verbose,
    )
    total_s = time.perf_counter() - t0

    # Length safety: OLA preserves T by construction, but be defensive about +/-1 smp.
    n = min(len(mix), len(clean))
    mix_t   = mix[:n]
    clean   = clean[:n]
    extracted_noise = (mix_t - clean).astype(np.float32)

    save_wav(clean,           out_dir / "2_unet_clean.wav")
    save_wav(extracted_noise, out_dir / "3_unet_extracted_noise.wav")
    print(f"        processed {n_chunks} chunks in {total_s:.2f}s  "
          f"(mean per-chunk inference = {mean_ms:.1f} ms)")
    print(f"        mix RMS = {rms(mix_t):.5f}   "
          f"clean RMS = {rms(clean):.5f}   "
          f"extracted-noise RMS = {rms(extracted_noise):.5f}")

    # -- 2b. LPF the residual -- same fix that took CRN from -14.92 -> +10.75 --
    tcn_input = low_pass_filter(extracted_noise)
    save_wav(tcn_input, out_dir / "3b_unet_extracted_noise_lpf.wav")
    print(f"        LPF (Cheby II 180 Hz, causal sosfilt)  "
          f"filtered-noise RMS = {rms(tcn_input):.5f}")

    # -- 3. TCN streaming over the LP-filtered residual --
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

    # -- 4. Report --
    nr_db = db(rms(actual), rms(residual))
    print("\n" + "-" * 64)
    print(f"  U-Net chunks          : {n_chunks}  "
          f"(chunk={args.chunk_sec:.2f}s hop={args.hop_sec:.2f}s)")
    print(f"  TCN windows           : {len(corrs)}  "
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
