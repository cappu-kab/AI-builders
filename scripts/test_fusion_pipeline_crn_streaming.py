"""
test_fusion_pipeline_crn_streaming.py
=====================================
CRN + Causal-TCN fusion driven by **streaming chunked** CRN inference,
plus an explicit listener-perceived output so you can audition the actual
ANC deliverable.

Why CRN here (vs the U-Net version)
-----------------------------------
Spectral analysis on the same real-world recording showed CRN preserves
~5x more energy in the 300-4000 Hz speech band than the U-Net denoiser
(CRN_MID ~18.6  vs  UNet_MID ~5.1, averaged over the speech-present
section).  The "muffled / hard to hear" character of the U-Net pipeline
was the U-Net over-suppressing the speech band, not chunking artefacts.
CRN keeps the speech intelligible and gives the TCN a residual that
still cancels well.

Why chunked + OLA
-----------------
Even though CRN's late-segment "dropout" issue on this file turned out
to be the audio truly having no speech past 28s, processing in fixed
overlapping windows is still the right pattern:

  * It mirrors how the Jetson runtime sees audio (one window at a time,
    never the whole utterance).
  * It bounds the SE attention pooling window, which we saw distort
    long-file inference with U-Net.
  * It keeps memory usage bounded for arbitrarily long recordings.

Outputs
-------
    1_mix.wav                       the noisy input
    2_crn_clean.wav                 CRN-enhanced speech
    3_crn_extracted_noise.wav       residual the TCN works on (mix - clean)
    3b_extracted_noise_lpf.wav      same, after the strictly-causal
                                    Cheby II 180 Hz LPF (TCN training-time front end)
    4_tcn_prediction.wav            TCN's forecast of the LF noise
    5_anti_noise.wav                speaker emission = -prediction
    6_residual.wav                  LF noise floor that survives cancellation
    7_perceived_at_listener.wav     mix + anti  (what the listener actually hears:
                                    speech + (original_noise - predicted_noise))

Run
---
    python test_fusion_pipeline_crn_streaming.py --mix recording.wav

    # Tighter latency (smaller window, more chunks, hardware-mirroring):
    python test_fusion_pipeline_crn_streaming.py --mix recording.wav \
        --chunk_sec 1.0 --hop_sec 0.5

    # Pick a different CRN / TCN checkpoint:
    python test_fusion_pipeline_crn_streaming.py --mix recording.wav \
        --crn_ckpt Run/CRN/checkpoints/best.pt \
        --tcn_ckpt ANC_Hardware_Test/tcn_predictor.pt
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

if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

SCRIPT_DIR = Path(__file__).resolve().parent
ANC_DIR    = SCRIPT_DIR / "ANC_Hardware_Test"
CRN_DIR    = SCRIPT_DIR / "Run" / "CRN"

for p in (ANC_DIR, CRN_DIR):
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

SR = SAMPLE_RATE


# ---------------------------------------------------------------------------
# I/O helpers
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


def soft_noise_gate(wav: np.ndarray, sr: int = SR,
                    threshold_db: float = -40.0,
                    knee_db:      float =   6.0,
                    attack_ms:    float =   5.0,
                    release_ms:   float =  80.0) -> np.ndarray:
    """Soft-knee noise gate with separate attack / release time constants.

    Same algorithm used by scripts/inference.py: per-frame RMS in dBFS ->
    cosine soft-knee gain in [0, 1] -> one-pole IIR smoothing (fast attack
    so speech onsets are not chopped, slow release so the gate stays open
    through brief inter-word pauses) -> linear interp back to sample rate.

    Defaults silence anything below -40 dBFS over a 6 dB knee, which kills
    residual model hiss in non-speech regions without touching quiet
    speech (typically above -30 dBFS).
    """
    n = len(wav)
    if n == 0:
        return wav
    hop = max(1, sr // 100)        # 10 ms
    win = max(hop, sr // 50)       # 20 ms
    n_frames = max(1, (n + hop - 1) // hop)
    rms_db = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        e = min(s + win, n)
        rms_db[i] = 20.0 * np.log10(
            float(np.sqrt(np.mean(wav[s:e] ** 2))) + 1e-12
        )
    half_knee = knee_db * 0.5
    lo, hi = threshold_db - half_knee, threshold_db + half_knee
    gain = np.where(
        rms_db <= lo, 0.0,
        np.where(
            rms_db >= hi, 1.0,
            0.5 * (1.0 - np.cos(np.pi * (rms_db - lo) / knee_db))
        )
    ).astype(np.float32)
    fps = sr / hop
    a_c = float(np.exp(-1.0 / max(1.0, attack_ms  * fps / 1000.0)))
    r_c = float(np.exp(-1.0 / max(1.0, release_ms * fps / 1000.0)))
    g = float(gain[0])
    for i in range(n_frames):
        target = float(gain[i])
        coeff  = a_c if target > g else r_c
        g      = coeff * g + (1.0 - coeff) * target
        gain[i] = g
    frame_pos    = np.arange(n_frames, dtype=np.float64) * hop
    sample_pos   = np.arange(n,        dtype=np.float64)
    gain_samples = np.interp(sample_pos, frame_pos, gain).astype(np.float32)
    return (wav * gain_samples).astype(np.float32)


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
# Streaming chunked inference (model-agnostic OLA driver)
# ---------------------------------------------------------------------------

def _make_ola_window(chunk: int, overlap: int) -> np.ndarray:
    """Trapezoidal cross-fade window. With 50% overlap the pair sums to 1.0
    across the interior; the divide-by-window-sum at the end keeps the
    edges level-flat too."""
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
    """Process the model in fixed chunks with cosine overlap-add.

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
# TCN streaming  (identical to other fusion scripts)
# ---------------------------------------------------------------------------

def tcn_stream(model: LightweightTCN, noise_track: np.ndarray,
               tcn_hop: int = PREDICT_LEN // 2) -> dict:
    """TCN streaming with cosine OLA between consecutive predictions.

    Originally we tiled PREDICT_LEN (80-sample) predictions back-to-back with
    no overlap.  Each prediction ended at a value uncorrelated with the next
    prediction's starting value -> step discontinuity at every 5 ms boundary
    -> audible "choppy" / "pulsing" artefact in the anti-noise and residual
    tracks.  Cross-fading consecutive predictions with a Hann window kills
    those steps.

    With tcn_hop = PREDICT_LEN // 2 (40 samples = 2.5 ms), every output sample
    receives a Hann-weighted contribution from two predictions, the window
    sum at the interior is constant 1.0, and the streaming cost roughly
    doubles (still ~6500 TCN calls for a 43-second clip, ~30 ms total).
    """
    if len(noise_track) < INPUT_LEN + PREDICT_LEN:
        raise ValueError(
            f"Noise track too short for TCN streaming: need >= "
            f"{INPUT_LEN + PREDICT_LEN} smp, got {len(noise_track)}."
        )
    hop = max(1, int(tcn_hop))
    if hop > PREDICT_LEN:
        hop = PREDICT_LEN  # never less overlap than zero
    n_preds = (len(noise_track) - INPUT_LEN - PREDICT_LEN) // hop + 1
    out_len = (n_preds - 1) * hop + PREDICT_LEN

    pred_acc   = np.zeros(out_len, dtype=np.float32)
    actual_acc = np.zeros(out_len, dtype=np.float32)
    norm       = np.zeros(out_len, dtype=np.float32)

    # Hann window over PREDICT_LEN samples — sums to ~1.0 in the interior
    # when consecutive windows are spaced hop apart with 50% overlap (hop =
    # PREDICT_LEN/2).  Trapezoidal would also work; Hann is smoother.
    win = np.hanning(PREDICT_LEN).astype(np.float32)
    if hop == PREDICT_LEN:
        win[:] = 1.0  # no overlap requested -> revert to flat tiling

    ring = StreamingRingBuffer(INPUT_LEN)
    ring.push(noise_track[:INPUT_LEN])

    corrs = np.empty(n_preds, dtype=np.float32)
    for i in range(n_preds):
        t        = INPUT_LEN + i * hop
        snapshot = ring.snapshot()
        pred     = predict_tcn(model, snapshot)
        actual   = noise_track[t : t + PREDICT_LEN]

        slc = slice(i * hop, i * hop + PREDICT_LEN)
        pred_acc  [slc] += pred  * win
        actual_acc[slc] += actual * win
        norm      [slc] += win

        corrs[i] = (np.corrcoef(pred, actual)[0, 1]
                    if actual.std() > 1e-9 else 0.0)

        # Open-loop / mic-wins: ring is always re-pushed with hop samples of
        # ACTUAL past, never with the model's prediction.  When hop <
        # PREDICT_LEN this also keeps the ring perfectly aligned to the next
        # iteration's `t`.
        ring.push(actual[:hop])

    norm[norm < 1e-6] = 1.0
    pred_out   = (pred_acc   / norm).astype(np.float32)
    actual_out = (actual_acc / norm).astype(np.float32)
    return dict(tcn_pred=pred_out, actual_future=actual_out, corrs=corrs)


# ---------------------------------------------------------------------------
# CRN loader
# ---------------------------------------------------------------------------

def load_crn_raw(ckpt: Path, device: torch.device):
    """Return the raw torch CRN module so we can drive it chunk-by-chunk."""
    from inference import load_model        # noqa: E402  (Run/CRN/inference.py)
    model, _ = load_model(str(ckpt), device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Streaming CRN + TCN fusion with listener-perceived "
                    "audio output (chunked OLA, hardware-mirroring)."
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
    ap.add_argument("--crn_ckpt", type=Path,
                    default=CRN_DIR / "checkpoints" / "best.pt",
                    help="CRN checkpoint (default: Run/CRN/checkpoints/best.pt).")
    ap.add_argument("--chunk_sec", type=float, default=2.0,
                    help="CRN inference chunk length in seconds (default 2.0).")
    ap.add_argument("--hop_sec",   type=float, default=1.0,
                    help="Hop between consecutive CRN chunks (default 1.0).")
    ap.add_argument("--verbose",   action="store_true",
                    help="Print per-chunk progress + per-chunk latency.")
    ap.add_argument("--gate_db",   type=float, default=-40.0,
                    help="Soft noise gate threshold in dBFS applied to the "
                         "CRN clean output (default -40, use -120 to disable).")
    ap.add_argument("--out_dir",   type=Path,
                    default=SCRIPT_DIR / "fusion_outputs_crn_streaming",
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

    # -- 2. STREAMING CRN inference (chunked OLA) --
    print(f"\n[CRN] loading {args.crn_ckpt}")
    crn = load_crn_raw(args.crn_ckpt, device)
    print(f"      chunked OLA: chunk={args.chunk_sec:.2f}s "
          f"hop={args.hop_sec:.2f}s  overlap={args.chunk_sec - args.hop_sec:.2f}s "
          f"({100*(1 - args.hop_sec/args.chunk_sec):.0f}%)")

    t0 = time.perf_counter()
    clean, n_chunks, mean_ms = enhance_waveform_streaming(
        crn, mix, device, sr=SR,
        chunk_sec=args.chunk_sec, hop_sec=args.hop_sec,
        verbose=args.verbose,
    )
    total_s = time.perf_counter() - t0

    n = min(len(mix), len(clean))
    mix_t   = mix[:n]
    clean   = clean[:n]
    extracted_noise = (mix_t - clean).astype(np.float32)

    save_wav(clean,           out_dir / "2_crn_clean.wav")
    save_wav(extracted_noise, out_dir / "3_crn_extracted_noise.wav")
    print(f"      processed {n_chunks} chunks in {total_s:.2f}s  "
          f"(mean per-chunk inference = {mean_ms:.1f} ms)")
    print(f"      mix RMS = {rms(mix_t):.5f}   "
          f"clean RMS = {rms(clean):.5f}   "
          f"extracted-noise RMS = {rms(extracted_noise):.5f}")

    # -- 2c. Soft noise gate on the clean output --
    # Suppresses residual model hiss during quiet / no-speech segments
    # without touching real speech.  This is the single biggest perceptual
    # win for casual listening to 2_crn_clean.wav.
    if args.gate_db > -119.0:
        clean_gated = soft_noise_gate(clean, sr=SR, threshold_db=args.gate_db)
        save_wav(clean_gated, out_dir / "2b_crn_clean_gated.wav")
        print(f"      noise gate ({args.gate_db:+.1f} dBFS, knee=6 dB, "
              f"attack=5 ms, release=80 ms)  gated RMS = {rms(clean_gated):.5f}")
    else:
        clean_gated = clean
        print(f"      noise gate disabled (--gate_db {args.gate_db})")

    # -- 2b. LPF the residual --
    tcn_input = low_pass_filter(extracted_noise)
    save_wav(tcn_input, out_dir / "3b_extracted_noise_lpf.wav")
    print(f"      LPF (Cheby II 180 Hz, causal sosfilt)  "
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

    # -- 3b. LISTENER-PERCEIVED AUDIO (headset-style ANC simulation) ---------
    # Headset model: the listener hears denoised speech from the playback
    # path PLUS the residual ambient noise that survives the speaker's
    # cancellation.  Mathematically this is exactly:
    #     perceived(t) = clean(t) + residual(t)
    #                  = denoised_speech(t) + (LF_noise(t) - LF_pred(t))
    # We use the gated clean output here -- the gate kills residual hiss in
    # the speech path, the TCN handles ambient LF noise in the air path,
    # together they deliver intelligible speech with reduced ambient rumble.
    perceived_start = INPUT_LEN
    perceived_end   = INPUT_LEN + len(residual)
    clean_for_mix   = clean_gated[perceived_start:perceived_end]
    perceived       = (clean_for_mix + residual).astype(np.float32)
    save_wav(perceived, out_dir / "7_perceived_at_listener.wav")

    # -- 4. Report -----------------------------------------------------------
    nr_db = db(rms(actual), rms(residual))
    print("\n" + "-" * 64)
    print(f"  CRN chunks            : {n_chunks}  "
          f"(chunk={args.chunk_sec:.2f}s hop={args.hop_sec:.2f}s)")
    print(f"  TCN windows           : {len(corrs)}  "
          f"(context={INPUT_LEN}, predict={PREDICT_LEN}, OLA hop={PREDICT_LEN // 2})")
    print(f"  TCN vs extracted-noise corr   "
          f"mean={corrs.mean():+.3f}   median={np.median(corrs):+.3f}   "
          f"p10={np.percentile(corrs,10):+.3f}   p90={np.percentile(corrs,90):+.3f}")
    print(f"  RMS extracted-noise   : {rms(actual):.5f}")
    print(f"  RMS anti-noise        : {rms(anti):.5f}")
    print(f"  RMS residual          : {rms(residual):.5f}")
    print(f"  cancellation (LF band): {nr_db:+.2f} dB   (TCN vs LP-filtered noise)")
    print("-" * 64)
    print(f"  WAVs written under    : {out_dir.resolve()}")
    listening = "7_perceived_at_listener.wav"
    listed = ["1_mix.wav", "2_crn_clean.wav", "2b_crn_clean_gated.wav",
              "3_crn_extracted_noise.wav", "3b_extracted_noise_lpf.wav",
              "4_tcn_prediction.wav", "5_anti_noise.wav", "6_residual.wav",
              "7_perceived_at_listener.wav"]
    for name in listed:
        marker = "  <-- LISTEN TO THIS (headset ANC simulation)" if name == listening else ""
        print(f"    {name}{marker}")


if __name__ == "__main__":
    main()
