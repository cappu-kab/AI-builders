#!/usr/bin/env python3
"""
crn_pipeline.py â€” Record â†’ Denoise (CRN) â†’ Play
Jetson Orin Nano, 16 kHz mono, ESP32-S3 @ /dev/ttyTHS0.

Usage:
    source ~/anc_env/bin/activate
    python3 ~/crn_pipeline.py [--duration 3] [--port /dev/ttyTHS0] [--baud 2000000]
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
import serial

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CRN_ROOT  = Path(__file__).parent.parent / "Run" / "CRN"
CKPT_PATH = CRN_ROOT / "checkpoints" / "best.pt"
if str(CRN_ROOT) not in sys.path:
    sys.path.insert(0, str(CRN_ROOT))

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SAMPLE_RATE  = 16_000
MIC_H0       = 0xCC
MIC_H1       = 0xDD
SPK_HEADER   = bytes([0xAA, 0xBB])
PLAY_CHUNK_BYTES = 512    # bytes per speaker packet payload (= 256 int16 samples)
                           # ESP32 firmware rejects length > CHUNK_SIZE+64 = 576 bytes
_CRN_CTX_N   = 47616      # BiLSTM training window size (â‰ˆ3 s at 16 kHz)
_SAFE_SINGLE = 64_000     # process in one pass if L â‰¤ this (shorter than 2 hops,
                           # overlap-add would give no benefit)
_STFT_MIN    = 257        # torch.stft center=True reflect-pad needs L > n_fft//2
_OLA_HOP     = _CRN_CTX_N // 2   # 23808 â€” 50% overlap hop for overlap-add
_PRE_PAD_S   = 8_192      # silence pre-padded before model to warm up the forward
                           # BiLSTM (~64 STFT frames); trimmed from output afterward.
                           # Without this the first ~0.5â€“1 s of output can be quiet
                           # because the forward LSTM starts from zero state.
_DRY_MIX     = 0.03       # keep tiny â€” higher values blend raw mic noise back in.
_TARGET_RMS  = 0.15       # RMS target after enhancement; normalises volume across runs.

RAW_WAV = "/tmp/raw_input.wav"
ENH_WAV = "/tmp/enhanced.wav"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# UART helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _wait_for(port: serial.Serial, token: bytes, timeout: float = 3.0) -> None:
    """Block until `token` appears in the incoming byte stream or timeout."""
    deadline = time.perf_counter() + timeout
    buf = b""
    while time.perf_counter() < deadline:
        data = port.read(max(1, port.in_waiting))
        if data:
            buf += data
            if token in buf:
                return
    raise TimeoutError(f"Timed out waiting for {token!r}; last bytes: {buf[-64:]!r}")


def _recv_mic_packets(port: serial.Serial, duration: float) -> np.ndarray:
    """
    Collect 0xCC 0xDD mic packets for `duration` seconds.
    Packet layout: [0xCC][0xDD][len_lo][len_hi][PCM bytes (int16 LE)]
    Returns 1-D int16 array of all received samples.
    """
    frames: List[np.ndarray] = []
    deadline = time.perf_counter() + duration
    buf = bytearray()

    while time.perf_counter() < deadline:
        avail = port.in_waiting
        if avail:
            buf.extend(port.read(avail))
        else:
            b = port.read(1)
            if b:
                buf.extend(b)
            continue

        # Drain all complete packets from buf
        while True:
            # Search for 0xCC 0xDD header
            i = 0
            while i < len(buf) - 1:
                if buf[i] == MIC_H0 and buf[i + 1] == MIC_H1:
                    break
                i += 1
            else:
                # Header not found; keep last byte (may be 0xCC of next header)
                buf = buf[-1:] if buf else bytearray()
                break

            if i > 0:
                del buf[:i]         # strip leading garbage before header

            if len(buf) < 4:
                break               # need at least header (2) + length (2)

            length = struct.unpack_from("<H", buf, 2)[0]
            total  = 4 + length
            if len(buf) < total:
                break               # payload not fully arrived yet

            pcm = np.frombuffer(bytes(buf[4:total]), dtype=np.int16)
            del buf[:total]
            if len(pcm):
                frames.append(pcm)

    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CRN helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_crn():
    """
    Load CRN checkpoint once, before any timing. Returns (model, device).

    Three-attempt strategy for Jetson cuDNN INTERNAL_ERROR in flatten_parameters():

    Attempt 1 â€” eval() before .to(cuda):
      Load to CPU (safe, no cuDNN involved), call eval(), then move to CUDA.
      In eval mode dropout is inactive, which avoids the cuDNN workspace
      allocation that triggers INTERNAL_ERROR on some Jetson cuDNN 8.x configs.

    Attempt 2 â€” cudnn.enabled=False:
      Disable cuDNN so flatten_parameters() is a no-op (is_acceptable=False).
      LSTM runs via the generic CUDA path â€” slower but correct.

    Attempt 3 â€” CPU fallback:
      All CUDA paths failed; run inference on CPU.
    """
    import torch
    from inference import load_model  # CRN_ROOT is on sys.path

    cuda = torch.device("cuda")
    cpu  = torch.device("cpu")

    if not torch.cuda.is_available():
        model, _ = load_model(CKPT_PATH, cpu)
        return model, cpu

    # Attempt 1: load to CPU â†’ eval() â†’ move to CUDA
    try:
        model, _ = load_model(CKPT_PATH, cpu)
        model.eval()
        model.to(cuda)
        return model, cuda
    except RuntimeError as exc:
        print(f"  [WARN] CUDA (eval-before-move) failed: {exc!s}")

    # Attempt 2: disable cuDNN so flatten_parameters() is a no-op, reload to CUDA
    try:
        torch.backends.cudnn.enabled = False
        model, _ = load_model(CKPT_PATH, cuda)
        print("  [INFO] cuDNN disabled â€” BiLSTM runs via generic CUDA path.")
        return model, cuda
    except RuntimeError as exc:
        print(f"  [WARN] CUDA+cuDNN-disabled also failed: {exc!s}")
        torch.backends.cudnn.enabled = True   # restore for other ops

    # Attempt 3: CPU fallback
    print("  [WARN] All CUDA paths failed â€” falling back to CPU.")
    model, _ = load_model(CKPT_PATH, cpu)
    return model, cpu


def _run_model(model, x_np: np.ndarray, device) -> np.ndarray:
    """
    Single model.forward() pass on a 1-D float32 array.
    Falls back to CPU if the CUDA path raises (known cuDNN LSTM issue on some
    Jetson configurations with BiLSTM).
    """
    import torch

    def _infer(dev):
        with torch.inference_mode():
            x   = torch.from_numpy(x_np).unsqueeze(0).to(dev)  # (1, T)
            out = model(x)                                        # (1, T)
            return out.squeeze(0).cpu().numpy().astype(np.float32)

    if device.type == "cuda":
        try:
            return _infer(device)
        except Exception as exc:
            print(f"  [WARN] CUDA inference failed ({exc!s}); retrying on CPU.")
            model.to("cpu")
            return _infer(torch.device("cpu"))
    return _infer(device)


def enhance_waveform_chunked(model, pcm_f32: np.ndarray, device) -> np.ndarray:
    """
    Run CRN on a 1-D float32 waveform in range [-1, 1].

    â‰¤ _SAFE_SINGLE (4 s): single forward pass â€” matches training length exactly.

    Longer recordings: Hann-windowed 50% overlap-add (OLA).
      Window W = _CRN_CTX_N = 47616 samples (exactly the BiLSTM training
      context).  Hop H = W/2 = 23808 samples.
      Every output sample is the weighted average of TWO independent model
      passes that each see the full W-sample bidirectional context.  At the
      centre of each window the Hann weight is 1.0; it tapers smoothly to 0
      at the edges, so boundary artefacts are suppressed by design â€” no hard
      cuts, no crossfade logic needed.
      The OLA normalisation factor (sum of squared Hann windows at each
      sample) is uniformly 0.5 for 50% overlap, so dividing by it is exact.
    """
    L = len(pcm_f32)
    if L == 0:
        return np.zeros(0, dtype=np.float32)

    if L <= _SAFE_SINGLE:
        enh = _run_model(model, pcm_f32, device)
        return enh[:L]

    W = _CRN_CTX_N          # window length
    H = _OLA_HOP             # hop = W/2

    # Synthesis window: Hann.  For 50% overlap the OLA normalisation sum
    # is exactly 0.5 everywhere (COLA condition), so we just divide by 0.5.
    win = np.hanning(W).astype(np.float32)

    # Number of windows needed to cover L samples
    n_win = max(1, int(np.ceil((L - W) / H)) + 1) if L > W else 1
    padded_len = (n_win - 1) * H + W

    pcm_pad = np.concatenate([
        pcm_f32,
        np.zeros(padded_len - L, dtype=np.float32),
    ])

    output = np.zeros(padded_len, dtype=np.float32)
    norm   = np.zeros(padded_len, dtype=np.float32)

    for i in range(n_win):
        s   = i * H
        seg = pcm_pad[s : s + W]
        enh = _run_model(model, seg, device)
        # Trim/pad to W (STFT center-pad can shift length by Â±1)
        if len(enh) > W:
            enh = enh[:W]
        elif len(enh) < W:
            enh = np.pad(enh, (0, W - len(enh)))
        output[s : s + W] += enh * win
        norm[s : s + W]   += win

    # Normalise by accumulated window weight (â‰ˆ0.5 everywhere with 50% overlap)
    result = (output / np.maximum(norm, 1e-8))[:L]
    return result


def _suppress_musical_noise(enh_f32: np.ndarray) -> np.ndarray:
    """
    Zero-phase spectral smoothing to reduce CRN musical noise.

    Musical noise = rapid frame-to-frame STFT magnitude bursts left behind
    by the mask.  They sound like a tonal warble or faint echo.

    Fix: re-analyse the enhanced waveform with a SHORT STFT (n_fft=256,
    hop=64 â†’ 4 ms frames), apply a 3-frame centred moving average over the
    magnitude spectrum, then reconstruct with the original phase.

    Centred (zero-phase) MA: no group delay, no added echo.
    Window = 3 Ã— 4 ms = 12 ms â€” far shorter than any speech phoneme (â‰¥30 ms),
    so speech clarity is preserved while isolated magnitude spikes are averaged
    away.
    """
    from scipy.ndimage import uniform_filter1d
    import scipy.signal as ss

    n_fft = 256
    hop   = 64

    _, _, Z  = ss.stft(enh_f32, nperseg=n_fft, noverlap=n_fft - hop)
    mag      = np.abs(Z)
    phase    = np.angle(Z)

    # 3-frame zero-phase moving average along the time axis (axis 1)
    smoothed = uniform_filter1d(mag, size=3, axis=1, mode="nearest")

    _, out = ss.istft(smoothed * np.exp(1j * phase), noverlap=n_fft - hop)
    out = out.astype(np.float32)

    L = len(enh_f32)
    if len(out) >= L:
        return out[:L]
    return np.pad(out, (0, L - len(out)))


def _despike(pcm: np.ndarray, sigma_mult: float = 2.5) -> np.ndarray:
    """Interpolate over impulse samples exceeding sigma_mult Ã— std."""
    sigma = float(np.std(pcm))
    if sigma < 1e-6:
        return pcm
    thresh = sigma * sigma_mult
    spike = np.abs(pcm) > thresh
    if not spike.any():
        return pcm
    idx = np.arange(len(pcm))
    good = ~spike
    if good.sum() > 2:
        out = pcm.copy()
        out[spike] = np.interp(idx[spike], idx[good], pcm[good])
        return out
    return pcm


def _highpass(pcm: np.ndarray, cutoff: float = 200.0, sr: int = 16000, order: int = 4) -> np.ndarray:
    """Butterworth high-pass at 200 Hz â€” cuts sub-200 Hz hum/buzz after CRN."""
    from scipy.signal import butter, sosfilt
    sos = butter(order, cutoff, btype='high', fs=sr, output='sos')
    return sosfilt(sos, pcm).astype(np.float32)


def _noise_gate(pcm: np.ndarray, thresh: float = 0.015, frame_n: int = 160) -> np.ndarray:
    """Soft noise gate: frames below thresh RMS get attenuated proportionally.
    Gain is smoothed over 4 frames to prevent clicks at gate boundaries."""
    gain = np.ones(len(pcm), dtype=np.float32)
    for i in range(0, len(pcm), frame_n):
        seg = pcm[i:i + frame_n]
        rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
        gain[i:i + len(seg)] = min(1.0, rms / thresh) if rms < thresh else 1.0
    kernel = np.ones(frame_n * 4, dtype=np.float32) / (frame_n * 4)
    gain = np.convolve(gain, kernel, mode='same')
    return pcm * gain


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Pipeline phases
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def phase1_record(port: serial.Serial, duration: float,
                  raw_wav: str = RAW_WAV) -> Tuple[int, float]:
    """Send startmic, collect packets for `duration` s, send stopmic, save WAV."""
    t0 = time.perf_counter()

    # Recover from any previous Ctrl+C â€” stop lingering mic stream
    port.write(b"stopmic\n")
    port.flush()
    time.sleep(0.3)
    port.reset_input_buffer()

    port.write(b"startmic\n")
    port.flush()
    _wait_for(port, b"OK:mic_start")

    print(f"  *** RECORDING {duration:.0f} s â€” SPEAK NOW ***", flush=True)
    pcm_i16 = _recv_mic_packets(port, duration)
    print("  Recording done.", flush=True)

    port.write(b"stopmic\n")
    port.flush()
    try:
        _wait_for(port, b"OK:mic_stop", timeout=2.0)
    except TimeoutError:
        pass  # non-fatal: data already captured

    with wave.open(raw_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_i16.tobytes())

    elapsed = time.perf_counter() - t0
    n = len(pcm_i16)
    s = n / SAMPLE_RATE
    print(f"Phase 1 [RECORD]:   {elapsed:.3f} s  ({n} frames = {s:.1f} s of audio)")
    return n, elapsed


def phase2_denoise(model, device, raw_wav: str = RAW_WAV,
                   volume: float = 1.0) -> Tuple[float, str, float]:
    """Read raw WAV, run CRN, save enhanced WAV. Times wav-read to wav-saved."""
    t0 = time.perf_counter()

    with wave.open(raw_wav, "rb") as wf:
        raw_bytes = wf.readframes(wf.getnframes())
    pcm_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
    pcm_f32 = pcm_i16.astype(np.float32) / 32768.0
    input_s = len(pcm_f32) / SAMPLE_RATE
    t_read = time.perf_counter()

    # De-spike pass 1: before CRN â€” remove impulse spikes from raw mic input
    pcm_f32 = _despike(pcm_f32)
    t_despike1 = time.perf_counter()

    # Pre-pad with silence so the forward BiLSTM has ~64 warmup frames
    # before it reaches real audio â€” prevents the quiet beginning.
    pre = np.zeros(_PRE_PAD_S, dtype=np.float32)
    enh_padded = enhance_waveform_chunked(model, np.concatenate([pre, pcm_f32]), device)
    enh_f32 = enh_padded[_PRE_PAD_S : _PRE_PAD_S + len(pcm_f32)]
    t_crn = time.perf_counter()

    # De-spike pass 2: after CRN â€” CRN can produce a burst artifact at spike locations
    # even after pass 1 removed the raw spike (STFT frames around spike still unusual)
    enh_f32 = _despike(enh_f32)
    t_despike2 = time.perf_counter()

    # Suppress CRN musical noise artifacts (tonal warble left by mask)
    enh_f32 = _suppress_musical_noise(enh_f32)
    t_suppress = time.perf_counter()

    # Dry mix: small blend keeps over-suppressed speech frames audible.
    # Keep low â€” higher values blend raw mic noise back in.
    enh_f32 = (1.0 - _DRY_MIX) * enh_f32 + _DRY_MIX * pcm_f32

    # High-pass at 200 Hz â€” second-layer LF noise removal after CRN + dry mix
    enh_f32 = _highpass(enh_f32)

    # v2 volume: RMS-normalize then multiply â€” same approach that worked at --volume 2.5
    rms_out = float(np.sqrt(np.mean(enh_f32 ** 2)))
    if rms_out > 1e-6:
        enh_f32 = enh_f32 * min(_TARGET_RMS / rms_out, 3.0)
    if volume != 1.0:
        enh_f32 = enh_f32 * volume
    peak = float(np.abs(enh_f32).max())

    # Fade out last 0.5 s â€” suppress tail noise from floating mic
    fade_n = min(int(0.50 * SAMPLE_RATE), len(enh_f32))
    enh_f32[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

    # Soft peak limit: tanh compressor at 0.95 â€” smoothly squashes any remaining
    # runaway peaks that survived de-spike; preserves loudness, avoids hard-clip pops
    enh_f32 = np.tanh(enh_f32 / 0.95) * 0.95
    t_post = time.perf_counter()

    enh_i16 = np.clip(enh_f32 * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(ENH_WAV, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(enh_i16.tobytes())
    t_end = time.perf_counter()

    elapsed  = t_end - t0
    dev_type = "GPU" if device.type == "cuda" else "CPU"
    print(f"Phase 2 [DENOISE]:  {elapsed:.3f} s  (model: {dev_type}, input: {input_s:.1f} s)")
    print(f"  Sub-stages (ms):")
    print(f"    WAV read:                         {(t_read     - t0         )*1000:6.1f} ms")
    print(f"    De-spike pre-CRN:                 {(t_despike1 - t_read     )*1000:6.1f} ms")
    print(f"    CRN (STFT + BiLSTM + ISTFT):      {(t_crn      - t_despike1 )*1000:6.1f} ms  [{input_s:.1f}s audio]")
    print(f"    De-spike post-CRN:                {(t_despike2 - t_crn      )*1000:6.1f} ms")
    print(f"    Musical-noise suppression (STFT): {(t_suppress  - t_despike2 )*1000:6.1f} ms")
    print(f"    Post-proc (dry-mix + HP + norm):  {(t_post      - t_suppress )*1000:6.1f} ms")
    print(f"    WAV write:                        {(t_end       - t_post     )*1000:6.1f} ms")
    print(f"  audio â€” RMS out: {rms_out:.4f}  peak: {peak:.3f}")
    return elapsed, dev_type, input_s


def phase3_play(port: serial.Serial) -> Tuple[float, int]:
    """
    Read enhanced WAV, packetize into 512-byte chunks, send to ESP32.

    Key constraints from ESP32 firmware (work_sound_v5.ino):
      - Packet payload must be â‰¤ CHUNK_SIZE+64 = 576 bytes (512 samples Ã— 2 bytes
        = 1024 bytes was silently rejected â€” that was the original silent-speaker bug).
      - CHUNK_SIZE in the firmware is 512 BYTES, so we send 512-byte payloads
        = 256 int16 samples per packet.
      - A flush command clears the DMA buffer before playback to avoid a
        pop/glitch from stale data.
      - Pacing at 110% real-time (matching jetson_audio_sender.py) prevents
        DAC underruns while keeping the UART RX buffer from overflowing.
    """
    with wave.open(ENH_WAV, "rb") as wf:
        raw_pcm = wf.readframes(wf.getnframes())   # raw int16 bytes, no conversion needed

    # Flush ESP32 I2S DMA before playing (clears any stale silence/pops)
    port.write(b"flush\n")
    port.flush()
    try:
        _wait_for(port, b"OK:flush", timeout=2.0)
    except TimeoutError:
        pass
    port.reset_input_buffer()

    BYTES_PER_SEC = SAMPLE_RATE * 2               # 32 000 bytes/s
    SEND_RATE     = BYTES_PER_SEC * 1.02          # 32 640 bytes/s â€” 102% real-time
    PREFILL_BYTES = BYTES_PER_SEC * 0.5           # 0.5 s pre-buffered
    total_bytes = 0
    n_packets   = (len(raw_pcm) + PLAY_CHUNK_BYTES - 1) // PLAY_CHUNK_BYTES
    dur_s       = len(raw_pcm) / BYTES_PER_SEC
    print(f"  Streaming {n_packets} packets ({dur_s:.1f} s) ...", flush=True)

    t0        = time.perf_counter()
    pcm_sent  = 0

    for pkt_i, start in enumerate(range(0, len(raw_pcm), PLAY_CHUNK_BYTES)):
        chunk = raw_pcm[start : start + PLAY_CHUNK_BYTES]
        if len(chunk) % 2:
            chunk += b'\x00'
        packet = SPK_HEADER + struct.pack("<H", len(chunk)) + chunk
        port.write(packet)
        total_bytes += len(packet)
        pcm_sent    += len(chunk)

        if pcm_sent > PREFILL_BYTES:
            expected = (pcm_sent - PREFILL_BYTES) / SEND_RATE
            elapsed  = time.perf_counter() - t0
            if expected > elapsed:
                time.sleep(expected - elapsed)

    port.flush()
    elapsed = time.perf_counter() - t0
    kb = total_bytes / 1024
    print(f"Phase 3 [PLAY]:     {elapsed:.3f} s  ({kb:.1f} KB sent)")
    return elapsed, total_bytes


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> None:
    ap = argparse.ArgumentParser(description="CRN pipeline: Record â†’ Denoise â†’ Play")
    ap.add_argument("--duration", type=float, default=3.0,
                    help="Recording duration in seconds (default: 3)")
    ap.add_argument("--port",  default="/dev/ttyTHS0", help="UART device")
    ap.add_argument("--baud",  type=int, default=2_000_000, help="Baud rate")
    ap.add_argument("--skip-record", action="store_true",
                    help="Skip Phase 1; read raw WAV from --raw-out instead of recording")
    ap.add_argument("--raw-out", default=RAW_WAV,
                    help=f"Path to raw recorded WAV (default: {RAW_WAV})")
    ap.add_argument("--volume", type=float, default=1.0,
                    help="Output volume multiplier applied after all processing (default: 1.0)")
    args = ap.parse_args()

    print(f"\n=== CRN Pipeline  [{args.duration}s | {args.port} | {args.baud} baud] ===\n")

    # Load model BEFORE opening UART and BEFORE any phase timing
    print(f"Loading CRN from {CKPT_PATH} ...", end=" ", flush=True)
    model, device = load_crn()
    dev_label = "GPU" if device.type == "cuda" else "CPU"
    print(f"OK  ({dev_label})\n")

    port = serial.Serial(
        port      = args.port,
        baudrate  = args.baud,
        bytesize  = serial.EIGHTBITS,
        parity    = serial.PARITY_NONE,
        stopbits  = serial.STOPBITS_ONE,
        timeout   = 0.05,           # 50 ms read timeout (non-blocking polling)
    )
    port.reset_input_buffer()

    try:
        pipeline_t0 = time.perf_counter()

        if args.skip_record:
            print(f"Phase 1 [RECORD]:   skipped  (reading {args.raw_out})")
            n_frames, t1 = 0, 0.0
        else:
            n_frames, t1 = phase1_record(port, args.duration, args.raw_out)
        t2, dev_used, in_s = phase2_denoise(model, device, args.raw_out, args.volume)
        t3, bytes_sent     = phase3_play(port)

        total = time.perf_counter() - pipeline_t0

        print()
        print("=== Pipeline Summary ===")
        if args.skip_record:
            print(f"  Phase 1 [RECORD]:   skipped  ({args.raw_out})")
        else:
            print(f"  Phase 1 [RECORD]:   {t1:.3f} s  â€”  {n_frames} frames"
                  f" = {n_frames / SAMPLE_RATE:.2f} s @ {SAMPLE_RATE} Hz")
        print(f"  Phase 2 [DENOISE]:  {t2:.3f} s  â€”  model: {dev_used},"
              f" input: {in_s:.2f} s")
        print(f"  Phase 3 [PLAY]:     {t3:.3f} s  â€”  {bytes_sent / 1024:.1f} KB sent")
        print(f"  Total:              {total:.3f} s")
        print()
    finally:
        port.close()


if __name__ == "__main__":
    main()
