"""
model_selector.py â€” UNet and Resemble-FT inference for local app.py.
CRN is handled by denoise_core.process() directly.
Public API: process(in_wav_path, model_name) -> out_wav_path
"""
from __future__ import annotations
import os

import sys
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.resolve()
_UNET_DIR      = _ROOT / "Run" / "U_net"
_UNET_CKPT     = _ROOT / "Colab_Ready_UNet" / "checkpoints_unet" / "best.pt"
_RESEMBLE_DIR  = Path(os.environ.get("RESEMBLE_DIR", "")) if os.environ.get("RESEMBLE_DIR") else None
_RESEMBLE_CKPT = _ROOT / "Run" / "benchmark_outputs" / "resemble_ft" / "denoiser_ft.pt"

_OUT_DIR = _ROOT / "outputs_tmp"
_OUT_DIR.mkdir(exist_ok=True)

_SAMPLE_RATE  = 16_000
_RESEMBLE_SR  = 44_100
_TARGET_RMS   = 0.15
_PEAK_GUARD   = 0.99

_unet_model     = None
_resemble_model = None


def _read_wav_mono_16k(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        sr   = wf.getframerate()
        n_ch = wf.getnchannels()
        sw   = wf.getsampwidth()
        raw  = wf.readframes(wf.getnframes())
    dtype = np.int16 if sw == 2 else np.int32
    pcm   = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    pcm  /= (32768.0 if sw == 2 else 2147483648.0)
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)
    if sr != _SAMPLE_RATE:
        import scipy.signal as ss
        pcm = ss.resample(pcm, int(len(pcm) * _SAMPLE_RATE / sr)).astype(np.float32)
    return pcm


def _write_wav(pcm: np.ndarray, stem: str, suffix: str) -> str:
    rms = float(np.sqrt(np.mean(pcm ** 2)))
    if rms > 1e-6:
        pcm = pcm * min(_TARGET_RMS / rms, 3.0)
    peak = float(np.abs(pcm).max())
    if peak > _PEAK_GUARD:
        pcm = pcm * (_PEAK_GUARD / peak)
    out = str(_OUT_DIR / f"{stem}_{suffix}.wav")
    i16 = np.clip(pcm * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(i16.tobytes())
    return out


def _get_unet():
    global _unet_model
    if _unet_model is None:
        import torch, importlib.util
        spec = importlib.util.spec_from_file_location(
            "unet_inference", _UNET_DIR / "inference.py"
        )
        unet_inf = importlib.util.module_from_spec(spec)
        # Register model_unet_advanced in sys.path so unet_inference can import it
        if str(_UNET_DIR) not in sys.path:
            sys.path.insert(0, str(_UNET_DIR))
        spec.loader.exec_module(unet_inf)
        _unet_model, _ = unet_inf.load_model(_UNET_CKPT, torch.device("cpu"))
        _unet_model.eval()
    return _unet_model


def _process_unet(in_wav_path: str) -> str:
    import torch
    model = _get_unet()
    pcm   = _read_wav_mono_16k(in_wav_path)
    with torch.inference_mode():
        out = model(torch.from_numpy(pcm).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
    return _write_wav(out, Path(in_wav_path).stem, "unet")


def _get_resemble():
    global _resemble_model
    if _resemble_model is None:
        import torch, pathlib
        _orig = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        if str(_RESEMBLE_DIR) not in sys.path:
            sys.path.insert(0, str(_RESEMBLE_DIR))
        from resemble_enhance.denoiser.train import Denoiser, HParams
        model = Denoiser(HParams())
        sd = torch.load(str(_RESEMBLE_CKPT), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(sd["denoiser"], strict=False)
        if missing:
            print(f"[resemble_ft] {len(missing)} missing keys")
        model.eval()
        pathlib.PosixPath = _orig
        _resemble_model = model
    return _resemble_model


def _process_resemble_ft(in_wav_path: str) -> str:
    import torch, torchaudio
    model = _get_resemble()
    if str(_RESEMBLE_DIR) not in sys.path:
        sys.path.insert(0, str(_RESEMBLE_DIR))
    from resemble_enhance.inference import inference
    pcm = _read_wav_mono_16k(in_wav_path)
    wav = torchaudio.functional.resample(
        torch.from_numpy(pcm), _SAMPLE_RATE, _RESEMBLE_SR
    )
    out = inference(model=model, dwav=wav, sr=_RESEMBLE_SR, device=torch.device("cpu"))
    if isinstance(out, tuple):
        out = out[0]
    out = torchaudio.functional.resample(out, _RESEMBLE_SR, _SAMPLE_RATE)
    return _write_wav(out.numpy().astype(np.float32), Path(in_wav_path).stem, "resemble_ft")


def process(in_wav_path: str, model_name: str) -> str:
    if model_name == "UNet":
        return _process_unet(in_wav_path)
    if model_name == "Resemble-FT":
        return _process_resemble_ft(in_wav_path)
    raise ValueError(f"Unknown model: {model_name}")
