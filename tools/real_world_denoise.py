"""
real_world_denoise.py
=====================
Denoise a single .mp3 file with all five models and save .wav outputs.

Usage
-----
    python real_world_denoise.py noisy_audio.mp3
    python real_world_denoise.py noisy_audio.mp3 --out_dir my_outputs

Models
------
  CRN          — Run/checkpoints_crn/best.pt
  U-Net        — Colab_Ready_UNet/checkpoints_unet/best.pt
  Resemble     — pretrained weights (auto-downloaded)
  Resemble-FT  — Run/benchmark_outputs/resemble_ft/denoiser_ft.pt
  MossFormer-FT— Run/benchmark_outputs/mossformer_ft/mossformer_ft.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

# Checkpoints saved on Linux contain PosixPath objects; remap on Windows.
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

SCRIPT_DIR   = Path(__file__).parent
RUN_DIR      = SCRIPT_DIR / "Run"
CRN_DIR      = RUN_DIR / "CRN"
UNET_DIR     = RUN_DIR / "U_net"
RESEMBLE_DIR = Path(os.environ["RESEMBLE_DIR"]) if "RESEMBLE_DIR" in os.environ else None

SR           = 16_000
_RESEMBLE_SR = 44_100


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_module(path: Path, name: str):
    """Load a .py file as a named module without permanently polluting sys.path."""
    parent = str(path.parent)
    added  = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        if added:
            sys.path.remove(parent)
    return mod


def load_audio(path: Path) -> np.ndarray:
    """Load any audio file and return 16 kHz mono float32 numpy array.

    Avoids torchaudio.load() to sidestep the torchcodec/FFmpeg DLL error on
    Windows.  soundfile handles WAV/FLAC/OGG without any subprocess; librosa
    (via audioread) handles MP3 using an ffmpeg subprocess, not a DLL.
    """
    import soundfile as sf
    import librosa

    wav: np.ndarray | None = None
    sr_in: int | None = None

    # soundfile: fast, no subprocess, covers WAV / FLAC / OGG / AIFF
    try:
        wav, sr_in = sf.read(str(path), always_2d=True, dtype="float32")
        wav = wav.mean(axis=1)          # (samples, channels) -> (samples,)
    except Exception:
        pass

    # librosa fallback: handles MP3 via audioread / ffmpeg subprocess
    if wav is None:
        wav, sr_in = librosa.load(str(path), sr=None, mono=True, dtype=np.float32)

    if sr_in != SR:
        wav = librosa.resample(wav, orig_sr=sr_in, target_sr=SR)

    return wav.astype(np.float32)


def save_wav(wav: np.ndarray, path: Path) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(wav, -1.0, 1.0).astype(np.float32), SR)


def wiener_post_filter(wav: np.ndarray,
                       n_fft: int = 512, hop: int = 128,
                       floor: float = 0.05,
                       alpha: float = 0.95) -> np.ndarray:
    """Residual noise suppression via recursive minimum-statistics Wiener filter.

    Estimates the residual noise PSD using a running minimum of the power
    spectrum (Martin 2001), then applies a floored Wiener gain.  This removes
    the faint steady-state noise haze that deep-learning models leave behind
    without distorting transient speech frames.

    Args:
        floor : minimum gain (0.05 = at most 26 dB suppression in silent bins)
        alpha : noise PSD smoothing (higher = slower tracking, less musical noise)
    """
    import librosa
    S       = librosa.stft(wav, n_fft=n_fft, hop_length=hop)
    power   = np.abs(S) ** 2                        # (F, T)

    # Initialise noise PSD from average of first 5 frames (assumed noise-only)
    noise   = np.mean(power[:, :5], axis=1, keepdims=True) * np.ones_like(power)
    # Recursive minimum: update each column
    for t in range(1, power.shape[1]):
        noise[:, t] = np.minimum(alpha * noise[:, t - 1] + (1 - alpha) * power[:, t],
                                 power[:, t])

    gain    = np.maximum(1.0 - noise / (power + 1e-12), floor)
    S_clean = gain * S
    out     = librosa.istft(S_clean, hop_length=hop, length=len(wav))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Model wrappers  (loading logic mirrors Run/benchmark.py)
# ---------------------------------------------------------------------------

class CRNWrapper:
    def __init__(self, ckpt: Path, device: torch.device):
        mod = _load_module(CRN_DIR / "inference.py", "crn_inference")
        self.model, _ = mod.load_model(str(ckpt), device)
        self._enhance  = mod.enhance_waveform
        self.device    = device

    def enhance(self, noisy: np.ndarray) -> np.ndarray:
        return self._enhance(self.model, noisy, self.device, sr=SR)


class UNetWrapper:
    def __init__(self, ckpt: Path, device: torch.device):
        mod = _load_module(UNET_DIR / "inference.py", "unet_inference")
        self.model, _ = mod.load_model(str(ckpt), device)
        self._enhance  = mod.enhance_waveform
        self.device    = device

    def enhance(self, noisy: np.ndarray) -> np.ndarray:
        return self._enhance(self.model, noisy, self.device, sr=SR)


class ResembleWrapper:
    """Wraps the base pretrained Resemble denoiser."""

    def __init__(self, device: torch.device):
        self.device   = device
        self._denoise = None
        self._setup()

    def _setup(self):
        if RESEMBLE_DIR.is_dir() and str(RESEMBLE_DIR) not in sys.path:
            sys.path.insert(0, str(RESEMBLE_DIR))
        try:
            from resemble_enhance.enhancer.inference import denoise
            self._denoise = denoise
        except ImportError as e:
            print(f"  [resemble] import failed: {e}")

    def enhance(self, noisy: np.ndarray) -> np.ndarray:
        if self._denoise is None:
            return noisy
        # Run on CPU — avoids a Windows CUDA weight-download bug in the base model.
        cpu = torch.device("cpu")
        try:
            wav = torch.from_numpy(noisy)
            out, out_sr = self._denoise(wav, SR, cpu, run_dir=None)
            out_np = out.cpu().numpy().squeeze()
            if out_sr != SR:
                t = torch.from_numpy(out_np).unsqueeze(0)
                t = torchaudio.functional.resample(t, out_sr, SR)
                out_np = t.squeeze(0).numpy()
            return out_np.astype(np.float32)
        except Exception as e:
            print(f"  [resemble] inference error: {e}")
            return noisy


class ResembleFTWrapper:
    """Fine-tuned Resemble denoiser loaded from a local checkpoint."""

    def __init__(self, ft_ckpt: Path, device: torch.device):
        self.device   = device
        self._model   = None
        self._setup(ft_ckpt)

    def _setup(self, ft_ckpt: Path):
        if RESEMBLE_DIR.is_dir() and str(RESEMBLE_DIR) not in sys.path:
            sys.path.insert(0, str(RESEMBLE_DIR))
        try:
            from resemble_enhance.denoiser.inference import load_denoiser
            model = load_denoiser(None, self.device)
            sd = torch.load(str(ft_ckpt), map_location=self.device)
            if "denoiser" in sd:
                sd = sd["denoiser"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            model.eval()
            self._model = model
            print(f"  [resemble-ft] loaded  (missing={len(missing)} unexpected={len(unexpected)})")
        except Exception as e:
            print(f"  [resemble-ft] load failed: {e}")

    def enhance(self, noisy: np.ndarray) -> np.ndarray:
        if self._model is None:
            return noisy
        try:
            wav = torch.from_numpy(noisy).unsqueeze(0)
            if SR != _RESEMBLE_SR:
                wav = torchaudio.functional.resample(wav, SR, _RESEMBLE_SR)
            wav = wav.to(self.device)
            with torch.no_grad():
                out = self._model(wav)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out.squeeze(0).cpu()
            if SR != _RESEMBLE_SR:
                out = torchaudio.functional.resample(out, _RESEMBLE_SR, SR)
            return out.numpy().astype(np.float32)
        except Exception as e:
            print(f"  [resemble-ft] inference error: {e}")
            return noisy


class MossFormerFTWrapper:
    """MossFormer2-SE-16kHz with optional fine-tuned weights patched in."""

    def __init__(self, device: torch.device, save_dir: str, ft_ckpt: Path | None = None):
        self.device = device
        self._model = None
        self._load(save_dir, ft_ckpt)

    def _load(self, save_dir: str, ft_ckpt: Path | None):
        import types as _types

        # Pre-import speechbrain.nnet sub-modules to bypass the k2_fsa lazy-import stub.
        try:
            import speechbrain.nnet.RNN    # noqa
            import speechbrain.nnet.linear # noqa
        except Exception:
            pass

        _k2_key = "speechbrain.integrations.k2_fsa"
        if _k2_key in sys.modules and not isinstance(sys.modules[_k2_key], _types.ModuleType):
            dummy = _types.ModuleType(_k2_key)
            dummy.__file__ = "<k2 not installed>"
            sys.modules[_k2_key] = dummy

        sb_device = (f"{self.device.type}:{self.device.index or 0}"
                     if self.device.type == "cuda" else "cpu")

        try:
            from speechbrain.inference.enhancement import SpectralMaskEnhancement
            self._model = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/MossFormer2-SE-16kHz",
                savedir=save_dir,
                run_opts={"device": sb_device},
            )
            print("  [mossformer] base model loaded")
        except Exception as e:
            print(f"  [mossformer] base model load failed: {e}")
            return

        if ft_ckpt and ft_ckpt.is_file():
            try:
                sd = torch.load(str(ft_ckpt), map_location=self.device)
                if "mods_state_dict" in sd:
                    sd = sd["mods_state_dict"]
                elif "mods" in sd:
                    sd = sd["mods"]
                missing, unexpected = self._model.mods.load_state_dict(sd, strict=False)
                self._model.mods.eval()
                print(f"  [mossformer-ft] FT weights loaded  (missing={len(missing)} unexpected={len(unexpected)})")
            except Exception as e:
                print(f"  [mossformer-ft] FT weights load failed: {e}")

    def enhance(self, noisy: np.ndarray) -> np.ndarray:
        if self._model is None:
            return noisy
        try:
            t   = torch.from_numpy(noisy).unsqueeze(0)
            out = self._model.enhance_batch(t, lengths=torch.ones(1)).squeeze(0)
            return out.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"  [mossformer-ft] inference error: {e}")
            return noisy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Denoise an MP3 through CRN, U-Net, Resemble, Resemble-FT, MossFormer-FT."
    )
    ap.add_argument("input",     help="Path to noisy audio file (.mp3 or any format torchaudio supports)")
    ap.add_argument("--out_dir", default="denoised_outputs", help="Output folder (default: denoised_outputs/)")
    args = ap.parse_args()

    inp     = Path(args.input)
    out_dir = Path(args.out_dir)
    stem    = inp.stem
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nInput : {inp.resolve()}")
    print(f"Device: {device}")
    print(f"OutDir: {out_dir.resolve()}\n")

    noisy = load_audio(inp)
    print(f"Loaded {len(noisy)/SR:.2f}s  ({len(noisy):,} samples @ {SR} Hz)\n")

    saved: list[Path] = []
    # Collect enhanced arrays for ensemble (keyed by model name)
    ensemble_pool: dict[str, np.ndarray] = {}

    def _run(label: str, fn, out_name: str) -> np.ndarray | None:
        """Run fn(), apply Wiener post-filter, save, and register for ensemble."""
        try:
            raw  = fn()
            filt = wiener_post_filter(raw)
            path = out_dir / out_name
            save_wav(filt, path)
            saved.append(path)
            ensemble_pool[label] = filt
            print(f"      ✓ {path}")
            return filt
        except Exception as e:
            print(f"      ✗ {e}")
            return None

    # ── 1. CRN ──────────────────────────────────────────────────────────────
    crn_ckpt = RUN_DIR / "checkpoints_crn" / "best.pt"
    print(f"[1/5] CRN  ← {crn_ckpt}")
    _run("CRN", lambda: CRNWrapper(crn_ckpt, device).enhance(noisy),
         f"{stem}_CRN.wav")

    # ── 2. U-Net ─────────────────────────────────────────────────────────────
    unet_ckpt = SCRIPT_DIR / "Colab_Ready_UNet" / "checkpoints_unet" / "best.pt"
    print(f"\n[2/5] U-Net  ← {unet_ckpt}")
    _run("UNet", lambda: UNetWrapper(unet_ckpt, device).enhance(noisy),
         f"{stem}_UNet.wav")

    # ── 3. Resemble (base pretrained) ────────────────────────────────────────
    print("\n[3/5] Resemble (base pretrained)")
    _run("Resemble", lambda: ResembleWrapper(device).enhance(noisy),
         f"{stem}_Resemble.wav")

    # ── 4. Resemble-FT ───────────────────────────────────────────────────────
    resemble_ft_ckpt = RUN_DIR / "benchmark_outputs" / "resemble_ft" / "denoiser_ft.pt"
    print(f"\n[4/5] Resemble-FT  ← {resemble_ft_ckpt}")
    _run("Resemble-FT", lambda: ResembleFTWrapper(resemble_ft_ckpt, device).enhance(noisy),
         f"{stem}_Resemble-FT.wav")

    # ── 5. MossFormer-FT ─────────────────────────────────────────────────────
    mf_save    = str(RUN_DIR / "pretrained_models" / "mossformer2")
    mf_ft_ckpt = RUN_DIR / "benchmark_outputs" / "mossformer_ft" / "mossformer_ft.pt"
    print(f"\n[5/5] MossFormer-FT  ← {mf_ft_ckpt}")
    _run("MossFormer-FT",
         lambda: MossFormerFTWrapper(device, save_dir=mf_save, ft_ckpt=mf_ft_ckpt).enhance(noisy),
         f"{stem}_MossFormer-FT.wav")

    # ── Ensemble ─────────────────────────────────────────────────────────────
    # Average all successful outputs.  Different architectures make different
    # errors; averaging reduces residual noise artefacts that slip past any
    # single model.  Uses the shorter of the two lengths to avoid padding bias.
    if len(ensemble_pool) >= 2:
        print("\n[+] Computing ensemble average …")
        try:
            min_len  = min(v.shape[0] for v in ensemble_pool.values())
            stack    = np.stack([v[:min_len] for v in ensemble_pool.values()], axis=0)
            ensemble = stack.mean(axis=0).astype(np.float32)
            # One more Wiener pass to clean up residual from averaging artifacts
            ensemble = wiener_post_filter(ensemble)
            path     = out_dir / f"{stem}_Ensemble.wav"
            save_wav(ensemble, path)
            saved.append(path)
            print(f"      ✓ {path}  (averaged {len(ensemble_pool)} models)")
        except Exception as e:
            print(f"      ✗ ensemble failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    n_model = len(saved) - (1 if len(ensemble_pool) >= 2 else 0)
    print("\n" + "─" * 60)
    print(f"Done — {n_model}/5 model outputs + ensemble → {out_dir.resolve()}/")
    for p in saved:
        print(f"  {p.name}")
    print("─" * 60)


if __name__ == "__main__":
    main()
