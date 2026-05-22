"""
benchmark.py
============
Unified benchmark: CRN | U-Net | Resemble Enhance | MossFormer2.

Produces
--------
  benchmark_outputs/
    summary_table.csv           — all models × all metrics
    summary_table.txt           — pretty-printed console table
    per_utt.csv                 — per-file scores for every model
    spectrograms/<src>_<stem>/
      comparison.png            — grid: noisy + all enhanced + clean
    audio/<src>_<stem>/
      noisy.wav / clean.wav / <model>.wav

Usage
-----
    cd C:\\Users\\rocha\\AI_builders\\Run
    python benchmark.py --data_root ./data_npy --max_files 200

    # include fine-tuned Resemble (after running finetune_resemble.py):
    python benchmark.py --resemble_ft_ckpt ./benchmark_outputs/resemble_ft/denoiser_ft.pt

Install notes
-------------
    pip install torchaudio speechbrain pesq speechmos jiwer pythainlp
    pip install -e C:\\Users\\rocha\\resemble-enhance   # or add --resemble_dir
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import pathlib
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Checkpoints saved on Linux pickle PosixPath objects; remap to WindowsPath on Windows.
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

import numpy as np
import torch
import torchaudio

ROOT     = Path(__file__).parent
CRN_DIR  = ROOT / "CRN"
UNET_DIR = ROOT / "U_net"

# ---------------------------------------------------------------------------
# Dynamic module loader — avoids sys.path pollution between CRN / U-Net
# ---------------------------------------------------------------------------

def _load_module(path: Path, name: str):
    """Load a Python file as a module without permanently polluting sys.path.

    Temporarily adds the file's parent directory so that relative imports
    inside the file (e.g. `from model_unet_advanced import ...`) resolve,
    then removes it again to avoid name clashes between CRN and speech_denoise
    modules that share identical filenames (dataset.py, model.py, etc.).
    """
    parent  = str(path.parent)
    added   = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod          # register so sub-imports find it
        spec.loader.exec_module(mod)
    finally:
        if added:
            sys.path.remove(parent)
    return mod


# ---------------------------------------------------------------------------
# SI-SDR (= SI-SNR)
# ---------------------------------------------------------------------------

_RESEMBLE_SR = 44_100   # Resemble denoiser's native sample rate


def si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha  = float(np.dot(est, ref) / (np.dot(ref, ref) + eps))
    target = alpha * ref
    noise  = est - target
    return float(10.0 * np.log10(
        (np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)))


# ---------------------------------------------------------------------------
# MOS scorer
# ---------------------------------------------------------------------------

class MOSScorer:
    def __init__(self, sr: int = 16_000):
        self.sr   = sr
        self.kind = "snr_proxy"
        self._fn: Optional[object] = None
        try:
            from speechmos import dnsmos
            self._fn  = lambda w: float(dnsmos.run(w, sr=self.sr)["ovrl_mos"])
            self.kind = "dnsmos"
            return
        except Exception:
            pass
        try:
            from pesq import pesq as _pesq
            self._fn  = lambda ref, deg: float(_pesq(self.sr, ref, deg, "wb"))
            self.kind = "pesq_wb"
        except Exception:
            pass

    def score(self, ref: np.ndarray, est: np.ndarray) -> float:
        if self.kind == "dnsmos":
            return self._fn(est)
        if self.kind == "pesq_wb":
            try:
                return self._fn(ref, est)
            except Exception:
                pass
        return float(np.clip(2.0 + si_sdr(ref, est) / 10.0, 1.0, 5.0))


# ---------------------------------------------------------------------------
# WER / CER
# ---------------------------------------------------------------------------

_HAVE_JIWER     = False
_HAVE_PYTHAINLP = False
try:
    import jiwer as _jiwer
    _HAVE_JIWER = True
except ImportError:
    pass
try:
    import pythainlp as _pythainlp
    _HAVE_PYTHAINLP = True
except ImportError:
    pass


def _tok(text: str) -> List[str]:
    if _HAVE_PYTHAINLP:
        from pythainlp import word_tokenize
        return [t for t in word_tokenize(text.strip(), engine="newmm",
                                          keep_whitespace=False) if t.strip()]
    return text.strip().split()


def _edit(a, b):
    m, n = len(a), len(b)
    if not m: return n
    if not n: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (0 if a[i-1]==b[j-1] else 1))
            prev = tmp
    return dp[n]


def score_wer_cer(ref: str, hyp: str) -> Tuple[float, float]:
    rt = _tok(ref); ht = _tok(hyp)
    rs = " ".join(rt); hs = " ".join(ht)
    if _HAVE_JIWER:
        return float(_jiwer.wer(rs, hs)), float(_jiwer.cer(ref.strip(), hyp.strip()))
    wer = (_edit(rt, ht) / len(rt)) if rt else (0.0 if not ht else 1.0)
    rc  = list(ref.strip()); hc = list(hyp.strip())
    cer = (_edit(rc, hc) / len(rc)) if rc else (0.0 if not hc else 1.0)
    return wer, cer


def _prefix_edit_rate(h: list, r: list) -> float:
    """Edit rate vs best-matching prefix of r; ignores trailing reference tokens."""
    m, n = len(h), len(r)
    if m == 0: return 0.0
    if n == 0: return 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] if h[i-1] == r[j-1] \
                       else 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    j_star = min(range(1, n + 1), key=lambda j: dp[m][j])
    return dp[m][j_star] / j_star


def _prefix_wer_cer(ref: str, hyp: str) -> Tuple[float, float]:
    """WER/CER ignoring trailing reference tokens not covered by the hypothesis.

    Handles the case where audio is cropped (e.g. 4 s) but the ground-truth
    label covers the full utterance (e.g. 10 s): trailing deletions in the
    reference are free, so only the matched prefix is penalised.
    """
    return (_prefix_edit_rate(_tok(ref), _tok(hyp)),
            _prefix_edit_rate(list(ref.strip()), list(hyp.strip())))


class WERScorer:
    def __init__(self, model_name: str = "openai/whisper-small",
                 sr: int = 16_000, language: str = "thai"):
        self.kind = "disabled"
        self.sr   = sr
        self.lang = language if language.lower() != "auto" else None
        self._pipe = None
        self._wh   = None
        try:
            from transformers import pipeline as _hfp
            # Always run on CPU — denoising models (Resemble, MossFormer) need full GPU VRAM.
            for _dev in ["cpu"]:
                try:
                    self._pipe = _hfp("automatic-speech-recognition", model=model_name,
                                      device=_dev, chunk_length_s=30, stride_length_s=5)
                    break
                except Exception as _e:
                    print(f"[WERScorer] {_dev} failed ({_e}), trying next …")
            if self._pipe is not None:
                self.kind = f"hf:{model_name}"
                print(f"[WERScorer] loaded {model_name} on {_dev}")
                return
        except Exception as e:
            print(f"[WERScorer] transformers pipeline unavailable: {e}")
        try:
            import whisper as _w
            self._wh  = _w.load_model(model_name.split("/")[-1].replace("whisper-",""), device="cpu")
            self.kind = f"openai-whisper"
        except Exception as e:
            print(f"[WERScorer] openai-whisper unavailable: {e}")

    def transcribe(self, wav: np.ndarray) -> Optional[str]:
        if self.kind == "disabled":
            return None
        gkw = {"task": "transcribe"}
        if self.lang:
            gkw["language"] = self.lang
        try:
            if self._pipe:
                return self._pipe({"array": wav.astype(np.float32),
                                   "sampling_rate": self.sr},
                                  generate_kwargs=gkw)["text"]
            if self._wh:
                return self._wh.transcribe(wav.astype(np.float32),
                                           fp16=False, verbose=False,
                                           language=self.lang)["text"]
        except Exception as e:
            print(f"[benchmark]   ASR error: {e}")
        return None


# ---------------------------------------------------------------------------
# Model wrappers  — all return np.ndarray (float32, 16 kHz)
# ---------------------------------------------------------------------------

class CRNWrapper:
    name = "CRN"

    def __init__(self, ckpt: Path, device: torch.device):
        mod           = _load_module(CRN_DIR / "inference.py", "crn_inference")
        self.model, _ = mod.load_model(str(ckpt), device)
        self._enhance = mod.enhance_waveform
        self.device   = device
        self.load_ok  = True

    def enhance(self, noisy: np.ndarray, sr: int) -> np.ndarray:
        return self._enhance(self.model, noisy, self.device, sr=sr)


class UNetWrapper:
    name = "U-Net"

    def __init__(self, ckpt: Path, device: torch.device):
        mod           = _load_module(UNET_DIR / "inference.py", "unet_inference")
        self.model, _ = mod.load_model(str(ckpt), device)
        self._enhance = mod.enhance_waveform
        self.device   = device
        self.load_ok  = True

    def enhance(self, noisy: np.ndarray, sr: int) -> np.ndarray:
        return self._enhance(self.model, noisy, self.device, sr=sr)


class ResembleWrapper:
    """Resemble Enhance wrapper.  Uses denoise() for fair comparison."""

    def __init__(self, device: torch.device, run_dir: Optional[Path] = None,
                 ft_ckpt: Optional[Path] = None, label: str = "Resemble"):
        self.device       = device
        self.run_dir      = run_dir
        self.ft_ckpt      = ft_ckpt
        self.name         = label
        self.load_ok      = False
        self._enhance_fn  = None
        self._denoise_fn  = None
        self._ft_denoiser = None   # populated by _load_ft_denoiser if ft_ckpt given
        self._load()

    def _load(self):
        try:
            from resemble_enhance.enhancer.inference import enhance, denoise
            self._enhance_fn = enhance
            self._denoise_fn = denoise
            self.load_ok     = True
            print(f"[benchmark] {self.name}: Resemble Enhance loaded")
            if self.ft_ckpt and self.ft_ckpt.is_file():
                self._load_ft_denoiser()
        except ImportError as e:
            print(f"[benchmark] {self.name}: resemble_enhance not importable — {e}")
            print("  Install: pip install -e <resemble-enhance-dir>  "
                  "or add --resemble_dir to benchmark args")

    def _load_ft_denoiser(self):
        """Load fine-tuned denoiser for direct inference (bypasses module-level denoise())."""
        try:
            from resemble_enhance.denoiser.inference import load_denoiser
            ft_denoiser = load_denoiser(self.run_dir, self.device)
            sd = torch.load(str(self.ft_ckpt), map_location=self.device)
            if "denoiser" in sd:
                sd = sd["denoiser"]
            missing, unexpected = ft_denoiser.load_state_dict(sd, strict=False)
            ft_denoiser.eval()
            self._ft_denoiser = ft_denoiser
            print(f"[benchmark] {self.name}: FT denoiser loaded from {self.ft_ckpt} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})")
        except Exception as e:
            print(f"[benchmark] {self.name}: could not load FT denoiser -- {e}; "
                  f"falling back to base Resemble weights")

    def enhance(self, noisy: np.ndarray, sr: int) -> np.ndarray:
        # FT path: call the fine-tuned denoiser directly
        if self._ft_denoiser is not None:
            try:
                wav = torch.from_numpy(noisy.astype(np.float32))
                # Resemble denoiser operates at 44.1 kHz; resample if source differs
                if sr != _RESEMBLE_SR:
                    wav = torchaudio.functional.resample(wav, sr, _RESEMBLE_SR)
                wav = wav.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self._ft_denoiser(wav)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                out = out.squeeze(0).cpu()
                if sr != _RESEMBLE_SR:
                    out = torchaudio.functional.resample(out, _RESEMBLE_SR, sr)
                return out.numpy().astype(np.float32)
            except Exception as e:
                print(f"[benchmark]   {self.name} FT inference error: {e}")
                return noisy

        # Base path: use module-level denoise()
        # Force CPU: CUDA path triggers weight-download failure on Windows when
        # model.pt is absent from model_repo cache, silently returning noisy audio.
        if self._denoise_fn is None:
            return noisy
        _cpu = torch.device("cpu")
        try:
            wav = torch.from_numpy(noisy.astype(np.float32))
            out, out_sr = self._denoise_fn(wav, sr, _cpu, run_dir=self.run_dir)
            out_np = out.cpu().numpy().squeeze()
            if out_sr != sr:
                t  = torch.from_numpy(out_np).unsqueeze(0)
                t  = torchaudio.functional.resample(t, out_sr, sr)
                out_np = t.squeeze(0).numpy()
            return out_np.astype(np.float32)
        except Exception as e:
            print(f"[benchmark]   {self.name} denoise() error: {e}")
            return noisy


class MossFormerWrapper:
    """MossFormer2-SE-16kHz via speechbrain, with MetricGAN+ fallback."""

    name = "MossFormer"

    def __init__(self, device: torch.device,
                 save_dir: str = "pretrained_models/mossformer2",
                 ft_ckpt: Optional[Path] = None,
                 label: str = "MossFormer",
                 source: Optional[str] = None):
        self.device      = device
        self.ft_ckpt     = ft_ckpt
        self.name        = label
        self.load_ok     = False
        self._model      = None
        self._model_type = "spectral"
        self._load(save_dir, source=source)

    def _load(self, save_dir: str, source: Optional[str] = None):
        import types as _types

        # Pre-import speechbrain.nnet sub-modules before from_hparams() runs.
        # SpeechBrain 1.x lazy-loads nnet via a k2_fsa integration stub; accessing
        # sb.nnet.RNN inside a HyperPyYAML file triggers the lazy-import chain and
        # raises ImportError (k2 not installed). Explicit import bypasses the stub.
        try:
            import speechbrain.nnet.RNN     # noqa: F401
            import speechbrain.nnet.linear  # noqa: F401
        except Exception:
            pass

        # Replace the k2_fsa LazyModule with a real dummy so Python's inspect
        # machinery (triggered by torch._dynamo) doesn't fire __getattr__ → ImportError.
        _k2_key = "speechbrain.integrations.k2_fsa"
        if _k2_key in sys.modules and not isinstance(sys.modules[_k2_key], _types.ModuleType):
            _dummy = _types.ModuleType(_k2_key)
            _dummy.__file__ = "<k2 not installed>"
            sys.modules[_k2_key] = _dummy

        # SpeechBrain requires "cuda:N" — str(torch.device("cuda")) gives bare "cuda"
        sb_device = (f"{self.device.type}:{self.device.index or 0}"
                     if self.device.type == "cuda" else "cpu")

        candidates = []
        if source:
            candidates.append((source, "custom"))
        candidates += [
            ("speechbrain/MossFormer2-SE-16kHz",    "MossFormer2"),
            ("speechbrain/metricgan-plus-voicebank", "MetricGAN+"),
        ]
        for src, label in candidates:
            slug = label.replace(" ", "_").replace("+", "plus")
            try:
                from speechbrain.inference.enhancement import SpectralMaskEnhancement
                self._model = SpectralMaskEnhancement.from_hparams(
                    source=src,
                    savedir=os.path.join(save_dir, slug),
                    run_opts={"device": sb_device},
                )
                if self.name == "MossFormer":
                    self.name = label
                self.load_ok = True
                print(f"[benchmark] MossFormer slot: loaded '{label}' from {src}")
                if self.ft_ckpt and self.ft_ckpt.is_file():
                    self._load_ft_weights()
                return
            except Exception as e:
                print(f"[benchmark] MossFormer: {src} failed -> {e}")
        print("[benchmark] ERROR: MossFormer — all candidates failed. "
              "Column will show FAILED in table.\n"
              "  Fix: huggingface-cli login   OR   --mossformer_source <local-dir>")

    def _load_ft_weights(self):
        """Patch mods with fine-tuned weights from finetune_mossformer.py."""
        try:
            sd = torch.load(str(self.ft_ckpt), map_location=self.device)
            # finetune_mossformer.py saves under "mods_state_dict" key
            if "mods_state_dict" in sd:
                sd = sd["mods_state_dict"]
            elif "mods" in sd:
                sd = sd["mods"]
            # else assume sd is already a raw state dict
            missing, unexpected = self._model.mods.load_state_dict(sd, strict=False)
            self._model.mods.eval()
            print(f"[benchmark] {self.name}: FT weights loaded from {self.ft_ckpt} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})")
        except Exception as e:
            print(f"[benchmark] {self.name}: could not load fine-tuned weights — {e}")

    def enhance(self, noisy: np.ndarray, sr: int) -> np.ndarray:
        if self._model is None:
            return noisy
        try:
            t   = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0)
            out = self._model.enhance_batch(t, lengths=torch.ones(1)).squeeze(0)
            result = out.cpu().numpy().astype(np.float32)
            if result.size < 100:
                print(f"[benchmark]   {self.name} output suspiciously small ({result.shape}); returning noisy")
                return noisy
            return result
        except Exception as e:
            print(f"[benchmark]   MossFormer inference error: {e}")
            return noisy


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_SOURCE_PATTERNS: List[Tuple[str, List[str]]] = [
    ("commonvoice", ["commonvoice"]),
    ("fleurs",      ["fleurs", "feulr"]),
    ("thai_isan",   ["thai_isan"]),
    ("thai_elder",  ["thai_elder", "elder"]),
]


def _detect_source(name: str) -> Optional[str]:
    n = name.lower()
    for src, pats in _SOURCE_PATTERNS:
        if any(p in n for p in pats):
            return src
    return None


def _read_labels(csv_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not csv_path.is_file():
        return out
    import csv as _csv
    _FNAME = {"file_name", "filename", "file", "name"}
    _TEXT  = {"transcription_cleaned", "transcription", "transcript", "text", "label"}
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = _csv.DictReader(fh)
        fields = reader.fieldnames or []
        fc = next((f for f in fields if f.lower().strip() in _FNAME), None)
        tc = next((f for f in fields if f.lower().strip() in _TEXT), None)
        for row in reader:
            if fc:
                fname = row[fc].strip()
                text  = (row[tc].strip() if tc and row.get(tc) else "")
                if fname:
                    out[fname] = text
    return out


def _crop_or_pad(x: np.ndarray, n: int) -> np.ndarray:
    if x.size >= n:
        return x[:n]
    return np.pad(x, (0, n - x.size)).astype(np.float32)


def _mix(clean: np.ndarray, noise: np.ndarray, snr_db: float):
    eps = 1e-8
    cr  = float(np.sqrt(np.mean(clean ** 2) + eps))
    nr  = float(np.sqrt(np.mean(noise ** 2) + eps))
    a   = cr / (nr * (10.0 ** (snr_db / 20.0)) + eps)
    return (clean + a * noise).astype(np.float32)


def _is_lf_noise(arr: np.ndarray, sr: int = 16_000,
                 cutoff_hz: float = 200.0, threshold: float = 0.70) -> bool:
    """True when ≥threshold fraction of noise energy is below cutoff_hz."""
    spec  = np.abs(np.fft.rfft(arr)) ** 2
    freqs = np.fft.rfftfreq(len(arr), d=1.0 / sr)
    return float(spec[freqs < cutoff_hz].sum()) / (float(spec.sum()) + 1e-8) >= threshold


def load_test_files(data_root: Path, split: str, max_files: int, seed: int = 42):
    split_dir   = data_root / split
    all_clean   = sorted((split_dir / "clean").glob("*.npy"))
    noise_files = sorted((split_dir / "noise").glob("*.npy"))
    labels      = _read_labels(split_dir / "labels_npy.csv")

    src_keys = [s for s, _ in _SOURCE_PATTERNS]
    groups: Dict[str, List[Path]] = {s: [] for s in src_keys}
    for cf in all_clean:
        src = _detect_source(cf.name)
        if src in groups:
            groups[src].append(cf)

    n_grp      = sum(1 for s in src_keys if groups[s])
    per_source = max_files // max(1, n_grp) if n_grp else max_files

    random.seed(seed)
    chosen: List[Path] = []
    for src in src_keys:
        pool   = groups[src]
        sample = random.sample(pool, min(per_source, len(pool)))
        chosen.extend(sample)
        print(f"[benchmark]   {src:14s}: {len(pool):5d} -> {len(sample):3d} selected")

    # Fallback: if no file matched any source pattern, sample uniformly from all files
    if not chosen:
        print(f"[benchmark] WARNING: no files matched source patterns "
              f"({', '.join(p for _, ps in _SOURCE_PATTERNS for p in ps)}). "
              f"Falling back to random sample of all {len(all_clean)} files.")
        chosen = random.sample(all_clean, min(max_files, len(all_clean)))

    random.shuffle(chosen)
    print(f"[benchmark] total evaluation files: {len(chosen)}")
    return chosen, noise_files, labels


# ---------------------------------------------------------------------------
# Spectrogram helpers
# ---------------------------------------------------------------------------

def _log_spec(wav: np.ndarray, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    t   = torch.from_numpy(wav.astype(np.float32))
    win = torch.hann_window(n_fft)
    s   = torch.stft(t, n_fft=n_fft, hop_length=hop,
                     window=win, return_complex=True, center=True)
    return torch.log1p(s.abs()).numpy()


def save_comparison_spectrogram(panels: List[Tuple[str, np.ndarray]],
                                 out_path: Path,
                                 title: str = "") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (label, wav) in zip(axes, panels):
        img = ax.imshow(_log_spec(wav), aspect="auto", origin="lower",
                        cmap="magma", vmin=0)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Freq bin")
        fig.colorbar(img, ax=ax, fraction=0.02, pad=0.02)
    axes[-1].set_xlabel("Time frame")
    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=90, bbox_inches="tight")
    plt.close(fig)


def save_wav(wav: np.ndarray, path: Path, sr: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav_f32 = np.clip(wav, -1.0, 1.0).astype(np.float32)
    try:
        import soundfile as _sf
        _sf.write(str(path), wav_f32, sr)
        return
    except Exception as e:
        print(f"[benchmark]   wav save failed ({path.name}): {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",       default="../data_npy")
    ap.add_argument("--split",           default="test")
    ap.add_argument("--crn_ckpt",        default="./checkpoints_crn/best.pt")
    ap.add_argument("--unet_ckpt",       default="../Colab_Ready_UNet/checkpoints_unet/best.pt")
    ap.add_argument("--resemble_dir",    default=str(Path.home() / "resemble-enhance"),
                    help="Path to the cloned resemble-enhance repo "
                         "(added to sys.path for import).")
    ap.add_argument("--resemble_ft_ckpt",
                    default="./benchmark_outputs/resemble_ft/denoiser_ft.pt",
                    help="Fine-tuned Resemble denoiser checkpoint "
                         "(from finetune_resemble.py). Empty = skip.")
    ap.add_argument("--mossformer_ft_ckpt",
                    default="./benchmark_outputs/mossformer_ft/mossformer_ft.pt",
                    help="Fine-tuned MossFormer mods checkpoint "
                         "(from finetune_mossformer.py). Empty = skip.")
    ap.add_argument("--no_mossformer",   action="store_true",
                    help="Skip the MossFormer column (saves time if not installed).")
    ap.add_argument("--mossformer_savedir",
                    default="./pretrained_models/mossformer2",
                    help="Where SpeechBrain caches / looks for MossFormer weights.")
    ap.add_argument("--mossformer_source", default="",
                    help="Local dir or HuggingFace model-id for MossFormer "
                         "(overrides default candidates). Leave empty to auto-select.")
    ap.add_argument("--max_files",       type=int, default=200)
    ap.add_argument("--snr",             type=float, default=5.0)
    ap.add_argument("--sample_rate",     type=int, default=16_000)
    ap.add_argument("--whisper_model",   default="openai/whisper-small")
    ap.add_argument("--language",        default="thai")
    ap.add_argument("--out_dir",         default="./benchmark_outputs")
    ap.add_argument("--save_audio",      action="store_true", default=True)
    ap.add_argument("--save_specs",      action="store_true", default=True)
    ap.add_argument("--seed",            type=int, default=42)
    return ap.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[benchmark] device={device}  split={args.split}  snr={args.snr} dB")

    # ---- Add resemble-enhance to import path ----
    resemble_dir = Path(args.resemble_dir)
    if resemble_dir.is_dir() and str(resemble_dir) not in sys.path:
        sys.path.insert(0, str(resemble_dir))
        print(f"[benchmark] added to sys.path: {resemble_dir}")

    # ---- Scorers (init before denoising models to claim GPU memory early) ----
    mos_scorer = MOSScorer(sr=args.sample_rate)
    lang       = args.language if args.language.lower() != "auto" else None
    wer_scorer = WERScorer(args.whisper_model, sr=args.sample_rate, language=lang or "auto")
    print(f"[benchmark] MOS={mos_scorer.kind}  ASR={wer_scorer.kind}")

    # ---- Load models ----
    wrappers = []

    crn_ckpt = Path(args.crn_ckpt)
    if crn_ckpt.is_file():
        try:
            wrappers.append(CRNWrapper(crn_ckpt, device))
            print(f"[benchmark] CRN loaded from {crn_ckpt}")
        except Exception as e:
            print(f"[benchmark] CRN load failed: {e}")
    else:
        print(f"[benchmark] ERROR: CRN checkpoint not found: {crn_ckpt}")

    # U-Net: primary path (Colab weights), fallback to Run/U_net/checkpoints/
    unet_ckpt = Path(args.unet_ckpt)
    if not unet_ckpt.is_file():
        _fb = ROOT / "U_net" / "checkpoints" / "best.pt"
        if _fb.is_file():
            print(f"[benchmark] U-Net primary not found; using fallback: {_fb}")
            unet_ckpt = _fb
    if unet_ckpt.is_file():
        try:
            wrappers.append(UNetWrapper(unet_ckpt, device))
            print(f"[benchmark] U-Net loaded from {unet_ckpt}")
        except Exception as e:
            print(f"[benchmark] U-Net load failed: {e}")
    else:
        print(f"[benchmark] ERROR: U-Net checkpoint not found at {args.unet_ckpt}")

    ft_ckpt = Path(args.resemble_ft_ckpt) if args.resemble_ft_ckpt else None
    wrappers.append(ResembleWrapper(device, label="Resemble"))
    if ft_ckpt and ft_ckpt.is_file():
        wrappers.append(ResembleWrapper(device, ft_ckpt=ft_ckpt, label="Resemble-FT"))

    if not args.no_mossformer:
        mf_save   = args.mossformer_savedir
        mf_source = args.mossformer_source or None
        wrappers.append(MossFormerWrapper(device, save_dir=mf_save, source=mf_source))
        mf_ft = Path(args.mossformer_ft_ckpt) if args.mossformer_ft_ckpt else None
        if mf_ft and mf_ft.is_file():
            wrappers.append(MossFormerWrapper(device, save_dir=mf_save,
                                              ft_ckpt=mf_ft, source=mf_source,
                                              label="MossFormer-FT"))

    failed_models = {w.name for w in wrappers if not w.load_ok}
    if failed_models:
        print(f"[benchmark] LOAD FAILED (will show FAILED in table): {sorted(failed_models)}")

    model_names = ["Noisy"] + [w.name for w in wrappers]
    print(f"[benchmark] models: {model_names}")


    # ---- Test data ----
    data_root = Path(args.data_root)
    print(f"[benchmark] loading test files (max {args.max_files}) …")
    clean_files, noise_files, labels = load_test_files(
        data_root, args.split, args.max_files, args.seed)

    rng = np.random.default_rng(args.seed)

    # accumulators: {model_name: {metric: sum, n: count, n_wer: count}}
    def _new_sums():
        return {nm: {"mos": 0.0, "sisdr": 0.0, "wer": 0.0, "cer": 0.0, "n": 0, "n_wer": 0}
                for nm in model_names}
    sums     = _new_sums()   # all noise combined
    gen_sums = _new_sums()   # general / broadband noise
    lf_sums  = _new_sums()   # LF-dominant noise (<200 Hz)

    rows: List[Dict] = []
    # examples_saved: save one audio+spectrogram per dataset source
    saved_srcs: set = set()

    for i, cf in enumerate(clean_files):
        ref_text = labels.get(cf.name, "")
        clean = np.load(cf).astype(np.float32).squeeze()
        if clean.ndim > 1:
            clean = clean.mean(axis=0)

        is_lf = False
        if noise_files:
            nf    = noise_files[int(rng.integers(0, len(noise_files)))]
            noise = np.load(nf).astype(np.float32).squeeze()
            if noise.ndim > 1:
                noise = noise.mean(axis=0)
            noise = _crop_or_pad(noise, clean.size)
            is_lf = _is_lf_noise(noise, args.sample_rate)
        else:
            noise = rng.standard_normal(clean.size).astype(np.float32)
        noisy = _mix(clean, noise, args.snr)

        row: Dict = {
            "file":      cf.name,
            "source":    _detect_source(cf.name) or "unknown",
            "snr_in_db": args.snr,
        }

        # enhanced outputs: {model_name: np.ndarray}
        outputs: Dict[str, np.ndarray] = {"Noisy": noisy}
        for w in wrappers:
            try:
                enh = w.enhance(noisy, args.sample_rate)
                outputs[w.name] = enh
                # Warn if output is identical to noisy (silent fallback indicator)
                if i < 3 and np.allclose(enh[:min(len(enh), len(noisy))],
                                          noisy[:min(len(enh), len(noisy))], atol=1e-6):
                    print(f"[benchmark]   WARNING {w.name}: output == noisy (silent fallback?)")
            except Exception as e:
                print(f"[benchmark]   {w.name} inference error on {cf.name}: {e}")
                outputs[w.name] = noisy

        # ---- first-file diagnostics ----
        if i == 0:
            print("[benchmark] --- first-file output stats ---")
            for nm, est in outputs.items():
                print(f"  {nm:20s}: len={len(est):6d}  mean={float(np.mean(est)):+.4f}  "
                      f"std={float(np.std(est)):.4f}  max={float(np.abs(est).max()):.4f}  "
                      f"SI-SNR={si_sdr(clean, _crop_or_pad(est, clean.size)):+.2f}")

        # ---- metrics ----
        transcriptions: Dict[str, Optional[str]] = {}
        for nm, est in outputs.items():
            est_c = _crop_or_pad(est, clean.size)
            m  = mos_scorer.score(clean, est_c)
            sd = si_sdr(clean, est_c)
            row[f"{nm}_mos"]   = round(m,  4)
            row[f"{nm}_sisdr"] = round(sd, 4)
            for _s in (sums, lf_sums if is_lf else gen_sums):
                _s[nm]["mos"]   += m
                _s[nm]["sisdr"] += sd
                _s[nm]["n"]     += 1

            if ref_text and wer_scorer.kind != "disabled":
                hyp = wer_scorer.transcribe(est_c)
                transcriptions[nm] = hyp
                if hyp is not None:
                    wer, cer = _prefix_wer_cer(ref_text, hyp)
                    row[f"{nm}_wer"] = round(wer, 4)
                    row[f"{nm}_cer"] = round(cer, 4)
                    for _s in (sums, lf_sums if is_lf else gen_sums):
                        _s[nm]["wer"]   += wer
                        _s[nm]["cer"]   += cer
                        _s[nm]["n_wer"] += 1

        rows.append(row)

        # ---- print progress ----
        noisy_s = row.get("Noisy_sisdr", float("nan"))
        for w in wrappers:
            print(f" [{i+1:3d}/{len(clean_files)}] {cf.name}  "
                  f"noisy_SI-SNR={noisy_s:+.1f}  "
                  f"{w.name}_SI-SNR={row.get(f'{w.name}_sisdr', float('nan')):+.1f}  "
                  f"{w.name}_MOS={row.get(f'{w.name}_mos', float('nan')):.3f}")
            break  # print once per file (first model only for brevity)

        # ---- save audio + spectrogram (one example per source) ----
        src = _detect_source(cf.name) or "unknown"
        if src not in saved_srcs:
            saved_srcs.add(src)
            stem = f"{src}_{Path(cf.name).stem}"

            if args.save_audio:
                audio_dir = out / "audio" / stem
                save_wav(clean, audio_dir / "clean.wav", args.sample_rate)
                save_wav(noisy, audio_dir / "noisy.wav", args.sample_rate)
                for nm, est in outputs.items():
                    if nm == "Noisy":
                        continue
                    save_wav(est, audio_dir / f"{nm.replace(' ', '_')}.wav",
                             args.sample_rate)

            if args.save_specs:
                panels = [("Noisy Input", noisy)]
                for nm, est in outputs.items():
                    if nm == "Noisy":
                        continue
                    panels.append((nm, _crop_or_pad(est, len(clean))))
                panels.append(("Clean (Reference)", clean))
                save_comparison_spectrogram(
                    panels,
                    out / "spectrograms" / stem / "comparison.png",
                    title=f"Source: {src}  |  {cf.name}",
                )
                print(f"[benchmark] spectrogram saved -> spectrograms/{stem}/comparison.png")

    # ---- per-utterance CSV ----
    per_utt_path = out / "per_utt.csv"
    with open(per_utt_path, "w", newline="", encoding="utf-8") as fh:
        if rows:
            keys = sorted({k for r in rows for k in r})
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"[benchmark] per-utterance CSV -> {per_utt_path}")

    # ---- summary table ----
    _build_summary(sums, model_names, mos_scorer.kind, wer_scorer.kind, out,
                   gen_sums=gen_sums, lf_sums=lf_sums, failed_models=failed_models)


def _build_summary(sums, model_names, mos_kind, asr_kind, out: Path,
                   gen_sums=None, lf_sums=None, failed_models=None):
    failed_models = failed_models or set()
    cols = ["Model", "SNR_before", "SNR_after", "SNR_improve", "WER", "CER"]

    def _table_rows(s):
        noisy_d      = s["Noisy"]
        baseline_snr = noisy_d["sisdr"] / max(1, noisy_d["n"])
        rows = []
        for nm in model_names:
            d = s[nm]; n, nw = max(1, d["n"]), d["n_wer"]
            if nm == "Noisy":
                rows.append({
                    "Model":       "Raw",
                    "SNR_before":  f"{baseline_snr:+.2f}",
                    "SNR_after":   f"{baseline_snr:+.2f}",
                    "SNR_improve": "+0.00",
                    "WER":         f"{d['wer']/nw:.3f}" if nw else "n/a",
                    "CER":         f"{d['cer']/nw:.3f}" if nw else "n/a",
                })
            elif nm in failed_models:
                rows.append({
                    "Model":       nm,
                    "SNR_before":  f"{baseline_snr:+.2f}",
                    "SNR_after":   "FAILED",
                    "SNR_improve": "FAILED",
                    "WER":         "FAILED",
                    "CER":         "FAILED",
                })
            else:
                snr_after   = d["sisdr"] / n
                snr_improve = snr_after - baseline_snr
                rows.append({
                    "Model":       nm,
                    "SNR_before":  f"{baseline_snr:+.2f}",
                    "SNR_after":   f"{snr_after:+.2f}",
                    "SNR_improve": f"{snr_improve:+.2f}",
                    "WER":         f"{d['wer']/nw:.3f}" if nw else "n/a",
                    "CER":         f"{d['cer']/nw:.3f}" if nw else "n/a",
                })
        return rows

    def _render(rows, title):
        widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in cols}
        sep = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"
        hdr = "|" + "|".join(f" {c:<{widths[c]}} " for c in cols) + "|"
        lines = [f"\n=== {title} ===", sep, hdr, sep]
        for r in rows:
            lines.append("|" + "|".join(f" {r[c]:<{widths[c]}} " for c in cols) + "|")
        lines.append(sep)
        return "\n".join(lines)

    def _save_csv(path, rows, extra=None):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fields = list(rows[0].keys()) + list(extra or {})
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({**r, **(extra or {})})

    txt_parts = []
    for title, s in (("Benchmark Summary — All Noise",  sums),
                     ("Split: General Noise",            gen_sums),
                     ("Split: LF Noise (<200 Hz)",       lf_sums)):
        if s is None:
            continue
        if s is not sums and not any(s[nm]["n"] > 0 for nm in model_names):
            continue
        rows = _table_rows(s)
        block = _render(rows, title)
        print(block)
        txt_parts.append(block)
        if s is sums:
            _save_csv(out / "summary_table.csv", rows,
                      {"MOS_kind": mos_kind, "ASR_kind": asr_kind})
        else:
            tag = "gen" if "General" in title else "lf"
            _save_csv(out / f"summary_{tag}.csv", rows)

    footer = f"\n  MOS: {mos_kind}  |  ASR: {asr_kind}  |  WER: prefix-tolerant"
    print(footer)
    txt_path = out / "summary_table.txt"
    txt_path.write_text("\n".join(txt_parts) + footer + "\n", encoding="utf-8")
    print(f"\n[benchmark] summary CSV  -> {out / 'summary_table.csv'}")
    print(f"[benchmark] summary text -> {txt_path}")


if __name__ == "__main__":
    main()
