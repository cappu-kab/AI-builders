"""
benchmark.py
============
6-model + noisy-baseline speech denoising comparison.

Systems evaluated
-----------------
    0. Noisy input       — lower bound (no processing)
    1. CRN               — our trained CRN from Run/CRN/
    2. U-Net             — our trained AdvancedUNetSE
    3. Resemble (orig)   — pre-trained Resemble Enhance denoiser (44.1 kHz)
    4. Resemble (ft)     — fine-tuned on our LF-noise data
    5. MossFormer2 (orig)— pre-trained MossFormer2 / modelscope
    6. MossFormer2 (ft)  — fine-tuned on our LF-noise data

Metrics (averaged over --max_files test utterances)
----------------------------------------------------
    SI-SDR (dB)  —  scale-invariant SDR
    MOS          —  DNSMOS (neural) → PESQ (reference) → SNR proxy
    WER / CER    —  via Whisper + pythainlp (Thai-aware)

Outputs
-------
    <out_dir>/summary.csv           — comparison table
    <out_dir>/per_utt.csv           — per-file metrics
    <out_dir>/spectrograms/         — 8-panel PNGs (one per dataset source)

Usage
-----
    cd Run\\speech_denoise
    ..\\..\\venv\\Scripts\\python benchmark.py ^
        --data_root      ../../data_npy ^
        --crn_ckpt       ../CRN/checkpoints/best.pt ^
        --unet_ckpt      checkpoints/best.pt ^
        --resemble_ft    resemble_finetuned/best.pt ^
        --mossformer_ft  mossformer_finetuned/best.pt ^
        --max_files      200 ^
        --out_dir        benchmark_results
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── local imports (speech_denoise is the working directory) ─────────────────
from dataset import _crop_or_pad, mix_at_snr, _read_labels, _SOURCE_PATTERNS, _detect_source
from inference import load_model as _unet_load, enhance_waveform as _unet_enhance

SCRIPT_DIR = Path(__file__).parent
CRN_DIR    = SCRIPT_DIR.parent / "CRN"
SR         = 16_000       # all metrics computed at 16 kHz

EnhanceFn  = Callable[[np.ndarray], np.ndarray]   # noisy → enhanced (16 kHz)


# ============================================================================
# Metric helpers
# ============================================================================

def si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha  = float(np.dot(est, ref) / (np.dot(ref, ref) + eps))
    target = alpha * ref
    noise  = est - target
    return float(10.0 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)))


class MOSScorer:
    def __init__(self, sr: int = SR):
        self.sr, self.kind, self._fn = sr, "snr_proxy", None
        try:
            from speechmos import dnsmos
            self._fn   = lambda wav: float(dnsmos.run(wav, sr=self.sr)["ovrl_mos"])
            self.kind  = "dnsmos"
            return
        except Exception:
            pass
        try:
            from pesq import pesq as _pesq
            self._fn  = lambda ref, deg: float(_pesq(self.sr, ref, deg, "wb"))
            self.kind = "pesq_wb"
        except Exception:
            pass

    def score(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        if self.kind == "dnsmos":
            try: return self._fn(estimate)
            except Exception: pass
        if self.kind == "pesq_wb":
            try: return self._fn(reference, estimate)
            except Exception: pass
        snr = si_sdr(reference, estimate)
        return float(np.clip(2.0 + snr / 10.0, 1.0, 5.0))


class WERScorer:
    def __init__(self, model_name: str = "openai/whisper-small",
                 language: Optional[str] = "thai"):
        self.kind, self._pipe, self._wh = "disabled", None, None
        try:
            from transformers import pipeline as _hf
            self._pipe = _hf("automatic-speech-recognition", model=model_name,
                              device=0 if torch.cuda.is_available() else -1,
                              chunk_length_s=30, stride_length_s=5)
            self.kind = f"hf:{model_name}"
            return
        except Exception:
            pass
        try:
            import whisper as _whisper
            name = model_name.split("/")[-1].replace("whisper-", "")
            self._wh = _whisper.load_model(name)
            self.kind = f"openai:{name}"
        except Exception:
            pass
        self._language = language

    def transcribe(self, wav: np.ndarray) -> Optional[str]:
        if self.kind == "disabled":
            return None
        gkw: dict = {"task": "transcribe"}
        if hasattr(self, "_language") and self._language and self._language.lower() != "auto":
            gkw["language"] = self._language
        try:
            if self._pipe is not None:
                return self._pipe({"array": wav.astype(np.float32), "sampling_rate": SR},
                                  generate_kwargs=gkw)["text"]
            if self._wh is not None:
                lang = self._language if (self._language and self._language.lower() != "auto") else None
                return self._wh.transcribe(wav.astype(np.float32), fp16=False, language=lang)["text"]
        except Exception:
            pass
        return None


def _wer(ref: str, hyp: str) -> Tuple[float, float]:
    try:
        import jiwer
        try:
            from pythainlp import word_tokenize
            ref_j = " ".join(word_tokenize(ref.strip(), engine="newmm", keep_whitespace=False))
            hyp_j = " ".join(word_tokenize(hyp.strip(), engine="newmm", keep_whitespace=False))
        except ImportError:
            ref_j, hyp_j = ref.strip(), hyp.strip()
        return float(jiwer.wer(ref_j, hyp_j)), float(jiwer.cer(ref.strip(), hyp.strip()))
    except ImportError:
        pass

    def _ed(a: list, b: list) -> int:
        m, n = len(a), len(b)
        if m == 0: return n
        if n == 0: return m
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]; dp[0] = i
            for j in range(1, n + 1):
                tmp = dp[j]; dp[j] = min(dp[j]+1, dp[j-1]+1, prev+(0 if a[i-1]==b[j-1] else 1))
                prev = tmp
        return dp[n]

    wt = ref.strip().split()
    ht = hyp.strip().split()
    wer = _ed(wt, ht) / max(1, len(wt))
    cer = _ed(list(ref.strip()), list(hyp.strip())) / max(1, len(ref.strip()))
    return wer, cer


# ============================================================================
# Model loaders — each returns an EnhanceFn: np.ndarray → np.ndarray (16 kHz)
# ============================================================================

def _pad_or_trim(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.size >= n:
        return arr[:n]
    return np.pad(arr, (0, n - arr.size), mode="constant").astype(np.float32)


def load_crn(ckpt_path: str, device: torch.device) -> Optional[EnhanceFn]:
    """Load CRN from Run/CRN/ without polluting U-Net module namespace."""
    if not Path(ckpt_path).is_file():
        print(f"[bench] CRN checkpoint not found: {ckpt_path}")
        return None
    sys.path.insert(0, str(CRN_DIR))
    try:
        from model import CRN
    except ImportError as e:
        print(f"[bench] CRN import error: {e}")
        return None
    finally:
        sys.path.pop(0)
        # Rename so U-Net imports (model_unet_advanced) aren't shadowed
        sys.modules.pop("model", None)
        sys.modules.pop("inference", None)
        sys.modules.pop("losses",   None)

    try:
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = ckpt.get("args", {}) or {}
        kwarg_keys = ("n_fft", "hop_length", "rnn_hidden", "rnn_layers",
                      "rnn_type", "bottleneck_dim", "mask_bound", "dropout_p")
        kwargs = {k: saved[k] for k in kwarg_keys if k in saved}
        if "no_se" in saved:
            kwargs["use_se"] = not saved["no_se"]
        model = CRN(**kwargs).to(device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"[bench] CRN loaded from {ckpt_path}")
    except Exception as e:
        print(f"[bench] CRN load failed: {e}")
        return None

    @torch.inference_mode()
    def _enhance(noisy: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).to(device)
        y = model(x)
        return y.squeeze(0).cpu().numpy().astype(np.float32)

    return _enhance


def load_unet(ckpt_path: str, device: torch.device) -> Optional[EnhanceFn]:
    if not Path(ckpt_path).is_file():
        print(f"[bench] U-Net checkpoint not found: {ckpt_path}")
        return None
    try:
        model, _ = _unet_load(ckpt_path, device)
        print(f"[bench] U-Net loaded from {ckpt_path}")
    except Exception as e:
        print(f"[bench] U-Net load failed: {e}")
        return None

    @torch.inference_mode()
    def _enhance(noisy: np.ndarray) -> np.ndarray:
        return _unet_enhance(model, noisy, device, sr=SR)

    return _enhance


def _load_resemble_denoiser(state_dict_or_run_dir, device: torch.device):
    """Shared helper: returns Resemble Denoiser given either weights dict or run_dir."""
    try:
        from resemble_enhance.denoiser.denoiser import Denoiser
        from resemble_enhance.denoiser.hparams import HParams
    except ImportError:
        return None
    hp    = HParams()
    model = Denoiser(hp).to(device)
    if isinstance(state_dict_or_run_dir, dict):
        model.load_state_dict(state_dict_or_run_dir, strict=True)
    else:                                   # Path to pre-trained run_dir
        run_dir = Path(state_dict_or_run_dir)
        pt_path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
        raw = torch.load(str(pt_path), map_location="cpu")
        model.load_state_dict(raw["module"], strict=True)
    model.eval()
    return model


def _resemble_fn(model, device: torch.device) -> EnhanceFn:
    from resemble_enhance.inference import inference as _re_inf
    try:
        from torchaudio.functional import resample as _rs
    except ImportError:
        raise RuntimeError("torchaudio required for Resemble inference")

    @torch.inference_mode()
    def _enhance(noisy: np.ndarray) -> np.ndarray:
        n   = len(noisy)
        wav = torch.from_numpy(noisy.astype(np.float32))
        out, out_sr = _re_inf(model, wav, sr=SR, device=device)
        out_16k = _rs(out.cpu(), out_sr, SR)
        return _pad_or_trim(out_16k.numpy().astype(np.float32), n)

    return _enhance


def load_resemble_orig(device: torch.device) -> Optional[EnhanceFn]:
    try:
        from resemble_enhance.enhancer.download import download as _dl
        run_dir = _dl()
        model = _load_resemble_denoiser(run_dir, device)
        if model is None:
            raise RuntimeError("model is None")
        print(f"[bench] Resemble (orig) pre-trained loaded")
        return _resemble_fn(model, device)
    except Exception as e:
        print(f"[bench] Resemble (orig) unavailable: {e}")
        return None


def load_resemble_ft(ckpt_path: str, device: torch.device) -> Optional[EnhanceFn]:
    if not Path(ckpt_path).is_file():
        print(f"[bench] Resemble (ft) checkpoint not found: {ckpt_path}")
        return None
    try:
        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd  = raw.get("state_dict", raw)
        model = _load_resemble_denoiser(sd, device)
        if model is None:
            raise RuntimeError("model is None")
        print(f"[bench] Resemble (ft) loaded from {ckpt_path}")
        return _resemble_fn(model, device)
    except Exception as e:
        print(f"[bench] Resemble (ft) load failed: {e}")
        return None


def _load_mossformer_model(device: torch.device) -> Optional[torch.nn.Module]:
    """Try alibabasglab then modelscope."""
    try:
        from mossformer2.models.mossformer2 import MossFormer2
        from mossformer2.utils.checkpoint import load_pretrained
        model = MossFormer2(in_channels=256, out_channels=256, num_layers=24, d_model=256)
        load_pretrained(model, tag="MossFormer2_SE_16K")
        return model.to(device)
    except ImportError:
        pass
    try:
        from modelscope.models import Model

        class _Wrap(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                out = self.inner({"noisy": x})
                return out.get("enhanced", list(out.values())[0]) if isinstance(out, dict) else out

        inner = Model.from_pretrained("damo/speech_mossformergan_se_16k")
        return _Wrap(inner).to(device)
    except Exception:
        pass
    return None


def _mossformer_fn(model, device: torch.device) -> EnhanceFn:
    @torch.inference_mode()
    def _enhance(noisy: np.ndarray) -> np.ndarray:
        n   = len(noisy)
        x   = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).to(device)
        out = model(x)
        arr = out.squeeze(0).cpu().numpy().astype(np.float32)
        return _pad_or_trim(arr, n)
    return _enhance


def load_mossformer_orig(device: torch.device) -> Optional[EnhanceFn]:
    model = _load_mossformer_model(device)
    if model is None:
        print("[bench] MossFormer2 (orig) unavailable — install mossformer2 or modelscope")
        return None
    model.eval()
    print("[bench] MossFormer2 (orig) pre-trained loaded")
    return _mossformer_fn(model, device)


def load_mossformer_ft(ckpt_path: str, device: torch.device) -> Optional[EnhanceFn]:
    if not Path(ckpt_path).is_file():
        print(f"[bench] MossFormer2 (ft) checkpoint not found: {ckpt_path}")
        return None
    model_base = _load_mossformer_model(device)
    if model_base is None:
        print("[bench] MossFormer2 (ft) unavailable — base model could not be loaded")
        return None
    try:
        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd  = raw.get("state_dict", raw)
        model_base.load_state_dict(sd, strict=False)
        model_base.eval()
        print(f"[bench] MossFormer2 (ft) loaded from {ckpt_path}")
        return _mossformer_fn(model_base, device)
    except Exception as e:
        print(f"[bench] MossFormer2 (ft) load failed: {e}")
        return None


# ============================================================================
# Spectrogram export (N-panel)
# ============================================================================

def _log_spec(wav: np.ndarray, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    t   = torch.from_numpy(wav.astype(np.float32))
    win = torch.hann_window(n_fft)
    s   = torch.stft(t, n_fft=n_fft, hop_length=hop, window=win,
                     return_complex=True, center=True)
    return torch.log1p(s.abs()).numpy()


def save_spectrograms(systems: List[Tuple[str, np.ndarray]],
                      out_path: Path, source: str, fname: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(systems)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (label, wav) in zip(axes, systems):
        spec = _log_spec(wav)
        im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma", vmin=0)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Freq bin")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("Time frame")
    fig.suptitle(f"{source}  |  {fname}", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"[bench] spectrogram → {out_path}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      default="../../data_npy")
    p.add_argument("--split",          default="test")
    p.add_argument("--snr",            type=float, default=5.0,
                   help="Input SNR (dB) for the noisy mixture.")
    p.add_argument("--max_files",      type=int,   default=200,
                   help="Total files across all sources (balanced sampling).")
    p.add_argument("--crn_ckpt",       default="../CRN/checkpoints/best.pt")
    p.add_argument("--unet_ckpt",      default="checkpoints/best.pt")
    p.add_argument("--resemble_ft",    default="resemble_finetuned/best.pt")
    p.add_argument("--mossformer_ft",  default="mossformer_finetuned/best.pt")
    p.add_argument("--whisper_model",  default="openai/whisper-small")
    p.add_argument("--language",       default="thai")
    p.add_argument("--out_dir",        default="benchmark_results")
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bench] device={device}  split={args.split}  snr={args.snr} dB")

    out_dir  = Path(args.out_dir)
    spec_dir = out_dir / "spectrograms"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load all models ----
    systems: Dict[str, Optional[EnhanceFn]] = {
        "Noisy":            lambda x: x.copy(),
        "CRN":              load_crn(args.crn_ckpt, device),
        "U-Net":            load_unet(args.unet_ckpt, device),
        "Resemble-orig":    load_resemble_orig(device),
        "Resemble-ft":      load_resemble_ft(args.resemble_ft, device),
        "MossFormer2-orig": load_mossformer_orig(device),
        "MossFormer2-ft":   load_mossformer_ft(args.mossformer_ft, device),
    }

    active = {name: fn for name, fn in systems.items() if fn is not None}
    skipped = [name for name, fn in systems.items() if fn is None and name != "Noisy"]
    if skipped:
        print(f"[bench] SKIPPED (not available): {skipped}")
    print(f"[bench] Evaluating {len(active)} systems: {list(active)}")

    # ---- Metrics ----
    mos_scorer = MOSScorer(sr=SR)
    wer_scorer = WERScorer(model_name=args.whisper_model,
                           language=(None if args.language.lower() == "auto" else args.language))
    print(f"[bench] MOS={mos_scorer.kind}  WER={wer_scorer.kind}")

    # ---- Data ----
    split_dir   = Path(args.data_root) / args.split
    all_clean   = sorted((split_dir / "clean").glob("*.npy"))
    noise_files = sorted((split_dir / "noise").glob("*.npy"))
    labels      = _read_labels(split_dir / "labels_npy.csv") if (split_dir / "labels_npy.csv").is_file() else {}

    _source_keys = [s for s, _ in _SOURCE_PATTERNS]
    groups: Dict[str, List[Path]] = {s: [] for s in _source_keys}
    for cf in all_clean:
        src = _detect_source(cf.name)
        if src in groups:
            groups[src].append(cf)

    n_groups   = sum(1 for s in _source_keys if groups[s])
    per_source = args.max_files // max(1, n_groups)
    random.seed(args.seed)
    clean_files: List[Path] = []
    for src in _source_keys:
        pool   = groups[src]
        sample = random.sample(pool, min(per_source, len(pool))) if pool else []
        clean_files.extend(sample)
    random.shuffle(clean_files)
    print(f"[bench] {len(clean_files)} files selected (balanced, seed={args.seed})")

    # ---- Evaluation ----
    rng  = np.random.default_rng(args.seed)
    sums = {name: {"mos": 0.0, "sisdr": 0.0, "wer": 0.0, "cer": 0.0, "n": 0, "n_wer": 0}
            for name in active}
    rows: List[Dict] = []
    example_data: Dict[str, Dict] = {}  # source → {"file", "panels": [(label, wav)]}

    for i, cf in enumerate(clean_files):
        ref_text = labels.get(cf.name, "")
        clean    = np.load(cf).astype(np.float32).squeeze()
        if clean.ndim > 1:
            clean = clean.mean(0)

        nf    = noise_files[int(rng.integers(0, len(noise_files)))]
        noise = np.load(nf).astype(np.float32).squeeze()
        if noise.ndim > 1:
            noise = noise.mean(0)
        noise = _crop_or_pad(noise, clean.size)
        noisy, _ = mix_at_snr(clean, noise, args.snr)

        row: Dict = {"file": cf.name, "snr_in_db": args.snr}
        panels: List[Tuple[str, np.ndarray]] = []

        hyps: Dict[str, Optional[str]] = {}
        for name, fn in active.items():
            enhanced = fn(noisy)
            enhanced = _pad_or_trim(enhanced, len(clean))

            m  = mos_scorer.score(clean, enhanced)
            sd = si_sdr(clean, enhanced)
            row[f"{name}_mos"]   = round(m,  4)
            row[f"{name}_sisdr"] = round(sd, 4)
            sums[name]["mos"]   += m
            sums[name]["sisdr"] += sd
            sums[name]["n"]     += 1

            if ref_text and wer_scorer.kind != "disabled":
                hyp = wer_scorer.transcribe(enhanced)
                hyps[name] = hyp
                if hyp is not None:
                    w, c = _wer(ref_text, hyp)
                    row[f"{name}_wer"] = round(w, 4)
                    row[f"{name}_cer"] = round(c, 4)
                    sums[name]["wer"]   += w
                    sums[name]["cer"]   += c
                    sums[name]["n_wer"] += 1

            panels.append((name, enhanced.copy()))

        # add clean reference at the bottom
        panels.append(("Clean (reference)", clean.copy()))
        rows.append(row)

        src = _detect_source(cf.name)
        if src and src not in example_data:
            example_data[src] = {"file": cf.name, "panels": panels}

        # console progress
        crn_sd = row.get("CRN_sisdr", row.get("Noisy_sisdr", "-"))
        print(f"  [{i+1:3d}/{len(clean_files)}] {cf.name[:50]}  "
              f"Noisy={row.get('Noisy_sisdr','-'):>6}  "
              f"CRN={row.get('CRN_sisdr','-'):>6}  "
              f"U-Net={row.get('U-Net_sisdr','-'):>6} dB")

    # ---- Per-utterance CSV ----
    per_utt_path = out_dir / "per_utt.csv"
    with open(per_utt_path, "w", newline="", encoding="utf-8") as fh:
        if rows:
            keys = sorted(set().union(*[r.keys() for r in rows]))
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    print(f"[bench] per_utt.csv → {per_utt_path}")

    # ---- Summary CSV + console ----
    summary_path = out_dir / "summary.csv"
    bar = "=" * 78
    print(f"\n{bar}")
    print(f"  {'System':<22} {'SI-SDR':>8} {'MOS':>7} {'WER':>7} {'CER':>7}  {'n':>5}")
    print(bar)

    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "SI-SDR_dB", "MOS", "WER", "CER", "n", "n_wer"])
        for name in active:
            s = sums[name]
            if s["n"] == 0:
                continue
            sd_avg  = s["sisdr"] / s["n"]
            mos_avg = s["mos"]   / s["n"]
            nw      = s["n_wer"]
            wer_avg = (s["wer"] / nw) if nw else float("nan")
            cer_avg = (s["cer"] / nw) if nw else float("nan")

            w.writerow([name, f"{sd_avg:.4f}", f"{mos_avg:.4f}",
                        f"{wer_avg:.4f}" if nw else "n/a",
                        f"{cer_avg:.4f}" if nw else "n/a",
                        s["n"], nw])

            wer_str = f"{wer_avg:.4f}" if nw else "  n/a"
            cer_str = f"{cer_avg:.4f}" if nw else "  n/a"
            flag = " ←best" if sd_avg == max(
                sums[n]["sisdr"] / max(1, sums[n]["n"]) for n in active) else ""
            print(f"  {name:<22} {sd_avg:>+8.2f} {mos_avg:>7.3f} "
                  f"{wer_str:>7} {cer_str:>7}  {s['n']:>5}{flag}")

    print(bar)
    print(f"\n[bench] summary.csv → {summary_path}")

    for skipped_name in skipped:
        print(f"[bench] {skipped_name:<22}  SKIPPED (not installed / checkpoint missing)")

    # ---- Spectrograms ----
    for src, data in example_data.items():
        out_png = spec_dir / f"{src}_{Path(data['file']).stem}.png"
        save_spectrograms(data["panels"], out_png, src, data["file"])

    print(f"\n[bench] Done.  Results in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
