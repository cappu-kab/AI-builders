"""
evaluate.py
===========
Quantitative evaluation of the speech-denoising CRN.

Compares three systems on the test split:
    1. raw noisy   (no processing)
    2. Wiener filter (classical baseline, scipy.signal.wiener)
    3. trained CRN (--ckpt)

Metrics
-------
    * MOS-style perceptual score (best-available: SpeechMOS → PESQ → SNR proxy)
    * WER  — pythainlp word tokenisation then jiwer edit-distance (Thai-aware)
    * CER  — character edit-distance via jiwer.cer
    * SI-SDR for sanity

Outputs
-------
    outputs/eval_per_utt.csv       — per-file metrics
    outputs/eval_summary.csv       — dataset-level averages
    outputs/spectrogram_examples/  — 3-panel PNGs for one file per dataset source

Usage
-----
    python evaluate.py --ckpt checkpoints/best.pt \\
                       --data_root ./data_npy --split test \\
                       --snr 5 --max_files 50
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from inference import enhance_waveform, load_model
from dataset import _crop_or_pad, mix_at_snr, _read_labels


# ---------------------------------------------------------------------------
# Wiener-filter baseline
# ---------------------------------------------------------------------------
def wiener_baseline(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import wiener
        y = wiener(x.astype(np.float64), mysize=129)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        peak = max(1e-9, float(np.max(np.abs(y))))
        if peak > 1.0:
            y = y / peak
        return y
    except Exception as e:
        print(f"[eval] scipy.signal.wiener unavailable ({e}); returning input")
        return x.astype(np.float32)


# ---------------------------------------------------------------------------
# SI-SDR
# ---------------------------------------------------------------------------
def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    ref = reference - reference.mean()
    est = estimate  - estimate.mean()
    alpha  = float(np.dot(est, ref) / (np.dot(ref, ref) + eps))
    target = alpha * ref
    noise  = est - target
    return float(10.0 * np.log10((np.sum(target ** 2) + eps) /
                                 (np.sum(noise  ** 2) + eps)))


# ---------------------------------------------------------------------------
# MOS scorer (best-available)
# ---------------------------------------------------------------------------
class MOSScorer:
    """
    Tries (in order):
      1. SpeechMOS  – `pip install speechmos`   (no reference, neural MOS)
      2. PESQ-NB    – `pip install pesq`         (reference required)
      3. SNR proxy  – pseudo-MOS = clip(2 + SNR/10, 1, 5)
    """
    def __init__(self, sr: int = 16_000):
        self.sr   = sr
        self.kind = "snr_proxy"
        self.scorer: Optional[object] = None
        try:
            from speechmos import dnsmos
            self.scorer = lambda wav: float(dnsmos.run(wav, sr=self.sr)["ovrl_mos"])
            self.kind = "dnsmos"
            return
        except Exception:
            pass
        try:
            from pesq import pesq as _pesq
            self.scorer = lambda ref, deg: float(_pesq(self.sr, ref, deg, "wb"))
            self.kind = "pesq_wb"
        except Exception:
            pass

    def score(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        if self.kind == "dnsmos":
            return self.scorer(estimate)
        if self.kind == "pesq_wb":
            try:
                return self.scorer(reference, estimate)
            except Exception:
                pass
        snr = si_sdr(reference, estimate)
        return float(np.clip(2.0 + snr / 10.0, 1.0, 5.0))


# ---------------------------------------------------------------------------
# WER / CER helpers
# ---------------------------------------------------------------------------

def _edit_distance(a: list, b: list) -> int:
    """Wagner-Fischer edit distance — works on any comparable list (tokens or chars)."""
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            tmp  = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[n]


_HAVE_JIWER = False
try:
    import jiwer as _jiwer
    _HAVE_JIWER = True
except ImportError:
    print("[eval] WARNING: jiwer not installed — WER/CER use built-in edit-distance. "
          "Install with: pip install jiwer")

_HAVE_PYTHAINLP = False
try:
    import pythainlp as _pythainlp
    _HAVE_PYTHAINLP = True
except ImportError:
    pass


def _tok_thai(text: str) -> List[str]:
    """Tokenise text into words.

    Uses ``pythainlp`` word_tokenize (newmm engine) for Thai — Thai text has no
    spaces between words, so space-splitting gives one giant 'token' per sentence.
    Falls back to whitespace splitting when pythainlp is not installed.
    """
    if _HAVE_PYTHAINLP:
        from pythainlp import word_tokenize
        tokens = word_tokenize(text.strip(), engine="newmm", keep_whitespace=False)
        return [t for t in tokens if t.strip()]
    return text.strip().split()


def _wer_tokens(ref_toks: List[str], hyp_toks: List[str]) -> float:
    """WER from pre-tokenised lists."""
    if not ref_toks:
        return 0.0 if not hyp_toks else 1.0
    if _HAVE_JIWER:
        return float(_jiwer.wer(" ".join(ref_toks), " ".join(hyp_toks)))
    return _edit_distance(ref_toks, hyp_toks) / len(ref_toks)


def _score_wer_cer(ref: str, hyp: str,
                   segment_seconds: float = 4.0,
                   words_per_second: float = 3.0) -> Tuple[float, float]:
    """Return *(WER, CER)* for a reference / hypothesis pair.

    Prefix matching
    ---------------
    When the reference label covers more audio than the 4-second segment being
    evaluated, a full-reference WER unfairly penalises correct transcriptions of
    the prefix.  This function tries both the full reference and a prefix of
    length ≈ segment_seconds × words_per_second, and returns the lower WER.

    WER: tokenised with ``_tok_thai`` (pythainlp when available).
    CER: character edit-distance on the raw stripped strings.
    """
    ref_toks = _tok_thai(ref)
    hyp_toks = _tok_thai(hyp)

    # Full-reference WER
    wer_full = _wer_tokens(ref_toks, hyp_toks)

    # Prefix WER: try a prefix ≈ expected words in `segment_seconds`
    n_prefix = max(1, int(round(segment_seconds * words_per_second)))
    if len(ref_toks) > n_prefix and len(hyp_toks) > 0:
        wer_prefix = _wer_tokens(ref_toks[:n_prefix], hyp_toks)
        wer = min(wer_full, wer_prefix)
    else:
        wer = wer_full

    # CER (character level — always against full ref, prefix CER not meaningful)
    if _HAVE_JIWER:
        cer = float(_jiwer.cer(ref.strip(), hyp.strip()))
    else:
        ref_chars = list(ref.strip())
        hyp_chars = list(hyp.strip())
        if not ref_chars:
            cer = 0.0 if not hyp_chars else 1.0
        else:
            cer = _edit_distance(ref_chars, hyp_chars) / len(ref_chars)

    return wer, cer


# ---------------------------------------------------------------------------
# ASR / WER scorer (HuggingFace transformers → openai-whisper fallback)
# ---------------------------------------------------------------------------
class WERScorer:
    """ASR-based WER + CER scorer.

    Load priority
    -------------
    1. HuggingFace ``transformers`` pipeline with Whisper (multilingual).
    2. ``openai-whisper`` pip package — fallback if transformers is absent.
    3. Disabled — WER/CER columns are omitted from output.

    Language forcing
    ----------------
    ``language`` is passed directly to Whisper's ``generate_kwargs`` so the
    decoder never wastes steps guessing the language.  For a Thai dataset set
    ``language="thai"`` (default).  Pass ``None`` or ``"auto"`` to let Whisper
    detect automatically.
    """

    def __init__(self, model_name: str = "openai/whisper-small",
                 sample_rate: int = 16_000,
                 device: Optional[str] = None,
                 language: Optional[str] = "thai"):
        self.kind        = "disabled"
        self.sample_rate = sample_rate
        self._language   = language  # e.g. "thai", "english", None/"auto"
        self._pipe       = None   # transformers pipeline
        self._wh         = None   # openai-whisper model

        if not _HAVE_JIWER:
            print("[eval] WARNING: jiwer not installed — WER/CER use built-in edit-distance.")
        if not _HAVE_PYTHAINLP:
            print("[eval] WARNING: pythainlp not installed — Thai WER will use whitespace split. "
                  "Install with: pip install pythainlp")

        # ---- 1. HuggingFace transformers (preferred) ----
        try:
            from transformers import pipeline as _hf_pipeline
            _dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"[eval] loading ASR model '{model_name}' on {_dev} …")
            self._pipe = _hf_pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=_dev,
                chunk_length_s=30,
                stride_length_s=5,
            )
            self.kind = f"hf:{model_name}"
            lang_tag = f"  language={self._language}" if self._language else ""
            print(f"[eval] ASR ready  ({self.kind}){lang_tag}  "
                  f"jiwer={'yes' if _HAVE_JIWER else 'fallback'}  "
                  f"pythainlp={'yes' if _HAVE_PYTHAINLP else 'fallback'}")
            return
        except Exception as exc:
            print(f"[eval] transformers pipeline unavailable ({exc}); trying openai-whisper …")

        # ---- 2. openai-whisper fallback ----
        try:
            import whisper as _whisper
            _wname = model_name.split("/")[-1].replace("whisper-", "")
            _dev   = device or ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[eval] loading openai-whisper '{_wname}' on {_dev} …")
            self._wh  = _whisper.load_model(_wname, device=_dev)
            self.kind = f"openai-whisper:{_wname}"
            print(f"[eval] ASR ready  ({self.kind})  "
                  f"jiwer={'yes' if _HAVE_JIWER else 'fallback'}")
        except Exception as exc:
            print(f"[eval] openai-whisper unavailable ({exc}) — WER/CER will be skipped.")

    # ------------------------------------------------------------------
    def _transcribe_raw(self, wav: np.ndarray) -> str:
        """Run ASR and return raw hypothesis string. Raises on failure."""
        gkw: dict = {"task": "transcribe"}
        if self._language and self._language.lower() != "auto":
            gkw["language"] = self._language

        if self._pipe is not None:
            result = self._pipe(
                {"array": wav.astype(np.float32), "sampling_rate": self.sample_rate},
                generate_kwargs=gkw,
            )
            return result["text"]

        if self._wh is not None:
            lang_arg = self._language if (self._language and
                                          self._language.lower() != "auto") else None
            return self._wh.transcribe(
                wav.astype(np.float32),
                fp16=False, verbose=False,
                language=lang_arg,
            )["text"]

        raise RuntimeError("No ASR backend available")

    # ------------------------------------------------------------------
    def transcribe(self, wav: np.ndarray) -> Optional[str]:
        """Transcribe *wav* and return the hypothesis string, or *None* on failure."""
        if self.kind == "disabled":
            return None
        try:
            return self._transcribe_raw(wav)
        except Exception as exc:
            print(f"[eval]   ⚠ transcription error — {exc}")
            return None

    # ------------------------------------------------------------------
    def __call__(self, reference: str, wav: np.ndarray) -> Optional[float]:
        """Backward-compatible interface: transcribe *wav* and return WER only."""
        hyp = self.transcribe(wav)
        if hyp is None:
            return None
        wer, _ = _score_wer_cer(reference, hyp)
        return wer


# ---------------------------------------------------------------------------
# Dataset-source detection from filename
# ---------------------------------------------------------------------------
_SOURCE_PATTERNS: List[Tuple[str, List[str]]] = [
    ("commonvoice", ["commonvoice"]),
    ("fleurs",      ["fleurs", "feulr"]),       # include the known typo variant
    ("thai_isan",   ["thai_isan"]),
    ("thai_elder",  ["thai_elder", "elder"]),
]


def _detect_source(filename: str) -> Optional[str]:
    """Return the dataset-source key (e.g. 'commonvoice') from a filename, or *None*."""
    name = filename.lower()
    for source, patterns in _SOURCE_PATTERNS:
        if any(p in name for p in patterns):
            return source
    return None


# ---------------------------------------------------------------------------
# Spectrogram export
# ---------------------------------------------------------------------------
def _save_spectrogram_examples(
    example_data: Dict[str, dict],
    out_dir: Path,
    sr: int = 16_000,
    n_fft: int = 512,
    hop: int = 128,
) -> None:
    """Save a 3-panel spectrogram PNG (noisy / enhanced / clean) for each example."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[eval] matplotlib not available; skipping spectrogram export.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    def _log_spec(wav: np.ndarray) -> np.ndarray:
        t   = torch.from_numpy(wav.astype(np.float32))
        win = torch.hann_window(n_fft)
        s   = torch.stft(t, n_fft=n_fft, hop_length=hop,
                         window=win, return_complex=True, center=True)
        return torch.log1p(s.abs()).numpy()

    sources_ordered = ["commonvoice", "fleurs", "thai_isan", "thai_elder"]
    for source in sources_ordered:
        data = example_data.get(source)
        if data is None:
            continue

        stem = Path(data["file"]).stem

        panels = [
            ("Noisy Input",          data["noisy"]),
            ("CRN Enhanced",         data["enhanced"]),
            ("Ground Truth (Clean)", data["clean"]),
        ]

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        for ax, (title, wav) in zip(axes, panels):
            spec_img = _log_spec(wav)
            im = ax.imshow(spec_img, aspect="auto", origin="lower",
                           cmap="magma", vmin=0)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("Frequency bin")
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        axes[-1].set_xlabel("Time frame")

        fig.suptitle(f"Dataset: {source}  |  {stem}",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()

        out_path = out_dir / f"{stem}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[eval] spectrogram saved → {out_path}")


# ---------------------------------------------------------------------------
# Console text-example block
# ---------------------------------------------------------------------------
_EXAMPLE_SOURCES = ("commonvoice", "fleurs", "thai_isan", "thai_elder")


def _print_text_examples(example_data: Dict[str, dict]) -> None:
    """Print a formatted block of ground-truth vs. transcription for each source."""
    bar = "=" * 72
    print(f"\n{bar}")
    print("  QUALITATIVE TEXT EXAMPLES  (one per dataset source)")
    print(bar)
    for source in _EXAMPLE_SOURCES:
        data = example_data.get(source)
        if data is None:
            print(f"\n  [{source}]  ← no file from this source found in evaluated set")
            continue
        crn_text   = data.get("crn_hyp")   or "(transcription failed / ASR disabled)"
        noisy_text = data.get("noisy_hyp") or "(transcription failed / ASR disabled)"
        print(f"\n  Dataset Source : {source}")
        print(f"  File           : {data['file']}")
        print(f"  Ground Truth   : {data['ref_text'] or '(no reference label)'}")
        print(f"  CRN Enhanced   : {crn_text}")
        print(f"  Noisy (raw)    : {noisy_text}")
    print(f"{bar}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",         default="./checkpoints/best.pt")
    ap.add_argument("--data_root",    default="./data_npy")
    ap.add_argument("--split",        default="test")
    ap.add_argument("--snr",          type=float, default=5.0)
    ap.add_argument("--max_files",    type=int,   default=456,
                    help="Cap evaluated files (default 456 = 10%% of train budget).")
    ap.add_argument("--sample_rate",  type=int,   default=16_000)
    ap.add_argument("--whisper_model", default="openai/whisper-small",
                    help="HuggingFace model ID or openai-whisper name. "
                         "Must be multilingual (no '.en' suffix) for Thai data. "
                         "Default: openai/whisper-small (~244M params, best Thai accuracy).")
    ap.add_argument("--language",     default="thai",
                    help="Force Whisper decoder language (e.g. 'thai', 'english'). "
                         "Pass 'auto' to let the model detect automatically.")
    ap.add_argument("--out_csv",     default="./outputs/eval_per_utt.csv")
    ap.add_argument("--out_summary", default="./outputs/eval_summary.csv")
    ap.add_argument("--spec_dir",    default="./outputs/spectrogram_examples",
                    help="Directory for comparative spectrogram PNGs.")
    ap.add_argument("--seed",        type=int, default=0)
    args = ap.parse_args()

    lang = args.language if args.language.lower() != "auto" else None

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.ckpt, device)
    mos  = MOSScorer(sr=args.sample_rate)
    werx = WERScorer(model_name=args.whisper_model,
                     sample_rate=args.sample_rate,
                     language=lang)
    print(f"[eval] MOS={mos.kind}  WER/CER={werx.kind}  device={device}")

    split_dir   = Path(args.data_root) / args.split
    all_clean   = sorted((split_dir / "clean").glob("*.npy"))
    noise_files = sorted((split_dir / "noise").glob("*.npy"))
    labels_path = split_dir / "labels_npy.csv"
    labels      = _read_labels(labels_path) if labels_path.is_file() else {}

    # ---- Balanced sampling: equal quota from each dataset source ----
    _source_keys = [s for s, _ in _SOURCE_PATTERNS]
    groups: Dict[str, List[Path]] = {s: [] for s in _source_keys}
    ungrouped: List[Path] = []
    for cf in all_clean:
        src = _detect_source(cf.name)
        if src in groups:
            groups[src].append(cf)
        else:
            ungrouped.append(cf)

    n_groups   = sum(1 for s in _source_keys if groups[s])
    per_source = (args.max_files // max(1, n_groups)) if args.max_files else None

    # Fixed seed so every run evaluates the identical balanced subset.
    random.seed(42)
    clean_files: List[Path] = []
    print(f"[eval] balanced sampling (seed=42, {per_source} per source):")
    for src in _source_keys:
        pool   = groups[src]
        sample = (random.sample(pool, min(per_source, len(pool)))
                  if per_source is not None else list(pool))
        clean_files.extend(sample)
        print(f"[eval]   {src:14s}: {len(pool):5d} available  →  {len(sample):4d} selected")
    if ungrouped:
        print(f"[eval]   (unknown src) : {len(ungrouped):5d} files skipped (no source tag)")

    random.shuffle(clean_files)
    print(f"[eval] evaluating {len(clean_files)} files (balanced) from '{args.split}' split")

    rng  = np.random.default_rng(args.seed)
    rows: List[Dict] = []
    sums = {sys: {"mos": 0.0, "wer": 0.0, "cer": 0.0,
                  "sisdr": 0.0, "n": 0, "n_wer": 0}
            for sys in ("noisy", "wiener", "crn")}

    # One captured example per dataset source for qualitative reporting.
    example_data: Dict[str, dict] = {}

    for i, cf in enumerate(clean_files):
        ref_text = labels.get(cf.name, "")
        clean    = np.load(cf).astype(np.float32).squeeze()
        if clean.ndim > 1:
            clean = clean.mean(axis=0)

        nf    = noise_files[int(rng.integers(0, len(noise_files)))]
        noise = np.load(nf).astype(np.float32).squeeze()
        if noise.ndim > 1:
            noise = noise.mean(axis=0)
        noise = _crop_or_pad(noise, clean.size)
        noisy, _ = mix_at_snr(clean, noise, args.snr)

        crn_out = enhance_waveform(model, noisy, device, sr=args.sample_rate)
        wnr_out = wiener_baseline(noisy)

        row: Dict = {
            "file":       cf.name,
            "snr_in_db":  args.snr,
            "ref_words":  len(ref_text.split()),
        }

        # ---- acoustic metrics (MOS + SI-SDR, no transcription) ----
        for sys_name, est in (("noisy", noisy), ("wiener", wnr_out), ("crn", crn_out)):
            m  = mos.score(clean, est)
            sd = si_sdr(clean, est)
            row[f"{sys_name}_mos"]   = round(m,  4)
            row[f"{sys_name}_sisdr"] = round(sd, 4)
            sums[sys_name]["mos"]   += m
            sums[sys_name]["sisdr"] += sd
            sums[sys_name]["n"]     += 1

        # ---- transcription + WER + CER ----
        transcriptions: Dict[str, Optional[str]] = {}
        if ref_text and werx.kind != "disabled":
            for sys_name, est in (("noisy", noisy), ("wiener", wnr_out), ("crn", crn_out)):
                hyp = werx.transcribe(est)
                transcriptions[sys_name] = hyp
                if hyp is not None:
                    wer, cer = _score_wer_cer(ref_text, hyp)
                    row[f"{sys_name}_wer"] = round(wer, 4)
                    row[f"{sys_name}_cer"] = round(cer, 4)
                    sums[sys_name]["wer"]   += wer
                    sums[sys_name]["cer"]   += cer
                    sums[sys_name]["n_wer"] += 1

        rows.append(row)
        print(f" [{i+1:3d}/{len(clean_files)}] {cf.name}  "
              f"noisy_mos={row.get('noisy_mos', '-'):>6}  "
              f"crn_mos={row.get('crn_mos',   '-'):>6}  "
              f"crn_wer={row.get('crn_wer',   '-'):>6}  "
              f"crn_cer={row.get('crn_cer',   '-'):>6}")

        # ---- capture one example per dataset source ----
        src = _detect_source(cf.name)
        if src and src not in example_data:
            example_data[src] = {
                "file":      cf.name,
                "ref_text":  ref_text,
                "noisy_hyp": transcriptions.get("noisy"),
                "crn_hyp":   transcriptions.get("crn"),
                "clean":     clean.copy(),
                "noisy":     noisy.copy(),
                "enhanced":  crn_out.copy(),
            }

    # ---------- per-utterance CSV ----------
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        if rows:
            all_keys = sorted(set().union(*[r.keys() for r in rows]))
            w = csv.DictWriter(fh, fieldnames=all_keys)
            w.writeheader()
            w.writerows(rows)

    # ---------- summary CSV ----------
    with open(args.out_summary, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "MOS", "SI-SDR(dB)", "WER", "CER", "n", "n_wer",
                    "mos_kind", "asr_kind",
                    "pythainlp", "language"])
        for sys_name in ("noisy", "wiener", "crn"):
            s       = sums[sys_name]
            mos_avg = s["mos"]   / max(1, s["n"])
            sd_avg  = s["sisdr"] / max(1, s["n"])
            n_w     = s["n_wer"]
            wer_avg = (s["wer"] / n_w) if n_w else float("nan")
            cer_avg = (s["cer"] / n_w) if n_w else float("nan")
            w.writerow([
                sys_name,
                f"{mos_avg:.4f}", f"{sd_avg:.4f}",
                f"{wer_avg:.4f}" if n_w else "n/a",
                f"{cer_avg:.4f}" if n_w else "n/a",
                s["n"], n_w,
                mos.kind, werx.kind,
                "yes" if _HAVE_PYTHAINLP else "no",
                args.language,
            ])

    # ---------- console summary ----------
    print("\n=== Summary ===")
    for sys_name in ("noisy", "wiener", "crn"):
        s       = sums[sys_name]
        mos_avg = s["mos"]   / max(1, s["n"])
        sd_avg  = s["sisdr"] / max(1, s["n"])
        n_w     = s["n_wer"]
        if n_w:
            print(f"  {sys_name:6s}  MOS={mos_avg:.3f}  SI-SDR={sd_avg:+.2f}dB  "
                  f"WER={s['wer']/n_w:.3f}  CER={s['cer']/n_w:.3f}  (n={n_w})")
        else:
            print(f"  {sys_name:6s}  MOS={mos_avg:.3f}  SI-SDR={sd_avg:+.2f}dB  "
                  f"WER=n/a  CER=n/a")

    # ---------- qualitative examples ----------
    if example_data:
        _print_text_examples(example_data)
    else:
        print("[eval] No source-tagged files found — check filename patterns "
              f"({[p for src, ps in _SOURCE_PATTERNS for p in ps]})")

    # ---------- spectrogram PNGs ----------
    if example_data:
        _save_spectrogram_examples(
            example_data,
            out_dir=Path(args.spec_dir),
            sr=args.sample_rate,
        )
    else:
        print("[eval] Skipping spectrogram export (no examples captured).")


if __name__ == "__main__":
    main()
