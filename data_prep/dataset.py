"""
Evaluation dataset.

Two operating modes:

  1. paired      -- you already have (noisy, clean) pairs on disk.
  2. on_the_fly  -- you have separate clean and noise corpora and want
                    to mix them at controlled SNRs at evaluation time.

Each item is returned as an `EvalSample` dataclass containing the noisy,
clean, and (computed) noise components, the sample rate, and the input SNR.
"""
from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import librosa


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class EvalSample:
    name: str
    clean: np.ndarray
    noise: np.ndarray
    noisy: np.ndarray
    sr: int
    snr_db: float
    category: str = "all"


def find_wavs_recursive(root: str) -> List[str]:
    """Walk `root` and return all .wav files (case-insensitive)."""
    out: list[str] = []
    if not root or not os.path.isdir(root):
        return out
    for dirpath, _, fnames in os.walk(root):
        for fn in fnames:
            if fn.lower().endswith(".wav"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load(path: str, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def _rms(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps))


def mix_at_snr(clean: np.ndarray,
               noise: np.ndarray,
               snr_db: float,
               eps: float = 1e-8
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mix `clean` and `noise` at the given SNR (in dB), tile the noise if
    it is shorter than the clean signal, and rescale all three signals
    together if the result would clip.

    Returns: (clean_out, noise_out, noisy_out) — all float32, same length.
    """
    clean = clean.astype(np.float32, copy=False)
    noise = noise.astype(np.float32, copy=False)

    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / max(1, len(noise))))
        noise = np.tile(noise, reps)
    noise = noise[: len(clean)]

    clean_rms = _rms(clean)
    noise_rms = _rms(noise)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0) + eps)
    noise_scaled = noise * (target_noise_rms / (noise_rms + eps))
    noisy = clean + noise_scaled

    peak = float(np.max(np.abs(noisy))) if len(noisy) else 0.0
    if peak > 0.99:
        s = 0.99 / peak
        return (clean * s).astype(np.float32), \
               (noise_scaled * s).astype(np.float32), \
               (noisy * s).astype(np.float32)
    return clean.astype(np.float32), \
           noise_scaled.astype(np.float32), \
           noisy.astype(np.float32)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class EvalDataset:
    """Indexable evaluation set. See classmethod constructors below."""

    def __init__(self, sr: int = 16000) -> None:
        self.sr = sr
        self._items: list = []
        self._mode: str | None = None

    # -- constructors -------------------------------------------------------- #
    @classmethod
    def from_paired(cls,
                    pairs: Sequence[Tuple[str, str]],
                    sr: int = 16000) -> "EvalDataset":
        """`pairs` is an iterable of (noisy_path, clean_path)."""
        ds = cls(sr=sr)
        ds._mode = "paired"
        ds._items = list(pairs)
        return ds

    @classmethod
    def from_on_the_fly(cls,
                        clean_paths: Sequence[str],
                        noise_paths,
                        snrs_db: Iterable[float] = (0, 5, 10, 15),
                        n_per_clean: int = 1,
                        seed: int = 0,
                        sr: int = 16000,
                        max_clean_samples: int | None = None
                        ) -> "EvalDataset":
        """
        Random pairing of clean files with noise files at given SNRs.

        `noise_paths` can be:
          * a flat list of noise file paths (single category, label="all")
          * a dict {category: [noise_paths]}: each clean file gets paired
            with one noise from EACH category, producing one labelled
            sample per category. This is what gives you the per-noise-source
            performance breakdown in the summary.
        """
        ds = cls(sr=sr)
        ds._mode = "otf"
        rng = np.random.default_rng(seed)
        snrs = list(snrs_db)

        if isinstance(noise_paths, dict):
            categories = noise_paths
        else:
            categories = {"all": list(noise_paths)}

        clean_list = list(clean_paths)
        if max_clean_samples is not None and len(clean_list) > max_clean_samples:
            # Deterministic subsampling.
            sub_rng = np.random.default_rng(seed + 1)
            idx = sub_rng.permutation(len(clean_list))[:max_clean_samples]
            clean_list = [clean_list[i] for i in sorted(idx.tolist())]

        items = []
        for cp in clean_list:
            for cat, npaths in categories.items():
                if not npaths:
                    continue
                for _ in range(n_per_clean):
                    np_idx = int(rng.integers(0, len(npaths)))
                    snr = float(snrs[int(rng.integers(0, len(snrs)))])
                    items.append((cp, npaths[np_idx], snr, cat))
        ds._items = items
        return ds

    @classmethod
    def from_dirs(cls,
                  clean_dir: str | None = None,
                  noise_dir=None,
                  noisy_dir: str | None = None,
                  sr: int = 16000,
                  recursive: bool = True,
                  **otf_kwargs) -> "EvalDataset":
        """
        Build either a paired or on-the-fly dataset from directory paths.

        `noise_dir` accepts:
          * a single path string,
          * a dict {category: path}, or
          * a list of (category, path) tuples
        and discovers WAVs recursively (case-insensitive).
        """
        finder = (find_wavs_recursive if recursive
                  else (lambda d: sorted(glob.glob(os.path.join(d, "*.wav")))))

        if noisy_dir and clean_dir:
            pairs = []
            for n in finder(noisy_dir):
                base = os.path.basename(n)
                c = os.path.join(clean_dir, base)
                if os.path.exists(c):
                    pairs.append((n, c))
            return cls.from_paired(pairs, sr=sr)

        if clean_dir and noise_dir:
            cleans = finder(clean_dir)
            if isinstance(noise_dir, dict):
                cats = {k: finder(v) for k, v in noise_dir.items()}
            elif isinstance(noise_dir, list):
                cats = {k: finder(v) for k, v in noise_dir}
            else:
                cats = {"all": finder(noise_dir)}
            return cls.from_on_the_fly(cleans, cats, sr=sr, **otf_kwargs)

        raise ValueError(
            "Provide either (noisy_dir, clean_dir) for paired mode "
            "or (clean_dir, noise_dir) for on-the-fly mode.")

    # -- protocol ------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> EvalSample:
        if self._mode == "paired":
            noisy_path, clean_path = self._items[idx]
            clean = _load(clean_path, self.sr)
            noisy = _load(noisy_path, self.sr)
            L = min(len(clean), len(noisy))
            clean, noisy = clean[:L], noisy[:L]
            noise = noisy - clean
            snr = 20.0 * np.log10(_rms(clean) / (_rms(noise) + 1e-8))
            name = os.path.splitext(os.path.basename(noisy_path))[0]
            return EvalSample(name=name, clean=clean, noise=noise,
                              noisy=noisy, sr=self.sr,
                              snr_db=float(snr), category="paired")

        if self._mode == "otf":
            item = self._items[idx]
            # Backward-compat: 3-tuple (no category) or 4-tuple.
            if len(item) == 4:
                clean_path, noise_path, snr, cat = item
            else:
                clean_path, noise_path, snr = item
                cat = "all"
            clean = _load(clean_path, self.sr)
            noise = _load(noise_path, self.sr)
            clean, noise_s, noisy = mix_at_snr(clean, noise, snr)
            name = (
                f"{cat}__"
                f"{os.path.splitext(os.path.basename(clean_path))[0]}__"
                f"{os.path.splitext(os.path.basename(noise_path))[0]}__"
                f"snr{snr:+.0f}"
            )
            return EvalSample(name=name, clean=clean, noise=noise_s,
                              noisy=noisy, sr=self.sr,
                              snr_db=snr, category=cat)

        raise RuntimeError("EvalDataset has no mode set; use a classmethod.")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]