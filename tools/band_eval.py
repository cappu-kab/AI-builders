"""
Frequency-band evaluation utilities.

This is the core module for the project's stated goal: measuring
noise reduction in the 0-200 Hz band specifically.

Convention:
    `band_noise_reduction_db` measures the residual-noise reduction
    of (enhanced - clean) vs. (noisy - clean) within each band.
    A positive value means the model attenuated the noise in that band.
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
import librosa


# Default band layout — focused on the project's <200 Hz priority.
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "low_0_200":  (0.0,    200.0),
    "mid_200_1k": (200.0,  1000.0),
    "high_1k_8k": (1000.0, 8000.0),
}


def _stft_power(signal: np.ndarray,
                sr: int,
                n_fft: int,
                hop_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (|STFT|^2, freqs)."""
    S = librosa.stft(signal,
                     n_fft=n_fft,
                     hop_length=hop_length,
                     win_length=n_fft,
                     center=True)
    power = np.abs(S) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return power, freqs


def _band_energy(power: np.ndarray,
                 freqs: np.ndarray,
                 low: float,
                 high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(power[mask, :].sum())


def band_noise_reduction_db(noisy: np.ndarray,
                            enhanced: np.ndarray,
                            clean: np.ndarray,
                            sr: int = 16000,
                            bands: Mapping[str, Tuple[float, float]] | None = None,
                            n_fft: int = 1024,
                            hop_length: int = 256,
                            eps: float = 1e-12) -> Dict[str, float]:
    """
    Per-band noise attenuation in dB.

        NR_band = 10 * log10( E_band(noisy - clean) / E_band(enhanced - clean) )

    Higher is better. Uses identical lengths for all three signals.
    """
    bands = bands or DEFAULT_BANDS
    L = min(len(noisy), len(enhanced), len(clean))
    noise_in = noisy[:L] - clean[:L]
    noise_out = enhanced[:L] - clean[:L]

    p_in, freqs = _stft_power(noise_in, sr, n_fft, hop_length)
    p_out, _ = _stft_power(noise_out, sr, n_fft, hop_length)

    out: Dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        e_in = _band_energy(p_in, freqs, lo, hi)
        e_out = _band_energy(p_out, freqs, lo, hi)
        out[name] = 10.0 * np.log10((e_in + eps) / (e_out + eps))
    return out


def band_energy_breakdown(signal: np.ndarray,
                          sr: int = 16000,
                          bands: Mapping[str, Tuple[float, float]] | None = None,
                          n_fft: int = 1024,
                          hop_length: int = 256,
                          eps: float = 1e-12) -> Dict[str, float]:
    """Per-band energy of a single signal in dB. Useful for plotting."""
    bands = bands or DEFAULT_BANDS
    power, freqs = _stft_power(signal, sr, n_fft, hop_length)
    return {
        name: 10.0 * np.log10(_band_energy(power, freqs, lo, hi) + eps)
        for name, (lo, hi) in bands.items()
    }


def speech_distortion_db(enhanced: np.ndarray,
                         clean: np.ndarray,
                         sr: int = 16000,
                         bands: Mapping[str, Tuple[float, float]] | None = None,
                         n_fft: int = 1024,
                         hop_length: int = 256,
                         eps: float = 1e-12) -> Dict[str, float]:
    """
    Per-band speech distortion in dB:
        SD_band = 10 * log10( E_band(enhanced - clean) / E_band(clean) )

    Lower (more negative) means less distortion of the speech.
    Useful to spot a model that 'wins' noise reduction by attenuating
    speech as well.
    """
    bands = bands or DEFAULT_BANDS
    L = min(len(enhanced), len(clean))
    diff = enhanced[:L] - clean[:L]

    p_diff, freqs = _stft_power(diff, sr, n_fft, hop_length)
    p_clean, _ = _stft_power(clean[:L], sr, n_fft, hop_length)

    out: Dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        e_d = _band_energy(p_diff, freqs, lo, hi)
        e_c = _band_energy(p_clean, freqs, lo, hi)
        out[name] = 10.0 * np.log10((e_d + eps) / (e_c + eps))
    return out
