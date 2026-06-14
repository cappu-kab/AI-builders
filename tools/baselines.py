"""
Non-learned baselines used for sanity-checking the trained model.

Every baseline implements the same interface as a model: it is a callable
that maps (noisy_waveform, sr) -> enhanced_waveform.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
import librosa


class IdentityBaseline:
    """Pass-through. Useful as a floor reference."""
    name = "identity"

    def __call__(self, noisy: np.ndarray, sr: int = 16000) -> np.ndarray:
        return noisy.astype(np.float32, copy=True)


class HighPassBaseline:
    """Butterworth high-pass filter. Trivially attenuates everything below cutoff."""
    name = "highpass_200"

    def __init__(self, cutoff: float = 200.0, order: int = 4) -> None:
        self.cutoff = cutoff
        self.order = order

    def __call__(self, noisy: np.ndarray, sr: int = 16000) -> np.ndarray:
        sos = butter(self.order, self.cutoff / (sr / 2.0),
                     btype="highpass", output="sos")
        return sosfiltfilt(sos, noisy).astype(np.float32, copy=False)


class WienerBaseline:
    """
    Spectral-subtraction-style Wiener filter.

    Noise PSD is estimated from the first `noise_estimate_seconds` of the
    input signal (assumed to be relatively speech-free). If you have an
    oracle noise signal you can pass it via `oracle_noise=` for an
    upper-bound reference.
    """
    name = "wiener"

    def __init__(self,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 noise_estimate_seconds: float = 0.5,
                 floor: float = 0.05) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_estimate_seconds = noise_estimate_seconds
        self.floor = floor  # min gain to avoid musical noise

    def __call__(self,
                 noisy: np.ndarray,
                 sr: int = 16000,
                 oracle_noise: np.ndarray | None = None) -> np.ndarray:
        S = librosa.stft(noisy, n_fft=self.n_fft, hop_length=self.hop_length)
        power = np.abs(S) ** 2

        if oracle_noise is not None:
            N = librosa.stft(oracle_noise,
                             n_fft=self.n_fft, hop_length=self.hop_length)
            noise_psd = np.mean(np.abs(N) ** 2, axis=1, keepdims=True)
        else:
            n_init = max(1, int(self.noise_estimate_seconds * sr / self.hop_length))
            n_init = min(n_init, power.shape[1])
            noise_psd = np.mean(power[:, :n_init], axis=1, keepdims=True)

        gain = np.maximum(1.0 - noise_psd / (power + 1e-12), self.floor)
        S_enh = gain * S
        y = librosa.istft(S_enh, hop_length=self.hop_length, length=len(noisy))
        return y.astype(np.float32, copy=False)
