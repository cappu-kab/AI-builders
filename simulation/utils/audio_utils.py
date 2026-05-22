"""
Signal-processing helpers for ANC Phase 1.

All functions operate on 1-D float32 numpy arrays normalised to [-1.0, 1.0].
They are pure (no side effects) and safe to call from any thread.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

from configs.config import LOW_FREQ_CUTOFF_HZ, SAMPLE_RATE

# ── Pre-build the Butterworth low-pass filter once at import time ──────────────
_LP_ORDER = 4
_NYQUIST  = SAMPLE_RATE / 2.0

if LOW_FREQ_CUTOFF_HZ >= _NYQUIST:
    raise ValueError(
        f"LOW_FREQ_CUTOFF_HZ ({LOW_FREQ_CUTOFF_HZ} Hz) must be less than "
        f"Nyquist frequency ({_NYQUIST} Hz)."
    )

_LP_B, _LP_A = butter(_LP_ORDER, LOW_FREQ_CUTOFF_HZ / _NYQUIST, btype="low")
_LP_ZI       = lfilter_zi(_LP_B, _LP_A)   # initial conditions for zero-phase start


def apply_lowpass(audio: np.ndarray) -> np.ndarray:
    """
    Band-limit `audio` to the ANC target band (<= LOW_FREQ_CUTOFF_HZ).

    Uses a causal IIR filter (no look-ahead delay) to stay compatible with
    real-time block processing.
    """
    filtered, _ = lfilter(_LP_B, _LP_A, audio, zi=_LP_ZI * audio[0])
    return filtered.astype(np.float32)


def phase_invert(noise_estimate: np.ndarray) -> np.ndarray:
    """
    Generate the anti-noise signal via destructive interference.

    Anti-noise = noise_estimate × -1  →  cancels the noise in physical air space
    when the speaker is positioned at the error microphone location.
    """
    return (noise_estimate * -1.0).astype(np.float32)


def clip_output(signal: np.ndarray, ceiling: float = 1.0) -> np.ndarray:
    """
    Hard-clip to [-ceiling, ceiling] to protect the MAX98357A speaker hardware.
    Any model prediction that saturates the DAC could damage the amplifier.
    """
    return np.clip(signal, -ceiling, ceiling).astype(np.float32)


def is_silent(chunk: np.ndarray, threshold: float = 1e-5) -> bool:
    """Return True if the chunk is effectively silent (all-zero or below threshold)."""
    return float(np.max(np.abs(chunk))) < threshold
