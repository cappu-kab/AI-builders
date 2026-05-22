"""
Mock noise-estimation model for development and hardware-absent testing.

Exposes the same `.predict()` interface as ModelRuntime so it can be swapped
transparently.  Internally applies a Butterworth low-pass filter as a
stand-in for the real AI model — it isolates the low-frequency band that
the trained network would learn to extract.
"""
from __future__ import annotations

import numpy as np

from utils.audio_utils import apply_lowpass
from utils.logger_setup import get_logger

log = get_logger(__name__)


class MockNoiseEstimator:
    """
    Drop-in replacement for a real noise-estimation model.

    Behaviour: passes the input chunk through a low-pass Butterworth filter
    (< LOW_FREQ_CUTOFF_HZ Hz) to produce a synthetic noise estimate.
    This is acoustically plausible for low-frequency engine noise and gives
    a non-zero signal to validate the full pipeline end-to-end.
    """

    def __init__(self) -> None:
        log.warning(
            "MockNoiseEstimator loaded — output is LP-filtered audio, NOT a "
            "trained ANC model.  Set USE_MOCK_MODEL=False to load the real checkpoint."
        )

    def predict(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Args:
            audio_chunk: 1-D float32 array, shape (CHUNK_SIZE,), range [-1, 1].

        Returns:
            noise_estimate: 1-D float32 array, same shape — low-frequency content only.
        """
        return apply_lowpass(audio_chunk)
