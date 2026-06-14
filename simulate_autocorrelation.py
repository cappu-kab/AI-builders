"""
simulate_autocorrelation.py
===========================
Proof-of-concept: detect acoustic delay via cross-correlation and
align the phase before ANC playback.

Physical scenario:
  Anti-noise leaves the speaker -> travels through air -> arrives at
  the target zone LATE by some unknown number of samples.
  Cross-correlation finds that lag so we can pre-shift the signal.

Pipeline:
  original_signal
      |
      +-> [pad 150 zeros + noise] --> received_signal
      |
      +-> correlate(original, received)
              |
              +-> detected_lag --> aligned_signal
                                       |
                                       +-> |original - aligned| ~= 0 check
"""

import numpy as np
from scipy.signal import correlate, correlation_lags

# ── Reproducibility ─────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)

SR        = 16000   # sample rate (Hz)
DURATION  = 1.0     # seconds
TRUE_LAG  = 150     # samples of simulated acoustic delay
NOISE_AMP = 0.005   # microphone static amplitude (small vs. signal)

# ── 1. Generate original signal ──────────────────────────────────────────────
print("[1/5] Generating original signal (1 s sine wave at 440 Hz) ...")
t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
original_signal = np.sin(2 * np.pi * 440 * t)   # shape: (16000,)
print(f"      samples={len(original_signal)}  sr={SR} Hz  duration={DURATION}s")

# ── 2. Simulate acoustic delay ───────────────────────────────────────────────
print(f"\n[2/5] Simulating {TRUE_LAG}-sample acoustic delay + microphone noise ...")
padding = np.zeros(TRUE_LAG)
noise   = rng.uniform(-NOISE_AMP, NOISE_AMP, size=len(original_signal))

# Prepend zeros (delay) then trim back to original length so arrays match
received_signal = (
    np.concatenate([padding, original_signal])[:len(original_signal)] + noise
)
print(f"      noise amplitude : +/-{NOISE_AMP}  (signal peak: +/-{original_signal.max():.3f})")
print(f"      SNR             : ~{20 * np.log10(original_signal.std() / noise.std()):.1f} dB")

# ── 3. Cross-correlation ─────────────────────────────────────────────────────
print("\n[3/5] Running cross-correlation (original vs. received) ...")
correlation = correlate(original_signal, received_signal, mode="full")
lags        = correlation_lags(len(original_signal), len(received_signal), mode="full")

detected_lag = lags[np.argmax(correlation)]
print(f"      correlation peak index : {np.argmax(correlation)}")
print(f"      detected lag           : {detected_lag} samples")
print(f"      true lag               : {TRUE_LAG} samples")

lag_match = (detected_lag == TRUE_LAG)
print(f"      lag detection          : {'PASS' if lag_match else 'FAIL'} "
      f"({'exact match' if lag_match else f'off by {abs(detected_lag - TRUE_LAG)} samples'})")

# ── 4. Phase alignment ───────────────────────────────────────────────────────
print(f"\n[4/5] Aligning received_signal by shifting back {detected_lag} samples ...")

if detected_lag >= 0:
    # Received arrived late -> drop the leading delay samples
    aligned_signal = np.concatenate([
        received_signal[detected_lag:],
        np.zeros(detected_lag)
    ])
else:
    # Received arrived early -> prepend zeros to push it forward
    shift = -detected_lag
    aligned_signal = np.concatenate([
        np.zeros(shift),
        received_signal[:len(received_signal) - shift]
    ])

print(f"      aligned shape: {aligned_signal.shape}  original shape: {original_signal.shape}")

# ── 5. Verification ──────────────────────────────────────────────────────────
print("\n[5/5] Verification: |original - aligned| ...")
diff     = original_signal - aligned_signal
max_diff = np.abs(diff).max()
rms_diff = np.sqrt(np.mean(diff**2))

print(f"      max |diff| = {max_diff:.6f}")
print(f"      rms |diff| = {rms_diff:.6f}")
print(f"      noise floor (2x amp) = {2 * NOISE_AMP:.6f}  (expected upper bound)")

# Pass if residual is within ~2x injected noise floor
threshold = 2 * NOISE_AMP
if max_diff <= threshold:
    print(f"\n  PASS: Phase Alignment Successful!")
    print(f"  Residual max={max_diff:.6f} is within noise floor ({threshold:.4f}).")
    print("  Math is correct -- ready to apply pre-shift before ANC playback.")
else:
    print(f"\n  FAIL: Residual max={max_diff:.6f} exceeds noise floor ({threshold:.4f}).")
    print("  Check lag detection or alignment logic.")

print()
