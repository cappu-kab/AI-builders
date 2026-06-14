"""
main_jetson_pipeline.py
=======================
Master ANC pipeline for Jetson Nano — AI Builders 2026 Showcase.

Hardware (not wired yet — swap placeholders when ready):
  - Sony Spresense   : roaming mic, USB serial
  - USB Sound Card   : line-in for reference mic + line-out to amp
  - 2x TPA3118 amp   : drives two 4-inch speakers

State machine:
  CALIBRATION -> NOISE_EXTRACTION -> ANTINOISE_GENERATION -> PLAYBACK_LOOP

Run:
  python main_jetson_pipeline.py
"""

import sys
import time
import numpy as np
from scipy.signal import correlate, correlation_lags

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE        = 16000   # Hz — must match model training SR
NOISE_RECORD_SECS  = 5       # seconds of ambient noise to record
PING_FREQ_HZ       = 1000    # calibration ping frequency
PING_DURATION_SECS = 0.5     # length of ping burst


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: HARDWARE I/O PLACEHOLDERS
# Swap these out for real hardware calls when wiring is complete.
# ─────────────────────────────────────────────────────────────────────────────

def record_from_spresense_serial(duration_secs: float) -> np.ndarray:
    """
    [PLACEHOLDER] Record audio from Sony Spresense over USB serial.

    Real implementation:
      - Open /dev/ttyUSB0 at 921600 baud
      - Read PCM16 frames for `duration_secs` seconds
      - Return float32 array normalised to [-1, 1]

    Returns dummy silence for now.
    """
    n_samples = int(SAMPLE_RATE * duration_secs)
    print(f"      [HW stub] Spresense: recording {duration_secs}s "
          f"({n_samples} samples) ...")
    time.sleep(0.1)  # simulate I/O latency
    return np.zeros(n_samples, dtype=np.float32)


def record_from_usb_soundcard(duration_secs: float) -> np.ndarray:
    """
    [PLACEHOLDER] Record from USB Sound Card line-in (reference mic).

    Real implementation:
      - Use sounddevice / pyaudio with device index for the USB card
      - Return float32 array normalised to [-1, 1]
    """
    n_samples = int(SAMPLE_RATE * duration_secs)
    print(f"      [HW stub] USB soundcard: recording {duration_secs}s ...")
    time.sleep(0.1)
    return np.zeros(n_samples, dtype=np.float32)


def play_to_usb_soundcard(audio_array: np.ndarray, loop: bool = False) -> None:
    """
    [PLACEHOLDER] Play audio through USB Sound Card -> TPA3118 amps -> speakers.

    Real implementation:
      - Use sounddevice.play(audio_array, samplerate=SAMPLE_RATE, loop=loop)
      - For loop=True: run in a background thread and block until KeyboardInterrupt
    """
    duration = len(audio_array) / SAMPLE_RATE
    print(f"      [HW stub] Playback: {len(audio_array)} samples "
          f"({duration:.3f}s)  loop={loop}")
    if loop:
        print("      [HW stub] Looping anti-noise (Ctrl+C to stop) ...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


def run_unet_denoiser(noisy_audio: np.ndarray) -> np.ndarray:
    """
    [PLACEHOLDER] Run the trained U-Net / CRN denoiser model.

    Real implementation:
      - Load ONNX model (crn_bilstm.onnx) with onnxruntime
      - Frame the input, run inference, overlap-add output
      - Return float32 clean audio, same length as input

    Returns the input unchanged (no denoising) for skeleton testing.
    """
    print(f"      [AI stub] Denoiser: processing {len(noisy_audio)} samples ...")
    time.sleep(0.1)
    return noisy_audio.copy()  # stub: pass-through


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: MATH UTILITIES  (verified in simulate_anc_math.py /
#                              simulate_autocorrelation.py)
# ─────────────────────────────────────────────────────────────────────────────

def generate_ping(freq_hz: float = PING_FREQ_HZ,
                  duration_secs: float = PING_DURATION_SECS,
                  sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Pure-tone burst used for acoustic delay calibration."""
    t = np.linspace(0, duration_secs, int(sample_rate * duration_secs), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def detect_acoustic_lag(reference: np.ndarray,
                         recorded: np.ndarray) -> int:
    """
    Cross-correlate `reference` (emitted) against `recorded` (mic pickup).
    Returns lag in samples: positive = recorded arrived late.

    Verified in simulate_autocorrelation.py.
    """
    corr = correlate(reference, recorded, mode="full")
    lags = correlation_lags(len(reference), len(recorded), mode="full")
    return int(lags[np.argmax(corr)])


def extract_noise_profile(noisy: np.ndarray,
                           clean: np.ndarray) -> np.ndarray:
    """
    Isolate the pure noise component.

    noise_profile = noisy - clean

    Verified in simulate_anc_math.py.
    """
    min_len = min(len(noisy), len(clean))
    return (noisy[:min_len] - clean[:min_len]).astype(np.float32)


def create_anti_noise(noise_profile: np.ndarray,
                      lag_samples: int) -> np.ndarray:
    """
    Phase-invert the noise profile, then pre-shift by `lag_samples` so the
    anti-noise arrives at the target zone exactly in phase with the real noise.

    anti_noise = roll(noise_profile * -1, -lag_samples)

    Positive lag means the speaker output needs to leave EARLY (negative roll).
    """
    inverted = noise_profile * -1.0
    aligned  = np.roll(inverted, -lag_samples)
    return aligned.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: PIPELINE STEPS
# ─────────────────────────────────────────────────────────────────────────────

def step1_calibration() -> int:
    """
    STEP 1 - CALIBRATION
    --------------------
    Emit a known ping from the speaker, record it at the target-zone mic,
    and use cross-correlation to measure the exact acoustic delay in samples.

    Returns:
        lag_samples (int): samples of travel delay from speaker to target zone.
    """
    print("\n" + "=" * 60)
    print("STEP 1: CALIBRATION -- Measuring Acoustic Delay")
    print("=" * 60)
    print("  Generating calibration ping ...")
    ping = generate_ping()

    print("  Playing ping through speakers ...")
    play_to_usb_soundcard(ping, loop=False)

    print("  Recording ping arrival at target-zone mic ...")
    recorded_ping = record_from_spresense_serial(PING_DURATION_SECS + 0.2)

    lag = detect_acoustic_lag(ping, recorded_ping[:len(ping)])
    print(f"\n  Detected acoustic lag : {lag} samples "
          f"({lag / SAMPLE_RATE * 1000:.2f} ms)")
    print(f"  Speed-of-sound dist   : ~{lag / SAMPLE_RATE * 343:.2f} cm")
    print("  CALIBRATION COMPLETE")
    return lag


def step2_noise_extraction() -> np.ndarray:
    """
    STEP 2 - NOISE EXTRACTION
    -------------------------
    Record ambient noise, run the AI denoiser, subtract to isolate the pure
    noise component. This is the 'quiet room with only AC hum' phase.

    Returns:
        noise_profile (np.ndarray): isolated noise waveform.
    """
    print("\n" + "=" * 60)
    print("STEP 2: NOISE EXTRACTION -- Recording Ambient Noise")
    print("=" * 60)
    print(f"  Recording {NOISE_RECORD_SECS}s of ambient noise via Spresense ...")
    noisy_audio = record_from_spresense_serial(NOISE_RECORD_SECS)

    print("  Running AI denoiser ...")
    clean_audio = run_unet_denoiser(noisy_audio)

    print("  Extracting noise profile: noise = noisy - clean ...")
    noise_profile = extract_noise_profile(noisy_audio, clean_audio)

    rms = np.sqrt(np.mean(noise_profile ** 2))
    print(f"  Noise profile RMS : {rms:.6f}")
    print("  NOISE EXTRACTION COMPLETE")
    return noise_profile


def step3_antinoise_generation(noise_profile: np.ndarray,
                                lag_samples: int) -> np.ndarray:
    """
    STEP 3 - ANTI-NOISE GENERATION
    --------------------------------
    Phase-invert the noise profile and pre-shift it so the cancellation
    wave arrives in perfect anti-phase at the target zone.

    Returns:
        anti_noise (np.ndarray): speaker-ready cancellation waveform.
    """
    print("\n" + "=" * 60)
    print("STEP 3: ANTI-NOISE GENERATION")
    print("=" * 60)
    print("  Phase inverting noise profile (* -1) ...")
    print(f"  Pre-shifting by -{lag_samples} samples to compensate travel delay ...")
    anti_noise = create_anti_noise(noise_profile, lag_samples)

    peak = np.abs(anti_noise).max()
    print(f"  Anti-noise peak amplitude : {peak:.6f}")
    print("  ANTI-NOISE READY")
    return anti_noise


def step4_playback_loop(anti_noise: np.ndarray) -> None:
    """
    STEP 4 - PLAYBACK LOOP
    ----------------------
    Continuously play the aligned anti-noise through both TPA3118 amplifiers.
    Press Ctrl+C to stop.
    """
    print("\n" + "=" * 60)
    print("STEP 4: PLAYBACK LOOP -- ANC Active")
    print("=" * 60)
    print("  Anti-noise is now playing in a loop.")
    print("  Press Ctrl+C to stop.\n")
    play_to_usb_soundcard(anti_noise, loop=True)
    print("\n  Playback stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Thai Speech Denoiser -- ANC Pipeline")
    print("  AI Builders 2026 | Jetson Nano")
    print("=" * 60)
    print("\nThis script will guide you through 4 steps.")
    print("Hardware stubs are active -- swap placeholder functions for")
    print("real I/O when wiring is complete.\n")

    # ── Step 1: Calibration ──────────────────────────────────────────────
    input(">>> Place the Spresense mic at the TARGET ZONE (the chair/listener).\n"
          "    Then press Enter to begin Phase Calibration ... ")
    lag_samples = step1_calibration()

    # ── Step 2: Noise Extraction ─────────────────────────────────────────
    input("\n>>> Keep the mic at the target zone. Make sure the room has\n"
          "    only ambient noise (AC, fan) -- no speech.\n"
          "    Press Enter to begin Noise Recording ... ")
    noise_profile = step2_noise_extraction()

    # ── Step 3: Anti-Noise Generation ────────────────────────────────────
    input("\n>>> Press Enter to generate the Anti-Noise wave ... ")
    anti_noise = step3_antinoise_generation(noise_profile, lag_samples)

    # ── Step 4: Playback Loop ────────────────────────────────────────────
    input("\n>>> Move the Spresense mic back to its normal position (on you).\n"
          "    Press Enter to START Anti-Noise playback (Ctrl+C to stop) ... ")
    step4_playback_loop(anti_noise)

    print("\nPipeline finished. Goodbye.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Pipeline stopped.")
        sys.exit(0)
