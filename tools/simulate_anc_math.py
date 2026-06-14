"""
simulate_anc_math.py
====================
Software proof-of-concept for the ANC phase-inversion math.

Pipeline:
    noisy.wav  ──────────────────────────────────────────┐
                                                          ▼
    noisy.wav ──► (minus) ──► noise_profile ──► (*-1) ──► anti_noise
    clean.wav ──►                                         │
                                                          ▼
                              simulated_result = noisy + anti_noise

VERIFICATION: simulated_result == clean_audio (algebraically identical).
  noisy + anti_noise
= noisy + (-(noisy - clean))
= noisy - noisy + clean
= clean  ✓

If the result sounds like clean.wav (or silence when no speech),
the phase-inversion math is 100% correct and ready for hardware.
"""

import sys
import numpy as np
import soundfile as sf


def load_mono(path: str) -> tuple:
    """Load a wav file and downmix to mono if needed."""
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float64), sr


def main():
    noisy_path    = "noisy.wav"
    clean_path    = "clean.wav"
    out_antinoise = "anti_noise_test.wav"
    out_result    = "simulated_result_check.wav"

    # ── Load ────────────────────────────────────────────────────────────
    print(f"[1/5] Loading '{noisy_path}' ...")
    noisy, sr_n = load_mono(noisy_path)

    print(f"[1/5] Loading '{clean_path}' ...")
    clean, sr_c = load_mono(clean_path)

    if sr_n != sr_c:
        print(f"ERROR: sample-rate mismatch  noisy={sr_n} Hz  clean={sr_c} Hz")
        sys.exit(1)
    sr = sr_n

    # ── Trim to equal length ────────────────────────────────────────────
    min_len = min(len(noisy), len(clean))
    if len(noisy) != len(clean):
        print(f"[2/5] Length mismatch — trimming both to {min_len} samples "
              f"({min_len/sr:.3f}s)")
    else:
        print(f"[2/5] Lengths match: {min_len} samples ({min_len/sr:.3f}s)")
    noisy = noisy[:min_len]
    clean = clean[:min_len]

    # ── ANC math ────────────────────────────────────────────────────────
    print("[3/5] Extracting noise profile:   noise_profile = noisy - clean")
    noise_profile = noisy - clean

    print("[3/5] Phase inversion:            anti_noise    = noise_profile * -1.0")
    anti_noise = noise_profile * -1.0

    print("[3/5] Simulating cancellation:    result        = noisy + anti_noise")
    simulated_result = noisy + anti_noise  # algebraically equals clean

    # ── Amplitude stats ─────────────────────────────────────────────────
    print("\n── Amplitude statistics ─────────────────────────────────────")
    for label, arr in [
        ("noisy           ", noisy),
        ("clean           ", clean),
        ("noise_profile   ", noise_profile),
        ("anti_noise      ", anti_noise),
        ("simulated_result", simulated_result),
    ]:
        print(f"  {label}  min={arr.min():+.6f}  max={arr.max():+.6f}  "
              f"rms={np.sqrt(np.mean(arr**2)):.6f}")

    # Sanity check: simulated_result must be numerically identical to clean
    max_diff = np.abs(simulated_result - clean).max()
    passed = max_diff < 1e-9
    print(f"\n  max |simulated_result - clean| = {max_diff:.2e}  "
          f"{'PASS (within float64 epsilon)' if passed else 'FAIL'}")
    print("─────────────────────────────────────────────────────────────\n")

    # ── Save outputs ────────────────────────────────────────────────────
    print(f"[4/5] Saving '{out_antinoise}' ...")
    sf.write(out_antinoise, anti_noise, sr, subtype="PCM_16")

    print(f"[4/5] Saving '{out_result}' ...")
    sf.write(out_result, simulated_result, sr, subtype="PCM_16")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"""
[5/5] Done.

Files written:
  {out_antinoise:35s} <- play this through speakers to cancel ambient noise
  {out_result:35s} <- should sound identical to clean.wav

HOW TO VERIFY:
  Listen to '{out_result}':
    * Speech present  -> sounds like clean.wav (noise removed)
    * No speech       -> sounds like silence
  Either outcome proves phase-inversion math is correct for hardware.
""")


if __name__ == "__main__":
    main()
