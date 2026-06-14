"""
tune_compare.py — AGC current vs stronger on noisy input.
Usage: python tune_compare.py <input.wav>
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent.parent))
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python tune_compare.py <input.wav>")
    sys.exit(1)

in_path = sys.argv[1]
if not Path(in_path).exists():
    print(f"File not found: {in_path}")
    sys.exit(1)

from app import denoise_core

stem    = Path(in_path).stem
out_dir = denoise_core._OUT_DIR

# --- current AGC ---
print(f"\nInput: {in_path}\n")
print(f"--- AGC current (target={denoise_core._AGC_TARGET_RMS}  "
      f"release={denoise_core._AGC_RELEASE_MS}ms  boost={denoise_core._AGC_MAX_BOOST}x) ---")
tmp = denoise_core.process(in_path, dry_mix=0.05, gate=True, agc=True)
out_current = out_dir / f"{stem}_agc_current.wav"
Path(tmp).replace(out_current)
print(f"  -> {out_current}")

# Patch to stronger: higher target + faster release
denoise_core._AGC_TARGET_RMS = 0.25
denoise_core._AGC_RELEASE_MS = 200

print(f"\n--- AGC stronger (target={denoise_core._AGC_TARGET_RMS}  "
      f"release={denoise_core._AGC_RELEASE_MS}ms  boost={denoise_core._AGC_MAX_BOOST}x) ---")
tmp = denoise_core.process(in_path, dry_mix=0.05, gate=True, agc=True)
out_strong = out_dir / f"{stem}_agc_strong.wav"
Path(tmp).replace(out_strong)
print(f"  -> {out_strong}")

print("\nDone.")
