import sys
from pathlib import Path as _P
_REPO = _P(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'models' / 'crn'))
sys.path.insert(0, str(_P(__file__).parent))
"""Synthetic end-to-end smoke test for the evaluation pipeline."""
import os
import shutil
import numpy as np
import soundfile as sf

from baselines import HighPassBaseline, IdentityBaseline, WienerBaseline
from dataset import EvalDataset
from evaluate import evaluate_dataset

SR = 16000
ROOT = "/tmp/smoke_anc_eval"
CLEAN_DIR = os.path.join(ROOT, "clean")
NOISE_DIR = os.path.join(ROOT, "noise")
OUT_DIR = os.path.join(ROOT, "results")

shutil.rmtree(ROOT, ignore_errors=True)
os.makedirs(CLEAN_DIR); os.makedirs(NOISE_DIR)

rng = np.random.default_rng(0)

# Build "clean" -> band-limited 200..3000 Hz signals to mimic speech band.
def synth_clean(seconds=2.0):
    t = np.arange(int(SR*seconds))/SR
    s = np.zeros_like(t)
    for f in [220.0, 440.0, 880.0, 1760.0]:
        s += np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi))
    s = s / np.max(np.abs(s)+1e-8) * 0.3
    return s.astype(np.float32)

# Low-frequency rumble: a 60 Hz tone + filtered low-band noise.
def synth_lowfreq_noise(seconds=2.0):
    t = np.arange(int(SR*seconds))/SR
    rumble = 0.5*np.sin(2*np.pi*60*t) + 0.3*np.sin(2*np.pi*120*t)
    noise = rng.standard_normal(t.shape).astype(np.float32)
    # crude low-pass: cumulative average smoother
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, 200/(SR/2), btype="lowpass", output="sos")
    noise = sosfiltfilt(sos, noise).astype(np.float32)
    out = (0.5*rumble + 1.0*noise).astype(np.float32)
    return (out / (np.max(np.abs(out))+1e-8) * 0.5).astype(np.float32)

for i in range(3):
    sf.write(os.path.join(CLEAN_DIR, f"clean_{i}.wav"), synth_clean(), SR)
    sf.write(os.path.join(NOISE_DIR, f"noise_{i}.wav"), synth_lowfreq_noise(), SR)

dataset = EvalDataset.from_dirs(
    clean_dir=CLEAN_DIR, noise_dir=NOISE_DIR,
    sr=SR, snrs_db=[0, 5], n_per_clean=1, seed=42,
)
print(f"Loaded {len(dataset)} samples")

enhancers = {
    "identity":     IdentityBaseline(),
    "highpass_200": HighPassBaseline(200.0),
    "wiener":       WienerBaseline(),
}

rows, summary = evaluate_dataset(
    enhancers=enhancers,
    dataset=dataset,
    out_dir=OUT_DIR,
    compute_pesq_stoi=False,
    save_audio=True,
    n_visualize=2,
)
print("rows:", len(rows))
print("summary keys:", list(summary.keys()))
print("CSV exists:", os.path.exists(os.path.join(OUT_DIR, "results.csv")))
print("audio dir items:", len(os.listdir(os.path.join(OUT_DIR, "audio"))))
print("figures:", len(os.listdir(os.path.join(OUT_DIR, "figures"))))
