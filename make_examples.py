"""
Generate 3 example WAV files for HF Space demo.
Thai speech from FLEURS + bandpass LF noise (20-300 Hz) at ~6-8 dB SNR, -20 dB RMS output.
"""
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from datasets import load_dataset

OUTPUT_DIR = r"C:\Users\rocha\AI_builders\hf_deploy\examples"
SAMPLE_RATE = 16000
TARGET_RMS_DB = -20.0
SNR_DB = 7.0  # middle of 6-8 dB range
MIN_DUR = 8.0
MAX_DUR = 15.0


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def rms_db(x):
    r = rms(x)
    return 20 * np.log10(r) if r > 1e-9 else -120.0


def normalize_rms(x, target_db):
    r = rms(x)
    if r < 1e-9:
        return x
    return x * (10 ** (target_db / 20.0) / r)


def make_lf_noise(n_samples, sr=SAMPLE_RATE, lo=20, hi=300):
    """Bandpass white noise 20-300 Hz."""
    white = np.random.randn(n_samples).astype(np.float32)
    sos = butter(6, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, white).astype(np.float32)


def mix_at_snr(speech, noise, snr_db):
    """Mix noise into speech at given SNR (dB)."""
    speech_rms = rms(speech)
    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    noise_rms = rms(noise)
    if noise_rms < 1e-9:
        return speech
    noise_scaled = noise * (target_noise_rms / noise_rms)
    if len(noise_scaled) > len(speech):
        noise_scaled = noise_scaled[:len(speech)]
    else:
        noise_scaled = np.pad(noise_scaled, (0, len(speech) - len(noise_scaled)))
    return speech + noise_scaled


def trim_or_pad(audio, min_dur, max_dur, sr):
    min_s, max_s = int(min_dur * sr), int(max_dur * sr)
    if len(audio) < min_s:
        audio = np.pad(audio, (0, min_s - len(audio)))
    if len(audio) > max_s:
        audio = audio[:max_s]
    return audio


def main():
    from math import gcd
    from scipy.signal import resample_poly
    import io

    print("Loading FLEURS th_th test split (decode=False)...")
    ds = load_dataset("google/fleurs", "th_th", split="test")
    ds = ds.cast_column("audio", __import__("datasets").Audio(decode=False))
    print(f"Dataset size: {len(ds)}")

    def load_audio(item):
        audio = item["audio"]
        raw = audio.get("bytes") or open(audio["path"], "rb").read()
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        if sr != SAMPLE_RATE:
            g = gcd(SAMPLE_RATE, sr)
            arr = resample_poly(arr, SAMPLE_RATE // g, sr // g).astype(np.float32)
        return arr

    chosen = []
    for item in ds:
        arr = load_audio(item)
        dur = len(arr) / SAMPLE_RATE
        if MIN_DUR <= dur <= MAX_DUR:
            chosen.append(arr)
        if len(chosen) == 3:
            break

    if len(chosen) < 3:
        print(f"Only {len(chosen)} utterances in 8-15s range. Falling back to trim/pad.")
        for item in ds:
            if len(chosen) == 3:
                break
            arr = load_audio(item)
            arr = trim_or_pad(arr, MIN_DUR, MAX_DUR, SAMPLE_RATE)
            chosen.append(arr)

    for i, speech in enumerate(chosen, 1):
        speech = trim_or_pad(speech, MIN_DUR, MAX_DUR, SAMPLE_RATE)
        noise = make_lf_noise(len(speech))
        mix = mix_at_snr(speech, noise, SNR_DB)
        mix = normalize_rms(mix, TARGET_RMS_DB)
        mix = np.clip(mix, -1.0, 1.0)

        out_path = f"{OUTPUT_DIR}/example_thai_speech_{i}.wav"
        sf.write(out_path, mix, SAMPLE_RATE, subtype="PCM_16")
        print(f"Wrote {out_path}  dur={len(mix)/SAMPLE_RATE:.1f}s  RMS={rms_db(mix):.1f} dB")


if __name__ == "__main__":
    main()
