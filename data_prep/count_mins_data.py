import os
import wave

PATHS = [
    r"C:\Users\rocha\AI_builders\data_sounds"
]

speech_duration = 0
noise_duration = 0
noisy_duration = 0

def get_category(path):
    path = path.lower()

    if "clean" in path:
        return None
    if "test" in path:
        return None
    # if "noisy" in path:
    #     return None

    if "speech" in path:
        return "speech"
    if "noise" in path:
        return "noise"
    if "noisy" in path:
        return "noisy"

    return None


def get_wav_duration(file_path):
    try:
        with wave.open(file_path, 'r') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except:
        return 0


for root_path in PATHS:
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if not file.endswith(".wav"):
                continue

            full_path = os.path.join(root, file)
            category = get_category(full_path)

            if category is None:
                continue

            duration = get_wav_duration(full_path)

            if category == "speech":
                speech_duration += duration
            elif category == "noise":
                noise_duration += duration
            elif category == "noisy":
                noisy_duration += duration


def format_time(sec):
    return f"{sec:.2f} sec | {sec/60:.2f} min | {sec/3600:.2f} hr"


print("Speech:", format_time(speech_duration))
print("Noise :", format_time(noise_duration))