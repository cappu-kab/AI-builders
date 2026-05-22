"""
Calculate total duration of audio files across MULTIPLE dataset roots,
separated into "speech" and "noise" categories based on their folder/file
paths.

Usage:
    Edit the PATHS list below, then run:
        python audio_duration.py

Requirements:
    pip install librosa soundfile
"""

import os
import librosa

# -----------------------------------------------------------------------------
# 1) Define your dataset roots here. The script will scan each one recursively.
#    Add or remove paths as needed.
# -----------------------------------------------------------------------------
PATHS = [
    r"C:\Users\rocha\AI_builders\data_sounds\extracted\clean",
    r"C:\Users\rocha\AI_builders\data_sounds\extracted\noise",
    r"C:\Users\rocha\AI_builders\data_sounds\NoisySpeech_training",
    r"C:\Users\rocha\AI_builders\data_sounds\NoisySpeech_training_singleNoise",
    r"C:\Users\rocha\AI_builders\data_sounds\NoisySpeech_training_singleNoise1",
    r"C:\Users\rocha\AI_builders\data_sounds\raw\noise",
    r"C:\Users\rocha\AI_builders\data_sounds\raw\speech",
    # ...add more paths here
]

# Audio file extensions we'll look for
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def get_category(file_path):
    """
    Decide if a file belongs to 'speech' or 'noise' based on its path.

    Changed: instead of a single substring check on the whole path, we now
    inspect each folder component from the deepest one upward. The CLOSEST
    folder to the file that clearly says "speech" or "noise" wins.

    This correctly handles paths like:
        .../speech-noise-dataset/speech/file.wav  -> speech
        .../speech-noise-dataset/noise/file.wav   -> noise
        .../CleanSpeech_training/file.wav         -> speech
        .../Noise_training/file.wav               -> noise

    Folders that contain BOTH keywords (e.g. "speech-noise-dataset") are
    ambiguous and skipped so we keep walking up until a clearer one is found.
    Returns None if no folder along the path matches.
    """
    parts = os.path.normpath(file_path).split(os.sep)
    # Walk from the immediate parent folder up toward the root,
    # ignoring the file name itself (parts[-1]).
    for part in reversed(parts[:-1]):
        part_lower = part.lower()
        has_speech = "speech" in part_lower
        has_noise = "noise" in part_lower  # "noisy" does NOT match "noise"

        # Skip ambiguous folders that mention both
        if has_speech and has_noise:
            continue
        if has_speech:
            return "speech"
        if has_noise:
            return "noise"
    return None


def get_duration(file_path):
    """
    Return the duration of an audio file in seconds.
    Returns 0 if the file cannot be read.
    """
    try:
        # librosa.get_duration is fast — it reads metadata when possible
        return librosa.get_duration(path=file_path)
    except Exception as e:
        print(f"  [warning] Could not read {file_path}: {e}")
        return 0


def format_duration(seconds):
    """Format seconds as 'SS sec | MM min | HH hours'."""
    minutes = seconds / 60
    hours = seconds / 3600
    return f"{seconds:.2f} sec | {minutes:.2f} min | {hours:.2f} hours"


def scan_path(dataset_path, totals, counts):
    """
    Walk a single dataset root and update the running totals/counts in place.
    Returns the number of audio files skipped (no speech/noise keyword found).
    """
    skipped = 0
    if not os.path.isdir(dataset_path):
        print(f"  [warning] '{dataset_path}' is not a valid directory — skipped.")
        return skipped

    for root, _dirs, files in os.walk(dataset_path):
        for filename in files:
            if not filename.lower().endswith(AUDIO_EXTENSIONS):
                continue

            file_path = os.path.join(root, filename)
            category = get_category(file_path)

            if category is None:
                skipped += 1
                continue

            duration = get_duration(file_path)
            totals[category] += duration
            counts[category] += 1

    return skipped


def main():
    # Shared running totals across ALL dataset roots
    totals = {"speech": 0.0, "noise": 0.0}
    counts = {"speech": 0, "noise": 0}
    total_skipped = 0

    # Loop over each user-defined root path
    for path in PATHS:
        print(f"Scanning '{path}'...")
        total_skipped += scan_path(path, totals, counts)

    # ------------------- Final report -------------------
    print("\n--- Results ---")
    print(f"Speech files: {counts['speech']}")
    print(f"Total speech duration: {format_duration(totals['speech'])}")
    print(f"\nNoise files:  {counts['noise']}")
    print(f"Total noise duration:  {format_duration(totals['noise'])}")

    if total_skipped:
        print(f"\n(Skipped {total_skipped} audio files with no 'speech' or "
              f"'noise' keyword anywhere in their folder path.)")


if __name__ == "__main__":
    main()