"""
ANC Phase 1 — Hardware Configuration
Target: Jetson Nano + INMP441 (I2S mic) + MAX98357A (I2S DAC/Amp)

ALL VALUES MARKED [DUMMY] MUST BE REPLACED BEFORE DEPLOYMENT.
See README.md → "Finding Real Hardware Values" for shell commands.
"""

# ══════════════════════════════════════════════════════════════
# AUDIO DEVICE INDICES  — [DUMMY]
# Replace with actual ALSA card indices found via:
#   python -m sounddevice
#   arecord -l   (for input/capture devices)
#   aplay   -l   (for output/playback devices)
# ══════════════════════════════════════════════════════════════
INPUT_DEVICE_INDEX  = 2   # [DUMMY] INMP441 I2S mic ALSA card index
OUTPUT_DEVICE_INDEX = 3   # [DUMMY] MAX98357A I2S DAC ALSA card index

# Alternatively, sounddevice also accepts ALSA name strings, e.g.:
#   INPUT_DEVICE_INDEX  = "hw:2,0"
#   OUTPUT_DEVICE_INDEX = "hw:3,0"

# ══════════════════════════════════════════════════════════════
# AUDIO STREAM PARAMETERS
# ══════════════════════════════════════════════════════════════
SAMPLE_RATE  = 16_000   # Hz — must match the model's training sample rate
CHUNK_SIZE   = 256      # Samples per processing block → 16 ms latency @ 16 kHz.
                        # Tight real-time target for Jetson Orin Nano (the Orin
                        # has enough headroom to run the ~30–60 ms CRN inference
                        # per call, which now exceeds the 16 ms chunk period and
                        # is absorbed by the streaming pipeline's deadline-sleep).
                        # Streaming NR is ~0.6 dB lower than at CHUNK_SIZE=1024
                        # because of higher per-chunk model variance.  See v5
                        # results in history.md §2 for the trade-off.
NUM_CHANNELS = 1        # Mono — INMP441 is a single-channel mic
DTYPE        = "float32"

# ══════════════════════════════════════════════════════════════
# WARM-UP MUTE
# ══════════════════════════════════════════════════════════════
# Anti-noise output is forced to zeros for the first WARMUP_CHUNKS
# to prevent cold-start model artefacts from reaching the speaker.
# Model inference still runs during this period to populate the
# context buffer and RNN hidden states.
# Sized to cover the CRN's 47616-sample sliding context, i.e.
# ceil(47872 / CHUNK_SIZE) chunks, with a small margin.
# 200 × 256 / 16000 ≈ 3.2 s   (matches v4/v5 warmup duration at 1024-sample
# chunks).  Re-tune if CHUNK_SIZE or _CRN_CTX_N changes.
WARMUP_CHUNKS = 200

# ══════════════════════════════════════════════════════════════
# FILE SIMULATOR — HARDWARE PHYSICS PARAMETERS
# ══════════════════════════════════════════════════════════════
# Acoustic propagation jitter: random per-chunk delay applied to
# the anti-noise path to simulate speaker→mic travel time variation.
# At 16 kHz: 8 smp ≈ 0.5 ms, 24 smp ≈ 1.5 ms (1–3 cm at 343 m/s)
SIM_PROP_DELAY_MIN  = 8    # samples
SIM_PROP_DELAY_MAX  = 24   # samples

# Acoustic feedback: fraction of anti-noise output injected back
# into the next mic input chunk to simulate speaker→mic echo leakage.
SIM_FEEDBACK_SCALE  = 0.003

# ══════════════════════════════════════════════════════════════
# AI MODEL  — [DUMMY]
# ══════════════════════════════════════════════════════════════
MODEL_PATH     = "/home/jetson/ANC_Hardware_Test/checkpoints/anc_noise_est.onnx"  # [DUMMY]
USE_MOCK_MODEL = True   # Set to False once the real ONNX checkpoint is deployed

# Expected ONNX I/O shapes (adjust if your model differs):
#   Input  name: "input"   shape: [1, 1, CHUNK_SIZE]  dtype: float32
#   Output name: "output"  shape: [1, 1, CHUNK_SIZE]  dtype: float32
MODEL_INPUT_NAME  = "input"
MODEL_OUTPUT_NAME = "output"

# ══════════════════════════════════════════════════════════════
# ANC SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════
LOW_FREQ_CUTOFF_HZ     = 200    # Hz — ANC targets engine noise below this frequency
PHASE_INVERSION_FACTOR = -1.0   # Anti-noise = noise_estimate × (-1)

# Block-boundary cross-fade length (samples) for the streaming noise estimate.
# Set to 0 with the look-ahead extraction in model_runtime._infer_crn_stateful:
# extracted segments come from the model's iSTFT interior (not the reflect-pad
# boundary), so consecutive chunks are already continuous and the cross-fade
# would just ramp the leading samples of cancellation toward zero.
# Set 1..CHUNK_SIZE to re-enable the smoothing ramp if a future model has
# residual chunk-boundary discontinuities (must satisfy 0 <= N <= CHUNK_SIZE).
ANC_XFADE_SAMPLES = 0

# ══════════════════════════════════════════════════════════════
# RESIDUAL NOISE GATE
# ══════════════════════════════════════════════════════════════
# Per-chunk gate applied to clean_chunk *after* cancellation, to push the
# residual noise floor toward silence during noise-only periods without
# touching speech segments.
#
# Detection signal: ratio = rms(clean_chunk) / rms(raw).  When the model
# cancels well (pure noise), the ratio is small (~0.2).  When speech is
# present (model preserves it), the ratio is larger (~0.5+).
#
# Envelope: immediate release (gate opens the first chunk a speech burst
# arrives) + smoothed attack (gate closes gradually so brief noise lulls
# don't pump audibly).
NOISE_GATE_ENABLE         = True
NOISE_GATE_RATIO_LOW      = 0.30   # ratio <= LOW  -> full attenuation
NOISE_GATE_RATIO_HIGH     = 0.45   # ratio >= HIGH -> no attenuation
NOISE_GATE_ATTENUATION_DB = -15.0  # extra suppression applied when fully gated

# Wall-clock attack time (gate closing).  The simulator converts this to a
# per-chunk smoothing coefficient at startup:
#     alpha = 1 - exp(-CHUNK_PERIOD_MS / NOISE_GATE_ATTACK_MS)
# Empirically tuned at CHUNK=256 to alpha ≈ 0.20 (-> 72 ms wall-clock).  This
# yields the best NR on real noise at 16 ms chunks; longer attack times let
# each false release leak too much energy back into the output.  At larger
# chunks (e.g. 1024) the same alpha emerges from a longer wall-clock setting
# (~290 ms), but the empirically optimal alpha for that case still works out
# to ~0.20.  Treat this knob as a per-CHUNK_SIZE empirical tuning point.
NOISE_GATE_ATTACK_MS      = 72.0

# Wall-clock smoothing of the detection signal (the clean/raw RMS ratio).
# Set very small (≈ 1 ms) to disable: at any reasonable CHUNK_PERIOD the
# resulting alpha rounds to 1.0, so the smoothed ratio = raw ratio.  Disabled
# by default because experiments at CHUNK=256 showed that smoothing the
# detection signal delays the gate's response to genuine noise drops more
# than it suppresses false releases on amplitude bursts, net hurting NR.
# Raise this knob (to e.g. 60–100 ms) if a future signal class shows the
# opposite trade-off.
NOISE_GATE_RATIO_SMOOTH_MS = 1.0

# ══════════════════════════════════════════════════════════════
# STREAM HEALTH & FAULT TOLERANCE
# ══════════════════════════════════════════════════════════════
STREAM_LATENCY           = "low"  # sounddevice latency hint to PortAudio
MAX_CONSECUTIVE_ERRORS   = 5      # Abort pipeline after this many back-to-back errors
OVERFLOW_LOG_THROTTLE_S  = 2.0    # Minimum seconds between repeated overflow warnings
QUEUE_MAXSIZE            = 8      # Max buffered chunks in input/output queues
HEALTH_CHECK_INTERVAL_S  = 0.5    # Seconds between stream health checks in main loop

# ══════════════════════════════════════════════════════════════
# VISUALIZATION DASHBOARD  (PyQtGraph)
# ══════════════════════════════════════════════════════════════
VIS_FPS          = 30    # Target dashboard refresh rate (Hz). 30 is safe on Jetson Nano.
                         # Raise to 60 only if the GPU/display pipeline can sustain it.
VIS_QUEUE_MAXSIZE = 3    # Shallow depth — Qt timer always drains to the LATEST frame,
                         # so we only need enough room to absorb one scheduling hiccup.
VIS_YRANGE       = (-1.1, 1.1)   # Fixed amplitude Y-range for plots 1-3
