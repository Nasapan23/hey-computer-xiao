from __future__ import annotations

AUTHORIZED_WAKE_LABEL = "authorized_user_wake"
UNKNOWN_USER_WAKE_LABEL = "unknown_user_wake"
NOT_WAKE_LABEL = "not_wake"

LABELS = [AUTHORIZED_WAKE_LABEL, UNKNOWN_USER_WAKE_LABEL, NOT_WAKE_LABEL]
LABEL_SOURCE_DIRS = {
    # Current 3-class setup:
    # - authorized_user_wake
    # - unknown_user_wake
    # - unknown_speech (mapped to not_wake)
    AUTHORIZED_WAKE_LABEL: ["authorized_user_wake"],
    UNKNOWN_USER_WAKE_LABEL: ["unknown_user_wake"],
    NOT_WAKE_LABEL: ["unknown_speech"],
}
# Legacy source folders kept on disk for compatibility but ignored by training.
LEGACY_IGNORED_SOURCE_DIRS = ("wake_word", "background_noise")

SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
STRIDE_SECONDS = 0.25
STRIDE_SAMPLES = int(SAMPLE_RATE * STRIDE_SECONDS)

FEATURE_FRAME_COUNT = 32
FEATURE_GROUP_COUNT = 3
FEATURE_COUNT = FEATURE_FRAME_COUNT * FEATURE_GROUP_COUNT
FEATURE_LAYOUT = ["energy", "derivative_energy", "zero_crossing_rate"]
ENERGY_FEATURE_GAIN = 200.0
DIFF_FEATURE_GAIN = 400.0
ZCR_FEATURE_GAIN = 4.0

TARGET_RMS = 0.04
MIN_GAIN = 0.5
MAX_GAIN = 20.0
REMOVE_DC_OFFSET = True

# Frontend DSP settings. These must stay aligned with esp32_tinyml_wake_word.ino.
PREEMPHASIS_ENABLED = True
PREEMPHASIS_ALPHA = 0.97
BANDPASS_ENABLED = True
BANDPASS_HIGHPASS_CUTOFF_HZ = 80.0
BANDPASS_LOWPASS_CUTOFF_HZ = 3600.0

# Model/training defaults tuned for stronger speaker separation.
MODEL_HIDDEN_UNITS = (128, 96, 48)
MODEL_DROPOUT_RATES = (0.15, 0.10)
MODEL_GAUSSIAN_NOISE_STDDEV = 0.02
MODEL_L2_REGULARIZATION = 1e-4
MODEL_LEARNING_RATE = 0.003

# Weighted loss tuned to improve authorized recall while keeping unknown-user rejection strong.
AUTHORIZED_LOSS_WEIGHT = 1.8
UNKNOWN_USER_LOSS_WEIGHT = 1.8
NOT_WAKE_LOSS_WEIGHT = 1.0

EARLY_STOPPING_PATIENCE = 12
LR_PLATEAU_PATIENCE = 4
LR_PLATEAU_FACTOR = 0.5
MIN_LEARNING_RATE = 5e-5
