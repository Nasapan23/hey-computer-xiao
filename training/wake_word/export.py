from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from .config import (
    AUTHORIZED_WAKE_LABEL,
    BANDPASS_ENABLED,
    BANDPASS_HIGHPASS_CUTOFF_HZ,
    BANDPASS_LOWPASS_CUTOFF_HZ,
    DIFF_FEATURE_GAIN,
    ENERGY_FEATURE_GAIN,
    FEATURE_COUNT,
    FEATURE_FRAME_COUNT,
    LABELS,
    NOT_WAKE_LABEL,
    PREEMPHASIS_ALPHA,
    PREEMPHASIS_ENABLED,
    SAMPLE_RATE,
    UNKNOWN_USER_WAKE_LABEL,
    WINDOW_SAMPLES,
    ZCR_FEATURE_GAIN,
)


def export_tflite(model: tf.keras.Model, representative_x: np.ndarray, output_path: Path) -> bytes:
    def representative_dataset():
        limit = min(len(representative_x), 100)
        for sample in representative_x[:limit]:
            yield [sample.astype(np.float32)[None, :]]

    input_signature = tf.TensorSpec(
        shape=[1, FEATURE_COUNT],
        dtype=tf.float32,
        name="voice_features",
    )
    concrete_function = tf.function(lambda voice_features: model(voice_features)).get_concrete_function(input_signature)
    frozen_function = convert_variables_to_constants_v2(concrete_function)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_function])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    return tflite_model


def c_array_bytes(data: bytes) -> str:
    values = [f"0x{byte:02x}" for byte in data]
    lines = []
    for start in range(0, len(values), 12):
        lines.append("  " + ", ".join(values[start : start + 12]))
    return ",\n".join(lines)


def write_model_header(tflite_model: bytes, header_path: Path) -> None:
    header_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ", ".join(f'"{label}"' for label in LABELS)
    data = c_array_bytes(tflite_model)

    header = f"""#pragma once
#include <cstdint>

constexpr int kWakeWordSampleRate = {SAMPLE_RATE};
constexpr int kWakeWordInputSamples = {WINDOW_SAMPLES};
constexpr int kWakeWordFeatureFrameCount = {FEATURE_FRAME_COUNT};
constexpr int kWakeWordFeatureCount = {FEATURE_COUNT};
constexpr float kWakeWordEnergyFeatureGain = {ENERGY_FEATURE_GAIN}f;
constexpr float kWakeWordDiffFeatureGain = {DIFF_FEATURE_GAIN}f;
constexpr float kWakeWordZcrFeatureGain = {ZCR_FEATURE_GAIN}f;
constexpr bool kWakeWordPreemphasisEnabled = {"true" if PREEMPHASIS_ENABLED else "false"};
constexpr float kWakeWordPreemphasisAlpha = {PREEMPHASIS_ALPHA}f;
constexpr bool kWakeWordBandpassEnabled = {"true" if BANDPASS_ENABLED else "false"};
constexpr float kWakeWordBandpassHighpassCutoffHz = {BANDPASS_HIGHPASS_CUTOFF_HZ}f;
constexpr float kWakeWordBandpassLowpassCutoffHz = {BANDPASS_LOWPASS_CUTOFF_HZ}f;
constexpr int kWakeWordLabelCount = {len(LABELS)};
constexpr const char *kWakeWordLabels[kWakeWordLabelCount] = {{{labels}}};
constexpr int kAuthorizedUserWakeIndex = {LABELS.index(AUTHORIZED_WAKE_LABEL)};
constexpr int kUnknownUserWakeIndex = {LABELS.index(UNKNOWN_USER_WAKE_LABEL)};
constexpr int kNotWakeIndex = {LABELS.index(NOT_WAKE_LABEL)};
constexpr int kWakeWordIndex = kAuthorizedUserWakeIndex;

alignas(16) const unsigned char wake_word_model_tflite[] = {{
{data}
}};

const unsigned int wake_word_model_tflite_len = {len(tflite_model)};
"""
    header_path.write_text(header, encoding="utf-8")
