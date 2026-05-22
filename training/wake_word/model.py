from __future__ import annotations

import tensorflow as tf

from .config import (
    FEATURE_COUNT,
    LABELS,
    MODEL_DROPOUT_RATES,
    MODEL_GAUSSIAN_NOISE_STDDEV,
    MODEL_HIDDEN_UNITS,
    MODEL_L2_REGULARIZATION,
    MODEL_LEARNING_RATE,
)


def build_model() -> tf.keras.Model:
    """Create a stronger regularized dense classifier for tiny speaker-aware KWS."""
    hidden_1, hidden_2, hidden_3 = MODEL_HIDDEN_UNITS
    dropout_1, dropout_2 = MODEL_DROPOUT_RATES
    regularizer = tf.keras.regularizers.l2(MODEL_L2_REGULARIZATION)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(FEATURE_COUNT,)),
            tf.keras.layers.GaussianNoise(MODEL_GAUSSIAN_NOISE_STDDEV),
            tf.keras.layers.Dense(hidden_1, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(dropout_1),
            tf.keras.layers.Dense(hidden_2, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(dropout_2),
            tf.keras.layers.Dense(hidden_3, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dense(len(LABELS), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=MODEL_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
