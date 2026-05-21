from __future__ import annotations

import tensorflow as tf

from .config import FEATURE_COUNT, LABELS


def build_model() -> tf.keras.Model:
    """Create a tiny dense classifier suitable for TFLite Micro deployment."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(FEATURE_COUNT,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(len(LABELS), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

