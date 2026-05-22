from __future__ import annotations

import shutil

import numpy as np
import tensorflow as tf

from .augmentation import oversample_and_augment
from .config import (
    AUTHORIZED_LOSS_WEIGHT,
    AUTHORIZED_WAKE_LABEL,
    EARLY_STOPPING_PATIENCE,
    LABELS,
    LR_PLATEAU_FACTOR,
    LR_PLATEAU_PATIENCE,
    MODEL_DROPOUT_RATES,
    MODEL_GAUSSIAN_NOISE_STDDEV,
    MODEL_HIDDEN_UNITS,
    MODEL_L2_REGULARIZATION,
    MODEL_LEARNING_RATE,
    MIN_LEARNING_RATE,
    NOT_WAKE_LABEL,
    NOT_WAKE_LOSS_WEIGHT,
    UNKNOWN_USER_LOSS_WEIGHT,
    UNKNOWN_USER_WAKE_LABEL,
)
from .dataset import make_dataset, split_dataset
from .export import export_tflite, write_model_header
from .features import extract_feature_batch
from .metrics import compute_per_class_metrics, make_confusion_matrix, print_confusion_matrix
from .model import build_model
from .reports import save_training_report


def count_labels(y: np.ndarray) -> list[int]:
    return [int(np.sum(y == label_index)) for label_index in range(len(LABELS))]


def train_and_export(args) -> None:
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    processed_dir = args.dataset / "processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WAV files and selecting 2 second windows from your longer recordings...")
    x, y, source_ids, manifest = make_dataset(args.dataset, args.max_windows_per_file, args.normalize_rms)

    x_train, y_train, x_test, y_test = split_dataset(x, y, source_ids, args.seed)
    evaluation_uses_training_data = False
    if len(y_test) == 0:
        print("Warning: no held-out test files available; using training data for validation/evaluation.")
        x_test = x_train
        y_test = y_train
        evaluation_uses_training_data = True

    y_train_before_augmentation = y_train.copy()
    x_train, y_train = oversample_and_augment(x_train, y_train, args.seed)

    x_train_model = extract_feature_batch(x_train)
    x_test_model = extract_feature_batch(x_test)

    print(f"Examples: train={len(x_train_model)} test={len(x_test_model)}")
    for label_index, label in enumerate(LABELS):
        print(f"  {label}: {int(np.sum(y == label_index))} windows")

    model = build_model()
    model.summary()

    class_weight = {
        LABELS.index(AUTHORIZED_WAKE_LABEL): AUTHORIZED_LOSS_WEIGHT,
        LABELS.index(UNKNOWN_USER_WAKE_LABEL): UNKNOWN_USER_LOSS_WEIGHT,
        LABELS.index(NOT_WAKE_LABEL): NOT_WAKE_LOSS_WEIGHT,
    }
    print("Class-weighted loss:")
    for label_index, label in enumerate(LABELS):
        print(f"  {label}: {class_weight[label_index]:.2f}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_PLATEAU_FACTOR,
            patience=LR_PLATEAU_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train_model,
        y_train,
        validation_data=(x_test_model, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    evaluation = model.evaluate(x_test_model, y_test, verbose=0)
    print(f"Test accuracy: {evaluation[1]:.3f}")
    y_pred = np.argmax(model.predict(x_test_model, verbose=0), axis=1)
    confusion_matrix = make_confusion_matrix(y_test, y_pred, len(LABELS))
    per_class_metrics = compute_per_class_metrics(confusion_matrix)
    print_confusion_matrix(confusion_matrix)

    keras_path = args.output / "wake_word_model.keras"
    tflite_path = args.output / "wake_word_model_int8.tflite"

    model.save(keras_path)
    tflite_model = export_tflite(model, x_train_model, tflite_path)
    write_model_header(tflite_model, args.header_output)
    save_training_report(
        args.output,
        manifest,
        history,
        evaluation,
        confusion_matrix,
        per_class_metrics,
        model,
        len(x_train_model),
        len(x_test_model),
        evaluation_uses_training_data,
        args.normalize_rms,
        {
            "all_windows": count_labels(y),
            "train_before_augmentation": count_labels(y_train_before_augmentation),
            "test_split": count_labels(y_test),
            "train_after_augmentation": count_labels(y_train),
        },
        {
            "model_hidden_units": list(MODEL_HIDDEN_UNITS),
            "model_dropout_rates": list(MODEL_DROPOUT_RATES),
            "model_gaussian_noise_stddev": MODEL_GAUSSIAN_NOISE_STDDEV,
            "model_l2_regularization": MODEL_L2_REGULARIZATION,
            "model_learning_rate": MODEL_LEARNING_RATE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "lr_plateau_patience": LR_PLATEAU_PATIENCE,
            "lr_plateau_factor": LR_PLATEAU_FACTOR,
            "min_learning_rate": MIN_LEARNING_RATE,
            "class_weight": {str(index): float(weight) for index, weight in class_weight.items()},
        },
    )

    print()
    print(f"Saved Keras model: {keras_path}")
    print(f"Saved int8 TFLite model: {tflite_path}")
    print(f"Generated Arduino model header: {args.header_output}")
    print(f"Saved confusion matrix CSV: {args.output / 'confusion_matrix.csv'}")
    print(f"Saved confusion matrix image: {args.output / 'confusion_matrix.png'}")
    print(f"Saved training curves image: {args.output / 'training_curves.png'}")
    print(f"Saved class distribution image: {args.output / 'class_distribution.png'}")
    print(f"Saved per-class metrics image: {args.output / 'per_class_metrics.png'}")
    print(f"Saved model summary: {args.output / 'model_summary.txt'}")
    print("Next: open esp32_tinyml_wake_word/esp32_tinyml_wake_word.ino and flash it.")
