from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

SEED = 42
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES: list[str] = []


@dataclass
class ModelSpec:
    name: str
    builder: Callable[..., tf.keras.Model]
    preprocess: Callable[[tf.Tensor], tf.Tensor]
    image_size: int = 224


def resolve_device() -> str:
    if tf.config.list_physical_devices("GPU"):
        return "gpu"
    return "cpu"


MODEL_SPECS: dict[str, ModelSpec] = {
    "efficientnetb0": ModelSpec(
        name="efficientnetb0",
        builder=tf.keras.applications.EfficientNetB0,
        preprocess=tf.keras.applications.efficientnet.preprocess_input,
    ),
    "mobilenetv3large": ModelSpec(
        name="mobilenetv3large",
        builder=tf.keras.applications.MobileNetV3Large,
        preprocess=tf.keras.applications.mobilenet_v3.preprocess_input,
    ),
    "convnexttiny": ModelSpec(
        name="convnexttiny",
        builder=tf.keras.applications.ConvNeXtTiny,
        preprocess=tf.keras.applications.convnext.preprocess_input,
    ),
}


def build_dataframe(data_dir: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for img_path in sorted(class_dir.glob("*.jpg")):
            rows.append({"filepath": str(img_path), "label": class_dir.name})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No JPG files found in {data_dir}")
    return df


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def decode_and_resize(path: tf.Tensor, label: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(
    df: pd.DataFrame,
    class_names: list[str],
    image_size: int,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    label_to_id = {label: idx for idx, label in enumerate(class_names)}
    paths = df["filepath"].tolist()
    labels = [label_to_id[label] for label in df["label"].tolist()]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: decode_and_resize(p, y, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_augmenter() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
            tf.keras.layers.RandomRotation(0.08, seed=SEED),
            tf.keras.layers.RandomZoom(0.12, 0.12, seed=SEED),
            tf.keras.layers.RandomContrast(0.15, seed=SEED),
            tf.keras.layers.RandomTranslation(0.05, 0.05, seed=SEED),
        ],
        name="augmentation",
    )


def build_model(spec: ModelSpec, class_names: list[str], dropout: float = 0.35) -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input(shape=(spec.image_size, spec.image_size, 3), name="image")
    augmenter = make_augmenter()
    x = augmenter(inputs)
    x = spec.preprocess(x)
    base_model = spec.builder(
        include_top=False,
        weights="imagenet",
        input_shape=(spec.image_size, spec.image_size, 3),
        pooling="avg",
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs, name=f"mango_{spec.name}")
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
    )


def fit_model(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    run_dir: Path,
    frozen_epochs: int,
    finetune_epochs: int,
    finetune_at: int,
) -> tf.keras.callbacks.History:
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=False),
    ]

    compile_model(model, learning_rate=1e-3)
    history_1 = model.fit(train_ds, validation_data=val_ds, epochs=frozen_epochs, callbacks=callbacks, verbose=1)

    base_model.trainable = True
    if finetune_at > 0:
        for layer in base_model.layers[:-finetune_at]:
            layer.trainable = False

    compile_model(model, learning_rate=1e-4)
    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=history_1.epoch[-1] + 1 if history_1.epoch else 0,
        epochs=(history_1.epoch[-1] + 1 if history_1.epoch else 0) + finetune_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return history_2


def evaluate_model(model: tf.keras.Model, ds: tf.data.Dataset, y_true: np.ndarray, class_names: list[str]) -> dict:
    probs = model.predict(ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "probabilities": probs.tolist(),
        "predictions": y_pred.tolist(),
    }


def save_split_metadata(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)


def load_precomputed_splits(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(args.train_split)
    val_df = pd.read_csv(args.val_split)
    test_df = pd.read_csv(args.test_split)
    return train_df, val_df, test_df


def run_experiment(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(data_dir)
    class_names = sorted(df["label"].unique().tolist())
    global CLASS_NAMES
    CLASS_NAMES = class_names

    if args.train_split and args.val_split and args.test_split:
        train_df, val_df, test_df = load_precomputed_splits(args)
    else:
        train_df, val_df, test_df = stratified_split(df)
        save_split_metadata(train_df, val_df, test_df, output_dir)

    y_test = np.array([class_names.index(label) for label in test_df["label"].tolist()])

    spec_names = args.models.split(",")
    summary: list[dict] = []

    for spec_name in spec_names:
        spec_name = spec_name.strip().lower()
        if spec_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {spec_name}. Choose from {list(MODEL_SPECS)}")
        spec = MODEL_SPECS[spec_name]
        run_dir = output_dir / spec_name
        run_dir.mkdir(parents=True, exist_ok=True)

        train_ds = make_dataset(train_df, class_names, spec.image_size, args.batch_size, training=True)
        val_ds = make_dataset(val_df, class_names, spec.image_size, args.batch_size, training=False)
        test_ds = make_dataset(test_df, class_names, spec.image_size, args.batch_size, training=False)

        model, base_model = build_model(spec, class_names, dropout=args.dropout)
        fit_model(
            model,
            base_model,
            train_ds,
            val_ds,
            run_dir,
            frozen_epochs=args.frozen_epochs,
            finetune_epochs=args.finetune_epochs,
            finetune_at=args.finetune_at,
        )

        best_model = tf.keras.models.load_model(run_dir / "best_model.keras")
        metrics = evaluate_model(best_model, test_ds, y_test, class_names)
        metrics.update(
            {
                "model": spec_name,
                "class_names": class_names,
                "device": resolve_device(),
                "num_train": int(len(train_df)),
                "num_val": int(len(val_df)),
                "num_test": int(len(test_df)),
                "image_size": spec.image_size,
                "batch_size": args.batch_size,
                "frozen_epochs": args.frozen_epochs,
                "finetune_epochs": args.finetune_epochs,
                "finetune_at": args.finetune_at,
            }
        )

        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        pd.DataFrame(metrics["classification_report"]).transpose().to_csv(run_dir / "classification_report.csv")
        pd.DataFrame(metrics["confusion_matrix"], index=class_names, columns=class_names).to_csv(run_dir / "confusion_matrix.csv")

        summary.append(
            {
                "model": spec_name,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "f1_weighted": metrics["f1_weighted"],
            }
        )

    summary_df = pd.DataFrame(summary).sort_values(by=["f1_macro", "accuracy"], ascending=False)
    summary_df.to_csv(output_dir / "experiment_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate mango leaf disease classifiers.")
    parser.add_argument("--data-dir", required=True, help="Path to the dataset directory with one subfolder per class.")
    parser.add_argument("--output-dir", required=True, help="Directory to store trained models and metrics.")
    parser.add_argument("--models", default="efficientnetb0,convnexttiny,mobilenetv3large", help="Comma-separated model names.")
    parser.add_argument("--train-split", help="Optional precomputed train split CSV.")
    parser.add_argument("--val-split", help="Optional precomputed validation split CSV.")
    parser.add_argument("--test-split", help="Optional precomputed test split CSV.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--frozen-epochs", type=int, default=6)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--finetune-at", type=int, default=30)
    args = parser.parse_args()
    run_experiment(args)
