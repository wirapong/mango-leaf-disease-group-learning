from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from build_grouped_split import build_dataframe, annotate_records, assign_groups
from train_mango_models import MODEL_SPECS, build_model, evaluate_model, fit_model, make_dataset, resolve_device

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold grouped CV for EfficientNetB0 on MangoLeafBD.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frozen-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--finetune-at", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = build_dataframe(data_dir)
    annotated_df = annotate_records(raw_df)
    grouped_df, group_audit = assign_groups(annotated_df)
    grouped_df = grouped_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    class_names = sorted(grouped_df["label"].unique().tolist())
    spec = MODEL_SPECS["efficientnetb0"]

    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    X = grouped_df["filepath"].values
    y = grouped_df["label"].values
    groups = grouped_df["group_id"].values

    fold_rows: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        tf.keras.backend.clear_session()
        gc.collect()

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = grouped_df.iloc[train_idx].reset_index(drop=True)
        val_df = grouped_df.iloc[val_idx].reset_index(drop=True)
        train_df.to_csv(fold_dir / "train_split.csv", index=False)
        val_df.to_csv(fold_dir / "val_split.csv", index=False)

        train_ds = make_dataset(train_df, class_names, spec.image_size, args.batch_size, training=True)
        val_ds = make_dataset(val_df, class_names, spec.image_size, args.batch_size, training=False)
        y_val = np.array([class_names.index(label) for label in val_df["label"].tolist()])

        model, base_model = build_model(spec, class_names, dropout=0.35)
        fit_model(
            model,
            base_model,
            train_ds,
            val_ds,
            fold_dir,
            frozen_epochs=args.frozen_epochs,
            finetune_epochs=args.finetune_epochs,
            finetune_at=args.finetune_at,
        )

        best_model = tf.keras.models.load_model(fold_dir / "best_model.keras")
        metrics = evaluate_model(best_model, val_ds, y_val, class_names)
        metrics.update(
            {
                "fold": fold_idx,
                "model": spec.name,
                "class_names": class_names,
                "device": resolve_device(),
                "num_train": int(len(train_df)),
                "num_val": int(len(val_df)),
                "num_train_groups": int(train_df["group_id"].nunique()),
                "num_val_groups": int(val_df["group_id"].nunique()),
            }
        )
        (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        fold_rows.append(
            {
                "fold": fold_idx,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "f1_weighted": metrics["f1_weighted"],
                "num_train": len(train_df),
                "num_val": len(val_df),
                "num_train_groups": int(train_df["group_id"].nunique()),
                "num_val_groups": int(val_df["group_id"].nunique()),
            }
        )

    summary_df = pd.DataFrame(fold_rows)
    summary_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    metric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    summary = {
        "model": spec.name,
        "n_splits": args.n_splits,
        "device": resolve_device(),
        "group_audit": group_audit,
        "folds": fold_rows,
        "mean": {col: float(summary_df[col].mean()) for col in metric_cols},
        "std": {col: float(summary_df[col].std(ddof=1)) for col in metric_cols},
        "min": {col: float(summary_df[col].min()) for col in metric_cols},
        "max": {col: float(summary_df[col].max()) for col in metric_cols},
    }
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["mean"], indent=2))
    print(json.dumps(summary["std"], indent=2))


if __name__ == "__main__":
    main()
