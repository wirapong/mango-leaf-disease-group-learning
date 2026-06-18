from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def load_metrics(model_dir: Path) -> dict:
    with open(model_dir / "metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    y_pred = probs.argmax(axis=1)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "predictions": y_pred.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", required=True, help="Comma-separated model subdirectories")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    payloads = [load_metrics(output_dir / model_name) for model_name in model_names]

    class_names = payloads[0]["class_names"]
    probs = [np.array(payload["probabilities"], dtype=float) for payload in payloads]
    preds = [np.array(payload["predictions"], dtype=int) for payload in payloads]
    if len({tuple(p.shape) for p in probs}) != 1:
        raise ValueError("Probability arrays have mismatched shapes")

    stacked = np.mean(probs, axis=0)
    y_true = preds[0] * 0
    # reconstruct y_true from classification_report support ordering is unreliable; use confusion matrix row sums + predictions impossible
    # so we read test_split.csv and map labels using class_names
    test_split = output_dir / "test_split.csv"
    if not test_split.exists():
        raise FileNotFoundError("test_split.csv not found in output directory")

    import pandas as pd

    df = pd.read_csv(test_split)
    y_true = np.array([class_names.index(label) for label in df["label"].tolist()])
    metrics = evaluate(y_true, stacked, class_names)
    metrics["ensemble_models"] = model_names

    with open(output_dir / "ensemble_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: metrics[k] for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"]}, indent=2))


if __name__ == "__main__":
    main()
