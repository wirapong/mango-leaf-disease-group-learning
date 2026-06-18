from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export best Keras mango disease model with metadata for deployment.")
    parser.add_argument("--model-dir", required=True, help="Directory containing best_model.keras and metrics.json")
    parser.add_argument("--output-dir", required=True, help="Directory to place deployable package")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = model_dir / "metrics.json"
    model_path = model_dir / "best_model.keras"
    if not metrics_path.exists() or not model_path.exists():
        raise FileNotFoundError("Expected metrics.json and best_model.keras in model-dir")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    class_names = metrics["class_names"]

    labels_path = output_dir / "labels.json"
    labels_payload = {
        "class_names": class_names,
        "label_to_id": {label: idx for idx, label in enumerate(class_names)},
        "id_to_label": {str(idx): label for idx, label in enumerate(class_names)},
        "image_size": metrics.get("image_size", 224),
        "preprocess": "tensorflow.keras.applications.efficientnet.preprocess_input",
        "model_family": metrics.get("model", "efficientnetb0"),
    }
    labels_path.write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

    model_card = output_dir / "MODEL_CARD.md"
    model_card.write_text(
        "# Mango Leaf Disease Classifier\n\n"
        f"- Model: {metrics.get('model', 'efficientnetb0')}\n"
        f"- Accuracy: {metrics.get('accuracy')}\n"
        f"- Weighted F1: {metrics.get('f1_weighted')}\n"
        "- Dataset: MangoLeafBD full 8-class local copy (4,000 images)\n"
        "- Deployment intent: Hugging Face Hub / Gradio / Streamlit\n",
        encoding="utf-8",
    )

    target_model_path = output_dir / "best_model.keras"
    with model_path.open("rb") as src, target_model_path.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)

    summary_path = output_dir / "metrics_summary.json"
    summary = {
        key: metrics[key]
        for key in [
            "model",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "image_size",
            "batch_size",
        ]
        if key in metrics
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Exported deployable package to: {output_dir}")


if __name__ == "__main__":
    main()
