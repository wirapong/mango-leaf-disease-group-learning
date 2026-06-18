from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

from disease_db import DISEASE_INFO, THAI_LABELS

HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "your-username/mango-leaf-disease-efficientnetb0")
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "")
DEVICE = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"


def _resolve_asset(filename: str) -> str:
    if LOCAL_MODEL_DIR:
        candidate = Path(LOCAL_MODEL_DIR) / filename
        if candidate.exists():
            return str(candidate)
    return hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=filename)


@functools.lru_cache(maxsize=1)
def load_label_bundle() -> dict:
    labels_path = _resolve_asset("labels.json")
    with open(labels_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


class _CompatBatchNormalization(tf.keras.layers.BatchNormalization):
    """Drop-in shim that strips legacy `renorm*` kwargs removed in newer Keras."""

    _COMPAT_STRIP = frozenset(["renorm", "renorm_clipping", "renorm_momentum"])

    def __init__(self, **kwargs: object) -> None:
        for k in self._COMPAT_STRIP:
            kwargs.pop(k, None)  # type: ignore[call-overload]
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config: dict) -> "_CompatBatchNormalization":
        for k in cls._COMPAT_STRIP:
            config.pop(k, None)
        return cls(**config)



@functools.lru_cache(maxsize=1)
def load_model() -> tf.keras.Model:
    import traceback
    try:
        print(f"[model] HF_MODEL_REPO_ID={HF_MODEL_REPO_ID!r} LOCAL_MODEL_DIR={LOCAL_MODEL_DIR!r}")
        model_path = _resolve_asset("best_model.keras")
        print(f"[model] resolved path: {model_path}")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"BatchNormalization": _CompatBatchNormalization},
        )
        print("[model] load OK")
        return model
    except Exception as exc:
        traceback.print_exc()
        raise RuntimeError(
            "ไม่สามารถโหลดโมเดล EfficientNetB0 ได้ กรุณาตรวจสอบค่า HF_MODEL_REPO_ID หรือ LOCAL_MODEL_DIR และไฟล์โมเดล"
        ) from exc


def preprocess_image(image: Image.Image, image_size: int) -> np.ndarray:
    arr = np.array(image.convert("RGB").resize((image_size, image_size)), dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict_disease(image: Image.Image) -> Tuple[str, float, List[List[object]], Dict[str, str]]:
    labels = load_label_bundle()
    class_names = labels["class_names"]
    image_size = int(labels.get("image_size", 224))

    model = load_model()
    batch = preprocess_image(image, image_size)
    probs = model.predict(batch, verbose=0)[0].astype(float)
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    pred_label_th = THAI_LABELS.get(pred_label, pred_label)

    probability_table = []
    for label, prob in sorted(zip(class_names, probs.tolist()), key=lambda x: x[1], reverse=True):
        probability_table.append([THAI_LABELS.get(label, label), round(prob * 100, 2)])

    disease_info = DISEASE_INFO.get(pred_label, {
        "thai_name": pred_label_th,
        "cause": "ไม่มีข้อมูล",
        "symptoms": "ไม่มีข้อมูล",
        "advice": "ไม่มีข้อมูล",
    })
    return pred_label_th, float(probs[pred_idx] * 100), probability_table, disease_info


def warmup_model() -> str:
    try:
        load_label_bundle()
        model = load_model()
        labels = load_label_bundle()
        dummy = np.zeros((1, int(labels.get("image_size", 224)), int(labels.get("image_size", 224)), 3), dtype=np.float32)
        model.predict(dummy, verbose=0)
        return f"โมเดลพร้อมใช้งาน ({DEVICE})"
    except Exception as exc:
        return f"การ warm-up โมเดลไม่สำเร็จ: {exc}"
