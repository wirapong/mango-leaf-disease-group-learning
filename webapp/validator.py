from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def validate_mango_leaf(image: Image.Image) -> Tuple[bool, str]:
    """Return (is_valid, rejection_reason_in_Thai)."""
    rgb = image.convert("RGB")
    arr = np.asarray(rgb).astype("float32") / 255.0
    h, w = arr.shape[:2]

    if h < 120 or w < 120:
        return False, "ภาพมีความละเอียดต่ำเกินไป ควรใช้ภาพที่คมชัดและเห็นรายละเอียดใบมะม่วงชัดเจน"

    aspect_ratio = max(h / w, w / h)
    if aspect_ratio > 3.2:
        return False, "สัดส่วนภาพไม่เหมาะสม ควรใช้ภาพใบมะม่วงที่ครอบภาพพอดีและไม่ยาวหรือแคบเกินไป"

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    green_dominance = np.mean((g > r * 0.95) & (g > b * 0.95))
    non_white_ratio = np.mean(np.mean(arr, axis=2) < 0.95)
    color_std = float(arr.std())

    gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
    # Use counts (not density) then normalise → proper probabilities in [0,1]
    hist, _ = np.histogram(gray, bins=32, range=(0, 1))
    hist = hist.astype("float64") / (hist.sum() + 1e-8) + 1e-8
    entropy = float(-(hist * np.log(hist)).sum())  # range ~0..3.47 for 32 bins

    if green_dominance < 0.08:
        return False, "ภาพนี้ดูไม่เหมือนใบมะม่วงหรือมีพื้นหลังมากเกินไป กรุณาอัปโหลดภาพใบมะม่วงที่ชัดเจน"
    if non_white_ratio < 0.25:
        return False, "พื้นที่ของใบในภาพน้อยเกินไป ควรถ่ายภาพให้ใบมะม่วงอยู่เด่นกลางภาพ"
    if color_std < 0.05:
        return False, "ภาพมีรายละเอียดสีต่ำเกินไป อาจไม่เหมาะสำหรับการวิเคราะห์โรค"
    if entropy < 1.5:
        return False, "ภาพมีรายละเอียดไม่เพียงพอหรือพื้นผิวไม่ชัดเจน กรุณาใช้ภาพใหม่ที่คมชัดกว่าเดิม"

    return True, "ผ่านการตรวจสอบเบื้องต้น"
