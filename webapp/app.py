from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Compatibility shim: huggingface_hub >= 0.20 removed HfFolder.
# Gradio 4.x still imports it from gradio/oauth.py. Patch before importing gradio.
# ---------------------------------------------------------------------------
try:
    from huggingface_hub import HfFolder as _check_hffolder  # noqa: F401
except ImportError:
    import huggingface_hub as _hfhub

    class _HfFolder:  # minimal stub
        @staticmethod
        def get_token():
            try:
                return _hfhub.get_token()
            except Exception:
                return None

        @staticmethod
        def save_token(token: str) -> None:
            try:
                _hfhub.login(token=token)
            except Exception:
                pass

        @staticmethod
        def delete_token() -> None:
            pass

    _hfhub.HfFolder = _HfFolder  # type: ignore[attr-defined]
    sys.modules["huggingface_hub"].HfFolder = _HfFolder  # type: ignore[assignment]
# ---------------------------------------------------------------------------

import gradio as gr

# ---------------------------------------------------------------------------
# Compatibility shim A: gradio_client bug — Gradio 4.44.x installed with a
# newer gradio_client produces APIInfoParseError when `additionalProperties`
# is a boolean (True/False) in the JSON schema of gr.Dataframe.
# We wrap the PUBLIC json_schema_to_python_type (called by blocks.py via
# `client_utils.json_schema_to_python_type`) with a try/except so any
# schema-parse failure silently returns "Any" instead of raising.
# ---------------------------------------------------------------------------
try:
    import gradio_client.utils as _gcu
    _orig_pub_schema = _gcu.json_schema_to_python_type

    def _safe_pub_schema(schema: object) -> str:  # type: ignore[misc]
        try:
            return _orig_pub_schema(schema)  # type: ignore[arg-type]
        except Exception:
            return "Any"

    _gcu.json_schema_to_python_type = _safe_pub_schema  # type: ignore[assignment]
except Exception:
    pass
# ---------------------------------------------------------------------------
# app_version: 2026-06-17-v8
from PIL import Image

from disease_db import CLASS_ORDER, DISEASE_INFO, THAI_LABELS
from model import HF_MODEL_REPO_ID, predict_disease, warmup_model
from validator import validate_mango_leaf

APP_TITLE_EN = "Mango Leaf Disease Diagnosis System"
APP_TITLE_TH = "ระบบวินิจฉัยโรคใบมะม่วงด้วยปัญญาประดิษฐ์"
MODEL_STATUS = warmup_model()


def format_disease_card(disease_info: dict) -> str:
    return (
        f"## {disease_info['thai_name']}\n\n"
        f"**สาเหตุของโรค**  \n{disease_info['cause']}\n\n"
        f"**อาการที่พบ**  \n{disease_info['symptoms']}\n\n"
        f"**คำแนะนำเบื้องต้น**  \n{disease_info['advice']}"
    )


def analyze_image(image: Image.Image | None, show_confidence: bool):
    if image is None:
        return (
            "ไม่สามารถวิเคราะห์ได้",
            "-",
            [],
            "## กรุณาอัปโหลดภาพใบมะม่วงก่อนเริ่มการวิเคราะห์",
            "ยังไม่พร้อมวิเคราะห์",
        )

    is_valid, reason = validate_mango_leaf(image)
    if not is_valid:
        return (
            "ไม่สามารถวิเคราะห์ได้",
            "-",
            [],
            f"## ไม่สามารถวิเคราะห์ได้\n\n{reason}",
            "ตรวจพบว่าภาพอยู่นอกขอบเขตของแบบจำลอง",
        )

    try:
        predicted_name, confidence_score, probability_table, disease_info = predict_disease(image)
        conf_text = f"{confidence_score:.1f}%" if show_confidence else "-"
        return (
            predicted_name,
            conf_text,
            probability_table,
            format_disease_card(disease_info),
            f"วิเคราะห์สำเร็จด้วยโมเดลจาก {HF_MODEL_REPO_ID}",
        )
    except Exception as exc:
        return (
            "ไม่สามารถวิเคราะห์ได้",
            "-",
            [],
            f"## เกิดข้อผิดพลาดระหว่างการวิเคราะห์\n\n{exc}",
            "ระบบไม่พร้อมใช้งานชั่วคราว",
        )


def build_intro_markdown() -> str:
    disease_list = "\n".join([f"- {THAI_LABELS[name]} ({name})" for name in CLASS_ORDER])
    return f"""
# {APP_TITLE_EN}
# {APP_TITLE_TH}

ระบบนี้พัฒนาขึ้นเพื่อช่วยวิเคราะห์ภาพใบมะม่วงและจำแนกโรคทางใบจากแบบจำลองที่ดีที่สุดของงานวิจัย (ปัจจุบันคือ EfficientNetB0) โดยตัวเว็บแอปและโมเดลถูกแยกออกจากกันเพื่อให้อัปเดตโมเดลได้โดยไม่ต้อง redeploy ทั้งระบบ และสามารถโหลดได้ทั้งจาก Hugging Face Hub หรือจากโฟลเดอร์โมเดลภายในเครื่อง

**วิธีใช้งาน 3 ขั้นตอน**
1. อัปโหลดภาพใบมะม่วง
2. กดปุ่มวิเคราะห์โรค
3. ดูผลการวินิจฉัยพร้อมคำแนะนำเบื้องต้น

**กลุ่มที่ระบบรองรับ (8 classes: 7 โรค + ใบปกติ)**
{disease_list}

> หมายเหตุ: ระบบนี้ฝึกจากภาพใบมะม่วงเท่านั้น หากอัปโหลดภาพวัตถุ คน สัตว์ หรือพืชชนิดอื่น ระบบอาจปฏิเสธการวิเคราะห์เพื่อรักษาความน่าเชื่อถือของผลลัพธ์

**สถานะโมเดล:** {MODEL_STATUS}
"""


with gr.Blocks(theme=gr.themes.Soft(), title=APP_TITLE_EN) as demo:
    gr.Markdown(build_intro_markdown())

    with gr.Tabs():
        with gr.TabItem("แนะนำระบบ"):
            gr.Markdown(build_intro_markdown())

        with gr.TabItem("วิเคราะห์โรค"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="อัปโหลดภาพใบมะม่วง", height=420)
                    status_box = gr.Textbox(value="พร้อมใช้งาน", label="สถานะ", interactive=False)
                    show_confidence = gr.Checkbox(label="แสดงค่าความมั่นใจของแบบจำลอง", value=True)
                    analyze_button = gr.Button("วิเคราะห์โรค", variant="primary")
                with gr.Column(scale=1):
                    predicted_name = gr.Textbox(label="ผลการวินิจฉัย", interactive=False)
                    confidence_output = gr.Textbox(label="ค่าความมั่นใจ (%)", interactive=False)
                    probability_table = gr.Dataframe(
                        headers=["กลุ่มโรค", "ค่าความน่าจะเป็น (%)"],
                        datatype=["str", "number"],
                        label="ตารางความน่าจะเป็นของทุกกลุ่ม",
                    )
                    disease_card = gr.Markdown("## รอผลการวินิจฉัย")

                clear_button = gr.ClearButton(
                    value="ล้างข้อมูล",
                    components=[image_input, status_box, predicted_name, confidence_output, probability_table, disease_card],
                )

            analyze_button.click(
                fn=analyze_image,
                inputs=[image_input, show_confidence],
                outputs=[predicted_name, confidence_output, probability_table, disease_card, status_box],
                api_name="analyze",
            )

# ---------------------------------------------------------------------------
# Launch: Gradio 4.44.x on HF Spaces fails the post-start localhost check.
# We catch that specific ValueError, keep the process alive, and rely on
# the uvicorn thread that was already started to handle requests.
# ---------------------------------------------------------------------------
try:
    demo.launch(server_name="0.0.0.0", server_port=7860)
except ValueError as _launch_err:
    if "localhost" in str(_launch_err).lower() or "shareable" in str(_launch_err).lower():
        import time as _time
        while True:  # keep process alive; uvicorn thread is still serving
            _time.sleep(60)
    raise
