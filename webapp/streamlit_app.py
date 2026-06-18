from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from PIL import Image

from model import HF_MODEL_REPO_ID, predict_disease, warmup_model
from validator import validate_mango_leaf

st.set_page_config(page_title="Mango Leaf Disease Diagnosis", layout="wide")

st.title("Mango Leaf Disease Diagnosis System")
st.subheader("ระบบวินิจฉัยโรคใบมะม่วงด้วยปัญญาประดิษฐ์")
st.caption(f"Model source: {HF_MODEL_REPO_ID}")
st.info(warmup_model())

uploaded = st.file_uploader("อัปโหลดภาพใบมะม่วง", type=["jpg", "jpeg", "png"])
show_confidence = st.checkbox("แสดงค่าความมั่นใจของแบบจำลอง", value=True)

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)
    with col2:
        valid, message = validate_mango_leaf(image)
        if not valid:
            st.error(f"ไม่สามารถวิเคราะห์ได้: {message}")
        else:
            predicted_name, confidence_score, probability_table, disease_info = predict_disease(image)
            st.success(f"ผลการวินิจฉัย: {predicted_name}")
            if show_confidence:
                st.metric("ค่าความมั่นใจ", f"{confidence_score:.2f}%")
            st.dataframe(pd.DataFrame(probability_table, columns=["กลุ่มโรค", "ค่าความน่าจะเป็น (%)"]))
            st.markdown(
                f"## {disease_info['thai_name']}\n\n"
                f"**สาเหตุของโรค**  \n{disease_info['cause']}\n\n"
                f"**อาการที่พบ**  \n{disease_info['symptoms']}\n\n"
                f"**คำแนะนำเบื้องต้น**  \n{disease_info['advice']}"
            )
else:
    st.write("พร้อมใช้งาน")
