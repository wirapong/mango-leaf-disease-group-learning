---
title: Mango Leaf Disease Diagnosis
emoji: 🥭
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: "1.36.0"
python_version: "3.10"
app_file: streamlit_app.py
pinned: false
suggested_hardware: cpu-basic
---

# Mango Leaf Disease Diagnosis Web App

เว็บแอปนี้ใช้สำหรับวินิจฉัยโรคใบมะม่วงจากภาพถ่าย โดยตัวแอปสร้างด้วย **Gradio** และเชื่อมกับโมเดล **EfficientNetB0** ที่ชนะการทดลองรอบปัจจุบันของงานวิจัย ตัวระบบออกแบบให้แยก **model registry** ออกจาก **application code** อย่างชัดเจน เพื่อให้สามารถอัปเดตโมเดลได้โดยไม่ต้องแก้ UI ใหม่ทั้งหมด

นอกจากนี้ยังมีไฟล์ `streamlit_app.py` สำหรับกรณีที่ต้องการ deploy แบบ Streamlit หรือรันบน GitHub/เครื่องท้องถิ่นได้ด้วย

## โครงสร้างไฟล์

- `app.py` — ส่วนติดต่อผู้ใช้ด้วย Gradio สำหรับ Hugging Face Spaces
- `streamlit_app.py` — เวอร์ชัน Streamlit สำหรับการใช้งานทางเลือก
- `model.py` — โมดูลโหลดโมเดลและรัน inference
- `validator.py` — ตรวจสอบว่า input น่าจะเป็นภาพใบมะม่วงก่อนวิเคราะห์
- `disease_db.py` — ฐานข้อมูลคำอธิบายโรคภาษาไทย
- `requirements.txt` — dependencies สำหรับ deploy

## แพ็กเกจโมเดลที่เตรียมไว้แล้ว

ผลการ export โมเดลพร้อม deploy อยู่ที่:

`/GGD/OpenClawMacMini/Mango Plant Disease/outputs/deployable_model_package/`

ภายในประกอบด้วยอย่างน้อย:
- `best_model.keras`
- `labels.json`
- `metrics_summary.json`
- `MODEL_CARD.md`

แพ็กเกจนี้สามารถอัปโหลดขึ้น Hugging Face Hub ได้ตรง ๆ ใน model repository หนึ่งตัว

## การตั้งค่า Environment Variables

กำหนดค่าเหล่านี้ใน Hugging Face Space หรือในเครื่องท้องถิ่น:

- `HF_MODEL_REPO_ID=your-username/mango-leaf-disease-efficientnetb0`
- `LOCAL_MODEL_DIR=` เว้นว่างได้ หากต้องการโหลดจาก HF Hub
- `GRADIO_THEME=soft`

> ถ้ากำหนด `LOCAL_MODEL_DIR` ระบบจะพยายามโหลด `best_model.keras` และ `labels.json` จากโฟลเดอร์นี้ก่อน

## ขั้นตอนการ deploy บน Hugging Face Spaces (Gradio)

### 1) สร้าง Space
1. เข้า Hugging Face และกด **New Space**
2. ตั้งชื่อโปรเจกต์ เช่น `mango-leaf-diagnosis`
3. เลือก **Gradio SDK**
4. เลือก hardware เป็น `cpu-basic` ก่อน หาก inference ช้าค่อยขยับเป็น `gpu-t4-small`

### 2) อัปโหลดโมเดลขึ้น Hugging Face Hub
1. สร้าง model repository ใหม่ เช่น `your-username/mango-leaf-disease-efficientnetb0`
2. อัปโหลดไฟล์จากโฟลเดอร์ `outputs/deployable_model_package/` ขึ้น repo นี้
3. ตรวจสอบว่า repo มีไฟล์ `best_model.keras` และ `labels.json` ครบ

### 3) อัปโหลดไฟล์เว็บแอปขึ้น Space
1. อัปโหลดไฟล์ทั้งหมดในโฟลเดอร์ `webapp/`
2. ตรวจสอบว่า `app.py` อยู่ระดับ root ของ Space
3. ตรวจสอบว่า `requirements.txt` ติดตั้ง package ครบ

### 4) ตั้งค่า Environment Variables
1. ไปที่ **Space Settings**
2. เพิ่มตัวแปร `HF_MODEL_REPO_ID`
3. เพิ่มตัวแปร `GRADIO_THEME=soft`
4. Save แล้วรอ Space rebuild อัตโนมัติ

### 5) ทดสอบระบบหลัง deploy
1. เปิดหน้า Space
2. อัปโหลดภาพใบมะม่วงที่ทราบกลุ่มโรค
3. กดปุ่ม **วิเคราะห์โรค**
4. ตรวจผลลัพธ์ชื่อโรค ค่าความมั่นใจ และตารางความน่าจะเป็น
5. ทดลองอัปโหลดภาพที่ไม่ใช่ใบมะม่วงเพื่อเช็ก validation

## ขั้นตอนการรันแบบ Streamlit / Local

### 1) ติดตั้ง dependencies
```bash
pip install -r requirements.txt
```

### 2) ตั้งค่าโมเดลภายในเครื่อง
```bash
export LOCAL_MODEL_DIR="/Users/wirapongc/Library/CloudStorage/GoogleDrive-wirach@kku.ac.th/My Drive/OpenClawMacMini/Mango Plant Disease/outputs/deployable_model_package"
```

### 3) รัน Streamlit
```bash
streamlit run streamlit_app.py
```

## หมายเหตุเชิงวิชาการ

เวอร์ชันนี้ normalize class ให้เป็น **8 classes** ตาม dataset ที่ใช้งานจริง ได้แก่ 7 โรคและ 1 class ใบปกติ (`Healthy`) แม้ prompt ตั้งต้นจะมีข้อความปะปนเรื่อง 6/7 classes ก็ตาม

ผลวิจัยปัจจุบันแยกเป็น 2 ระดับการใช้งาน:
- **Best deployable single model:** EfficientNetB0
- **Best research model:** soft-voting heterogeneous ensemble ของ EfficientNetB0 + MobileNetV3Large + ConvNeXtTiny

## ข้อเสนอแนะเพิ่มเติม

- หากต้องการความเร็วสูงขึ้น ให้แยกทำ lightweight validator model บน Hub อีกหนึ่งตัว
- หากต้องการใช้ในภาคสนาม ควรเพิ่มตัวอย่างภาพที่อยู่นอกโดเมนเพื่อปรับ threshold ของ validation
- ควรบันทึก version ของโมเดลและผลประเมินลงใน model card บน Hugging Face Hub ทุกครั้ง
