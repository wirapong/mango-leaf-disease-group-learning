# 🥭 Mango Leaf Disease Diagnosis System

**Leakage-Aware Deep Learning for Mango Leaf Disease Classification: Transfer Learning, Explainable AI, and Deployment-Oriented Evaluation**

A reproducible deep-learning pipeline and a live web application for diagnosing eight classes of mango leaf conditions from a single leaf image. The repository accompanies the manuscript above and emphasises **leakage-aware validation**, **explainability (Grad-CAM)**, and **deployment-oriented evaluation** rather than headline accuracy alone.

🔗 **Live demo (Hugging Face Space):** <https://huggingface.co/spaces/wirapongc/mango-leaf-diagnosis>

---

## ✨ Highlights

- **8-class classification** on the full MangoLeafBD formulation (4,000 images).
- **Transfer-learning benchmark** of three modern backbones: EfficientNetB0, MobileNetV3Large, ConvNeXtTiny.
- **Leakage audit** using cleaned filename stems + perceptual hashes (`aHash`, `dHash`) to build a **group-aware split** with verified zero overlap across train/val/test.
- **Leakage-reassessed primary results** (more honest than the naive split):
  - EfficientNetB0 — accuracy **0.9950**, weighted F1 **0.9950**
  - 5-fold grouped CV — mean accuracy **0.9963** (± 0.0020)
- **Explainable AI:** Grad-CAM heatmaps for class-evidence inspection.
- **Deployment-oriented evaluation:** latency benchmarking + a working Streamlit/Gradio web app with leaf-validity pre-filter.
- **Beats the prior JISEM VGG-19 bagging benchmark** (Thalor et al., 2025: 95% accuracy) even *after* leakage correction.

---

## 📁 Repository Structure

```
.
├── Experiments/                  # Training, evaluation, XAI, and benchmarking scripts
│   ├── build_grouped_split.py        # Leakage-resistant grouped split (stems + aHash + dHash)
│   ├── train_mango_models.py         # Transfer-learning trainer (multi-backbone)
│   ├── ensemble_from_saved_probs.py  # Post-hoc soft-voting ensemble
│   ├── grouped_cv_efficientnet.py    # 5-fold grouped cross-validation
│   ├── export_best_model.py          # Export deployable model + metadata
│   ├── generate_gradcam.py           # Grad-CAM visualisations
│   └── latency_benchmark.py          # Inference latency profiling
│
├── Outputs/                      # Experiment artefacts (CSVs, logs, figures, model package)
│   ├── grouped_split_v1/             # Verified leakage-resistant split
│   ├── grouped_model_search_v1/      # Leakage-reassessed model comparison
│   ├── grouped_cv_efficientnet_v1/   # 5-fold grouped CV results
│   ├── gradcam_v1/                   # Grad-CAM heatmaps
│   ├── latency_v1/                   # Latency benchmark CSVs
│   ├── model_search_v1/              # Original (naive-split) comparison — for reference only
│   └── deployable_model_package/     # Exported EfficientNetB0 + labels + metrics + MODEL_CARD
│
└── Webapp/                       # Web application (Streamlit + Gradio backends)
    ├── app.py                        # Gradio entry point (used by HF Space)
    ├── streamlit_app.py              # Streamlit entry point
    ├── model.py                      # Model loader (Hugging Face Hub or local)
    ├── validator.py                  # Leaf-validity pre-filter
    ├── disease_db.py                 # Disease info / management knowledge base
    ├── requirements.txt              # App dependencies
    └── README.md                     # App-specific notes
```

---

## 🧪 Dataset

- **Source formulation:** MangoLeafBD (8 classes × 500 images = 4,000 JPGs).
- **Classes:** Anthracnose · Bacterial Canker · Cutting Weevil · Die Back · Gall Midge · Healthy · Powdery Mildew · Sooty Mould.
- The dataset itself is **not redistributed** in this repository. Please obtain the original MangoLeafBD release from its authors and place it under `dataset/<class_name>/*.jpg` before running the experiments.

---

## 🔬 Methodology in Brief

1. **Leakage audit.** Compute cleaned filename stems and 64-bit perceptual hashes (`aHash`, `dHash`) for every image. Group duplicates and near-duplicates so that identical or nearly identical leaves never cross train/val/test.
2. **Group-aware split.** 2,800 train / 601 val / 599 test, with **zero overlap** in `group_id`, `clean_stem`, `aHash`, `dHash`.
3. **Transfer learning.** Three ImageNet-pretrained backbones (EfficientNetB0, MobileNetV3Large, ConvNeXtTiny) fine-tuned with a unified head and augmentation pipeline.
4. **Leakage-reassessed evaluation.** Report metrics on the grouped hold-out test set, not the optimistic naive split.
5. **5-fold grouped cross-validation** for the strongest backbone (EfficientNetB0) to quantify stability.
6. **Explainability.** Grad-CAM on representative test images per class.
7. **Deployment evaluation.** Latency profiling and an end-to-end web app with a leaf-validity pre-filter (rejects non-leaf inputs before classification).

---

## 📊 Key Results (leakage-reassessed)

| Model                          | Accuracy   | Weighted F1 |
|--------------------------------|:----------:|:-----------:|
| **EfficientNetB0** (primary)   | **0.9950** | **0.9950**  |
| MobileNetV3Large               | 0.9833     | 0.9833      |
| Soft-voting (EffB0 + MNV3L)    | 0.9950     | 0.9950      |

**5-fold grouped CV (EfficientNetB0):** mean accuracy **0.9963**, mean weighted F1 **0.9962**, std ≈ 0.0020.

> **Note.** The earlier naive-split run produced near-perfect numbers (incl. a 1.000 ensemble). Those results were retained under `Outputs/model_search_v1/` purely for transparency — the leakage-reassessed numbers above are the values reported in the manuscript.

---

## 🚀 Quick Start

### 1. Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r Webapp/requirements.txt
# For training/experiments you will also need: tensorflow, scikit-learn, pandas, pillow, imagehash, matplotlib, opencv-python
```

### 2. Reproduce the leakage-resistant split

```bash
python Experiments/build_grouped_split.py \
  --data_dir dataset/ \
  --out_dir Outputs/grouped_split_v1/
```

### 3. Train a backbone on the grouped split

```bash
python Experiments/train_mango_models.py \
  --split_dir Outputs/grouped_split_v1/ \
  --backbone efficientnet_b0 \
  --out_dir Outputs/grouped_model_search_v1/efficientnet_b0/
```

### 4. 5-fold grouped CV (EfficientNetB0)

```bash
python Experiments/grouped_cv_efficientnet.py \
  --data_dir dataset/ \
  --out_dir Outputs/grouped_cv_efficientnet_v1/
```

### 5. Grad-CAM visualisations

```bash
python Experiments/generate_gradcam.py \
  --model_path Outputs/deployable_model_package/best_model.keras \
  --out_dir Outputs/gradcam_v1/
```

### 6. Latency benchmark

```bash
python Experiments/latency_benchmark.py \
  --model_path Outputs/deployable_model_package/best_model.keras \
  --out_dir Outputs/latency_v1/
```

---

## 🌐 Web Application

The deployable EfficientNetB0 model is wrapped in a lightweight web app with two interchangeable frontends.

**Live demo:** <https://huggingface.co/spaces/wirapongc/mango-leaf-diagnosis>

### Run locally (Gradio — same as HF Space)

```bash
cd Webapp
export LOCAL_MODEL_DIR=../Outputs/deployable_model_package
python app.py
```

### Run locally (Streamlit)

```bash
cd Webapp
export LOCAL_MODEL_DIR=../Outputs/deployable_model_package
streamlit run streamlit_app.py
```

### Model source

The app loads the model from either:

- a Hugging Face model repo (`HF_MODEL_REPO_ID=<user>/<repo>`), or
- a local directory (`LOCAL_MODEL_DIR=path/to/deployable_model_package`).

A **leaf-validity pre-filter** (`validator.py`) rejects clearly non-leaf inputs before classification to reduce confident-but-wrong predictions in real-world usage.

---

## 📦 Deployable Model Package

Located at `Outputs/deployable_model_package/`:

- `best_model.keras` — exported EfficientNetB0 checkpoint
- `labels.json` — class index ↔ class name mapping
- `metrics_summary.json` — final leakage-reassessed metrics
- `MODEL_CARD.md` — intended use, dataset, training setup, limitations

---

## 📜 Citation

If you use this code, the deployable model, or the web application in your work, please cite:

> Chansanam, W. (2026). *Leakage-Aware Deep Learning for Mango Leaf Disease Classification: Transfer Learning, Explainable AI, and Deployment-Oriented Evaluation.* Manuscript.

BibTeX:

```bibtex
@article{chansanam2026mangoleaf,
  title   = {Leakage-Aware Deep Learning for Mango Leaf Disease Classification:
             Transfer Learning, Explainable AI, and Deployment-Oriented Evaluation},
  author  = {Chansanam, Wirapong},
  year    = {2026},
  note    = {Manuscript}
}
```

---

## ⚠️ Limitations & Intended Use

- Trained and validated on the MangoLeafBD formulation; performance on orchard-level, device-level, or geographically unseen images has **not** been guaranteed and is identified as future work.
- Intended as a **decision-support tool** for researchers and practitioners — **not** a substitute for in-field agronomic diagnosis.
- The leaf-validity pre-filter mitigates but does not eliminate confident misclassification on out-of-distribution inputs.

---

## 🙏 Acknowledgements

- The MangoLeafBD dataset authors for the original image collection.
- The TensorFlow/Keras, Hugging Face, Gradio, and Streamlit communities.
- Prior work by Thalor, Mate, Shiralkar & Shinde (2025, JISEM) as a comparative benchmark.

---

## 📄 License

Released for academic and research use. Please consult the repository owner before commercial use or redistribution of the trained model weights.
