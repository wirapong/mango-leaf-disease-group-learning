"""Grad-CAM visualization for deployed EfficientNetB0 mango disease classifier."""
from pathlib import Path
import json, os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

BASE = Path('/Users/wirapongc/Library/CloudStorage/GoogleDrive-wirach@kku.ac.th/My Drive/OpenClawMacMini/Mango Plant Disease')
MODEL_PATH = BASE / 'outputs/deployable_model_package/best_model.keras'
LABELS_PATH = BASE / 'outputs/deployable_model_package/labels.json'
TEST_CSV    = BASE / 'outputs/grouped_split_v1/test_split.csv'
OUT_DIR     = BASE / 'outputs/gradcam_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

THAI_MAP = {
    'Anthracnose': 'แอนแทรคโนส',
    'Bacterial Canker': 'แบคทีเรียแคงเกอร์',
    'Cutting Weevil': 'มอดเจาะกิ่ง',
    'Die Back': 'ไดแบ็ก',
    'Gall Midge': 'แมลงวันตัวอ่อน',
    'Healthy': 'ใบปกติ',
    'Powdery Mildew': 'ราแป้ง',
    'Sooty Mould': 'ราดำ',
}

def load_preprocess(path: str, size: int = 224) -> np.ndarray:
    img = Image.open(path).convert('RGB').resize((size, size))
    arr = tf.keras.applications.efficientnet.preprocess_input(np.array(img, dtype=np.float32))
    return np.expand_dims(arr, 0), img

def build_gradcam_pipeline(model):
    backbone  = model.get_layer('efficientnetb0')
    aug_layer = model.get_layer('augmentation')
    last_conv = backbone.get_layer('top_activation')
    avg_pool  = backbone.get_layer('avg_pool')

    pre_conv = tf.keras.Model(backbone.inputs, last_conv.output)

    ci = tf.keras.Input(shape=last_conv.output.shape[1:])
    post_conv = tf.keras.Model(ci, avg_pool(ci))

    hi = tf.keras.Input(shape=backbone.output.shape[1:])
    x = hi
    for lname in ('batch_normalization', 'dropout', 'dense'):
        x = model.get_layer(lname)(x)
    head = tf.keras.Model(hi, x)

    return aug_layer, pre_conv, post_conv, head

def compute_gradcam(aug_layer, pre_conv, post_conv, head, img_array, class_idx):
    aug_out = aug_layer(img_array, training=False)
    conv_features = pre_conv(aug_out, training=False)
    with tf.GradientTape() as tape:
        tape.watch(conv_features)
        pool_out = post_conv(conv_features, training=False)
        preds    = head(pool_out, training=False)
        score    = preds[:, class_idx]
    grads = tape.gradient(score, conv_features)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = (conv_features[0] @ pooled_grads[..., tf.newaxis]).numpy().squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8
    return heatmap, preds.numpy()[0]

def make_overlay(orig_img: Image.Image, heatmap: np.ndarray, alpha=0.45) -> np.ndarray:
    orig = np.array(orig_img.convert('RGB').resize((224, 224)), dtype=np.float32)
    hm_u8 = np.uint8(heatmap * 255)
    hm_res = np.array(Image.fromarray(hm_u8, 'L').resize((224, 224), Image.BILINEAR))
    jet_rgb = (cm.jet(np.arange(256))[:, :3] * 255).astype(np.uint8)
    colored = jet_rgb[hm_res].astype(np.float32)
    overlay = (1 - alpha) * orig + alpha * colored
    return np.clip(overlay, 0, 255).astype(np.uint8)

def main():
    print('Loading model...')
    model = tf.keras.models.load_model(MODEL_PATH)
    labels_data = json.loads(LABELS_PATH.read_text())
    class_names = labels_data['class_names']

    aug_layer, pre_conv, post_conv, head = build_gradcam_pipeline(model)

    test_df = pd.read_csv(TEST_CSV)
    # metrics to pick correctly-predicted samples per class
    metrics_path = BASE / 'outputs/grouped_model_search_v1/efficientnetb0/metrics.json'
    metrics = json.loads(metrics_path.read_text())
    preds_list  = metrics['predictions']
    probs_list  = metrics['probabilities']

    # get one correctly-classified image per class
    samples = {}
    for idx, (row, pred_idx) in enumerate(zip(test_df.itertuples(), preds_list)):
        true_idx = class_names.index(row.label)
        if pred_idx == true_idx and row.label not in samples:
            samples[row.label] = row.filepath
        if len(samples) == len(class_names):
            break

    print(f'Selected {len(samples)} sample images')

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    results = {}
    for ax_i, cls in enumerate(class_names):
        img_path = samples.get(cls)
        if img_path is None:
            # fallback: first image of class
            row = test_df[test_df['label'] == cls].iloc[0]
            img_path = row.filepath
        img_array, orig_img = load_preprocess(img_path)
        cls_idx = class_names.index(cls)
        heatmap, pred_probs = compute_gradcam(aug_layer, pre_conv, post_conv, head, img_array, cls_idx)
        overlay = make_overlay(orig_img, heatmap)

        # save individual overlay
        overlay_path = OUT_DIR / f'gradcam_{cls.replace(" ", "_")}.png'
        Image.fromarray(overlay).save(overlay_path, dpi=(400, 400))

        axes[ax_i].imshow(overlay)
        axes[ax_i].set_title(f'{cls}\n({THAI_MAP[cls]})', fontsize=9, pad=4)
        axes[ax_i].axis('off')
        results[cls] = {
            'image': str(img_path),
            'pred_class': class_names[int(np.argmax(pred_probs))],
            'confidence': float(np.max(pred_probs)),
            'gradcam_file': str(overlay_path),
        }
        print(f'  [{ax_i+1}/8] {cls} → pred {results[cls]["pred_class"]} ({results[cls]["confidence"]*100:.1f}%)')

    fig.suptitle('Grad-CAM Visualizations — EfficientNetB0 (Grouped-Split Evaluation)', fontsize=11, y=1.01)
    plt.tight_layout()
    grid_path = OUT_DIR / 'Figure_GradCAM_grid.png'
    fig.savefig(grid_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'\nGrid figure saved: {grid_path}')

    (OUT_DIR / 'gradcam_results.json').write_text(json.dumps(results, indent=2))
    print('Done.')

if __name__ == '__main__':
    main()
