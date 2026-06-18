"""Latency and throughput benchmark for deployed EfficientNetB0."""
import json, os, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

BASE       = Path('/Users/wirapongc/Library/CloudStorage/GoogleDrive-wirach@kku.ac.th/My Drive/OpenClawMacMini/Mango Plant Disease')
MODEL_PATH = BASE / 'outputs/deployable_model_package/best_model.keras'
LABELS_PATH= BASE / 'outputs/deployable_model_package/labels.json'
TEST_CSV   = BASE / 'outputs/grouped_split_v1/test_split.csv'
OUT_DIR    = BASE / 'outputs/latency_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_WARMUP = 5
N_ITERS  = 100
BATCH_SIZES = [1, 4, 8, 16]

def load_preprocess(path: str, size: int = 224) -> np.ndarray:
    img = Image.open(path).convert('RGB').resize((size, size))
    arr = tf.keras.applications.efficientnet.preprocess_input(np.array(img, dtype=np.float32))
    return np.expand_dims(arr, 0)

def main():
    import pandas as pd

    labels = json.loads(LABELS_PATH.read_text())
    image_size = int(labels.get('image_size', 224))

    # ── Cold-start load time ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    model = tf.keras.models.load_model(MODEL_PATH)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f'Model load time: {load_ms:.1f} ms')

    # ── Grab a sample image ───────────────────────────────────────────────────
    df = pd.read_csv(TEST_CSV)
    sample_path = df.iloc[0]['filepath']
    batch1 = load_preprocess(sample_path, image_size)

    # ── Warmup ────────────────────────────────────────────────────────────────
    for _ in range(N_WARMUP):
        _ = model.predict(batch1, verbose=0)

    # ── Single-image latency ──────────────────────────────────────────────────
    times_ms = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        _ = model.predict(batch1, verbose=0)
        times_ms.append((time.perf_counter() - t0) * 1000)
    times_ms = np.array(times_ms)
    single = {
        'n_iters': N_ITERS,
        'mean_ms': float(np.mean(times_ms)),
        'std_ms':  float(np.std(times_ms, ddof=1)),
        'min_ms':  float(np.min(times_ms)),
        'p50_ms':  float(np.percentile(times_ms, 50)),
        'p95_ms':  float(np.percentile(times_ms, 95)),
        'max_ms':  float(np.max(times_ms)),
    }
    fps_single = 1000.0 / single['mean_ms']
    print(f'\nSingle-image inference (batch=1, n={N_ITERS}):')
    print(f'  mean {single["mean_ms"]:.1f} ms | p95 {single["p95_ms"]:.1f} ms | throughput {fps_single:.1f} img/s')

    # ── Batch throughput ──────────────────────────────────────────────────────
    batch_results = []
    for bs in BATCH_SIZES:
        # build batch
        paths = df['filepath'].iloc[:bs * 10].tolist()
        imgs = np.concatenate([load_preprocess(p, image_size) for p in paths[:bs]], axis=0)
        # warmup
        for _ in range(N_WARMUP):
            _ = model.predict(imgs, verbose=0)
        # measure
        bt = []
        for _ in range(max(10, N_ITERS // bs)):
            t0 = time.perf_counter()
            _ = model.predict(imgs, verbose=0)
            bt.append((time.perf_counter() - t0) * 1000)
        bt = np.array(bt)
        per_img = bt / bs
        fps = 1000.0 / per_img.mean()
        row = {'batch_size': bs, 'batch_mean_ms': float(bt.mean()), 'per_img_ms': float(per_img.mean()), 'throughput_img_per_s': float(fps)}
        batch_results.append(row)
        print(f'  batch={bs}: {bt.mean():.1f} ms total | {per_img.mean():.1f} ms/img | {fps:.1f} img/s')

    # ── Save summary ─────────────────────────────────────────────────────────
    summary = {
        'device': 'gpu' if tf.config.list_physical_devices('GPU') else 'cpu',
        'model': 'efficientnetb0',
        'image_size': image_size,
        'load_ms': load_ms,
        'single_image': single,
        'single_image_throughput_img_per_s': fps_single,
        'batch_throughput': batch_results,
    }
    out_path = OUT_DIR / 'latency_summary.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f'\nSummary saved: {out_path}')
    return summary

if __name__ == '__main__':
    main()
