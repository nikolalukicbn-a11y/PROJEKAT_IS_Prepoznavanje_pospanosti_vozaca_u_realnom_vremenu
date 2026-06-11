#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKRIPTA 3: TESTIRANJE
- Merenje brzine inference-a (razliciti batch size-ovi)
- Grafikoni metrika i trenda učenja
- Testiranje sa optimizovanim parametrima (conf=0.55, agnostic_nms=True)
- CSV izveštaj
"""

import sys
import time
import csv
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

import torch
from ultralytics import YOLO

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# KONFIGURACIJA
# ============================================================
BASE_DIR = Path(__file__).parent
TEST_ZIP_PATH = BASE_DIR / 'test_finale.zip'
TEST_EXTRACT_DIR = BASE_DIR / 'test_dataset'

CONF_THRESHOLD = 0.55
AGNOSTIC_NMS = True
IOU_THRESHOLD = 0.7

OUTPUT_DIR = Path('test_results')
OUTPUT_DIR.mkdir(exist_ok=True)

matplotlib.rcParams['figure.dpi'] = 150


def find_best_model():
    runs_dir = Path('runs/detect')
    if runs_dir.exists():
        for subdir in sorted(runs_dir.iterdir(), reverse=True):
            best = subdir / 'weights' / 'best.pt'
            if best.exists():
                return best

    for pt in Path('.').glob('*.pt'):
        if pt.name != 'yolov8s.pt' and pt.name != 'yolov8n.pt':
            return pt
    return None


def measure_inference_speed(model, device):
    print("\n" + "=" * 60)
    print("1. MERENJE BRZINE INFERENCE-A")
    print("=" * 60)

    val_images = Path('val/images')
    if not val_images.exists():
        print("⚠️ val/images ne postoji.")
        return

    test_images = list(val_images.glob('*.jpg')) + list(val_images.glob('*.png'))
    test_images = test_images[:min(100, len(test_images))]

    if not test_images:
        print("❌ Nema slika za testiranje!")
        return

    print(f"📸 Test slika: {len(test_images)}")

    for _ in range(5):
        _ = model(test_images[0])

    times = []
    batch_sizes = [1, 4, 16, 32]

    for bs in batch_sizes:
        if bs > len(test_images):
            break
        batch_times = []

        for i in tqdm(range(0, len(test_images), bs), desc=f"Batch {bs}"):
            batch = test_images[i:i + bs]
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(batch)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            batch_times.append(elapsed / len(batch))

        avg_ms = np.mean(batch_times) * 1000
        std_ms = np.std(batch_times) * 1000
        times.append({
            'batch_size': bs,
            'avg_ms': avg_ms,
            'std_ms': std_ms,
            'fps': 1000 / avg_ms
        })

        print(f"  Batch {bs:2d}: {avg_ms:6.2f} ms | FPS: {1000/avg_ms:6.1f} | std: {std_ms:.2f} ms")

    df_times = pd.DataFrame(times)
    df_times.to_csv(OUTPUT_DIR / 'inference_speed.csv', index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(df_times['batch_size'], df_times['avg_ms'], color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Vreme (ms)')
    ax1.set_title('Inference Time po Batch Size-u')
    ax1.grid(alpha=0.3)

    ax2.plot(df_times['batch_size'], df_times['fps'], 'b-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('FPS')
    ax2.set_title('FPS po Batch Size-u')
    ax2.grid(alpha=0.3)

    plt.suptitle('Analiza brzine inference-a', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'inference_speed.png')
    plt.close()
    print(f"✅ Grafikon sačuvan: {OUTPUT_DIR / 'inference_speed.png'}")


def plot_learning_curves():
    print("\n" + "=" * 60)
    print("2. GRAFIKONI TRENDA UČENJA")
    print("=" * 60)

    results_csv = None
    runs_dir = Path('runs/detect')
    if runs_dir.exists():
        for subdir in sorted(runs_dir.iterdir(), reverse=True):
            rcsv = subdir / 'results.csv'
            if rcsv.exists():
                results_csv = rcsv
                break

    if results_csv is None:
        print("⚠️ results.csv nije pronađen.")
        return

    df = pd.read_csv(results_csv)
    print(f"📁 {results_csv} — {len(df)} epoha")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if 'metrics/mAP50(B)' in df.columns:
        ax = axes[0, 0]
        ax.plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP50')
        ax.set_title('mAP50 tokom treninga')
        ax.grid(alpha=0.3)
        ax.axhline(y=df['metrics/mAP50(B)'].max(), color='r', linestyle='--', alpha=0.5)
        ax.text(0.7, 0.95, f"Max: {df['metrics/mAP50(B)'].max():.4f}",
                transform=ax.transAxes, fontsize=9)

    ax = axes[0, 1]
    ax.plot(df['epoch'], df['val/box_loss'], 'r-', linewidth=2, label='Box Loss')
    ax.plot(df['epoch'], df['val/cls_loss'], 'g-', linewidth=2, label='Cls Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validacioni gubici')
    ax.legend()
    ax.grid(alpha=0.3)

    if 'metrics/precision(B)' in df.columns:
        ax = axes[1, 0]
        ax.plot(df['epoch'], df['metrics/precision(B)'], 'b-', linewidth=2, label='Precision')
        ax.plot(df['epoch'], df['metrics/recall(B)'], 'orange', linewidth=2, label='Recall')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Vrednost')
        ax.set_title('Precision i Recall')
        ax.legend()
        ax.grid(alpha=0.3)

    if 'metrics/mAP50-95(B)' in df.columns:
        ax = axes[1, 1]
        ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP50-95')
        ax.set_title('mAP50-95 tokom treninga')
        ax.grid(alpha=0.3)

    plt.suptitle('Trend učenja — Sve epohe', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'learning_curves.png')
    plt.close()
    print(f"✅ Grafikon sačuvan: {OUTPUT_DIR / 'learning_curves.png'}")


def test_with_optimized_params(model):
    print("\n" + "=" * 60)
    print("3. TESTIRANJE SA OPTIMIZOVANIM PARAMETRIMA")
    print("=" * 60)
    print(f"  conf={CONF_THRESHOLD}, agnostic_nms={AGNOSTIC_NMS}, iou={IOU_THRESHOLD}")

    test_images = []
    if TEST_ZIP_PATH.exists():
        if TEST_EXTRACT_DIR.exists():
            shutil.rmtree(TEST_EXTRACT_DIR)
        TEST_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(TEST_ZIP_PATH, 'r') as zf:
            zf.extractall(TEST_EXTRACT_DIR)

        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(TEST_EXTRACT_DIR.rglob(ext))
        print(f"📸 Test skup (ZIP): {len(test_images)} slika")
    else:
        val_dir = Path('val/images')
        if val_dir.exists():
            test_images = list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png'))
            print(f"📸 Validacioni skup: {len(test_images)} slika")

    if not test_images:
        print("❌ Nema test slika!")
        return

    total = len(test_images)
    both_classes = 0
    total_detections = 0
    all_conf = []
    results_rows = []

    for img_path in tqdm(test_images, desc="Testiranje"):
        results = model(img_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, agnostic_nms=AGNOSTIC_NMS)

        if len(results[0].boxes) > 0:
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            total_detections += len(classes)
            all_conf.extend(confs.tolist())

            if len(set(classes)) >= 2:
                both_classes += 1

            results_rows.append({
                'image': img_path.name,
                'detections': len(classes),
                'avg_confidence': float(confs.mean()),
                'classes': ','.join(str(int(c)) for c in classes)
            })
        else:
            results_rows.append({
                'image': img_path.name,
                'detections': 0,
                'avg_confidence': 0.0,
                'classes': ''
            })

    pd.DataFrame(results_rows).to_csv(OUTPUT_DIR / 'test_results.csv', index=False)

    print(f"\n📊 REZULTATI:")
    print(f"  Ukupno slika:           {total}")
    print(f"  Slike sa obe klase:     {both_classes} ({both_classes/total*100:.1f}%)")
    print(f"  Ukupno detekcija:       {total_detections}")
    print(f"  Prosečno detekcija:     {total_detections/total:.2f}")
    if all_conf:
        print(f"  Prosečan confidence:    {np.mean(all_conf):.3f}")

    if both_classes == 0:
        print("\n  ✅ NEMA slika sa obe klase istovremeno!")
    else:
        print(f"\n  ⚠️ {both_classes} slika i dalje ima obe klase.")

    print(f"\n✅ CSV sačuvan: {OUTPUT_DIR / 'test_results.csv'}")


def main():
    print("=" * 60)
    print("SKRIPTA 3: TESTIRANJE I ANALIZA")
    print("=" * 60)

    model_path = find_best_model()
    if model_path is None:
        print("❌ Nema istreniranog modela! Pokreni prvo 02_trening.py")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📁 Model: {model_path}")
    print(f"🖥️ Uređaj: {device.upper()}")

    model = YOLO(model_path)

    measure_inference_speed(model, device)
    plot_learning_curves()
    test_with_optimized_params(model)

    print("\n" + "=" * 60)
    print(f"✅ SKRIPTA 3 ZAVRŠENA — Rezultati u: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
