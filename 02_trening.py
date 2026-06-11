#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKRIPTA 2: TRENING
- Provera podataka i GPU
- Treniranje YOLOv8s modela
- Logovanje metrika tokom treninga
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from ultralytics import YOLO

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# KONFIGURACIJA
# ============================================================
EPOCHS = 150
BATCH_SIZE = 64
IMG_SIZE = 640
PATIENCE = 25
WORKERS = 4
MODEL_NAME = 'drowsiness_v8s_improved'
BASE_MODEL = 'yolov8s.pt'
DATA_YAML = 'data.yaml'

MIXUP = 0.2
COPY_PASTE = 0.1
CLS_PW = 1.0


def check_data():
    print("=" * 60)
    print("1. PROVERA PODATAKA")
    print("=" * 60)

    train_img = len(list(Path('train/images').glob('*')))
    train_lbl = len(list(Path('train/labels').glob('*.txt')))
    val_img = len(list(Path('val/images').glob('*')))
    val_lbl = len(list(Path('val/labels').glob('*.txt')))

    print(f"  Train: {train_img} slika, {train_lbl} anotacija")
    print(f"  Val:   {val_img} slika, {val_lbl} anotacija")

    if train_img == 0 or val_img == 0:
        print("❌ Nema dovoljno podataka! Pokreni prvo 01_podaci.py")
        return False
    return True


def check_gpu():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Uređaj: {device.upper()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    return device


def train(device):
    print("\n" + "=" * 60)
    print("2. TRENING MODELA")
    print("=" * 60)
    print(f"  Model:      {BASE_MODEL}")
    print(f"  Epohe:      {EPOCHS}")
    print(f"  Batch:      {BATCH_SIZE}")
    print(f"  Img size:   {IMG_SIZE}")
    print(f"  Patience:   {PATIENCE}")
    print(f"  Mixup:      {MIXUP}")
    print(f"  Copy-paste: {COPY_PASTE}")
    print(f"  cls_pw:     {CLS_PW}")
    print("=" * 60)

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=WORKERS,
        patience=PATIENCE,
        save=True,
        save_period=10,
        name=MODEL_NAME,
        verbose=True,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        cls_pw=CLS_PW,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
    )

    return results


def log_results():
    print("\n" + "=" * 60)
    print("3. LOGOVANJE REZULTATA")
    print("=" * 60)

    results_csv = Path(f'runs/detect/{MODEL_NAME}/results.csv')
    if not results_csv.exists():
        print("⚠️ results.csv nije pronađen.")
        return

    df = pd.read_csv(results_csv)

    log_folder = Path('training_logs')
    log_folder.mkdir(exist_ok=True)

    validation_log = log_folder / f'validation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    with open(validation_log, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'box_loss', 'cls_loss', 'precision', 'recall', 'mAP50', 'mAP50-95', 'timestamp'])
        for _, row in df.iterrows():
            writer.writerow([
                int(row['epoch']),
                row.get('val/box_loss', 0),
                row.get('val/cls_loss', 0),
                row.get('metrics/precision(B)', 0),
                row.get('metrics/recall(B)', 0),
                row.get('metrics/mAP50(B)', 0),
                row.get('metrics/mAP50-95(B)', 0),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

    print(f"✅ Log sačuvan: {validation_log}")

    print("\n📊 FINALNI REZULTATI:")
    if 'metrics/mAP50(B)' in df.columns:
        best_map50 = df['metrics/mAP50(B)'].max()
        best_epoch = df['metrics/mAP50(B)'].idxmax() + 1
        print(f"  Najbolji mAP50:  {best_map50:.4f} (epoha {best_epoch})")
        print(f"  Finalni mAP50:   {df['metrics/mAP50(B)'].iloc[-1]:.4f}")

    if 'metrics/mAP50-95(B)' in df.columns:
        print(f"  Najbolji mAP50-95: {df['metrics/mAP50-95(B)'].max():.4f}")

    if 'metrics/precision(B)' in df.columns:
        print(f"  Precision: {df['metrics/precision(B)'].iloc[-1]:.4f}")
        print(f"  Recall:    {df['metrics/recall(B)'].iloc[-1]:.4f}")


def main():
    print("=" * 60)
    print("SKRIPTA 2: TRENING YOLOv8s")
    print("=" * 60)

    if not check_data():
        sys.exit(1)

    device = check_gpu()
    train(device)
    log_results()

    print("\n" + "=" * 60)
    print("✅ SKRIPTA 2 ZAVRŠENA — Model istreniran!")
    print(f"   Model: runs/detect/{MODEL_NAME}/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
