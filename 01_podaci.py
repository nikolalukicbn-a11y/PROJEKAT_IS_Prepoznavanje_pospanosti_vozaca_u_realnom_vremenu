#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKRIPTA 1: PODACI
- Raspakivanje tar fajla
- Provera strukture dataseta
- Podela 80/20 (train/val)
- Padding slika na uniforman format
- Kreiranje data.yaml
"""

import sys
import os
import re
import shutil
import random
import tarfile
import yaml
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# KONFIGURACIJA
# ============================================================
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / 'dataset'
DATASET_TAR_PATH = BASE_DIR / 'trening+val.tar'

TARGET_SIZE = 640
RANDOM_SEED = 42
TRAIN_RATIO = 0.8


def sanitize_filename(name):
    parts = name.split('/')
    parts = [re.sub(r'[<>:"|?*]', '_', p) for p in parts]
    return '/'.join(parts)


def extract_tar():
    print("=" * 60)
    print("1. RASPAKIVANJE TAR FAJLA")
    print("=" * 60)

    if not DATASET_TAR_PATH.exists():
        print(f"❌ Tar fajl nije pronadjen: {DATASET_TAR_PATH}")
        return False

    velicina = DATASET_TAR_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ Fajl pronadjen: {DATASET_TAR_PATH}")
    print(f"   Veličina: {velicina:.2f} MB")

    if DATASET_DIR.exists():
        print(f"🗑️ Brišem stari dataset: {DATASET_DIR}")
        shutil.rmtree(DATASET_DIR)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📦 Raspakujem u '{DATASET_DIR}'...")

    with tarfile.open(DATASET_TAR_PATH, 'r') as tar:
        for member in tar.getmembers():
            member.name = sanitize_filename(member.name)
            try:
                tar.extract(member, path=str(DATASET_DIR), filter='fully_trusted')
            except OSError as e:
                print(f"   ⚠️ Preskacem: {member.name} ({e})")

    print("✅ Raspakivanje završeno!")
    return True


def check_structure():
    print("\n" + "=" * 60)
    print("2. PROVERA STRUKTURE DATASETA")
    print("=" * 60)

    images_path = DATASET_DIR / 'images' / 'train'
    labels_path = DATASET_DIR / 'labels' / 'train'

    print(f"📸 Slike: {images_path} — postoji: {images_path.exists()}")
    print(f"📝 Anotacije: {labels_path} — postoji: {labels_path.exists()}")

    if images_path.exists():
        slike = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        print(f"   Broj slika: {len(slike)}")
    if labels_path.exists():
        anotacije = list(labels_path.glob('*.txt'))
        print(f"   Broj anotacija: {len(anotacije)}")

    return images_path if images_path.exists() else None, labels_path if labels_path.exists() else None


def split_dataset(images_dir, labels_dir):
    print("\n" + "=" * 60)
    print("3. PODELA DATASETA 80/20")
    print("=" * 60)

    sve_slike = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        sve_slike.extend(images_dir.glob(ext))

    if not sve_slike:
        print(f"❌ Nema slika u {images_dir}")
        return False

    print(f"   Ukupno slika: {len(sve_slike)}")

    img_ann_map = {}
    bez_anotacije = []

    for img in sve_slike:
        ann = labels_dir / f"{img.stem}.txt"
        if ann.exists():
            img_ann_map[img] = ann
        else:
            bez_anotacije.append(img)

    print(f"   Sa anotacijama: {len(img_ann_map)}")
    print(f"   Bez anotacija:   {len(bez_anotacije)}")

    if len(img_ann_map) == 0:
        print("❌ Nema anotacija!")
        return False

    slike_lista = list(img_ann_map.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(slike_lista)

    train_count = int(len(slike_lista) * TRAIN_RATIO)
    train_slike = slike_lista[:train_count]
    val_slike = slike_lista[train_count:]

    print(f"\n   Trening (80%): {len(train_slike)} slika")
    print(f"   Validacija (20%): {len(val_slike)} slika")

    for folder in ['train', 'val']:
        if Path(folder).exists():
            shutil.rmtree(folder)

    os.makedirs('train/images', exist_ok=True)
    os.makedirs('train/labels', exist_ok=True)
    os.makedirs('val/images', exist_ok=True)
    os.makedirs('val/labels', exist_ok=True)

    print("\n📦 Kopiranje fajlova...")
    for img in train_slike:
        shutil.copy2(img, f'train/images/{img.name}')
        shutil.copy2(img_ann_map[img], f'train/labels/{img_ann_map[img].name}')
    for img in val_slike:
        shutil.copy2(img, f'val/images/{img.name}')
        shutil.copy2(img_ann_map[img], f'val/labels/{img_ann_map[img].name}')

    print("✅ Kopiranje završeno!")

    def count_classes(folder):
        cc = {0: 0, 1: 0}
        for lbl in Path(folder).glob('*.txt'):
            with open(lbl, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        if cls in cc:
                            cc[cls] += 1
        return cc

    tc = count_classes('train/labels')
    vc = count_classes('val/labels')
    print(f"\n   Trening — Pospan: {tc[0]}, Budan: {tc[1]}")
    print(f"   Validacija — Pospan: {vc[0]}, Budan: {vc[1]}")

    data_yaml = {
        'path': '.',
        'train': 'train/images',
        'val': 'val/images',
        'nc': 2,
        'names': ['pospan', 'budan']
    }
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print("\n✅ data.yaml kreiran")
    return True


def resize_with_padding(image, target_size=TARGET_SIZE, pad_color=(114, 114, 114)):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    padded = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)

    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded, (x_offset, y_offset, new_w, new_h)


def update_annotations_padding(label_path, original_size, pad_info):
    x_offset, y_offset, new_w, new_h = pad_info
    orig_w, orig_h = original_size

    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = map(float, parts)

        x_px = xc * orig_w
        y_px = yc * orig_h
        wp = w * orig_w
        hp = h * orig_h

        sx = new_w / orig_w
        sy = new_h / orig_h

        x_final = (x_px * sx) + x_offset
        y_final = (y_px * sy) + y_offset
        w_scaled = wp * sx
        h_scaled = hp * sy

        xc_new = max(0.0, min(1.0, x_final / TARGET_SIZE))
        yc_new = max(0.0, min(1.0, y_final / TARGET_SIZE))
        wn = max(0.0, min(1.0, w_scaled / TARGET_SIZE))
        hn = max(0.0, min(1.0, h_scaled / TARGET_SIZE))

        new_lines.append(f"{int(cls)} {xc_new:.6f} {yc_new:.6f} {wn:.6f} {hn:.6f}\n")

    with open(label_path, 'w') as f:
        f.writelines(new_lines)


def process_padding():
    print("\n" + "=" * 60)
    print("4. PADDING SLIKA (UNIFORMAN FORMAT 640x640)")
    print("=" * 60)

    for split_name, img_dir in [('Trening', 'train/images'), ('Validacija', 'val/images')]:
        lbl_dir = img_dir.replace('images', 'labels')
        images = list(Path(img_dir).glob('*.*'))
        images = [i for i in images if i.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        print(f"\n🖼️ {split_name} skup: {len(images)} slika")
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            original_size = (img.shape[1], img.shape[0])
            padded, pad_info = resize_with_padding(img)
            cv2.imwrite(str(img_path), padded)

            lbl = Path(lbl_dir) / f"{img_path.stem}.txt"
            if lbl.exists():
                update_annotations_padding(lbl, original_size, pad_info)

    print("\n✅ Sve slike su 640x640!")


def main():
    print("=" * 60)
    print("SKRIPTA 1: PRIPREMA PODATAKA")
    print("=" * 60)
    print(f"  Dataset tar:  {DATASET_TAR_PATH}")
    print(f"  Dataset dir:  {DATASET_DIR}")
    print(f"  Target size:  {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"  Train ratio:  {int(TRAIN_RATIO * 100)}%")
    print("=" * 60)

    if not extract_tar():
        print("❌ Ekstrakcija nije uspela.")
        sys.exit(1)

    images_dir, labels_dir = check_structure()
    if images_dir is None or labels_dir is None:
        print("❌ Neispravna struktura dataseta.")
        sys.exit(1)

    if not split_dataset(images_dir, labels_dir):
        print("❌ Podela nije uspela.")
        sys.exit(1)

    process_padding()

    print("\n" + "=" * 60)
    print("✅ SKRIPTA 1 ZAVRŠENA — Podaci spremni za trening!")
    print("=" * 60)


if __name__ == "__main__":
    main()
