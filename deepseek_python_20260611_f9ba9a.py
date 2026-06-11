#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DROWSINESS DETECTION - YOLOv8 FINAL PROJECT
Detekcija pospanosti koristeći YOLOv8s model

Autor: Student
Datum: Jun 2026

Optimizovano:
- agnostic_nms=True (rešava problem duplih klasa na istom licu)
- conf=0.55 (optimalan balans za eliminaciju lažnih detekcija)
"""

import os
import sys
import tarfile
import zipfile
import shutil
import random
import csv
import yaml
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO

# ============================================================
# 0. INSTALACIJA (samo ako nije instalirano)
# ============================================================
def install_packages():
    """Instalira potrebne pakete"""
    print("📦 Instalacija potrebnih paketa...")
    os.system("pip install ultralytics pyyaml tqdm pandas matplotlib opencv-python-headless pillow -q")
    print("✅ Instalacija završena")

# ============================================================
# 1. MOUNT GOOGLE DISK (samo za Colab)
# ============================================================
def mount_google_drive():
    """Mountuje Google Drive - preskače ako nije Colab"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Disk je uspešno povezan!")
        return True
    except ImportError:
        print("⚠️ Nije detektovan Colab environment - preskačem mount Drive-a")
        return False

# ============================================================
# 2. RASKUJ TAR FAJL SA DISKA
# ============================================================
def extract_tar_file(tar_path, dest_path):
    """Raspakuje TAR fajl"""
    if not os.path.exists(tar_path):
        print(f"❌ Fajl nije pronađen: {tar_path}")
        return False
    
    print(f"✅ Fajl pronadjen: {tar_path}")
    velicina = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"   Veličina: {velicina:.2f} MB")
    
    Path(dest_path).mkdir(parents=True, exist_ok=True)
    print(f"\n📦 Raspakujem u '{dest_path}'...")
    
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=dest_path)
    
    print("✅ Raspakivanje završeno!")
    return True

# ============================================================
# 3. PROVERA STRUKTURE DATASETA
# ============================================================
def check_dataset_structure(dataset_path):
    """Proverava strukturu dataseta"""
    print("\n" + "="*60)
    print("🔍 PROVERA STRUKTURE DATASETA")
    print("="*60)
    
    images_path = Path(dataset_path) / 'images' / 'train'
    labels_path = Path(dataset_path) / 'labels' / 'train'
    
    print(f"📸 Slike: {images_path}")
    print(f"📝 Anotacije: {labels_path}")
    
    if images_path.exists():
        slike = list(images_path.glob('*.*'))
        print(f"   Slike: {len([s for s in slike if s.suffix.lower() in ['.jpg', '.jpeg', '.png']])}")
    else:
        print(f"   ❌ Folder ne postoji: {images_path}")
    
    if labels_path.exists():
        anotacije = list(labels_path.glob('*.txt'))
        print(f"   Anotacije: {len(anotacije)}")
    else:
        print(f"   ❌ Folder ne postoji: {labels_path}")
    
    return images_path, labels_path

# ============================================================
# 4. PODELA DATASETA 80/20
# ============================================================
def split_dataset(images_dir, labels_dir):
    """Delj dataset na train/val 80/20"""
    print("\n" + "="*60)
    print("🔧 PODELA DATASET-A 80/20")
    print("="*60)
    
    print(f"📸 Slike: {images_dir}")
    print(f"📝 Anotacije: {labels_dir}")
    
    # Pronađi sve slike
    sve_slike = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        sve_slike.extend(images_dir.glob(ext))
    
    print(f"   Ukupno slika: {len(sve_slike)}")
    
    # Mapiranje slika sa anotacijama
    img_ann_map = {}
    slike_bez_anotacije = []
    
    for img in sve_slike:
        ann_path = labels_dir / f"{img.stem}.txt"
        if ann_path.exists():
            img_ann_map[img] = ann_path
        else:
            slike_bez_anotacije.append(img)
    
    print(f"   Slike sa anotacijama: {len(img_ann_map)}")
    print(f"   Slike bez anotacija: {len(slike_bez_anotacije)}")
    
    if len(img_ann_map) == 0:
        print("❌ Nema anotacija! Prekidam.")
        return False
    
    # Mešanje i podela
    print("\n🎲 Mešam i delim 80/20...")
    slike_lista = list(img_ann_map.keys())
    random.seed(42)
    random.shuffle(slike_lista)
    
    total = len(slike_lista)
    train_count = int(total * 0.8)
    val_count = total - train_count
    
    train_slike = slike_lista[:train_count]
    val_slike = slike_lista[train_count:]
    
    print(f"\n📊 PODELA:")
    print(f"   Trening (80%): {len(train_slike)} slika")
    print(f"   Validacija (20%): {len(val_slike)} slika")
    
    # Kreiranje foldera
    for folder in ['train', 'val']:
        if Path(folder).exists():
            shutil.rmtree(folder)
    
    os.makedirs('train/images', exist_ok=True)
    os.makedirs('train/labels', exist_ok=True)
    os.makedirs('val/images', exist_ok=True)
    os.makedirs('val/labels', exist_ok=True)
    
    # Kopiranje fajlova
    print("\n📦 Kopiranje fajlova...")
    for img in train_slike:
        shutil.copy2(img, f'train/images/{img.name}')
        shutil.copy2(img_ann_map[img], f'train/labels/{img_ann_map[img].name}')
    
    for img in val_slike:
        shutil.copy2(img, f'val/images/{img.name}')
        shutil.copy2(img_ann_map[img], f'val/labels/{img_ann_map[img].name}')
    
    print("✅ Kopiranje završeno!")
    
    # Statistika po klasama
    def count_classes(folder):
        class_count = {0: 0, 1: 0}
        for label in Path(folder).glob('*.txt'):
            with open(label, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        if cls in class_count:
                            class_count[cls] += 1
        return class_count
    
    train_classes = count_classes('train/labels')
    val_classes = count_classes('val/labels')
    
    print("\n📊 STATISTIKA PO KLASAMA:")
    print(f"   Trening - Pospan: {train_classes[0]}, Budan: {train_classes[1]}")
    print(f"   Validacija - Pospan: {val_classes[0]}, Budan: {val_classes[1]}")
    
    # Kreiranje data.yaml
    data_yaml = {
        'path': '.',
        'train': 'train/images',
        'val': 'val/images',
        'nc': 2,
        'names': ['pospan', 'budan']
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print("\n✅ Kreiran data.yaml fajl")
    return True

# ============================================================
# 5. ISPRAVLJANJE SLIKA (PADDING)
# ============================================================
def resize_with_padding(image, target_size=640, pad_color=(114, 114, 114)):
    """Resize sliku dodavanjem padding-a"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    padded = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded, (x_offset, y_offset, new_w, new_h)

def update_annotations_for_padding(label_path, original_size, new_size, pad_info):
    """Ažurira anotacije za padding"""
    x_offset, y_offset, new_w, new_h = pad_info
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        
        cls, x_center, y_center, width, height = parts
        x_center, y_center, width, height = map(float, [x_center, y_center, width, height])
        
        orig_w, orig_h = original_size
        x_pixel = x_center * orig_w
        y_pixel = y_center * orig_h
        w_pixel = width * orig_w
        h_pixel = height * orig_h
        
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        
        x_pixel_final = (x_pixel * scale_x) + x_offset
        y_pixel_final = (y_pixel * scale_y) + y_offset
        w_pixel_scaled = w_pixel * scale_x
        h_pixel_scaled = h_pixel * scale_y
        
        x_center_new = max(0.0, min(1.0, x_pixel_final / new_size))
        y_center_new = max(0.0, min(1.0, y_pixel_final / new_size))
        width_new = max(0.0, min(1.0, w_pixel_scaled / new_size))
        height_new = max(0.0, min(1.0, h_pixel_scaled / new_size))
        
        new_lines.append(f"{cls} {x_center_new:.6f} {y_center_new:.6f} {width_new:.6f} {height_new:.6f}\n")
    
    with open(label_path, 'w') as f:
        f.writelines(new_lines)

def process_dataset_padding():
    """Primenjuje padding na sve slike"""
    print("\n" + "="*60)
    print("🔧 ISPRAVLJANJE SLIKA NA UNIFORMAN FORMAT (PADDING)")
    print("="*60)
    
    TARGET_SIZE = 640
    TRAIN_IMAGES = Path('train/images')
    VAL_IMAGES = Path('val/images')
    
    print("\n🖼️ OBRADA TRENING SKUPA")
    train_images = list(TRAIN_IMAGES.glob('*.*'))
    for img_path in tqdm(train_images, desc="   Obrada trening slika"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        original_size = (img.shape[1], img.shape[0])
        padded_img, pad_info = resize_with_padding(img, TARGET_SIZE)
        cv2.imwrite(str(img_path), padded_img)
        
        label_path = Path('train/labels') / f"{img_path.stem}.txt"
        if label_path.exists():
            update_annotations_for_padding(label_path, original_size, TARGET_SIZE, pad_info)
    
    print("\n🖼️ OBRADA VALIDACIONOG SKUPA")
    val_images = list(VAL_IMAGES.glob('*.*'))
    for img_path in tqdm(val_images, desc="   Obrada validacionih slika"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        original_size = (img.shape[1], img.shape[0])
        padded_img, pad_info = resize_with_padding(img, TARGET_SIZE)
        cv2.imwrite(str(img_path), padded_img)
        
        label_path = Path('val/labels') / f"{img_path.stem}.txt"
        if label_path.exists():
            update_annotations_for_padding(label_path, original_size, TARGET_SIZE, pad_info)
    
    print("\n✅ SVE SLIKE SU DOVEDENE U ISTI FORMAT (640x640)!")

# ============================================================
# 6. TRENING MODELA
# ============================================================
def train_model():
    """Pokreće trening YOLOv8s modela"""
    print("\n" + "="*60)
    print("🚀 POKRETANJE TRENINGA (YOLOv8s, mixup, patience=25)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Uređaj: {device.upper()}")
    
    print("\n🔧 KONFIGURACIJA TRENINGA:")
    print("   Model: YOLOv8s (11.2M parametara)")
    print("   Epohe: 150")
    print("   Patience: 25")
    print("   Mixup: 0.2")
    print("   Copy-paste: 0.1")
    print("   cls_pw: 1.0")
    
    model = YOLO('yolov8s.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=150,
        imgsz=640,
        batch=64,
        device=device,
        workers=4,
        patience=25,
        save=True,
        save_period=10,
        name='drowsiness_v8s_improved',
        verbose=True,
        mixup=0.2,
        copy_paste=0.1,
        cls_pw=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
    )
    
    print("\n✅ TRENING ZAVRŠEN!")
    return results

# ============================================================
# 7. TESTIRANJE SA OPTIMIZOVANIM PARAMETRIMA
# ============================================================
def test_model(test_zip_path=None):
    """Testira model na test skupu sa conf=0.55 i agnostic_nms=True"""
    print("\n" + "="*60)
    print("🚀 TESTIRANJE SA OPTIMIZOVANIM PARAMETRIMA")
    print("="*60)
    print("   ⚙️ agnostic_nms=True - sprečava duple klase na istom licu")
    print("   ⚙️ conf=0.55 - optimalan balans za eliminaciju lažnih detekcija")
    print("="*60)
    
    # Optimizovani parametri
    CONF_THRESHOLD = 0.55
    AGNOSTIC_NMS = True
    IOU_THRESHOLD = 0.7
    
    print(f"\n⚙️ Postavke testiranja:")
    print(f"   confidence threshold: {CONF_THRESHOLD}")
    print(f"   agnostic_nms: {AGNOSTIC_NMS}")
    print(f"   iou_threshold: {IOU_THRESHOLD}")
    
    # Pronađi najbolji model
    best_model_path = None
    runs_dir = Path('runs/detect')
    if runs_dir.exists():
        for subdir in runs_dir.iterdir():
            if 'drowsiness_v8s_improved' in subdir.name:
                best_path = subdir / 'weights' / 'best.pt'
                if best_path.exists():
                    best_model_path = best_path
                    break
    
    if best_model_path is None and runs_dir.exists():
        for subdir in runs_dir.iterdir():
            best_path = subdir / 'weights' / 'best.pt'
            if best_path.exists():
                best_model_path = best_path
                break
    
    if best_model_path is None:
        pt_files = list(Path('.').rglob('*.pt'))
        if pt_files:
            best_model_path = pt_files[0]
    
    if best_model_path is None:
        print("❌ Nema modela za testiranje!")
        return
    
    print(f"\n📁 Model: {best_model_path}")
    model = YOLO(best_model_path)
    
    # Pronađi test slike
    test_images = []
    if test_zip_path and os.path.exists(test_zip_path):
        extract_path = "/content/test_dataset"
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(Path(extract_path).rglob(ext))
        print(f"📸 Test slika sa Drive-a: {len(test_images)}")
    else:
        val_images = Path('val/images')
        if val_images.exists():
            test_images = list(val_images.glob('*.jpg')) + list(val_images.glob('*.png'))
            print(f"📸 Validacione slike: {len(test_images)}")
    
    if not test_images:
        print("❌ Nema test slika!")
        return
    
    # Statistika
    total_images = len(test_images)
    images_with_both_classes = 0
    total_detections = 0
    
    for img_path in tqdm(test_images, desc="Testiranje"):
        results = model(img_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, agnostic_nms=AGNOSTIC_NMS)
        
        if len(results[0].boxes) > 0:
            classes = results[0].boxes.cls.cpu().numpy()
            total_detections += len(classes)
            if len(set(classes)) == 2:
                images_with_both_classes += 1
    
    print(f"\n📊 REZULTATI TESTIRANJA:")
    print(f"   📸 Ukupno test slika: {total_images}")
    print(f"   🔴 Slike sa OBE KLASE: {images_with_both_classes} ({images_with_both_classes/total_images*100:.1f}%)")
    print(f"   📦 Ukupno detekcija: {total_detections}")
    print(f"   📊 Prosečno detekcija po slici: {total_detections/total_images:.2f}")
    
    if images_with_both_classes == 0:
        print("\n   ✅ USPJEH! Nema slika sa obe klase istovremeno!")
    else:
        print(f"\n   ⚠️ I dalje ima {images_with_both_classes} slika sa obe klase.")
    
    print("\n✅ PREPORUKA ZA KORIŠĆENJE:")
    print("   results = model('slika.jpg', conf=0.55, agnostic_nms=True)")

# ============================================================
# 8. GLAVNA FUNKCIJA
# ============================================================
def main():
    """Glavna funkcija koja pokreće ceo pipeline"""
    print("\n" + "="*60)
    print("🎯 DROWSINESS DETECTION - YOLOv8 FINAL PROJECT")
    print("="*60)
    
    # Putanje (PROMENITI PREMA POTREBI!)
    TAR_PATH = "/content/drive/MyDrive/inteligentni/trening+val.tar"
    TEST_ZIP_PATH = "/content/drive/MyDrive/inteligentni/test/test_finale.zip"
    DATASET_PATH = "/content/dataset"
    
    # 1. Mount Drive (samo za Colab)
    is_colab = mount_google_drive()
    
    # 2. Raspakuj TAR
    if not extract_tar_file(TAR_PATH, DATASET_PATH):
        print("❌ Nije moguće nastaviti bez dataseta!")
        return
    
    # 3. Proveri strukturu
    images_dir, labels_dir = check_dataset_structure(DATASET_PATH)
    
    # 4. Podela dataseta
    if not split_dataset(images_dir, labels_dir):
        return
    
    # 5. Padding slika
    process_dataset_padding()
    
    # 6. Trening modela
    train_model()
    
    # 7. Testiranje na validacionom skupu
    test_model()
    
    # 8. Testiranje na Drive test skupu (ako postoji)
    if os.path.exists(TEST_ZIP_PATH):
        test_model(TEST_ZIP_PATH)
    
    print("\n" + "="*60)
    print("🎉 PROJEKAT ZAVRŠEN!")
    print("="*60)
    print("\n📌 FINALNA PREPORUKA:")
    print("   from ultralytics import YOLO")
    print("   model = YOLO('runs/detect/drowsiness_v8s_improved/weights/best.pt')")
    print("   results = model('slika.jpg', conf=0.55, agnostic_nms=True)")

# ============================================================
# POKRETANJE
# ============================================================
if __name__ == "__main__":
    main()