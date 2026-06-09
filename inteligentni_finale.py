# -*- coding: utf-8 -*-
"""INTELIGENTNI_FINALE - konvertovano iz Colab .ipynb u .py"""

import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# KONFIGURACIJA - Podesi putanje pre pokretanja!
# ============================================================

# Putanja do glavnog dataset foldera (umesto /content/dataset)
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')

# Putanja do tar fajla sa podacima (ostavi None ako vec imas raspakovane podatke)
DATASET_TAR_PATH = os.path.join(os.getcwd(), 'trening+val.tar')

# Putanja do test ZIP fajla (ostavi None ako nemas test podatke)
TEST_ZIP_PATH = os.path.join(os.getcwd(), 'test_finale.zip')

# Putanja za ekstrakciju test podataka
TEST_EXTRACT_DIR = os.path.join(os.getcwd(), 'test_dataset')

# Pre pokretanja instaliraj:
#   pip install ultralytics pyyaml torch pandas matplotlib pillow tqdm

print("=" * 60)
print("KONFIGURACIJA:")
print(f"  Dataset folder: {DATASET_DIR}")
print(f"  Dataset tar:    {DATASET_TAR_PATH}")
print(f"  Test ZIP:       {TEST_ZIP_PATH}")
print(f"  Test extract:   {TEST_EXTRACT_DIR}")
print("=" * 60)

# ============================================================
# 2. PRONAĐI I RASKUJ TAR FAJL SA DISKA
# ============================================================

import tarfile
import re
import shutil

PUTANJA_NA_DISKU = DATASET_TAR_PATH
DESTINACIJA = DATASET_DIR

import re

def sanitize_filename(name):
    """Zameni karaktere nelegalne za Windows u imenu fajla, ali zadrzi strukturu foldera"""
    parts = name.split('/')
    parts = [re.sub(r'[<>:"|?*]', '_', p) for p in parts]
    return '/'.join(parts)

if PUTANJA_NA_DISKU and os.path.exists(PUTANJA_NA_DISKU):
    print(f"✅ Fajl pronadjen: {PUTANJA_NA_DISKU}")

    velicina = os.path.getsize(PUTANJA_NA_DISKU) / (1024 * 1024)
    print(f"   Veličina: {velicina:.2f} MB")

    if Path(DESTINACIJA).exists():
        print(f"🗑️ Brišem stari dataset folder: {DESTINACIJA}")
        shutil.rmtree(DESTINACIJA)

    Path(DESTINACIJA).mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Raspakujem u '{DESTINACIJA}'...")
    with tarfile.open(PUTANJA_NA_DISKU, 'r') as tar:
        for member in tar.getmembers():
            member.name = sanitize_filename(member.name)
            try:
                tar.extract(member, path=DESTINACIJA, filter='fully_trusted')
            except OSError as e:
                print(f"   ⚠️ Preskacem fajl: {member.name} ({e})")

    print("✅ Raspakivanje završeno!")

    print("\n📁 Struktura raspakovanog foldera:")
    for item in Path(DESTINACIJA).iterdir():
        if item.is_dir():
            print(f"   📂 {item.name}/")
            for i, sub in enumerate(item.iterdir()):
                if i < 2 and sub.is_dir():
                    print(f"      📂 {sub.name}/")
            if len(list(item.iterdir())) > 2:
                print(f"      ...")
        else:
            print(f"   📄 {item.name}")

elif PUTANJA_NA_DISKU:
    print(f"❌ GRESKA: Tar fajl nije pronadjen: {PUTANJA_NA_DISKU}")

else:
    print("ℹ️ DATASET_TAR_PATH nije podešen, preskacem ekstrakciju tar fajla.")
    print("   Ako su podaci vec raspakovani, podesi DATASET_DIR na tu putanju.")

# ============================================================
# 3. PROVERA DA LI SU SLIKE ISPRAVNO RASKUJENE
# ============================================================

DESTINACIJA = DATASET_DIR

# Pretražujemo slike (common formati)
slike = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    slike.extend(Path(DESTINACIJA).rglob(ext))

print(f"🖼️ Ukupno pronađenih slika: {len(slike)}")

# Prikaz strukture foldera
print("\n📁 Struktura foldera sa slikama:")
for folder in Path(DESTINACIJA).iterdir():
    if folder.is_dir():
        broj_slika = len(list(folder.rglob('*.jpg')) + list(folder.rglob('*.png')))
        print(f"   📂 {folder.name}/ - {broj_slika} slika")

        # Prikaz podfoldera
        for sub in folder.iterdir():
            if sub.is_dir():
                broj_sub = len(list(sub.rglob('*.jpg')) + list(sub.rglob('*.png')))
                print(f"      📂 {sub.name}/ - {broj_sub} slika")

from pathlib import Path
import os

print("🔍 DIJAGNOSTIKA - PRONALAŽENJE FAJLOVA")
print("="*60)

# 1. Proveri trenutni direktorijum
print(f"\n📁 Trenutni direktorijum: {os.getcwd()}")
print("   Sadržaj:")
for item in Path('.').iterdir():
    if item.is_dir():
        print(f"   📂 {item.name}/")
    else:
        print(f"   📄 {item.name}")

# 2. Proveri dataset_raspakovan (alternativno ime)
dataset_path = Path(DATASET_DIR) if Path(DATASET_DIR).exists() else Path(DATASET_DIR.parent / 'dataset_raspakovan')
print(f"\n📁 Dataset path: {dataset_path}")
print(f"   Postoji: {dataset_path.exists()}")

if dataset_path.exists():
    print("\n   Sadržaj dataset_raspakovan:")
    for item in dataset_path.iterdir():
        if item.is_dir():
            print(f"   📂 {item.name}/")
            # Prikaži podfoldere
            for sub in item.iterdir():
                if sub.is_dir():
                    print(f"      📂 {sub.name}/")
                    # Prikaži fajlove u podfolderu
                    files = list(sub.glob('*'))[:3]
                    for f in files:
                        print(f"         📄 {f.name}")
        else:
            print(f"   📄 {item.name}")

# 3. Pretraži ceo disk za slike
print("\n" + "="*60)
print("🔍 PRETRAGA ZA SLIKAMA (.jpg, .png)")

slike = []
for ext in ['*.jpg', '*.jpeg', '*.png']:
    slike.extend(Path(DATASET_DIR).rglob(ext))

print(f"Ukupno pronađeno slika: {len(slike)}")

if slike:
    print("\n   Primeri pronađenih slika:")
    for img in slike[:10]:
        print(f"   - {img}")

        # Proveri da li postoji anotacija pored slike
        ann = img.with_suffix('.txt')
        if ann.exists():
            print(f"     ✅ Anotacija postoji: {ann.name}")
        else:
            print(f"     ❌ Nema anotacije")
else:
    print("   ❌ NEMA SLIKA!")

# 4. Proveri dataset folder
print("\n" + "="*60)
print("🔍 PROVERA DATASET FOLDERA")

if Path(DATASET_DIR).exists():
    print(f"✅ Dataset folder postoji: {DATASET_DIR}")

    tar_files = list(Path(DATASET_DIR).rglob('*.tar'))
    print(f"\n   Pronađeno .tar fajlova: {len(tar_files)}")
    for tf in tar_files[:5]:
        print(f"   - {tf}")
else:
    print(f"❌ Dataset folder ne postoji: {DATASET_DIR}")

print("\n" + "="*60)
print("✅ DIJAGNOSTIKA ZAVRŠENA")

from pathlib import Path
import os

print("🔍 PRETRAGA ANOTACIJA")
print("="*60)

# Putanja do slika
images_path = Path(DATASET_DIR) / 'images' / 'train'
print(f"📸 Slike u: {images_path}")
print(f"   Postoji: {images_path.exists()}")

if images_path.exists():
    slike = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
    print(f"   Broj slika: {len(slike)}")

# Pretraži anotacije u mogućim lokacijama
moguce_lokacije_anotacija = [
    str(Path(DATASET_DIR) / 'labels' / 'train'),
    str(Path(DATASET_DIR) / 'labels'),
    str(Path(DATASET_DIR) / 'annotations'),
    str(Path(DATASET_DIR).parent / 'dataset_raspakovan' / 'labels' / 'train'),
    str(Path(DATASET_DIR).parent / 'dataset_raspakovan' / 'labels'),
    str(Path(DATASET_DIR) / 'train' / 'labels'),
]

print("\n🔍 Tražim anotacije...")
pronadjene_anotacije = []

for lokacija in moguce_lokacije_anotacija:
    path = Path(lokacija)
    if path.exists():
        anotacije = list(path.glob('*.txt'))
        if anotacije:
            print(f"\n✅ Anotacije pronađene u: {lokacija}")
            print(f"   Broj anotacija: {len(anotacije)}")
            print(f"   Primer: {anotacije[0].name}")
            pronadjene_anotacije = anotacije
            break

if not pronadjene_anotacije:
    print("\n❌ Nema anotacija u standardnim lokacijama!")
    print("\n🔍 Pretražujem ceo dataset za .txt fajlovima...")

    svi_txt = list(Path(DATASET_DIR).rglob('*.txt'))
    print(f"   Ukupno .txt fajlova: {len(svi_txt)}")

    if svi_txt:
        print("\n   Primeri .txt fajlova:")
        for txt in svi_txt[:10]:
            print(f"   - {txt}")
            # Pokušaj da otvoriš prvi red
            try:
                with open(txt, 'r') as f:
                    first_line = f.readline().strip()
                    print(f"     Prva linija: {first_line[:50]}...")
            except:
                pass

# Proveri da li u folderu dataset postoji labels folder
dataset_path = Path(DATASET_DIR)
if dataset_path.exists():
    print(f"\n📁 Sadržaj {DATASET_DIR}:")
    for item in dataset_path.iterdir():
        if item.is_dir():
            print(f"   📂 {item.name}/")
            # Prikaži podfoldere
            for sub in item.iterdir():
                if sub.is_dir():
                    print(f"      📂 {sub.name}/")
                    # Broj fajlova
                    broj = len(list(sub.glob('*')))
                    print(f"         {broj} fajlova")

"""#PODELA TRENING VAL 80/20"""

import os
import shutil
import random
from pathlib import Path

print("🔧 PODELA DATASET-A 80/20 (FINALNO)")
print("="*60)

# ============================================================
# 1. PUTANJE (tačne na osnovu dijagnostike)
# ============================================================

IMAGES_DIR = Path(DATASET_DIR) / 'images' / 'train'
LABELS_DIR = Path(DATASET_DIR) / 'labels' / 'train'

print(f"📸 Slike: {IMAGES_DIR}")
print(f"📝 Anotacije: {LABELS_DIR}")
print(f"   Slike: {len(list(IMAGES_DIR.glob('*.jpg')))} + {len(list(IMAGES_DIR.glob('*.png')))}")
print(f"   Anotacije: {len(list(LABELS_DIR.glob('*.txt')))}")

# ============================================================
# 2. MAPIRANJE SLIKA SA ANOTACIJAMA
# ============================================================

print("\n📊 Mapiranje slika i anotacija...")

# Pronađi sve slike
sve_slike = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    sve_slike.extend(IMAGES_DIR.glob(ext))

print(f"   Ukupno slika: {len(sve_slike)}")

# Samo slike koje imaju anotaciju
img_ann_map = {}
slike_bez_anotacije = []

for img in sve_slike:
    ann_path = LABELS_DIR / f"{img.stem}.txt"
    if ann_path.exists():
        img_ann_map[img] = ann_path
    else:
        slike_bez_anotacije.append(img)

print(f"   Slike sa anotacijama: {len(img_ann_map)}")
print(f"   Slike bez anotacija: {len(slike_bez_anotacije)}")

if len(img_ann_map) == 0:
    print("❌ Nema anotacija! Prekidam.")
    sys.exit(1)

# ============================================================
# 3. MEŠANJE I PODELA 80/20
# ============================================================

print("\n" + "="*60)
print("🎲 Mešam i delim 80/20...")

slike_lista = list(img_ann_map.keys())
random.seed(42)  # Fiksni seed za reprodukovanje
random.shuffle(slike_lista)

total = len(slike_lista)
train_count = int(total * 0.8)
val_count = total - train_count

train_slike = slike_lista[:train_count]
val_slike = slike_lista[train_count:]

print(f"\n📊 PODELA:")
print(f"   Trening (80%): {len(train_slike)} slika")
print(f"   Validacija (20%): {len(val_slike)} slika")

# ============================================================
# 4. BRISANJE STARIH FOLDERA
# ============================================================

print("\n🗑️ Brišem stare YOLO foldere...")
for folder in ['train', 'val']:
    if Path(folder).exists():
        shutil.rmtree(folder)
    if Path(folder).exists():
        shutil.rmtree(folder)

# ============================================================
# 5. KREIRANJE NOVIH FOLDERA
# ============================================================

print("📁 Kreiram nove YOLO foldere...")
os.makedirs('train/images', exist_ok=True)
os.makedirs('train/labels', exist_ok=True)
os.makedirs('val/images', exist_ok=True)
os.makedirs('val/labels', exist_ok=True)

# ============================================================
# 6. KOPIRANJE FAJLOVA
# ============================================================

print("\n📦 Kopiranje trening fajlova (80%)...")
for img in train_slike:
    shutil.copy2(img, f'train/images/{img.name}')
    shutil.copy2(img_ann_map[img], f'train/labels/{img_ann_map[img].name}')

print("📦 Kopiranje validacionih fajlova (20%)...")
for img in val_slike:
    shutil.copy2(img, f'val/images/{img.name}')
    shutil.copy2(img_ann_map[img], f'val/labels/{img_ann_map[img].name}')

print("✅ Kopiranje završeno!")

# ============================================================
# 7. FINALNA PROVERA
# ============================================================

print("\n" + "="*60)
print("🔍 FINALNA PROVERA:")

train_img_count = len(list(Path('train/images').glob('*')))
train_lbl_count = len(list(Path('train/labels').glob('*.txt')))
val_img_count = len(list(Path('val/images').glob('*')))
val_lbl_count = len(list(Path('val/labels').glob('*.txt')))

print(f"\n   ✅ train/images: {train_img_count}")
print(f"   ✅ train/labels: {train_lbl_count}")
print(f"   ✅ val/images: {val_img_count}")
print(f"   ✅ val/labels: {val_lbl_count}")

if train_img_count == train_lbl_count and train_img_count > 0:
    print("\n✅ Trening set konzistentan!")
else:
    print(f"\n⚠️ Trening: {train_img_count} slika, {train_lbl_count} anotacija")

if val_img_count == val_lbl_count and val_img_count > 0:
    print("✅ Validacioni set konzistentan!")
else:
    print(f"⚠️ Validacija: {val_img_count} slika, {val_lbl_count} anotacija")

# ============================================================
# 8. STATISTIKA PO KLASAMA
# ============================================================

print("\n📊 STATISTIKA PO KLASAMA:")

def count_classes(folder):
    """Broji pojavljivanja klasa u folderu anotacija"""
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

# Trening
train_classes = count_classes('train/labels')
print(f"\n   Trening skup:")
print(f"      - Pospan (klasa 0): {train_classes[0]} anotacija")
print(f"      - Budan (klasa 1): {train_classes[1]} anotacija")
print(f"      - Ukupno: {train_classes[0] + train_classes[1]}")

# Validacija
val_classes = count_classes('val/labels')
print(f"\n   Validacioni skup:")
print(f"      - Pospan (klasa 0): {val_classes[0]} anotacija")
print(f"      - Budan (klasa 1): {val_classes[1]} anotacija")
print(f"      - Ukupno: {val_classes[0] + val_classes[1]}")

# ============================================================
# 9. KREIRANJE data.yaml
# ============================================================

data_yaml = {
    'path': '.',
    'train': 'train/images',
    'val': 'val/images',
    'nc': 2,
    'names': ['pospan', 'budan']
}

import yaml
with open('data.yaml', 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("\n✅ Kreiran data.yaml fajl:")
print(f"   train: {data_yaml['train']}")
print(f"   val: {data_yaml['val']}")
print(f"   broj klasa: {data_yaml['nc']}")
print(f"   klase: {data_yaml['names']}")

# ============================================================
# 10. PRIKAZ PRIMERA
# ============================================================

if train_lbl_count > 0:
    print("\n📄 Primer anotacije iz trening skupa:")
    primer_label = list(Path('train/labels').glob('*.txt'))[0]
    with open(primer_label, 'r') as f:
        content = f.read().strip()
        print(f"   {primer_label.name}: {content}")

        parts = content.split()
        if len(parts) == 5:
            class_id = int(parts[0])
            klasa = 'Pospan' if class_id == 0 else 'Budan'
            print(f"   → Klasa: {klasa} (ID: {class_id})")
            print(f"   → YOLO format: [class] [x_center] [y_center] [width] [height]")
else:
    print("\n⚠️ Nema anotacija za prikaz primera!")

# ============================================================
# 11. SAČUVAJ INFORMACIJU O PODELI
# ============================================================

with open('dataset_split_info.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("DATASET SPLIT INFO\n")
    f.write("="*60 + "\n\n")
    f.write(f"Originalne slike: {len(sve_slike)}\n")
    f.write(f"Slike sa anotacijama: {len(img_ann_map)}\n")
    f.write(f"Slike bez anotacija: {len(slike_bez_anotacije)}\n\n")
    f.write(f"Trening (80%): {train_img_count} slika, {train_lbl_count} anotacija\n")
    f.write(f"  - Pospan: {train_classes[0]}\n")
    f.write(f"  - Budan: {train_classes[1]}\n\n")
    f.write(f"Validacija (20%): {val_img_count} slika, {val_lbl_count} anotacija\n")
    f.write(f"  - Pospan: {val_classes[0]}\n")
    f.write(f"  - Budan: {val_classes[1]}\n")

print("\n📁 Informacija o podeli sačuvana u: dataset_split_info.txt")

# ============================================================
# 12. PROVERA PRVIH 5 SLIKA
# ============================================================

print("\n🔍 Provera prvih 5 slika u trening skupu:")
for i, img in enumerate(list(Path('train/images').glob('*'))[:5]):
    print(f"   {i+1}. {img.name}")

print("\n" + "="*60)
print("🎉 SPREMAN ZA YOLO TRENING!")
print("="*60)

"""#TRENING"""

from ultralytics import YOLO
import torch
from pathlib import Path
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

print("🚀 POKRETANJE TRENINGA SA DETALJNIM PRAĆENJEM SVAKE EPOCHE")
print("="*60)

# ============================================================
# 1. PROVERA GPU
# ============================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Uređaj: {device.upper()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 2. PROVERA PODATAKA
# ============================================================
train_img = len(list(Path('train/images').glob('*')))
train_lbl = len(list(Path('train/labels').glob('*.txt')))
val_img = len(list(Path('val/images').glob('*')))
val_lbl = len(list(Path('val/labels').glob('*.txt')))

print(f"\n📊 Podaci za trening:")
print(f"   Train: {train_img} slika, {train_lbl} anotacija")
print(f"   Val: {val_img} slika, {val_lbl} anotacija")

if train_img == 0 or val_img == 0:
    print("❌ Nema dovoljno podataka! Prvo pokrenite ćeliju za podelu.")
else:
    # ============================================================
    # 3. Učitaj validacione slike za vizuelni prikaz
    # ============================================================
    val_images = list(Path('val/images').glob('*'))
    random.seed(42)
    sample_val_images = random.sample(val_images, min(4, len(val_images)))
    print(f"\n📸 Uzorak validacionih slika za praćenje: {len(sample_val_images)}")

    # ============================================================
    # 4. KREIRANJE data.yaml AKO NE POSTOJI
    # ============================================================
    if not Path('data.yaml').exists():
        import yaml
        data_config = {
            'path': '.',
            'train': './train/images',
            'val': './val/images',
            'nc': 2,
            'names': ['Pospan', 'Budan']
        }
        with open('data.yaml', 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        print("\n✅ data.yaml kreiran")

    # ============================================================
    # 5. KREIRANJE LOGOVA
    # ============================================================
    log_folder = Path('training_logs')
    log_folder.mkdir(exist_ok=True)
    viz_folder = Path('validation_viz')
    viz_folder.mkdir(exist_ok=True)

    validation_log = log_folder / f'validation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    with open(validation_log, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'box_loss', 'cls_loss', 'precision', 'recall', 'mAP50', 'mAP50-95', 'timestamp'])

    # ============================================================
    # 6. FUNKCIJA ZA VIZUELIZACIJU PREDIKCIJA
    # ============================================================
    def visualize_predictions(model, images, epoch, save=True):
        """Prikazuje predikcije modela na uzorku validacionih slika."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, img_path in enumerate(images):
            if idx >= 4:
                break

            try:
                results = model(img_path)
                img = Image.open(img_path)
                axes[idx].imshow(img)
                axes[idx].axis('off')
                axes[idx].set_title(f"Epoch {epoch}", fontsize=10)

                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()

                    colors = {0: 'red', 1: 'green'}
                    class_names = {0: 'Pospan', 1: 'Budan'}

                    for box, conf, cls in zip(boxes, confs, classes):
                        x1, y1, x2, y2 = box
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                                linewidth=2, edgecolor=colors.get(int(cls), 'blue'),
                                                facecolor='none')
                        axes[idx].add_patch(rect)
                        axes[idx].text(x1, y1-5, f"{class_names.get(int(cls), '?')}: {conf:.2f}",
                                     fontsize=8, color='white',
                                     bbox=dict(boxstyle='round', facecolor=colors.get(int(cls), 'blue'), alpha=0.7))
            except Exception as e:
                print(f"   ⚠️ Greška pri vizuelizaciji {img_path.name}: {e}")

        plt.tight_layout()

        if save:
            plt.savefig(viz_folder / f'epoch_{epoch}_predictions.png', dpi=150, bbox_inches='tight')

        plt.show()
        plt.close()

    # ============================================================
    # 7. UČITAVANJE MODELA
    # ============================================================
    model = YOLO('yolov8n.pt')

    # ============================================================
    # 8. TRENING BEZ CALLBACK ARGUMENTA
    # ============================================================
    print("\n🎯 Počinje trening...")
    print("="*60)

    # Pokreni trening (callbacks se ne prosleđuju direktno)
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=64,
        device=device,
        workers=4,
        patience=10,
        save=True,
        save_period=5,
        name='drowsiness_tracking',
        verbose=True
    )

    # ============================================================
    # 9. NAKON TRENINGA - PRIKAZI REZULTATE
    # ============================================================

    results_csv = Path('runs/detect/drowsiness_tracking/results.csv')

    if results_csv.exists():
        df = pd.read_csv(results_csv)

        # Čuvanje u log fajl
        with open(validation_log, 'a', newline='') as f:
            writer = csv.writer(f)
            for idx, row in df.iterrows():
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

        print(f"\n✅ Log validacije sačuvan u: {validation_log}")

        # ============================================================
        # 10. GRAFIKON NAPREDOVANJA
        # ============================================================

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # mAP50
        if 'metrics/mAP50(B)' in df.columns:
            axes[0, 0].plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('mAP50')
            axes[0, 0].set_title('mAP50 tokom treninga')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(y=df['metrics/mAP50(B)'].max(), color='r', linestyle='--', alpha=0.5)
            axes[0, 0].text(0.7, 0.95, f"Max: {df['metrics/mAP50(B)'].max():.4f}", transform=axes[0, 0].transAxes)

        # Loss
        axes[0, 1].plot(df['epoch'], df['val/box_loss'], 'r-', linewidth=2, label='Box Loss')
        axes[0, 1].plot(df['epoch'], df['val/cls_loss'], 'g-', linewidth=2, label='Cls Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Validacioni gubici')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Precision & Recall
        if 'metrics/precision(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], 'b-', linewidth=2, label='Precision')
            axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], 'orange', linewidth=2, label='Recall')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Vrednost')
            axes[1, 0].set_title('Precision i Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # mAP50-95
        if 'metrics/mAP50-95(B)' in df.columns:
            axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], 'purple', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('mAP50-95')
            axes[1, 1].set_title('mAP50-95 tokom treninga')
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Napredak treninga - Sve epohe', fontsize=14)
        plt.tight_layout()
        plt.savefig(log_folder / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()

        # ============================================================
        # 11. STATISTIKA
        # ============================================================
        print("\n" + "="*60)
        print("📊 FINALNI REZULTATI TRENINGA")
        print("="*60)

        if 'metrics/mAP50(B)' in df.columns:
            print(f"\n   ✅ Najbolji mAP50: {df['metrics/mAP50(B)'].max():.4f}")
            print(f"      (epoha: {df['metrics/mAP50(B)'].idxmax() + 1})")
            print(f"   ✅ Finalni mAP50: {df['metrics/mAP50(B)'].iloc[-1]:.4f}")

        if 'metrics/mAP50-95(B)' in df.columns:
            print(f"   ✅ Najbolji mAP50-95: {df['metrics/mAP50-95(B)'].max():.4f}")

        if 'metrics/precision(B)' in df.columns:
            print(f"   ✅ Finalna Precision: {df['metrics/precision(B)'].iloc[-1]:.4f}")
            print(f"   ✅ Finalni Recall: {df['metrics/recall(B)'].iloc[-1]:.4f}")

        # Prikaz poslednjih 10 validacija
        print("\n📋 POSLEDNJIH 10 VALIDACIJA:")
        cols = ['epoch', 'val/box_loss', 'val/cls_loss']
        if 'metrics/mAP50(B)' in df.columns:
            cols.append('metrics/mAP50(B)')
        print(df[cols].tail(10).to_string(index=False))

        # ============================================================
        # 12. VIZUELIZACIJA FINALNIH PREDIKCIJA
        # ============================================================
        print("\n🎯 Prikaz finalnih predikcija na validacionom skupu...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, img_path in enumerate(sample_val_images):
            if idx >= 4:
                break

            try:
                results_final = model(img_path)
                img = Image.open(img_path)
                axes[idx].imshow(img)
                axes[idx].axis('off')
                axes[idx].set_title(f"Finalna predikcija (epoha {len(df)})", fontsize=10)

                if len(results_final[0].boxes) > 0:
                    boxes = results_final[0].boxes.xyxy.cpu().numpy()
                    confs = results_final[0].boxes.conf.cpu().numpy()
                    classes = results_final[0].boxes.cls.cpu().numpy()

                    colors = {0: 'red', 1: 'green'}
                    class_names = {0: 'Pospan', 1: 'Budan'}

                    for box, conf, cls in zip(boxes, confs, classes):
                        x1, y1, x2, y2 = box
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                                linewidth=2, edgecolor=colors.get(int(cls), 'blue'),
                                                facecolor='none')
                        axes[idx].add_patch(rect)
                        axes[idx].text(x1, y1-5, f"{class_names.get(int(cls), '?')}: {conf:.2f}",
                                     fontsize=8, color='white',
                                     bbox=dict(boxstyle='round', facecolor=colors.get(int(cls), 'blue'), alpha=0.7))
            except Exception as e:
                print(f"   ⚠️ Greška: {e}")

        plt.tight_layout()
        plt.savefig(log_folder / 'final_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()

        # ============================================================
        # 13. EXTRA: Prikaz svake epohe kroz CSV (YOLO već čuva)
        # ============================================================
        print(f"\n📁 Detalji treninga sačuvani u:")
        print(f"   - {results_csv}")
        print(f"   - {validation_log}")

    else:
        print("⚠️ results.csv nije kreiran, proverite da li je trening uspešno završen.")
        print("\n📁 Mogući folderi sa rezultatima:")
        runs_dir = Path('runs/detect')
        if runs_dir.exists():
            for d in runs_dir.iterdir():
                if d.is_dir():
                    print(f"   - {d}")

    print("\n" + "="*60)
    print("✅ TRENING ZAVRŠEN!")
    print(f"📁 Logovi sačuvani u: {log_folder}")
    print(f"📁 Vizualizacije sačuvane u: {viz_folder}")
    print("="*60)

"""#PRIKAZ PO EPOHAMA"""

from ultralytics import YOLO
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import numpy as np

print("🚀 VIZUELNI PREGLED UČENJA MODELA KROZ EPOCHE")
print("="*60)

# ============================================================
# 1. PRONAĐI SVE CHECKPOINTOVE (modele za svaku epohu)
# ============================================================

def pronadji_sve_modele():
    """Pronalazi sve dostupne modele (checkpoint-ove)"""
    models = []

    # 1. Proveri runs/detect folder
    runs_dir = Path('runs/detect')
    if runs_dir.exists():
        for subdir in runs_dir.iterdir():
            if subdir.is_dir():
                # Proveri weights folder
                weights_dir = subdir / 'weights'
                if weights_dir.exists():
                    for pt in weights_dir.glob('*.pt'):
                        # Izdvoj broj epohe iz naziva
                        epoch_num = None
                        if 'epoch' in pt.stem:
                            try:
                                epoch_num = int(pt.stem.replace('epoch', ''))
                            except:
                                epoch_num = 0
                        models.append((epoch_num, pt, subdir.name))

                # Proveri direktno u folderu
                for pt in subdir.glob('epoch*.pt'):
                    try:
                        epoch_num = int(pt.stem.replace('epoch', ''))
                        models.append((epoch_num, pt, subdir.name))
                    except:
                        pass

    # 2. Proveri trenutni folder
    for pt in Path('.').glob('*.pt'):
        models.append((None, pt, 'current'))

    # Sortiraj po epoch broju
    models.sort(key=lambda x: x[0] if x[0] is not None else 999)

    return models

models = pronadji_sve_modele()

if not models:
    print("❌ Nema pronađenih modela!")
    print("\nPokreni trening prvo, ili proveri putanje.")
    sys.exit(1)

print(f"📁 Pronađeno modela: {len(models)}")
for epoch, path, folder in models:
    epoch_str = f"epoha {epoch}" if epoch is not None else "nepoznata epoha"
    print(f"   - {path.name} ({epoch_str}) - folder: {folder}")

# ============================================================
# 2. PRONAĐI VALIDACIONE SLIKE (sa ground truth anotacijama)
# ============================================================

# Potraži validacione slike i njihove anotacije
val_images = []
val_annotations = {}

# Mape mogućih lokacija
possible_paths = [
    ('val/images', 'val/labels'),
    ('dataset_raspakovan/images/val', 'dataset_raspakovan/labels/val'),
    ('images/val', 'labels/val'),
    ('valid/images', 'valid/labels'),
]

for img_path, lbl_path in possible_paths:
    img_dir = Path(img_path)
    lbl_dir = Path(lbl_path)

    if img_dir.exists() and lbl_dir.exists():
        print(f"📸 Pronađene slike: {img_dir}")
        print(f"📝 Pronađene anotacije: {lbl_dir}")

        for img_file in img_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Potraži odgovarajuću anotaciju
                ann_file = lbl_dir / f"{img_file.stem}.txt"
                if ann_file.exists():
                    val_images.append(img_file)
                    val_annotations[img_file] = ann_file
        break

if not val_images:
    print("⚠️ Nema validacionih slika sa anotacijama, koristim samo slike...")
    # Pokušaj samo slike
    for img_dir in [Path('val/images'), Path('images/val')]:
        if img_dir.exists():
            val_images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            break

print(f"📸 Pronađeno slika za validaciju: {len(val_images)}")

if len(val_images) == 0:
    print("❌ Nema slika za prikaz!")
    sys.exit(1)

# Odaberi 4 slike za prikaz
random.seed(42)
sample_images = random.sample(val_images, min(4, len(val_images)))
print(f"🖼️ Odabrano slika za prikaz: {len(sample_images)}")

# ============================================================
# 3. FUNKCIJA ZA CRTANJE GROUND TRUTH BOX-OVA
# ============================================================

def nacrtaj_ground_truth(ax, img_path, annotations_dict):
    """Crta ground truth bounding box-ove na slici"""

    # Potraži anotaciju
    ann_path = annotations_dict.get(img_path)
    if not ann_path or not ann_path.exists():
        return

    try:
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_center, y_center, width, height = map(float, parts)

                    # YOLO format -> pixel koordinate
                    img = Image.open(img_path)
                    img_w, img_h = img.size

                    x1 = (x_center - width/2) * img_w
                    y1 = (y_center - height/2) * img_h
                    x2 = (x_center + width/2) * img_w
                    y2 = (y_center + height/2) * img_h

                    boja = 'darkred' if cls == 0 else 'darkgreen'
                    naziv = 'POSPAN' if cls == 0 else 'BUDAN'

                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                            linewidth=2, edgecolor=boja,
                                            facecolor='none', linestyle='--')
                    ax.add_patch(rect)
                    ax.text(x1, y1-3, f"GT: {naziv}",
                           fontsize=8, color='black', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor=boja, alpha=0.5))
    except Exception as e:
        pass

# ============================================================
# 4. FUNKCIJA ZA CRTANJE PREDIKCIJA
# ============================================================

def nacrtaj_predikcije(ax, results):
    """Crta predikcije modela na slici"""

    if len(results[0].boxes) == 0:
        ax.text(0.5, 0.5, "NEMA DETEKCIJE",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='gray', fontweight='bold')
        return

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = box
        boja = 'red' if cls == 0 else 'lime'
        naziv = 'POSPAN' if cls == 0 else 'BUDAN'

        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=3, edgecolor=boja,
                                facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"{naziv}: {conf:.3f}",
               fontsize=9, color='white', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=boja, alpha=0.8))

# ============================================================
# 5. FUNKCIJA ZA UPOREDNI PRIKAZ (GT vs PREDIKCIJA)
# ============================================================

def prikazi_uporedno(model, images, annotations, epoch_num, save_path=None):
    """Prikazuje uporedno: Ground Truth (levo) i Predikciju (desno)"""

    fig, axes = plt.subplots(len(images), 2, figsize=(14, 4 * len(images)))

    if len(images) == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(images):
        # Leva kolona: Ground Truth
        ax_gt = axes[idx, 0]
        img = Image.open(img_path)
        ax_gt.imshow(img)
        ax_gt.axis('off')
        ax_gt.set_title(f"STVARNO STANJE (Ground Truth)", fontsize=10, fontweight='bold', color='darkblue')

        # Nacrtaj ground truth box-ove
        nacrtaj_ground_truth(ax_gt, img_path, annotations)

        # Desna kolona: Predikcija modela
        ax_pred = axes[idx, 1]
        ax_pred.imshow(img)
        ax_pred.axis('off')
        ax_pred.set_title(f"PREDIKCIJA MODELA - Epoha {epoch_num}", fontsize=10, fontweight='bold', color='darkgreen')

        # Nacrtaj predikcije
        results = model(img_path)
        nacrtaj_predikcije(ax_pred, results)

        # Dodaj ime slike
        ax_gt.text(0.5, -0.08, img_path.name, transform=ax_gt.transAxes,
                  ha='center', fontsize=8, color='gray')
        ax_pred.text(0.5, -0.08, img_path.name, transform=ax_pred.transAxes,
                    ha='center', fontsize=8, color='gray')

    plt.suptitle(f'📊 Epoha {epoch_num} - Poređenje: Ground Truth vs Predikcija',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    return fig

# ============================================================
# 6. GLAVNI PROGRAM - PRIKAZ ZA SVAKU EPOHU
# ============================================================

output_folder = Path('epoch_comparison')
output_folder.mkdir(exist_ok=True)

print("\n" + "="*60)
print("🎯 PRIKAZUJEM UČENJE KROZ EPOCHE")
print("="*60)

# Prikaži za svaki pronađeni model
for epoch_num, model_path, folder_name in models:
    if epoch_num is None:
        continue

    print(f"\n📊 Epoha {epoch_num} - Učitavam model: {model_path.name}")

    # Učitaj model
    model = YOLO(model_path)

    # Prikaži uporedno
    save_path = output_folder / f'epoha_{epoch_num}_comparison.png'
    prikazi_uporedno(model, sample_images, val_annotations, epoch_num, save_path)

    print(f"   ✅ Sačuvano: {save_path}")

# ============================================================
# 7. KREIRAJ VIDEO/ANIMACIJU NAPREDOVANJA
# ============================================================

print("\n" + "="*60)
print("🎬 KREIRAM ANIMACIJU NAPREDOVANJA")
print("="*60)

try:
    from PIL import Image as PILImage
    import glob

    # Sortiraj slike po epochama
    image_files = sorted(glob.glob(str(output_folder / 'epoha_*_comparison.png')),
                        key=lambda x: int(x.split('epoha_')[1].split('_')[0]))

    if len(image_files) > 1:
        images = []
        for img_file in image_files:
            img = PILImage.open(img_file)
            images.append(img)

        gif_path = output_folder / 'learning_progress.gif'
        images[0].save(gif_path, save_all=True, append_images=images[1:],
                      duration=800, loop=0)
        print(f"✅ GIF animacija sačuvana: {gif_path}")
        print(f"   (prikazuje kako model uči kroz epohe)")
    else:
        print("⚠️ Nema dovoljno slika za animaciju (potrebno 2+ epohe)")

except Exception as e:
    print(f"⚠️ Greška pri kreiranju animacije: {e}")

# ============================================================
# 8. FINALNA STATISTIKA
# ============================================================

print("\n" + "="*60)
print("📊 ZAVRŠENO!")
print("="*60)

print(f"""
✅ Vizuelni prikaz učenja modela kroz epohe:

📁 Rezultati sačuvani u: {output_folder}
   - epoha_X_comparison.png - uporedni prikaz za svaku epohu
   - learning_progress.gif - animacija napredovanja

🖼️ Na prikazu:
   - Leva strana: STVARNO STANJE (Ground Truth)
     (tamno crvena/zuta - gde su ljudi zapravo)
   - Desna strana: PREDIKCIJA MODELA
     (svetlo crvena/zelena - šta model misli)

🎯 Ono što treba da gledaš:
   - Da li se bounding box-ovi približavaju pravima?
   - Da li model počinje da detektuje prave osobe?
   - Da li confidenc skor (pouzdanost) raste?
   - Da li model "halucinira" (detektuje tamo gde nema ljudi)?
""")

print("🎉 GOTOVO! Pregledaj slike i vidi kako model napreduje!")

"""#MERENJE VREMENA OBRADE SLIKE"""

from ultralytics import YOLO
import torch
from pathlib import Path
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

print("🚀 MERENJE INFERENCE TIME - YOLO MODEL")
print("="*60)

# ============================================================
# 1. PRONAĐI MODEL
# ============================================================

def pronadji_najbolji_model():
    """Pronalazi najbolji dostupni model"""

    # Prvo proveri runs/detect
    runs_dir = Path('runs/detect')
    if runs_dir.exists():
        for subdir in runs_dir.iterdir():
            if subdir.is_dir():
                # Proveri best.pt
                best = subdir / 'weights' / 'best.pt'
                if best.exists():
                    return best
                # Proveri last.pt
                last = subdir / 'weights' / 'last.pt'
                if last.exists():
                    return last

    # Proveri trenutni folder
    for pt in Path('.').glob('*.pt'):
        return pt

    return None

model_path = pronadji_najbolji_model()

if model_path is None:
    print("❌ Nema pronađenog modela!")
    sys.exit(1)

print(f"📁 Korišćen model: {model_path}")
print(f"   Veličina: {model_path.stat().st_size / (1024 * 1024):.1f} MB")

# Učitaj model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path)
print(f"🖥️ Uređaj: {device.upper()}")

# ============================================================
# 2. PRONAĐI SLIKE ZA TESTIRANJE
# ============================================================

# Potraži slike u validacionom skupu
test_images = []

possible_paths = [
    Path('val/images'),
    Path('dataset_raspakovan/images/val'),
    Path('images/val'),
    Path('test/images'),
]

for path in possible_paths:
    if path.exists():
        test_images = list(path.glob('*.jpg')) + list(path.glob('*.png')) + list(path.glob('*.jpeg'))
        if test_images:
            print(f"📸 Slike za testiranje: {len(test_images)} iz {path}")
            break

if not test_images:
    # Probaj bilo gde
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        test_images.extend(Path('.').glob(f'**/{ext}'))
    test_images = list(set(test_images))[:100]  # Maks 100 slika
    print(f"📸 Pronađeno slika: {len(test_images)}")

if not test_images:
    print("❌ Nema slika za testiranje!")
    sys.exit(1)

# Odaberi 100 slika za testiranje (ili sve ako ih je manje)
num_tests = min(100, len(test_images))
test_images = test_images[:num_tests]
print(f"🔬 Testiranje na {num_tests} slika")

# ============================================================
# 3. MERENJE INFERENCE TIME
# ============================================================

print("\n" + "="*60)
print("⏱️ MERENJE VREMENA...")
print("="*60)

# Zagrevanje (warm-up) - prve predikcije su često sporije
print("🔥 Zagrevanje modela...")
for _ in range(5):
    _ = model(test_images[0])

times = []
batch_sizes = [1, 4, 16, 32, 64]

for batch_size in batch_sizes:
    print(f"\n📊 Batch size: {batch_size}")
    batch_times = []

    # Pokreni testiranje
    for i in tqdm(range(0, len(test_images), batch_size), desc=f"Batch {batch_size}"):
        batch = test_images[i:i+batch_size]

        # Počni merenje
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.perf_counter()

        # Predikcija
        results = model(batch)

        # Završi merenje
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.perf_counter()

        # Vreme po slici u batch-u
        time_per_image = (end_time - start_time) / len(batch)
        batch_times.append(time_per_image)

    avg_time = np.mean(batch_times) * 1000  # konvertuj u milisekunde
    std_time = np.std(batch_times) * 1000
    times.append({
        'batch_size': batch_size,
        'avg_ms': avg_time,
        'std_ms': std_time,
        'fps': 1000 / avg_time
    })

    print(f"   ✅ Prosečno vreme: {avg_time:.2f} ms po slici")
    print(f"   📊 FPS: {1000/avg_time:.1f} slika/sekundi")
    print(f"   📈 Std dev: {std_time:.2f} ms")

# ============================================================
# 4. DETALJNO MERENJE (pojedinačne slike)
# ============================================================

print("\n" + "="*60)
print("🔬 DETALJNO MERENJE (batch size = 1)")
print("="*60)

single_times = []
confidences = []
detections = []

for img_path in tqdm(test_images[:50], desc="Merenje pojedinačno"):
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()

    results = model(img_path)

    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.perf_counter()

    inference_ms = (end - start) * 1000
    single_times.append(inference_ms)

    # Broj detekcija i confidenc
    if len(results[0].boxes) > 0:
        detections.append(len(results[0].boxes))
        confidences.append(results[0].boxes.conf.cpu().numpy().mean())
    else:
        detections.append(0)
        confidences.append(0)

# ============================================================
# 5. STATISTIKA
# ============================================================

print("\n" + "="*60)
print("📊 STATISTIKA INFERENCE TIME")
print("="*60)

print(f"\n📈 Batch size performance:")
df_times = pd.DataFrame(times)
print(df_times.to_string(index=False))

print(f"\n📈 Pojedinačne predikcije (batch size=1):")
print(f"   ✅ Prosečno vreme:  {np.mean(single_times):.2f} ms")
print(f"   ✅ Medijana:        {np.median(single_times):.2f} ms")
print(f"   ✅ Minimum:         {np.min(single_times):.2f} ms")
print(f"   ✅ Maximum:         {np.max(single_times):.2f} ms")
print(f"   ✅ Std dev:         {np.std(single_times):.2f} ms")
print(f"   📊 FPS:             {1000/np.mean(single_times):.1f} slika/sek")

print(f"\n📈 Detekcije po slici:")
print(f"   ✅ Prosečno detekcija: {np.mean(detections):.2f}")
print(f"   ✅ Prosečna confidenc: {np.mean(confidences):.3f}")

# ============================================================
# 6. GRAFIKONI
# ============================================================

print("\n" + "="*60)
print("📊 KREIRANJE GRAFIKONA...")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Grafikon 1: Vreme po batch size
ax1 = axes[0, 0]
ax1.bar(df_times['batch_size'], df_times['avg_ms'], color='skyblue', edgecolor='navy')
ax1.errorbar(df_times['batch_size'], df_times['avg_ms'], yerr=df_times['std_ms'],
             fmt='none', capsize=5, color='red')
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Vreme (ms)')
ax1.set_title('Inference Time po Batch Size-u')
ax1.grid(True, alpha=0.3)

# Grafikon 2: FPS po batch size
ax2 = axes[0, 1]
ax2.plot(df_times['batch_size'], df_times['fps'], 'b-o', linewidth=2, markersize=8)
ax2.set_xlabel('Batch Size')
ax2.set_ylabel('FPS (slike/sekund)')
ax2.set_title('Performanse (FPS) po Batch Size-u')
ax2.grid(True, alpha=0.3)

# Grafikon 3: Histogram vremena
ax3 = axes[1, 0]
ax3.hist(single_times, bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
ax3.axvline(np.mean(single_times), color='red', linestyle='--', label=f'Mean: {np.mean(single_times):.2f} ms')
ax3.axvline(np.median(single_times), color='blue', linestyle='--', label=f'Median: {np.median(single_times):.2f} ms')
ax3.set_xlabel('Vreme (ms)')
ax3.set_ylabel('Broj slika')
ax3.set_title('Distribucija Inference Time (batch size=1)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Grafikon 4: Vreme kroz slike
ax4 = axes[1, 1]
ax4.plot(single_times, 'b-', alpha=0.5, linewidth=1)
ax4.fill_between(range(len(single_times)), single_times, alpha=0.3)
ax4.axhline(np.mean(single_times), color='red', linestyle='--', label=f'Mean: {np.mean(single_times):.2f} ms')
ax4.set_xlabel('Slika redom')
ax4.set_ylabel('Vreme (ms)')
ax4.set_title('Inference Time po slici')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle(f'YOLO Inference Time Analysis - {model_path.name}', fontsize=14, fontweight='bold')
plt.tight_layout()

# Sačuvaj grafikon
output_folder = Path('inference_results')
output_folder.mkdir(exist_ok=True)
plt.savefig(output_folder / 'inference_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 7. SAČUVAJ REZULTATE
# ============================================================

# Sačuvaj CSV sa rezultatima
results_df = pd.DataFrame({
    'image': [str(p) for p in test_images[:len(single_times)]],
    'inference_ms': single_times,
    'detections': detections[:len(single_times)],
    'confidence': confidences[:len(single_times)]
})
results_df.to_csv(output_folder / 'inference_times.csv', index=False)

# Sačuvaj statistiku
stats = {
    'model': str(model_path),
    'device': device,
    'num_images': len(single_times),
    'mean_inference_ms': np.mean(single_times),
    'median_inference_ms': np.median(single_times),
    'std_inference_ms': np.std(single_times),
    'min_inference_ms': np.min(single_times),
    'max_inference_ms': np.max(single_times),
    'fps': 1000 / np.mean(single_times),
    'best_batch_size': df_times.loc[df_times['fps'].idxmax(), 'batch_size'],
    'best_batch_fps': df_times['fps'].max()
}

stats_df = pd.DataFrame([stats])
stats_df.to_csv(output_folder / 'inference_stats.csv', index=False)

# ============================================================
# 8. FINALNI PRIKAZ
# ============================================================

print("\n" + "="*60)
print("📊 FINALNI REZULTATI")
print("="*60)

print(f"""
✅ INFERENCE TIME ANALIZA - ZAKLJUČAK
{'='*50}

📁 Model: {model_path.name}
🖥️ Uređaj: {device.upper()}
📸 Testirano slika: {len(single_times)}

⏱️ VREME PO SLICI (batch size=1):
   • Prosečno: {np.mean(single_times):.2f} ms
   • Medijana: {np.median(single_times):.2f} ms
   • Opseg: {np.min(single_times):.2f} - {np.max(single_times):.2f} ms

🚀 PERFORMANSE:
   • FPS (batch=1): {1000/np.mean(single_times):.1f} slika/sekund
   • Najbolji batch size: {stats['best_batch_size']} (FPS: {stats['best_batch_fps']:.1f})

📁 Rezultati sačuvani u: {output_folder}
   - inference_analysis.png (grafikoni)
   - inference_times.csv (detalji po slici)
   - inference_stats.csv (statistika)
""")

print("\n🎉 MERENJE ZAVRŠENO!")

"""#TEST SKUP"""

# ============================================================
# TESTIRANJE MODELA NA NOVOM TEST SKUPU
# ============================================================

print("🚀 TESTIRANJE MODELA NA NOVOM TEST SKUPU")
print("="*60)

import zipfile
import shutil
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

# ============================================================
# 1. SKINI ZIP SA DRIVE-a
# ============================================================

TEST_ZIP = TEST_ZIP_PATH
EXTRACT_PATH = TEST_EXTRACT_DIR

if not TEST_ZIP or not os.path.exists(TEST_ZIP):
    print(f"ℹ️ Test ZIP nije podešen ili ne postoji: {TEST_ZIP}")
    print("   Podesi TEST_ZIP_PATH na vrhu fajla pa pokreni ponovo.")
    print("   Preskacem testiranje...")
    # Preskoci test sekciju
    should_run_test = False
else:
    should_run_test = True

if should_run_test:

    print(f"✅ Pronađen ZIP: {TEST_ZIP}")

    if os.path.exists(EXTRACT_PATH):
        shutil.rmtree(EXTRACT_PATH)

    os.makedirs(EXTRACT_PATH, exist_ok=True)

    with zipfile.ZipFile(TEST_ZIP, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

    print(f"✅ ZIP raspakovan u: {EXTRACT_PATH}")

    # ============================================================
    # 2. PRONAĐI SVE SLIKE U TEST SKUPU
    # ============================================================

    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        test_images.extend(Path(EXTRACT_PATH).rglob(ext))

    print(f"📸 Pronađeno test slika: {len(test_images)}")

    if len(test_images) == 0:
        print("❌ Nema slika u test skupu!")
        sys.exit(1)

    # ============================================================
    # 3. UČITAJ TRENIRANI MODEL
    # ============================================================

    model_path = None
    runs_dir = Path('runs/detect')
    if runs_dir.exists():
        for subdir in runs_dir.iterdir():
            if subdir.is_dir():
                best_model = subdir / 'weights' / 'best.pt'
                if best_model.exists():
                    model_path = best_model
                    break

    if model_path is None:
        pt_files = list(Path('.').rglob('*.pt'))
        if pt_files:
            model_path = pt_files[0]

    if model_path is None:
        print("❌ Nema istreniranog modela! Prvo pokreni trening.")
        sys.exit(1)

    print(f"📁 Učitavam model: {model_path}")
    model = YOLO(model_path)

    # ============================================================
    # 4. TESTIRANJE I VIZUELIZACIJA
    # ============================================================

    print("\n🎯 Pokrećem testiranje...")
    print("="*60)

    num_to_show = min(6, len(test_images))
    sample_images = random.sample(test_images, num_to_show)

    rows = (num_to_show + 1) // 2
    cols = 2 if num_to_show > 1 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if num_to_show == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    print(f"\n📊 Prikazujem {num_to_show} nasumičnih test slika:\n")

    for idx, img_path in enumerate(sample_images):
        results = model(img_path)
        img = Image.open(img_path)

        ax = axes[idx]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{img_path.name}", fontsize=10)

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = box
                boja = 'red' if cls == 0 else 'lime'
                naziv = 'POSPAN' if cls == 0 else 'BUDAN'

                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                        linewidth=2, edgecolor=boja,
                                        facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-5, f"{naziv}: {conf:.2f}",
                       fontsize=9, color='white', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=boja, alpha=0.7))
        else:
            ax.text(0.5, 0.5, "NEMA DETEKCIJE",
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, color='red', fontweight='bold')

    plt.suptitle(f'TESTIRANJE MODELA - {len(test_images)} slika', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ============================================================
    # 5. STATISTIKA TESTIRANJA
    # ============================================================

    print("\n" + "="*60)
    print("📊 STATISTIKA TESTIRANJA")
    print("="*60)

    total_detections = 0
    total_conf = 0
    images_with_detections = 0

    for img_path in test_images[:50]:
        results = model(img_path)
        if len(results[0].boxes) > 0:
            images_with_detections += 1
            total_detections += len(results[0].boxes)
            total_conf += results[0].boxes.conf.cpu().numpy().mean()

    if images_with_detections > 0:
        avg_conf = total_conf / images_with_detections
        avg_detections = total_detections / images_with_detections
    else:
        avg_conf = 0
        avg_detections = 0

    print(f"\n   📸 Ukupno test slika: {len(test_images)}")
    print(f"   ✅ Slike sa detekcijama: {images_with_detections}/{min(50, len(test_images))}")
    print(f"   📦 Prosečno detekcija po slici: {avg_detections:.2f}")
    print(f"   🎯 Prosečna confidence: {avg_conf:.3f}")

    # ============================================================
    # 6. SAČUVAJ REZULTATE
    # ============================================================

    test_results_folder = Path('test_results')
    test_results_folder.mkdir(exist_ok=True)

    print("\n💾 Čuvanje svih test predikcija...")
    for i, img_path in enumerate(test_images):
        results = model(img_path)
        save_path = test_results_folder / f"pred_{img_path.name}"
        results[0].save(filename=str(save_path))

        if (i + 1) % 50 == 0:
            print(f"   Sačuvano {i+1}/{len(test_images)} slika...")

    print(f"\n✅ Sve test predikcije sačuvane u: {test_results_folder}")

    # ============================================================
    # 7. DODATNO: Sačuvaj i prikaži CSV sa rezultatima
    # ============================================================

    import csv
    csv_path = test_results_folder / 'test_results.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'detections', 'avg_confidence'])

        for img_path in test_images:
            results = model(img_path)
            if len(results[0].boxes) > 0:
                confs = results[0].boxes.conf.cpu().numpy()
                writer.writerow([img_path.name, len(results[0].boxes), confs.mean()])
            else:
                writer.writerow([img_path.name, 0, 0])

    print(f"✅ CSV sa rezultatima: {csv_path}")

    # ============================================================
    # 8. ARHIVIRAJ REZULTATE (opciono)
    # ============================================================

    print("\n📁 Želiš da arhiviraš rezultate testiranja?")
    download_choice = input("   Unesi 'da' za arhiviranje, bilo šta drugo za preskakanje: ")

    if download_choice.lower() == 'da':
        shutil.make_archive('test_results', 'zip', test_results_folder)
        print(f"   ✅ Arhiva kreirana: test_results.zip")

    print("\n" + "="*60)
    print("🎉 TESTIRANJE ZAVRŠENO!")
    print(f"📁 Rezultati u folderu: {test_results_folder}")
    print("="*60)