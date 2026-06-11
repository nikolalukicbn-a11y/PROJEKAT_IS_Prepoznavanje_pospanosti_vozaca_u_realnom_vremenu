#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKRIPTA 5: ČUVANJE NAJBOLJEG MODELA
- Pronalazi najbolji model (best.pt)
- Kopira ga u export folder
- Eksportuje u ONNX, TorchScript i OpenVINO format
- Generiše info fajl sa metrikama
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

import torch

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# KONFIGURACIJA
# ============================================================
EXPORT_DIR = Path('exported_models')
EXPORT_DIR.mkdir(exist_ok=True)


def find_best_model():
    candidates = []

    runs_dir = Path('runs/detect')
    if runs_dir.exists():
        for subdir in runs_dir.iterdir():
            if not subdir.is_dir():
                continue
            best = subdir / 'weights' / 'best.pt'
            if best.exists():
                candidates.append((subdir.name, best))

    candidates.sort(reverse=True)
    return candidates


def export_onnx(model, model_path, run_name):
    print("\n📦 Export u ONNX format...")
    try:
        onnx_path = EXPORT_DIR / f'{run_name}.onnx'
        model.export(format='onnx', imgsz=640, half=False)
        src = Path(f'runs/detect/{run_name}/weights/best.onnx')
        if src.exists():
            shutil.copy2(src, onnx_path)
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {onnx_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"   ⚠️ ONNX export nije uspeo: {e}")


def export_torchscript(model, model_path, run_name):
    print("\n📦 Export u TorchScript format...")
    try:
        ts_path = EXPORT_DIR / f'{run_name}.torchscript'
        model.export(format='torchscript', imgsz=640)
        src = Path(f'runs/detect/{run_name}/weights/best.torchscript')
        if src.exists():
            shutil.copy2(src, ts_path)
            size_mb = ts_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {ts_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"   ⚠️ TorchScript export nije uspeo: {e}")


def copy_best_model(candidates):
    print("=" * 60)
    print("1. KOPIRANJE NAJBOLJEG MODELA")
    print("=" * 60)

    if not candidates:
        print("❌ Nema pronađenih modela!")
        return None

    for run_name, model_path in candidates:
        dest = EXPORT_DIR / f'{run_name}_best.pt'
        shutil.copy2(model_path, dest)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ {run_name}/weights/best.pt -> {dest} ({size_mb:.1f} MB)")

        src_pt = Path('.') / f'{run_name}_best.pt'
        if not src_pt.exists():
            shutil.copy2(model_path, src_pt)
            print(f"✅ Kopija i u root-u: {src_pt}")

        return run_name, model_path

    return None


def generate_model_info(run_name):
    print("\n" + "=" * 60)
    print("2. INFORMACIJE O MODELU")
    print("=" * 60)

    info = []
    info.append("=" * 60)
    info.append("MODEL EXPORT INFO")
    info.append("=" * 60)
    info.append(f"Datum:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info.append(f"Run:      {run_name}")
    info.append("")

    args_path = Path(f'runs/detect/{run_name}/args.yaml')
    if args_path.exists():
        import yaml
        with open(args_path, 'r') as f:
            args = yaml.safe_load(f)
        info.append("ARGUMENTI TRENINGA:")
        for k, v in args.items():
            info.append(f"  {k}: {v}")
        info.append("")

    results_csv = Path(f'runs/detect/{run_name}/results.csv')
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        info.append("FINALNE METRIKE:")
        if 'metrics/mAP50(B)' in df.columns:
            best_idx = df['metrics/mAP50(B)'].idxmax()
            info.append(f"  Najbolji mAP50:      {df['metrics/mAP50(B)'].max():.4f} (epoha {best_idx + 1})")
            info.append(f"  Finalni mAP50:       {df['metrics/mAP50(B)'].iloc[-1]:.4f}")
        if 'metrics/mAP50-95(B)' in df.columns:
            info.append(f"  Najbolji mAP50-95:   {df['metrics/mAP50-95(B)'].max():.4f}")
        if 'metrics/precision(B)' in df.columns:
            info.append(f"  Precision:           {df['metrics/precision(B)'].iloc[-1]:.4f}")
            info.append(f"  Recall:              {df['metrics/recall(B)'].iloc[-1]:.4f}")
        info.append(f"  Ukupno epoha:        {len(df)}")

    info_path = EXPORT_DIR / f'{run_name}_info.txt'
    with open(info_path, 'w') as f:
        f.write('\n'.join(info))

    print('\n'.join(info))
    print(f"\n✅ Info sačuvan: {info_path}")


def main():
    print("=" * 60)
    print("SKRIPTA 5: ČUVANJE NAJBOLJEG MODELA")
    print("=" * 60)

    candidates = find_best_model()
    if not candidates:
        print("❌ Nema istreniranih modela! Pokreni prvo 02_trening.py")
        sys.exit(1)

    result = copy_best_model(candidates)
    if result is None:
        sys.exit(1)

    run_name, model_path = result
    generate_model_info(run_name)

    from ultralytics import YOLO
    model = YOLO(model_path)

    export_onnx(model, model_path, run_name)
    export_torchscript(model, model_path, run_name)

    print("\n" + "=" * 60)
    print(f"✅ SKRIPTA 5 ZAVRŠENA — Modeli sačuvani u: {EXPORT_DIR}")
    print("=" * 60)
    print(f"\n📂 Sadržaj {EXPORT_DIR}:")
    for f in sorted(EXPORT_DIR.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
