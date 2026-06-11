#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKRIPTA 6: DETEKCIJA POSPANOSTI U REALNOM VREMENU (WEBCAM)
- Koristi istrenirani YOLOv8s model
- Prikazuje webcam feed sa bounding box-ovima
- Target: 30 FPS
- Pritiskom 'q' izlazi, 's' snima screenshot
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# KONFIGURACIJA
# ============================================================
CONF_THRESHOLD = 0.55
AGNOSTIC_NMS = True
IOU_THRESHOLD = 0.7
TARGET_FPS = 50
CAMERA_ID = 0

CLASS_NAMES = {0: 'POSPAN', 1: 'BUDAN'}
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}  # BGR: crvena, zelena

OUTPUT_DIR = Path('realtime_results')
OUTPUT_DIR.mkdir(exist_ok=True)


def find_model():
    runs_dir = Path('runs/detect')
    if runs_dir.exists():
        for subdir in sorted(runs_dir.iterdir(), reverse=True):
            best = subdir / 'weights' / 'best.pt'
            if best.exists():
                return best

    for pt in Path('.').glob('best.pt'):
        return pt

    return None


def draw_overlay(frame, results, fps, frame_count):
    h, w = frame.shape[:2]

    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls)
            label = f"{CLASS_NAMES.get(cls_id, '?')} {conf:.2f}"
            color = CLASS_COLORS.get(cls_id, (255, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        detekcije = f"Detekcija: {len(boxes)}"
    else:
        detekcije = "Detekcija: NEMA"

    overlay = np.zeros((120, w, 3), dtype=np.uint8)
    overlay[:, :] = (30, 30, 30)

    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(overlay, detekcije, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Frame: {frame_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    status = f"conf={CONF_THRESHOLD} | agnostic_nms={AGNOSTIC_NMS} | Q=izlaz S=snimak"
    cv2.putText(overlay, status, (w - 500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return np.vstack([overlay, frame])


def main():
    print("=" * 60)
    print("SKRIPTA 6: DETEKCIJA POSPANOSTI U REALNOM VREMENU")
    print("=" * 60)

    model_path = find_model()
    if model_path is None:
        print("❌ Nema modela! Pokreni prvo 02_trening.py ili importuj best.pt")
        sys.exit(1)

    print(f"📁 Model: {model_path}")
    print(f"🎯 Target: {TARGET_FPS} FPS")

    model = YOLO(model_path)

    print("\n📷 Otvaram kameru...")
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("❌ Ne mogu da otvorim kameru!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 50)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"   Kamera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {actual_fps:.0f} FPS")

    print("\n⚙️ Kontrole:")
    print("   Q = izlaz")
    print("   S = snimi screenshot")
    print("=" * 60)

    frame_count = 0
    fps_history = []
    fps_start = time.perf_counter()
    fps_counter = 0
    current_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Greška pri čitanju frame-a")
            break

        frame_count += 1
        fps_counter += 1

        results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, agnostic_nms=AGNOSTIC_NMS, verbose=False)

        now = time.perf_counter()
        elapsed = now - fps_start
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_history.append(current_fps)
            fps_counter = 0
            fps_start = now

        display = draw_overlay(frame.copy(), results, current_fps, frame_count)

        cv2.imshow('Detekcija pospanosti - YOLOv8s', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            path = OUTPUT_DIR / f'screenshot_{frame_count:06d}.png'
            cv2.imwrite(str(path), display)
            print(f"📸 Sačuvano: {path}")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("📊 STATISTIKA SESIJE")
    print("=" * 60)
    print(f"  Ukupno frame-ova: {frame_count}")

    if fps_history:
        avg_fps = np.mean(fps_history)
        print(f"  Prosečan FPS:     {avg_fps:.1f}")
        print(f"  Min FPS:          {min(fps_history):.1f}")
        print(f"  Max FPS:          {max(fps_history):.1f}")
        target_hit = sum(1 for f in fps_history if f >= TARGET_FPS)
        print(f"  Iznad {TARGET_FPS} FPS:  {target_hit}/{len(fps_history)} ({target_hit/len(fps_history)*100:.0f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()
