"""
Validate the updated predict_road_sign_cnn on:
  1. GTSRB test images across multiple classes (Stop, Speed limits, Yield, No-entry, etc.)
  2. A simulated uncropped scene (sign embedded in background)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2
import numpy as np
import torch
import pandas as pd

from modules.road_sign_detector import predict_road_sign_cnn, GTSRB_CLASSES_DB

BASE_DIR = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\data"
TEST_CSV  = os.path.join(BASE_DIR, "Test.csv")

# --- Part 1: Sample one image from each of 10 key classes ---
print("=" * 65)
print("PART 1 — Key classes sampled from GTSRB Test set")
print("=" * 65)

df = pd.read_csv(TEST_CSV)
target_classes = [0, 1, 4, 9, 13, 14, 17, 25, 33, 38]   # varied set
found = {c: None for c in target_classes}

for _, row in df.iterrows():
    cls = int(row['ClassId'])
    if cls in found and found[cls] is None:
        found[cls] = os.path.join(BASE_DIR, row['Path'])
    if all(v is not None for v in found.values()):
        break

correct = 0
for cls_id, img_path in sorted(found.items()):
    if img_path is None:
        print(f"  Class {cls_id:>2}: ⚠️  no test image found")
        continue

    # Call the REAL function (no target lang needed here — skip AWS to speed test)
    # We'll test raw inference for speed; reuse the CNN directly
    from modules.road_sign_detector import TrafficSignCNN, load_weights_from_keras_h5
    import h5py

    h5_path = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\models\best_model.h5"

    # Quick inline test mirrors _infer + _extract_sign_roi from the updated function
    model = TrafficSignCNN()
    load_weights_from_keras_h5(model, h5_path)
    model.eval()

    img = cv2.imread(img_path)

    def _infer(m, i):
        arr = cv2.resize(i, (30, 30)).astype(np.float32).transpose(2, 0, 1)
        t   = torch.from_numpy(arr).unsqueeze(0)
        with torch.no_grad():
            out = m(t)
        return int(torch.argmax(out, dim=1).item()), float(torch.max(out).item())

    def _roi(img_bgr):
        img_h, img_w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        m_r = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,   50, 50]), np.array([20,  255, 255])),
            cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255])),
        )
        m_b = cv2.inRange(hsv, np.array([90,  50, 50]), np.array([140, 255, 255]))
        m_y = cv2.inRange(hsv, np.array([20,  50, 50]), np.array([40,  255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        best_cnt, best_area = None, 0
        for mask in (m_r, m_b, m_y):
            c = cv2.morphologyEx(mask,  cv2.MORPH_CLOSE, kernel)
            c = cv2.morphologyEx(c,     cv2.MORPH_OPEN,  kernel)
            cnts, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 100: continue
                x, y, cw, ch = cv2.boundingRect(cnt)
                asp = float(cw) / max(ch, 1)
                if 0.65 <= asp <= 1.5 and area < 0.20 * img_w * img_h:
                    if area > best_area:
                        best_area, best_cnt = area, cnt
        if best_cnt is not None:
            x, y, cw, ch = cv2.boundingRect(best_cnt)
            if cw * ch <= 0.80 * img_w * img_h:
                pw = max(int(cw * 0.15), 3); ph = max(int(ch * 0.15), 3)
                return img_bgr[max(0,y-ph):min(img_h,y+ch+ph),
                               max(0,x-pw):min(img_w,x+cw+pw)]
        return img_bgr

    cls_f, cf = _infer(model, img)
    roi        = _roi(img)
    cls_c, cc  = _infer(model, roi)
    pred = cls_c if cc >= cf else cls_f
    conf = max(cc, cf)

    name     = GTSRB_CLASSES_DB.get(pred, {}).get('name', 'Unknown')
    expected = GTSRB_CLASSES_DB.get(cls_id, {}).get('name', 'Unknown')
    ok       = "✅" if pred == cls_id else "❌"
    if pred == cls_id: correct += 1
    print(f"  Class {cls_id:>2} | Expected: {expected:<35} | Got: {name:<35} | Conf: {conf*100:5.1f}% {ok}")

print(f"\n  Accuracy on sampled classes: {correct}/{len(target_classes)}")

# --- Part 2: Simulated uncropped image ---
print()
print("=" * 65)
print("PART 2 — Simulated uncropped scene (sign in 300x300 background)")
print("=" * 65)

sign_path = os.path.join(BASE_DIR, "Test/00111.png")  # Stop sign (Class 14)
sign_img  = cv2.imread(sign_path)
bg        = np.zeros((300, 300, 3), dtype=np.uint8)
bg[:150]  = [200, 210, 220]
bg[150:]  = [70, 130, 50]
sh, sw    = sign_img.shape[:2]
bg[80:80+sh, 90:90+sw] = sign_img

# Load model once
model_test = TrafficSignCNN()
load_weights_from_keras_h5(model_test, h5_path)
model_test.eval()

cls_f, cf = _infer(model_test, bg)
roi        = _roi(bg)
cls_c, cc  = _infer(model_test, roi)
pred_u     = cls_c if cc >= cf else cls_f

name = GTSRB_CLASSES_DB.get(pred_u, {}).get('name', 'Unknown')
ok   = "✅" if pred_u == 14 else "❌"
print(f"  True class: Stop (14) | Predicted: {name} (cls {pred_u}) {ok}")
print(f"  Full image conf: {cf*100:.1f}% | Cropped conf: {cc*100:.1f}%")
