import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2, numpy as np, torch
from modules.road_sign_detector import TrafficSignCNN, load_weights_from_keras_h5, GTSRB_CLASSES_DB

h5 = r'Traffic-Signs-Recognition-using-CNN-Keras-main/models/best_model.h5'
model = TrafficSignCNN()
load_weights_from_keras_h5(model, h5)
model.eval()

def infer_top3(img):
    arr = cv2.resize(img, (30, 30)).astype('float32').transpose(2, 0, 1)
    t = torch.from_numpy(arr).unsqueeze(0)
    with torch.no_grad():
        out = model(t)[0]
    vals, idx = torch.topk(out, 3)
    return [(int(idx[i]), float(vals[i])) for i in range(3)]

def get_tier(conf):
    if conf >= 0.70: return 'HIGH (shown normally)'
    if conf >= 0.40: return 'MEDIUM (shown with warning)'
    return 'LOW -> UNKNOWN SIGN shown'

print("=" * 60)
print("CONFIDENCE THRESHOLD TEST — out-of-distribution signs")
print("=" * 60)

# Test 1: Cattle crossing (yellow triangle with black silhouette — NOT in GTSRB)
img1 = np.ones((100, 100, 3), dtype='uint8') * 255
# Yellow triangle on white background
pts = np.array([[50,5],[5,95],[95,95]], np.int32)
cv2.fillPoly(img1, [pts], (50, 200, 230))   # yellow (BGR)
cv2.circle(img1, (50, 60), 20, (0, 80, 150), -1)  # brown blob = cattle silhouette
res1 = infer_top3(img1)
print("\nTest 1: Simulated cattle crossing sign (yellow triangle + silhouette)")
for rank, (cls, conf) in enumerate(res1):
    name = GTSRB_CLASSES_DB.get(cls, {}).get('name', 'Unknown')
    print(f"  #{rank+1}  Class {cls:>2}  {name:<40} {conf*100:5.1f}%")
print(f"  => Tier: {get_tier(res1[0][1])}")

# Test 2: Railway crossing sign (big red circle with X — NOT in GTSRB)
img2 = np.ones((100, 100, 3), dtype='uint8') * 240
cv2.circle(img2, (50, 50), 45, (0, 0, 200), 6)   # red ring
cv2.line(img2, (20, 20), (80, 80), (0, 0, 200), 5)
cv2.line(img2, (80, 20), (20, 80), (0, 0, 200), 5)
res2 = infer_top3(img2)
print("\nTest 2: Simulated railway crossing sign (red circle + X)")
for rank, (cls, conf) in enumerate(res2):
    name = GTSRB_CLASSES_DB.get(cls, {}).get('name', 'Unknown')
    print(f"  #{rank+1}  Class {cls:>2}  {name:<40} {conf*100:5.1f}%")
print(f"  => Tier: {get_tier(res2[0][1])}")

# Test 3: Standard GTSRB Stop sign (should be HIGH)
img3 = cv2.imread(r'Traffic-Signs-Recognition-using-CNN-Keras-main/data/Test/00111.png')
res3 = infer_top3(img3)
print("\nTest 3: Real GTSRB Stop sign (should be HIGH confidence)")
for rank, (cls, conf) in enumerate(res3):
    name = GTSRB_CLASSES_DB.get(cls, {}).get('name', 'Unknown')
    print(f"  #{rank+1}  Class {cls:>2}  {name:<40} {conf*100:5.1f}%")
print(f"  => Tier: {get_tier(res3[0][1])}")

# Test 4: Real GTSRB Yield sign
import pandas as pd
df = pd.read_csv(r'Traffic-Signs-Recognition-using-CNN-Keras-main/data/Test.csv')
yield_row = df[df['ClassId'] == 13].iloc[0]
img4 = cv2.imread(r'Traffic-Signs-Recognition-using-CNN-Keras-main/data/' + yield_row['Path'])
res4 = infer_top3(img4)
print("\nTest 4: Real GTSRB Yield sign (should be HIGH confidence)")
for rank, (cls, conf) in enumerate(res4):
    name = GTSRB_CLASSES_DB.get(cls, {}).get('name', 'Unknown')
    print(f"  #{rank+1}  Class {cls:>2}  {name:<40} {conf*100:5.1f}%")
print(f"  => Tier: {get_tier(res4[0][1])}")
