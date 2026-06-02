import os
import cv2
import numpy as np
import sys

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
    # Load the simulated uncropped image
    img_path = "scratch/simulated_uncropped.png"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} does not exist.")
        sys.exit(1)
        
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, c = img.shape
    
    # Wide saturation/value mask
    # We want saturated pixels (sat > 40) that are reasonably bright (val > 40)
    # and not green (hue between 40 and 85)
    sat_mask = hsv[:, :, 1] > 40
    val_mask = hsv[:, :, 2] > 40
    
    # Green hue mask
    hue = hsv[:, :, 0]
    is_green = (hue >= 40) & (hue <= 85)
    
    # Combined mask: saturated, bright, and NOT green
    combined_mask = sat_mask & val_mask & ~is_green
    combined_mask = (combined_mask * 255).astype(np.uint8)
    
    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} candidate contours.")
    
    best_cnt = None
    best_area = 0
    
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = float(cw) / ch
        print(f"Contour {idx}: Area={area:.1f}, BBox=({cx},{cy},{cw},{ch}), AspectRatio={aspect_ratio:.2f}")
        
        if area >= 100 and 0.5 <= aspect_ratio <= 2.0:
            if area > best_area:
                best_area = area
                best_cnt = cnt
                
    if best_cnt is not None:
        cx, cy, cw, ch = cv2.boundingRect(best_cnt)
        # Pad by 10%
        pad_w = int(cw * 0.1)
        pad_h = int(ch * 0.1)
        x1 = max(0, cx - pad_w)
        y1 = max(0, cy - pad_h)
        x2 = min(w, cx + cw + pad_w)
        y2 = min(h, cy + ch + pad_h)
        print(f"Selected BBox: ({cx},{cy},{cw},{ch}) -> Padded BBox: ({x1},{y1},{x2},{y2})")
        cropped = img[y1:y2, x1:x2]
        cv2.imwrite("scratch/simulated_cropped_wide.png", cropped)
    else:
        print("No suitable contour found!")
