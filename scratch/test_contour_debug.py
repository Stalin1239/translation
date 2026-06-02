import os
import cv2
import numpy as np
import sys

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
    img_path = "scratch/simulated_uncropped.png"
    if not os.path.exists(img_path):
        print("Error: simulated image not found.")
        sys.exit(1)
        
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. Red Mask (looser thresholds)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # 2. Blue Mask
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 3. Yellow/Orange Mask
    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine them
    combined_mask = cv2.bitwise_or(mask_red, mask_blue)
    combined_mask = cv2.bitwise_or(combined_mask, mask_yellow)
    
    # Let's see what is matched in the region of the sign [80:110, 80:110]
    sign_region = combined_mask[80:110, 80:110]
    print(f"Number of matched pixels in the sign region (30x30): {np.sum(sign_region > 0)}")
    
    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\nTotal contours found: {len(contours)}")
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = float(cw) / ch
        print(f"Contour {idx}: Area={area:.1f}, BBox=({cx},{cy},{cw},{ch}), AspectRatio={aspect_ratio:.2f}")
