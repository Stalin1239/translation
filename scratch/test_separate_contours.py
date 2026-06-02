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
    h, w, c = img.shape
    
    # 1. Red Mask (covers STOP, Speed Limit borders, warning triangles, prohibitory)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # 2. Blue Mask (covers mandatory direction circles)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 3. Yellow/Orange Mask (covers caution/priority diamonds/triangles)
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Analyze each color mask separately
    color_masks = [
        ("Red", mask_red),
        ("Blue", mask_blue),
        ("Yellow/Orange", mask_yellow)
    ]
    
    best_cnt = None
    best_area = 0
    best_color = None
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    for color_name, mask in color_masks:
        # Clean mask
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"\n--- Color: {color_name} | Found {len(contours)} contours ---")
        
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = float(cw) / ch
            print(f"  Contour {idx}: Area={area:.1f}, BBox=({cx},{cy},{cw},{ch}), AspectRatio={aspect_ratio:.2f}")
            
            # Filter criteria:
            # 1. Minimum area to filter noise
            # 2. Aspect ratio close to 1.0 (between 0.6 and 1.6)
            # 3. Make sure it doesn't cover the entire image (unless it is a cropped image)
            if area >= 100 and 0.6 <= aspect_ratio <= 1.6:
                # Limit the maximum size to 50% of the image area to exclude sky/backgrounds
                if area < 0.5 * (w * h):
                    if area > best_area:
                        best_area = area
                        best_cnt = cnt
                        best_color = color_name
                        
    if best_cnt is not None:
        cx, cy, cw, ch = cv2.boundingRect(best_cnt)
        pad_w = int(cw * 0.15)
        pad_h = int(ch * 0.15)
        x1 = max(0, cx - pad_w)
        y1 = max(0, cy - pad_h)
        x2 = min(w, cx + cw + pad_w)
        y2 = min(h, cy + ch + pad_h)
        print(f"\n🏆 SUCCESS! Selected best contour from {best_color} mask:")
        print(f"  BBox: ({cx},{cy},{cw},{ch}) -> Padded Crop: ({x1},{y1},{x2},{y2})")
    else:
        print("\n❌ No suitable road sign contour found.")
