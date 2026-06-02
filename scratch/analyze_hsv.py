import os
import cv2
import numpy as np
import sys

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
    test_image_path = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\data\Test\00122.png"
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"Error: Could not load {test_image_path}")
        sys.exit(1)
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, c = img.shape
    print(f"Image shape: {h}x{w}x{c}")
    
    # Flatten the image to analyze pixels
    h_vals = hsv[:, :, 0].flatten()
    s_vals = hsv[:, :, 1].flatten()
    v_vals = hsv[:, :, 2].flatten()
    
    # In GTSRB, the red border is typically saturated red. Let's find pixels with high saturation/value or standard red hue.
    # Red hue is usually < 10 or > 170.
    red_pixels = []
    for y in range(h):
        for x in range(w):
            hue = hsv[y, x, 0]
            sat = hsv[y, x, 1]
            val = hsv[y, x, 2]
            # Red hue ranges: 0..10 or 160..180
            if (hue < 15 or hue > 165) and sat > 40 and val > 40:
                red_pixels.append((x, y, hue, sat, val))
                
    print(f"Found {len(red_pixels)} red-like pixels out of {h*w} total pixels.")
    if red_pixels:
        hues = [p[2] for p in red_pixels]
        sats = [p[3] for p in red_pixels]
        vals = [p[4] for p in red_pixels]
        print(f"Hue: min={min(hues)}, max={max(hues)}, mean={np.mean(hues):.2f}")
        print(f"Saturation: min={min(sats)}, max={max(sats)}, mean={np.mean(sats):.2f}")
        print(f"Value: min={min(vals)}, max={max(vals)}, mean={np.mean(vals):.2f}")
    else:
        # Let's print out a histogram of the hues to see what hues exist
        print("No red-like pixels found under thresholds! Hue histogram:")
        hist, bin_edges = np.histogram(h_vals, bins=10, range=(0, 180))
        for i in range(len(hist)):
            print(f"Hue {bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}: {hist[i]} pixels")
            
    # Let's see what values are actually present in the image
    print("\nSample BGR pixels:")
    print("Center pixel:", img[h//2, w//2])
    print("Top-left pixel:", img[0, 0])
