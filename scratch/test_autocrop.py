import sys
import os
import cv2
import numpy as np
import torch

# Add the project directory to sys.path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.road_sign_detector import TrafficSignCNN, load_weights_from_keras_h5, GTSRB_CLASSES_DB

def get_road_sign_roi(image_bgr):
    """
    HSV-based color segmentation on individual color channels and contour geometry
    to locate and crop a road sign from an uncropped image.
    """
    h, w, c = image_bgr.shape
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
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
    
    # Analyze each color mask separately to avoid merging with background
    color_masks = [
        ("Red", mask_red),
        ("Blue", mask_blue),
        ("Yellow/Orange", mask_yellow)
    ]
    
    best_cnt = None
    best_area = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    for color_name, mask in color_masks:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise
            if area < 100:
                continue
                
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = float(cw) / ch
            
            # Road signs are symmetric (aspect ratio close to 1.0)
            if 0.65 <= aspect_ratio <= 1.5:
                # Exclude massive background elements (e.g. sky or background panels)
                # A sign in an uncropped image rarely covers more than 20% of the image
                if area < 0.2 * (w * h):
                    if area > best_area:
                        best_area = area
                        best_cnt = cnt
                        
    if best_cnt is not None:
        x, y, cw, ch = cv2.boundingRect(best_cnt)
        # If the detected sign covers almost the whole image already, no need to crop
        if cw * ch > 0.8 * (w * h):
            print("Auto-crop: Detected sign already covers most of the image. Skipping crop.")
            return image_bgr
            
        # Add 15% padding around the bounding box to capture the full sign beautifully
        pad_w = int(cw * 0.15)
        pad_h = int(ch * 0.15)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + cw + pad_w)
        y2 = min(h, y + ch + pad_h)
        
        print(f"Auto-crop: Cropped road sign bounding box from ({x}, {y}, {cw}, {ch}) to ({x1}, {y1}, {x2}, {y2})")
        return image_bgr[y1:y2, x1:x2]
        
    print("Auto-crop: No road sign contour detected. Using full image.")
    return image_bgr

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
    h5_path = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\models\best_model.h5"
    model = TrafficSignCNN()
    load_weights_from_keras_h5(model, h5_path)
    model.eval()
    
    # Load a test image (e.g. 00122.png, which is Speed Limit 30, Class 1)
    test_image_path = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\data\Test\00122.png"
    sign_img = cv2.imread(test_image_path)
    
    # Embed the small sign (e.g. 30x30) inside a larger background image (e.g. 300x300)
    bg_h, bg_w = 300, 300
    # Create background (green landscape and sky representation)
    bg_img = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
    bg_img[:150, :] = [230, 210, 180]  # Light sky-blue BGR
    bg_img[150:, :] = [70, 130, 50]    # Green field BGR
    
    # Overlay the road sign onto the background
    sign_h, sign_w = sign_img.shape[:2]
    # Place sign at coordinates (80, 80)
    offset_y = 80
    offset_x = 80
    bg_img[offset_y:offset_y+sign_h, offset_x:offset_x+sign_w] = sign_img
    
    # Save simulated uncropped image for reference
    os.makedirs("scratch", exist_ok=True)
    simulated_path = "scratch/simulated_uncropped.png"
    cv2.imwrite(simulated_path, bg_img)
    print(f"Created simulated uncropped image at: {simulated_path}")
    
    # ------------------ TEST 1: WITHOUT AUTO-CROP ------------------
    print("\n--- TEST 1: Inference WITHOUT Auto-Cropping ---")
    img_resized = cv2.resize(bg_img, (30, 30))
    img_array = img_resized.astype(np.float32)
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
        conf = torch.max(output).item()
    print(f"Predicted Class: {pred} ({GTSRB_CLASSES_DB.get(pred, {}).get('name', 'Unknown')}) | Conf: {conf*100:.2f}%")
    
    # ------------------ TEST 2: WITH AUTO-CROP ------------------
    print("\n--- TEST 2: Inference WITH Auto-Cropping ---")
    cropped_img = get_road_sign_roi(bg_img)
    cv2.imwrite("scratch/simulated_cropped.png", cropped_img)
    print(f"Cropped image shape: {cropped_img.shape}")
    
    img_resized_crop = cv2.resize(cropped_img, (30, 30))
    img_array_crop = img_resized_crop.astype(np.float32)
    img_tensor_crop = torch.from_numpy(img_array_crop.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        output_crop = model(img_tensor_crop)
        pred_crop = torch.argmax(output_crop, dim=1).item()
        conf_crop = torch.max(output_crop).item()
    print(f"Predicted Class: {pred_crop} ({GTSRB_CLASSES_DB.get(pred_crop, {}).get('name', 'Unknown')}) | Conf: {conf_crop*100:.2f}%")
    
    if pred_crop == 1:
        print("\n🎉 SUCCESS! Auto-cropping successfully extracted the road sign and correctly predicted Class 1!")
    else:
        print("\n❌ FAILURE! Predicted incorrect class.")
