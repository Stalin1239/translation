import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the project directory to sys.path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.road_sign_detector import TrafficSignCNN, load_weights_from_keras_h5, GTSRB_CLASSES_DB

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
    h5_path = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\models\best_model.h5"
    model = TrafficSignCNN()
    load_weights_from_keras_h5(model, h5_path)
    model.eval()
    
    test_image_111 = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\data\Test\00111.png" # True Class: 14 (Stop)
    test_image_122 = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\data\Test\00122.png" # True Class: 1 (Speed Limit 30)
    
    for label, img_path, true_class in [("00111.png (Stop)", test_image_111, 14), ("00122.png (Speed limit 30)", test_image_122, 1)]:
        print(f"\n=================== TESTING IMAGE: {label} [True Class: {true_class}] ===================")
        
        # Load image via cv2 (BGR)
        img_bgr = cv2.imread(img_path)
        
        interpolations = [
            ("INTER_NEAREST", cv2.INTER_NEAREST),
            ("INTER_LINEAR", cv2.INTER_LINEAR),
            ("INTER_CUBIC", cv2.INTER_CUBIC),
            ("INTER_AREA", cv2.INTER_AREA),
        ]
        
        for interp_name, interp_val in interpolations:
            img_resized = cv2.resize(img_bgr, (30, 30), interpolation=interp_val)
            img_array = img_resized.astype(np.float32)
            
            # Transpose to channels-first (3, 30, 30)
            img_tensor = img_array.transpose(2, 0, 1)
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)
            
            with torch.no_grad():
                output = model(img_tensor)
                confidence = torch.max(output).item()
                predicted_class = torch.argmax(output, dim=1).item()
                
            status = "🔥 CORRECT!" if predicted_class == true_class else "❌ WRONG"
            class_name = GTSRB_CLASSES_DB.get(predicted_class, {}).get('name', 'Unknown')
            print(f"Interp: {interp_name:<15} -> Predicted: {predicted_class} ({class_name}) | Conf: {confidence*100:.2f}% | {status}")
