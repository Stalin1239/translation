import sys
import os
import cv2
import pandas as pd
import numpy as np
import torch

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.road_sign_detector import TrafficSignCNN, load_weights_from_keras_h5

if __name__ == "__main__":
    h5_path = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\models\best_model.h5"
    model = TrafficSignCNN()
    load_weights_from_keras_h5(model, h5_path)
    model.eval()
    
    base_dir = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\data"
    test_csv_path = os.path.join(base_dir, 'Test.csv')
    
    df = pd.read_csv(test_csv_path)
    print(f"Total test images in CSV: {len(df)}")
    
    correct = 0
    total = 0
    
    # Let's test first 500 images
    for idx, row in df.head(500).iterrows():
        img_rel_path = row['Path']
        true_class = row['ClassId']
        
        img_path = os.path.join(base_dir, img_rel_path)
        if not os.path.exists(img_path):
            continue
            
        # Load and preprocess
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        img_resized = cv2.resize(img_bgr, (30, 30))
        img_array = img_resized.astype(np.float32)
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
            
        if pred == true_class:
            correct += 1
        total += 1
        
    accuracy = (correct / total) * 100
    print(f"PyTorch model accuracy on actual test images: {accuracy:.2f}% ({correct}/{total})")
