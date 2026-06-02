import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

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
    
    data_dir = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\data"
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    true_labels = np.argmax(y_test, axis=1)
    
    print("Finding some correctly classified images in X_test...")
    count = 0
    for idx in range(len(X_test)):
        img = X_test[idx].astype(np.float32)
        # Transpose
        img_t = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_t).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
            conf = torch.max(output).item()
            
        if pred == true_labels[idx]:
            class_name = GTSRB_CLASSES_DB.get(pred, {}).get('name', 'Unknown')
            print(f"Index: {idx:<4} | Class: {pred:<2} ({class_name:<30}) | Conf: {conf*100:.2f}%")
            count += 1
            if count >= 10:
                break
