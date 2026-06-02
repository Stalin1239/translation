import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the project directory to sys.path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.road_sign_detector import TrafficSignCNN, load_weights_from_keras_h5

if __name__ == "__main__":
    h5_path = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\models\best_model.h5"
    model = TrafficSignCNN()
    load_weights_from_keras_h5(model, h5_path)
    model.eval()
    
    data_dir = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\data"
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # y_test is one-hot encoded: (num_samples, 43)
    true_labels = np.argmax(y_test, axis=1)
    
    correct = 0
    total = len(X_test)
    
    # Process in batches
    batch_size = 64
    for i in range(0, total, batch_size):
        batch_x = X_test[i:i+batch_size].astype(np.float32)
        batch_y = true_labels[i:i+batch_size]
        
        # In PyTorch: expect input of shape (batch, 3, 30, 30)
        # X_test is in BGR format and shape (batch, 30, 30, 3)
        batch_x_t = batch_x.transpose(0, 3, 1, 2)
        
        inputs = torch.from_numpy(batch_x_t)
        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).numpy()
            
        correct += np.sum(preds == batch_y)
        
    accuracy = (correct / total) * 100
    print(f"PyTorch Model Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
