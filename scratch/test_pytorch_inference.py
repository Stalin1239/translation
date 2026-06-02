import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py

class TrafficSignCNN(nn.Module):
    def __init__(self):
        super(TrafficSignCNN, self).__init__()
        # Conv layer 1: input 3 channels, output 32, kernel 5x5
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        # Conv layer 2: input 32, output 32, kernel 5x5
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv layer 3: input 32, output 64, kernel 3x3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        # Conv layer 4: input 64, output 64, kernel 3x3
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Dense layers
        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256, 43)
        
    def forward(self, x):
        # x is (batch, 3, 30, 30)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = x.permute(0, 2, 3, 1) # Keras expects channels-last flat, PyTorch is channels-first. Let's make sure the flattening matches!
        # Wait, in Keras: Flatten() of (batch, H, W, C) flattens in row-major order: H -> W -> C.
        # If we just flatten (batch, C, H, W) in PyTorch, the layout is C -> H -> W.
        # So we MUST permute to channels-last first (batch, H, W, C), then flatten, to match Keras's Flatten layer exactly!
        # This is extremely important: x = x.permute(0, 2, 3, 1) (batch, H, W, C) -> then flatten.
        x = x.reshape(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def load_weights_from_keras_h5(pytorch_model, h5_path):
    with h5py.File(h5_path, 'r') as f:
        # Load conv1
        k_conv1 = f['model_weights/conv2d/conv2d/kernel:0'][:]
        b_conv1 = f['model_weights/conv2d/conv2d/bias:0'][:]
        # Keras: (H, W, InC, OutC) -> PyTorch: (OutC, InC, H, W)
        pytorch_model.conv1.weight.data = torch.from_numpy(k_conv1.transpose(3, 2, 0, 1))
        pytorch_model.conv1.bias.data = torch.from_numpy(b_conv1)
        
        # Load conv2
        k_conv2 = f['model_weights/conv2d_1/conv2d_1/kernel:0'][:]
        b_conv2 = f['model_weights/conv2d_1/conv2d_1/bias:0'][:]
        pytorch_model.conv2.weight.data = torch.from_numpy(k_conv2.transpose(3, 2, 0, 1))
        pytorch_model.conv2.bias.data = torch.from_numpy(b_conv2)
        
        # Load conv3
        k_conv3 = f['model_weights/conv2d_2/conv2d_2/kernel:0'][:]
        b_conv3 = f['model_weights/conv2d_2/conv2d_2/bias:0'][:]
        pytorch_model.conv3.weight.data = torch.from_numpy(k_conv3.transpose(3, 2, 0, 1))
        pytorch_model.conv3.bias.data = torch.from_numpy(b_conv3)
        
        # Load conv4
        k_conv4 = f['model_weights/conv2d_3/conv2d_3/kernel:0'][:]
        b_conv4 = f['model_weights/conv2d_3/conv2d_3/bias:0'][:]
        pytorch_model.conv4.weight.data = torch.from_numpy(k_conv4.transpose(3, 2, 0, 1))
        pytorch_model.conv4.bias.data = torch.from_numpy(b_conv4)
        
        # Load fc1
        k_fc1 = f['model_weights/dense/dense/kernel:0'][:]
        b_fc1 = f['model_weights/dense/dense/bias:0'][:]
        # Keras: (InFeatures, OutFeatures) -> PyTorch: (OutFeatures, InFeatures)
        pytorch_model.fc1.weight.data = torch.from_numpy(k_fc1.transpose(1, 0))
        pytorch_model.fc1.bias.data = torch.from_numpy(b_fc1)
        
        # Load fc2
        k_fc2 = f['model_weights/dense_1/dense_1/kernel:0'][:]
        b_fc2 = f['model_weights/dense_1/dense_1/bias:0'][:]
        pytorch_model.fc2.weight.data = torch.from_numpy(k_fc2.transpose(1, 0))
        pytorch_model.fc2.bias.data = torch.from_numpy(b_fc2)

if __name__ == "__main__":
    h5_path = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\models\best_model.h5"
    model = TrafficSignCNN()
    load_weights_from_keras_h5(model, h5_path)
    model.eval()
    
    # Generate dummy input of shape (1, 3, 30, 30)
    dummy_input = torch.randn(1, 3, 30, 30)
    output = model(dummy_input)
    print("PyTorch model loaded weights successfully!")
    print("Output shape:", output.shape)
    print("Class predicted:", torch.argmax(output, dim=1).item())
