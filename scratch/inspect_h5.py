import h5py

model_path = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\models\best_model.h5"

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
    else:
        print(f"Group: {name}")

with h5py.File(model_path, 'r') as f:
    f.visititems(print_structure)
