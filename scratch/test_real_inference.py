import sys
import os
import torch

# Reconfigure stdout to support utf-8 in Windows console
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add the project directory to sys.path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.road_sign_detector import predict_road_sign_cnn

if __name__ == "__main__":
    # Define test images
    base_test_dir = r"c:\Users\stali\translation\Traffic-Signs-Recognition-using-CNN-Keras-main\data\Test"
    
    # 00111.png is GTSRB class 14 (Stop sign)
    # 00122.png is GTSRB class 2 or 3 (Speed Limit)
    test_cases = [
        ("00111.png", "kn"),  # Translate to Kannada
        ("00122.png", "hi"),  # Translate to Hindi
        ("00122.png", "ml"),  # Translate to Malayalam
    ]
    
    print("--- Running AI CNN Road Sign Inference & Translation Test ---")
    for filename, lang in test_cases:
        image_path = os.path.join(base_test_dir, filename)
        if not os.path.exists(image_path):
            print(f"Test image {image_path} does not exist!")
            continue
            
        print(f"\nScanning {filename} with target language: '{lang}'...")
        result = predict_road_sign_cnn(image_path, target_lang=lang)
        
        print(f"Detected: {result['detected']}")
        if result['detected']:
            print(f"Class ID: {result['class_id']}")
            print(f"Original Name: {result['symbol_name']}")
            print(f"Category: {result['category']}")
            print(f"Meaning (Translated): {result['meaning']}")
            print(f"Safety Alert (Translated): {result['alert']}")
            print(f"Confidence: {result['confidence']:.4f}")
        else:
            print(f"Error: {result['meaning']}")
            print(f"Alert: {result['alert']}")
