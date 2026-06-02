import sys
import os
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.append(os.getcwd())

from modules.image_translation import perform_ocr

def run_test():
    # 1. Create a dummy image with high-contrast Hindi text
    img = Image.new('RGB', (300, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    # Write some simple English and a Hindi word if supported, or just use Hindi lang_list
    d.text((50, 30), "नमस्ते", fill=(0, 0, 0))
    
    test_img_path = "temp_ocr_test.jpg"
    img.save(test_img_path)
    
    print("--- TESTING HINDI OCR ---")
    try:
        results = perform_ocr(test_img_path, use_engine='easyocr', lang_list=['en', 'hi'], ocr_mode='scanned_doc')
        print(f"OCR Detections count: {len(results) if results else 0}")
        print("Results detail:")
        print(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        
    # Clean up test image
    if os.path.exists(test_img_path):
        os.remove(test_img_path)

if __name__ == '__main__':
    run_test()
