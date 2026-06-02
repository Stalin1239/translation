import os
from PIL import Image, ImageDraw, ImageFont

def generate_test_image():
    # 1. Create a large, high-contrast white sheet
    img = Image.new('RGB', (600, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    
    # 2. Load Nirmala UI font (native to Windows, supports Hindi and all Indian scripts)
    font_path = "C:\\Windows\\Fonts\\Nirmala.ttf"
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, 48)
    else:
        # Fallback to default
        font = ImageFont.load_default()
        
    # 3. Write bold regional words
    d.text((50, 60), "नमस्ते स्वागत है", fill=(0, 0, 0), font=font)
    
    output_path = os.path.abspath("temp_hindi_test.png")
    img.save(output_path)
    print(f"Generated test image at: {output_path}")

if __name__ == '__main__':
    generate_test_image()
