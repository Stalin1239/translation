import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from modules.text_translation import translate_text
from modules.utils import save_translation_history

# Setup Pytesseract command path
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def preprocess_image_for_ocr(image_path, target_engine='easyocr', ocr_mode='scanned_doc'):
    """
    Optimizes image readability. For EasyOCR (deep learning), contrast enhancement
    and resizing is preferred. For Tesseract, binarization/thresholding is preferred.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
            
        h, w = img.shape[:2]
        # 1. Resize if image is too small to improve OCR accuracy
        scaled = False
        if max(h, w) < 1200:
            scale = 1200.0 / max(h, w)
            img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            scaled = True
            
        # 2. If it's a scanned doc and we didn't need to scale, return raw to avoid noise
        if ocr_mode == 'scanned_doc' and not scaled:
            return image_path
            
        if target_engine == 'easyocr':
            if ocr_mode == 'scene_photo':
                # EasyOCR (neural engine) performs best on high-definition COLOR scene images.
                # We enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
                # in LAB color space to preserve color information while eliminating lighting shadows.
                # We also apply a bilateral filter to smooth noise while keeping text edges extremely sharp.
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b_ch = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                denoised_l = cv2.bilateralFilter(cl, 9, 75, 75)
                limg = cv2.merge((denoised_l, a, b_ch))
                enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            else:
                # Scanned Doc (that was scaled): Keep color gradients, apply light bilateral smoothing to remove scale noise
                enhanced = cv2.bilateralFilter(img, 5, 50, 50)
                
            base, ext = os.path.splitext(image_path)
            preprocessed_path = f"{base}_preprocessed{ext}"
            cv2.imwrite(preprocessed_path, enhanced)
            return preprocessed_path
        else:
            # Tesseract performs best on high-contrast grayscale binary images.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            thresholded = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            base, ext = os.path.splitext(image_path)
            preprocessed_path = f"{base}_preprocessed{ext}"
            cv2.imwrite(preprocessed_path, thresholded)
            return preprocessed_path
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return image_path  # Fallback to original image

def merge_ocr_boxes(ocr_results, y_threshold=20, x_threshold=70):
    """
    Layout Analysis: Groups and merges adjacent OCR word fragments that lie
    on the same horizontal line. This constructs complete sentences/phrases
    which guarantees contextually accurate sentence-level translation rather than
    broken word-by-word translation.
    """
    if not ocr_results:
        return []
    
    # Sort boxes top-to-bottom, then left-to-right
    sorted_boxes = sorted(ocr_results, key=lambda r: (r['box'][1], r['box'][0]))
    
    merged = []
    while sorted_boxes:
        curr = sorted_boxes.pop(0)
        curr_box = curr['box']
        curr_text = curr['text']
        curr_conf = curr['confidence']
        
        # Look for boxes on the same horizontal line (similar Y) and close horizontally (small X delta)
        merged_any = True
        while merged_any:
            merged_any = False
            for i, other in enumerate(sorted_boxes):
                other_box = other['box']
                
                # Check Y overlap
                y_delta = abs(curr_box[1] - other_box[1])
                # Check horizontal proximity (other X starts shortly after curr X ends)
                x_delta = other_box[0] - curr_box[2]
                
                if y_delta < y_threshold and 0 <= x_delta < x_threshold:
                    # Merge boxes
                    new_box = (
                        min(curr_box[0], other_box[0]),
                        min(curr_box[1], other_box[1]),
                        max(curr_box[2], other_box[2]),
                        max(curr_box[3], other_box[3])
                    )
                    curr_text = f"{curr_text} {other['text']}"
                    curr_conf = (curr_conf + other['confidence']) / 2.0
                    curr_box = new_box
                    
                    # Remove from list
                    sorted_boxes.pop(i)
                    merged_any = True
                    break
                    
        merged.append({
            'text': curr_text.strip(),
            'box': curr_box,
            'confidence': curr_conf
        })
        
    return merged

def perform_ocr(image_path, use_engine='easyocr', lang_list=None, ocr_mode='scanned_doc'):
    """
    Performs OCR using EasyOCR or Pytesseract.
    Returns: A list of dicts containing 'text', 'box', 'confidence'
    """
    if lang_list is None:
        # Default to a highly-robust, comprehensive list of major Indian scripts + English
        lang_list = ['en', 'hi', 'kn', 'ta', 'te', 'ml', 'mr', 'bn']
        
    results = []
    
    # Preprocess image specifically optimized for the chosen engine and capture type
    preprocessed_path = preprocess_image_for_ocr(image_path, target_engine=use_engine, ocr_mode=ocr_mode)
    
    # Try 1: EasyOCR
    if use_engine == 'easyocr':
        try:
            import easyocr
            # Reader handles GPU acceleration dynamically
            reader = easyocr.Reader(lang_list, gpu=True)
            ocr_res = reader.readtext(
                preprocessed_path, 
                contrast_ths=0.1, 
                adjust_contrast=0.7, 
                text_threshold=0.4, 
                low_text=0.3
            )
            
            for (bbox, text, prob) in ocr_res:
                x0, y0 = int(bbox[0][0]), int(bbox[0][1])
                x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                results.append({
                    'text': text.strip(),
                    'box': (x0, y0, x2, y2),
                    'confidence': float(prob)
                })
                
            # Fallback: If preprocessed image yielded no results, retry on the raw original image!
            if not results and preprocessed_path != image_path:
                print("Preprocessing resulted in zero text detections. Retrying OCR on original raw image...")
                ocr_res = reader.readtext(
                    image_path, 
                    contrast_ths=0.1, 
                    adjust_contrast=0.7, 
                    text_threshold=0.4, 
                    low_text=0.3
                )
                for (bbox, text, prob) in ocr_res:
                    x0, y0 = int(bbox[0][0]), int(bbox[0][1])
                    x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                    results.append({
                        'text': text.strip(),
                        'box': (x0, y0, x2, y2),
                        'confidence': float(prob)
                    })
            
            if results:
                # Apply layout-analysis sentence grouping
                return merge_ocr_boxes(results)
        except Exception as e:
            import traceback
            try:
                log_dir = os.path.join(os.path.dirname(image_path), '..', 'temp')
                if not os.path.exists(log_dir):
                    log_dir = os.path.dirname(image_path)
                log_path = os.path.join(log_dir, 'ocr_debug.log')
                with open(log_path, 'a', encoding='utf-8') as log_f:
                    log_f.write(f"\n--- EasyOCR Error for {os.path.basename(image_path)} ---\n")
                    traceback.print_exc(file=log_f)
            except Exception:
                pass
            print(f"EasyOCR failed or not installed, falling back to Pytesseract. Error: {e}")
            use_engine = 'pytesseract'

    # Try 2: Pytesseract
    if use_engine == 'pytesseract' or not results:
        try:
            tess_langs = []
            for l in lang_list:
                if l == 'en': tess_langs.append('eng')
                elif l == 'hi': tess_langs.append('hin')
                elif l == 'kn': tess_langs.append('kan')
                elif l == 'ta': tess_langs.append('tam')
                elif l == 'te': tess_langs.append('tel')
                elif l == 'ml': tess_langs.append('mal')
                elif l == 'mr': tess_langs.append('mar')
                elif l == 'bn': tess_langs.append('ben')
            
            lang_str = "+".join(tess_langs) if tess_langs else 'eng'
            
            data = pytesseract.image_to_data(
                Image.open(preprocessed_path), 
                lang=lang_str, 
                output_type=pytesseract.Output.DICT
            )
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = float(data['conf'][i])
                if text and conf > 15.0:  # Higher threshold for cleaner Pytesseract text
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    results.append({
                        'text': text,
                        'box': (x, y, x + w, y + h),
                        'confidence': conf / 100.0
                    })
            
            if results:
                return merge_ocr_boxes(results)
        except Exception as e:
            import traceback
            try:
                log_dir = os.path.join(os.path.dirname(image_path), '..', 'temp')
                if not os.path.exists(log_dir):
                    log_dir = os.path.dirname(image_path)
                log_path = os.path.join(log_dir, 'ocr_debug.log')
                with open(log_path, 'a', encoding='utf-8') as log_f:
                    log_f.write(f"\n--- Pytesseract Error for {os.path.basename(image_path)} ---\n")
                    traceback.print_exc(file=log_f)
            except Exception:
                pass
            print(f"Pytesseract OCR failed. Error: {e}")

    # Remove temporary preprocessed file if created
    if preprocessed_path != image_path and os.path.exists(preprocessed_path):
        try:
            os.remove(preprocessed_path)
        except Exception:
            pass

    return results

def wrap_text(text, font, max_width):
    """
    Fits and wraps translated text into multiple lines that comfortably fit 
    within the horizontal bounding box boundary.
    """
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        # Calculate size of test line
        if hasattr(font, 'getbbox'):
            w = font.getbbox(test_line)[2]
        else:
            w = font.getsize(test_line)[0]
            
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            
    if current_line:
        lines.append(' '.join(current_line))
        
    return lines

def overlay_translation(image_path, ocr_results, target_lang='en'):
    """
    Translates detected OCR sentence blocks, draws solid colored boxes over the original
    text in the image to cover them, and overlays the newly translated text with multi-line wrapping
    and dynamic Nirmala UI Unicode font support.
    Returns: Absolute path to the output image file.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        
        # Check standard Windows paths for Nirmala UI or Arial.
        # Nirmala.ttf is Windows' default standard font for ALL Indian regional scripts.
        font_search_paths = [
            r"C:\Windows\Fonts\Nirmala.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\seguisb.ttf"
        ]
        
        font_path = None
        for path in font_search_paths:
            if os.path.exists(path):
                font_path = path
                break
                
        for item in ocr_results:
            orig_text = item['text']
            box = item['box']  # (x0, y0, x2, y2)
            x0, y0, x2, y2 = box
            
            box_height = max(12, y2 - y0)
            box_width = max(20, x2 - x0)
            
            # 1. Translate sentence-level text contextually
            translated_text, _, _ = translate_text(orig_text, 'auto', target_lang)
            
            # 2. Cover original text with high-end neon overlay box
            draw.rectangle(
                [x0, y0, x2, y2], 
                fill=(15, 15, 28, 240),      # Dark neon cyber blue
                outline=(0, 242, 254, 255),  # Glowing cyan active borders
                width=2
            )
            
            # 3. Dynamic font size calculation
            font_size = max(11, int(box_height * 0.70))
            if font_path:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except Exception:
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
                
            # 4. Wrap text to fit boundaries
            lines = wrap_text(translated_text, font, box_width - 8)
            
            # Calculate line height
            if hasattr(font, 'getbbox'):
                line_height = font.getbbox("A")[3]
            else:
                line_height = font.getsize("A")[1]
                
            # Center the lines vertically in the bounding box
            total_text_height = len(lines) * (line_height + 2)
            if total_text_height < box_height:
                current_y = y0 + (box_height - total_text_height) // 2
            else:
                current_y = y0 + 3
                
            # Draw text lines
            for line in lines:
                if current_y + line_height > y2 + 8:
                    break  # Prevent visual text overflow past the box boundary
                draw.text((x0 + 6, current_y), line, fill=(255, 255, 255, 255), font=font)
                current_y += line_height + 2
            
            # Save translation log
            save_translation_history("Visual OCR Overlay", "auto", target_lang, orig_text, translated_text, item['confidence'])

        # Save overlaid image to outputs/
        output_filename = f"ocr_translated_{os.path.basename(image_path)}"
        output_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'outputs')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_filename)
        
        # Convert back to RGB and save
        rgb_img = img.convert("RGB")
        rgb_img.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error overlaying text onto image: {e}")
        return image_path  # Fallback to original image
