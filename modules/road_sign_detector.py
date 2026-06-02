import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image
from modules.text_translation import translate_text

# A dictionary of standard Indian regional road sign texts, their English meaning, and corresponding safety actions.
ROAD_SIGNS_DB = {
    # --- Kannada ---
    'ವೇಗ ಮಿತಿ': {
        'meaning': 'Speed Limit',
        'alert': '⚠️ Speed Limit Zone Ahead. Please monitor your speed indicator.',
        'category': 'Regulatory'
    },
    'ನಿಲ್ಲಿಸಿ': {
        'meaning': 'Stop',
        'alert': '🛑 Halt! Complete stop required before proceeding.',
        'category': 'Regulatory'
    },
    'ಏಕಮುಖ ರಸ್ತೆ': {
        'meaning': 'One Way',
        'alert': '➡️ One Way Traffic Ahead. Do not enter from opposite direction.',
        'category': 'Regulatory'
    },
    'ಮುಂದೆ ಹೋಗಬೇಡಿ': {
        'meaning': 'No Entry',
        'alert': '🚫 No Entry! Entry is prohibited.',
        'category': 'Regulatory'
    },
    'ನಿಧಾನವಾಗಿ ಹೋಗಿ': {
        'meaning': 'Go Slow',
        'alert': '⚠️ Go Slow! Pedestrian crossing or narrow road ahead.',
        'category': 'Cautionary'
    },
    'ಅಪಾಯ': {
        'meaning': 'Danger',
        'alert': '🔥 Danger Zone! Exercise extreme caution.',
        'category': 'Warning'
    },
    
    # --- Hindi ---
    'गति सीमा': {
        'meaning': 'Speed Limit',
        'alert': '⚠️ Speed Limit Zone. Adjust vehicle speed.',
        'category': 'Regulatory'
    },
    'रुकिए': {
        'meaning': 'Stop',
        'alert': '🛑 Stop! Halt your vehicle.',
        'category': 'Regulatory'
    },
    'एकतरफा रास्ता': {
        'meaning': 'One Way',
        'alert': '➡️ One Way Street. Follow traffic flow direction.',
        'category': 'Regulatory'
    },
    'प्रवेश निषेध': {
        'meaning': 'No Entry',
        'alert': '🚫 No Entry! Entry restricted for vehicles.',
        'category': 'Regulatory'
    },
    'धीरे चलें': {
        'meaning': 'Go Slow',
        'alert': '⚠️ Slow Down. Work in progress or school zone ahead.',
        'category': 'Cautionary'
    },
    'खतरा': {
        'meaning': 'Danger',
        'alert': '🔥 Danger! Hazard ahead on road.',
        'category': 'Warning'
    },
    
    # --- Tamil ---
    'வேகக் கட்டுப்பாடு': {
        'meaning': 'Speed Limit',
        'alert': '⚠️ Speed Limit area. Drive safely.',
        'category': 'Regulatory'
    },
    'நில்': {
        'meaning': 'Stop',
        'alert': '🛑 Stop! Halt.',
        'category': 'Regulatory'
    },
    'நுழைவு இல்லை': {
        'meaning': 'No Entry',
        'alert': '🚫 No Entry! Closed road.',
        'category': 'Regulatory'
    },
    'மெதுவாகச் செல்லவும்': {
        'meaning': 'Go Slow',
        'alert': '⚠️ Go Slow. School zone or bend ahead.',
        'category': 'Cautionary'
    },
    'ஆபத்து': {
        'meaning': 'Danger',
        'alert': '🔥 Danger! Drive carefully.',
        'category': 'Warning'
    },
    
    # --- English ---
    'speed limit': {
        'meaning': 'Speed Limit',
        'alert': '⚠️ Speed Limit Zone Ahead. Please monitor your speed indicator.',
        'category': 'Regulatory'
    },
    'stop': {
        'meaning': 'Stop',
        'alert': '🛑 Halt! Complete stop required before proceeding.',
        'category': 'Regulatory'
    },
    'one way': {
        'meaning': 'One Way',
        'alert': '➡️ One Way Traffic Ahead. Do not enter from opposite direction.',
        'category': 'Regulatory'
    },
    'no entry': {
        'meaning': 'No Entry',
        'alert': '🚫 No Entry! Entry is prohibited.',
        'category': 'Regulatory'
    },
    'go slow': {
        'meaning': 'Go Slow',
        'alert': '⚠️ Go Slow! Pedestrian crossing or narrow road ahead.',
        'category': 'Cautionary'
    },
    'slow down': {
        'meaning': 'Slow Down',
        'alert': '⚠️ Slow Down! Hazards or construction work ahead.',
        'category': 'Cautionary'
    },
    'danger': {
        'meaning': 'Danger',
        'alert': '🔥 Danger Zone! Exercise extreme caution.',
        'category': 'Warning'
    }
}

def analyze_road_sign(extracted_text, target_lang='en'):
    """
    Scans the extracted OCR text for known Indian road sign matches.
    If matched, returns sign details and high-visibility alerts.
    If not explicitly matched in DB, translates dynamically and suggests standard driving guidance.
    """
    text_clean = extracted_text.strip().lower()
    
    # Look for partial matches in the database
    matched_sign = None
    for key, info in ROAD_SIGNS_DB.items():
        if key.lower() in text_clean or text_clean in key.lower():
            matched_sign = {
                'detected_sign': key,
                'meaning_en': info['meaning'],
                'alert_en': info['alert'],
                'category': info['category']
            }
            break
            
    if matched_sign:
        # Translate the safety alert to the requested target language
        translated_meaning, _, _ = translate_text(matched_sign['meaning_en'], 'en', target_lang)
        translated_alert, _, _ = translate_text(matched_sign['alert_en'], 'en', target_lang)
        
        return {
            'is_known_sign': True,
            'original_sign': matched_sign['detected_sign'],
            'meaning': translated_meaning,
            'safety_alert': translated_alert,
            'category': matched_sign['category']
        }
    else:
        # Dynamic fallback translation for unknown signs
        translated_meaning, _, _ = translate_text(extracted_text, 'auto', target_lang)
        translated_alert, _, _ = translate_text("⚠️ Attention: Regulatory or informative road sign detected. Follow local traffic rules.", 'en', target_lang)
        
        return {
            'is_known_sign': False,
            'original_sign': extracted_text,
            'meaning': translated_meaning,
            'safety_alert': translated_alert,
            'category': 'General Informative'
        }

def detect_visual_sign_symbols(image_path, target_lang='en'):
    """
    Uses HSV color thresholding and contour geometry (approxPolyDP)
    to detect letterless geometric road signs:
    - Red Octagon: STOP Sign
    - Yellow/Orange Diamond/Triangle: Road Work / Caution
    - Red Ring/Circle: Regulatory / Speed Limit Border
    - Blue Circle: Direction / Mandatory
    Returns a dict with: 'detected', 'symbol_name', 'category', 'alert'
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 1. Color Segmentation Masks
        # Red has two ranges in HSV
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
        
        # Yellow/Orange (for Caution / Work signs)
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Blue (for Mandatory direction signs)
        lower_blue = np.array([100, 80, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Clean masks using morphological opening
        kernel = np.ones((5,5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        
        h, w = img.shape[:2]
        min_area = (h * w) * 0.005  # Sign must cover at least 0.5% of the image to prevent background noise
        
        # A. Check RED Contours (STOP / Regulatory circles)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if area > min_area:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.035 * perimeter, True)
                vertices = len(approx)
                
                # Stop sign is an octagon (approx 8 vertices) or approx circular red block
                if vertices == 8 or (7 <= vertices <= 9):
                    raw_alert = '🛑 Halt! Complete stop required. Watch for crossing vehicles.'
                    translated_alert, _, _ = translate_text(raw_alert, 'en', target_lang)
                    translated_meaning, _, _ = translate_text('STOP Sign Shape Detected', 'en', target_lang)
                    return {
                        'detected': True,
                        'symbol_name': 'STOP Sign Shape',
                        'category': 'Regulatory / Stop Sign',
                        'meaning': translated_meaning,
                        'alert': translated_alert
                    }
                # Circular outline (regulatory speed ring or no vehicles)
                elif vertices > 6:
                    raw_alert = '⚠️ Regulatory Restriction Zone. Observe speed limits and local vehicle bans.'
                    translated_alert, _, _ = translate_text(raw_alert, 'en', target_lang)
                    translated_meaning, _, _ = translate_text('Regulatory Circle Shape Detected', 'en', target_lang)
                    return {
                        'detected': True,
                        'symbol_name': 'Regulatory Circle / Speed Limit Sign Shape',
                        'category': 'Regulatory Restriction',
                        'meaning': translated_meaning,
                        'alert': translated_alert
                    }
                    
        # B. Check YELLOW/ORANGE Contours (Caution / Construction)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_yellow:
            area = cv2.contourArea(cnt)
            if area > min_area:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
                vertices = len(approx)
                
                # Diamond is 4 vertices, Triangle is 3 vertices
                if vertices == 4 or vertices == 3:
                    raw_alert = '⚠️ Caution Zone / Road Construction Ahead. Slow down and check for obstacles or workers.'
                    translated_alert, _, _ = translate_text(raw_alert, 'en', target_lang)
                    translated_meaning, _, _ = translate_text('Caution / Road Construction Shape Detected', 'en', target_lang)
                    return {
                        'detected': True,
                        'symbol_name': 'Caution / Construction Diamond',
                        'category': 'Cautionary / Road Work Warning',
                        'meaning': translated_meaning,
                        'alert': translated_alert
                    }
                    
        # C. Check BLUE Contours (Mandatory direction)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_blue:
            area = cv2.contourArea(cnt)
            if area > min_area:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.035 * perimeter, True)
                vertices = len(approx)
                if vertices > 5:  # Circle
                    raw_alert = '➡️ Mandatory Direction Indicator. Follow lane markings and sign arrows.'
                    translated_alert, _, _ = translate_text(raw_alert, 'en', target_lang)
                    translated_meaning, _, _ = translate_text('Mandatory Directional Shape Detected', 'en', target_lang)
                    return {
                        'detected': True,
                        'symbol_name': 'Mandatory Blue Arrow Sign Shape',
                        'category': 'Mandatory / Directional Instruction',
                        'meaning': translated_meaning,
                        'alert': translated_alert
                    }
    except Exception:
        pass
        
    return None

# --- GTSRB dataset classes (43 classes) with categories & English details ---
GTSRB_CLASSES_DB = {
    0: {
        'name': 'Speed limit (20km/h)',
        'alert': '⚠️ Speed Limit 20 km/h. Please slow down and drive extremely slowly in this zone.',
        'category': 'Regulatory / Speed Limit'
    },
    1: {
        'name': 'Speed limit (30km/h)',
        'alert': '⚠️ Speed Limit 30 km/h. Slow down and maintain a safe speed in residential or school areas.',
        'category': 'Regulatory / Speed Limit'
    },
    2: {
        'name': 'Speed limit (50km/h)',
        'alert': '⚠️ Speed Limit 50 km/h. Maintain normal city driving speed limit.',
        'category': 'Regulatory / Speed Limit'
    },
    3: {
        'name': 'Speed limit (60km/h)',
        'alert': '⚠️ Speed Limit 60 km/h. Moderate speed zone ahead. Adjust vehicle speed.',
        'category': 'Regulatory / Speed Limit'
    },
    4: {
        'name': 'Speed limit (70km/h)',
        'alert': '⚠️ Speed Limit 70 km/h. Keep to moderate speed limit on highways or sub-arterial roads.',
        'category': 'Regulatory / Speed Limit'
    },
    5: {
        'name': 'Speed limit (80km/h)',
        'alert': '⚠️ Speed Limit 80 km/h. High speed limit zone. Ensure proper lane driving.',
        'category': 'Regulatory / Speed Limit'
    },
    6: {
        'name': 'End of speed limit (80km/h)',
        'alert': '✅ End of 80 km/h Speed Limit. Resume standard highway speed regulations.',
        'category': 'Regulatory / End of Restriction'
    },
    7: {
        'name': 'Speed limit (100km/h)',
        'alert': '⚠️ Speed Limit 100 km/h. High speed zone. Maintain vehicle stability and highway safety rules.',
        'category': 'Regulatory / Speed Limit'
    },
    8: {
        'name': 'Speed limit (120km/h)',
        'alert': '⚠️ Speed Limit 120 km/h. Maximum speed limit zone. Exercise caution and do not exceed this speed.',
        'category': 'Regulatory / Speed Limit'
    },
    9: {
        'name': 'No passing',
        'alert': '🚫 No Overtaking / No Passing! Overtaking is prohibited for all vehicles.',
        'category': 'Regulatory / Prohibitory'
    },
    10: {
        'name': 'No passing for vehicles over 3.5 metric tons',
        'alert': '🚫 Heavy vehicles over 3.5 tons are prohibited from overtaking other vehicles.',
        'category': 'Regulatory / Prohibitory'
    },
    11: {
        'name': 'Right-of-way at the next intersection',
        'alert': '⚠️ Intersection Ahead. You have right-of-way, but proceed with caution.',
        'category': 'Priority / Cautionary'
    },
    12: {
        'name': 'Priority road',
        'alert': '🔷 Priority Road. Vehicles on this road have priority at intersections.',
        'category': 'Priority / Informative'
    },
    13: {
        'name': 'Yield',
        'alert': '🔻 Yield / Give Way! Slow down and prepare to stop if necessary to let other traffic pass.',
        'category': 'Regulatory / Give Way'
    },
    14: {
        'name': 'Stop',
        'alert': '🛑 Halt! Complete stop required before proceeding. Check for cross traffic.',
        'category': 'Regulatory / Stop Sign'
    },
    15: {
        'name': 'No vehicles',
        'alert': '🚫 No Vehicles Allowed! Entry closed for all vehicles in both directions.',
        'category': 'Regulatory / Prohibitory'
    },
    16: {
        'name': 'Vehicles over 3.5 metric tons prohibited',
        'alert': '🚫 Heavy vehicles exceeding 3.5 metric tons are prohibited from entering.',
        'category': 'Regulatory / Prohibitory'
    },
    17: {
        'name': 'No entry',
        'alert': '🚫 No Entry! Entry is strictly prohibited for all vehicles.',
        'category': 'Regulatory / Prohibitory'
    },
    18: {
        'name': 'General caution',
        'alert': '⚠️ General Caution! Unspecified danger ahead on road. Stay vigilant.',
        'category': 'Cautionary'
    },
    19: {
        'name': 'Dangerous curve to the left',
        'alert': '↩️ Sharp left curve ahead. Slow down and stay in your lane.',
        'category': 'Cautionary'
    },
    20: {
        'name': 'Dangerous curve to the right',
        'alert': '↪️ Sharp right curve ahead. Slow down and negotiate the curve carefully.',
        'category': 'Cautionary'
    },
    21: {
        'name': 'Double curve',
        'alert': '🔄 Double Curve Ahead (first to the left then right). Reduce speed.',
        'category': 'Cautionary'
    },
    22: {
        'name': 'Bumpy road',
        'alert': '⚠️ Bumpy / Uneven Road Ahead. Reduce speed to prevent vehicle damage.',
        'category': 'Cautionary'
    },
    23: {
        'name': 'Slippery road',
        'alert': '🌧️ Slippery Road Conditions Ahead. Drive carefully and avoid sudden braking.',
        'category': 'Cautionary'
    },
    24: {
        'name': 'Road narrows on the right',
        'alert': '⚠️ Road Narrows on Right Side. Watch for oncoming traffic and merge safely.',
        'category': 'Cautionary'
    },
    25: {
        'name': 'Road work',
        'alert': '🚧 Road Work / Construction Ahead. Watch for workers, equipment, and detours.',
        'category': 'Cautionary / Road Work'
    },
    26: {
        'name': 'Traffic signals',
        'alert': '🚦 Traffic Signals Ahead. Be prepared to stop or obey signal lights.',
        'category': 'Cautionary / Signals'
    },
    27: {
        'name': 'Pedestrians',
        'alert': '🚶 Pedestrian Crossing Ahead. Watch for pedestrians and yield right of way.',
        'category': 'Cautionary / Pedestrian'
    },
    28: {
        'name': 'Children crossing',
        'alert': '🚸 School Zone / Children Crossing Ahead. Drive slow and look out for children.',
        'category': 'Cautionary / School'
    },
    29: {
        'name': 'Bicycles crossing',
        'alert': '🚴 Bicycle Crossing Ahead. Watch for cyclists on or near the road.',
        'category': 'Cautionary'
    },
    30: {
        'name': 'Beware of ice/snow',
        'alert': '❄️ Ice or Snow Warning. Road may be highly slippery. Drive with maximum care.',
        'category': 'Cautionary'
    },
    31: {
        'name': 'Wild animals crossing',
        'alert': '🦌 Wild Animals Crossing Zone. Be prepared for sudden animal movements.',
        'category': 'Cautionary'
    },
    32: {
        'name': 'End of all speed and passing limits',
        'alert': '✅ End of all Speed and Overtaking Limits. Follow general traffic rules.',
        'category': 'Regulatory / End of Restriction'
    },
    33: {
        'name': 'Turn right ahead',
        'alert': '➡️ Mandatory Turn Right Ahead. Follow the arrows.',
        'category': 'Mandatory / Directional'
    },
    34: {
        'name': 'Turn left ahead',
        'alert': '⬅️ Mandatory Turn Left Ahead. Follow the arrows.',
        'category': 'Mandatory / Directional'
    },
    35: {
        'name': 'Ahead only',
        'alert': '⬆️ Mandatory Straight Ahead Only. Do not turn right or left.',
        'category': 'Mandatory / Directional'
    },
    36: {
        'name': 'Go straight or right',
        'alert': '↗️ Mandatory Go Straight or Right. Turning left is prohibited.',
        'category': 'Mandatory / Directional'
    },
    37: {
        'name': 'Go straight or left',
        'alert': '↖️ Mandatory Go Straight or Left. Turning right is prohibited.',
        'category': 'Mandatory / Directional'
    },
    38: {
        'name': 'Keep right',
        'alert': '➡️ Keep Right! Pass the obstacle on the right side.',
        'category': 'Mandatory / Directional'
    },
    39: {
        'name': 'Keep left',
        'alert': '⬅️ Keep Left! Pass the obstacle on the left side.',
        'category': 'Mandatory / Directional'
    },
    40: {
        'name': 'Roundabout mandatory',
        'alert': '🔄 Mandatory Roundabout / Rotary. Give way to vehicles already in the circle.',
        'category': 'Mandatory / Directional'
    },
    41: {
        'name': 'End of no passing',
        'alert': '✅ End of Overtaking Ban. Normal passing permitted when safe.',
        'category': 'Regulatory / End of Restriction'
    },
    42: {
        'name': 'End of no passing by vehicles over 3.5 metric tons',
        'alert': '✅ End of Overtaking Ban for Heavy Vehicles. Normal rules apply.',
        'category': 'Regulatory / End of Restriction'
    }
}

# --- PyTorch Custom Keras Model Loader ---
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
        # x shape is (batch, 3, 30, 30)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # Permute to channels-last before reshape to match Keras Row-Major flatten exactly
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def load_weights_from_keras_h5(pytorch_model, h5_path):
    """
    Extracts Keras weight datasets from the H5 file and adapts/loads them into our PyTorch layers.
    Transposes:
      - Keras Convolution Kernels (H, W, InC, OutC) -> PyTorch (OutC, InC, H, W)
      - Keras Dense Kernels (InFeatures, OutFeatures) -> PyTorch (OutFeatures, InFeatures)
    """
    with h5py.File(h5_path, 'r') as f:
        # Load conv1
        k_conv1 = f['model_weights/conv2d/conv2d/kernel:0'][:]
        b_conv1 = f['model_weights/conv2d/conv2d/bias:0'][:]
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
        pytorch_model.fc1.weight.data = torch.from_numpy(k_fc1.transpose(1, 0))
        pytorch_model.fc1.bias.data = torch.from_numpy(b_fc1)
        
        # Load fc2
        k_fc2 = f['model_weights/dense_1/dense_1/kernel:0'][:]
        b_fc2 = f['model_weights/dense_1/dense_1/bias:0'][:]
        pytorch_model.fc2.weight.data = torch.from_numpy(k_fc2.transpose(1, 0))
        pytorch_model.fc2.bias.data = torch.from_numpy(b_fc2)

def predict_road_sign_cnn(image_path, target_lang='en'):
    """
    Classifies the road sign using the pre-trained CNN model via PyTorch.

    Strategy:
      1. Try to auto-crop the sign region using per-channel HSV segmentation
         (Red, Blue, Yellow/Orange) with contour area and aspect-ratio filters.
      2. Run the CNN on both the full image and the cropped region.
      3. Return the result with higher softmax confidence (dual-prediction voting).
      4. Translate output via AWS Translate.
    """
    try:
        # --- locate model ---
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir  = os.path.dirname(curr_dir)
        h5_path   = os.path.join(
            root_dir,
            'Traffic-Signs-Recognition-using-CNN-Keras-main',
            'models',
            'best_model.h5'
        )
        if not os.path.exists(h5_path):
            return {
                'detected':    False,
                'symbol_name': 'Model Not Found',
                'category':    'System Error',
                'meaning':     'Model file not found.',
                'alert':       'Please ensure best_model.h5 is present in the models directory.',
                'confidence':  0.0,
            }

        # --- load model ---
        model = TrafficSignCNN()
        load_weights_from_keras_h5(model, h5_path)
        model.eval()

        # --- load image (BGR, raw [0,255] — same as training) ---
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            pil_img   = Image.open(image_path).convert('RGB')
            image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # --- helper: resize to 30×30 and run one forward pass ---
        def _infer(img_bgr):
            arr    = cv2.resize(img_bgr, (30, 30)).astype(np.float32)
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
            with torch.no_grad():
                out  = model(tensor)
                conf = float(torch.max(out).item())
                cls  = int(torch.argmax(out, dim=1).item())
            return cls, conf

        # --- helper: HSV auto-crop using per-channel colour segmentation ---
        def _extract_sign_roi(img_bgr):
            img_h, img_w = img_bgr.shape[:2]
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            # Red hue wraps at 180, so use two ranges
            m_r = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0,   50, 50]), np.array([20,  255, 255])),
                cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255])),
            )
            # Blue (mandatory direction signs)
            m_b = cv2.inRange(hsv, np.array([90,  50, 50]), np.array([140, 255, 255]))
            # Yellow / Orange (caution, priority)
            m_y = cv2.inRange(hsv, np.array([20,  50, 50]), np.array([40,  255, 255]))

            kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            best_cnt   = None
            best_area  = 0

            for mask in (m_r, m_b, m_y):
                clean  = cv2.morphologyEx(mask,  cv2.MORPH_CLOSE, kernel)
                clean  = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  kernel)
                cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if area < 100:
                        continue
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    aspect = float(cw) / max(ch, 1)
                    # Square-ish signs; must not cover > 20 % of the image
                    if 0.65 <= aspect <= 1.5 and area < 0.20 * img_w * img_h:
                        if area > best_area:
                            best_area = area
                            best_cnt  = cnt

            if best_cnt is not None:
                x, y, cw, ch = cv2.boundingRect(best_cnt)
                # Skip crop when sign already fills most of the image
                if cw * ch <= 0.80 * img_w * img_h:
                    pw = max(int(cw * 0.15), 3)
                    ph = max(int(ch * 0.15), 3)
                    x1 = max(0,     x  - pw)
                    y1 = max(0,     y  - ph)
                    x2 = min(img_w, x + cw + pw)
                    y2 = min(img_h, y + ch + ph)
                    return img_bgr[y1:y2, x1:x2]

            return img_bgr   # fallback: full image

        # --- dual-prediction: full image vs. auto-cropped region ---
        cls_full,  conf_full  = _infer(image_bgr)
        roi                   = _extract_sign_roi(image_bgr)
        cls_crop,  conf_crop  = _infer(roi)

        # Pick whichever prediction is more confident
        if conf_crop >= conf_full:
            predicted_class, confidence = cls_crop, conf_crop
        else:
            predicted_class, confidence = cls_full, conf_full

        # --- compute top-3 predictions from full image for richer UI feedback ---
        arr_full   = cv2.resize(image_bgr, (30, 30)).astype(np.float32)
        t_full     = torch.from_numpy(arr_full.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            probs_full = model(t_full)[0]          # shape (43,)
        top3_vals, top3_idx = torch.topk(probs_full, 3)
        top3 = [
            {
                'class_id':    int(top3_idx[i].item()),
                'name':        GTSRB_CLASSES_DB.get(int(top3_idx[i].item()), {}).get('name', 'Unknown'),
                'confidence':  float(top3_vals[i].item()),
            }
            for i in range(3)
        ]

        # --- confidence thresholding ---
        # The CNN is trained on 43 GTSRB classes only.
        # Signs outside these 43 classes (e.g. cattle crossing, school zone, Indian-specific
        # signs) will produce low softmax confidence — the model is "unsure".
        # Thresholds:
        #   ≥ 0.70  → HIGH    — show the sign result clearly
        #   0.40–0.69 → MEDIUM — show the result with an uncertainty warning
        #   < 0.40  → LOW     — unknown/non-standard sign; avoid showing a wrong answer
        if confidence >= 0.70:
            confidence_tier = 'high'
        elif confidence >= 0.40:
            confidence_tier = 'medium'
        else:
            confidence_tier = 'low'

        # For low confidence, report as unknown instead of giving a wrong answer
        if confidence_tier == 'low':
            unknown_msg = (
                'This does not appear to be one of the 43 standard road signs this model was '
                'trained on (GTSRB dataset). It may be a country-specific sign such as a '
                'cattle crossing, school zone, or local informational board. '
                'Please refer to local traffic authority guidelines.'
            )
            translated_unknown, _, _ = translate_text(unknown_msg, 'en', target_lang)
            translated_alert_uk, _, _ = translate_text(
                '⚠️ Non-standard or unknown road sign. Drive cautiously and consult local traffic rules.',
                'en', target_lang
            )
            return {
                'detected':         True,
                'is_known':         False,
                'class_id':         predicted_class,
                'symbol_name':      'Unknown / Non-standard Sign',
                'category':         'Unknown',
                'meaning':          translated_unknown,
                'alert':            translated_alert_uk,
                'confidence':       confidence,
                'confidence_tier':  'low',
                'top3':             top3,
            }

        # --- look up sign details ---
        sign_info = GTSRB_CLASSES_DB.get(predicted_class, {
            'name':     'Unknown Road Sign',
            'alert':    '⚠️ Unknown or unrecognized road sign. Please drive carefully.',
            'category': 'Unknown',
        })

        # --- translate via AWS Translate ---
        translated_meaning,  _, _ = translate_text(sign_info['name'],     'en', target_lang)
        translated_alert,    _, _ = translate_text(sign_info['alert'],    'en', target_lang)
        translated_category, _, _ = translate_text(sign_info['category'], 'en', target_lang)

        return {
            'detected':         True,
            'is_known':         True,
            'class_id':         predicted_class,
            'symbol_name':      sign_info['name'],
            'category':         translated_category,
            'meaning':          translated_meaning,
            'alert':            translated_alert,
            'confidence':       confidence,
            'confidence_tier':  confidence_tier,
            'top3':             top3,
        }

    except Exception as e:
        return {
            'detected':    False,
            'symbol_name': 'Error',
            'category':    'Inference Error',
            'meaning':     f'CNN inference failed: {e}',
            'alert':       'Please check system dependencies and model files.',
            'confidence':  0.0,
        }
