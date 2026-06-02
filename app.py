import os
import sqlite3
import hashlib
import base64
import time
import streamlit as st
from PIL import Image

# Import BhashaBridge modular configurations and tools
from config import init_db, SUPPORTED_LANGUAGES, DB_PATH, get_aws_client, BASE_DIR
from modules.text_translation import translate_text, transliterate_text, google_transliterate
from modules.grammar_corrector import correct_grammar
from modules.image_translation import perform_ocr, overlay_translation
from modules.road_sign_detector import analyze_road_sign, detect_visual_sign_symbols, predict_road_sign_cnn
from modules.audio_translation import transcribe_audio_file, text_to_speech
from modules.video_translation import generate_subtitles
from modules.utils import (
    save_translation_history, 
    get_translation_history, 
    clear_translation_history, 
    clean_directories, 
    check_gpu_support
)

# Initialize DB on load
init_db()

# Page configuration
st.set_page_config(
    page_title="BhashaBridge AI - Indian Language Multimodal Translator",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ACCESSIBILITY STATE INITS ---
if "accessibility" not in st.session_state: st.session_state.accessibility = False
if "high_contrast" not in st.session_state: st.session_state.high_contrast = False
if "large_text" not in st.session_state: st.session_state.large_text = False
if "tts_auto_play" not in st.session_state: st.session_state.tts_auto_play = False
if "auth" not in st.session_state: st.session_state.auth = True  # Auto login for MCA Demo convenience

# --- CSS STYLING INJECTORS ---
def inject_styles():
    # Detect theme state
    large_text = st.session_state.large_text
    high_contrast = st.session_state.high_contrast
    accessibility = st.session_state.accessibility
    
    font_size_main = "1.3rem" if large_text else "0.95rem"
    font_size_h1 = "3rem" if large_text else "2.2rem"
    font_size_h2 = "2rem" if large_text else "1.5rem"
    font_size_h3 = "1.6rem" if large_text else "1.15rem"
    btn_padding = "1rem 2rem" if large_text else "0.6rem 1.5rem"
    
    if accessibility and high_contrast:
        # High Contrast Yellow on Black Theme for Vision Impairment
        st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
        
        .stApp {{
            background: #000000 !important;
            color: #FFFF00 !important;
            font-family: 'Inter', sans-serif !important;
        }}
        
        div.stButton > button, div.stDownloadButton > button {{
            background-color: #FFFF00 !important;
            color: #000000 !important;
            border: 3px solid #FFFFFF !important;
            font-weight: 900 !important;
            font-size: {font_size_main} !important;
            border-radius: 0px !important;
            padding: {btn_padding} !important;
            text-transform: uppercase !important;
        }}
        
        .nexus-card {{
            background-color: #000000 !important;
            border: 4px solid #FFFF00 !important;
            padding: 2rem;
            margin-bottom: 1.5rem;
        }}
        
        h1, h2, h3, h4, h5, h6, p, label, span, li {{
            color: #FFFF00 !important;
            font-size: {font_size_main} !important;
        }}
        h1 {{ font-size: {font_size_h1} !important; }}
        h2 {{ font-size: {font_size_h2} !important; }}
        h3 {{ font-size: {font_size_h3} !important; }}
        
        .stTextArea textarea, .stTextInput input {{
            background-color: #000000 !important;
            color: #FFFF00 !important;
            border: 2px solid #FFFF00 !important;
            font-size: {font_size_main} !important;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        # Futuristic Dark Mode Glassmorphism Theme
        st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Inter:wght@300;400;500;700&display=swap');
        
        .stApp {{
            background: radial-gradient(circle at top right, #1a1a2e, #0d0d18, #05050a);
            color: #f1f1f6;
            font-family: 'Inter', sans-serif;
            font-size: {font_size_main};
        }}
        
        /* Glassmorphism Containers */
        .nexus-card {{
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 20px;
            padding: 2.2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.45);
            transition: all 0.3s ease;
        }}
        
        .nexus-card:hover {{
            border-color: rgba(0, 242, 254, 0.3);
            box-shadow: 0 10px 40px 0 rgba(0, 242, 254, 0.1);
        }}
        
        /* Neon Highlights and Text */
        h1, h2, h3 {{
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 2px;
            margin-bottom: 1.5rem;
        }}
        h1 {{ font-size: {font_size_h1}; }}
        h2 {{ font-size: {font_size_h2}; }}
        h3 {{ font-size: {font_size_h3}; }}
        
        /* Premium Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: rgba(6, 6, 12, 0.98) !important;
            border-right: 1px solid rgba(0, 242, 254, 0.15);
        }}
        
        /* Glowing Buttons */
        div.stButton > button, div.stDownloadButton > button {{
            background: linear-gradient(45deg, #00f2fe 0%, #4facfe 100%) !important;
            color: #000000 !important;
            font-family: 'Orbitron', sans-serif !important;
            font-weight: 700 !important;
            padding: {btn_padding} !important;
            border-radius: 50px !important;
            border: none !important;
            font-size: 0.9rem !important;
            letter-spacing: 1px !important;
            transition: all 0.4s ease !important;
            box-shadow: 0 4px 15px rgba(0, 242, 254, 0.2) !important;
            text-transform: uppercase !important;
            width: 100% !important;
        }}
        
        div.stButton > button:hover, div.stDownloadButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 0 25px #00f2fe !important;
            color: #ffffff !important;
        }}
        
        /* Textareas and Text Inputs */
        .stTextArea textarea, .stTextInput input {{
            background-color: rgba(255, 255, 255, 0.03) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            padding: 10px !important;
            font-size: {font_size_main} !important;
        }}
        
        .stTextArea textarea:focus, .stTextInput input:focus {{
            border-color: #00f2fe !important;
            box-shadow: 0 0 10px rgba(0, 242, 254, 0.3) !important;
        }}
        
        /* Status Elements */
        .stAlert {{
            border-radius: 12px !important;
            background-color: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(0, 242, 254, 0.2) !important;
            color: #ffffff !important;
        }}
        </style>
        """, unsafe_allow_html=True)

# Inject CSS layout rules
inject_styles()

def play_audio_helper(text, target_lang):
    """
    Synthesizes speech and outputs an HTML5 autoplay audio player inside the app.
    """
    try:
        audio_bytes = text_to_speech(text, target_lang, use_aws=True)
        if audio_bytes:
            b64 = base64.b64encode(audio_bytes).decode()
            md = f"""
                <audio controls autoplay="true" style="width: 100%; margin-top: 10px;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)
            st.success("🗣️ Speech synthesized successfully!")
        else:
            st.error("❌ Audio synthesis failed.")
    except Exception as e:
        st.error(f"❌ Sound Synthesis Error: {e}")

# --- LOGIN GATEWAY ---
if not st.session_state.auth:
    st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
    st.title("🔐 BHASHABRIDGE GATEWAY")
    tab1, tab2 = st.tabs(["🔒 Secure Login", "📝 Create Identity"])
    
    with tab1:
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")
        if st.button("AUTHENTICATE"):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            hp = hashlib.sha256(p.encode()).hexdigest()
            cursor.execute("SELECT * FROM users WHERE user=? AND pw=?", (u, hp))
            user_exists = cursor.fetchone()
            conn.close()
            if user_exists or (u == "admin" and p == "admin"): # MCA convenience pass
                st.session_state.auth = True
                st.success("Access Granted. Deploying core modules...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid credentials.")
                
    with tab2:
        nu = st.text_input("New Username", key="reg_u")
        np = st.text_input("New Password", type="password", key="reg_p")
        if st.button("CREATE SECURE IDENTITY"):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            hp = hashlib.sha256(np.encode()).hexdigest()
            try:
                cursor.execute("INSERT INTO users VALUES (?,?)", (nu, hp))
                conn.commit()
                st.success("Identity registered successfully. You may login.")
            except Exception:
                st.error("Username already registered.")
            finally:
                conn.close()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # --- MAIN PORTAL ---
    # Sidebar
    st.sidebar.markdown("<h2 style='text-align: center; margin: 0;'>💠 BHASHA BRIDGE</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center; color: #00f2fe; font-size: 0.8rem; margin-bottom: 2rem;'>MCA Multimodal Cloud/AI Project</p>", unsafe_allow_html=True)
    
    # Navigation
    nav = st.sidebar.radio(
        "NAVIGATION NODE", 
        [
            "🌐 Text Translation", 
            "👁️ Image OCR & Translation", 
            "🚏 Road Sign Scanner",
            "🔊 Speech & Audio Node", 
            "🎬 Video Subtitle Engine", 
            "♿ Accessibility Hub",
            "ℹ️ Tourist & Travel Helper",
            "📊 History & Diagnostics"
        ]
    )
    
    st.sidebar.markdown("---")
    # Quick Logoff
    if st.sidebar.button("🔒 EXIT GATEWAY"):
        st.session_state.auth = False
        st.rerun()

    # --- 1. TEXT TRANSLATION MODULE ---
    if nav == "🌐 Text Translation":
        st.title("🌐 NEURAL TEXT TRANSLATION")
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            src_lang = st.selectbox("Source Language", ["auto"] + list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: "Auto Detect" if x == "auto" else SUPPORTED_LANGUAGES[x])
        with col2:
            target_lang = st.selectbox("Target Language", list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x], index=1) # Default to Hindi
        
        # Select Translation Engine Mode
        translation_mode = st.radio(
            "Translation Mode",
            ["neural_translate", "phonetic_transliterate"],
            format_func=lambda x: {
                "neural_translate": "🌐 Contextual Neural Translation (Translates Meaning, e.g. 'how are you' -> 'आप कैसे हैं?')",
                "phonetic_transliterate": "🔤 Phonetic Transliteration / Hinglish Support (Converts sounds to regional script, e.g. 'hi bhai' -> 'हाय भाई')"
            }[x],
            horizontal=True
        )
        
        # User input text
        original_text = st.text_area("Input Text", height=150, placeholder="Type regional phrases or English text here...")
        
        # Grammar corrector checkbox
        correct_check = st.checkbox("⚙️ Smart Auto-Grammar Correction")
        
        # Initialize Text Translation Session State variables individually if not present
        if 'txt_active_input' not in st.session_state:
            st.session_state.txt_active_input = None
        if 'txt_active_src_lang' not in st.session_state:
            st.session_state.txt_active_src_lang = None
        if 'txt_active_target_lang' not in st.session_state:
            st.session_state.txt_active_target_lang = None
        if 'txt_active_mode' not in st.session_state:
            st.session_state.txt_active_mode = None
        if 'txt_active_correct' not in st.session_state:
            st.session_state.txt_active_correct = None
        if 'txt_translated' not in st.session_state:
            st.session_state.txt_translated = None
        if 'txt_detected_lang' not in st.session_state:
            st.session_state.txt_detected_lang = None
        if 'txt_confidence' not in st.session_state:
            st.session_state.txt_confidence = None
            
        # Reset caches if any inputs change
        state_changed = (
            st.session_state.txt_active_input != original_text or
            st.session_state.txt_active_src_lang != src_lang or
            st.session_state.txt_active_target_lang != target_lang or
            st.session_state.txt_active_mode != translation_mode or
            st.session_state.txt_active_correct != correct_check
        )
        if state_changed:
            st.session_state.txt_active_input = original_text
            st.session_state.txt_active_src_lang = src_lang
            st.session_state.txt_active_target_lang = target_lang
            st.session_state.txt_active_mode = translation_mode
            st.session_state.txt_active_correct = correct_check
            st.session_state.txt_translated = None
            st.session_state.txt_detected_lang = None
            st.session_state.txt_confidence = None
            
        # Trigger Translation execution
        if st.button("EXECUTE TRANSLATION") or st.session_state.txt_translated is not None:
            if original_text.strip():
                if st.session_state.txt_translated is None:
                    with st.spinner("Processing neural link..."):
                        if translation_mode == "phonetic_transliterate":
                            # Perform sound-to-script transliteration
                            transliterated = google_transliterate(original_text, target_lang)
                            st.session_state.txt_translated = transliterated
                            st.session_state.txt_detected_lang = src_lang
                            st.session_state.txt_confidence = 1.0
                        else:
                            # Step 1: Grammar check if requested
                            cleaned_text = original_text
                            grammar_details = []
                            if correct_check and src_lang in ['auto', 'en']:
                                cleaned_text, grammar_details = correct_grammar(original_text)
                                if grammar_details:
                                    st.info(f"✨ Grammar corrected text: {cleaned_text}")
                                    
                            # Step 2: Translate
                            translated, detected, confidence = translate_text(cleaned_text, src_lang, target_lang)
                            
                            st.session_state.txt_translated = translated
                            st.session_state.txt_detected_lang = detected
                            st.session_state.txt_confidence = confidence
                
                # Render results persistently
                if st.session_state.txt_translated is not None:
                    translated = st.session_state.txt_translated
                    detected = st.session_state.txt_detected_lang
                    confidence = st.session_state.txt_confidence
                    
                    if translation_mode == "phonetic_transliterate":
                        st.success(f"🎯 Phonetic Transliteration completed successfully!")
                    else:
                        st.success(f"🎯 Detected Source: {SUPPORTED_LANGUAGES.get(detected, 'Unknown')} | Confidence: {confidence:.2f}")
                        
                    st.text_area("Translated Output", value=translated, height=150, key="txt_out")
                    
                    # Action buttons
                    sub_col1, sub_col2, sub_col3 = st.columns(3)
                    with sub_col1:
                        # Pronounce button
                        if st.button("🗣️ Narrate Translation"):
                            play_audio_helper(translated, target_lang)
                    with sub_col2:
                        st.download_button("📥 Download Text", data=translated, file_name="translated_text.txt")
                    with sub_col3:
                        # Transliteration phonetic guide
                        if st.button("🔤 Transliterate Text"):
                            pronun = transliterate_text(original_text, target_lang)
                            st.info(f"Phonetic Guide: {pronun}")
            else:
                st.warning("Please enter text to translate.")
                
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 2. IMAGE OCR & TRANSLATION ---
    elif nav == "👁️ Image OCR & Translation":
        st.title("👁️ NEURAL IMAGE OCR & TRANSLATION")
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        
        ocr_source = st.radio("Select Scan Source", ["Upload Image File", "Real-Time Camera Capture"])
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            source_lang = st.selectbox(
                "Source Script in Image", 
                ["hi", "en", "kn", "ta", "te", "ml", "mr", "bn", "auto_multi"],
                format_func=lambda x: {
                    "hi": "Hindi (Devanagari)",
                    "en": "English Only",
                    "kn": "Kannada",
                    "ta": "Tamil",
                    "te": "Telugu",
                    "ml": "Malayalam",
                    "mr": "Marathi",
                    "bn": "Bengali",
                    "auto_multi": "Auto-Detect / Multi-Script"
                }[x],
                index=0  # Default to Hindi Devanagari!
            )
        with col_s2:
            target_lang = st.selectbox("Translate OCR Text to", list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x], index=0)
            
        # Add visual optimization toggle
        ocr_mode = st.selectbox(
            "Image Capture Type (Optimizes Text Contrast)",
            ["scanned_doc", "scene_photo"],
            format_func=lambda x: {
                "scanned_doc": "📄 Digital Document / Clean Scan (No Preprocessing Noise)",
                "scene_photo": "📸 Outdoor Scene Photo / Road Sign (Shadow & Contrast Boost)"
            }[x],
            index=0 # Default to scanned document! Perfect for clean poetry sheets
        )
            
        # Initialize OCR Session State variables individually if not present to support hot-reloads cleanly
        if 'ocr_active_file' not in st.session_state:
            st.session_state.ocr_active_file = None
        if 'ocr_active_source_lang' not in st.session_state:
            st.session_state.ocr_active_source_lang = None
        if 'ocr_active_target_lang' not in st.session_state:
            st.session_state.ocr_active_target_lang = None
        if 'ocr_active_ocr_mode' not in st.session_state:
            st.session_state.ocr_active_ocr_mode = None
        if 'ocr_results' not in st.session_state:
            st.session_state.ocr_results = None
        if 'ocr_full_extracted_text' not in st.session_state:
            st.session_state.ocr_full_extracted_text = None
        if 'ocr_translated_ocr' not in st.session_state:
            st.session_state.ocr_translated_ocr = None
        if 'ocr_overlay_path' not in st.session_state:
            st.session_state.ocr_overlay_path = None
            
        img_file = None
        if ocr_source == "Upload Image File":
            img_file = st.file_uploader("Scan Visual Data (PNG/JPG)", type=["png", "jpg", "jpeg"])
        else:
            img_file = st.camera_input("Scanner Camera")
            
        if img_file:
            # Generate a unique key for the active file
            if hasattr(img_file, 'name'):
                file_key = f"{img_file.name}_{img_file.size}"
            else:
                file_key = f"camera_{len(img_file.getvalue())}"
                
            # If the user changed the file, source script, target language, or ocr mode, clear previous session state
            state_changed = (
                st.session_state.ocr_active_file != file_key or
                st.session_state.ocr_active_source_lang != source_lang or
                st.session_state.ocr_active_target_lang != target_lang or
                st.session_state.ocr_active_ocr_mode != ocr_mode
            )
            if state_changed:
                st.session_state.ocr_active_file = file_key
                st.session_state.ocr_active_source_lang = source_lang
                st.session_state.ocr_active_target_lang = target_lang
                st.session_state.ocr_active_ocr_mode = ocr_mode
                st.session_state.ocr_results = None
                st.session_state.ocr_full_extracted_text = None
                st.session_state.ocr_translated_ocr = None
                st.session_state.ocr_overlay_path = None
                
            # Save visual image locally for processing
            temp_path = os.path.join(BASE_DIR, 'temp', f"uploaded_scan_{file_key}.jpg")
            if not os.path.exists(os.path.dirname(temp_path)):
                os.makedirs(os.path.dirname(temp_path))
            
            with open(temp_path, "wb") as f:
                f.write(img_file.getvalue())
                
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(img_file, caption="Captured Image Preview", use_container_width=True)
                
            # Trigger OCR execution
            if st.button("EXECUTE VISUAL OCR") or st.session_state.ocr_results is not None:
                if st.session_state.ocr_results is None:
                    with st.spinner("Extracting visual glyphs & text..."):
                        # Perform OCR with the advanced layout analysis and preprocessor
                        if source_lang == "auto_multi":
                            lang_list = ['en', 'hi']
                        else:
                            lang_list = ['en', source_lang]
                            
                        ocr_results = perform_ocr(temp_path, use_engine='easyocr', lang_list=lang_list, ocr_mode=ocr_mode)
                        
                        if ocr_results:
                            full_extracted_text = " ".join([r['text'] for r in ocr_results])
                            overlay_path = overlay_translation(temp_path, ocr_results, target_lang)
                            
                            # Normal contextual sentence-level translation
                            translated_ocr, _, _ = translate_text(full_extracted_text, 'auto', target_lang)
                            
                            # Cache in session state
                            st.session_state.ocr_results = ocr_results
                            st.session_state.ocr_full_extracted_text = full_extracted_text
                            st.session_state.ocr_translated_ocr = translated_ocr
                            if os.path.exists(overlay_path):
                                st.session_state.ocr_overlay_path = overlay_path
                        else:
                            st.session_state.ocr_results = []
                
                # Render results if OCR succeeded
                if st.session_state.ocr_results:
                    full_extracted_text = st.session_state.ocr_full_extracted_text
                    translated_ocr = st.session_state.ocr_translated_ocr
                    overlay_path = st.session_state.ocr_overlay_path
                    
                    with col_img2:
                        st.info(f"📝 Raw OCR Extracted Text:\n\n`{full_extracted_text}`")
                        st.success(f"🎯 Translated OCR Text ({SUPPORTED_LANGUAGES[target_lang]}):\n\n{translated_ocr}")
                            
                        # Audio narration
                        if st.button("🗣️ Read Aloud OCR Translation", key="btn_ocr_audio"):
                            play_audio_helper(translated_ocr, target_lang)
                            
                    # Display translation overlay
                    if overlay_path and os.path.exists(overlay_path):
                        st.markdown("### 🖼️ TRANSLATION VISUAL OVERLAY")
                        st.image(overlay_path, caption="Translated Text Bounding Box Overlay", use_container_width=True)
                        with open(overlay_path, "rb") as file:
                            st.download_button(
                                label="📥 Download Translated Image",
                                data=file,
                                file_name=f"translated_overlay_{os.path.basename(temp_path)}",
                                mime="image/jpeg",
                                key="btn_download_overlay"
                            )
                elif st.session_state.ocr_results == []:
                    st.error("No clear textual data could be identified in the image.")
                    
            # Clean up temp image
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
                
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 2.5 ROAD SIGN SCANNER ---
    elif nav == "🚏 Road Sign Scanner":
        st.title("🚏 SMART ROAD SIGN SCANNER")
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        
        sign_source = st.radio("Select Sign Scan Source", ["Upload Sign Image", "Real-Time Camera Capture"])
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            source_lang = st.selectbox(
                "Source Regional Script", 
                ["hi", "en", "kn", "ta", "te", "ml", "mr", "bn", "auto_multi"],
                format_func=lambda x: {
                    "hi": "Hindi (Devanagari)",
                    "en": "English Only",
                    "kn": "Kannada",
                    "ta": "Tamil",
                    "te": "Telugu",
                    "ml": "Malayalam",
                    "mr": "Marathi",
                    "bn": "Bengali",
                    "auto_multi": "Auto-Detect / Multi-Script"
                }[x],
                index=0  # Default to Hindi Devanagari!
            )
        with col_s2:
            target_lang = st.selectbox("Translate Sign Info to", list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x], index=0)
            
        # Initialize Sign Session State variables individually if not present to support hot-reloads cleanly
        if 'sign_active_file' not in st.session_state:
            st.session_state.sign_active_file = None
        if 'sign_active_source_lang' not in st.session_state:
            st.session_state.sign_active_source_lang = None
        if 'sign_active_target_lang' not in st.session_state:
            st.session_state.sign_active_target_lang = None
        if 'sign_active_mode' not in st.session_state:
            st.session_state.sign_active_mode = None
        if 'sign_results' not in st.session_state:
            st.session_state.sign_results = None
        if 'sign_visual_match' not in st.session_state:
            st.session_state.sign_visual_match = None
        if 'sign_cnn_match' not in st.session_state:
            st.session_state.sign_cnn_match = None
        if 'sign_full_extracted_text' not in st.session_state:
            st.session_state.sign_full_extracted_text = None
        if 'sign_road_sign_analysis' not in st.session_state:
            st.session_state.sign_road_sign_analysis = None
        if 'sign_overlay_path' not in st.session_state:
            st.session_state.sign_overlay_path = None
            
        # Unified mode — CNN always runs (handles all sign types, text or no text)
        # OCR supplements only when text is also visible on the sign
        st.info("🧠 **AI CNN Classifier** detects ALL road signs automatically — signs with or without text, symbols, arrows, speed limits, stop, yield, and more. No selection needed.")
        sign_mode = "cnn_model"  # Always use CNN as the primary engine
            
        img_file = None
        if sign_source == "Upload Sign Image":
            img_file = st.file_uploader("Upload Road Sign Photo (PNG/JPG)", type=["png", "jpg", "jpeg"])
        else:
            img_file = st.camera_input("Sign Camera Scanner")
            
        if img_file:
            # Generate a unique key for the active file
            if hasattr(img_file, 'name'):
                file_key = f"{img_file.name}_{img_file.size}"
            else:
                file_key = f"camera_sign_{len(img_file.getvalue())}"
                
            # If the user changed inputs, clear sign session cache
            state_changed = (
                st.session_state.sign_active_file != file_key or
                st.session_state.sign_active_source_lang != source_lang or
                st.session_state.sign_active_target_lang != target_lang
            )
            if state_changed:
                st.session_state.sign_active_file = file_key
                st.session_state.sign_active_source_lang = source_lang
                st.session_state.sign_active_target_lang = target_lang
                st.session_state.sign_active_mode = sign_mode
                st.session_state.sign_results = None
                st.session_state.sign_visual_match = None
                st.session_state.sign_cnn_match = None
                st.session_state.sign_full_extracted_text = None
                st.session_state.sign_road_sign_analysis = None
                st.session_state.sign_overlay_path = None
                
            # Save visual image locally for processing
            temp_path = os.path.join(BASE_DIR, 'temp', f"uploaded_sign_{file_key}.jpg")
            if not os.path.exists(os.path.dirname(temp_path)):
                os.makedirs(os.path.dirname(temp_path))
            
            with open(temp_path, "wb") as f:
                f.write(img_file.getvalue())
                
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(img_file, caption="Target Road Sign", use_container_width=True)
                
            # Trigger Sign execution
            execution_triggered = (
                st.button("SCAN ROAD SIGN") or 
                st.session_state.sign_cnn_match is not None or
                st.session_state.sign_results is not None or 
                st.session_state.sign_visual_match is not None
            )
            
            if execution_triggered:
                # If cache is empty, run the unified pipeline
                if (st.session_state.sign_cnn_match is None and
                    st.session_state.sign_results is None):

                    with st.spinner("🧠 Scanning road sign with AI..."):
                        # Step 1 — Always run CNN (works for ALL sign types, text or no text)
                        cnn_match = predict_road_sign_cnn(temp_path, target_lang)
                        st.session_state.sign_cnn_match = cnn_match

                        # Step 2 — Supplement with OCR only if text is also visible
                        if source_lang == "auto_multi":
                            lang_list = ['en', 'hi']
                        else:
                            lang_list = ['en', source_lang]

                        ocr_results = perform_ocr(temp_path, use_engine='easyocr', lang_list=lang_list, ocr_mode='scene_photo')
                        if ocr_results:
                            full_extracted_text = " ".join([r['text'] for r in ocr_results])
                            road_sign_analysis = analyze_road_sign(full_extracted_text, target_lang)
                            overlay_path = overlay_translation(temp_path, ocr_results, target_lang)
                            st.session_state.sign_results = ocr_results
                            st.session_state.sign_full_extracted_text = full_extracted_text
                            st.session_state.sign_road_sign_analysis = road_sign_analysis
                            if overlay_path and os.path.exists(overlay_path):
                                st.session_state.sign_overlay_path = overlay_path
                        else:
                            st.session_state.sign_results = []
                            st.session_state.sign_full_extracted_text = ""
                            st.session_state.sign_road_sign_analysis = None
                                
                # Retrieve from state cache
                cnn_match = st.session_state.sign_cnn_match
                results = st.session_state.sign_results
                full_extracted_text = st.session_state.sign_full_extracted_text
                road_sign_analysis = st.session_state.sign_road_sign_analysis
                visual_match = st.session_state.sign_visual_match
                overlay_path = st.session_state.sign_overlay_path
                
                with col_img2:
                    # --- PRIMARY: CNN result (always shown — works for ALL sign types) ---
                    if cnn_match is not None:
                        if cnn_match.get('detected') and cnn_match.get('is_known', True):
                            tier = cnn_match.get('confidence_tier', 'high')

                            if tier == 'high':
                                st.markdown("### 🧠 AI SIGN RECOGNITION")
                            else:
                                st.markdown("### 🧠 AI SIGN RECOGNITION *(uncertain)*")
                                st.warning(
                                    "⚠️ **Low confidence match.** This may not be a standard road sign "
                                    "recognised by this model — or the image is unclear. "
                                    "Result shown as best guess only."
                                )

                            st.error(f"**Sign Category:** {cnn_match.get('category')}")
                            st.warning(f"**Identified Sign:** {cnn_match.get('meaning')}  \n*({cnn_match.get('symbol_name')})*")
                            st.info(f"**Safety Alert:** {cnn_match.get('alert')}")
                            st.metric(
                                label="Recognition Confidence",
                                value=f"{cnn_match.get('confidence', 0.0) * 100:.1f}%"
                            )

                            # Top-3 alternatives
                            top3 = cnn_match.get('top3', [])
                            if top3 and tier != 'high':
                                with st.expander("🔍 See top-3 model guesses"):
                                    for i, t in enumerate(top3):
                                        st.caption(f"#{i+1}  {t['name']}  —  {t['confidence']*100:.1f}%")

                            narration_text = f"Traffic sign detected: {cnn_match.get('meaning')}. Safety advice: {cnn_match.get('alert')}"
                            if st.button("🗣️ Play Audio Alert", key="btn_sign_audio_cnn"):
                                play_audio_helper(narration_text, target_lang)

                        elif cnn_match.get('detected') and not cnn_match.get('is_known', True):
                            # Unknown / non-standard sign (low confidence)
                            st.markdown("### ❓ UNKNOWN / NON-STANDARD SIGN")
                            st.warning(
                                "🚫 **This sign was not recognised** by the AI model.\n\n"
                                "The model is trained on 43 standard international road signs (GTSRB). "
                                "Signs like **cattle crossing, school zone, railway crossing, "
                                "Indian-specific signs** are not in this dataset and will not be detected correctly."
                            )
                            st.info(f"ℹ️ {cnn_match.get('meaning')}")
                            st.metric(label="Best-guess Confidence (too low to trust)", value=f"{cnn_match.get('confidence', 0.0) * 100:.1f}%")

                            # Show top-3 guesses
                            top3 = cnn_match.get('top3', [])
                            if top3:
                                with st.expander("🔍 Model's closest guesses (unreliable for this sign)"):
                                    for i, t in enumerate(top3):
                                        st.caption(f"#{i+1}  {t['name']}  —  {t['confidence']*100:.1f}%")

                            # Geometric HSV fallback
                            st.markdown("---")
                            st.markdown("#### 📐 Shape & Colour Analysis (Fallback)")
                            hsv_result = detect_visual_sign_symbols(temp_path, target_lang)
                            if hsv_result and hsv_result.get('detected'):
                                st.success(f"**Detected Shape:** {hsv_result.get('symbol_name')}")
                                st.info(f"**Guidance:** {hsv_result.get('alert')}")
                            else:
                                st.caption("Shape analysis could not identify the sign geometry either.")

                            if st.button("🗣️ Play Unknown Sign Alert", key="btn_sign_audio_unknown"):
                                play_audio_helper(cnn_match.get('alert', '⚠️ Unknown road sign detected. Please drive carefully.'), target_lang)

                        else:
                            st.error(f"⚠️ CNN could not classify this sign: {cnn_match.get('meaning')}")
                            st.info("Try uploading a clearer, closer photo of the road sign.")

                    # --- SUPPLEMENTARY: OCR text if text was also visible on the sign ---
                    if results and len(results) > 0:
                        st.markdown("---")
                        st.markdown("### 📝 Text Also Detected on Sign")
                        st.caption(f"Extracted text: `{full_extracted_text}`")
                        if road_sign_analysis:
                            translated_meaning = road_sign_analysis.get('meaning', '')
                            if translated_meaning:
                                st.success(f"**Sign Text Meaning ({SUPPORTED_LANGUAGES.get(target_lang, target_lang)}):** {translated_meaning}")
                        if overlay_path and os.path.exists(overlay_path):
                            st.image(overlay_path, caption="Text Localisation Overlay", use_container_width=True)

            # Clean up temp image
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
                
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 3. SPEECH & AUDIO NODE ---
    elif nav == "🔊 Speech & Audio Node":
        st.title("🔊 NEURAL VOICE & AUDIO NODE")
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        
        audio_source = st.radio("Audio Node Input", ["Upload WAV/MP3 File", "Text-to-Speech Voice Synthesizer"])
        
        if audio_source == "Upload WAV/MP3 File":
            st.markdown("### 🎙️ Speech-to-Text Translator")
            audio_file = st.file_uploader("Upload Speech Track", type=["wav", "mp3", "ogg"])
            target_lang = st.selectbox("Translate Speech Output to", list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x], index=0)
            
            if audio_file and st.button("TRANSCRIBE & TRANSLATE SPEECH"):
                with st.spinner("Extracting auditory frequencies..."):
                    # Save audio locally
                    temp_audio_path = os.path.join(BASE_DIR, 'temp', f"audio_upload_{int(time.time())}.wav")
                    if not os.path.exists(os.path.dirname(temp_audio_path)):
                        os.makedirs(os.path.dirname(temp_audio_path))
                        
                    with open(temp_audio_path, "wb") as f:
                        f.write(audio_file.getvalue())
                    
                    # Transcribe
                    transcription = transcribe_audio_file(temp_audio_path, use_engine='whisper')
                    
                    st.info(f"🗣️ Transcribed English Text: {transcription}")
                    
                    # Translate
                    translated_speech, _, _ = translate_text(transcription, 'auto', target_lang)
                    st.success(f"🎯 Translated Speech ({SUPPORTED_LANGUAGES[target_lang]}): {translated_speech}")
                    
                    # Narrate translation
                    if st.button("🗣️ Synthesize Translated Audio"):
                        play_audio_helper(translated_speech, target_lang)
                        
                    # Clean up temp audio
                    try:
                        os.remove(temp_audio_path)
                    except Exception:
                        pass
        else:
            st.markdown("### 🗣️ Indian Accent Text-to-Speech Synthesizer")
            tts_text = st.text_area("Enter Text for Voice Synthesis", "Hello, welcome to BhashaBridge Cloud services.")
            target_lang = st.selectbox("Select Target Language & Accent", list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x], index=0)
            
            if st.button("SYNTHESIZE VOICE"):
                if tts_text.strip():
                    with st.spinner("Linking to neural voice oscillators..."):
                        play_audio_helper(tts_text, target_lang)
                else:
                    st.warning("Please enter text to synthesize.")
                    
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 4. VIDEO SUBTITLE ENGINE ---
    elif nav == "🎬 Video Subtitle Engine":
        st.title("🎬 VIDEO SUBTITLING CORE")
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        
        st.info("💡 Subtitle Engine processes files locally. It will extract vocal frequencies, generate SubRip SRT scripts with exact timestamps, translate them, and export a burned subtitle mp4 video.")
        
        vid_file = st.file_uploader("Upload Video Track (MP4)", type=["mp4"])
        target_lang = st.selectbox("Generate Subtitles in", list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x], index=1)
        
        if vid_file and st.button("PROCESS VIDEO SUBTITLES"):
            with st.spinner("Processing neural frames & audio tracks... This can take up to 2-3 minutes depending on video duration."):
                # Save uploaded video locally
                temp_vid_path = os.path.join(BASE_DIR, 'temp', f"temp_video_{int(time.time())}.mp4")
                if not os.path.exists(os.path.dirname(temp_vid_path)):
                    os.makedirs(os.path.dirname(temp_vid_path))
                    
                with open(temp_vid_path, "wb") as f:
                    f.write(vid_file.getvalue())
                
                # Process subtitles
                sub_results = generate_subtitles(temp_vid_path, target_lang)
                
                if "error" in sub_results:
                    st.error(f"❌ Subtitle Processing Failed: {sub_results['error']}")
                else:
                    st.success("🎯 Video subtitle compilation succeeded!")
                    
                    if "burning_warning" in sub_results:
                        st.warning(sub_results['burning_warning'])
                        
                    # Show translated video
                    col_vid1, col_vid2 = st.columns(2)
                    with col_vid1:
                        st.markdown("#### Original/Output Video Playback")
                        st.video(sub_results['output_video_path'])
                        
                    with col_vid2:
                        st.markdown("#### Translated Subtitle SRT Preview")
                        with open(sub_results['srt_path'], 'r', encoding='utf-8') as f:
                            srt_text = f.read()
                        st.text_area("Subtitles SRT", value=srt_text, height=250)
                        
                        # Downloads
                        st.download_button(
                            label="📥 Download Subtitle SRT File",
                            data=srt_text,
                            file_name=f"translated_subtitles_{SUPPORTED_LANGUAGES[target_lang]}.srt",
                            mime="text/plain"
                        )
                        
                        if os.path.exists(sub_results['output_video_path']):
                            with open(sub_results['output_video_path'], 'rb') as f:
                                st.download_button(
                                    label="📥 Download Video with Subtitles",
                                    data=f,
                                    file_name="translated_video_output.mp4",
                                    mime="video/mp4"
                                )
                                
                # Cleanup temp file
                try:
                    os.remove(temp_vid_path)
                except Exception:
                    pass
                    
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 5. ACCESSIBILITY HUB ---
    elif nav == "♿ Accessibility Hub":
        st.title("♿ ACCESSIBILITY CONTROLS")
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        st.write("Configure cognitive and visual assistance settings for the platform:")
        
        acc_toggle = st.toggle("♿ Activate Accessibility Enhancements", value=st.session_state.accessibility)
        if acc_toggle != st.session_state.accessibility:
            st.session_state.accessibility = acc_toggle
            st.rerun()
            
        if st.session_state.accessibility:
            col_acc1, col_acc2 = st.columns(2)
            with col_acc1:
                st.markdown("### 👓 Visual Preferences")
                hc_toggle = st.toggle("🌙 High Contrast Mode (Yellow text on Black)", value=st.session_state.high_contrast)
                lt_toggle = st.toggle("🔎 Large Casing & Typography (+50% Scale)", value=st.session_state.large_text)
                
                if hc_toggle != st.session_state.high_contrast or lt_toggle != st.session_state.large_text:
                    st.session_state.high_contrast = hc_toggle
                    st.session_state.large_text = lt_toggle
                    st.rerun()
            with col_acc2:
                st.markdown("### 🗣️ Cognitive Audio Assister")
                auto_tts = st.toggle("⚡ Autoplay Audio Narration on Translation", value=st.session_state.tts_auto_play)
                st.session_state.tts_auto_play = auto_tts
                
                if st.button("🗣️ Test Accessibility Narrator"):
                    play_audio_helper("Cognitive auditory assistance node active. Welcome to BhashaBridge.", "en")
        else:
            st.info("Enable the toggle above to activate Large Text, High Contrast, and voice narration guides.")
            
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 6. TOURIST & TRAVEL HELPER ---
    elif nav == "ℹ️ Tourist & Travel Helper":
        st.title("ℹ️ TOURIST SAFETY & EMERGENCY HELPER")
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        
        st.write("Quick-access translations and phonetic guides for emergency, medical, and travel interactions.")
        
        helper_category = st.selectbox("Interaction Context", ["Emergency Alerts", "Medical Conversations", "General Tourist Phrasebook"])
        target_lang = st.selectbox("Target Regional Accent", list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x], index=1)
        
        phrase_list = []
        if helper_category == "Emergency Alerts":
            phrase_list = [
                "Stop", "Danger", "Fire", "Call the police", "I need help", "I am lost"
            ]
        elif helper_category == "Medical Conversations":
            phrase_list = [
                "Where is the hospital?", "I am sick", "I need a doctor", "I have pain", "Take me to a clinic", "Medicine"
            ]
        else:
            phrase_list = [
                "Hello", "How are you?", "Thank you", "How much is this?", "Where is the train station?", "Water"
            ]
            
        st.markdown("### 🗂️ Phrase Quick Cards")
        for phrase in phrase_list:
            col_p1, col_p2 = st.columns([3, 1])
            with col_p1:
                # Get dynamic phonetic guides
                phonetic = transliterate_text(phrase, target_lang)
                st.info(f"**English:** {phrase}  \n**{SUPPORTED_LANGUAGES[target_lang]}:** {phonetic}")
            with col_p2:
                # Single button triggers TTS
                if st.button(f"🗣️ Play '{phrase}'", key=f"tts_{phrase}"):
                    translated, _, _ = translate_text(phrase, 'en', target_lang)
                    play_audio_helper(translated, target_lang)
                    
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 7. HISTORY & DIAGNOSTICS ---
    elif nav == "📊 History & Diagnostics":
        st.title("📊 SYSTEM HISTORY & DIAGNOSTICS")
        
        # 1. Diagnostics Card
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Local System Diagnostics")
        
        gpu_ok, gpu_name = check_gpu_support()
        aws_translate_ok = get_aws_client('translate') is not None
        aws_polly_ok = get_aws_client('polly') is not None
        
        col_diag1, col_diag2, col_diag3 = st.columns(3)
        with col_diag1:
            st.metric("CUDA Hardware Accelerator", gpu_name, delta="GPU Enabled" if gpu_ok else "CPU Fallback")
        with col_diag2:
            st.metric("AWS Polly Voice Engine", "Connected" if aws_polly_ok else "Offline", delta="gTTS Fallback Active" if not aws_polly_ok else "AWS Premium")
        with col_diag3:
            st.metric("AWS Neural Translator", "Connected" if aws_translate_ok else "Offline", delta="Google/MyMemory Fallback" if not aws_translate_ok else "AWS Premium")
            
        if st.button("🧹 Clear Temp Directories Cache"):
            count = clean_directories()
            st.success(f"Cleaned up {count} cached files from temp/, outputs/, and uploads/ directories.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 2. Database History Card
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        st.subheader("📜 Translation History Log")
        
        hist_df = get_translation_history()
        
        if hist_df.empty:
            st.write("No translation operations logged yet in nexus.db.")
        else:
            st.dataframe(hist_df, use_container_width=True)
            
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                # Download CSV
                csv_data = hist_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export History Report (CSV)",
                    data=csv_data,
                    file_name="bhashabridge_history_report.csv",
                    mime="text/csv"
                )
            with sub_col2:
                if st.button("🗑️ Clear Entire History Log"):
                    clear_translation_history()
                    st.success("All database history logs cleared.")
                    time.sleep(1)
                    st.rerun()
                    
        st.markdown('</div>', unsafe_allow_html=True)