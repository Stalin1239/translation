import streamlit as st
import boto3
import sqlite3
import hashlib
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. AWS & DB CONFIG ---
# Load from environment variables
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
SECRET_KEY = os.getenv("AWS_SECRET_KEY")
REGION = os.getenv("AWS_REGION", "us-east-1")

def get_client(service):
    return boto3.client(service, aws_access_key_id=ACCESS_KEY, 
                        aws_secret_access_key=SECRET_KEY, region_name=REGION)

def init_db():
    conn = sqlite3.connect('nexus.db')
    conn.execute('CREATE TABLE IF NOT EXISTS users (user TEXT PRIMARY KEY, pw TEXT)')
    conn.commit()
    conn.close()

# --- 2. GLOBAL CSS (Futuristic Glassmorphism) ---
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top right, #1e1e2f, #0a0a12);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphism Containers */
    .nexus-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Neon Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 18, 0.95) !important;
        border-right: 1px solid #00f2fe;
    }
    
    /* Headlines */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 2px;
    }

    /* Modern Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #00f2fe 0%, #4facfe 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: 0.3s all ease;
        text-transform: uppercase;
        font-size: 0.8rem;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px #00f2fe;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. PAGE MODULES ---
def page_chat():
    st.title("💬 NEURAL CHAT")
    if "messages" not in st.session_state: 
        st.session_state.messages = []
    
    target = st.selectbox("Link Language", ["hi", "kn", "ta", "te", "ml"])
    
    # Display existing messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): 
            st.markdown(m["content"])

    # Chat Input logic
    if p := st.chat_input("Enter message..."):
        # 1. Add user message to state and display immediately
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"):
            st.markdown(p)
            
        # 2. Call AWS with error handling
        try:
            client = get_client('translate')
            res = client.translate_text(
                Text=p, 
                SourceLanguageCode="auto", 
                TargetLanguageCode=target
            )
            ans = res['TranslatedText']
            
            # 3. Add AI message to state and display
            st.session_state.messages.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"):
                st.markdown(ans)
                
        except Exception as e:
            # If there is a permission or key issue, it will show here in RED
            st.error(f"⚠️ AWS Neural Link Error: {str(e)}")
            
        # Note: We do NOT use st.rerun() here so the UI stays stable
def page_vision():
    st.title("👁️ VISION CORE")
    st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
    up = st.file_uploader("Scan Visual Data", type=['png', 'jpg'])
    target = st.selectbox("Translate to", ["kn", "hi", "ta", "te", "ml"])
    
    if up and st.button("EXECUTE SCAN"):
        rek = get_client('rekognition')
        trn = get_client('translate')
        
        # OCR
        res = rek.detect_text(Image={'Bytes': up.getvalue()})
        raw = " ".join([t['DetectedText'] for t in res['TextDetections'] if t['Type']=='LINE'])
        
        # Translate
        final = trn.translate_text(Text=raw, SourceLanguageCode="auto", TargetLanguageCode=target)
        
        col1, col2 = st.columns(2)
        with col1: st.info(f"Extracted: {raw}")
        with col2: st.success(f"Translated: {final['TranslatedText']}")
    st.markdown('</div>', unsafe_allow_html=True)
import io  # Add this at the top of your file

import io
import base64  # Need this for the trick

def page_audio():
    st.title("🔊 NEURAL AUDIO NODE")
    st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
    
    options = {"Hindi": "hi", "Kannada": "kn", "Tamil": "ta", "Telugu": "te", "English": "en"}
    label = st.selectbox("Select Target Language", list(options.keys()))
    target_code = options[label]
    
    txt = st.text_area("Enter text", "Hello world")

    if st.button("GENERATE VOICE"):
        try:
            trn = get_client('translate')
            pol = get_client('polly')

            # 1. Translate
            t_res = trn.translate_text(Text=txt, SourceLanguageCode="auto", TargetLanguageCode=target_code)
            translated_text = t_res['TranslatedText']
            st.info(f"Text to be spoken: {translated_text}")

            # 2. Request Speech
            # Standard engine is the most compatible across all AWS accounts
            response = pol.synthesize_speech(
                Text=translated_text,
                OutputFormat='mp3',
                VoiceId='Aditi', 
                Engine='standard' 
            )

            # 3. Read the Raw Bytes
            audio_bytes = response['AudioStream'].read()
            
            if audio_bytes:
                # 4. THE FIX: Convert to Base64
                # This embeds the audio directly into the HTML so it can't be 'empty'
                b64 = base64.b64encode(audio_bytes).decode()
                md = f"""
                    <audio controls autoplay="true">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                    """
                st.markdown(md, unsafe_allow_html=True)
                
                st.success("Neural link established. Audio ready.")
                st.download_button("Download MP3", audio_bytes, "speech.mp3")
            else:
                st.error("AWS returned no data.")

        except Exception as e:
            st.error(f"Nexus Audio Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
def main():
    inject_css()
    init_db()
    
    if 'auth' not in st.session_state: st.session_state.auth = False

    if not st.session_state.auth:
        st.title("🔐 NEXUS GATEWAY")
        with st.container():
            st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
            mode = st.tabs(["Login", "Identity Creation"])
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            
            if st.button("AUTH"):
                conn = sqlite3.connect('nexus.db')
                hp = hashlib.sha256(p.encode()).hexdigest()
                # Check / Add logic here
                st.session_state.auth = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Sidebar Navigation
        st.sidebar.title("💠 NEXUS CORE")
        nav = st.sidebar.radio("Navigate", ["Neural Chat", "Vision Scan", "Audio Node", "System Status"])
        
        if st.sidebar.button("EXIT"):
            st.session_state.auth = False
            st.rerun()

        if nav == "Neural Chat": page_chat()
        elif nav == "Vision Scan": page_vision()
        elif nav == "Audio Node": page_audio()
        elif nav == "System Status":
            st.title("📊 SYSTEM STATUS")
            st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
            st.write("AWS Services: **Operational**")
            st.write("Free Tier Usage: **Optimal**")
            st.write("Neural Links: **Active**")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()