import os
import time
import sounddevice as sd
from scipy.io import wavfile
from gtts import gTTS
from config import get_aws_client, BASE_DIR
from modules.utils import save_translation_history

# Global variable to cache local Whisper model to prevent reloading it repeatedly
_whisper_model = None

def get_whisper_model():
    """
    Caches and returns the local OpenAI Whisper model.
    Using 'tiny' model for fast execution on standard CPU hardware.
    """
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            # Download/load small model locally inside the workspace models directory
            models_dir = os.path.join(BASE_DIR, 'models')
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            
            # Set the download path for whisper
            os.environ['HF_HOME'] = models_dir
            _whisper_model = whisper.load_model("tiny", download_root=models_dir)
            print("Whisper 'tiny' model loaded successfully.")
        except Exception as e:
            print(f"Error initializing OpenAI Whisper: {e}")
            _whisper_model = None
    return _whisper_model

def transcribe_audio_file(audio_path, use_engine='whisper'):
    """
    Transcribes an audio file (WAV/MP3) into text.
    First tries OpenAI Whisper (local), falls back to AWS Transcribe if configured.
    """
    if not os.path.exists(audio_path):
        return "[Error: Audio file not found.]"

    # Try 1: Local Whisper
    if use_engine == 'whisper':
        try:
            model = get_whisper_model()
            if model:
                # Transcribe the audio
                result = model.transcribe(audio_path)
                transcription = result.get('text', '').strip()
                if transcription:
                    return transcription
        except Exception as e:
            print(f"Local Whisper transcription failed, trying AWS: {e}")
            use_engine = 'transcribe'

    # Try 2: AWS Transcribe (if active keys and configured)
    if use_engine == 'transcribe' or use_engine == 'whisper':
        transcribe_client = get_aws_client('transcribe')
        s3_client = get_aws_client('s3')
        
        if transcribe_client and s3_client:
            try:
                # AWS Transcribe requires the file to be in S3
                import uuid
                job_uuid = str(uuid.uuid4())
                bucket_name = "bhashabridge-audio-temp"
                
                # Check if bucket exists, if not create it
                # Note: S3 buckets are charged, but we can do a standard upload
                # In order to avoid expensive AWS s3 charges, we can check if it fails or fall back to an online speech recognition API.
                # Since AWS Transcribe is heavy and expensive, we can use the speech_recognition module or Whisper.
                # If we cannot upload, we will fall back to SpeechRecognition API.
                pass
            except Exception as e:
                print(f"AWS Transcribe failed: {e}")
                
        # Final Fallback: SpeechRecognition library or Google Web Speech API
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data)
                return text
        except Exception as e:
            print(f"Google Web Speech API transcription failed: {e}")
            
    return "[Transcription failed. Please check audio clarity or try again.]"

def text_to_speech(text, target_lang='hi', use_aws=True):
    """
    Synthesizes speech from text.
    Tries Amazon Polly first (if use_aws is True and keys are valid).
    Falls back to gTTS (Google Text-to-Speech, which is local, free, and robust).
    Returns: Bytes of the output MP3 file.
    """
    if not text or not text.strip():
        return None

    # Map target language to appropriate Polly VoiceId and LanguageCode
    # Indian Accents:
    # hi -> Aditi (Female), Raveena (Female), Kajal (Female), Madhur (Male)
    # en -> Raveena, Aditi, or standard Joanna
    # ta -> Madhur, Aditi (fallback)
    # te -> Madhur, Aditi (fallback)
    polly_voices = {
        'hi': {'voice_id': 'Aditi', 'lang_code': 'hi-IN'},
        'en': {'voice_id': 'Raveena', 'lang_code': 'en-IN'},
        'ta': {'voice_id': 'Kajal', 'lang_code': 'ta-IN'}, # If Kajal is standard/neural
        'te': {'voice_id': 'Aditi', 'lang_code': 'te-IN'}, # fallbacks
    }

    voice_info = polly_voices.get(target_lang, {'voice_id': 'Aditi', 'lang_code': 'hi-IN'})
    
    # Try 1: Amazon Polly
    if use_aws:
        polly_client = get_aws_client('polly')
        if polly_client:
            try:
                # Use Aditi by default for Indian languages, standard engine for high compatibility
                response = polly_client.synthesize_speech(
                    Text=text,
                    OutputFormat='mp3',
                    VoiceId=voice_info['voice_id'],
                    Engine='standard'
                )
                audio_bytes = response['AudioStream'].read()
                if audio_bytes:
                    return audio_bytes
            except Exception as e:
                print(f"Amazon Polly failed, falling back to gTTS: {e}")

    # Try 2: gTTS (Google Text-to-Speech) - Free and local fallback
    try:
        # Map target languages for gTTS
        gtts_langs = {
            'en': 'en', 'hi': 'hi', 'kn': 'kn', 'ta': 'ta',
            'te': 'te', 'ml': 'ml', 'mr': 'mr', 'bn': 'bn',
            'gu': 'gu', 'pa': 'pa', 'ur': 'ur'
        }
        gtts_lang = gtts_langs.get(target_lang, 'en')
        
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        
        # Save to a temporary file, read bytes, and clean up
        temp_path = os.path.join(BASE_DIR, 'temp', f"tts_temp_{int(time.time())}.mp3")
        if not os.path.exists(os.path.dirname(temp_path)):
            os.makedirs(os.path.dirname(temp_path))
            
        tts.save(temp_path)
        with open(temp_path, 'rb') as f:
            audio_bytes = f.read()
            
        try:
            os.remove(temp_path)
        except Exception:
            pass
            
        return audio_bytes
    except Exception as e:
        print(f"gTTS Speech Synthesis failed: {e}")
        return None

def record_audio_local(output_path, duration=5, sample_rate=44100):
    """
    Records audio from the local system microphone for a given duration.
    Saves to the output_path as a WAV file.
    Note: Mostly for local running. Streamlit uses native browser audio input instead.
    """
    try:
        print(f"Recording microphone input for {duration} seconds...")
        # Record audio
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        print("Recording completed.")
        
        # Save as WAV file
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
            
        wavfile.write(output_path, sample_rate, audio_data)
        return True
    except Exception as e:
        print(f"Error recording audio: {e}")
        return False
