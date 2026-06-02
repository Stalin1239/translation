import traceback
import requests
import json
import urllib.parse
from langdetect import detect as detect_lang
from deep_translator import GoogleTranslator, MyMemoryTranslator
from config import SUPPORTED_LANGUAGES, get_aws_client
from modules.utils import save_translation_history

def detect_language(text):
    """
    Detects the source language of a given text.
    """
    try:
        lang = detect_lang(text)
        if lang in SUPPORTED_LANGUAGES:
            return lang
        return 'en'  # Fallback to English
    except Exception:
        return 'en'

def google_transliterate(text, target_lang='hi'):
    """
    Phonetically transliterates Latin script input (e.g. "hi bhai how are you")
    into the target Indian regional script (e.g. "हाय भाई हाउ आर यू")
    using Google Input Tools API.
    """
    if not text or not text.strip():
        return ""
        
    try:
        lang_map = {
            'hi': 'hi-t-i0-und',
            'kn': 'kn-t-i0-und',
            'ta': 'ta-t-i0-und',
            'te': 'te-t-i0-und',
            'ml': 'ml-t-i0-und',
            'mr': 'mr-t-i0-und',
            'bn': 'bn-t-i0-und',
            'gu': 'gu-t-i0-und',
            'pa': 'pa-t-i0-und'
        }
        
        itc = lang_map.get(target_lang)
        if not itc:
            # Fall back to standard translation if target script is not supported in Google Input Tools
            val, _, _ = translate_text(text, 'auto', target_lang)
            return val
            
        url = f"https://inputtools.google.com/request?text={urllib.parse.quote(text)}&itc={itc}&num=1&cp=0&cs=1&ie=utf-8&oe=utf-8&app=demopage"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data[0] == "SUCCESS":
                candidates = []
                for word_info in data[1]:
                    if word_info.get("candidate"):
                        candidates.append(word_info["candidate"][0])
                return " ".join(candidates)
    except Exception as e:
        print(f"Transliteration error: {e}")
    # Fall back to translating if API fails
    val, _, _ = translate_text(text, 'auto', target_lang)
    return val

def translate_text(text, source_lang='auto', target_lang='en'):
    """
    Translates text from source_lang to target_lang.
    Uses AWS Translate if available, then deep-translator (Google), then MyMemory, then googletrans.
    """
    if not text or not text.strip():
        return "", "auto", 0.0

    # Auto detect language locally for history tracking and interface mapping
    detected_lang = source_lang
    if source_lang == 'auto':
        detected_lang = detect_language(text)

    # If same language, no translation needed
    if detected_lang == target_lang:
        return text, detected_lang, 1.0

    # Try 1: AWS Translate (if keys valid)
    aws_client = get_aws_client('translate')
    if aws_client:
        try:
            # Map 'auto' to auto for AWS
            aws_source = "auto" if source_lang == 'auto' else source_lang
            response = aws_client.translate_text(
                Text=text,
                SourceLanguageCode=aws_source,
                TargetLanguageCode=target_lang
            )
            translated = response.get('TranslatedText', '')
            if translated:
                save_translation_history("Text Translation (AWS)", detected_lang, target_lang, text, translated, 1.0)
                return translated, detected_lang, 1.0
        except Exception:
            pass  # Fall back to free options

    # Try 2: Deep Translator (GoogleTranslator)
    try:
        # Use raw source_lang (which may be 'auto') to leverage Google's internal deep neural auto-detection
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        if translated:
            save_translation_history("Text Translation (Google)", detected_lang, target_lang, text, translated, 0.95)
            return translated, detected_lang, 0.95
    except Exception:
        pass

    # Try 3: Deep Translator (MyMemoryTranslator)
    try:
        translator = MyMemoryTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        if translated:
            save_translation_history("Text Translation (MyMemory)", detected_lang, target_lang, text, translated, 0.85)
            return translated, detected_lang, 0.85
    except Exception:
        pass

    # Try 4: googletrans (as final fallback)
    try:
        from googletrans import Translator as GoogleTransTranslator
        translator = GoogleTransTranslator()
        res = translator.translate(text, src=source_lang, dest=target_lang)
        if res and res.text:
            save_translation_history("Text Translation (googletrans)", detected_lang, target_lang, text, res.text, 0.80)
            return res.text, detected_lang, 0.80
    except Exception as e:
        err_msg = f"Translation error: {str(e)}"
        print(err_msg)
        traceback.print_exc()
        return f"[Translation Error: {err_msg}]", detected_lang, 0.0

def transliterate_text(text, target_lang='hi'):
    """
    Provides a simple phonetics transliteration/pronunciation guide
    for tourist/tourist assistance modes.
    """
    # Simple phonetic dictionary for basic emergency / tourist phrases
    PHONETIC_PHRASES = {
        'en': {
            'hi': {
                'hello': 'Namaste (नमस्ते)',
                'how are you?': 'Aap kaise hain? (आप कैसे हैं?)',
                'where is the hospital?': 'Hospital kahaan hai? (अस्पताल कहाँ है?)',
                'please help me': 'Kripya meri madad karein (कृपया मेरी मदद करें)',
                'thank you': 'Dhanyavaad (धन्यवाद)',
                'stop': 'Rukiye (रुकिए)',
                'danger': 'Khatra (खतरा)',
                'how much is this?': 'Yeh kitne ka hai? (यह कितने का है?)'
            },
            'kn': {
                'hello': 'Namaskara (ನಮಸ್ಕಾರ)',
                'how are you?': 'Hegiddira? (ಹೇಗಿದ್ದೀರಾ?)',
                'where is the hospital?': 'Hospital elli ide? (ಆಸ್ಪತ್ರೆ ಎಲ್ಲಿದೆ?)',
                'please help me': 'Dayavittu nanage sahaya madi (ದಯವಿಟ್ಟು ನನಗೆ ಸಹಾಯ ಮಾಡಿ)',
                'thank you': 'Dhanayavadagalu (ಧನ್ಯವಾದಗಳು)',
                'stop': 'Nillisi (ನಿಲ್ಲಿಸಿ)',
                'danger': 'Apaya (ಅಪಾಯ)',
                'how much is this?': 'Idu eshtu? (ಇದು ಎಷ್ಟು?)'
            }
        }
    }

    # Clean the input phrase
    phrase = text.strip().lower()
    
    # Check dictionary first
    if 'en' in PHONETIC_PHRASES and target_lang in PHONETIC_PHRASES['en']:
        if phrase in PHONETIC_PHRASES['en'][target_lang]:
            return PHONETIC_PHRASES['en'][target_lang][phrase]
    
    # Dynamic fallback: Use Google Translation API to get pronunciation via deep-translator or googletrans
    try:
        from googletrans import Translator as GoogleTransTranslator
        translator = GoogleTransTranslator()
        res = translator.translate(text, src='auto', dest=target_lang)
        if res and hasattr(res, 'pronunciation') and res.pronunciation:
            return f"{res.pronunciation} ({res.text})"
        return res.text
    except Exception:
        # Fall back to translating
        val, _, _ = translate_text(text, 'auto', target_lang)
        return val
