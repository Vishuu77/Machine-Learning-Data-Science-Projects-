import os
import io
import pyttsx3
from gtts import gTTS

LANGUAGES = {
    "English (English)": "en",
    "Hindi (हिन्दी)": "hi",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Telugu (తెలుగు)": "te",
    "Tamil (தமிழ்)": "ta",
    "Bengali (বাংলা)": "bn",
    "Marathi (मराठी)": "mr"
}

SAMPLE_TEXTS = {
    "en": "Hello! Welcome to the multilingual Text-to-Speech converter with tone and speed controls.",
    "hi": "नमस्ते! यह एक बहुभाषी टेक्स्ट-टू-स्पीच कनवर्टर है।",
    "kn": "ನಮಸ್ಕಾರ! ಇದು ಬಹುಭಾಷಾ ಧ್ವನಿ ಪರಿವರ್ತಕ.",
    "te": "నమస్కారం! ఇది టెక్స్ట్-టు-స్పీచ్ కన్వర్టర్.",
    "ta": "வணக்கம்! இது டெக்ஸ்ட்-டு-ஸ்பீச் மாற்றி.",
    "bn": "নমস্কার! এটি টেক্সট-টু-স্পিচ কনভার্টার।",
    "mr": "नमस्कार! हा टेक्स्ट-टू-स्पीच कनवर्टर आहे."
}

def generate_gtts_audio(text, lang_code, slow=False):
    if not text.strip():
        raise ValueError("Text cannot be empty.")
    tts = gTTS(text=text, lang=lang_code, slow=slow)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read(), "mp3"

def generate_pyttsx3_audio(text, voice_id, rate):
    if not text.strip():
        raise ValueError("Text cannot be empty.")
    
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    if voice_id:
        engine.setProperty('voice', voice_id)
        
    temp_path = os.path.abspath("temp_speech.wav")
    engine.save_to_file(text, temp_path)
    engine.runAndWait()
    
    with open(temp_path, "rb") as f:
        audio_bytes = f.read()
    
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except:
            pass
            
    return audio_bytes, "wav"

def get_sapi5_voices():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        return {v.name: v.id for v in voices}
    except:
        return {"Default": ""}
