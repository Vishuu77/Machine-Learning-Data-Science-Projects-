import streamlit as st
import io
from tts_core import LANGUAGES, SAMPLE_TEXTS, generate_gtts_audio, generate_pyttsx3_audio, get_sapi5_voices

st.set_page_config(page_title="Advanced Multilingual TTS & Voice Studio", layout="centered", page_icon="🗣️")

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stSidebar { background-color: #1a1d29; }
    h1, h2, h3 { color: #00E5A0; }
    .stButton>button { background-color: #00E5A0; color: #0e1117; font-weight: bold; border-radius: 6px; width: 100%; }
    .stButton>button:hover { background-color: #00b37d; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

st.title("🗣️ Multilingual Voice Studio")
st.markdown("Convert text into speech with **language selection (English, Kannada, Hindi, Telugu, Tamil, Bengali, Marathi)**, **voice tone/profile selection**, and **speed adjustments**.")

# Sidebar Settings
st.sidebar.header("Configure Voice")

# 1. Select Language (Always visible)
selected_lang_name = st.sidebar.selectbox("Select Language", list(LANGUAGES.keys()))
lang_code = LANGUAGES[selected_lang_name]

# 2. Select Voice Style / Tone based on language
st.sidebar.markdown("---")
st.sidebar.markdown("**Tone & Voice Profile**")

voice_options = []
if lang_code == "en":
    voice_options = ["Natural Voice (Google)", "Male Voice (David - SAPI5)", "Female Voice (Zira - SAPI5)"]
else:
    voice_options = ["Natural Accent (Google)"]

selected_voice = st.sidebar.selectbox("Voice Tone", voice_options)

# 3. Speech Rate (Fine-grained slider restored as requested)
st.sidebar.markdown("---")
st.sidebar.markdown("**Speed Control**")
speed_rate = st.sidebar.slider("Speech Rate (Words Per Minute)", 100, 350, 180, 10)

# Map controls to engines
is_gtts = "Google" in selected_voice or "Natural" in selected_voice
# For gtts: if speed is below 150, trigger slow mode
is_slow = (speed_rate < 150)

# Preset sample text helper
default_text = SAMPLE_TEXTS.get(lang_code, "Hello!")

st.subheader(f"Language: {selected_lang_name} | Voice: {selected_voice}")

text_input = st.text_area("Enter Text to Convert:", value=default_text, height=150)

if st.button("🔊 Synthesize & Generate Speech"):
    if not text_input.strip():
        st.warning("Please enter some text to convert.")
    else:
        try:
            with st.spinner("Synthesizing voice audio..."):
                if is_gtts:
                    audio_bytes, audio_format = generate_gtts_audio(text_input, lang_code, slow=is_slow)
                else:
                    # Windows local SAPI5 voices
                    voice_dict = get_sapi5_voices()
                    voice_id = ""
                    if "David" in selected_voice:
                        voice_id = voice_dict.get("Microsoft David Desktop - English (United States)", "")
                    elif "Zira" in selected_voice:
                        voice_id = voice_dict.get("Microsoft Zira Desktop - English (United States)", "")
                    
                    audio_bytes, audio_format = generate_pyttsx3_audio(text_input, voice_id, speed_rate)
                    
                st.session_state["audio_bytes"] = audio_bytes
                st.session_state["audio_format"] = audio_format
                st.session_state["file_ext"] = audio_format
            st.success("Speech generated successfully!")
        except Exception as e:
            st.error(f"Error generating speech: {e}")

if "audio_bytes" in st.session_state:
    st.markdown("---")
    st.subheader("🎧 Play & Download Audio")
    
    audio_data = st.session_state["audio_bytes"]
    fmt = st.session_state.get("audio_format", "mp3")
    
    st.audio(audio_data, format=f"audio/{fmt}")
    
    st.download_button(
        label=f"📥 Download Audio File (.{fmt})",
        data=audio_data,
        file_name=f"speech_{lang_code}.{fmt}",
        mime=f"audio/{fmt}"
    )
