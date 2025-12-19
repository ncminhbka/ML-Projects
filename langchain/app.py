import os
import requests
import whisper
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from streamlit_chat import message
from elevenlabs import ElevenLabs

# Load env
load_dotenv()
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

# ElevenLabs client
client = ElevenLabs(api_key=ELEVEN_API_KEY)

# Constants
TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # đổi sang model bạn đã pull
    
# Load Whisper local
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

# Transcribe audio
def transcribe_audio(audio_file_path):
    try:
        result = whisper_model.transcribe(audio_file_path, fp16=False)
        return result["text"]
    except Exception as e:
        st.error(f"Whisper error: {str(e)}")
        return None

# Record + transcribe
def record_and_transcribe_audio():
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format=AUDIO_FORMAT)
        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        if st.button("Transcribe"):
            transcription = transcribe_audio(TEMP_AUDIO_PATH)
            os.remove(TEMP_AUDIO_PATH)
            st.write(f"Transcription: {transcription}")
    return transcription

# Query Ollama local
def query_ollama(prompt):
    try:
        resp = requests.post(
            OLLAMA_API,
            json={"model": OLLAMA_MODEL, "prompt": prompt},
            stream=True
        )
        output = ""
        for line in resp.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '"response":"' in data:
                    text_part = data.split('"response":"')[1].split('"')[0]
                    output += text_part
        return output.strip()
    except Exception as e:
        st.error(f"Ollama error: {str(e)}")
        return "Error: could not connect to Ollama."

# ElevenLabs TTS (new SDK)
def speak(text, voice_id):
    try:
        response = client.text_to_speech.convert(
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            text=text
        )

        tmp_path = "output.mp3"
        with open(tmp_path, "wb") as f:
            for chunk in response:
                if isinstance(chunk, bytes):
                    f.write(chunk)
        return tmp_path
    except Exception as e:
        st.error(f"ElevenLabs error: {str(e)}")
        return None

# Get all voices (cache để không gọi nhiều lần)
@st.cache_resource
def get_voices():
    voices = client.voices.get_all()
    return {v.name: v.voice_id for v in voices.voices}

# Main app
def main():
    st.title("🎙️ Local Voice Chatbot (Whisper + Ollama + ElevenLabs)")

    # Chọn giọng
    voices = get_voices()
    voice_name = st.selectbox("Chọn giọng đọc:", list(voices.keys()))
    voice_id = voices[voice_name]

    # Record and transcribe
    transcription = record_and_transcribe_audio()

    # Input (text hoặc từ voice)
    user_input = st.text_input("Your message:", value=transcription if transcription else "")

    # Session state
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hello! I'm ready."]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hi there!"]

    # Xử lý input với Ollama
    if user_input:
        response = query_ollama(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)

    # Hiển thị hội thoại
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=f"{i}_user")
            message(st.session_state["generated"][i], key=f"{i}")
            audio_path = speak(st.session_state["generated"][i], voice_id)
            if audio_path:
                st.audio(audio_path, format="audio/mp3")

if __name__ == "__main__":
    main()
