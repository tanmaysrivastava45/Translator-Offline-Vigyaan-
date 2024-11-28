import os
import json
import streamlit as st
import sounddevice as sd
import vosk
from transformers import MarianMTModel, MarianTokenizer
import threading

# Initialize pygame for audio playback
import pygame
pygame.mixer.init()

# Path to the Vosk Hindi model
model_path = r"D:\vigy-exp\Vigyaan-Translator-STS-main\vosk-model-small-hi-0.22"  # Change this to the correct path
model = vosk.Model(model_path)

# Initialize the recognizer with the model and sample rate (16000Hz)
recognizer = vosk.KaldiRecognizer(model, 16000)

# Specify the model for translating from Hindi to English
nmt_model_name = "Helsinki-NLP/opus-mt-hi-en"
tokenizer = MarianTokenizer.from_pretrained(nmt_model_name)
nmt_model = MarianMTModel.from_pretrained(nmt_model_name)

# Flag to control the translation process
isTranslateOn = False
audio_stream = None

# Function for text-to-speech using eSpeak-NG
def speak_text(text, language="en"):
    os.system(f'espeak-ng -v {language} "{text}"')

# Callback function to process audio in real time
def callback(indata, frames, time, status):
    if not isTranslateOn:
        return  # Stop processing if translation is off

    if recognizer.AcceptWaveform(bytes(indata)):
        result = json.loads(recognizer.Result())
        recognized_text = result['text']
        
        if recognized_text:
            # Translate recognized Hindi text to English
            tokenized_text = tokenizer.prepare_seq2seq_batch([recognized_text], return_tensors="pt")
            translated = nmt_model.generate(**tokenized_text)
            english_translation = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            
            # Speak the translated text
            speak_text(english_translation[0], language="en")

def start_translation():
    global audio_stream
    audio_stream = sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback)
    audio_stream.start()

    while isTranslateOn:
        sd.sleep(100)  # Sleep to avoid high CPU usage

    # Close the audio stream when isTranslateOn becomes False
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()

def stop_translation():
    global isTranslateOn, audio_stream
    isTranslateOn = False
    if audio_stream is not None:
        audio_stream.stop()  # Stop the stream immediately
        audio_stream.close()
        audio_stream = None  # Reset stream to None
    output_placeholder.markdown("<h3 style='color:red;'>🚫 Translation Stopped</h3>", unsafe_allow_html=True)

# UI layout
st.title("🌐 Hindi to English Language Translator")

# Instructions text
st.markdown("""
    <div style='background-color: black; padding: 10px; border-radius: 10px; color: white;'>
        <h3>Instructions:</h3>
        <p>1. Press "Start" to begin translation.</p>
        <p>2. Press "Stop" to end the translation process.</p>
    </div>
""", unsafe_allow_html=True)

# Buttons for Start and Stop
col1, col2 = st.columns(2)
with col1:
    start_button = st.button("🟢 Start", use_container_width=True)
with col2:
    stop_button = st.button("🔴 Stop", use_container_width=True)

# Output section
output_placeholder = st.empty()

# Check if "Start" button is clicked
if start_button and not isTranslateOn:
    isTranslateOn = True
    output_placeholder.markdown("<h3 style='color:blue;'>🎙️ Listening... Speak now.</h3>", unsafe_allow_html=True)

    # Start a new thread for the audio stream
    threading.Thread(target=start_translation, daemon=True).start()

# Check if "Stop" button is clicked
if stop_button and isTranslateOn:
    stop_translation()
