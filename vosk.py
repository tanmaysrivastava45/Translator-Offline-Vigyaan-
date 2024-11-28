import vosk
import sounddevice as sd
import json

model = vosk.Model("C:\Users\TANMAY SRIVASTAVA\Downloads\vosk-model-small-en-us-0.15.zip")
recognizer = vosk.KaldiRecognizer(model, 16000)

def callback(indata, frames, time, status):
    if recognizer.AcceptWaveform(indata):
        result = json.loads(recognizer.Result())
        print(result['text'])

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    print("Listening...")
    sd.sleep(10000)