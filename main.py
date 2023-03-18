from typing import Union
from fastapi import FastAPI
import whisper

app = FastAPI()


@app.on_event("startup")
def on_startup():
    global whisper_model

    whisper_model = whisper.load_model(large)


@app.post("/transcribe")
async def download_file(audio_file):

    detected_language, mel = get_language(whisper_model, audio_file)
    transcription = transcriber(whisper_model, mel)

    return transcription


def get_language(model, audio_file):

    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    #print(f"Detected language: {max(probs, key=probs.get)}")
    lang_set = {max(probs, key=probs.get)}
    
    for i in lang_set:
        lang = str(i)

    return lang, mel

def transcriber(model, mel):

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    transcription = result.text

    return transcription