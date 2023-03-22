from typing import Union
import uvicorn
from fastapi import FastAPI, UploadFile, File
import whisper
import time
import os

app = FastAPI()
LANGUAGE = "es"


@app.on_event("startup")
def on_startup():
    global whisper_model

    whisper_model = whisper.load_model('large')

@app.post("/upload")
async def receive_file(file: UploadFile = File(...)):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    filename = f'{dir_path}/uploads/{time.time()}-{file.filename}'
    with open(filename, 'wb') as f:
        content = await file.read()
        f.write(content)


@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):

    # Upload File
    print('Uploading file...')

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = f'{dir_path}/uploads/{time.time()}-{file.filename}'
    with open(filename, 'wb') as f:
        content = await file.read()
        f.write(content)
    print('File saved as: ', filename)

    print('Getting audio language...')
    detected_language, mel = get_language(whisper_model, filename)

    if str(detected_language) != LANGUAGE:
        transcription = "ERROR: wrong language"
    else:
        print('Transcribing...')
        transcription = transcriber(whisper_model, mel)

        print('Transcrition: ', transcription)

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)