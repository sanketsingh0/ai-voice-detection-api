from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import tempfile
import os

app = FastAPI(
    title="AI-Generated Voice Detection API",
    version="1.0"
)

API_KEY = "GUVI1234"


class VoiceRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str


@app.post("/detect-voice")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    # 1️⃣ API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if request.audio_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 audio is supported")

    # 2️⃣ Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        if len(audio_bytes) == 0:
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # 3️⃣ Save MP3 temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3:
            temp_mp3.write(audio_bytes)
            temp_mp3_path = temp_mp3.name
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save audio file")

    # 4️⃣ Load audio using librosa
    try:
        y, sr = librosa.load(temp_mp3_path, sr=None)
    except Exception:
        os.remove(temp_mp3_path)
        raise HTTPException(status_code=400, detail="Audio file could not be processed")

    # 5️⃣ Clean up temp file
    os.remove(temp_mp3_path)

    # 6️⃣ Extract simple features
    duration = librosa.get_duration(y=y, sr=sr)
    rms_energy = np.mean(librosa.feature.rms(y=y))

    # 7️⃣ Simple AI vs Human decision logic
    if duration < 1.5 and rms_energy < 0.02:
        classification = "AI-generated"
        confidence = 0.85
        explanation = "Short duration and flat energy patterns suggest synthetic speech."
    else:
        classification = "Human-generated"
        confidence = 0.82
        explanation = "Natural duration and energy variations indicate human speech."

    # 8️⃣ Final response
    return {
        "classification": classification,
        "confidence": confidence,
        "explanation": explanation
    }
