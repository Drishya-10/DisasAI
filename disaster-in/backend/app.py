"""
Emergency Call Ingestion, Denoising, ASR, Translation, LLM Intelligence, & Dispatch Server.
Implements the 7-stage Mission-Critical Architecture:
1. Noisy emergency call
2. Denoiser (Butterworth -> Spectral Gating -> Optional GRU)
3. Gnani ASR: native-language speech -> native-language text
4. Translation: native text -> English text
5. LLM: Classify emergency type, rate severity, extract location/casualties/hazards, output structured JSON
6. Dispatch logic: map type + severity -> teams to send
7. Human operator confirms (recommended for emergencies)
"""

import os
import io
import time
import uuid
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import numpy as np

from denoiser import AudioDenoiser
from asr_service import GnaniASRService
from translation_service import TranslationService
from llm_analyzer import EmergencyLLMAnalyzer
from dispatch_engine import EmergencyDispatchEngine
from config import GNANI_API_KEY, GNANI_TOKEN, GEMINI_API_KEY
from models import (
    EmergencyTypeEnum,
    SeverityLevelEnum,
    ProcessCallResponse,
    OperatorConfirmationRequest,
    DispatchPlan
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

app = FastAPI(
    title="Emergency Call AI Dispatch System",
    description="Multi-Stage Pipeline: Denoiser -> Gnani ASR -> Translation -> LLM -> Dispatch -> Operator Confirmation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services singletons (Gnani key pre-loaded from config)
asr_service = GnaniASRService(api_key=GNANI_API_KEY, token=GNANI_TOKEN)
translation_service = TranslationService(api_key=GNANI_API_KEY, token=GNANI_TOKEN)
llm_analyzer = EmergencyLLMAnalyzer(gemini_api_key=GEMINI_API_KEY)

# In-memory Incident Storage (for live command & control dashboard)
INCIDENT_STORE: Dict[str, Dict[str, Any]] = {}
AUDIO_STORE: Dict[str, bytes] = {}


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "denoiser": "Butterworth + Spectral Gating + PyTorch GRU ready",
            "asr": "Gnani Indic Speech-to-Text ready",
            "translation": "Deep Translator / Gnani ready",
            "llm": "Emergency JSON Intelligence ready",
            "dispatch_engine": "Tactical Multi-Agency Matrix ready"
        }
    }


@app.get("/api/scenarios")
def get_scenarios():
    """Returns preset emergency call scenarios (Hindi, Tamil, Telugu, Kannada, Bengali, English)."""
    meta_path = os.path.join(SAMPLES_DIR, "scenarios_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.get("/api/audio/{category}/{audio_id}")
def get_audio_file(category: str, audio_id: str):
    """Streams original or denoised WAV audio files."""
    if category == "sample":
        file_path = os.path.join(SAMPLES_DIR, audio_id)
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="audio/wav")
    elif category == "processed":
        if audio_id in AUDIO_STORE:
            return Response(content=AUDIO_STORE[audio_id], media_type="audio/wav")
        file_path = os.path.join(PROCESSED_DIR, f"{audio_id}.wav")
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="audio/wav")

    raise HTTPException(status_code=404, detail="Audio file not found")


@app.post("/api/process-audio")
async def process_emergency_call(
    audio_file: Optional[UploadFile] = File(None),
    scenario_id: Optional[str] = Form(None),
    enable_butterworth: bool = Form(True),
    enable_spectral_gating: bool = Form(True),
    enable_gru: bool = Form(True),
    language_hint: str = Form("auto"),
    gnani_api_key: Optional[str] = Form(None),
    gnani_token: Optional[str] = Form(None),
    gemini_api_key: Optional[str] = Form(None)
):
    """
    Full 7-Step Pipeline Execution:
    1. Ingestion: Reads noisy emergency audio (upload or preset scenario)
    2. Denoiser: Butterworth bandpass -> Spectral Gating -> PyTorch GRU
    3. Gnani ASR: Native speech -> Native transcript
    4. Translation: Native transcript -> English transcript
    5. LLM: Structured JSON extraction (type, severity, casualties, location, hazards)
    6. Dispatch Engine: Automated tactical unit & agency routing
    7. Queued for Step 7: Human Operator Confirmation Console
    """
    call_id = f"CALL-{uuid.uuid4().hex[:8].upper()}"
    t_start = time.time()

    # 1. Ingestion
    audio_np = None
    sr = 16000

    if scenario_id:
        # Load from preset samples
        sample_path = os.path.join(SAMPLES_DIR, f"{scenario_id}.wav")
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail=f"Scenario sample {scenario_id} not found")
        data, file_sr = sf.read(sample_path, dtype='float32')
        audio_np = data
        sr = file_sr
    elif audio_file:
        content = await audio_file.read()
        try:
            data, file_sr = sf.read(io.BytesIO(content), dtype='float32')
            audio_np = data
            sr = file_sr
        except Exception:
            # Try pydub for webm / mp3 / m4a from browser media recorder
            from pydub import AudioSegment
            audio_seg = AudioSegment.from_file(io.BytesIO(content))
            audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
            raw = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
            audio_np = raw / 32768.0
            sr = 16000
    else:
        raise HTTPException(status_code=400, detail="Either audio_file or scenario_id must be provided")

    # Store original audio for playback
    raw_wav_bytes = AudioDenoiser.to_wav_bytes(audio_np, sr)
    AUDIO_STORE[f"{call_id}_raw"] = raw_wav_bytes

    # 2. Denoiser (Butterworth -> Spectral Gating -> GRU)
    denoise_result = AudioDenoiser.process_pipeline(
        audio=audio_np,
        sr=sr,
        enable_butterworth=enable_butterworth,
        enable_spectral_gating=enable_spectral_gating,
        enable_gru=enable_gru
    )
    denoised_audio = denoise_result["final_audio"]
    denoised_wav_bytes = AudioDenoiser.to_wav_bytes(denoised_audio, sr)
    AUDIO_STORE[f"{call_id}_denoised"] = denoised_wav_bytes

    # 3. Gnani ASR: Native Speech -> Native Text
    # If a specific scenario was selected, match language hint
    if scenario_id:
        for suffix, code in [("_hi", "hi"), ("_ta", "ta"), ("_te", "te"), ("_kn", "kn"), ("_bn", "bn"), ("_en", "en")]:
            if scenario_id.endswith(suffix):
                language_hint = code
                break

    asr_result = asr_service.transcribe(
        audio_data=denoised_audio,
        sr=sr,
        target_lang=language_hint,
        api_key_override=gnani_api_key or GNANI_API_KEY,
        token_override=gnani_token or GNANI_TOKEN
    )
    native_transcript = asr_result["native_transcript"]
    detected_lang = asr_result["detected_language"]

    # 4. Translation: Native Text -> English Text
    trans_result = translation_service.translate_to_english(native_transcript, source_lang=detected_lang)
    english_translation = trans_result["translated_text"]

    # 5. LLM Analysis: Text -> Structured JSON
    llm_result = llm_analyzer.analyze_text(
        text=english_translation,
        gemini_key_override=gemini_api_key
    )

    # 6. Dispatch Engine: Map Type + Severity -> Tactical Teams
    dispatch_plan = EmergencyDispatchEngine.generate_dispatch_plan(llm_result)

    elapsed_ms = round((time.time() - t_start) * 1000, 1)

    # 7. Store Incident in Memory (Status: PENDING_CONFIRMATION)
    incident_record = {
        "call_id": call_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time_ms": elapsed_ms,
        "audio_metrics": denoise_result["metrics"],
        "waveforms": denoise_result["waveforms"],
        "original_audio_url": f"/api/audio/processed/{call_id}_raw",
        "denoised_audio_url": f"/api/audio/processed/{call_id}_denoised",
        "detected_language": detected_lang,
        "asr_provider": asr_result.get("provider", "Gnani ASR"),
        "native_transcript": native_transcript,
        "english_translation": english_translation,
        "translation_provider": trans_result.get("provider", "Gnani / Deep Neural Translator"),
        "llm_analysis": llm_result.model_dump(),
        "dispatch_plan": dispatch_plan.model_dump(),
        "status": "PENDING_CONFIRMATION",
        "operator_action": None
    }
    INCIDENT_STORE[call_id] = incident_record

    return incident_record


@app.post("/api/operator/confirm")
def operator_confirm_dispatch(payload: OperatorConfirmationRequest):
    """
    Step 7 in Architecture: Human Operator Confirms.
    Validates, edits, and authorizes the AI dispatch plan.
    Logs operator signature and broadcasts tactical alert to field units.
    """
    call_id = payload.call_id
    if call_id not in INCIDENT_STORE:
        raise HTTPException(status_code=404, detail="Incident Call ID not found")

    incident = INCIDENT_STORE[call_id]

    # Update state with human operator overrides
    incident["status"] = "DISPATCHED" if payload.action == "DISPATCH" else payload.action
    incident["operator_action"] = {
        "operator_id": payload.operator_id,
        "confirmed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": payload.action,
        "confirmed_emergency_type": payload.confirmed_emergency_type.value,
        "confirmed_severity": payload.confirmed_severity.value,
        "confirmed_location": payload.confirmed_location,
        "confirmed_people_count": payload.confirmed_people_count,
        "modified_units": payload.modified_units or [u["unit_type"] for u in incident["dispatch_plan"]["recommended_units"]],
        "operator_notes": payload.operator_notes or "Immediate response authorized by Incident Dispatch Controller.",
        "broadcast_code": f"TX-DISP-{uuid.uuid4().hex[:6].upper()}"
    }

    # Reflect overrides back in analysis
    incident["llm_analysis"]["emergency_type"] = payload.confirmed_emergency_type.value
    incident["llm_analysis"]["severity"] = payload.confirmed_severity.value
    incident["llm_analysis"]["location"] = payload.confirmed_location

    return {
        "success": True,
        "call_id": call_id,
        "status": incident["status"],
        "operator_action": incident["operator_action"],
        "dispatch_plan": incident["dispatch_plan"]
    }


@app.get("/api/incidents")
def get_all_incidents():
    """Returns audit feed of all processed calls for command center log."""
    return list(INCIDENT_STORE.values())


@app.get("/api/incidents/{call_id}")
def get_incident(call_id: str):
    if call_id in INCIDENT_STORE:
        return INCIDENT_STORE[call_id]
    raise HTTPException(status_code=404, detail="Incident not found")


# Mount static frontend
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
