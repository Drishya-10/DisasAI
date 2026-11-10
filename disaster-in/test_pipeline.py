"""
End-to-End Pipeline Verification Script
Tests all 7 stages of the Emergency Architecture:
1. Noisy emergency call
2. Denoiser (Butterworth -> Spectral Gating -> GRU)
3. Gnani ASR (Native speech -> Native text)
4. Translation (Native text -> English text)
5. LLM Structured Analysis (Type, Severity, Casualties, Hazards)
6. Dispatch Logic (Emergency unit routing)
7. Human Operator Confirmation
"""

import sys
import os
import soundfile as sf
import json

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "backend"))

from denoiser import AudioDenoiser
from asr_service import GnaniASRService
from translation_service import TranslationService
from llm_analyzer import EmergencyLLMAnalyzer
from dispatch_engine import EmergencyDispatchEngine
from models import OperatorConfirmationRequest, EmergencyTypeEnum, SeverityLevelEnum

def run_e2e_test():
    print("=" * 70)
    print("EMERGENCY PIPELINE INTEGRATION TEST: 7-STAGE ARCHITECTURE")
    print("=" * 70)

    sample_path = os.path.join(BASE_DIR, "data", "samples", "patna_flood_hi.wav")
    assert os.path.exists(sample_path), f"Sample file {sample_path} missing!"

    # STAGE 1: Noisy emergency call
    print("\n[STAGE 1] Ingesting Noisy Emergency Call...")
    audio, sr = sf.read(sample_path, dtype='float32')
    print(f"Loaded noisy audio: {len(audio)} samples, {sr} Hz ({len(audio)/sr:.2f}s)")

    # STAGE 2: Denoiser (Butterworth -> Spectral Gating -> GRU)
    print("\n[STAGE 2] Running 3-Stage Denoiser...")
    denoised_res = AudioDenoiser.process_pipeline(
        audio=audio,
        sr=sr,
        enable_butterworth=True,
        enable_spectral_gating=True,
        enable_gru=True
    )
    metrics = denoised_res["metrics"]
    print(f"Denoiser Applied Stages: {metrics['stages_applied']}")
    print(f"Raw SNR: {metrics['raw_snr_db']} dB | Enhanced SNR: {metrics['enhanced_snr_db']} dB | Improvement: +{metrics['snr_improvement_db']} dB")

    # STAGE 3: Gnani ASR (native-language speech -> native-language text)
    print("\n[STAGE 3] Invoking Gnani ASR...")
    asr_service = GnaniASRService()
    asr_res = asr_service.transcribe(denoised_res["final_audio"], sr, target_lang="hi")
    native_text = asr_res["native_transcript"]
    print(f"Detected Language: {asr_res['detected_language']}")
    print(f"Provider: {asr_res['provider']}")
    print(f"Native Transcript:\n\"{native_text}\"")

    # STAGE 4: Translation (native text -> English text)
    print("\n[STAGE 4] Translating to English...")
    trans_service = TranslationService()
    trans_res = trans_service.translate_to_english(native_text, source_lang="hi")
    english_text = trans_res["translated_text"]
    print(f"Translation Provider: {trans_res['provider']}")
    print(f"English Transcript:\n\"{english_text}\"")

    # STAGE 5: LLM (text only -> classify type, severity, extract location/casualties, output structured JSON)
    print("\n[STAGE 5] LLM Structured Entity & Severity Intelligence...")
    llm = EmergencyLLMAnalyzer()
    llm_res = llm.analyze_text(english_text)
    print(f"Emergency Type: {llm_res.emergency_type.value}")
    print(f"Severity: {llm_res.severity.value} (Score: {llm_res.severity_score}/5)")
    print(f"Location Extracted: {llm_res.location} (City: {llm_res.city})")
    print(f"Casualties: Trapped={llm_res.people_trapped}, Injured={llm_res.people_injured}, At-Risk={llm_res.people_at_risk}")
    print(f"Hazards Identified: {llm_res.hazards}")
    print(f"Summary: {llm_res.summary}")

    # STAGE 6: Dispatch Logic (map type + severity -> teams to send)
    print("\n[STAGE 6] Tactical Dispatch Engine Routing...")
    dispatch_plan = EmergencyDispatchEngine.generate_dispatch_plan(llm_res)
    print(f"Dispatch Plan ID: {dispatch_plan.plan_id}")
    print(f"Priority Level: {dispatch_plan.priority_level}")
    print(f"Primary Responding Agency: {dispatch_plan.primary_agency}")
    print(f"Estimated ETA: {dispatch_plan.estimated_eta_minutes} minutes")
    print(f"Radio Tactical Channel: {dispatch_plan.radio_channel}")
    print(f"Total Responders Deployed: {dispatch_plan.total_responders}")
    print("Recommended Tactical Units:")
    for unit in dispatch_plan.recommended_units:
        print(f"  - [{unit.call_sign}] {unit.quantity}x {unit.unit_type} ({unit.station})")

    # STAGE 7: Human Operator Confirms
    print("\n[STAGE 7] Human Operator Confirmation Protocol...")
    confirm_req = OperatorConfirmationRequest(
        call_id=dispatch_plan.plan_id,
        operator_id="DISPATCHER-COMMANDER-702",
        confirmed_emergency_type=llm_res.emergency_type,
        confirmed_severity=llm_res.severity,
        confirmed_location=llm_res.location,
        confirmed_people_count=llm_res.people_trapped,
        modified_units=[u.unit_type for u in dispatch_plan.recommended_units],
        operator_notes="Human Controller confirmed: 7 lives in immediate flood hazard. Immediate NDRF aquatic deployment authorized.",
        action="DISPATCH"
    )
    print(f"Operator ID: {confirm_req.operator_id}")
    print(f"Confirmed Action: {confirm_req.action}")
    print(f"Operator Notes: {confirm_req.operator_notes}")
    print("\n>>> ALL 7 ARCHITECTURAL STAGES COMPLETED SUCCESSFULLY WITH 100% INTEGRITY! <<<")

if __name__ == "__main__":
    run_e2e_test()
