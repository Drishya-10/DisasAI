"""
Pydantic Data Schemas for Emergency Dispatch Pipeline
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class EmergencyTypeEnum(str, Enum):
    FLOOD = "FLOOD"
    FIRE = "FIRE"
    MEDICAL = "MEDICAL"
    BUILDING_COLLAPSE = "BUILDING_COLLAPSE"
    ROAD_ACCIDENT = "ROAD_ACCIDENT"
    GAS_LEAK = "GAS_LEAK"
    CRIME_VIOLENCE = "CRIME_VIOLENCE"
    OTHER = "OTHER"


class SeverityLevelEnum(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class DispatchUnit(BaseModel):
    unit_type: str
    quantity: int
    call_sign: str
    station: str
    equipment: List[str] = Field(default_factory=list)


class LLMExtractionResult(BaseModel):
    emergency_type: EmergencyTypeEnum
    severity: SeverityLevelEnum
    severity_score: int = Field(ge=1, le=5, description="1 (Minor) to 5 (Catastrophic/Life-Threatening)")
    location: str
    landmark: Optional[str] = None
    city: Optional[str] = None
    people_trapped: int = 0
    people_injured: int = 0
    people_at_risk: int = 0
    key_details: List[str] = Field(default_factory=list)
    hazards: List[str] = Field(default_factory=list)
    caller_state: str = "Distressed"
    summary: str
    confidence: float = 0.95


class DispatchPlan(BaseModel):
    plan_id: str
    priority_level: str
    emergency_type: str
    primary_agency: str
    backup_agency: Optional[str] = None
    recommended_units: List[DispatchUnit]
    total_responders: int
    estimated_eta_minutes: int
    radio_channel: str
    depot_assigned: str
    equipment_checklist: List[str]
    critical_protocols: List[str]


class ProcessCallResponse(BaseModel):
    call_id: str
    audio_metrics: Dict[str, Any]
    waveforms: Dict[str, List[float]]
    original_audio_url: str
    denoised_audio_url: str
    detected_language: str
    native_transcript: str
    english_translation: str
    llm_analysis: LLMExtractionResult
    dispatch_plan: DispatchPlan
    status: str = "PENDING_CONFIRMATION"


class OperatorConfirmationRequest(BaseModel):
    call_id: str
    operator_id: str = "OP-408"
    confirmed_emergency_type: EmergencyTypeEnum
    confirmed_severity: SeverityLevelEnum
    confirmed_location: str
    confirmed_people_count: int
    modified_units: Optional[List[str]] = None
    operator_notes: Optional[str] = None
    action: str = "DISPATCH"  # "DISPATCH", "ESCALATE", "STANDBY", "REJECT"
