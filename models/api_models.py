"""
API models and request/response schemas for the Healthcare System.
"""

from pydantic import BaseModel
from typing import Dict, Any, Optional, List


class HealthcareRequest(BaseModel):
    patient_data: Dict[str, Any]
    workflow_type: str = "simple"
    priority: str = "standard"
    voice_data: Optional[str] = None


class VoiceProcessingRequest(BaseModel):
    voice_input: str
    patient_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class AccuracyTestRequest(BaseModel):
    test_cases: Optional[List[Dict[str, Any]]] = None
    categories: Optional[List[str]] = None


class AccuracyResponse(BaseModel):
    overall_accuracy: float
    category_accuracy: Dict[str, float]
    total_cases: int
    correct_predictions: int
    confidence_scores: List[float]
    evaluation_timestamp: str


class VoiceAnalysisResponse(BaseModel):
    transcription: Optional[str] = None
    sentiment: Optional[str] = None
    sentiment_confidence: Optional[float] = None
    stress_level: Optional[str] = None
    stress_confidence: Optional[float] = None
    quality: Optional[str] = None
    clarity_score: Optional[float] = None
    medical_indicators: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[str] = None


class VisionAnalysisResponse(BaseModel):
    diagnosis: Optional[str] = None
    confidence: Optional[float] = None
    quality: Optional[str] = None
    resolution: Optional[str] = None
    processing_time: Optional[str] = None
    findings: Optional[List[Dict[str, Any]]] = None
    measurements: Optional[Dict[str, str]] = None
    recommendations: Optional[str] = None
    model_used: Optional[str] = None
    image_format: Optional[str] = None
    # Enhanced health report fields
    health_report: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    clinical_notes: Optional[str] = None
    urgency_level: Optional[str] = None
    follow_up_required: Optional[bool] = None
    specialist_referral: Optional[str] = None
    patient_education: Optional[List[str]] = None
