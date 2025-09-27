"""
Voice processing endpoints for the Healthcare System.
"""

from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from models.api_models import VoiceAnalysisResponse
from services.health_report_service import analyze_medical_voice_content
from database import get_db, store_voice_analysis
from utils.logging import get_logger

logger = get_logger("voice_endpoints")
router = APIRouter(prefix="/voice", tags=["Voice Processing"])


@router.post("/process", response_model=VoiceAnalysisResponse)
async def process_voice_audio(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    patient_id: Optional[str] = Form("PATIENT_001"),
    db: Session = Depends(get_db)
):
    """Process voice audio file or text input for medical analysis with database storage."""
    try:
        logger.info("Processing voice input for medical analysis")

        # Initialize response data
        analysis_result = {
            "transcription": None,
            "sentiment": "neutral",
            "sentiment_confidence": 0.85,
            "stress_level": "normal",
            "stress_confidence": 0.80,
            "quality": "good",
            "clarity_score": 0.85,
            "medical_indicators": [],
            "recommendations": None
        }

        input_text = ""

        # Handle audio file upload
        if audio:
            logger.info(f"Processing audio file: {audio.filename}")
            # For demo purposes, simulate audio transcription
            analysis_result["transcription"] = "Patient reports chest pain and difficulty breathing"
            input_text = analysis_result["transcription"]

        # Handle text input
        elif text:
            logger.info("Processing text input for voice analysis")
            input_text = text
            analysis_result["transcription"] = text

        else:
            raise HTTPException(status_code=400, detail="Either audio file or text input is required")

        # Perform medical voice analysis
        if input_text:
            try:
                from orchestrator_minimal import comprehensive_orchestrator as orchestrator
                voice_agent = orchestrator.agents["voice"]
                voice_result = await voice_agent.process({
                    "audio_text": input_text,
                    "analysis_type": "medical_voice_analysis"
                })
            except Exception as e:
                logger.warning(f"Voice agent processing failed, using fallback: {e}")

            # Enhanced medical analysis
            medical_analysis = await analyze_medical_voice_content(input_text)

            # Update analysis result with medical insights
            analysis_result.update({
                "sentiment": medical_analysis.get("sentiment", "neutral"),
                "sentiment_confidence": medical_analysis.get("sentiment_confidence", 0.85),
                "stress_level": medical_analysis.get("stress_level", "normal"),
                "stress_confidence": medical_analysis.get("stress_confidence", 0.80),
                "medical_indicators": medical_analysis.get("medical_indicators", []),
                "recommendations": medical_analysis.get("recommendations", "Continue monitoring symptoms")
            })

        # Store analysis in database
        try:
            stored_analysis = store_voice_analysis(db, patient_id, analysis_result)
            logger.info(f"Voice analysis stored in database with ID: {stored_analysis.id}")
        except Exception as e:
            logger.error(f"Failed to store voice analysis in database: {e}")

        logger.info("Voice analysis completed successfully")
        return VoiceAnalysisResponse(**analysis_result)

    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")
