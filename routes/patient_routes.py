"""
Patient management endpoints for the Healthcare System.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db, get_patient_history
from utils.logging import get_logger

logger = get_logger("patient_endpoints")
router = APIRouter(prefix="/patients", tags=["Patient Management"])


@router.get("/{patient_id}/history")
async def get_patient_complete_history(patient_id: str, db: Session = Depends(get_db)):
    """Get complete patient history from database."""
    try:
        history = get_patient_history(db, patient_id)

        return {
            "patient_id": patient_id,
            "patient_info": history["patient"].__dict__ if history["patient"] else None,
            "voice_analyses_count": len(history["voice_analyses"]),
            "vision_analyses_count": len(history["vision_analyses"]),
            "diagnostic_tests_count": len(history["diagnostic_tests"]),
            "recent_voice_analyses": [
                {
                    "id": va.id,
                    "transcription": va.transcription,
                    "sentiment": va.sentiment,
                    "created_at": va.created_at
                } for va in history["voice_analyses"][-5:]  # Last 5
            ],
            "recent_vision_analyses": [
                {
                    "id": va.id,
                    "filename": va.filename,
                    "diagnosis": va.diagnosis,
                    "confidence": va.confidence,
                    "created_at": va.created_at
                } for va in history["vision_analyses"][-5:]  # Last 5
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get patient history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get patient history: {str(e)}")
