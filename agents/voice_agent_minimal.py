"""
Minimal Voice Agent for basic voice processing without heavy ML dependencies.
Can be upgraded later with Whisper and other advanced features.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Union
from agents.base_agent import BaseAgent
from config.settings import config
from utils.logging import get_logger
from utils.error_handling import AudioProcessingError, handle_exception

logger = get_logger("voice_agent_minimal")


class VoiceAgent(BaseAgent):
    """
    Minimal voice agent that processes text commands without heavy ML dependencies.
    Can be upgraded later with Whisper and speech recognition features.
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id, specialized_task="voice_processing")
        self.medical_vocabulary = self._load_medical_vocabulary()
        logger.info(f"VoiceAgent {self.agent_id} initialized (minimal mode)")

    def _load_medical_vocabulary(self) -> List[str]:
        """Load medical terminology for improved recognition accuracy."""
        return [
            "patient", "diagnosis", "symptoms", "medication", "prescription",
            "appointment", "doctor", "physician", "nurse", "hospital",
            "clinic", "treatment", "therapy", "surgery", "procedure",
            "blood pressure", "heart rate", "temperature", "weight", "height",
            "allergies", "medical history", "emergency", "urgent", "pain",
            "headache", "fever", "nausea", "dizziness", "fatigue"
        ]

    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process voice input data (text-based for now)."""
        try:
            if isinstance(input_data, str):
                return await self._process_text_command(input_data, context)
            elif isinstance(input_data, dict):
                if "command" in input_data:
                    return await self._process_voice_command(input_data["command"], context)

            raise AudioProcessingError("Invalid input format for voice processing")

        except Exception as e:
            logger.error(f"Voice processing failed: {str(e)}")
            raise AudioProcessingError(f"Voice processing error: {str(e)}")

    @handle_exception
    async def _process_text_command(self, text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process text command using AI for healthcare intent recognition."""

        system_prompt = """You are a healthcare voice assistant AI. Analyze voice commands and extract:
        1. Intent (appointment, medication, symptom_report, emergency, information_request)
        2. Entities (dates, times, medications, symptoms, body parts)
        3. Priority level (low, medium, high, emergency)
        4. Required actions
        5. Confidence score
        
        Respond in JSON format with structured data."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this healthcare voice command: '{text}'"}
        ]

        ai_response = await self.get_ai_response(messages, temperature=0.1)

        try:
            analysis = json.loads(ai_response["response"])
        except json.JSONDecodeError:
            analysis = await self._fallback_command_analysis(text)

        response_text = await self._generate_voice_response(analysis, context)

        return {
            "input_text": text,
            "analysis": analysis,
            "response_text": response_text,
            "model_info": ai_response["model_info"],
            "processing_type": "text_command"
        }

    async def _process_voice_command(self, command_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process structured voice command data."""
        command_type = command_data.get("type", "general")
        command_text = command_data.get("text", "")

        if command_type == "emergency":
            return await self._handle_emergency_command(command_text, context)
        else:
            return await self._process_text_command(command_text, context)

    async def _handle_emergency_command(self, command: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle emergency voice commands with high priority."""

        system_prompt = """You are handling a medical emergency voice command. 
        Provide immediate, clear, and actionable guidance. 
        Always recommend calling emergency services when appropriate."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"EMERGENCY: {command}"}
        ]

        ai_response = await self.get_ai_response(messages, temperature=0.0)

        return {
            "command": command,
            "response": ai_response["response"],
            "priority": "EMERGENCY",
            "requires_immediate_action": True,
            "emergency_services_recommended": True,
            "model_info": ai_response["model_info"],
            "processing_type": "emergency_command"
        }

    async def _fallback_command_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback analysis when JSON parsing fails."""
        text_lower = text.lower()

        if any(word in text_lower for word in ["emergency", "urgent", "help", "pain", "chest pain"]):
            intent = "emergency"
            priority = "emergency"
        elif any(word in text_lower for word in ["appointment", "schedule", "book"]):
            intent = "appointment"
            priority = "medium"
        elif any(word in text_lower for word in ["medication", "prescription", "pill"]):
            intent = "medication"
            priority = "high"
        else:
            intent = "general"
            priority = "low"

        return {
            "intent": intent,
            "entities": [],
            "priority": priority,
            "confidence": 0.7
        }

    async def _generate_voice_response(self, analysis: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """Generate appropriate voice response based on analysis."""
        intent = analysis.get("intent", "general")

        if intent == "emergency":
            return "I understand this is an emergency. Please call 911 immediately if you're experiencing a medical emergency."
        elif intent == "appointment":
            return "I'll help you with your appointment request. Let me process this for you."
        elif intent == "medication":
            return "I've noted your medication inquiry. For safety, I recommend discussing medication questions with your healthcare provider."
        else:
            return "I've processed your request. How else can I assist you with your healthcare needs?"

    async def get_voice_capabilities(self) -> Dict[str, Any]:
        """Get current voice processing capabilities."""
        return {
            "agent_id": self.agent_id,
            "mode": "minimal_text_processing",
            "medical_vocabulary_size": len(self.medical_vocabulary),
            "supported_intents": ["emergency", "appointment", "medication", "general"],
            "ai_powered_analysis": True,
            "upgrade_available": "Install whisper and speech_recognition for full audio support"
        }
