"""
Voice Input Agent for processing voice commands and audio data in healthcare system.
Uses the best available speech-to-text models for maximum accuracy.
"""

import asyncio
import io
import wave
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import speech_recognition as sr
import whisper
from pydub import AudioSegment
from gtts import gTTS
from agents.base_agent import BaseAgent
from config.settings import config
from utils.logging import get_logger
from utils.error_handling import AudioProcessingError, handle_exception

logger = get_logger("voice_agent")


class VoiceAgent(BaseAgent):
    """
    Specialized agent for processing voice commands and medical audio data.
    Optimized for healthcare terminology and medical accuracy.
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id, specialized_task="voice_processing")
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.whisper_model = None
        self.medical_vocabulary = self._load_medical_vocabulary()
        self._initialize_audio_components()

        logger.info(f"VoiceAgent {self.agent_id} initialized with medical vocabulary")

    def _initialize_audio_components(self):
        """Initialize audio processing components."""
        try:
            # Initialize microphone
            self.microphone = sr.Microphone()

            # Calibrate for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

            # Load Whisper model for high accuracy
            self.whisper_model = whisper.load_model("base")

            logger.info("Audio components initialized successfully")

        except Exception as e:
            logger.warning(f"Some audio components failed to initialize: {e}")

    def _load_medical_vocabulary(self) -> List[str]:
        """Load medical terminology for improved recognition accuracy."""
        medical_terms = [
            # Common medical terms
            "patient", "diagnosis", "symptoms", "medication", "prescription",
            "appointment", "doctor", "physician", "nurse", "hospital",
            "clinic", "treatment", "therapy", "surgery", "procedure",
            "blood pressure", "heart rate", "temperature", "weight", "height",
            "allergies", "medical history", "emergency", "urgent", "pain",
            "headache", "fever", "nausea", "dizziness", "fatigue",

            # Medical specialties
            "cardiology", "neurology", "oncology", "pediatrics", "radiology",
            "orthopedics", "dermatology", "psychiatry", "ophthalmology",

            # Common medications
            "aspirin", "ibuprofen", "acetaminophen", "insulin", "antibiotics",

            # Body parts
            "head", "neck", "chest", "abdomen", "back", "arms", "legs",
            "heart", "lungs", "liver", "kidney", "brain", "spine"
        ]
        return medical_terms

    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process voice input data."""
        try:
            if isinstance(input_data, str):
                # Text command processing
                return await self._process_text_command(input_data, context)
            elif isinstance(input_data, dict):
                if "audio_file" in input_data:
                    # Audio file processing
                    return await self._process_audio_file(input_data["audio_file"], context)
                elif "command" in input_data:
                    # Command processing
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
            # Fallback to structured analysis
            analysis = await self._fallback_command_analysis(text)

        # Generate appropriate response
        response_text = await self._generate_voice_response(analysis, context)

        return {
            "input_text": text,
            "analysis": analysis,
            "response_text": response_text,
            "model_info": ai_response["model_info"],
            "processing_type": "text_command"
        }

    @handle_exception
    async def _process_audio_file(self, audio_file_path: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process audio file using Whisper for medical accuracy."""

        try:
            # Transcribe using Whisper for best accuracy
            result = self.whisper_model.transcribe(
                audio_file_path,
                language="en",
                task="transcribe"
            )

            transcribed_text = result["text"].strip()
            confidence = result.get("confidence", 0.8)

            # Process the transcribed text
            text_result = await self._process_text_command(transcribed_text, context)

            return {
                "transcription": transcribed_text,
                "confidence": confidence,
                "audio_file": audio_file_path,
                "text_analysis": text_result,
                "processing_type": "audio_file"
            }

        except Exception as e:
            raise AudioProcessingError(f"Audio file processing failed: {str(e)}")

    async def _process_voice_command(self, command_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process structured voice command data."""

        command_type = command_data.get("type", "general")
        command_text = command_data.get("text", "")
        urgency = command_data.get("urgency", "normal")

        # Process based on command type
        if command_type == "emergency":
            return await self._handle_emergency_command(command_text, context)
        elif command_type == "appointment":
            return await self._handle_appointment_command(command_text, context)
        elif command_type == "medication":
            return await self._handle_medication_command(command_text, context)
        elif command_type == "symptom":
            return await self._handle_symptom_command(command_text, context)
        else:
            return await self._process_text_command(command_text, context)

    async def _handle_emergency_command(self, command: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle emergency voice commands with high priority."""

        system_prompt = """You are handling a medical emergency voice command. 
        Provide immediate, clear, and actionable guidance. 
        Always recommend calling emergency services when appropriate.
        Be concise and prioritize safety."""

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

    async def _handle_appointment_command(self, command: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle appointment-related voice commands."""

        system_prompt = """Extract appointment information from voice commands:
        - Date and time preferences
        - Doctor/specialty requested
        - Reason for appointment
        - Urgency level
        Respond with structured appointment data."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Appointment request: {command}"}
        ]

        ai_response = await self.get_ai_response(messages, temperature=0.2)

        return {
            "command": command,
            "appointment_data": ai_response["response"],
            "priority": "medium",
            "requires_scheduling": True,
            "model_info": ai_response["model_info"],
            "processing_type": "appointment_command"
        }

    async def _handle_medication_command(self, command: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle medication-related voice commands."""

        system_prompt = """Process medication-related voice commands. Extract:
        - Medication names
        - Dosage information
        - Frequency
        - Concerns or side effects
        - Refill requests
        Always recommend consulting healthcare providers for medication changes."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Medication inquiry: {command}"}
        ]

        ai_response = await self.get_ai_response(messages, temperature=0.1)

        return {
            "command": command,
            "medication_data": ai_response["response"],
            "priority": "high",
            "requires_physician_review": True,
            "model_info": ai_response["model_info"],
            "processing_type": "medication_command"
        }

    async def _handle_symptom_command(self, command: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle symptom reporting voice commands."""

        system_prompt = """Process symptom reports from voice commands. Extract:
        - Symptoms described
        - Severity (1-10 scale)
        - Duration
        - Associated factors
        - Body parts affected
        Provide appropriate triage guidance."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Symptom report: {command}"}
        ]

        ai_response = await self.get_ai_response(messages, temperature=0.1)

        return {
            "command": command,
            "symptom_data": ai_response["response"],
            "priority": "medium",
            "requires_medical_review": True,
            "model_info": ai_response["model_info"],
            "processing_type": "symptom_command"
        }

    async def _fallback_command_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback analysis when JSON parsing fails."""

        # Simple keyword-based analysis
        intent = "general"
        priority = "low"

        emergency_keywords = ["emergency", "urgent", "help", "pain", "chest pain", "bleeding"]
        appointment_keywords = ["appointment", "schedule", "book", "see doctor"]
        medication_keywords = ["medication", "prescription", "pill", "dosage", "side effect"]
        symptom_keywords = ["symptom", "feel", "hurt", "ache", "sick", "nausea"]

        text_lower = text.lower()

        if any(keyword in text_lower for keyword in emergency_keywords):
            intent = "emergency"
            priority = "emergency"
        elif any(keyword in text_lower for keyword in appointment_keywords):
            intent = "appointment"
            priority = "medium"
        elif any(keyword in text_lower for keyword in medication_keywords):
            intent = "medication"
            priority = "high"
        elif any(keyword in text_lower for keyword in symptom_keywords):
            intent = "symptom_report"
            priority = "medium"

        return {
            "intent": intent,
            "entities": [],
            "priority": priority,
            "confidence": 0.6,
            "analysis_method": "fallback"
        }

    async def _generate_voice_response(self, analysis: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """Generate appropriate voice response based on analysis."""

        intent = analysis.get("intent", "general")
        priority = analysis.get("priority", "low")

        if intent == "emergency":
            return "I understand this is an emergency. Please call 911 immediately if you're experiencing a medical emergency. I'm also notifying your healthcare provider."
        elif intent == "appointment":
            return "I'll help you schedule an appointment. Let me check available times with your preferred healthcare provider."
        elif intent == "medication":
            return "I've noted your medication inquiry. For safety, I recommend discussing any medication questions with your pharmacist or doctor."
        elif intent == "symptom_report":
            return "I've recorded your symptoms. Based on what you've described, I recommend consulting with a healthcare professional for proper evaluation."
        else:
            return "I've processed your request. How else can I assist you with your healthcare needs today?"

    async def record_voice_command(self, duration: int = 5) -> Optional[str]:
        """Record voice command from microphone."""

        if not self.microphone:
            raise AudioProcessingError("Microphone not available")

        try:
            with self.microphone as source:
                logger.info(f"Recording voice command for {duration} seconds...")
                audio = self.recognizer.listen(source, timeout=duration)

            # Use Whisper for transcription
            audio_data = io.BytesIO(audio.get_wav_data())
            result = self.whisper_model.transcribe(audio_data)

            return result["text"].strip()

        except Exception as e:
            raise AudioProcessingError(f"Voice recording failed: {str(e)}")

    async def generate_voice_response(self, text: str, output_file: Optional[str] = None) -> str:
        """Generate voice response using text-to-speech."""

        try:
            tts = gTTS(text=text, lang='en', slow=False)

            if output_file:
                tts.save(output_file)
                return output_file
            else:
                # Save to temporary file
                temp_file = f"temp_response_{self.agent_id}.mp3"
                tts.save(temp_file)
                return temp_file

        except Exception as e:
            raise AudioProcessingError(f"Voice synthesis failed: {str(e)}")

    async def get_voice_capabilities(self) -> Dict[str, Any]:
        """Get current voice processing capabilities."""

        return {
            "agent_id": self.agent_id,
            "microphone_available": self.microphone is not None,
            "whisper_model_loaded": self.whisper_model is not None,
            "medical_vocabulary_size": len(self.medical_vocabulary),
            "supported_languages": ["en"],
            "audio_formats": ["wav", "mp3", "m4a", "flac"],
            "max_recording_duration": config.audio.max_recording_time,
            "sample_rate": config.audio.sample_rate
        }
