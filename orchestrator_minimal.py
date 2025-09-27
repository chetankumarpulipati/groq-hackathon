"""
Simplified orchestrator that can run immediately without heavy ML dependencies.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4
from agents.voice_agent_minimal import VoiceAgent
from config.settings import config
from utils.logging import get_logger

logger = get_logger("orchestrator_minimal")


class HealthcareOrchestrator:
    """
    Minimal orchestration system to get the healthcare system running quickly.
    """

    def __init__(self):
        self.session_id = str(uuid4())
        self.agents = {}
        self.active_sessions = {}
        self.system_status = "initializing"
        self._initialize_agents()
        logger.info(f"HealthcareOrchestrator initialized (minimal mode)")

    def _initialize_agents(self):
        """Initialize minimal agents."""
        try:
            self.agents = {
                "voice": VoiceAgent()
            }
            self.system_status = "ready"
            logger.info(f"Minimal agents initialized: {list(self.agents.keys())}")
        except Exception as e:
            self.system_status = "error"
            logger.error(f"Agent initialization failed: {e}")

    async def process_healthcare_request(self, request_data: Dict[str, Any], workflow_type: str = "simple") -> Dict[str, Any]:
        """Process a simple healthcare request."""
        session_id = str(uuid4())
        start_time = datetime.utcnow()

        logger.info(f"Processing healthcare request: {session_id}")

        try:
            # Simple processing for now
            voice_agent = self.agents["voice"]

            # Extract patient symptoms or voice input
            patient_data = request_data.get("patient_data", {})
            symptoms = patient_data.get("symptoms", [])
            voice_data = request_data.get("voice_data", "")

            # Process the input
            if voice_data:
                result = await voice_agent.execute_task({"input": voice_data})
            elif symptoms:
                symptom_text = " ".join([s.get("name", "") for s in symptoms])
                result = await voice_agent.execute_task({"input": f"Patient reports: {symptom_text}"})
            else:
                result = {"result": {"analysis": "No symptoms or voice data provided"}}

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            return {
                "session_id": session_id,
                "status": "success",
                "workflow_type": workflow_type,
                "results": {"voice_analysis": result},
                "duration": duration,
                "timestamp": end_time.isoformat()
            }

        except Exception as e:
            logger.error(f"Healthcare request failed: {e}")
            return {
                "session_id": session_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "system_status": self.system_status,
            "orchestrator_session_id": self.session_id,
            "active_sessions": len(self.active_sessions),
            "agents": {name: agent.get_status() for name, agent in self.agents.items()},
            "database": {"overall": "disabled"},
            "api_gateway": {"overall": "disabled"},
            "mode": "minimal",
            "timestamp": datetime.utcnow().isoformat()
        }

    async def shutdown(self):
        """Shutdown orchestrator."""
        self.system_status = "shutdown"
        logger.info("System shutdown completed")


# Global orchestrator instance
orchestrator = HealthcareOrchestrator()
