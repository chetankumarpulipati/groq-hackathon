"""
Fixed comprehensive orchestrator for the healthcare system with proper agent initialization.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4
from agents.voice_agent_minimal import VoiceAgent
from agents.diagnostic_agent import DiagnosticAgent
from agents.patient_analysis_agent import PatientAnalysisAgent
from agents.medical_record_agent import MedicalRecordAgent
from agents.patient_monitoring_agent import PatientMonitoringAgent
from agents.validation_agent import DataValidationAgent
from config.settings import config
from utils.logging import get_logger

logger = get_logger("healthcare_orchestrator")


class HealthcareOrchestrator:
    """
    Complete orchestration system for healthcare AI with all working agents.
    """

    def __init__(self):
        self.session_id = str(uuid4())
        self.agents = {}
        self.active_sessions = {}
        self.system_status = "initializing"
        self._initialize_agents()
        logger.info(f"HealthcareOrchestrator initialized")

    def _initialize_agents(self):
        """Initialize all working healthcare agents."""
        try:
            # Only initialize agents that we know work
            self.agents = {
                "voice": VoiceAgent(),
                "diagnostic": DiagnosticAgent(),
                "patient_analysis": PatientAnalysisAgent(),
                "medical_record": MedicalRecordAgent(),
                "patient_monitoring": PatientMonitoringAgent(),
                "validation": DataValidationAgent()
            }
            self.system_status = "ready"
            logger.info(f"Healthcare agents initialized successfully: {list(self.agents.keys())}")
        except Exception as e:
            self.system_status = "error"
            logger.error(f"Agent initialization failed: {e}")

    async def process_comprehensive_healthcare_request(self, request_data: Dict[str, Any], workflow_type: str = "comprehensive") -> Dict[str, Any]:
        """Process a comprehensive healthcare request using all agents."""
        session_id = str(uuid4())
        start_time = datetime.utcnow()

        logger.info(f"Processing comprehensive healthcare request: {session_id}")

        try:
            patient_data = request_data.get("patient_data", {})
            request_type = request_data.get("request_type", "general_assessment")

            workflow_results = {}

            # Step 1: Data Validation
            if "validation" in self.agents:
                validation_result = await self.agents["validation"].process({
                    "data": patient_data,
                    "validation_type": "comprehensive"
                })
                workflow_results["validation"] = validation_result

            # Step 2: Patient Data Analysis
            if "patient_analysis" in self.agents:
                analysis_result = await self.agents["patient_analysis"].process({
                    "patient_data": patient_data,
                    "analysis_type": "comprehensive"
                })
                workflow_results["patient_analysis"] = analysis_result

            # Step 3: Medical Record Processing (if record data available)
            if "medical_record" in self.agents and patient_data.get("medical_records"):
                record_result = await self.agents["medical_record"].process({
                    "record_data": patient_data["medical_records"],
                    "processing_type": "comprehensive"
                })
                workflow_results["medical_record_processing"] = record_result

            # Step 4: Diagnostic Analysis
            if "diagnostic" in self.agents:
                diagnostic_result = await self.agents["diagnostic"].process({
                    "patient_data": patient_data,
                    "analysis_type": "comprehensive"
                })
                workflow_results["diagnostic_analysis"] = diagnostic_result

            # Step 5: Patient Monitoring Setup
            if "patient_monitoring" in self.agents:
                monitoring_result = await self.agents["patient_monitoring"].process({
                    "patient_data": patient_data,
                    "monitoring_type": "continuous"
                })
                workflow_results["patient_monitoring"] = monitoring_result

            # Compile comprehensive response
            response = {
                "session_id": session_id,
                "request_type": request_type,
                "workflow_type": workflow_type,
                "patient_id": patient_data.get("patient_id", "unknown"),
                "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                "system_status": self.system_status,
                "workflow_results": workflow_results,
                "summary": self._generate_workflow_summary(workflow_results),
                "recommendations": self._generate_comprehensive_recommendations(workflow_results),
                "next_steps": self._determine_next_steps(workflow_results),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed"
            }

            # Store session
            self.active_sessions[session_id] = response

            logger.info(f"Comprehensive healthcare request completed: {session_id}")
            return response

        except Exception as e:
            logger.error(f"Comprehensive healthcare processing failed: {e}")
            return {
                "session_id": session_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def process_patient_analysis_only(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process only patient data analysis."""
        if "patient_analysis" not in self.agents:
            return {"status": "error", "error": "Patient analysis agent not available"}

        return await self.agents["patient_analysis"].process({
            "patient_data": patient_data,
            "analysis_type": "comprehensive"
        })

    async def process_diagnostic_assistance(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process diagnostic assistance request."""
        if "diagnostic" not in self.agents:
            return {"status": "error", "error": "Diagnostic agent not available"}

        return await self.agents["diagnostic"].process({
            "patient_data": patient_data,
            "analysis_type": "comprehensive"
        })

    async def process_medical_records(self, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process medical record analysis."""
        if "medical_record" not in self.agents:
            return {"status": "error", "error": "Medical record agent not available"}

        return await self.agents["medical_record"].process({
            "record_data": record_data,
            "processing_type": "comprehensive"
        })

    async def start_patient_monitoring(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Start patient monitoring."""
        if "patient_monitoring" not in self.agents:
            return {"status": "error", "error": "Patient monitoring agent not available"}

        return await self.agents["patient_monitoring"].process({
            "patient_data": patient_data,
            "monitoring_type": "continuous"
        })

    def _generate_workflow_summary(self, workflow_results: Dict) -> Dict:
        """Generate summary of workflow results."""
        summary = {
            "agents_executed": len(workflow_results),
            "successful_processes": len([r for r in workflow_results.values() if r.get("status") == "completed"]),
            "alerts_generated": 0,
            "recommendations_count": 0
        }

        # Count alerts and recommendations
        for result in workflow_results.values():
            if isinstance(result, dict) and result.get("status") == "completed":
                # Count alerts from monitoring
                monitoring_result = result.get("monitoring_result", {})
                if "active_alerts" in monitoring_result:
                    summary["alerts_generated"] += len(monitoring_result["active_alerts"])

                # Count recommendations from various agents
                if "recommendations" in result:
                    if isinstance(result["recommendations"], list):
                        summary["recommendations_count"] += len(result["recommendations"])

                # Count recommendations from analysis
                analysis_result = result.get("analysis", {})
                if "recommendations" in analysis_result:
                    if isinstance(analysis_result["recommendations"], list):
                        summary["recommendations_count"] += len(analysis_result["recommendations"])

        return summary

    def _generate_comprehensive_recommendations(self, workflow_results: Dict) -> List[str]:
        """Generate comprehensive recommendations from all workflow results."""
        all_recommendations = []

        # Collect recommendations from all agents
        for agent_name, result in workflow_results.items():
            if isinstance(result, dict) and result.get("status") == "completed":
                # Extract recommendations based on agent type
                if agent_name == "patient_analysis":
                    analysis_result = result.get("analysis", {})
                    recommendations = analysis_result.get("recommendations", [])
                    all_recommendations.extend(recommendations)

                elif agent_name == "diagnostic_analysis":
                    diagnosis = result.get("diagnosis", {})
                    treatment_recs = diagnosis.get("treatment_recommendations", {})
                    if "immediate_interventions" in treatment_recs:
                        all_recommendations.extend(treatment_recs["immediate_interventions"])
                    if "follow_up_instructions" in treatment_recs:
                        all_recommendations.extend(treatment_recs["follow_up_instructions"])

                elif agent_name == "patient_monitoring":
                    monitoring_result = result.get("monitoring_result", {})
                    recommendations = monitoring_result.get("recommendations", [])
                    all_recommendations.extend(recommendations)

                elif agent_name == "medical_record_processing":
                    processing_result = result.get("processing_result", {})
                    recommendations = processing_result.get("recommendations", [])
                    all_recommendations.extend(recommendations)

        # Remove duplicates and return unique recommendations
        return list(set(all_recommendations))

    def _determine_next_steps(self, workflow_results: Dict) -> List[str]:
        """Determine next steps based on workflow results."""
        next_steps = []

        # Analyze results to determine priority actions
        diagnostic_results = workflow_results.get("diagnostic_analysis", {})
        if diagnostic_results.get("status") == "completed":
            diagnosis = diagnostic_results.get("diagnosis", {})
            red_flags = diagnosis.get("red_flag_assessment", {})

            if red_flags.get("overall_urgency") == "critical":
                next_steps.append("Immediate medical evaluation required")
            elif red_flags.get("overall_urgency") == "urgent":
                next_steps.append("Urgent medical consultation needed")
            else:
                next_steps.append("Routine follow-up recommended")

        # Check monitoring alerts
        monitoring_results = workflow_results.get("patient_monitoring", {})
        if monitoring_results.get("status") == "completed":
            monitoring_data = monitoring_results.get("monitoring_result", {})
            next_assessment = monitoring_data.get("next_assessment", {})

            if next_assessment.get("priority") == "urgent":
                next_steps.append("Schedule urgent follow-up assessment")
            else:
                next_steps.append("Continue routine monitoring")

        # Default next steps if none specific
        if not next_steps:
            next_steps = [
                "Continue current care plan",
                "Monitor symptoms",
                "Follow up as scheduled"
            ]

        return next_steps

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_status": self.system_status,
            "active_sessions": len(self.active_sessions),
            "available_agents": list(self.agents.keys()),
            "agent_status": {name: "ready" for name in self.agents.keys()},
            "last_updated": datetime.utcnow().isoformat()
        }

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a specific session."""
        return self.active_sessions.get(session_id)


# Create global orchestrator instance
healthcare_orchestrator = HealthcareOrchestrator()
