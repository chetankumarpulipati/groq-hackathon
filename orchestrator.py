"""
Healthcare System Orchestrator - Main coordination system for multi-agent healthcare platform.
Manages agent workflows, task distribution, and system coordination.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from uuid import uuid4
from agents.voice_agent import VoiceAgent
from agents.vision_agent import VisionAgent
from agents.validation_agent import DataValidationAgent
from agents.diagnostic_agent import DiagnosticAgent
from agents.notification_agent import NotificationAgent
from mcp_connectors.database_connector import database_connector
from mcp_connectors.api_gateway_connector import api_gateway_connector
from config.settings import config
from utils.logging import get_logger
from utils.error_handling import HealthcareSystemError, handle_exception

logger = get_logger("orchestrator")


class HealthcareOrchestrator:
    """
    Main orchestration system that coordinates all healthcare agents and services.
    Manages workflows, task distribution, and inter-agent communication.
    """

    def __init__(self):
        self.session_id = str(uuid4())
        self.agents = {}
        self.active_sessions = {}
        self.workflow_templates = self._initialize_workflow_templates()
        self.system_status = "initializing"

        # Initialize agents
        self._initialize_agents()

        logger.info(f"HealthcareOrchestrator initialized with session ID: {self.session_id}")

    def _initialize_agents(self):
        """Initialize all healthcare agents."""
        try:
            self.agents = {
                "voice": VoiceAgent(),
                "vision": VisionAgent(),
                "validation": DataValidationAgent(),
                "diagnostic": DiagnosticAgent(),
                "notification": NotificationAgent()
            }

            self.system_status = "ready"
            logger.info(f"All agents initialized successfully: {list(self.agents.keys())}")

        except Exception as e:
            self.system_status = "error"
            logger.error(f"Agent initialization failed: {e}")
            raise HealthcareSystemError(f"System initialization failed: {str(e)}")

    def _initialize_workflow_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize predefined workflow templates for common healthcare scenarios."""
        return {
            "comprehensive_diagnosis": {
                "description": "Complete diagnostic workflow with multi-modal data analysis",
                "steps": [
                    {"agent": "validation", "task": "validate_input_data"},
                    {"agent": "voice", "task": "process_symptoms", "condition": "has_voice_input"},
                    {"agent": "vision", "task": "analyze_images", "condition": "has_medical_images"},
                    {"agent": "diagnostic", "task": "perform_diagnosis"},
                    {"agent": "notification", "task": "send_results"}
                ],
                "parallel_execution": ["voice", "vision"],
                "critical_path": ["validation", "diagnostic", "notification"]
            },
            "emergency_triage": {
                "description": "Emergency triage workflow for urgent cases",
                "steps": [
                    {"agent": "voice", "task": "assess_emergency_symptoms"},
                    {"agent": "diagnostic", "task": "emergency_assessment"},
                    {"agent": "notification", "task": "emergency_alert"}
                ],
                "parallel_execution": [],
                "critical_path": ["voice", "diagnostic", "notification"],
                "max_duration": 300  # 5 minutes
            },
            "routine_checkup": {
                "description": "Routine health checkup workflow",
                "steps": [
                    {"agent": "validation", "task": "validate_patient_data"},
                    {"agent": "diagnostic", "task": "routine_assessment"},
                    {"agent": "notification", "task": "send_summary"}
                ],
                "parallel_execution": [],
                "critical_path": ["validation", "diagnostic", "notification"]
            },
            "medication_review": {
                "description": "Medication review and interaction checking",
                "steps": [
                    {"agent": "validation", "task": "validate_medication_data"},
                    {"agent": "diagnostic", "task": "medication_analysis"},
                    {"agent": "notification", "task": "medication_alerts"}
                ],
                "parallel_execution": [],
                "critical_path": ["validation", "diagnostic", "notification"]
            }
        }

    @handle_exception
    async def process_healthcare_request(
        self,
        request_data: Dict[str, Any],
        workflow_type: str = "comprehensive_diagnosis"
    ) -> Dict[str, Any]:
        """Process a complete healthcare request using the specified workflow."""

        session_id = str(uuid4())
        start_time = datetime.utcnow()

        logger.info(f"Starting healthcare request processing: {session_id} with workflow: {workflow_type}")

        # Initialize session
        session_data = {
            "session_id": session_id,
            "workflow_type": workflow_type,
            "request_data": request_data,
            "start_time": start_time,
            "status": "processing",
            "agent_results": {},
            "workflow_steps": [],
            "errors": []
        }

        self.active_sessions[session_id] = session_data

        try:
            # Execute workflow
            workflow_result = await self._execute_workflow(session_id, workflow_type, request_data)

            # Store results in database
            await self._store_session_results(session_id, workflow_result)

            # Update session status
            session_data["status"] = "completed"
            session_data["end_time"] = datetime.utcnow()
            session_data["duration"] = (session_data["end_time"] - start_time).total_seconds()

            logger.info(f"Healthcare request completed: {session_id} in {session_data['duration']:.2f}s")

            return {
                "session_id": session_id,
                "status": "success",
                "workflow_type": workflow_type,
                "results": workflow_result,
                "duration": session_data["duration"],
                "timestamp": session_data["end_time"].isoformat()
            }

        except Exception as e:
            session_data["status"] = "error"
            session_data["error"] = str(e)
            logger.error(f"Healthcare request failed: {session_id} - {str(e)}")

            # Send error notification if critical
            if workflow_type == "emergency_triage":
                await self._send_error_notification(session_id, str(e))

            raise HealthcareSystemError(f"Healthcare request processing failed: {str(e)}")

    async def _execute_workflow(
        self,
        session_id: str,
        workflow_type: str,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific healthcare workflow."""

        if workflow_type not in self.workflow_templates:
            raise HealthcareSystemError(f"Unknown workflow type: {workflow_type}")

        workflow = self.workflow_templates[workflow_type]
        session_data = self.active_sessions[session_id]

        workflow_results = {
            "workflow_type": workflow_type,
            "agent_results": {},
            "workflow_metadata": {
                "steps_completed": 0,
                "steps_total": len(workflow["steps"]),
                "parallel_executed": False,
                "critical_path_completed": False
            }
        }

        # Check for parallel execution opportunities
        parallel_steps = workflow.get("parallel_execution", [])
        if parallel_steps:
            await self._execute_parallel_steps(session_id, parallel_steps, request_data, workflow_results)
            workflow_results["workflow_metadata"]["parallel_executed"] = True

        # Execute sequential steps
        for step in workflow["steps"]:
            agent_name = step["agent"]
            task_name = step["task"]
            condition = step.get("condition")

            # Check execution condition
            if condition and not self._check_condition(condition, request_data):
                logger.debug(f"Skipping step {agent_name}/{task_name} - condition not met: {condition}")
                continue

            # Skip if already executed in parallel
            if agent_name in parallel_steps and agent_name in workflow_results["agent_results"]:
                continue

            # Execute agent task
            try:
                agent_result = await self._execute_agent_task(
                    agent_name,
                    task_name,
                    request_data,
                    workflow_results
                )

                workflow_results["agent_results"][agent_name] = agent_result
                workflow_results["workflow_metadata"]["steps_completed"] += 1

                # Record workflow step
                session_data["workflow_steps"].append({
                    "agent": agent_name,
                    "task": task_name,
                    "status": "completed",
                    "timestamp": datetime.utcnow().isoformat(),
                    "result_summary": agent_result.get("result", {}).get("processing_type", "unknown")
                })

                logger.debug(f"Workflow step completed: {agent_name}/{task_name}")

            except Exception as e:
                logger.error(f"Workflow step failed: {agent_name}/{task_name} - {str(e)}")
                session_data["errors"].append({
                    "agent": agent_name,
                    "task": task_name,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Decide whether to continue or abort based on criticality
                if agent_name in workflow.get("critical_path", []):
                    raise HealthcareSystemError(f"Critical workflow step failed: {agent_name}/{task_name}")

        # Mark critical path as completed if no critical errors
        workflow_results["workflow_metadata"]["critical_path_completed"] = True

        return workflow_results

    async def _execute_parallel_steps(
        self,
        session_id: str,
        parallel_agents: List[str],
        request_data: Dict[str, Any],
        workflow_results: Dict[str, Any]
    ):
        """Execute multiple agent tasks in parallel for efficiency."""

        logger.info(f"Executing parallel steps: {parallel_agents}")

        # Create tasks for parallel execution
        parallel_tasks = []
        for agent_name in parallel_agents:
            if agent_name in self.agents:
                # Determine appropriate task based on request data
                task_name = self._determine_agent_task(agent_name, request_data)
                if task_name:
                    task = self._execute_agent_task(agent_name, task_name, request_data, workflow_results)
                    parallel_tasks.append((agent_name, task))

        # Execute tasks in parallel
        if parallel_tasks:
            results = await asyncio.gather(
                *[task for _, task in parallel_tasks],
                return_exceptions=True
            )

            # Process results
            for i, (agent_name, _) in enumerate(parallel_tasks):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"Parallel task failed: {agent_name} - {str(result)}")
                    workflow_results["agent_results"][agent_name] = {
                        "status": "error",
                        "error": str(result)
                    }
                else:
                    workflow_results["agent_results"][agent_name] = result

    async def _execute_agent_task(
        self,
        agent_name: str,
        task_name: str,
        request_data: Dict[str, Any],
        workflow_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific task on an agent."""

        if agent_name not in self.agents:
            raise HealthcareSystemError(f"Unknown agent: {agent_name}")

        agent = self.agents[agent_name]

        # Prepare task data based on agent type and task
        task_data = self._prepare_task_data(agent_name, task_name, request_data, workflow_context)

        # Execute agent task
        result = await agent.execute_task(task_data)

        return result

    def _prepare_task_data(
        self,
        agent_name: str,
        task_name: str,
        request_data: Dict[str, Any],
        workflow_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare task-specific data for agent execution."""

        base_task_data = {
            "input": request_data,
            "context": {
                "workflow_context": workflow_context,
                "session_id": workflow_context.get("session_id"),
                "task_name": task_name
            }
        }

        # Agent-specific task data preparation
        if agent_name == "voice" and task_name == "process_symptoms":
            base_task_data["input"] = request_data.get("voice_data", request_data.get("symptoms", ""))

        elif agent_name == "vision" and task_name == "analyze_images":
            base_task_data["input"] = request_data.get("medical_images", [])

        elif agent_name == "validation":
            base_task_data["input"] = {
                "data_type": request_data.get("data_type", "patient_data"),
                "data": request_data.get("patient_data", request_data)
            }

        elif agent_name == "diagnostic":
            # Combine all available data for diagnosis
            base_task_data["input"] = {
                "patient_data": self._compile_patient_data(request_data, workflow_context)
            }

        elif agent_name == "notification":
            base_task_data["input"] = {
                "notification_type": self._determine_notification_type(task_name, workflow_context),
                "recipients": request_data.get("notification_recipients", []),
                "message_data": self._compile_notification_data(request_data, workflow_context)
            }

        return base_task_data

    def _compile_patient_data(self, request_data: Dict[str, Any], workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive patient data from all sources."""

        patient_data = request_data.get("patient_data", {})

        # Add results from other agents
        agent_results = workflow_context.get("agent_results", {})

        if "voice" in agent_results:
            voice_result = agent_results["voice"].get("result", {})
            if "analysis" in voice_result:
                patient_data["symptoms"] = voice_result["analysis"]

        if "vision" in agent_results:
            vision_result = agent_results["vision"].get("result", {})
            if "medical_analysis" in vision_result:
                patient_data["imaging"] = [vision_result["medical_analysis"]]

        if "validation" in agent_results:
            validation_result = agent_results["validation"].get("result", {})
            patient_data["data_quality"] = validation_result.get("data_quality_score", 0)

        return patient_data

    def _check_condition(self, condition: str, request_data: Dict[str, Any]) -> bool:
        """Check if a workflow condition is met."""

        condition_checks = {
            "has_voice_input": "voice_data" in request_data or "symptoms" in request_data,
            "has_medical_images": "medical_images" in request_data and len(request_data["medical_images"]) > 0,
            "has_patient_data": "patient_data" in request_data,
            "is_emergency": request_data.get("priority") == "emergency"
        }

        return condition_checks.get(condition, False)

    def _determine_agent_task(self, agent_name: str, request_data: Dict[str, Any]) -> Optional[str]:
        """Determine appropriate task for an agent based on request data."""

        task_mappings = {
            "voice": "process_symptoms" if "voice_data" in request_data else None,
            "vision": "analyze_images" if "medical_images" in request_data else None,
            "validation": "validate_patient_data",
            "diagnostic": "perform_diagnosis",
            "notification": "send_results"
        }

        return task_mappings.get(agent_name)

    def _determine_notification_type(self, task_name: str, workflow_context: Dict[str, Any]) -> str:
        """Determine notification type based on task and context."""

        if task_name == "emergency_alert":
            return "emergency_alert"
        elif task_name == "medication_alerts":
            return "medication_alert"
        elif task_name == "send_results":
            return "diagnostic_result"
        else:
            return "general_notification"

    def _compile_notification_data(self, request_data: Dict[str, Any], workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Compile data for notifications."""

        notification_data = {
            "patient_name": request_data.get("patient_data", {}).get("first_name", "Unknown"),
            "patient_id": request_data.get("patient_data", {}).get("patient_id", "Unknown"),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }

        # Add diagnostic results if available
        agent_results = workflow_context.get("agent_results", {})
        if "diagnostic" in agent_results:
            diagnostic_result = agent_results["diagnostic"].get("result", {})
            notification_data.update({
                "diagnostic_summary": str(diagnostic_result.get("differential_diagnosis", {})),
                "recommendations": str(diagnostic_result.get("clinical_recommendations", {})),
                "priority": diagnostic_result.get("triage_level", "standard")
            })

        return notification_data

    async def _store_session_results(self, session_id: str, workflow_result: Dict[str, Any]):
        """Store session results in database."""

        session_data = self.active_sessions[session_id]

        diagnostic_session_data = {
            "session_id": session_id,
            "patient_id": session_data["request_data"].get("patient_data", {}).get("patient_id", "unknown"),
            "agent_id": self.session_id,
            "session_data": session_data,
            "diagnosis_result": workflow_result,
            "confidence_score": workflow_result.get("agent_results", {}).get("diagnostic", {}).get("result", {}).get("diagnostic_confidence", 0.0),
            "requires_review": True,
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat()
        }

        try:
            await database_connector.store_diagnostic_session(diagnostic_session_data)
            logger.info(f"Session results stored in database: {session_id}")
        except Exception as e:
            logger.error(f"Failed to store session results: {e}")

    async def _send_error_notification(self, session_id: str, error_message: str):
        """Send error notification for critical failures."""

        try:
            await self.agents["notification"].execute_task({
                "input": {
                    "notification_type": "system_error",
                    "priority": "urgent",
                    "recipients": ["system_admin@healthcare.com"],
                    "message_data": {
                        "session_id": session_id,
                        "error_message": error_message,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            })
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""

        # Check agent status
        agent_status = {}
        for name, agent in self.agents.items():
            agent_status[name] = agent.get_status()

        # Check database connectivity
        db_health = await database_connector.health_check()

        # Check API gateway connectivity
        api_health = await api_gateway_connector.health_check()

        return {
            "system_status": self.system_status,
            "orchestrator_session_id": self.session_id,
            "active_sessions": len(self.active_sessions),
            "agents": agent_status,
            "database": db_health,
            "api_gateway": api_health,
            "available_workflows": list(self.workflow_templates.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def shutdown(self):
        """Gracefully shutdown the orchestrator and all components."""

        logger.info("Initiating system shutdown...")

        # Complete active sessions
        for session_id in list(self.active_sessions.keys()):
            session_data = self.active_sessions[session_id]
            if session_data["status"] == "processing":
                session_data["status"] = "interrupted"
                session_data["end_time"] = datetime.utcnow()

        # Close API gateway connections
        await api_gateway_connector.close()

        self.system_status = "shutdown"
        logger.info("System shutdown completed")


# Global orchestrator instance
orchestrator = HealthcareOrchestrator()
