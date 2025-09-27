"""
Base agent class for all healthcare system agents.
Provides common functionality and interfaces with multi-model support for maximum accuracy.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4
from config.settings import config
from utils.logging import get_logger
from utils.error_handling import handle_exception, HealthcareSystemError
from utils.multi_model_client import multi_client

logger = get_logger("base_agent")


class BaseAgent(ABC):
    """Abstract base class for all healthcare agents with multi-model support."""

    def __init__(self, agent_id: Optional[str] = None, specialized_task: str = "general"):
        self.agent_id = agent_id or str(uuid4())
        self.agent_type = self.__class__.__name__
        self.specialized_task = specialized_task
        self.created_at = datetime.utcnow()
        self.multi_client = multi_client
        self.status = "initialized"
        self.task_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized {self.agent_type} with ID: {self.agent_id} for {specialized_task} tasks")

    @abstractmethod
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input data and return results."""
        pass
    
    @handle_exception
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with logging and error handling."""
        task_id = str(uuid4())
        start_time = datetime.utcnow()
        
        logger.info(f"Agent {self.agent_id} starting task {task_id} of type {self.specialized_task}")

        try:
            self.status = "processing"
            result = await self.process(task_data.get("input"), task_data.get("context"))
            
            # Log successful completion
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            task_record = {
                "task_id": task_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "status": "completed",
                "input_summary": self._summarize_input(task_data.get("input")),
                "result_summary": self._summarize_result(result),
                "model_used": result.get("model_info", {}).get("provider", "unknown")
            }
            
            self.task_history.append(task_record)
            self.status = "ready"
            
            logger.info(f"Agent {self.agent_id} completed task {task_id} in {duration:.2f}s using {task_record['model_used']}")

            return {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "success",
                "result": result,
                "metadata": {
                    "duration": duration,
                    "timestamp": end_time.isoformat(),
                    "model_used": task_record["model_used"]
                }
            }
            
        except Exception as e:
            self.status = "error"
            logger.error(f"Agent {self.agent_id} failed task {task_id}: {str(e)}")
            raise
    
    def _summarize_input(self, input_data: Any) -> str:
        """Create a summary of input data for logging."""
        if isinstance(input_data, str):
            return f"Text input ({len(input_data)} chars)"
        elif isinstance(input_data, dict):
            return f"Dict input with keys: {list(input_data.keys())}"
        elif hasattr(input_data, '__len__'):
            return f"List/Array input ({len(input_data)} items)"
        else:
            return f"Input type: {type(input_data).__name__}"
    
    def _summarize_result(self, result: Any) -> str:
        """Create a summary of result data for logging."""
        if isinstance(result, dict):
            return f"Dict result with keys: {list(result.keys())}"
        elif isinstance(result, str):
            return f"Text result ({len(result)} chars)"
        else:
            return f"Result type: {type(result).__name__}"
    
    async def get_ai_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        provider_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get AI response using the best model for this agent's specialized task."""
        try:
            response_text = await self.multi_client.get_completion(
                messages=messages,
                task_type=self.specialized_task,
                max_tokens=max_tokens,
                temperature=temperature,
                provider_override=provider_override
            )

            provider_used = provider_override or self.multi_client.select_best_model(self.specialized_task)

            return {
                "response": response_text,
                "model_info": {
                    "provider": provider_used,
                    "task_type": self.specialized_task,
                    "temperature": temperature or config.models.temperature,
                    "max_tokens": max_tokens or config.models.max_tokens
                }
            }
        except Exception as e:
            raise HealthcareSystemError(f"AI response error: {str(e)}", "AI_RESPONSE_ERROR")

    async def get_medical_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get specialized medical analysis for healthcare tasks."""
        return await self.multi_client.get_medical_analysis(
            patient_data=patient_data,
            analysis_type=self.specialized_task
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and statistics."""
        completed_tasks = len([t for t in self.task_history if t["status"] == "completed"])
        avg_duration = sum(t["duration"] for t in self.task_history) / len(self.task_history) if self.task_history else 0
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "tasks_completed": completed_tasks,
            "average_task_duration": avg_duration,
            "total_tasks": len(self.task_history)
        }
