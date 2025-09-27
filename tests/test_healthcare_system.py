"""
Comprehensive test suite for the Multi-Agent Healthcare System.
Tests all agents, workflows, and system integration.
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from agents.voice_agent import VoiceAgent
from agents.vision_agent import VisionAgent
from agents.validation_agent import DataValidationAgent
from agents.diagnostic_agent import DiagnosticAgent
from agents.notification_agent import NotificationAgent
from orchestrator import HealthcareOrchestrator
from utils.multi_model_client import MultiModelClient


class TestVoiceAgent:
    """Test cases for Voice Agent."""

    @pytest.fixture
    def voice_agent(self):
        return VoiceAgent()

    @pytest.mark.asyncio
    async def test_text_command_processing(self, voice_agent):
        """Test processing of text-based voice commands."""

        with patch.object(voice_agent, 'get_ai_response', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = {
                "response": '{"intent": "appointment", "priority": "medium", "confidence": 0.9}',
                "model_info": {"provider": "openai"}
            }

            result = await voice_agent.process("I need to schedule an appointment with Dr. Smith")

            assert result["processing_type"] == "text_command"
            assert "analysis" in result
            mock_ai.assert_called_once()

    @pytest.mark.asyncio
    async def test_emergency_command_handling(self, voice_agent):
        """Test emergency voice command handling."""

        with patch.object(voice_agent, 'get_ai_response', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = {
                "response": "Call 911 immediately. This appears to be a medical emergency.",
                "model_info": {"provider": "openai"}
            }

            emergency_data = {
                "type": "emergency",
                "text": "I'm having severe chest pain and can't breathe",
                "urgency": "critical"
            }

            result = await voice_agent.process({"command": emergency_data})

            assert result["priority"] == "EMERGENCY"
            assert result["requires_immediate_action"] is True
            assert result["emergency_services_recommended"] is True


class TestVisionAgent:
    """Test cases for Vision Agent."""

    @pytest.fixture
    def vision_agent(self):
        return VisionAgent()

    @pytest.mark.asyncio
    async def test_image_type_detection(self, vision_agent):
        """Test medical image type detection."""

        # Mock image processing
        with patch('PIL.Image.open') as mock_image:
            mock_image.return_value.size = (512, 512)
            mock_image.return_value.mode = 'RGB'
            mock_image.return_value.convert.return_value = mock_image.return_value

            with patch.object(vision_agent, 'get_ai_response', new_callable=AsyncMock) as mock_ai:
                mock_ai.return_value = {
                    "response": "xray",
                    "model_info": {"provider": "google"}
                }

                # Mock file existence
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.stat') as mock_stat:
                        mock_stat.return_value.st_size = 1024000

                        result = await vision_agent._detect_medical_image_type(
                            mock_image.return_value,
                            "chest_xray.jpg"
                        )

                        assert result == "xray"

    @pytest.mark.asyncio
    async def test_medical_image_analysis(self, vision_agent):
        """Test comprehensive medical image analysis."""

        with patch.object(vision_agent, '_validate_and_load_image', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = {
                "image": Mock(),
                "detected_type": "xray",
                "format": "jpg",
                "size": (512, 512)
            }

            with patch.object(vision_agent, '_perform_medical_image_analysis', new_callable=AsyncMock) as mock_analysis:
                mock_analysis.return_value = {
                    "analysis_text": "Normal chest X-ray with no acute findings",
                    "image_type": "xray",
                    "model_info": {"provider": "google"}
                }

                with patch.object(vision_agent, '_generate_clinical_report', new_callable=AsyncMock) as mock_report:
                    mock_report.return_value = {
                        "clinical_report": "FINDINGS: Normal cardiac and pulmonary structures",
                        "requires_physician_review": True
                    }

                    result = await vision_agent.process("test_xray.jpg")

                    assert result["processing_type"] == "medical_image_file"
                    assert result["requires_radiologist_review"] is True
                    assert "medical_analysis" in result


class TestDataValidationAgent:
    """Test cases for Data Validation Agent."""

    @pytest.fixture
    def validation_agent(self):
        return DataValidationAgent()

    @pytest.mark.asyncio
    async def test_patient_data_validation(self, validation_agent):
        """Test patient demographics validation."""

        patient_data = {
            "patient_id": "P123456",
            "first_name": "John",
            "last_name": "Doe",
            "date_of_birth": "1980-01-01",
            "email": "john.doe@email.com",
            "phone": "555-123-4567"
        }

        input_data = {
            "data_type": "patient_demographics",
            "data": patient_data,
            "validation_level": "standard"
        }

        result = await validation_agent.process(input_data)

        assert result["is_valid"] is True
        assert result["data_quality_score"] > 0.8
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_vital_signs_validation(self, validation_agent):
        """Test vital signs validation with critical values."""

        vital_signs_data = {
            "patient_id": "P123456",
            "timestamp": "2025-01-01T10:00:00Z",
            "systolic_bp": 200,  # Critical high
            "diastolic_bp": 120,  # Critical high
            "heart_rate": 45,     # Critical low
            "temperature": 104.5,  # Critical high
            "oxygen_saturation": 88.0  # Critical low
        }

        input_data = {
            "data_type": "vital_signs",
            "data": vital_signs_data,
            "validation_level": "standard"
        }

        result = await validation_agent.process(input_data)

        assert len(result["critical_issues"]) > 0
        assert result["is_valid"] is False  # Due to critical values


class TestDiagnosticAgent:
    """Test cases for Diagnostic Agent."""

    @pytest.fixture
    def diagnostic_agent(self):
        return DiagnosticAgent()

    @pytest.mark.asyncio
    async def test_comprehensive_diagnosis(self, diagnostic_agent):
        """Test comprehensive diagnostic workflow."""

        patient_data = {
            "demographics": {
                "age": 45,
                "gender": "male",
                "patient_id": "P123456"
            },
            "symptoms": [
                {"name": "chest_pain", "severity": "severe", "duration": "2_hours"},
                {"name": "shortness_of_breath", "severity": "moderate", "duration": "1_hour"}
            ],
            "vital_signs": {
                "systolic_bp": 150,
                "diastolic_bp": 95,
                "heart_rate": 110,
                "temperature": 98.6
            },
            "medical_history": {
                "conditions": ["hypertension"],
                "medications": ["lisinopril"]
            }
        }

        input_data = {
            "patient_data": patient_data,
            "diagnostic_mode": "comprehensive"
        }

        with patch.object(diagnostic_agent, 'get_ai_response', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = {
                "response": "Differential diagnosis includes: 1. Acute coronary syndrome (high probability), 2. Anxiety disorder (moderate probability)",
                "model_info": {"provider": "openai"}
            }

            result = await diagnostic_agent.process(input_data)

            assert "differential_diagnosis" in result
            assert "risk_assessment" in result
            assert result["requires_physician_review"] is True


class TestNotificationAgent:
    """Test cases for Notification Agent."""

    @pytest.fixture
    def notification_agent(self):
        return NotificationAgent()

    @pytest.mark.asyncio
    async def test_emergency_alert_sending(self, notification_agent):
        """Test emergency alert notification."""

        alert_data = {
            "patient_name": "John Doe",
            "patient_id": "P123456",
            "alert_type": "Cardiac Emergency",
            "severity": "Critical",
            "clinical_summary": "Patient experiencing severe chest pain with ECG changes"
        }

        recipients = ["emergency@hospital.com", "cardiology@hospital.com"]

        with patch.object(notification_agent, '_send_email_notification', new_callable=AsyncMock) as mock_email:
            mock_email.return_value = {
                "success": True,
                "channel": "email",
                "timestamp": datetime.utcnow().isoformat()
            }

            result = await notification_agent.send_emergency_alert(alert_data, recipients)

            assert result["overall_success"] is True
            assert result["priority"] == "critical"
            mock_email.assert_called()


class TestMultiModelClient:
    """Test cases for Multi Model Client."""

    @pytest.fixture
    def multi_client(self):
        return MultiModelClient()

    def test_model_selection(self, multi_client):
        """Test intelligent model selection for different tasks."""

        # Test diagnostic task selection
        diagnostic_model = multi_client.select_best_model("medical_diagnosis")
        assert diagnostic_model in multi_client.clients

        # Test vision task selection
        vision_model = multi_client.select_best_model("vision")
        assert vision_model in multi_client.clients

        # Test voice task selection
        voice_model = multi_client.select_best_model("voice_processing")
        assert voice_model in multi_client.clients

    @pytest.mark.asyncio
    async def test_medical_analysis(self, multi_client):
        """Test specialized medical analysis."""

        patient_data = {
            "symptoms": ["chest pain", "shortness of breath"],
            "vital_signs": {"bp": "150/95", "hr": 110},
            "age": 55,
            "gender": "male"
        }

        with patch.object(multi_client, 'get_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = "Based on the symptoms and vital signs, differential diagnosis includes acute coronary syndrome."

            result = await multi_client.get_medical_analysis(patient_data, "cardiology")

            assert result["requires_physician_review"] is True
            assert "analysis" in result
            mock_completion.assert_called_once()


class TestHealthcareOrchestrator:
    """Test cases for Healthcare Orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        # Create a test orchestrator instance
        orchestrator = HealthcareOrchestrator()
        return orchestrator

    @pytest.mark.asyncio
    async def test_comprehensive_workflow(self, orchestrator):
        """Test complete healthcare workflow execution."""

        request_data = {
            "patient_data": {
                "patient_id": "P123456",
                "demographics": {"age": 45, "gender": "male"},
                "symptoms": [{"name": "chest_pain", "severity": "severe"}]
            },
            "workflow_type": "comprehensive_diagnosis",
            "priority": "urgent"
        }

        # Mock agent execution
        for agent_name, agent in orchestrator.agents.items():
            with patch.object(agent, 'execute_task', new_callable=AsyncMock) as mock_task:
                mock_task.return_value = {
                    "status": "success",
                    "result": {"processing_type": f"{agent_name}_result"},
                    "agent_id": agent.agent_id
                }

        # Mock database storage
        with patch('mcp_connectors.database_connector.database_connector.store_diagnostic_session', new_callable=AsyncMock):
            result = await orchestrator.process_healthcare_request(request_data, "comprehensive_diagnosis")

            assert result["status"] == "success"
            assert "session_id" in result
            assert result["workflow_type"] == "comprehensive_diagnosis"

    @pytest.mark.asyncio
    async def test_emergency_workflow(self, orchestrator):
        """Test emergency triage workflow."""

        request_data = {
            "patient_data": {
                "patient_id": "P123456",
                "symptoms": [{"name": "chest_pain", "severity": "critical"}]
            },
            "priority": "emergency"
        }

        # Mock agent execution for emergency workflow
        for agent_name, agent in orchestrator.agents.items():
            with patch.object(agent, 'execute_task', new_callable=AsyncMock) as mock_task:
                mock_task.return_value = {
                    "status": "success",
                    "result": {"triage_level": "emergency"},
                    "agent_id": agent.agent_id
                }

        with patch('mcp_connectors.database_connector.database_connector.store_diagnostic_session', new_callable=AsyncMock):
            result = await orchestrator.process_healthcare_request(request_data, "emergency_triage")

            assert result["status"] == "success"
            assert result["workflow_type"] == "emergency_triage"


class TestSystemIntegration:
    """Integration tests for the complete system."""

    @pytest.mark.asyncio
    async def test_end_to_end_diagnosis(self):
        """Test complete end-to-end diagnostic process."""

        # This would test the complete flow from API request to response
        # Including all agents, database operations, and notifications

        orchestrator = HealthcareOrchestrator()

        request_data = {
            "patient_data": {
                "patient_id": "TEST001",
                "demographics": {
                    "first_name": "Jane",
                    "last_name": "Smith",
                    "age": 35,
                    "gender": "female"
                },
                "symptoms": [
                    {"name": "fever", "severity": "moderate", "duration": "3_days"},
                    {"name": "cough", "severity": "mild", "duration": "2_days"}
                ],
                "vital_signs": {
                    "temperature": 101.2,
                    "heart_rate": 95,
                    "blood_pressure": "120/80"
                }
            },
            "workflow_type": "comprehensive_diagnosis"
        }

        # Mock all external dependencies
        with patch('mcp_connectors.database_connector.database_connector.store_diagnostic_session', new_callable=AsyncMock):
            with patch('utils.multi_model_client.multi_client.get_completion', new_callable=AsyncMock) as mock_ai:
                mock_ai.return_value = "Patient presents with viral syndrome. Recommend supportive care and follow-up."

                result = await orchestrator.process_healthcare_request(request_data)

                assert result["status"] == "success"
                assert "session_id" in result
                assert "results" in result


# Test fixtures and configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Performance tests
class TestPerformance:
    """Performance and load testing."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test system performance under concurrent load."""

        orchestrator = HealthcareOrchestrator()

        async def create_test_request():
            request_data = {
                "patient_data": {
                    "patient_id": f"TEST{datetime.utcnow().microsecond}",
                    "symptoms": [{"name": "headache", "severity": "mild"}]
                }
            }

            with patch('mcp_connectors.database_connector.database_connector.store_diagnostic_session', new_callable=AsyncMock):
                with patch('utils.multi_model_client.multi_client.get_completion', new_callable=AsyncMock) as mock_ai:
                    mock_ai.return_value = "Mild headache, recommend rest and hydration."
                    return await orchestrator.process_healthcare_request(request_data)

        # Test 10 concurrent requests
        tasks = [create_test_request() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all requests completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
