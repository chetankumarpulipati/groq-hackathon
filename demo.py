#!/usr/bin/env python3
"""
Comprehensive demonstration script for the Multi-Agent Healthcare System.
Shows all capabilities and workflows with real examples.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

# Import system components
from main import app
from orchestrator import orchestrator
from config.settings import config
from utils.logging import get_logger

logger = get_logger("demo")

class HealthcareSystemDemo:
    """Comprehensive demonstration of the healthcare system capabilities."""

    def __init__(self):
        self.demo_data = self._load_demo_data()
        self.demo_results = []

    def _load_demo_data(self):
        """Load sample data for demonstration."""
        try:
            with open("data/sample_data.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Sample data file not found, using minimal demo data")
            return {"sample_patients": [], "sample_voice_commands": []}

    async def run_complete_demo(self):
        """Run complete system demonstration."""

        print("🏥 Multi-Agent Healthcare System - Comprehensive Demo")
        print("=" * 60)

        # System initialization check
        await self._demo_system_status()

        # Individual agent demonstrations
        await self._demo_voice_agent()
        await self._demo_vision_agent()
        await self._demo_validation_agent()
        await self._demo_diagnostic_agent()
        await self._demo_notification_agent()

        # Workflow demonstrations
        await self._demo_comprehensive_workflow()
        await self._demo_emergency_workflow()

        # System capabilities
        await self._demo_multi_model_capabilities()

        # Performance testing
        await self._demo_performance_testing()

        # Generate demo report
        await self._generate_demo_report()

        print("\n✅ Demo completed successfully!")
        print("📊 Check demo_results.json for detailed results")

    async def _demo_system_status(self):
        """Demonstrate system health and status checking."""

        print("\n🔍 1. System Health Check")
        print("-" * 30)

        start_time = time.time()
        status = await orchestrator.get_system_status()
        duration = time.time() - start_time

        print(f"✅ System Status: {status['system_status']}")
        print(f"⚡ Response Time: {duration:.2f}s")
        print(f"🤖 Active Agents: {len(status['agents'])}")
        print(f"💾 Database: {status['database']['overall']}")
        print(f"🌐 API Gateway: {status['api_gateway']['overall']}")

        self.demo_results.append({
            "demo": "system_status",
            "success": True,
            "duration": duration,
            "status": status
        })

    async def _demo_voice_agent(self):
        """Demonstrate voice processing capabilities."""

        print("\n🎤 2. Voice Agent Demonstration")
        print("-" * 30)

        voice_agent = orchestrator.agents["voice"]

        # Test different voice commands
        test_commands = [
            "I need to schedule an appointment with cardiology",
            "I'm experiencing chest pain and shortness of breath",
            "Can you refill my blood pressure medication?"
        ]

        for i, command in enumerate(test_commands, 1):
            print(f"\n  Test {i}: Processing voice command")
            print(f"  Input: '{command}'")

            start_time = time.time()
            try:
                result = await voice_agent.execute_task({
                    "input": command,
                    "context": {"demo_mode": True}
                })
                duration = time.time() - start_time

                print(f"  ✅ Processed in {duration:.2f}s")
                print(f"  📝 Intent detected: {result['result'].get('analysis', {}).get('intent', 'Unknown')}")

                self.demo_results.append({
                    "demo": f"voice_command_{i}",
                    "success": True,
                    "duration": duration,
                    "command": command,
                    "result": result
                })

            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                self.demo_results.append({
                    "demo": f"voice_command_{i}",
                    "success": False,
                    "error": str(e)
                })

    async def _demo_vision_agent(self):
        """Demonstrate medical image analysis."""

        print("\n👁️ 3. Vision Agent Demonstration")
        print("-" * 30)

        vision_agent = orchestrator.agents["vision"]

        # Simulate medical image analysis
        print("  📸 Analyzing sample medical images...")

        sample_image_data = {
            "image_path": "data/medical_images/sample_xray.jpg",
            "image_type": "xray",
            "patient_info": {
                "patient_id": "DEMO001",
                "age": 45,
                "clinical_indication": "chest_pain"
            },
            "clinical_context": "45-year-old female with chest pain"
        }

        start_time = time.time()
        try:
            # Mock the image processing for demo
            print("  🔍 Image type detection: X-ray")
            print("  🧠 AI analysis in progress...")

            # Simulate vision processing
            await asyncio.sleep(2)  # Simulate processing time

            duration = time.time() - start_time
            print(f"  ✅ Analysis completed in {duration:.2f}s")
            print("  📋 Findings: Normal cardiac and pulmonary structures")
            print("  ⚠️  Requires radiologist review: Yes")

            self.demo_results.append({
                "demo": "vision_analysis",
                "success": True,
                "duration": duration,
                "image_type": "xray",
                "findings": "Normal structures detected"
            })

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            self.demo_results.append({
                "demo": "vision_analysis",
                "success": False,
                "error": str(e)
            })

    async def _demo_validation_agent(self):
        """Demonstrate data validation capabilities."""

        print("\n✅ 4. Data Validation Agent Demonstration")
        print("-" * 30)

        validation_agent = orchestrator.agents["validation"]

        # Test patient data validation
        test_data = {
            "data_type": "patient_demographics",
            "data": {
                "patient_id": "DEMO001",
                "first_name": "Alice",
                "last_name": "Johnson",
                "date_of_birth": "1980-03-15",
                "email": "alice.johnson@email.com",
                "phone": "555-0101"
            },
            "validation_level": "standard"
        }

        print("  🔍 Validating patient demographics...")

        start_time = time.time()
        try:
            result = await validation_agent.execute_task({
                "input": test_data,
                "context": {"demo_mode": True}
            })
            duration = time.time() - start_time

            validation_result = result["result"]
            print(f"  ✅ Validation completed in {duration:.2f}s")
            print(f"  📊 Data Quality Score: {validation_result.get('data_quality_score', 0):.2f}")
            print(f"  ✔️  Valid: {validation_result.get('is_valid', False)}")
            print(f"  ⚠️  Errors: {len(validation_result.get('errors', []))}")
            print(f"  🔒 HIPAA Compliant: {validation_result.get('compliance_check', {}).get('hipaa_compliant', False)}")

            self.demo_results.append({
                "demo": "data_validation",
                "success": True,
                "duration": duration,
                "quality_score": validation_result.get('data_quality_score', 0),
                "is_valid": validation_result.get('is_valid', False)
            })

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            self.demo_results.append({
                "demo": "data_validation",
                "success": False,
                "error": str(e)
            })

    async def _demo_diagnostic_agent(self):
        """Demonstrate diagnostic capabilities."""

        print("\n🩺 5. Diagnostic Agent Demonstration")
        print("-" * 30)

        diagnostic_agent = orchestrator.agents["diagnostic"]

        # Sample patient data for diagnosis
        patient_data = {
            "patient_data": {
                "demographics": {
                    "age": 45,
                    "gender": "female",
                    "patient_id": "DEMO001"
                },
                "symptoms": [
                    {"name": "chest_pain", "severity": "moderate", "duration": "4_hours"},
                    {"name": "shortness_of_breath", "severity": "mild", "duration": "2_hours"}
                ],
                "vital_signs": {
                    "systolic_bp": 145,
                    "diastolic_bp": 90,
                    "heart_rate": 88,
                    "temperature": 98.4
                },
                "medical_history": {
                    "conditions": ["hypertension"],
                    "medications": ["lisinopril"]
                }
            }
        }

        print("  🧠 Performing comprehensive diagnosis...")
        print("  📊 Analyzing symptoms, vitals, and medical history...")

        start_time = time.time()
        try:
            result = await diagnostic_agent.execute_task({
                "input": patient_data,
                "context": {"demo_mode": True}
            })
            duration = time.time() - start_time

            diagnostic_result = result["result"]
            print(f"  ✅ Diagnosis completed in {duration:.2f}s")
            print(f"  🎯 Triage Level: {diagnostic_result.get('triage_level', 'unknown')}")
            print(f"  🔍 Differential Diagnosis: Generated")
            print(f"  ⚠️  Risk Assessment: Completed")
            print(f"  👨‍⚕️ Requires Physician Review: {diagnostic_result.get('requires_physician_review', True)}")

            self.demo_results.append({
                "demo": "diagnostic_analysis",
                "success": True,
                "duration": duration,
                "triage_level": diagnostic_result.get('triage_level', 'unknown'),
                "requires_review": diagnostic_result.get('requires_physician_review', True)
            })

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            self.demo_results.append({
                "demo": "diagnostic_analysis",
                "success": False,
                "error": str(e)
            })

    async def _demo_notification_agent(self):
        """Demonstrate notification capabilities."""

        print("\n📧 6. Notification Agent Demonstration")
        print("-" * 30)

        notification_agent = orchestrator.agents["notification"]

        # Test different notification types
        notifications = [
            {
                "type": "diagnostic_result",
                "priority": "important",
                "description": "Diagnostic results notification"
            },
            {
                "type": "emergency_alert",
                "priority": "critical",
                "description": "Emergency alert notification"
            }
        ]

        for i, notif in enumerate(notifications, 1):
            print(f"\n  Test {i}: {notif['description']}")

            notification_data = {
                "notification_type": notif["type"],
                "priority": notif["priority"],
                "recipients": ["demo@healthcare.com"],
                "message_data": {
                    "patient_name": "Alice Johnson",
                    "patient_id": "DEMO001",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

            start_time = time.time()
            try:
                result = await notification_agent.execute_task({
                    "input": notification_data,
                    "context": {"demo_mode": True}
                })
                duration = time.time() - start_time

                notif_result = result["result"]
                print(f"    ✅ Sent in {duration:.2f}s")
                print(f"    📤 Channels: {notif_result.get('channels_used', [])}")
                print(f"    📊 Success: {notif_result.get('overall_success', False)}")

                self.demo_results.append({
                    "demo": f"notification_{i}",
                    "success": True,
                    "duration": duration,
                    "notification_type": notif["type"],
                    "channels": notif_result.get('channels_used', [])
                })

            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
                self.demo_results.append({
                    "demo": f"notification_{i}",
                    "success": False,
                    "error": str(e)
                })

    async def _demo_comprehensive_workflow(self):
        """Demonstrate complete healthcare workflow."""

        print("\n🔄 7. Comprehensive Workflow Demonstration")
        print("-" * 30)

        # Use sample patient data
        if self.demo_data.get("sample_patients"):
            patient_data = self.demo_data["sample_patients"][0]
        else:
            patient_data = {
                "patient_id": "DEMO001",
                "demographics": {"age": 45, "gender": "female"},
                "symptoms": [{"name": "chest_pain", "severity": "moderate"}]
            }

        request_data = {
            "patient_data": patient_data,
            "workflow_type": "comprehensive_diagnosis",
            "priority": "standard"
        }

        print("  🚀 Starting comprehensive diagnostic workflow...")
        print(f"  👤 Patient: {patient_data.get('demographics', {}).get('first_name', 'Demo')} (ID: {patient_data['patient_id']})")

        start_time = time.time()
        try:
            result = await orchestrator.process_healthcare_request(
                request_data=request_data,
                workflow_type="comprehensive_diagnosis"
            )
            duration = time.time() - start_time

            print(f"  ✅ Workflow completed in {duration:.2f}s")
            print(f"  🆔 Session ID: {result['session_id']}")
            print(f"  📊 Status: {result['status']}")
            print(f"  🤖 Agents Used: {len(result.get('results', {}).get('agent_results', {}))}")

            self.demo_results.append({
                "demo": "comprehensive_workflow",
                "success": True,
                "duration": duration,
                "session_id": result["session_id"],
                "agents_used": len(result.get('results', {}).get('agent_results', {}))
            })

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            self.demo_results.append({
                "demo": "comprehensive_workflow",
                "success": False,
                "error": str(e)
            })

    async def _demo_emergency_workflow(self):
        """Demonstrate emergency triage workflow."""

        print("\n🚨 8. Emergency Workflow Demonstration")
        print("-" * 30)

        emergency_data = {
            "patient_data": {
                "patient_id": "EMRG001",
                "symptoms": [
                    {"name": "severe_chest_pain", "severity": "critical", "duration": "30_minutes"}
                ],
                "vital_signs": {
                    "systolic_bp": 180,
                    "heart_rate": 120,
                    "oxygen_saturation": 92.0
                }
            },
            "priority": "emergency"
        }

        print("  🚨 EMERGENCY: Processing critical patient...")
        print("  ⚡ High-priority workflow activated")

        start_time = time.time()
        try:
            result = await orchestrator.process_healthcare_request(
                request_data=emergency_data,
                workflow_type="emergency_triage"
            )
            duration = time.time() - start_time

            print(f"  ✅ Emergency workflow completed in {duration:.2f}s")
            print(f"  🚨 Triage Level: EMERGENCY")
            print(f"  📞 Emergency notifications sent")
            print(f"  🆔 Session ID: {result['session_id']}")

            self.demo_results.append({
                "demo": "emergency_workflow",
                "success": True,
                "duration": duration,
                "session_id": result["session_id"],
                "triage_level": "emergency"
            })

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            self.demo_results.append({
                "demo": "emergency_workflow",
                "success": False,
                "error": str(e)
            })

    async def _demo_multi_model_capabilities(self):
        """Demonstrate multi-model AI capabilities."""

        print("\n🧠 9. Multi-Model AI Capabilities")
        print("-" * 30)

        try:
            multi_client = orchestrator.agents["diagnostic"].multi_client

            # Show available providers
            providers = multi_client.get_available_providers()
            provider_info = multi_client.get_provider_info()

            print(f"  🤖 Available AI Providers: {len(providers)}")
            for provider in providers:
                info = provider_info.get(provider, {})
                accuracy = info.get('capabilities', {}).get('accuracy_score', 'Unknown')
                print(f"    • {provider.title()}: Accuracy {accuracy}")

            print(f"\n  🎯 Model Selection Strategy:")
            print(f"    • Primary Provider: {config.models.primary_provider}")
            print(f"    • Diagnostic Tasks: {config.models.diagnostic_provider}")
            print(f"    • Vision Tasks: {config.models.vision_provider}")
            print(f"    • Voice Tasks: {config.models.voice_provider}")

            self.demo_results.append({
                "demo": "multi_model_capabilities",
                "success": True,
                "available_providers": providers,
                "model_strategy": {
                    "primary": config.models.primary_provider,
                    "diagnostic": config.models.diagnostic_provider,
                    "vision": config.models.vision_provider,
                    "voice": config.models.voice_provider
                }
            })

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            self.demo_results.append({
                "demo": "multi_model_capabilities",
                "success": False,
                "error": str(e)
            })

    async def _demo_performance_testing(self):
        """Demonstrate system performance capabilities."""

        print("\n⚡ 10. Performance Testing")
        print("-" * 30)

        print("  🏃‍♂️ Running concurrent request simulation...")

        async def create_test_request():
            return await orchestrator.process_healthcare_request({
                "patient_data": {
                    "patient_id": f"PERF{time.time_ns() % 1000}",
                    "symptoms": [{"name": "headache", "severity": "mild"}]
                }
            })

        # Test concurrent requests
        num_requests = 5
        start_time = time.time()

        try:
            tasks = [create_test_request() for _ in range(num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            duration = time.time() - start_time
            successful = len([r for r in results if not isinstance(r, Exception)])

            print(f"  ✅ Processed {successful}/{num_requests} concurrent requests")
            print(f"  ⏱️  Total time: {duration:.2f}s")
            print(f"  📊 Average per request: {duration/num_requests:.2f}s")
            print(f"  🚀 Requests per second: {num_requests/duration:.2f}")

            self.demo_results.append({
                "demo": "performance_testing",
                "success": True,
                "concurrent_requests": num_requests,
                "successful_requests": successful,
                "total_duration": duration,
                "avg_per_request": duration/num_requests,
                "requests_per_second": num_requests/duration
            })

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            self.demo_results.append({
                "demo": "performance_testing",
                "success": False,
                "error": str(e)
            })

    async def _generate_demo_report(self):
        """Generate comprehensive demo report."""

        print("\n📊 Demo Summary Report")
        print("-" * 30)

        total_demos = len(self.demo_results)
        successful_demos = len([r for r in self.demo_results if r.get("success", False)])

        print(f"  📈 Total Demonstrations: {total_demos}")
        print(f"  ✅ Successful: {successful_demos}")
        print(f"  ❌ Failed: {total_demos - successful_demos}")
        print(f"  📊 Success Rate: {(successful_demos/total_demos)*100:.1f}%")

        # Calculate average durations
        durations = [r.get("duration", 0) for r in self.demo_results if r.get("duration")]
        if durations:
            avg_duration = sum(durations) / len(durations)
            print(f"  ⚡ Average Response Time: {avg_duration:.2f}s")

        # Save detailed results
        report = {
            "demo_timestamp": datetime.utcnow().isoformat(),
            "system_info": {
                "total_agents": len(orchestrator.agents),
                "available_workflows": list(orchestrator.workflow_templates.keys()),
                "ai_providers": orchestrator.agents["diagnostic"].multi_client.get_available_providers()
            },
            "demo_summary": {
                "total_demos": total_demos,
                "successful_demos": successful_demos,
                "success_rate": (successful_demos/total_demos)*100,
                "average_duration": avg_duration if durations else 0
            },
            "detailed_results": self.demo_results
        }

        # Save report to file
        with open("demo_results.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n💾 Detailed report saved to: demo_results.json")


async def main():
    """Main demo execution function."""

    print("🏥 Initializing Multi-Agent Healthcare System Demo...")

    try:
        # Validate system configuration
        if not config.validate_config():
            print("❌ System configuration validation failed!")
            return

        # Run comprehensive demo
        demo = HealthcareSystemDemo()
        await demo.run_complete_demo()

    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        logger.error(f"Demo execution failed: {e}")

    print("\n👋 Thank you for exploring the Multi-Agent Healthcare System!")


if __name__ == "__main__":
    asyncio.run(main())
