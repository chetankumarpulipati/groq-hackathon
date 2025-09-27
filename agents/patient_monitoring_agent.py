"""
Patient Monitoring Agent for continuous health tracking and real-time alerts.
Monitors patient vital signs, medication adherence, and health status changes.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from utils.logging import get_logger
from utils.error_handling import handle_exception

logger = get_logger("patient_monitoring_agent")


class PatientMonitoringAgent(BaseAgent):
    """
    Specialized agent for continuous patient monitoring and health status tracking.
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id, specialized_task="patient_monitoring")
        self.monitoring_protocols = self._initialize_monitoring_protocols()
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.active_monitors = {}
        logger.info(f"PatientMonitoringAgent {self.agent_id} initialized")

    def _initialize_monitoring_protocols(self) -> Dict[str, Any]:
        """Initialize patient monitoring protocols and schedules."""
        return {
            "vital_signs_monitoring": {
                "frequency": {
                    "high_risk": "every_15_minutes",
                    "moderate_risk": "hourly",
                    "low_risk": "every_4_hours",
                    "stable": "daily"
                },
                "parameters": [
                    "blood_pressure", "heart_rate", "temperature",
                    "respiratory_rate", "oxygen_saturation", "pain_level"
                ]
            },
            "medication_monitoring": {
                "adherence_tracking": True,
                "side_effect_monitoring": True,
                "therapeutic_levels": True,
                "interaction_alerts": True
            },
            "chronic_condition_monitoring": {
                "diabetes": {
                    "glucose_monitoring": "4_times_daily",
                    "hba1c_tracking": "quarterly",
                    "complications_screening": "annually"
                },
                "hypertension": {
                    "bp_monitoring": "twice_daily",
                    "medication_adherence": "daily",
                    "lifestyle_factors": "weekly"
                },
                "heart_disease": {
                    "cardiac_symptoms": "continuous",
                    "exercise_tolerance": "daily",
                    "medication_effects": "continuous"
                }
            },
            "post_operative_monitoring": {
                "wound_healing": "daily",
                "infection_signs": "continuous",
                "pain_management": "continuous",
                "mobility_assessment": "daily"
            }
        }

    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Initialize alert thresholds for various health parameters."""
        return {
            "vital_signs": {
                "blood_pressure": {
                    "critical_high": {"systolic": 180, "diastolic": 110},
                    "warning_high": {"systolic": 160, "diastolic": 100},
                    "critical_low": {"systolic": 90, "diastolic": 60}
                },
                "heart_rate": {
                    "critical_high": 120,
                    "warning_high": 100,
                    "critical_low": 50,
                    "warning_low": 60
                },
                "temperature": {
                    "critical_high": 104.0,
                    "warning_high": 101.5,
                    "critical_low": 95.0,
                    "warning_low": 96.5
                },
                "oxygen_saturation": {
                    "critical_low": 88,
                    "warning_low": 92
                },
                "respiratory_rate": {
                    "critical_high": 30,
                    "warning_high": 24,
                    "critical_low": 8,
                    "warning_low": 12
                }
            },
            "laboratory_values": {
                "glucose": {
                    "critical_high": 400,
                    "warning_high": 250,
                    "critical_low": 50,
                    "warning_low": 70
                },
                "potassium": {
                    "critical_high": 6.0,
                    "warning_high": 5.5,
                    "critical_low": 3.0,
                    "warning_low": 3.5
                }
            },
            "symptoms": {
                "pain_scale": {
                    "severe": 8,
                    "moderate": 5
                },
                "breathing_difficulty": {
                    "severe": "unable_to_speak",
                    "moderate": "short_sentences"
                }
            }
        }

    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process patient monitoring request."""
        try:
            monitoring_type = input_data.get("monitoring_type", "continuous") if isinstance(input_data, dict) else "continuous"
            patient_data = input_data.get("patient_data", {}) if isinstance(input_data, dict) else {}
            monitoring_duration = input_data.get("duration", "ongoing") if isinstance(input_data, dict) else "ongoing"

            # Process monitoring request
            monitoring_result = await self._process_patient_monitoring(patient_data, monitoring_type, monitoring_duration)

            return {
                "agent_id": self.agent_id,
                "monitoring_type": monitoring_type,
                "patient_id": patient_data.get("patient_id", "unknown"),
                "monitoring_result": monitoring_result,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }

        except Exception as e:
            logger.error(f"Patient monitoring failed: {e}")
            return {
                "agent_id": self.agent_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _process_patient_monitoring(self, patient_data: Dict, monitoring_type: str, duration: str) -> Dict:
        """Process patient monitoring based on type and duration."""

        patient_id = patient_data.get("patient_id", "unknown")

        # Initialize monitoring session
        monitoring_session = await self._initialize_monitoring_session(patient_id, patient_data, monitoring_type)

        # Analyze current patient status
        current_status = await self._assess_current_patient_status(patient_data)

        # Generate monitoring plan
        monitoring_plan = await self._create_monitoring_plan(patient_data, current_status)

        # Check for alerts
        alerts = await self._check_for_alerts(patient_data)

        # Generate recommendations
        recommendations = self._generate_monitoring_recommendations(patient_data, current_status)

        monitoring_result = {
            "monitoring_session": monitoring_session,
            "current_status": current_status,
            "monitoring_plan": monitoring_plan,
            "active_alerts": alerts,
            "recommendations": recommendations,
            "next_assessment": self._schedule_next_assessment(patient_data, current_status),
            "emergency_protocols": self._get_emergency_protocols(patient_data)
        }

        # Store monitoring session
        self.active_monitors[patient_id] = monitoring_result

        return monitoring_result

    async def _initialize_monitoring_session(self, patient_id: str, patient_data: Dict, monitoring_type: str) -> Dict:
        """Initialize a new monitoring session for the patient."""

        risk_level = self._assess_patient_risk_level(patient_data)
        monitoring_frequency = self._determine_monitoring_frequency(risk_level, monitoring_type)

        session = {
            "session_id": f"MONITOR_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "patient_id": patient_id,
            "monitoring_type": monitoring_type,
            "risk_level": risk_level,
            "frequency": monitoring_frequency,
            "start_time": datetime.now().isoformat(),
            "status": "active",
            "parameters_monitored": self._get_monitoring_parameters(patient_data, risk_level)
        }

        logger.info(f"Initialized monitoring session {session['session_id']} for patient {patient_id}")
        return session

    async def _assess_current_patient_status(self, patient_data: Dict) -> Dict:
        """Assess current patient health status."""

        vital_signs = patient_data.get("vital_signs", {})
        symptoms = patient_data.get("symptoms", [])
        medical_history = patient_data.get("medical_history", {})
        current_medications = patient_data.get("current_medications", [])

        status_assessment = {
            "overall_stability": self._assess_overall_stability(vital_signs, symptoms),
            "vital_signs_status": self._assess_vital_signs_status(vital_signs),
            "symptom_severity": self._assess_symptom_severity(symptoms),
            "medication_status": self._assess_medication_status(current_medications),
            "risk_indicators": self._identify_risk_indicators(patient_data),
            "trending_concerns": self._identify_trending_concerns(patient_data),
            "immediate_needs": self._identify_immediate_needs(patient_data)
        }

        return status_assessment

    async def _create_monitoring_plan(self, patient_data: Dict, current_status: Dict) -> Dict:
        """Create personalized monitoring plan based on patient status."""

        medical_history = patient_data.get("medical_history", {})
        conditions = medical_history.get("conditions", [])
        risk_level = current_status.get("overall_stability", "moderate")

        monitoring_plan = {
            "vital_signs_schedule": self._create_vital_signs_schedule(risk_level),
            "medication_monitoring": self._create_medication_monitoring_plan(patient_data),
            "symptom_tracking": self._create_symptom_tracking_plan(patient_data),
            "condition_specific_monitoring": self._create_condition_monitoring(conditions),
            "alert_protocols": self._define_alert_protocols(patient_data),
            "escalation_procedures": self._define_escalation_procedures(risk_level)
        }

        return monitoring_plan

    async def _check_for_alerts(self, patient_data: Dict) -> List[Dict]:
        """Check patient data against alert thresholds."""

        alerts = []

        # Check vital signs alerts
        vital_signs = patient_data.get("vital_signs", {})
        vital_alerts = self._check_vital_signs_alerts(vital_signs)
        alerts.extend(vital_alerts)

        # Check symptom alerts
        symptoms = patient_data.get("symptoms", [])
        symptom_alerts = self._check_symptom_alerts(symptoms)
        alerts.extend(symptom_alerts)

        # Check medication alerts
        medications = patient_data.get("current_medications", [])
        medication_alerts = self._check_medication_alerts(medications)
        alerts.extend(medication_alerts)

        # Check lab value alerts
        lab_results = patient_data.get("recent_lab_results", [])
        lab_alerts = self._check_lab_alerts(lab_results)
        alerts.extend(lab_alerts)

        # Prioritize alerts
        prioritized_alerts = self._prioritize_alerts(alerts)

        return prioritized_alerts

    def _assess_patient_risk_level(self, patient_data: Dict) -> str:
        """Assess patient risk level for monitoring intensity."""

        risk_score = 0

        # Age-based risk
        demographics = patient_data.get("demographics", {})
        age = demographics.get("age", 0)
        if age > 75:
            risk_score += 3
        elif age > 65:
            risk_score += 2
        elif age > 45:
            risk_score += 1

        # Condition-based risk
        medical_history = patient_data.get("medical_history", {})
        conditions = medical_history.get("conditions", [])
        high_risk_conditions = [
            "heart_disease", "diabetes", "copd", "kidney_disease",
            "cancer", "stroke", "sepsis"
        ]

        for condition in conditions:
            if condition in high_risk_conditions:
                risk_score += 2

        # Symptom-based risk
        symptoms = patient_data.get("symptoms", [])
        for symptom in symptoms:
            if symptom.get("severity") == "severe":
                risk_score += 2
            elif symptom.get("severity") == "moderate":
                risk_score += 1

        # Vital signs risk
        vital_signs = patient_data.get("vital_signs", {})
        if self._has_critical_vitals(vital_signs):
            risk_score += 3

        # Determine risk level
        if risk_score >= 8:
            return "critical"
        elif risk_score >= 5:
            return "high"
        elif risk_score >= 2:
            return "moderate"
        else:
            return "low"

    def _determine_monitoring_frequency(self, risk_level: str, monitoring_type: str) -> Dict:
        """Determine monitoring frequency based on risk level."""

        base_frequencies = self.monitoring_protocols["vital_signs_monitoring"]["frequency"]

        frequency_map = {
            "critical": {"vital_signs": "every_5_minutes", "assessments": "continuous"},
            "high": {"vital_signs": "every_15_minutes", "assessments": "hourly"},
            "moderate": {"vital_signs": "hourly", "assessments": "every_4_hours"},
            "low": {"vital_signs": "every_4_hours", "assessments": "twice_daily"}
        }

        return frequency_map.get(risk_level, frequency_map["moderate"])

    def _get_monitoring_parameters(self, patient_data: Dict, risk_level: str) -> List[str]:
        """Get list of parameters to monitor based on patient condition and risk."""

        base_parameters = ["blood_pressure", "heart_rate", "temperature", "respiratory_rate"]

        # Add condition-specific parameters
        conditions = patient_data.get("medical_history", {}).get("conditions", [])

        if "diabetes" in conditions:
            base_parameters.extend(["blood_glucose", "ketones"])
        if "heart_disease" in conditions:
            base_parameters.extend(["oxygen_saturation", "cardiac_rhythm"])
        if "respiratory_condition" in conditions:
            base_parameters.extend(["oxygen_saturation", "peak_flow"])

        # Add risk-based parameters
        if risk_level in ["critical", "high"]:
            base_parameters.extend(["pain_level", "consciousness_level", "urine_output"])

        return list(set(base_parameters))  # Remove duplicates

    def _assess_overall_stability(self, vital_signs: Dict, symptoms: List) -> str:
        """Assess overall patient stability."""

        stability_score = 10  # Start with stable

        # Check vital signs stability
        if self._has_critical_vitals(vital_signs):
            stability_score -= 5
        elif self._has_abnormal_vitals(vital_signs):
            stability_score -= 2

        # Check symptom severity
        severe_symptoms = [s for s in symptoms if s.get("severity") == "severe"]
        moderate_symptoms = [s for s in symptoms if s.get("severity") == "moderate"]

        stability_score -= len(severe_symptoms) * 2
        stability_score -= len(moderate_symptoms)

        # Determine stability level
        if stability_score >= 8:
            return "stable"
        elif stability_score >= 5:
            return "concerning"
        elif stability_score >= 2:
            return "unstable"
        else:
            return "critical"

    def _assess_vital_signs_status(self, vital_signs: Dict) -> Dict:
        """Assess status of individual vital signs."""

        status = {}

        for vital, value in vital_signs.items():
            if vital in self.alert_thresholds["vital_signs"]:
                thresholds = self.alert_thresholds["vital_signs"][vital]
                status[vital] = self._classify_vital_sign_status(value, thresholds)
            else:
                status[vital] = {"value": value, "status": "unknown"}

        return status

    def _assess_symptom_severity(self, symptoms: List) -> Dict:
        """Assess overall symptom severity."""

        if not symptoms:
            return {"overall_severity": "none", "symptom_count": 0}

        severity_counts = {"mild": 0, "moderate": 0, "severe": 0}

        for symptom in symptoms:
            severity = symptom.get("severity", "mild")
            if severity in severity_counts:
                severity_counts[severity] += 1

        # Determine overall severity
        if severity_counts["severe"] > 0:
            overall_severity = "severe"
        elif severity_counts["moderate"] > 0:
            overall_severity = "moderate"
        else:
            overall_severity = "mild"

        return {
            "overall_severity": overall_severity,
            "symptom_count": len(symptoms),
            "severity_breakdown": severity_counts
        }

    def _assess_medication_status(self, medications: List) -> Dict:
        """Assess medication-related status."""

        return {
            "total_medications": len(medications),
            "adherence_concerns": [],  # Would be populated with real adherence data
            "interaction_risks": [],   # Would check for drug interactions
            "side_effect_monitoring": "active"
        }

    def _identify_risk_indicators(self, patient_data: Dict) -> List[str]:
        """Identify current risk indicators."""

        indicators = []

        # Vital signs risks
        vital_signs = patient_data.get("vital_signs", {})
        if vital_signs.get("systolic_bp", 0) > 160:
            indicators.append("Hypertensive crisis risk")
        if vital_signs.get("heart_rate", 0) > 120:
            indicators.append("Tachycardia concern")

        # Symptom risks
        symptoms = patient_data.get("symptoms", [])
        chest_pain = [s for s in symptoms if "chest" in s.get("name", "").lower()]
        if chest_pain:
            indicators.append("Cardiac event risk")

        return indicators

    def _identify_trending_concerns(self, patient_data: Dict) -> List[str]:
        """Identify trending health concerns."""

        # In real implementation, would analyze historical data trends
        return [
            "Blood pressure trending upward",
            "Pain levels increasing",
            "Medication adherence declining"
        ]

    def _identify_immediate_needs(self, patient_data: Dict) -> List[str]:
        """Identify immediate patient care needs."""

        needs = []

        # Check for severe symptoms
        symptoms = patient_data.get("symptoms", [])
        severe_symptoms = [s for s in symptoms if s.get("severity") == "severe"]

        if severe_symptoms:
            needs.append("Immediate symptom management required")

        # Check for critical vitals
        vital_signs = patient_data.get("vital_signs", {})
        if self._has_critical_vitals(vital_signs):
            needs.append("Critical vital signs - immediate intervention needed")

        return needs

    def _create_vital_signs_schedule(self, risk_level: str) -> Dict:
        """Create vital signs monitoring schedule."""

        frequency_map = {
            "critical": "every_5_minutes",
            "high": "every_15_minutes",
            "moderate": "hourly",
            "low": "every_4_hours"
        }

        return {
            "frequency": frequency_map.get(risk_level, "hourly"),
            "parameters": ["blood_pressure", "heart_rate", "temperature", "respiratory_rate"],
            "alert_on_change": True,
            "trending_analysis": True
        }

    def _create_medication_monitoring_plan(self, patient_data: Dict) -> Dict:
        """Create medication monitoring plan."""

        return {
            "adherence_tracking": "daily",
            "side_effect_monitoring": "continuous",
            "therapeutic_levels": "as_indicated",
            "interaction_screening": "continuous"
        }

    def _create_symptom_tracking_plan(self, patient_data: Dict) -> Dict:
        """Create symptom tracking plan."""

        symptoms = patient_data.get("symptoms", [])

        return {
            "symptom_assessment_frequency": "every_2_hours",
            "pain_scale_monitoring": "continuous",
            "new_symptom_alerts": True,
            "symptom_trending": True
        }

    def _create_condition_monitoring(self, conditions: List[str]) -> Dict:
        """Create condition-specific monitoring protocols."""

        condition_protocols = {}

        for condition in conditions:
            if condition in self.monitoring_protocols["chronic_condition_monitoring"]:
                condition_protocols[condition] = self.monitoring_protocols["chronic_condition_monitoring"][condition]

        return condition_protocols

    def _define_alert_protocols(self, patient_data: Dict) -> Dict:
        """Define alert protocols for the patient."""

        return {
            "critical_alerts": {
                "notify_immediately": ["physician", "nurse", "family"],
                "escalation_time": "5_minutes"
            },
            "warning_alerts": {
                "notify": ["nurse"],
                "escalation_time": "30_minutes"
            },
            "routine_alerts": {
                "notify": ["care_team"],
                "escalation_time": "2_hours"
            }
        }

    def _define_escalation_procedures(self, risk_level: str) -> Dict:
        """Define escalation procedures based on risk level."""

        procedures = {
            "critical": {
                "immediate_response": "activate_rapid_response_team",
                "notification_chain": ["attending_physician", "nurse_manager", "family"],
                "max_response_time": "2_minutes"
            },
            "high": {
                "immediate_response": "notify_physician",
                "notification_chain": ["physician", "primary_nurse"],
                "max_response_time": "15_minutes"
            },
            "moderate": {
                "immediate_response": "assess_and_document",
                "notification_chain": ["primary_nurse"],
                "max_response_time": "1_hour"
            },
            "low": {
                "immediate_response": "routine_monitoring",
                "notification_chain": ["care_team"],
                "max_response_time": "4_hours"
            }
        }

        return procedures.get(risk_level, procedures["moderate"])

    def _check_vital_signs_alerts(self, vital_signs: Dict) -> List[Dict]:
        """Check vital signs against alert thresholds."""

        alerts = []

        for vital, value in vital_signs.items():
            if vital in self.alert_thresholds["vital_signs"]:
                thresholds = self.alert_thresholds["vital_signs"][vital]
                alert = self._check_threshold_alert(vital, value, thresholds)
                if alert:
                    alerts.append(alert)

        return alerts

    def _check_symptom_alerts(self, symptoms: List) -> List[Dict]:
        """Check symptoms for alert conditions."""

        alerts = []

        for symptom in symptoms:
            severity = symptom.get("severity", "mild")
            name = symptom.get("name", "")

            if severity == "severe":
                alerts.append({
                    "type": "symptom_alert",
                    "severity": "critical",
                    "message": f"Severe symptom: {name}",
                    "timestamp": datetime.now().isoformat(),
                    "requires_immediate_attention": True
                })

        return alerts

    def _check_medication_alerts(self, medications: List) -> List[Dict]:
        """Check for medication-related alerts."""

        alerts = []

        # Check for missing medications (simplified)
        critical_medications = ["insulin", "warfarin", "digoxin"]

        for critical_med in critical_medications:
            if not any(critical_med in med.get("name", "").lower() for med in medications):
                # Would check if patient should be on this medication
                pass

        return alerts

    def _check_lab_alerts(self, lab_results: List) -> List[Dict]:
        """Check laboratory results for alerts."""

        alerts = []

        for result in lab_results:
            test_name = result.get("test_name", "")
            value = result.get("value", 0)

            if test_name in self.alert_thresholds["laboratory_values"]:
                thresholds = self.alert_thresholds["laboratory_values"][test_name]
                alert = self._check_threshold_alert(test_name, value, thresholds)
                if alert:
                    alert["type"] = "laboratory_alert"
                    alerts.append(alert)

        return alerts

    def _check_threshold_alert(self, parameter: str, value: Any, thresholds: Dict) -> Optional[Dict]:
        """Check if a value exceeds alert thresholds."""

        try:
            numeric_value = float(value)

            # Check critical thresholds first
            if "critical_high" in thresholds and numeric_value >= thresholds["critical_high"]:
                return {
                    "type": "vital_sign_alert",
                    "parameter": parameter,
                    "value": numeric_value,
                    "threshold_type": "critical_high",
                    "severity": "critical",
                    "message": f"{parameter} critically high: {numeric_value}",
                    "timestamp": datetime.now().isoformat(),
                    "requires_immediate_attention": True
                }
            elif "critical_low" in thresholds and numeric_value <= thresholds["critical_low"]:
                return {
                    "type": "vital_sign_alert",
                    "parameter": parameter,
                    "value": numeric_value,
                    "threshold_type": "critical_low",
                    "severity": "critical",
                    "message": f"{parameter} critically low: {numeric_value}",
                    "timestamp": datetime.now().isoformat(),
                    "requires_immediate_attention": True
                }
            elif "warning_high" in thresholds and numeric_value >= thresholds["warning_high"]:
                return {
                    "type": "vital_sign_alert",
                    "parameter": parameter,
                    "value": numeric_value,
                    "threshold_type": "warning_high",
                    "severity": "warning",
                    "message": f"{parameter} elevated: {numeric_value}",
                    "timestamp": datetime.now().isoformat(),
                    "requires_immediate_attention": False
                }
            elif "warning_low" in thresholds and numeric_value <= thresholds["warning_low"]:
                return {
                    "type": "vital_sign_alert",
                    "parameter": parameter,
                    "value": numeric_value,
                    "threshold_type": "warning_low",
                    "severity": "warning",
                    "message": f"{parameter} low: {numeric_value}",
                    "timestamp": datetime.now().isoformat(),
                    "requires_immediate_attention": False
                }

        except (ValueError, TypeError):
            return {
                "type": "data_quality_alert",
                "parameter": parameter,
                "value": value,
                "severity": "warning",
                "message": f"Invalid {parameter} value: {value}",
                "timestamp": datetime.now().isoformat(),
                "requires_immediate_attention": False
            }

        return None

    def _prioritize_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """Prioritize alerts by severity and urgency."""

        priority_order = {"critical": 1, "warning": 2, "info": 3}

        return sorted(alerts, key=lambda x: (
            priority_order.get(x.get("severity", "info"), 3),
            not x.get("requires_immediate_attention", False)
        ))

    def _has_critical_vitals(self, vital_signs: Dict) -> bool:
        """Check if patient has any critical vital signs."""

        for vital, value in vital_signs.items():
            if vital in self.alert_thresholds["vital_signs"]:
                thresholds = self.alert_thresholds["vital_signs"][vital]
                try:
                    numeric_value = float(value)
                    if ("critical_high" in thresholds and numeric_value >= thresholds["critical_high"]) or \
                       ("critical_low" in thresholds and numeric_value <= thresholds["critical_low"]):
                        return True
                except (ValueError, TypeError):
                    continue

        return False

    def _has_abnormal_vitals(self, vital_signs: Dict) -> bool:
        """Check if patient has any abnormal vital signs."""

        for vital, value in vital_signs.items():
            if vital in self.alert_thresholds["vital_signs"]:
                thresholds = self.alert_thresholds["vital_signs"][vital]
                try:
                    numeric_value = float(value)
                    if ("warning_high" in thresholds and numeric_value >= thresholds["warning_high"]) or \
                       ("warning_low" in thresholds and numeric_value <= thresholds["warning_low"]):
                        return True
                except (ValueError, TypeError):
                    continue

        return False

    def _classify_vital_sign_status(self, value: Any, thresholds: Dict) -> Dict:
        """Classify vital sign status against thresholds."""

        try:
            numeric_value = float(value)

            if "critical_high" in thresholds and numeric_value >= thresholds["critical_high"]:
                status = "critical_high"
            elif "critical_low" in thresholds and numeric_value <= thresholds["critical_low"]:
                status = "critical_low"
            elif "warning_high" in thresholds and numeric_value >= thresholds["warning_high"]:
                status = "warning_high"
            elif "warning_low" in thresholds and numeric_value <= thresholds["warning_low"]:
                status = "warning_low"
            else:
                status = "normal"

            return {
                "value": numeric_value,
                "status": status,
                "requires_attention": status.startswith("critical")
            }

        except (ValueError, TypeError):
            return {
                "value": value,
                "status": "invalid",
                "requires_attention": True
            }

    def _generate_monitoring_recommendations(self, patient_data: Dict, current_status: Dict) -> List[str]:
        """Generate monitoring recommendations based on patient status."""

        recommendations = []

        # Risk-based recommendations
        stability = current_status.get("overall_stability", "stable")

        if stability == "critical":
            recommendations.extend([
                "Continuous monitoring required",
                "Frequent vital sign assessments",
                "Consider ICU-level care",
                "Immediate physician evaluation"
            ])
        elif stability == "unstable":
            recommendations.extend([
                "Increased monitoring frequency",
                "Regular physician assessments",
                "Consider telemetry monitoring"
            ])
        elif stability == "concerning":
            recommendations.extend([
                "Close observation",
                "Regular vital sign monitoring",
                "Document all changes"
            ])
        else:
            recommendations.extend([
                "Continue routine monitoring",
                "Maintain current care plan"
            ])

        # Condition-specific recommendations
        conditions = patient_data.get("medical_history", {}).get("conditions", [])

        if "diabetes" in conditions:
            recommendations.append("Monitor blood glucose levels closely")
        if "hypertension" in conditions:
            recommendations.append("Regular blood pressure monitoring")
        if "heart_disease" in conditions:
            recommendations.append("Cardiac rhythm monitoring")

        return recommendations

    def _schedule_next_assessment(self, patient_data: Dict, current_status: Dict) -> Dict:
        """Schedule next patient assessment based on current status."""

        stability = current_status.get("overall_stability", "stable")

        time_intervals = {
            "critical": timedelta(minutes=15),
            "unstable": timedelta(hours=1),
            "concerning": timedelta(hours=4),
            "stable": timedelta(hours=8)
        }

        next_assessment_time = datetime.now() + time_intervals.get(stability, timedelta(hours=4))

        return {
            "scheduled_time": next_assessment_time.isoformat(),
            "assessment_type": "comprehensive" if stability in ["critical", "unstable"] else "routine",
            "priority": "urgent" if stability == "critical" else "routine"
        }

    def _get_emergency_protocols(self, patient_data: Dict) -> Dict:
        """Get emergency protocols for the patient."""

        return {
            "code_blue_criteria": [
                "Cardiac arrest",
                "Respiratory arrest",
                "Severe hypotension"
            ],
            "rapid_response_criteria": [
                "Systolic BP < 90 or > 180",
                "Heart rate < 50 or > 120",
                "Respiratory rate < 8 or > 30",
                "O2 saturation < 90%",
                "Altered mental status"
            ],
            "notification_contacts": {
                "primary_physician": "Dr. Smith (555-0123)",
                "nurse_manager": "Jane Doe RN (555-0456)",
                "emergency_contact": patient_data.get("demographics", {}).get("emergency_contact", "Not provided")
            }
        }
