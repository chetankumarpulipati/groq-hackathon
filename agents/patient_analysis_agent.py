"""
Patient Data Analysis Agent for comprehensive healthcare analytics.
Analyzes patient data trends, health metrics, and generates insights.
"""

import asyncio
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from utils.logging import get_logger
from utils.error_handling import handle_exception

logger = get_logger("patient_analysis_agent")


class PatientAnalysisAgent(BaseAgent):
    """
    Specialized agent for comprehensive patient data analysis and health insights.
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id, specialized_task="patient_analysis")
        self.analysis_models = self._initialize_analysis_models()
        logger.info(f"PatientAnalysisAgent {self.agent_id} initialized")

    def _initialize_analysis_models(self) -> Dict[str, Any]:
        """Initialize analysis models and thresholds."""
        return {
            "vital_signs_ranges": {
                "systolic_bp": {"normal": (90, 120), "elevated": (120, 130), "high": (130, 180)},
                "diastolic_bp": {"normal": (60, 80), "elevated": (80, 90), "high": (90, 120)},
                "heart_rate": {"normal": (60, 100), "low": (40, 60), "high": (100, 150)},
                "temperature": {"normal": (97.0, 99.5), "fever": (99.5, 102.0), "high_fever": (102.0, 106.0)},
                "respiratory_rate": {"normal": (12, 20), "low": (8, 12), "high": (20, 30)},
                "oxygen_saturation": {"normal": (95, 100), "low": (90, 95), "critical": (0, 90)}
            },
            "risk_factors": {
                "cardiovascular": ["hypertension", "diabetes", "high_cholesterol", "smoking", "family_history_cardiac"],
                "respiratory": ["asthma", "copd", "smoking", "allergies"],
                "metabolic": ["diabetes", "obesity", "metabolic_syndrome"]
            }
        }

    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process patient data analysis request."""
        try:
            patient_data = input_data.get("patient_data", {}) if isinstance(input_data, dict) else {}
            analysis_type = input_data.get("analysis_type", "comprehensive") if isinstance(input_data, dict) else "comprehensive"

            # Perform comprehensive patient analysis
            analysis_result = await self._analyze_patient_data(patient_data, analysis_type)

            return {
                "agent_id": self.agent_id,
                "analysis_type": analysis_type,
                "patient_id": patient_data.get("patient_id", "unknown"),
                "analysis": analysis_result,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }

        except Exception as e:
            logger.error(f"Patient analysis failed: {e}")
            return {
                "agent_id": self.agent_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _analyze_patient_data(self, patient_data: Dict, analysis_type: str) -> Dict:
        """Perform comprehensive patient data analysis."""

        analysis_result = {
            "demographic_analysis": self._analyze_demographics(patient_data.get("demographics", {})),
            "vital_signs_analysis": self._analyze_vital_signs(patient_data.get("vital_signs", {})),
            "medical_history_analysis": self._analyze_medical_history(patient_data.get("medical_history", {})),
            "symptoms_analysis": self._analyze_symptoms(patient_data.get("symptoms", [])),
            "risk_assessment": self._assess_health_risks(patient_data),
            "health_trends": await self._analyze_health_trends(patient_data),
            "recommendations": self._generate_recommendations(patient_data)
        }

        # Add overall health score
        analysis_result["overall_health_score"] = self._calculate_health_score(analysis_result)

        return analysis_result

    def _analyze_demographics(self, demographics: Dict) -> Dict:
        """Analyze patient demographic information."""
        age = demographics.get("age", 0)
        gender = demographics.get("gender", "unknown")

        age_category = "unknown"
        if age < 18:
            age_category = "pediatric"
        elif age < 65:
            age_category = "adult"
        else:
            age_category = "elderly"

        risk_factors = []
        if age > 65:
            risk_factors.append("advanced_age")
        if age > 45 and gender.lower() == "male":
            risk_factors.append("male_over_45")
        if age > 55 and gender.lower() == "female":
            risk_factors.append("female_over_55")

        return {
            "age_category": age_category,
            "demographic_risk_factors": risk_factors,
            "age_related_considerations": self._get_age_considerations(age)
        }

    def _analyze_vital_signs(self, vital_signs: Dict) -> Dict:
        """Analyze vital signs and identify abnormalities."""
        analysis = {
            "normal_ranges": {},
            "abnormal_findings": [],
            "severity_flags": []
        }

        for vital, value in vital_signs.items():
            if vital in self.analysis_models["vital_signs_ranges"]:
                ranges = self.analysis_models["vital_signs_ranges"][vital]
                status = self._classify_vital_sign(value, ranges)
                analysis["normal_ranges"][vital] = status

                if status != "normal":
                    analysis["abnormal_findings"].append({
                        "vital_sign": vital,
                        "value": value,
                        "status": status
                    })

                    if status in ["high", "critical", "high_fever"]:
                        analysis["severity_flags"].append(f"Critical {vital}: {value}")

        return analysis

    def _analyze_medical_history(self, medical_history: Dict) -> Dict:
        """Analyze patient medical history for risk patterns."""
        conditions = medical_history.get("conditions", [])
        medications = medical_history.get("medications", [])
        allergies = medical_history.get("allergies", [])
        family_history = medical_history.get("family_history", [])

        # Identify risk categories
        risk_categories = {}
        for category, risk_conditions in self.analysis_models["risk_factors"].items():
            patient_risks = [cond for cond in conditions if cond in risk_conditions]
            family_risks = [cond for cond in family_history if cond in risk_conditions]

            if patient_risks or family_risks:
                risk_categories[category] = {
                    "patient_conditions": patient_risks,
                    "family_history": family_risks
                }

        return {
            "chronic_conditions": conditions,
            "medication_count": len(medications),
            "allergy_alerts": allergies,
            "risk_categories": risk_categories,
            "polypharmacy_risk": len(medications) > 5
        }

    def _analyze_symptoms(self, symptoms: List) -> Dict:
        """Analyze current symptoms for severity and patterns."""
        if not symptoms:
            return {"status": "no_symptoms", "severity": "none"}

        severity_levels = []
        symptom_categories = {
            "cardiac": ["chest_pain", "palpitations", "shortness_of_breath"],
            "respiratory": ["cough", "shortness_of_breath", "wheezing"],
            "neurological": ["headache", "dizziness", "confusion"],
            "gastrointestinal": ["nausea", "vomiting", "abdominal_pain"]
        }

        active_categories = {}
        for symptom in symptoms:
            severity = symptom.get("severity", "mild")
            severity_levels.append(severity)

            symptom_name = symptom.get("name", "")
            for category, category_symptoms in symptom_categories.items():
                if symptom_name in category_symptoms:
                    if category not in active_categories:
                        active_categories[category] = []
                    active_categories[category].append(symptom)

        # Determine overall severity
        overall_severity = "mild"
        if "severe" in severity_levels:
            overall_severity = "severe"
        elif "moderate" in severity_levels:
            overall_severity = "moderate"

        return {
            "symptom_count": len(symptoms),
            "overall_severity": overall_severity,
            "affected_systems": active_categories,
            "urgent_symptoms": [s for s in symptoms if s.get("severity") == "severe"]
        }

    def _assess_health_risks(self, patient_data: Dict) -> Dict:
        """Assess overall health risks based on all patient data."""
        demographics = patient_data.get("demographics", {})
        medical_history = patient_data.get("medical_history", {})
        vital_signs = patient_data.get("vital_signs", {})
        symptoms = patient_data.get("symptoms", [])

        risk_score = 0
        risk_factors = []

        # Age-based risk
        age = demographics.get("age", 0)
        if age > 65:
            risk_score += 2
            risk_factors.append("Advanced age")
        elif age > 45:
            risk_score += 1

        # Chronic conditions risk
        conditions = medical_history.get("conditions", [])
        high_risk_conditions = ["diabetes", "hypertension", "heart_disease", "copd"]
        for condition in conditions:
            if condition in high_risk_conditions:
                risk_score += 2
                risk_factors.append(f"Chronic condition: {condition}")

        # Vital signs risk
        vital_analysis = self._analyze_vital_signs(vital_signs)
        if vital_analysis["severity_flags"]:
            risk_score += 3
            risk_factors.extend(vital_analysis["severity_flags"])

        # Symptoms risk
        symptom_analysis = self._analyze_symptoms(symptoms)
        if symptom_analysis.get("overall_severity") == "severe":
            risk_score += 3
            risk_factors.append("Severe symptoms present")
        elif symptom_analysis.get("overall_severity") == "moderate":
            risk_score += 1

        # Categorize risk level
        if risk_score >= 6:
            risk_level = "high"
        elif risk_score >= 3:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "immediate_attention_needed": risk_score >= 6
        }

    async def _analyze_health_trends(self, patient_data: Dict) -> Dict:
        """Analyze health trends over time (simulated for demo)."""
        # In real implementation, this would analyze historical data
        return {
            "trend_analysis": "Historical data analysis not available",
            "improvement_areas": ["Regular monitoring recommended"],
            "deterioration_indicators": [],
            "stability_metrics": "Baseline assessment completed"
        }

    def _generate_recommendations(self, patient_data: Dict) -> List[str]:
        """Generate personalized health recommendations."""
        recommendations = []

        # Base recommendations
        recommendations.append("Maintain regular follow-up appointments")
        recommendations.append("Monitor vital signs regularly")

        # Risk-based recommendations
        medical_history = patient_data.get("medical_history", {})
        conditions = medical_history.get("conditions", [])

        if "hypertension" in conditions:
            recommendations.append("Monitor blood pressure daily")
            recommendations.append("Follow low-sodium diet")

        if "diabetes" in conditions:
            recommendations.append("Monitor blood glucose levels")
            recommendations.append("Maintain regular meal schedule")

        # Symptom-based recommendations
        symptoms = patient_data.get("symptoms", [])
        for symptom in symptoms:
            if symptom.get("severity") == "severe":
                recommendations.append(f"Seek immediate medical attention for {symptom.get('name')}")

        return recommendations

    def _calculate_health_score(self, analysis: Dict) -> int:
        """Calculate overall health score (0-100)."""
        base_score = 80

        # Deduct for risk factors
        risk_level = analysis.get("risk_assessment", {}).get("risk_level", "low")
        if risk_level == "high":
            base_score -= 30
        elif risk_level == "moderate":
            base_score -= 15

        # Deduct for abnormal vital signs
        vital_analysis = analysis.get("vital_signs_analysis", {})
        abnormal_count = len(vital_analysis.get("abnormal_findings", []))
        base_score -= abnormal_count * 5

        # Deduct for severe symptoms
        symptoms_analysis = analysis.get("symptoms_analysis", {})
        if symptoms_analysis.get("overall_severity") == "severe":
            base_score -= 20
        elif symptoms_analysis.get("overall_severity") == "moderate":
            base_score -= 10

        return max(0, min(100, base_score))

    def _classify_vital_sign(self, value: float, ranges: Dict) -> str:
        """Classify vital sign value against normal ranges."""
        try:
            value = float(value)
            for status, (min_val, max_val) in ranges.items():
                if min_val <= value <= max_val:
                    return status
            return "abnormal"
        except (ValueError, TypeError):
            return "invalid"

    def _get_age_considerations(self, age: int) -> List[str]:
        """Get age-specific health considerations."""
        if age < 18:
            return ["Pediatric care protocols", "Growth and development monitoring"]
        elif age > 65:
            return ["Geriatric care considerations", "Fall risk assessment", "Medication review"]
        else:
            return ["Adult preventive care", "Lifestyle counseling"]
