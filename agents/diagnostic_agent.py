"""
Diagnostic Agent for medical analysis and clinical decision support.
Uses advanced AI models for accurate healthcare diagnosis.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from agents.base_agent import BaseAgent
from utils.logging import get_logger

logger = get_logger("diagnostic_agent")

class DiagnosticAgent(BaseAgent):
    """
    Specialized agent for medical diagnosis and clinical analysis with advanced diagnostic assistance.
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id, specialized_task="diagnosis")
        self.diagnostic_knowledge = self._initialize_diagnostic_knowledge()
        self.clinical_guidelines = self._initialize_clinical_guidelines()
        self.diagnostic_algorithms = self._initialize_diagnostic_algorithms()
        logger.info(f"Enhanced DiagnosticAgent {self.agent_id} initialized")

    def _initialize_diagnostic_knowledge(self) -> Dict[str, Any]:
        """Initialize comprehensive diagnostic knowledge base."""
        return {
            "symptom_patterns": {
                "cardiovascular": {
                    "chest_pain": {
                        "differential": [
                            {"condition": "myocardial_infarction", "probability": 0.25, "urgency": "critical"},
                            {"condition": "angina", "probability": 0.20, "urgency": "urgent"},
                            {"condition": "pericarditis", "probability": 0.15, "urgency": "moderate"},
                            {"condition": "costochondritis", "probability": 0.30, "urgency": "low"},
                            {"condition": "gastroesophageal_reflux", "probability": 0.10, "urgency": "low"}
                        ],
                        "red_flags": ["crushing pain", "radiation to arm", "diaphoresis", "nausea"],
                        "investigations": ["ECG", "cardiac_enzymes", "chest_xray", "echocardiogram"]
                    },
                    "shortness_of_breath": {
                        "differential": [
                            {"condition": "heart_failure", "probability": 0.30, "urgency": "urgent"},
                            {"condition": "pulmonary_embolism", "probability": 0.20, "urgency": "critical"},
                            {"condition": "asthma", "probability": 0.25, "urgency": "moderate"},
                            {"condition": "pneumonia", "probability": 0.25, "urgency": "urgent"}
                        ],
                        "red_flags": ["acute onset", "hemoptysis", "chest pain", "syncope"],
                        "investigations": ["chest_xray", "d_dimer", "arterial_blood_gas", "echocardiogram"]
                    }
                },
                "respiratory": {
                    "cough": {
                        "differential": [
                            {"condition": "viral_infection", "probability": 0.40, "urgency": "low"},
                            {"condition": "bacterial_pneumonia", "probability": 0.25, "urgency": "moderate"},
                            {"condition": "asthma", "probability": 0.20, "urgency": "moderate"},
                            {"condition": "copd_exacerbation", "probability": 0.15, "urgency": "moderate"}
                        ],
                        "red_flags": ["hemoptysis", "fever", "weight_loss", "night_sweats"],
                        "investigations": ["chest_xray", "sputum_culture", "complete_blood_count"]
                    }
                },
                "neurological": {
                    "headache": {
                        "differential": [
                            {"condition": "tension_headache", "probability": 0.45, "urgency": "low"},
                            {"condition": "migraine", "probability": 0.25, "urgency": "low"},
                            {"condition": "cluster_headache", "probability": 0.10, "urgency": "moderate"},
                            {"condition": "intracranial_pressure", "probability": 0.10, "urgency": "critical"},
                            {"condition": "meningitis", "probability": 0.10, "urgency": "critical"}
                        ],
                        "red_flags": ["sudden onset", "worst headache ever", "fever", "neck stiffness", "altered consciousness"],
                        "investigations": ["ct_head", "lumbar_puncture", "mri_brain"]
                    }
                }
            },
            "risk_factors": {
                "cardiovascular": ["age>45_male", "age>55_female", "smoking", "diabetes", "hypertension", "hyperlipidemia", "family_history"],
                "respiratory": ["smoking", "occupational_exposure", "family_history_asthma", "allergies"],
                "neurological": ["hypertension", "diabetes", "smoking", "atrial_fibrillation", "previous_stroke"],
                "metabolic": ["obesity", "sedentary_lifestyle", "family_history_diabetes", "gestational_diabetes"]
            },
            "age_considerations": {
                "pediatric": {
                    "common_conditions": ["viral_infections", "asthma", "allergic_reactions"],
                    "red_flags": ["fever_in_infant", "difficulty_breathing", "poor_feeding", "lethargy"]
                },
                "adult": {
                    "common_conditions": ["viral_infections", "stress_related", "lifestyle_disorders"],
                    "screening_recommendations": ["blood_pressure", "cholesterol", "diabetes", "cancer_screening"]
                },
                "elderly": {
                    "common_conditions": ["cardiovascular_disease", "diabetes", "cognitive_decline", "falls"],
                    "red_flags": ["confusion", "falls", "medication_interactions", "polypharmacy"],
                    "geriatric_syndromes": ["frailty", "cognitive_impairment", "incontinence"]
                }
            }
        }

    def _initialize_clinical_guidelines(self) -> Dict[str, Any]:
        """Initialize clinical practice guidelines."""
        return {
            "chest_pain_protocol": {
                "triage_criteria": {
                    "critical": ["st_elevation", "hemodynamic_instability", "cardiogenic_shock"],
                    "urgent": ["unstable_angina", "nstemi", "aortic_dissection"],
                    "moderate": ["stable_angina", "pericarditis"],
                    "low": ["musculoskeletal", "gerd", "anxiety"]
                },
                "investigations_sequence": [
                    {"test": "ECG", "timing": "immediate", "indication": "all_chest_pain"},
                    {"test": "cardiac_enzymes", "timing": "stat", "indication": "suspected_acs"},
                    {"test": "chest_xray", "timing": "urgent", "indication": "exclude_other_causes"},
                    {"test": "ct_angiogram", "timing": "urgent", "indication": "suspected_pe_or_dissection"}
                ]
            },
            "shortness_of_breath_protocol": {
                "assessment_steps": [
                    "vital_signs_assessment",
                    "oxygen_saturation",
                    "respiratory_examination",
                    "cardiovascular_examination"
                ],
                "investigations": [
                    {"test": "arterial_blood_gas", "indication": "severe_dyspnea"},
                    {"test": "chest_xray", "indication": "all_cases"},
                    {"test": "d_dimer", "indication": "suspected_pe"},
                    {"test": "bnp_or_nt_probnp", "indication": "suspected_heart_failure"}
                ]
            },
            "fever_protocol": {
                "age_specific_guidelines": {
                    "infant_0_3_months": {
                        "threshold": 38.0,
                        "urgency": "critical",
                        "workup": ["blood_culture", "urine_culture", "lumbar_puncture"]
                    },
                    "child_3_months_to_3_years": {
                        "threshold": 39.0,
                        "assessment": "clinical_appearance_based",
                        "workup": ["focused_examination", "urinalysis"]
                    },
                    "adult": {
                        "threshold": 38.5,
                        "assessment": "symptom_directed",
                        "workup": ["history_and_physical", "targeted_investigations"]
                    }
                }
            }
        }

    def _initialize_diagnostic_algorithms(self) -> Dict[str, Any]:
        """Initialize diagnostic decision algorithms."""
        return {
            "ottawa_rules": {
                "ankle": {
                    "criteria": [
                        "bone_tenderness_at_posterior_edge_lateral_malleolus",
                        "bone_tenderness_at_posterior_edge_medial_malleolus",
                        "inability_to_bear_weight_immediately_and_in_ed"
                    ],
                    "sensitivity": 0.99,
                    "specificity": 0.40
                },
                "knee": {
                    "criteria": [
                        "age_55_or_older",
                        "isolated_tenderness_of_patella",
                        "tenderness_at_head_of_fibula",
                        "inability_to_flex_to_90_degrees",
                        "inability_to_bear_weight_immediately_and_in_ed"
                    ]
                }
            },
            "wells_score": {
                "pulmonary_embolism": {
                    "criteria": [
                        {"item": "clinical_signs_dvt", "points": 3.0},
                        {"item": "pe_most_likely_diagnosis", "points": 3.0},
                        {"item": "heart_rate_over_100", "points": 1.5},
                        {"item": "immobilization_surgery", "points": 1.5},
                        {"item": "previous_pe_dvt", "points": 1.5},
                        {"item": "hemoptysis", "points": 1.0},
                        {"item": "malignancy", "points": 1.0}
                    ],
                    "interpretation": {
                        "low": {"range": [0, 4.0], "pe_probability": "12.1%"},
                        "moderate": {"range": [4.0, 6.0], "pe_probability": "28.5%"},
                        "high": {"range": [6.0, float('inf')], "pe_probability": "59.0%"}
                    }
                }
            },
            "chads_vasc_score": {
                "criteria": [
                    {"item": "congestive_heart_failure", "points": 1},
                    {"item": "hypertension", "points": 1},
                    {"item": "age_75_or_older", "points": 2},
                    {"item": "diabetes", "points": 1},
                    {"item": "stroke_tia_thromboembolism", "points": 2},
                    {"item": "vascular_disease", "points": 1},
                    {"item": "age_65_74", "points": 1},
                    {"item": "female_gender", "points": 1}
                ]
            }
        }

    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process comprehensive diagnostic analysis request."""
        try:
            patient_data = input_data.get("patient_data", {}) if isinstance(input_data, dict) else {}
            analysis_type = input_data.get("analysis_type", "comprehensive") if isinstance(input_data, dict) else "comprehensive"

            # Perform comprehensive diagnostic analysis
            diagnosis_result = await self._perform_comprehensive_diagnosis(patient_data, analysis_type)

            return {
                "agent_id": self.agent_id,
                "analysis_type": analysis_type,
                "patient_id": patient_data.get("patient_id", "unknown"),
                "diagnosis": diagnosis_result,
                "confidence_score": diagnosis_result.get("overall_confidence", 0.85),
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }

        except Exception as e:
            logger.error(f"Diagnostic processing failed: {e}")
            return {
                "agent_id": self.agent_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _perform_comprehensive_diagnosis(self, patient_data: Dict, analysis_type: str) -> Dict:
        """Perform comprehensive diagnostic analysis with clinical decision support."""

        # Extract patient information
        demographics = patient_data.get("demographics", {})
        symptoms = patient_data.get("symptoms", [])
        medical_history = patient_data.get("medical_history", {})
        vital_signs = patient_data.get("vital_signs", {})
        physical_exam = patient_data.get("physical_exam", {})

        # Perform multi-step diagnostic analysis
        diagnosis_result = {
            "clinical_assessment": await self._perform_clinical_assessment(patient_data),
            "differential_diagnosis": self._generate_differential_diagnosis(symptoms, demographics, medical_history),
            "risk_stratification": self._perform_risk_stratification(patient_data),
            "diagnostic_workup": self._recommend_diagnostic_workup(symptoms, demographics),
            "clinical_decision_support": await self._provide_clinical_decision_support(patient_data),
            "treatment_recommendations": self._generate_treatment_recommendations(patient_data),
            "follow_up_plan": self._create_follow_up_plan(patient_data),
            "red_flag_assessment": self._assess_red_flags(symptoms, vital_signs),
            "clinical_reasoning": self._generate_clinical_reasoning(patient_data)
        }

        # Calculate overall confidence
        diagnosis_result["overall_confidence"] = self._calculate_diagnostic_confidence(diagnosis_result)

        return diagnosis_result

    async def _perform_clinical_assessment(self, patient_data: Dict) -> Dict:
        """Perform systematic clinical assessment."""

        demographics = patient_data.get("demographics", {})
        symptoms = patient_data.get("symptoms", [])
        vital_signs = patient_data.get("vital_signs", {})

        assessment = {
            "chief_complaint": self._identify_chief_complaint(symptoms),
            "severity_assessment": self._assess_symptom_severity(symptoms),
            "acuity_level": self._determine_acuity_level(symptoms, vital_signs),
            "system_review": self._perform_systems_review(symptoms),
            "functional_status": self._assess_functional_status(patient_data),
            "psychosocial_factors": self._assess_psychosocial_factors(patient_data)
        }

        return assessment

    def _generate_differential_diagnosis(self, symptoms: List, demographics: Dict, medical_history: Dict) -> Dict:
        """Generate prioritized differential diagnosis list."""

        if not symptoms:
            return {"status": "insufficient_data", "differentials": []}

        # Identify primary symptom patterns
        primary_symptoms = [s.get("name", "") for s in symptoms if s.get("severity") in ["moderate", "severe"]]

        differential_list = []

        # Analyze each symptom pattern
        for symptom in primary_symptoms:
            symptom_differentials = self._get_symptom_differentials(symptom, demographics, medical_history)
            differential_list.extend(symptom_differentials)

        # Consolidate and rank differentials
        consolidated_differentials = self._consolidate_differentials(differential_list)

        # Apply clinical context
        contextualized_differentials = self._apply_clinical_context(consolidated_differentials, demographics, medical_history)

        return {
            "primary_differentials": contextualized_differentials[:5],  # Top 5
            "secondary_differentials": contextualized_differentials[5:10],  # Next 5
            "total_considered": len(contextualized_differentials),
            "reasoning": self._generate_differential_reasoning(contextualized_differentials)
        }

    def _perform_risk_stratification(self, patient_data: Dict) -> Dict:
        """Perform comprehensive risk stratification."""

        demographics = patient_data.get("demographics", {})
        symptoms = patient_data.get("symptoms", [])
        medical_history = patient_data.get("medical_history", {})

        risk_assessment = {
            "cardiovascular_risk": self._assess_cardiovascular_risk(demographics, medical_history),
            "respiratory_risk": self._assess_respiratory_risk(demographics, medical_history, symptoms),
            "infectious_risk": self._assess_infectious_risk(patient_data),
            "metabolic_risk": self._assess_metabolic_risk(demographics, medical_history),
            "neurological_risk": self._assess_neurological_risk(demographics, medical_history, symptoms),
            "overall_acuity": self._calculate_overall_acuity(patient_data),
            "mortality_risk": self._assess_mortality_risk(patient_data)
        }

        return risk_assessment

    def _recommend_diagnostic_workup(self, symptoms: List, demographics: Dict) -> Dict:
        """Recommend evidence-based diagnostic workup."""

        workup = {
            "immediate_tests": [],
            "urgent_tests": [],
            "routine_tests": [],
            "specialized_studies": [],
            "consultation_recommendations": []
        }

        # Analyze symptoms for test recommendations
        for symptom in symptoms:
            symptom_name = symptom.get("name", "")
            severity = symptom.get("severity", "mild")

            if "chest_pain" in symptom_name:
                if severity in ["moderate", "severe"]:
                    workup["immediate_tests"].extend(["ECG", "cardiac_enzymes"])
                    workup["urgent_tests"].extend(["chest_xray", "d_dimer"])
                else:
                    workup["routine_tests"].extend(["ECG", "chest_xray"])

            elif "shortness_of_breath" in symptom_name:
                workup["immediate_tests"].extend(["pulse_oximetry", "chest_xray"])
                if severity == "severe":
                    workup["urgent_tests"].extend(["arterial_blood_gas", "d_dimer"])

            elif "headache" in symptom_name:
                if severity == "severe" or any("sudden" in str(s) for s in symptoms):
                    workup["urgent_tests"].extend(["ct_head", "lumbar_puncture"])
                    workup["consultation_recommendations"].append("neurology")

        # Age-appropriate screening
        age = demographics.get("age", 0)
        if age > 50:
            workup["routine_tests"].extend(["complete_blood_count", "comprehensive_metabolic_panel"])

        # Remove duplicates
        for category in workup:
            if isinstance(workup[category], list):
                workup[category] = list(set(workup[category]))

        return workup

    async def _provide_clinical_decision_support(self, patient_data: Dict) -> Dict:
        """Provide evidence-based clinical decision support."""

        support = {
            "clinical_guidelines": self._apply_clinical_guidelines(patient_data),
            "decision_algorithms": self._apply_decision_algorithms(patient_data),
            "risk_calculators": self._calculate_risk_scores(patient_data),
            "evidence_based_recommendations": self._generate_evidence_recommendations(patient_data),
            "contraindications": self._check_contraindications(patient_data),
            "drug_interactions": self._check_drug_interactions(patient_data)
        }

        return support

    def _generate_treatment_recommendations(self, patient_data: Dict) -> Dict:
        """Generate evidence-based treatment recommendations."""

        symptoms = patient_data.get("symptoms", [])
        medical_history = patient_data.get("medical_history", {})

        recommendations = {
            "immediate_interventions": [],
            "pharmacological_treatment": [],
            "non_pharmacological_treatment": [],
            "lifestyle_modifications": [],
            "monitoring_requirements": [],
            "follow_up_instructions": []
        }

        # Symptom-based recommendations
        for symptom in symptoms:
            symptom_name = symptom.get("name", "")
            severity = symptom.get("severity", "mild")

            if "pain" in symptom_name and severity in ["moderate", "severe"]:
                recommendations["pharmacological_treatment"].append("appropriate_analgesics")
                recommendations["monitoring_requirements"].append("pain_assessment_scale")

            if "fever" in symptom_name:
                recommendations["pharmacological_treatment"].append("antipyretics_if_indicated")
                recommendations["monitoring_requirements"].append("temperature_monitoring")

        # Condition-specific recommendations
        conditions = medical_history.get("conditions", [])
        for condition in conditions:
            if condition == "hypertension":
                recommendations["lifestyle_modifications"].extend([
                    "low_sodium_diet", "regular_exercise", "weight_management"
                ])
            elif condition == "diabetes":
                recommendations["monitoring_requirements"].extend([
                    "blood_glucose_monitoring", "hba1c_tracking"
                ])

        return recommendations

    def _create_follow_up_plan(self, patient_data: Dict) -> Dict:
        """Create comprehensive follow-up care plan."""

        symptoms = patient_data.get("symptoms", [])
        risk_level = self._calculate_overall_risk_level(patient_data)

        follow_up = {
            "timeline": self._determine_follow_up_timeline(risk_level, symptoms),
            "monitoring_parameters": self._identify_monitoring_parameters(patient_data),
            "warning_signs": self._identify_warning_signs(patient_data),
            "specialist_referrals": self._recommend_specialist_referrals(patient_data),
            "patient_education": self._generate_patient_education(patient_data),
            "care_coordination": self._plan_care_coordination(patient_data)
        }

        return follow_up

    def _assess_red_flags(self, symptoms: List, vital_signs: Dict) -> Dict:
        """Assess for red flag symptoms requiring immediate attention."""

        red_flags = []

        # Vital signs red flags
        if vital_signs.get("systolic_bp", 0) > 180:
            red_flags.append({
                "category": "cardiovascular",
                "finding": "hypertensive_crisis",
                "urgency": "critical",
                "action": "immediate_intervention_required"
            })

        if vital_signs.get("heart_rate", 0) > 120:
            red_flags.append({
                "category": "cardiovascular",
                "finding": "tachycardia",
                "urgency": "urgent",
                "action": "cardiac_evaluation_needed"
            })

        # Symptom-based red flags
        for symptom in symptoms:
            symptom_name = symptom.get("name", "")
            description = symptom.get("description", "")

            if "chest_pain" in symptom_name and any(flag in description.lower() for flag in ["crushing", "radiating", "severe"]):
                red_flags.append({
                    "category": "cardiovascular",
                    "finding": "acute_coronary_syndrome_possible",
                    "urgency": "critical",
                    "action": "emergency_evaluation_required"
                })

        return {
            "red_flags_present": len(red_flags) > 0,
            "total_red_flags": len(red_flags),
            "red_flag_details": red_flags,
            "overall_urgency": self._determine_overall_urgency(red_flags)
        }

    def _generate_clinical_reasoning(self, patient_data: Dict) -> Dict:
        """Generate clinical reasoning and diagnostic thought process."""

        reasoning = {
            "clinical_presentation_summary": self._summarize_clinical_presentation(patient_data),
            "diagnostic_approach": self._describe_diagnostic_approach(patient_data),
            "key_decision_points": self._identify_key_decision_points(patient_data),
            "alternative_considerations": self._consider_alternatives(patient_data),
            "uncertainty_areas": self._identify_uncertainties(patient_data),
            "learning_points": self._extract_learning_points(patient_data)
        }

        return reasoning

    # Helper methods for diagnostic analysis

    def _identify_chief_complaint(self, symptoms: List) -> str:
        """Identify the primary chief complaint."""
        if not symptoms:
            return "No specific complaint identified"

        # Find most severe symptom
        severe_symptoms = [s for s in symptoms if s.get("severity") == "severe"]
        if severe_symptoms:
            return severe_symptoms[0].get("name", "Severe symptom")

        # Find moderate symptoms
        moderate_symptoms = [s for s in symptoms if s.get("severity") == "moderate"]
        if moderate_symptoms:
            return moderate_symptoms[0].get("name", "Moderate symptom")

        return symptoms[0].get("name", "Mild symptom")

    def _assess_symptom_severity(self, symptoms: List) -> Dict:
        """Assess overall symptom severity."""
        if not symptoms:
            return {"overall_severity": "none", "severity_score": 0}

        severity_scores = {"mild": 1, "moderate": 2, "severe": 3}
        total_score = sum(severity_scores.get(s.get("severity", "mild"), 1) for s in symptoms)
        max_possible = len(symptoms) * 3

        severity_percentage = (total_score / max_possible) * 100 if max_possible > 0 else 0

        if severity_percentage > 75:
            overall_severity = "high"
        elif severity_percentage > 50:
            overall_severity = "moderate"
        else:
            overall_severity = "low"

        return {
            "overall_severity": overall_severity,
            "severity_score": severity_percentage,
            "severe_symptom_count": len([s for s in symptoms if s.get("severity") == "severe"])
        }

    def _determine_acuity_level(self, symptoms: List, vital_signs: Dict) -> str:
        """Determine patient acuity level."""
        acuity_score = 0

        # Vital signs contribution
        if vital_signs.get("systolic_bp", 0) > 160 or vital_signs.get("systolic_bp", 0) < 100:
            acuity_score += 2
        if vital_signs.get("heart_rate", 0) > 100 or vital_signs.get("heart_rate", 0) < 60:
            acuity_score += 1

        # Symptom contribution
        severe_symptoms = [s for s in symptoms if s.get("severity") == "severe"]
        acuity_score += len(severe_symptoms) * 2

        if acuity_score >= 4:
            return "high"
        elif acuity_score >= 2:
            return "moderate"
        else:
            return "low"

    def _get_symptom_differentials(self, symptom: str, demographics: Dict, medical_history: Dict) -> List[Dict]:
        """Get differential diagnoses for a specific symptom."""
        differentials = []

        # Search knowledge base for symptom patterns
        for system, patterns in self.diagnostic_knowledge["symptom_patterns"].items():
            for pattern_name, pattern_data in patterns.items():
                if symptom.lower() in pattern_name.lower():
                    for diff in pattern_data.get("differential", []):
                        # Adjust probability based on patient factors
                        adjusted_prob = self._adjust_probability(
                            diff["probability"], demographics, medical_history, diff["condition"]
                        )
                        differentials.append({
                            "condition": diff["condition"],
                            "probability": adjusted_prob,
                            "urgency": diff["urgency"],
                            "system": system
                        })

        return differentials

    def _adjust_probability(self, base_prob: float, demographics: Dict, medical_history: Dict, condition: str) -> float:
        """Adjust probability based on patient-specific factors."""
        adjusted_prob = base_prob

        age = demographics.get("age", 0)
        gender = demographics.get("gender", "").lower()
        conditions = medical_history.get("conditions", [])

        # Age adjustments
        if condition == "myocardial_infarction":
            if age > 65:
                adjusted_prob *= 1.5
            elif age < 40:
                adjusted_prob *= 0.3

        # Gender adjustments
        if condition == "myocardial_infarction" and gender == "male":
            adjusted_prob *= 1.3

        # Risk factor adjustments
        if condition in ["myocardial_infarction", "angina"] and "diabetes" in conditions:
            adjusted_prob *= 1.4

        return min(1.0, adjusted_prob)

    def _consolidate_differentials(self, differential_list: List[Dict]) -> List[Dict]:
        """Consolidate and rank differential diagnoses."""
        # Group by condition
        condition_groups = {}
        for diff in differential_list:
            condition = diff["condition"]
            if condition not in condition_groups:
                condition_groups[condition] = []
            condition_groups[condition].append(diff)

        # Calculate consolidated probabilities
        consolidated = []
        for condition, diffs in condition_groups.items():
            avg_probability = sum(d["probability"] for d in diffs) / len(diffs)
            max_urgency = max(diffs, key=lambda x: self._urgency_score(x["urgency"]))["urgency"]

            consolidated.append({
                "condition": condition,
                "probability": avg_probability,
                "urgency": max_urgency,
                "confidence": min(1.0, len(diffs) * 0.2)  # Higher confidence with more evidence
            })

        # Sort by probability and urgency
        return sorted(consolidated, key=lambda x: (self._urgency_score(x["urgency"]), x["probability"]), reverse=True)

    def _urgency_score(self, urgency: str) -> int:
        """Convert urgency level to numeric score."""
        scores = {"critical": 4, "urgent": 3, "moderate": 2, "low": 1}
        return scores.get(urgency, 1)

    def _calculate_diagnostic_confidence(self, diagnosis_result: Dict) -> float:
        """Calculate overall diagnostic confidence score."""
        confidence_factors = []

        # Differential diagnosis confidence
        differentials = diagnosis_result.get("differential_diagnosis", {}).get("primary_differentials", [])
        if differentials:
            top_diff_prob = differentials[0].get("probability", 0)
            confidence_factors.append(top_diff_prob)

        # Clinical assessment confidence
        red_flags = diagnosis_result.get("red_flag_assessment", {})
        if not red_flags.get("red_flags_present", False):
            confidence_factors.append(0.8)  # Higher confidence if no red flags
        else:
            confidence_factors.append(0.6)

        # Overall confidence
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.7  # Default moderate confidence

    def _calculate_overall_risk_level(self, patient_data: Dict) -> str:
        """Calculate overall patient risk level."""
        # Simplified risk calculation
        symptoms = patient_data.get("symptoms", [])
        severe_symptoms = [s for s in symptoms if s.get("severity") == "severe"]

        if len(severe_symptoms) > 1:
            return "high"
        elif len(severe_symptoms) == 1:
            return "moderate"
        else:
            return "low"

    # Placeholder methods for complex clinical functions
    def _perform_systems_review(self, symptoms: List) -> Dict:
        return {"systems_reviewed": ["cardiovascular", "respiratory", "neurological"]}

    def _assess_functional_status(self, patient_data: Dict) -> Dict:
        return {"functional_level": "independent", "mobility": "ambulatory"}

    def _assess_psychosocial_factors(self, patient_data: Dict) -> Dict:
        return {"stress_level": "moderate", "support_system": "adequate"}

    def _apply_clinical_context(self, differentials: List, demographics: Dict, medical_history: Dict) -> List:
        return differentials  # Already applied in _adjust_probability

    def _generate_differential_reasoning(self, differentials: List) -> str:
        if not differentials:
            return "Insufficient data for differential diagnosis"
        top_diff = differentials[0]
        return f"Primary consideration: {top_diff['condition']} (probability: {top_diff['probability']:.2f})"

    # Additional placeholder methods
    def _assess_cardiovascular_risk(self, demographics: Dict, medical_history: Dict) -> str:
        return "moderate"

    def _assess_respiratory_risk(self, demographics: Dict, medical_history: Dict, symptoms: List) -> str:
        return "low"

    def _assess_infectious_risk(self, patient_data: Dict) -> str:
        return "low"

    def _assess_metabolic_risk(self, demographics: Dict, medical_history: Dict) -> str:
        return "moderate"

    def _assess_neurological_risk(self, demographics: Dict, medical_history: Dict, symptoms: List) -> str:
        return "low"

    def _calculate_overall_acuity(self, patient_data: Dict) -> str:
        return "moderate"

    def _assess_mortality_risk(self, patient_data: Dict) -> str:
        return "low"

    def _apply_clinical_guidelines(self, patient_data: Dict) -> List:
        return ["Follow chest pain protocol", "Consider cardiac workup"]

    def _apply_decision_algorithms(self, patient_data: Dict) -> Dict:
        return {"wells_score": "not_calculated", "ottawa_rules": "not_applicable"}

    def _calculate_risk_scores(self, patient_data: Dict) -> Dict:
        return {"cardiovascular_risk": 0.15, "pe_risk": 0.05}

    def _generate_evidence_recommendations(self, patient_data: Dict) -> List:
        return ["Evidence-based workup recommended", "Follow clinical guidelines"]

    def _check_contraindications(self, patient_data: Dict) -> List:
        return []

    def _check_drug_interactions(self, patient_data: Dict) -> List:
        return []

    def _determine_follow_up_timeline(self, risk_level: str, symptoms: List) -> Dict:
        timelines = {
            "high": {"immediate": "24_hours", "short_term": "1_week"},
            "moderate": {"immediate": "72_hours", "short_term": "2_weeks"},
            "low": {"immediate": "1_week", "short_term": "1_month"}
        }
        return timelines.get(risk_level, timelines["moderate"])

    def _identify_monitoring_parameters(self, patient_data: Dict) -> List:
        return ["vital_signs", "symptom_progression", "functional_status"]

    def _identify_warning_signs(self, patient_data: Dict) -> List:
        return ["worsening_symptoms", "new_chest_pain", "difficulty_breathing"]

    def _recommend_specialist_referrals(self, patient_data: Dict) -> List:
        return ["cardiology", "internal_medicine"]

    def _generate_patient_education(self, patient_data: Dict) -> List:
        return ["symptom_recognition", "when_to_seek_care", "medication_compliance"]

    def _plan_care_coordination(self, patient_data: Dict) -> Dict:
        return {"primary_care": "coordinate_with_pcp", "specialists": "facilitate_referrals"}

    def _determine_overall_urgency(self, red_flags: List) -> str:
        if not red_flags:
            return "routine"
        critical_flags = [rf for rf in red_flags if rf.get("urgency") == "critical"]
        if critical_flags:
            return "critical"
        urgent_flags = [rf for rf in red_flags if rf.get("urgency") == "urgent"]
        if urgent_flags:
            return "urgent"
        return "moderate"

    def _summarize_clinical_presentation(self, patient_data: Dict) -> str:
        symptoms = patient_data.get("symptoms", [])
        if not symptoms:
            return "No specific symptoms reported"
        primary_symptom = symptoms[0].get("name", "symptom")
        return f"Patient presents with {primary_symptom} and associated symptoms"

    def _describe_diagnostic_approach(self, patient_data: Dict) -> str:
        return "Systematic evaluation of symptoms with risk stratification and evidence-based workup"

    def _identify_key_decision_points(self, patient_data: Dict) -> List:
        return ["symptom_severity_assessment", "risk_factor_evaluation", "diagnostic_test_selection"]

    def _consider_alternatives(self, patient_data: Dict) -> List:
        return ["alternative_diagnoses", "atypical_presentations", "rare_conditions"]

    def _identify_uncertainties(self, patient_data: Dict) -> List:
        return ["incomplete_history", "atypical_presentation", "need_for_additional_testing"]

    def _extract_learning_points(self, patient_data: Dict) -> List:
        return ["clinical_decision_making", "evidence_based_practice", "patient_centered_care"]
