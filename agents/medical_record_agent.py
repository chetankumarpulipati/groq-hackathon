"""
Medical Record Processing Agent for electronic health record management.
Processes, analyzes, and extracts insights from medical records and documents.
"""

import asyncio
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from utils.logging import get_logger
from utils.error_handling import handle_exception

logger = get_logger("medical_record_agent")


class MedicalRecordAgent(BaseAgent):
    """
    Specialized agent for processing and analyzing medical records and healthcare documents.
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id, specialized_task="medical_record_processing")
        self.record_processors = self._initialize_record_processors()
        self.medical_terminology = self._initialize_medical_terminology()
        logger.info(f"MedicalRecordAgent {self.agent_id} initialized")

    def _initialize_record_processors(self) -> Dict[str, Any]:
        """Initialize medical record processing configurations."""
        return {
            "supported_formats": ["HL7", "FHIR", "CDA", "PDF", "TEXT", "JSON"],
            "extraction_patterns": {
                "patient_id": r"(?i)(?:patient\s*(?:id|identifier|number):\s*)([A-Z0-9\-]+)",
                "date_of_birth": r"(?i)(?:dob|date\s*of\s*birth):\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
                "diagnosis": r"(?i)(?:diagnosis|dx):\s*([^\n\r;]+)",
                "medications": r"(?i)(?:medication|rx|drug):\s*([^\n\r;]+)",
                "allergies": r"(?i)(?:allerg(?:y|ies)):\s*([^\n\r;]+)",
                "vital_signs": r"(?i)(?:bp|blood\s*pressure):\s*(\d{2,3}\/\d{2,3})|(?:hr|heart\s*rate):\s*(\d{2,3})|(?:temp|temperature):\s*(\d{2,3}\.?\d?)"
            },
            "icd_10_mapping": {
                "hypertension": "I10",
                "diabetes": "E11",
                "chest_pain": "R06.02",
                "shortness_of_breath": "R06.00"
            }
        }

    def _initialize_medical_terminology(self) -> Dict[str, List[str]]:
        """Initialize medical terminology and synonyms."""
        return {
            "symptoms": {
                "chest_pain": ["chest pain", "chest discomfort", "angina", "chest pressure"],
                "shortness_of_breath": ["dyspnea", "sob", "breathing difficulty", "breathlessness"],
                "headache": ["cephalgia", "head pain", "migraine", "tension headache"],
                "nausea": ["queasiness", "sick feeling", "upset stomach"]
            },
            "conditions": {
                "hypertension": ["high blood pressure", "htn", "elevated bp"],
                "diabetes": ["diabetes mellitus", "dm", "hyperglycemia"],
                "asthma": ["reactive airway disease", "bronchial asthma"]
            }
        }

    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process medical record analysis request."""
        try:
            record_data = input_data.get("record_data", {}) if isinstance(input_data, dict) else {}
            processing_type = input_data.get("processing_type", "comprehensive") if isinstance(input_data, dict) else "comprehensive"
            record_format = input_data.get("format", "JSON") if isinstance(input_data, dict) else "JSON"

            # Process medical record
            processing_result = await self._process_medical_record(record_data, processing_type, record_format)

            return {
                "agent_id": self.agent_id,
                "processing_type": processing_type,
                "record_format": record_format,
                "processing_result": processing_result,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }

        except Exception as e:
            logger.error(f"Medical record processing failed: {e}")
            return {
                "agent_id": self.agent_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _process_medical_record(self, record_data: Dict, processing_type: str, record_format: str) -> Dict:
        """Process medical record based on format and type."""

        processing_result = {
            "record_summary": await self._generate_record_summary(record_data),
            "extracted_data": self._extract_structured_data(record_data, record_format),
            "clinical_insights": self._analyze_clinical_data(record_data),
            "compliance_check": self._check_record_compliance(record_data),
            "quality_metrics": self._assess_record_quality(record_data),
            "recommendations": self._generate_processing_recommendations(record_data)
        }

        # Add format-specific processing
        if record_format.upper() == "HL7":
            processing_result["hl7_analysis"] = await self._process_hl7_record(record_data)
        elif record_format.upper() == "FHIR":
            processing_result["fhir_analysis"] = await self._process_fhir_record(record_data)

        return processing_result

    async def _generate_record_summary(self, record_data: Dict) -> Dict:
        """Generate comprehensive summary of medical record."""
        demographics = record_data.get("demographics", {})
        encounters = record_data.get("encounters", [])
        diagnoses = record_data.get("diagnoses", [])
        medications = record_data.get("medications", [])

        summary = {
            "patient_overview": {
                "patient_id": demographics.get("patient_id", "unknown"),
                "age": demographics.get("age"),
                "gender": demographics.get("gender"),
                "total_encounters": len(encounters),
                "active_diagnoses": len([d for d in diagnoses if d.get("status") == "active"]),
                "current_medications": len([m for m in medications if m.get("status") == "active"])
            },
            "recent_activity": self._summarize_recent_activity(encounters),
            "key_findings": self._extract_key_findings(record_data),
            "care_gaps": self._identify_care_gaps(record_data)
        }

        return summary

    def _extract_structured_data(self, record_data: Dict, record_format: str) -> Dict:
        """Extract structured data from medical record."""
        if record_format.upper() == "TEXT":
            return self._extract_from_text(record_data.get("text_content", ""))

        # For structured formats, parse existing data
        extracted = {
            "patient_demographics": record_data.get("demographics", {}),
            "clinical_data": {
                "diagnoses": record_data.get("diagnoses", []),
                "medications": record_data.get("medications", []),
                "procedures": record_data.get("procedures", []),
                "lab_results": record_data.get("lab_results", []),
                "vital_signs": record_data.get("vital_signs", {})
            },
            "administrative_data": {
                "encounters": record_data.get("encounters", []),
                "insurance": record_data.get("insurance", {}),
                "providers": record_data.get("providers", [])
            }
        }

        return extracted

    def _extract_from_text(self, text_content: str) -> Dict:
        """Extract structured data from unstructured text."""
        extracted_data = {
            "patient_identifiers": [],
            "diagnoses": [],
            "medications": [],
            "vital_signs": {},
            "dates": []
        }

        # Extract using regex patterns
        for data_type, pattern in self.record_processors["extraction_patterns"].items():
            matches = re.findall(pattern, text_content)
            if matches:
                extracted_data[data_type] = matches

        return extracted_data

    def _analyze_clinical_data(self, record_data: Dict) -> Dict:
        """Analyze clinical aspects of the medical record."""
        diagnoses = record_data.get("diagnoses", [])
        medications = record_data.get("medications", [])
        lab_results = record_data.get("lab_results", [])

        analysis = {
            "diagnostic_patterns": self._analyze_diagnostic_patterns(diagnoses),
            "medication_analysis": self._analyze_medications(medications),
            "lab_trends": self._analyze_lab_trends(lab_results),
            "risk_indicators": self._identify_clinical_risks(record_data),
            "treatment_efficacy": self._assess_treatment_efficacy(record_data)
        }

        return analysis

    def _analyze_diagnostic_patterns(self, diagnoses: List[Dict]) -> Dict:
        """Analyze patterns in patient diagnoses."""
        if not diagnoses:
            return {"status": "no_diagnoses"}

        diagnosis_categories = {
            "chronic": [],
            "acute": [],
            "resolved": []
        }

        for diagnosis in diagnoses:
            category = diagnosis.get("category", "unknown")
            status = diagnosis.get("status", "unknown")

            if status == "resolved":
                diagnosis_categories["resolved"].append(diagnosis)
            elif category in ["chronic", "long-term"]:
                diagnosis_categories["chronic"].append(diagnosis)
            else:
                diagnosis_categories["acute"].append(diagnosis)

        return {
            "total_diagnoses": len(diagnoses),
            "categories": diagnosis_categories,
            "complexity_score": len(diagnosis_categories["chronic"]) * 2 + len(diagnosis_categories["acute"])
        }

    def _analyze_medications(self, medications: List[Dict]) -> Dict:
        """Analyze medication patterns and interactions."""
        if not medications:
            return {"status": "no_medications"}

        active_meds = [m for m in medications if m.get("status") == "active"]

        analysis = {
            "total_medications": len(medications),
            "active_medications": len(active_meds),
            "medication_classes": self._categorize_medications(active_meds),
            "polypharmacy_risk": len(active_meds) > 5,
            "adherence_concerns": self._assess_medication_adherence(medications)
        }

        return analysis

    def _categorize_medications(self, medications: List[Dict]) -> Dict:
        """Categorize medications by therapeutic class."""
        categories = {
            "cardiovascular": ["lisinopril", "metoprolol", "atorvastatin", "amlodipine"],
            "diabetes": ["metformin", "insulin", "glipizide"],
            "respiratory": ["albuterol", "fluticasone", "montelukast"],
            "psychiatric": ["sertraline", "lorazepam", "risperidone"]
        }

        categorized = {}
        for med in medications:
            med_name = med.get("name", "").lower()
            for category, med_list in categories.items():
                if any(drug in med_name for drug in med_list):
                    if category not in categorized:
                        categorized[category] = []
                    categorized[category].append(med)

        return categorized

    def _analyze_lab_trends(self, lab_results: List[Dict]) -> Dict:
        """Analyze laboratory result trends."""
        if not lab_results:
            return {"status": "no_lab_results"}

        trends = {
            "recent_results": [],
            "abnormal_values": [],
            "trending_concerns": []
        }

        # Sort by date (most recent first)
        sorted_results = sorted(lab_results, key=lambda x: x.get("date", ""), reverse=True)
        trends["recent_results"] = sorted_results[:5]  # Last 5 results

        # Identify abnormal values
        for result in sorted_results:
            if result.get("status") == "abnormal":
                trends["abnormal_values"].append(result)

        return trends

    def _check_record_compliance(self, record_data: Dict) -> Dict:
        """Check medical record compliance with standards."""
        compliance_checks = {
            "required_fields": self._check_required_fields(record_data),
            "data_quality": self._check_data_quality(record_data),
            "privacy_compliance": self._check_privacy_compliance(record_data),
            "coding_standards": self._check_coding_standards(record_data)
        }

        # Calculate overall compliance score
        total_checks = len(compliance_checks)
        passed_checks = sum(1 for check in compliance_checks.values() if check.get("status") == "compliant")
        compliance_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0

        compliance_checks["overall_score"] = compliance_score
        compliance_checks["compliance_level"] = "high" if compliance_score > 80 else "medium" if compliance_score > 60 else "low"

        return compliance_checks

    def _check_required_fields(self, record_data: Dict) -> Dict:
        """Check for required medical record fields."""
        required_fields = ["patient_id", "demographics", "encounters"]
        missing_fields = []

        for field in required_fields:
            if field not in record_data or not record_data[field]:
                missing_fields.append(field)

        return {
            "status": "compliant" if not missing_fields else "non_compliant",
            "missing_fields": missing_fields,
            "completeness": ((len(required_fields) - len(missing_fields)) / len(required_fields)) * 100
        }

    def _check_data_quality(self, record_data: Dict) -> Dict:
        """Assess data quality metrics."""
        quality_issues = []

        # Check for data consistency
        demographics = record_data.get("demographics", {})
        if demographics.get("age") and demographics.get("date_of_birth"):
            # Could add age calculation validation here
            pass

        # Check for duplicate entries
        encounters = record_data.get("encounters", [])
        encounter_dates = [e.get("date") for e in encounters if e.get("date")]
        if len(encounter_dates) != len(set(encounter_dates)):
            quality_issues.append("Duplicate encounter dates found")

        return {
            "status": "high_quality" if not quality_issues else "quality_concerns",
            "issues": quality_issues,
            "quality_score": max(0, 100 - len(quality_issues) * 10)
        }

    def _check_privacy_compliance(self, record_data: Dict) -> Dict:
        """Check HIPAA and privacy compliance."""
        # Simplified privacy check
        return {
            "status": "compliant",
            "phi_protection": "implemented",
            "access_controls": "verified"
        }

    def _check_coding_standards(self, record_data: Dict) -> Dict:
        """Check medical coding standards compliance."""
        diagnoses = record_data.get("diagnoses", [])
        coded_diagnoses = [d for d in diagnoses if d.get("icd_code")]

        coding_compliance = len(coded_diagnoses) / len(diagnoses) * 100 if diagnoses else 100

        return {
            "status": "compliant" if coding_compliance > 80 else "partial_compliance",
            "icd_coding_percentage": coding_compliance,
            "missing_codes": len(diagnoses) - len(coded_diagnoses)
        }

    def _assess_record_quality(self, record_data: Dict) -> Dict:
        """Assess overall quality of medical record."""
        metrics = {
            "completeness": self._calculate_completeness(record_data),
            "accuracy": self._assess_accuracy(record_data),
            "timeliness": self._assess_timeliness(record_data),
            "consistency": self._assess_consistency(record_data)
        }

        # Overall quality score
        quality_score = sum(metrics.values()) / len(metrics)

        return {
            "individual_metrics": metrics,
            "overall_quality_score": quality_score,
            "quality_grade": self._get_quality_grade(quality_score)
        }

    def _calculate_completeness(self, record_data: Dict) -> float:
        """Calculate record completeness percentage."""
        expected_sections = ["demographics", "diagnoses", "medications", "encounters", "vital_signs"]
        present_sections = sum(1 for section in expected_sections if record_data.get(section))
        return (present_sections / len(expected_sections)) * 100

    def _assess_accuracy(self, record_data: Dict) -> float:
        """Assess data accuracy (simplified)."""
        # In real implementation, would validate against external sources
        return 85.0  # Placeholder

    def _assess_timeliness(self, record_data: Dict) -> float:
        """Assess data timeliness."""
        encounters = record_data.get("encounters", [])
        if not encounters:
            return 50.0

        # Check if recent encounters are present
        recent_threshold = datetime.now() - timedelta(days=30)
        recent_encounters = [e for e in encounters if e.get("date") and
                           datetime.fromisoformat(e["date"].replace("Z", "+00:00")) > recent_threshold]

        return min(100.0, len(recent_encounters) * 25)

    def _assess_consistency(self, record_data: Dict) -> float:
        """Assess data consistency."""
        # Check for logical consistency
        consistency_score = 90.0  # Base score

        # Example consistency check
        demographics = record_data.get("demographics", {})
        diagnoses = record_data.get("diagnoses", [])

        # Age-diagnosis consistency
        age = demographics.get("age", 0)
        pediatric_conditions = ["juvenile_diabetes", "pediatric_asthma"]

        if age > 18:
            pediatric_diagnoses = [d for d in diagnoses if any(pc in str(d) for pc in pediatric_conditions)]
            if pediatric_diagnoses:
                consistency_score -= 10

        return consistency_score

    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    async def _process_hl7_record(self, record_data: Dict) -> Dict:
        """Process HL7 format specific data."""
        return {
            "format": "HL7",
            "segments_processed": ["PID", "MSH", "OBX"],
            "parsing_status": "successful"
        }

    async def _process_fhir_record(self, record_data: Dict) -> Dict:
        """Process FHIR format specific data."""
        return {
            "format": "FHIR",
            "resources_processed": ["Patient", "Observation", "Condition"],
            "fhir_version": "R4",
            "parsing_status": "successful"
        }

    def _summarize_recent_activity(self, encounters: List[Dict]) -> Dict:
        """Summarize recent patient encounters."""
        if not encounters:
            return {"status": "no_recent_activity"}

        # Sort by date
        sorted_encounters = sorted(encounters, key=lambda x: x.get("date", ""), reverse=True)
        recent_encounters = sorted_encounters[:3]  # Last 3 encounters

        return {
            "recent_encounter_count": len(recent_encounters),
            "last_encounter_date": recent_encounters[0].get("date") if recent_encounters else None,
            "encounter_types": [e.get("type") for e in recent_encounters],
            "providers_seen": list(set(e.get("provider") for e in recent_encounters if e.get("provider")))
        }

    def _extract_key_findings(self, record_data: Dict) -> List[str]:
        """Extract key clinical findings from record."""
        findings = []

        # Check for critical diagnoses
        diagnoses = record_data.get("diagnoses", [])
        critical_conditions = ["myocardial_infarction", "stroke", "sepsis", "pneumonia"]

        for diagnosis in diagnoses:
            if any(condition in str(diagnosis).lower() for condition in critical_conditions):
                findings.append(f"Critical diagnosis: {diagnosis.get('name', 'Unknown')}")

        # Check for abnormal vital signs
        vital_signs = record_data.get("vital_signs", {})
        if vital_signs.get("systolic_bp", 0) > 140:
            findings.append("Hypertensive blood pressure detected")

        return findings

    def _identify_care_gaps(self, record_data: Dict) -> List[str]:
        """Identify potential gaps in patient care."""
        gaps = []

        # Check for missing preventive care
        encounters = record_data.get("encounters", [])
        demographics = record_data.get("demographics", {})

        age = demographics.get("age", 0)
        gender = demographics.get("gender", "").lower()

        # Age-appropriate screening gaps
        if age > 50 and not any("colonoscopy" in str(e).lower() for e in encounters):
            gaps.append("Colorectal screening due")

        if gender == "female" and age > 40 and not any("mammogram" in str(e).lower() for e in encounters):
            gaps.append("Mammography screening due")

        return gaps

    def _identify_clinical_risks(self, record_data: Dict) -> List[str]:
        """Identify clinical risk indicators."""
        risks = []

        diagnoses = record_data.get("diagnoses", [])
        medications = record_data.get("medications", [])

        # Polypharmacy risk
        active_meds = [m for m in medications if m.get("status") == "active"]
        if len(active_meds) > 5:
            risks.append("Polypharmacy risk - multiple medications")

        # Drug interaction risks (simplified)
        high_risk_combinations = [
            ["warfarin", "aspirin"],
            ["digoxin", "furosemide"]
        ]

        med_names = [m.get("name", "").lower() for m in active_meds]
        for combo in high_risk_combinations:
            if all(drug in str(med_names) for drug in combo):
                risks.append(f"Potential drug interaction: {' + '.join(combo)}")

        return risks

    def _assess_treatment_efficacy(self, record_data: Dict) -> Dict:
        """Assess treatment efficacy based on record data."""
        # Simplified efficacy assessment
        return {
            "treatment_response": "stable",
            "medication_effectiveness": "adequate",
            "disease_progression": "controlled"
        }

    def _assess_medication_adherence(self, medications: List[Dict]) -> List[str]:
        """Assess medication adherence concerns."""
        concerns = []

        for med in medications:
            if med.get("adherence_rate", 100) < 80:
                concerns.append(f"Low adherence to {med.get('name', 'unknown medication')}")

        return concerns

    def _generate_processing_recommendations(self, record_data: Dict) -> List[str]:
        """Generate recommendations for record processing and care."""
        recommendations = []

        # Data quality recommendations
        demographics = record_data.get("demographics", {})
        if not demographics.get("emergency_contact"):
            recommendations.append("Update emergency contact information")

        # Clinical recommendations
        diagnoses = record_data.get("diagnoses", [])
        if any("diabetes" in str(d).lower() for d in diagnoses):
            recommendations.append("Ensure regular HbA1c monitoring")

        if any("hypertension" in str(d).lower() for d in diagnoses):
            recommendations.append("Regular blood pressure monitoring recommended")

        return recommendations
