"""
Data Validation Agent for ensuring healthcare data integrity and compliance.
Validates medical data against healthcare standards and regulations.
"""

import asyncio
import re
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, date
from decimal import Decimal
import pandas as pd
from agents.base_agent import BaseAgent
from config.settings import config
from utils.logging import get_logger
from utils.error_handling import DataValidationError, handle_exception

logger = get_logger("validation_agent")


class DataValidationAgent(BaseAgent):
    """
    Specialized agent for validating healthcare data integrity, compliance,
    and adherence to medical standards and regulations.
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id, specialized_task="data_validation")
        self.validation_rules = self._initialize_validation_rules()
        self.medical_codes = self._initialize_medical_codes()
        self.compliance_standards = self._initialize_compliance_standards()

        logger.info(f"DataValidationAgent {self.agent_id} initialized with healthcare validation rules")

    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive healthcare data validation rules."""
        return {
            "patient_demographics": {
                "required_fields": ["patient_id", "first_name", "last_name", "date_of_birth"],
                "field_types": {
                    "patient_id": "string",
                    "first_name": "string",
                    "last_name": "string",
                    "date_of_birth": "date",
                    "phone": "phone_number",
                    "email": "email",
                    "ssn": "ssn",
                    "insurance_id": "string"
                },
                "constraints": {
                    "patient_id": {"min_length": 1, "max_length": 50},
                    "first_name": {"min_length": 1, "max_length": 100},
                    "last_name": {"min_length": 1, "max_length": 100},
                    "phone": {"format": r"^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$"},
                    "email": {"format": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
                }
            },
            "vital_signs": {
                "required_fields": ["timestamp", "patient_id"],
                "field_types": {
                    "systolic_bp": "integer",
                    "diastolic_bp": "integer",
                    "heart_rate": "integer",
                    "temperature": "decimal",
                    "respiratory_rate": "integer",
                    "oxygen_saturation": "decimal",
                    "weight": "decimal",
                    "height": "decimal"
                },
                "ranges": {
                    "systolic_bp": {"min": 70, "max": 250, "critical_low": 90, "critical_high": 180},
                    "diastolic_bp": {"min": 40, "max": 150, "critical_low": 60, "critical_high": 110},
                    "heart_rate": {"min": 30, "max": 200, "critical_low": 50, "critical_high": 120},
                    "temperature": {"min": 95.0, "max": 110.0, "critical_low": 96.0, "critical_high": 103.0},
                    "respiratory_rate": {"min": 8, "max": 40, "critical_low": 12, "critical_high": 25},
                    "oxygen_saturation": {"min": 70.0, "max": 100.0, "critical_low": 95.0},
                    "weight": {"min": 1.0, "max": 1000.0},
                    "height": {"min": 10.0, "max": 300.0}
                }
            },
            "medications": {
                "required_fields": ["medication_name", "dosage", "frequency", "patient_id"],
                "field_types": {
                    "medication_name": "string",
                    "dosage": "string",
                    "frequency": "string",
                    "route": "string",
                    "prescriber": "string",
                    "start_date": "date",
                    "end_date": "date"
                },
                "constraints": {
                    "medication_name": {"min_length": 2, "max_length": 200},
                    "dosage": {"format": r"^\d+(\.\d+)?\s*(mg|g|ml|l|units?|mcg|iu)$"},
                    "frequency": {"allowed": ["once daily", "twice daily", "three times daily", "four times daily", "as needed", "every 4 hours", "every 6 hours", "every 8 hours", "every 12 hours"]},
                    "route": {"allowed": ["oral", "IV", "IM", "topical", "sublingual", "inhalation", "rectal", "transdermal"]}
                }
            },
            "lab_results": {
                "required_fields": ["test_name", "result_value", "reference_range", "patient_id"],
                "field_types": {
                    "test_name": "string",
                    "result_value": "string",
                    "reference_range": "string",
                    "units": "string",
                    "abnormal_flag": "string"
                },
                "constraints": {
                    "abnormal_flag": {"allowed": ["Normal", "High", "Low", "Critical High", "Critical Low", "Abnormal"]}
                }
            }
        }

    def _initialize_medical_codes(self) -> Dict[str, List[str]]:
        """Initialize medical coding systems for validation."""
        return {
            "icd10_sample": [
                "A00-B99",  # Infectious diseases
                "C00-D49",  # Neoplasms
                "E00-E89",  # Endocrine diseases
                "I00-I99",  # Circulatory system
                "J00-J99",  # Respiratory system
                "K00-K95",  # Digestive system
            ],
            "cpt_sample": [
                "99201-99499",  # Evaluation and Management
                "10021-69990",  # Surgery
                "70010-79999",  # Radiology
                "80047-89398",  # Pathology and Laboratory
            ],
            "loinc_sample": [
                "33747-0",  # Blood pressure
                "8867-4",   # Heart rate
                "8310-5",   # Body temperature
                "9279-1",   # Respiratory rate
            ]
        }

    def _initialize_compliance_standards(self) -> Dict[str, Dict[str, Any]]:
        """Initialize healthcare compliance standards."""
        return {
            "hipaa": {
                "phi_fields": [
                    "ssn", "medical_record_number", "account_number",
                    "certificate_license_number", "vehicle_identifier",
                    "device_identifier", "web_url", "ip_address",
                    "biometric_identifier", "photograph", "email"
                ],
                "minimum_necessary": True,
                "encryption_required": True
            },
            "hl7_fhir": {
                "required_resources": ["Patient", "Observation", "Condition", "MedicationRequest"],
                "data_types": ["string", "integer", "decimal", "boolean", "dateTime", "code"],
                "cardinality_rules": True
            },
            "fda_regulations": {
                "device_requirements": ["UDI", "safety_reporting", "adverse_events"],
                "drug_requirements": ["NDC", "lot_number", "expiration_date"]
            }
        }

    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process data validation request."""
        try:
            validation_result = await self._validate_healthcare_data(input_data)

            return {
                "agent_id": self.agent_id,
                "validation_result": validation_result,
                "data_valid": validation_result.get("is_valid", False),
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }

        except Exception as e:
            logger.error(f"Validation processing failed: {e}")
            return {
                "agent_id": self.agent_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _validate_healthcare_data(self, data: Any) -> Dict:
        """Validate healthcare data for compliance and integrity."""

        validation_result = {
            "is_valid": True,
            "validation_checks": [],
            "warnings": [],
            "errors": []
        }

        if isinstance(data, dict):
            # Check required fields
            if "patient_id" in data:
                validation_result["validation_checks"].append("Patient ID present")

            # Validate medical data formats
            if "age" in data and isinstance(data["age"], (int, float)):
                if 0 <= data["age"] <= 120:
                    validation_result["validation_checks"].append("Age within valid range")
                else:
                    validation_result["warnings"].append("Age outside typical range")

        return validation_result

    async def _validate_structured_data(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate structured healthcare data against predefined rules."""

        data_type = input_data["data_type"]
        data = input_data["data"]
        validation_level = input_data.get("validation_level", "standard")

        validation_results = {
            "data_type": data_type,
            "validation_level": validation_level,
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "critical_issues": [],
            "compliance_check": {},
            "data_quality_score": 0.0
        }

        # Get validation rules for data type
        rules = self.validation_rules.get(data_type, {})

        if not rules:
            # Use AI to determine validation approach for unknown data types
            ai_validation = await self._ai_assisted_validation(data, data_type, context)
            validation_results.update(ai_validation)
        else:
            # Standard rule-based validation
            validation_results = await self._apply_validation_rules(data, rules, validation_results)

        # Compliance validation
        compliance_results = await self._validate_compliance(data, data_type, context)
        validation_results["compliance_check"] = compliance_results

        # Calculate overall data quality score
        validation_results["data_quality_score"] = self._calculate_quality_score(validation_results)

        # Determine final validity
        validation_results["is_valid"] = (
            len(validation_results["errors"]) == 0 and
            len(validation_results["critical_issues"]) == 0
        )

        return validation_results

    async def _apply_validation_rules(
        self,
        data: Dict[str, Any],
        rules: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply validation rules to healthcare data."""

        # Check required fields
        required_fields = rules.get("required_fields", [])
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                validation_results["errors"].append({
                    "field": field,
                    "error": "required_field_missing",
                    "message": f"Required field '{field}' is missing or empty"
                })

        # Check field types
        field_types = rules.get("field_types", {})
        for field, expected_type in field_types.items():
            if field in data and data[field] is not None:
                validation_error = self._validate_field_type(field, data[field], expected_type)
                if validation_error:
                    validation_results["errors"].append(validation_error)

        # Check constraints
        constraints = rules.get("constraints", {})
        for field, constraint_rules in constraints.items():
            if field in data and data[field] is not None:
                constraint_errors = self._validate_field_constraints(field, data[field], constraint_rules)
                validation_results["errors"].extend(constraint_errors)

        # Check value ranges (for vital signs, lab results, etc.)
        ranges = rules.get("ranges", {})
        for field, range_rules in ranges.items():
            if field in data and data[field] is not None:
                range_validation = self._validate_field_range(field, data[field], range_rules)
                if range_validation["is_critical"]:
                    validation_results["critical_issues"].append(range_validation)
                elif not range_validation["is_valid"]:
                    validation_results["warnings"].append(range_validation)

        return validation_results

    def _validate_field_type(self, field: str, value: Any, expected_type: str) -> Optional[Dict[str, Any]]:
        """Validate field data type."""

        try:
            if expected_type == "string" and not isinstance(value, str):
                return {"field": field, "error": "type_mismatch", "message": f"Field '{field}' must be a string"}

            elif expected_type == "integer":
                int(value)

            elif expected_type == "decimal":
                float(value)

            elif expected_type == "date":
                if isinstance(value, str):
                    datetime.strptime(value, "%Y-%m-%d")
                elif not isinstance(value, (date, datetime)):
                    raise ValueError("Invalid date format")

            elif expected_type == "phone_number":
                if not re.match(r"^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$", str(value)):
                    return {"field": field, "error": "invalid_format", "message": f"Invalid phone number format: {value}"}

            elif expected_type == "email":
                if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", str(value)):
                    return {"field": field, "error": "invalid_format", "message": f"Invalid email format: {value}"}

            elif expected_type == "ssn":
                if not re.match(r"^\d{3}-?\d{2}-?\d{4}$", str(value)):
                    return {"field": field, "error": "invalid_format", "message": f"Invalid SSN format: {value}"}

            return None

        except (ValueError, TypeError) as e:
            return {"field": field, "error": "type_conversion", "message": f"Cannot convert '{value}' to {expected_type}"}

    def _validate_field_constraints(self, field: str, value: Any, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate field constraints."""

        errors = []

        # Length constraints
        if "min_length" in constraints and len(str(value)) < constraints["min_length"]:
            errors.append({
                "field": field,
                "error": "min_length_violation",
                "message": f"Field '{field}' must be at least {constraints['min_length']} characters"
            })

        if "max_length" in constraints and len(str(value)) > constraints["max_length"]:
            errors.append({
                "field": field,
                "error": "max_length_violation",
                "message": f"Field '{field}' must be at most {constraints['max_length']} characters"
            })

        # Format constraints
        if "format" in constraints and not re.match(constraints["format"], str(value)):
            errors.append({
                "field": field,
                "error": "format_violation",
                "message": f"Field '{field}' does not match required format"
            })

        # Allowed values
        if "allowed" in constraints and value not in constraints["allowed"]:
            errors.append({
                "field": field,
                "error": "invalid_value",
                "message": f"Field '{field}' has invalid value. Allowed: {constraints['allowed']}"
            })

        return errors

    def _validate_field_range(self, field: str, value: Any, range_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate numeric field ranges with critical value detection."""

        try:
            numeric_value = float(value)

            result = {
                "field": field,
                "value": numeric_value,
                "is_valid": True,
                "is_critical": False,
                "message": ""
            }

            # Check critical ranges first
            if "critical_low" in range_rules and numeric_value < range_rules["critical_low"]:
                result.update({
                    "is_valid": False,
                    "is_critical": True,
                    "message": f"CRITICAL: {field} value {numeric_value} is below critical threshold {range_rules['critical_low']}"
                })
            elif "critical_high" in range_rules and numeric_value > range_rules["critical_high"]:
                result.update({
                    "is_valid": False,
                    "is_critical": True,
                    "message": f"CRITICAL: {field} value {numeric_value} is above critical threshold {range_rules['critical_high']}"
                })

            # Check normal ranges
            elif "min" in range_rules and numeric_value < range_rules["min"]:
                result.update({
                    "is_valid": False,
                    "message": f"{field} value {numeric_value} is below minimum {range_rules['min']}"
                })
            elif "max" in range_rules and numeric_value > range_rules["max"]:
                result.update({
                    "is_valid": False,
                    "message": f"{field} value {numeric_value} is above maximum {range_rules['max']}"
                })

            return result

        except (ValueError, TypeError):
            return {
                "field": field,
                "value": value,
                "is_valid": False,
                "is_critical": False,
                "message": f"Cannot validate range for non-numeric value: {value}"
            }

    async def _ai_assisted_validation(
        self,
        data: Dict[str, Any],
        data_type: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use AI to validate unknown or complex data types."""

        system_prompt = f"""You are a healthcare data validation expert. Analyze this {data_type} data for:
        
        1. Data completeness and quality
        2. Medical accuracy and plausibility  
        3. Potential data entry errors
        4. Missing critical information
        5. Inconsistencies or anomalies
        6. Compliance with healthcare standards
        
        Return a JSON response with:
        - is_valid: boolean
        - errors: array of error objects
        - warnings: array of warning objects  
        - critical_issues: array of critical findings
        - suggestions: array of improvement suggestions
        - confidence_score: 0.0-1.0"""

        user_prompt = f"""
        Data Type: {data_type}
        Data: {json.dumps(data, indent=2)}
        Context: {context or 'None provided'}
        
        Please validate this healthcare data thoroughly."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            ai_response = await self.get_ai_response(messages, temperature=0.1)
            ai_validation = json.loads(ai_response["response"])

            return {
                "ai_validation": ai_validation,
                "model_info": ai_response["model_info"],
                "validation_method": "ai_assisted"
            }

        except json.JSONDecodeError:
            return {
                "errors": [{"error": "ai_validation_failed", "message": "AI validation response was not valid JSON"}],
                "validation_method": "ai_assisted_failed"
            }

    async def _validate_compliance(
        self,
        data: Dict[str, Any],
        data_type: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate healthcare compliance requirements."""

        compliance_results = {
            "hipaa_compliant": True,
            "phi_detected": [],
            "encryption_required": False,
            "hl7_fhir_compliant": True,
            "fda_compliant": True,
            "compliance_issues": []
        }

        # HIPAA PHI Detection
        phi_fields = self.compliance_standards["hipaa"]["phi_fields"]
        for field in data.keys():
            if field.lower() in [f.lower() for f in phi_fields]:
                compliance_results["phi_detected"].append(field)
                compliance_results["encryption_required"] = True

        # Check for PHI patterns in data values
        phi_patterns = {
            "ssn": r"\d{3}-?\d{2}-?\d{4}",
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}"
        }

        for field, value in data.items():
            if isinstance(value, str):
                for pattern_name, pattern in phi_patterns.items():
                    if re.search(pattern, value):
                        compliance_results["phi_detected"].append(f"{field}_{pattern_name}")

        # Additional compliance checks can be added here
        # HL7 FHIR validation, FDA requirements, etc.

        return compliance_results

    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""

        total_issues = (
            len(validation_results["errors"]) +
            len(validation_results["warnings"]) +
            len(validation_results["critical_issues"]) * 3  # Weight critical issues more
        )

        if total_issues == 0:
            return 1.0

        # Simple scoring algorithm - can be made more sophisticated
        base_score = 1.0
        penalty_per_issue = 0.1

        quality_score = max(0.0, base_score - (total_issues * penalty_per_issue))
        return round(quality_score, 2)

    async def validate_batch_data(self, data_list: List[Dict[str, Any]], data_type: str) -> Dict[str, Any]:
        """Validate a batch of healthcare data records."""

        batch_results = {
            "total_records": len(data_list),
            "valid_records": 0,
            "invalid_records": 0,
            "critical_records": 0,
            "batch_quality_score": 0.0,
            "individual_results": []
        }

        for i, record in enumerate(data_list):
            record_validation = await self._validate_structured_data({
                "data_type": data_type,
                "data": record
            }, None)

            record_validation["record_index"] = i
            batch_results["individual_results"].append(record_validation)

            if record_validation["is_valid"]:
                batch_results["valid_records"] += 1
            else:
                batch_results["invalid_records"] += 1

            if len(record_validation["critical_issues"]) > 0:
                batch_results["critical_records"] += 1

        # Calculate batch quality score
        if batch_results["total_records"] > 0:
            batch_results["batch_quality_score"] = batch_results["valid_records"] / batch_results["total_records"]

        return batch_results

    async def get_validation_capabilities(self) -> Dict[str, Any]:
        """Get current data validation capabilities."""

        return {
            "agent_id": self.agent_id,
            "supported_data_types": list(self.validation_rules.keys()),
            "validation_methods": ["rule_based", "ai_assisted", "batch_validation"],
            "compliance_standards": list(self.compliance_standards.keys()),
            "medical_coding_systems": list(self.medical_codes.keys()),
            "critical_value_detection": True,
            "phi_detection": True,
            "data_quality_scoring": True
        }
