"""
Medical Benchmark Dataset Generator
Creates standardized test cases for healthcare AI evaluation
"""

import json
import random
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

class MedicalBenchmarkGenerator:
    """Generate comprehensive medical test datasets for accuracy evaluation."""

    def __init__(self):
        self.benchmark_data = {}

    def generate_symptom_diagnosis_benchmark(self) -> List[Dict[str, Any]]:
        """Generate symptom-to-diagnosis benchmark cases."""
        cases = [
            # Cardiovascular
            {
                "symptoms": ["chest pain", "shortness of breath", "sweating", "nausea"],
                "patient_age": 55,
                "patient_gender": "male",
                "correct_diagnosis": "myocardial_infarction",
                "urgency": "emergency",
                "accuracy_target": 0.95
            },
            {
                "symptoms": ["chest pain", "pain radiates to arm", "jaw pain"],
                "patient_age": 48,
                "patient_gender": "female",
                "correct_diagnosis": "angina_pectoris",
                "urgency": "urgent",
                "accuracy_target": 0.92
            },

            # Neurological
            {
                "symptoms": ["sudden severe headache", "vision changes", "weakness on one side"],
                "patient_age": 65,
                "patient_gender": "male",
                "correct_diagnosis": "stroke",
                "urgency": "emergency",
                "accuracy_target": 0.96
            },
            {
                "symptoms": ["memory loss", "confusion", "difficulty speaking"],
                "patient_age": 72,
                "patient_gender": "female",
                "correct_diagnosis": "dementia",
                "urgency": "routine",
                "accuracy_target": 0.88
            },

            # Respiratory
            {
                "symptoms": ["persistent cough", "fever", "difficulty breathing", "chest tightness"],
                "patient_age": 42,
                "patient_gender": "female",
                "correct_diagnosis": "pneumonia",
                "urgency": "urgent",
                "accuracy_target": 0.91
            },
            {
                "symptoms": ["wheezing", "shortness of breath", "chest tightness"],
                "patient_age": 28,
                "patient_gender": "male",
                "correct_diagnosis": "asthma_exacerbation",
                "urgency": "urgent",
                "accuracy_target": 0.89
            },

            # Gastrointestinal
            {
                "symptoms": ["severe abdominal pain", "nausea", "vomiting", "fever"],
                "patient_age": 35,
                "patient_gender": "female",
                "correct_diagnosis": "appendicitis",
                "urgency": "emergency",
                "accuracy_target": 0.93
            },
            {
                "symptoms": ["stomach pain", "bloating", "changes in bowel movements"],
                "patient_age": 45,
                "patient_gender": "male",
                "correct_diagnosis": "irritable_bowel_syndrome",
                "urgency": "routine",
                "accuracy_target": 0.85
            },

            # Infectious Diseases
            {
                "symptoms": ["high fever", "headache", "neck stiffness", "sensitivity to light"],
                "patient_age": 22,
                "patient_gender": "male",
                "correct_diagnosis": "meningitis",
                "urgency": "emergency",
                "accuracy_target": 0.97
            },
            {
                "symptoms": ["fever", "fatigue", "muscle aches", "cough"],
                "patient_age": 30,
                "patient_gender": "female",
                "correct_diagnosis": "influenza",
                "urgency": "routine",
                "accuracy_target": 0.87
            }
        ]

        return cases

    def generate_drug_interaction_benchmark(self) -> List[Dict[str, Any]]:
        """Generate drug interaction test cases."""
        return [
            {
                "drug_1": "warfarin",
                "drug_2": "aspirin",
                "interaction_type": "bleeding_risk",
                "severity": "high",
                "recommendation": "monitor_closely",
                "accuracy_target": 0.94
            },
            {
                "drug_1": "simvastatin",
                "drug_2": "grapefruit_juice",
                "interaction_type": "increased_toxicity",
                "severity": "moderate",
                "recommendation": "avoid_combination",
                "accuracy_target": 0.91
            },
            {
                "drug_1": "metformin",
                "drug_2": "contrast_dye",
                "interaction_type": "kidney_damage",
                "severity": "high",
                "recommendation": "stop_before_procedure",
                "accuracy_target": 0.96
            },
            {
                "drug_1": "digoxin",
                "drug_2": "furosemide",
                "interaction_type": "electrolyte_imbalance",
                "severity": "moderate",
                "recommendation": "monitor_levels",
                "accuracy_target": 0.89
            }
        ]

    def generate_treatment_recommendation_benchmark(self) -> List[Dict[str, Any]]:
        """Generate treatment recommendation test cases."""
        return [
            {
                "condition": "type_2_diabetes",
                "patient_factors": {"age": 45, "bmi": 32, "hba1c": 8.5},
                "correct_treatment": "metformin_plus_lifestyle",
                "contraindications": [],
                "accuracy_target": 0.92
            },
            {
                "condition": "hypertension",
                "patient_factors": {"age": 65, "kidney_disease": True, "race": "african_american"},
                "correct_treatment": "calcium_channel_blocker",
                "contraindications": ["ace_inhibitor"],
                "accuracy_target": 0.90
            },
            {
                "condition": "pneumonia",
                "patient_factors": {"age": 75, "copd": True, "hospitalized": True},
                "correct_treatment": "broad_spectrum_antibiotics",
                "contraindications": [],
                "accuracy_target": 0.94
            }
        ]

    def create_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Create complete benchmark dataset."""
        benchmark = {
            "metadata": {
                "created_date": datetime.now().isoformat(),
                "version": "1.0",
                "total_cases": 0
            },
            "symptom_diagnosis": self.generate_symptom_diagnosis_benchmark(),
            "drug_interactions": self.generate_drug_interaction_benchmark(),
            "treatment_recommendations": self.generate_treatment_recommendation_benchmark()
        }

        # Calculate total cases
        total = (len(benchmark["symptom_diagnosis"]) +
                len(benchmark["drug_interactions"]) +
                len(benchmark["treatment_recommendations"]))
        benchmark["metadata"]["total_cases"] = total

        return benchmark

    def save_benchmark(self, filepath: str) -> str:
        """Save benchmark to file."""
        benchmark = self.create_comprehensive_benchmark()

        with open(filepath, 'w') as f:
            json.dump(benchmark, f, indent=2)

        return filepath


# Generate the benchmark dataset
if __name__ == "__main__":
    generator = MedicalBenchmarkGenerator()
    benchmark_path = "evaluation/medical_benchmark_dataset.json"
    generator.save_benchmark(benchmark_path)
    print(f"Medical benchmark dataset created: {benchmark_path}")
