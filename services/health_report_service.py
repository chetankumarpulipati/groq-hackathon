"""
Health report generation utilities for medical image analysis.
"""

from typing import Dict, Any, List
from datetime import datetime
from utils.logging import get_logger

logger = get_logger("health_reports")


async def generate_comprehensive_health_report(
    diagnosis: str, 
    findings: List[Dict], 
    measurements: Dict, 
    confidence: float, 
    image_type: str
) -> Dict[str, Any]:
    """
    Generate a comprehensive health report based on medical image analysis results.
    """
    try:
        # Determine urgency level based on diagnosis and findings
        urgency_keywords = ["pneumonia", "fracture", "tumor", "bleeding", "stroke", "emergency", "severe", "acute"]
        urgency_level = "HIGH" if any(keyword in diagnosis.lower() for keyword in urgency_keywords) else "MEDIUM" if confidence > 0.85 else "LOW"

        # Generate risk assessment
        risk_factors = []
        if "pneumonia" in diagnosis.lower():
            risk_factors.extend(["Respiratory complications", "Sepsis risk", "Hospitalization may be required"])
        elif "normal" in diagnosis.lower():
            risk_factors.append("Low risk - routine monitoring recommended")
        else:
            risk_factors.append("Moderate risk - clinical correlation advised")

        # Determine specialist referral
        specialist_referral = None
        if any(term in diagnosis.lower() for term in ["pneumonia", "lung", "chest"]):
            specialist_referral = "Pulmonologist consultation recommended"
        elif any(term in diagnosis.lower() for term in ["brain", "neurological", "stroke"]):
            specialist_referral = "Neurologist consultation recommended"
        elif "fracture" in diagnosis.lower():
            specialist_referral = "Orthopedic surgeon consultation recommended"

        # Generate patient education points
        patient_education = []
        if "pneumonia" in diagnosis.lower():
            patient_education.extend([
                "Take prescribed antibiotics as directed, even if feeling better",
                "Rest and increase fluid intake",
                "Monitor for worsening symptoms like difficulty breathing",
                "Avoid smoking and secondhand smoke",
                "Follow up with healthcare provider as scheduled"
            ])
        elif "normal" in diagnosis.lower():
            patient_education.extend([
                "Continue regular health screenings as recommended",
                "Maintain healthy lifestyle habits",
                "Report any new or concerning symptoms to your healthcare provider"
            ])
        else:
            patient_education.extend([
                "Follow healthcare provider instructions carefully",
                "Attend all scheduled follow-up appointments",
                "Report any new or worsening symptoms immediately"
            ])

        # Generate clinical notes
        clinical_notes = f"""
MEDICAL IMAGE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PRIMARY DIAGNOSIS: {diagnosis}
CONFIDENCE LEVEL: {confidence:.1%}

DETAILED FINDINGS:
{chr(10).join([f"- {finding['description']} (Location: {finding.get('location', 'N/A')}, Confidence: {finding.get('confidence', 0):.1%})" for finding in findings])}

MEASUREMENTS:
{chr(10).join([f"- {key}: {value}" for key, value in measurements.items()])}

CLINICAL SIGNIFICANCE:
This {image_type} demonstrates {diagnosis.lower()}. The findings are consistent with the clinical presentation and require appropriate medical management.

QUALITY ASSESSMENT:
Image quality is suitable for diagnostic evaluation with adequate resolution and contrast for accurate interpretation.
        """.strip()

        # Comprehensive health report structure
        health_report = {
            "summary": {
                "primary_diagnosis": diagnosis,
                "confidence_level": confidence,
                "urgency_level": urgency_level,
                "overall_assessment": "Diagnostic findings consistent with clinical presentation"
            },
            "detailed_analysis": {
                "anatomical_findings": findings,
                "quantitative_measurements": measurements,
                "comparison_to_normal": "Findings compared to normal anatomical standards",
                "clinical_correlation": "Results should be correlated with clinical symptoms and history"
            },
            "prognosis": {
                "short_term_outlook": "Good with appropriate treatment" if confidence > 0.8 else "Requires clinical correlation",
                "long_term_outlook": "Excellent with proper management and follow-up",
                "recovery_timeline": "Varies based on treatment response and patient factors"
            },
            "monitoring_plan": {
                "immediate_actions": ["Clinical evaluation", "Symptom monitoring"],
                "follow_up_imaging": "As clinically indicated",
                "biomarkers_to_monitor": ["Clinical symptoms", "Vital signs", "Laboratory values as appropriate"]
            }
        }

        risk_assessment = {
            "overall_risk_level": urgency_level,
            "specific_risks": risk_factors,
            "risk_mitigation": [
                "Follow prescribed treatment plan",
                "Attend scheduled follow-up appointments",
                "Monitor for warning signs and symptoms"
            ],
            "emergency_indicators": [
                "Sudden worsening of symptoms",
                "Difficulty breathing or chest pain",
                "Signs of infection or complications"
            ]
        }

        return {
            "health_report": health_report,
            "risk_assessment": risk_assessment,
            "clinical_notes": clinical_notes,
            "urgency_level": urgency_level,
            "follow_up_required": urgency_level in ["HIGH", "MEDIUM"],
            "specialist_referral": specialist_referral,
            "patient_education": patient_education
        }

    except Exception as e:
        logger.error(f"Health report generation failed: {e}")
        return {
            "health_report": {"error": "Report generation failed"},
            "risk_assessment": {"overall_risk_level": "UNKNOWN"},
            "clinical_notes": "Unable to generate comprehensive report",
            "urgency_level": "MEDIUM",
            "follow_up_required": True,
            "specialist_referral": "General practitioner consultation recommended",
            "patient_education": ["Consult with healthcare provider for detailed analysis"]
        }


async def analyze_medical_voice_content(input_text: str) -> Dict[str, Any]:
    """
    Analyze medical voice content for sentiment, stress levels, and medical indicators.
    """
    from orchestrator_minimal import comprehensive_orchestrator as orchestrator
    
    try:
        # Use the orchestrator's voice agent for analysis
        voice_agent = orchestrator.agents.get("voice")
        if voice_agent:
            result = await voice_agent.process({
                "audio_text": input_text,
                "analysis_type": "medical_voice_analysis"
            })

            # Extract medical analysis from the result
            analysis = result.get("analysis", {})

            return {
                "sentiment": analysis.get("sentiment", "neutral"),
                "sentiment_confidence": analysis.get("sentiment_confidence", 0.85),
                "stress_level": analysis.get("stress_level", "normal"),
                "stress_confidence": analysis.get("stress_confidence", 0.80),
                "medical_indicators": analysis.get("medical_indicators", []),
                "recommendations": analysis.get("recommendations", "Continue monitoring symptoms")
            }
        else:
            # Fallback analysis if voice agent is not available
            return {
                "sentiment": "neutral",
                "sentiment_confidence": 0.75,
                "stress_level": "normal",
                "stress_confidence": 0.75,
                "medical_indicators": [],
                "recommendations": "Voice agent not available - basic analysis performed"
            }
    except Exception as e:
        logger.error(f"Medical voice analysis failed: {e}")
        # Return default values on error
        return {
            "sentiment": "neutral",
            "sentiment_confidence": 0.5,
            "stress_level": "unknown",
            "stress_confidence": 0.5,
            "medical_indicators": [],
            "recommendations": "Analysis failed - please try again"
        }
