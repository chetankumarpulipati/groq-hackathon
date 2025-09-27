# 🏥 Multi-Agent Healthcare System API Usage Guide

Your healthcare API is now running at **http://localhost:8000** with comprehensive medical AI capabilities!

## 🚀 Quick Start

### 1. Check System Health
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "service": "Multi-Agent Healthcare System", 
  "version": "2.0.0",
  "model_accuracy": "18.2% accuracy",
  "agents": ["voice", "diagnostic", "validation"],
  "endpoints": {
    "accuracy_metrics": "/accuracy/metrics",
    "accuracy_report": "/accuracy/report"
  }
}
```

---

## 📋 Core Healthcare Endpoints

### 2. Process Healthcare Request
**Endpoint:** `POST /healthcare/process`

**Use Case:** Submit patient data for comprehensive medical analysis

```bash
curl -X POST http://localhost:8000/healthcare/process \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "symptoms": ["chest pain", "shortness of breath", "sweating"],
      "age": 55,
      "gender": "male",
      "medical_history": ["hypertension"]
    },
    "workflow_type": "simple",
    "priority": "urgent"
  }'
```

**Response Example:**
```json
{
  "success": true,
  "message": "Healthcare request processed successfully",
  "result": {
    "analysis": "Potential myocardial infarction - requires immediate medical attention",
    "confidence": 0.94,
    "recommendations": ["Emergency room evaluation", "ECG", "Cardiac enzymes"],
    "urgency": "emergency"
  }
}
```

### 3. Voice Processing
**Endpoint:** `POST /agents/voice/process`

**Use Case:** Process voice commands or spoken medical queries

```bash
curl -X POST http://localhost:8000/agents/voice/process \
  -H "Content-Type: application/json" \
  -d '{
    "voice_input": "Patient complains of severe headache and vision changes",
    "patient_id": "P12345",
    "context": {
      "environment": "emergency_department"
    }
  }'
```

**Response Example:**
```json
{
  "success": true,
  "message": "Voice input processed successfully", 
  "result": {
    "interpretation": "Neurological emergency symptoms detected",
    "suggested_diagnosis": "Possible stroke or intracranial pressure",
    "confidence": 0.89,
    "next_steps": ["Immediate CT scan", "Neurological assessment"]
  }
}
```

---

## 📊 Model Accuracy & Evaluation Endpoints

### 4. Get Current Accuracy Metrics
**Endpoint:** `GET /accuracy/metrics`

```bash
curl http://localhost:8000/accuracy/metrics
```

**Response:**
```json
{
  "overall_accuracy": 0.182,
  "category_accuracy": {
    "symptom_analysis": 0.0,
    "drug_interaction": 0.5,
    "complex_diagnosis": 0.5
  },
  "total_cases": 11,
  "correct_predictions": 2,
  "confidence_scores": [0.95, 0.92, 0.85],
  "evaluation_timestamp": "2025-09-27T13:05:01"
}
```

### 5. Run Full Accuracy Evaluation
**Endpoint:** `POST /accuracy/evaluate`

```bash
curl -X POST http://localhost:8000/accuracy/evaluate
```

**Response:**
```json
{
  "message": "Accuracy evaluation started",
  "status": "running", 
  "estimated_duration": "2-3 minutes"
}
```

### 6. View Accuracy Report (HTML)
**Endpoint:** `GET /accuracy/report`

Open in browser: **http://localhost:8000/accuracy/report**

This provides a beautiful HTML dashboard showing:
- Overall accuracy percentage
- Category breakdowns
- Performance metrics
- Visual charts

### 7. Get Accuracy Visualizations
**Endpoint:** `GET /accuracy/visualizations`

```bash
curl http://localhost:8000/accuracy/visualizations --output accuracy_chart.png
```

Returns PNG charts showing model performance across categories.

### 8. Generate Medical Benchmark Dataset
**Endpoint:** `GET /accuracy/benchmark`

```bash
curl http://localhost:8000/accuracy/benchmark
```

**Response:**
```json
{
  "message": "Medical benchmark dataset generated",
  "benchmark_path": "evaluation/medical_benchmark_dataset.json",
  "total_cases": 17,
  "categories": ["symptom_diagnosis", "drug_interactions", "treatment_recommendations"],
  "sample_case": {
    "symptoms": ["chest pain", "shortness of breath"],
    "correct_diagnosis": "myocardial_infarction"
  }
}
```

### 9. Test Custom Medical Case
**Endpoint:** `POST /accuracy/test-custom`

**Test a specific medical scenario:**

```bash
curl -X POST http://localhost:8000/accuracy/test-custom \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Patient with diabetes taking metformin, now prescribed ciprofloxacin",
    "ground_truth": "drug_interaction_warning",
    "expected_keywords": ["interaction", "blood sugar", "monitor"],
    "confidence_threshold": 0.8
  }'
```

**Response:**
```json
{
  "input_case": {
    "input": "Patient with diabetes taking metformin, now prescribed ciprofloxacin"
  },
  "model_prediction": {
    "diagnosis": "Potential drug interaction between metformin and ciprofloxacin",
    "confidence": 0.87,
    "keywords": ["interaction", "monitor", "blood sugar"],
    "action": "monitor_closely"
  },
  "evaluation": {
    "is_correct": true,
    "confidence_met": true,
    "keyword_matches": 3
  }
}
```

---

## 🎯 Practical Usage Examples

### Medical Emergency Triage
```bash
curl -X POST http://localhost:8000/healthcare/process \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "symptoms": ["unconscious", "weak pulse", "cold skin"],
      "vital_signs": {"bp": "70/40", "hr": 120},
      "age": 67
    },
    "workflow_type": "emergency",
    "priority": "critical"
  }'
```

### Drug Interaction Check
```bash
curl -X POST http://localhost:8000/agents/voice/process \
  -H "Content-Type: application/json" \
  -d '{
    "voice_input": "Check interaction between warfarin and aspirin for elderly patient",
    "context": {"medication_review": true}
  }'
```

### Symptom Analysis
```bash
curl -X POST http://localhost:8000/healthcare/process \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "symptoms": ["persistent cough", "fever", "night sweats", "weight loss"],
      "duration": "3 weeks",
      "age": 45,
      "smoking_history": true
    },
    "workflow_type": "diagnostic"
  }'
```

---

## 🔧 Request Schemas

### HealthcareRequest Schema
```json
{
  "patient_data": {
    "symptoms": ["string"],
    "age": "integer", 
    "gender": "string",
    "medical_history": ["string"],
    "medications": ["string"],
    "vital_signs": {}
  },
  "workflow_type": "simple|emergency|diagnostic", 
  "priority": "standard|urgent|critical",
  "voice_data": "string (optional)"
}
```

### VoiceProcessingRequest Schema
```json
{
  "voice_input": "string (required)",
  "patient_id": "string (optional)",
  "context": {
    "environment": "string",
    "specialty": "string"
  }
}
```

---

## 🌐 Interactive Testing

### Option 1: Swagger UI Documentation
Visit: **http://localhost:8000/docs**
- Interactive API explorer
- Test all endpoints directly
- View request/response schemas
- Try different parameters

### Option 2: ReDoc Documentation  
Visit: **http://localhost:8000/redoc**
- Clean, readable API documentation
- Detailed schema descriptions

---

## 🚀 Advanced Features

### Real-time Accuracy Monitoring
```bash
# Check current accuracy
curl http://localhost:8000/accuracy/metrics

# Run new evaluation
curl -X POST http://localhost:8000/accuracy/evaluate

# View results
curl http://localhost:8000/accuracy/report
```

### Multi-Model AI Processing
Your system automatically selects the best AI model for each task:
- **Medical Diagnosis**: Uses `openai/gpt-oss-20b` for highest accuracy
- **Drug Interactions**: Specialized model selection
- **Emergency Triage**: Prioritized processing

### Error Handling
All endpoints return structured error responses:
```json
{
  "success": false,
  "error": "Detailed error message",
  "error_code": "SPECIFIC_ERROR_CODE"
}
```

---

## 🎯 Best Practices

1. **Always include patient context** for better accuracy
2. **Use appropriate priority levels** for urgent cases  
3. **Monitor accuracy metrics** regularly
4. **Test custom cases** to validate specific scenarios
5. **Check system health** before processing critical requests

Your Multi-Agent Healthcare System is now ready for comprehensive medical AI processing! 🏥✨
