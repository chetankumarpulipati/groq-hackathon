# API Documentation

## Multi-Agent Healthcare System API

This comprehensive API documentation covers all endpoints and capabilities of the Multi-Agent Healthcare System.

## Base URL
```
http://localhost:8000
```

## Authentication

Currently, the system uses API key authentication. In production, implement OAuth2 or JWT tokens.

```bash
# Example request with authentication (when implemented)
curl -H "Authorization: Bearer your_api_token" http://localhost:8000/healthcare/process
```

## Core Endpoints

### 1. System Health Check

**GET** `/health`

Check system health and status of all components.

**Response:**
```json
{
  "overall_health": "healthy",
  "detailed_status": {
    "system_status": "ready",
    "agents": {
      "voice": {"status": "ready", "tasks_completed": 0},
      "vision": {"status": "ready", "tasks_completed": 0},
      "validation": {"status": "ready", "tasks_completed": 0},
      "diagnostic": {"status": "ready", "tasks_completed": 0},
      "notification": {"status": "ready", "tasks_completed": 0}
    },
    "database": {"overall": "healthy"},
    "api_gateway": {"overall": "healthy"}
  }
}
```

### 2. Process Healthcare Request

**POST** `/healthcare/process`

Main endpoint for comprehensive healthcare processing using multi-agent workflows.

**Request Body:**
```json
{
  "patient_data": {
    "patient_id": "P123456",
    "demographics": {
      "first_name": "John",
      "last_name": "Doe",
      "age": 45,
      "gender": "male",
      "date_of_birth": "1980-01-01"
    },
    "symptoms": [
      {
        "name": "chest_pain",
        "severity": "severe",
        "duration": "2_hours",
        "description": "Sharp chest pain, radiating to left arm"
      }
    ],
    "vital_signs": {
      "systolic_bp": 150,
      "diastolic_bp": 95,
      "heart_rate": 110,
      "temperature": 98.6,
      "oxygen_saturation": 95.0
    },
    "medical_history": {
      "conditions": ["hypertension", "diabetes"],
      "medications": ["lisinopril", "metformin"],
      "allergies": ["penicillin"]
    }
  },
  "workflow_type": "comprehensive_diagnosis",
  "priority": "urgent",
  "notification_recipients": ["doctor@hospital.com"],
  "voice_data": "Patient reports severe chest pain",
  "medical_images": ["/path/to/chest_xray.jpg"]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Healthcare request processed successfully",
  "result": {
    "session_id": "session_20250927_143022",
    "status": "success",
    "workflow_type": "comprehensive_diagnosis",
    "results": {
      "agent_results": {
        "diagnostic": {
          "differential_diagnosis": "...",
          "risk_assessment": "...",
          "triage_level": "urgent"
        }
      }
    },
    "duration": 15.6,
    "timestamp": "2025-09-27T14:30:22Z"
  }
}
```

## Agent-Specific Endpoints

### 3. Voice Processing

**POST** `/agents/voice/process`

Process voice commands and audio input.

**Request Body:**
```json
{
  "voice_input": "I need to schedule an appointment with cardiology",
  "patient_id": "P123456",
  "context": {
    "urgency": "routine",
    "preferred_date": "next_week"
  }
}
```

### 4. Vision Analysis

**POST** `/agents/vision/analyze`

Analyze medical images.

**Request Body:**
```json
{
  "image_path": "/path/to/medical_image.jpg",
  "image_type": "xray",
  "patient_info": {
    "patient_id": "P123456",
    "age": 45,
    "clinical_indication": "chest_pain"
  },
  "clinical_context": "Patient presents with acute chest pain"
}
```

### 5. Data Validation

**POST** `/agents/validation/validate`

Validate healthcare data for integrity and compliance.

**Request Body:**
```json
{
  "data_type": "patient_demographics",
  "data": {
    "patient_id": "P123456",
    "first_name": "John",
    "last_name": "Doe",
    "date_of_birth": "1980-01-01",
    "email": "john.doe@email.com"
  },
  "validation_level": "standard"
}
```

### 6. Send Notifications

**POST** `/agents/notification/send`

Send healthcare notifications and alerts.

**Request Body:**
```json
{
  "notification_type": "emergency_alert",
  "recipients": ["emergency@hospital.com"],
  "message_data": {
    "patient_name": "John Doe",
    "alert_type": "Cardiac Emergency",
    "severity": "Critical"
  },
  "priority": "critical"
}
```

## System Information Endpoints

### 7. Agent Capabilities

**GET** `/agents/capabilities`

Get detailed information about agent capabilities.

### 8. Available Models

**GET** `/system/models`

Get information about available AI models and providers.

### 9. System Statistics

**GET** `/system/stats`

Get system usage statistics and performance metrics.

## Emergency Endpoints

### 10. Emergency Alert

**POST** `/emergency/alert`

Send immediate emergency alerts.

**Query Parameters:**
- `patient_id` (required): Patient identifier
- `alert_type` (required): Type of emergency
- `severity` (optional): Emergency severity level
- `clinical_summary` (optional): Clinical summary

## Workflow Types

The system supports several predefined workflows:

1. **comprehensive_diagnosis** - Complete diagnostic workflow with multi-modal analysis
2. **emergency_triage** - Rapid emergency assessment and triage
3. **routine_checkup** - Standard health assessment workflow
4. **medication_review** - Medication analysis and interaction checking

## Error Handling

All endpoints return standardized error responses:

```json
{
  "success": false,
  "error": "Error description",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-09-27T14:30:22Z"
}
```

## Rate Limiting

- Standard endpoints: 100 requests per minute
- Emergency endpoints: 1000 requests per minute
- System endpoints: 10 requests per minute

## Data Privacy and Security

- All patient data is encrypted in transit and at rest
- HIPAA compliant data handling
- PHI detection and protection
- Audit logging for all operations

## SDK and Integration Examples

### Python SDK Example

```python
import requests

# Initialize client
client = HealthcareSystemClient(base_url="http://localhost:8000")

# Process healthcare request
result = client.process_healthcare_request({
    "patient_data": {...},
    "workflow_type": "comprehensive_diagnosis"
})

print(f"Diagnosis completed: {result['session_id']}")
```

### JavaScript/Node.js Example

```javascript
const client = new HealthcareSystemClient('http://localhost:8000');

const result = await client.processHealthcareRequest({
  patient_data: {...},
  workflow_type: 'comprehensive_diagnosis'
});

console.log('Diagnosis completed:', result.session_id);
```

## WebSocket Support (Future)

Real-time updates and streaming will be available via WebSocket connections:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Real-time update:', update);
};
```

## Monitoring and Observability

- Health check endpoints for monitoring
- Prometheus metrics at `/metrics`
- Structured logging with correlation IDs
- Performance tracking and alerting
