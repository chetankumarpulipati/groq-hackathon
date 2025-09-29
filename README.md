## Screenshots

Below are screenshots demonstrating various features and interfaces of the system

![Figure 1](screenshots/Figure_1.png)
*Figure 1*

![Figure 2](screenshots/Figure_2.png)
*Figure 2*

![Figure 3](screenshots/Figure_3.png)
*Figure 3*

![Figure 4](screenshots/Figure_4.png)
*Figure 4*

![Figure 5](screenshots/Figure_5.png)
*Figure 5*

![Figure 6](screenshots/Figure_6.png)
*Figure 6*

![Figure 7](screenshots/Figure_7.png)
*Figure 7*

![Figure 8](screenshots/Figure_8.png)
*Figure 8*

![Figure 9](screenshots/Figure_9.png)
*Figure 9*

![Figure 10](screenshots/Figure_10.png)
*Figure 10*

![Figure 11](screenshots/Figure_11.png)
*Figure 11*

![Figure 12](screenshots/Figure_12.png)
*Figure 12*

![Figure 13](screenshots/Figure_13.png)
*Figure 13*

# Multi-Agent Healthcare System with Groq Integration

## Architecture Overview

This project implements a comprehensive multi-agent healthcare system that leverages Groq's accelerated inference capabilities for processing multi-modal medical data including voice commands, medical images, and patient records.

## System Components

### Agent Modules
- **Voice Input Agent**: Processes voice commands using ASR
- **Vision Processing Agent**: Analyzes medical images using AI models
- **Data Validation Agent**: Validates healthcare data against medical rules
- **Diagnostic Agent**: Fuses multi-modal data for diagnosis generation
- **Notification Agent**: Manages alerts and notifications

### MCP Integration Layer
- Database connectors for patient records
- API Gateway interfaces
- File system access for medical documents
- Notification service integration

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=your_database_url
NOTIFICATION_SERVICE_URL=your_notification_url
```

3. Run the system:
```bash
python main.py
```

## Usage Examples

### Voice Command Processing
```python
from agents.voice_agent import VoiceAgent
agent = VoiceAgent()
result = agent.process_voice_command("schedule appointment with Dr. Smith")
```

### Medical Image Analysis
```python
from agents.vision_agent import VisionAgent
agent = VisionAgent()
diagnosis = agent.analyze_medical_image("path/to/xray.jpg")
```

## Testing

Run unit tests:
```bash
python -m pytest tests/
```

Run integration tests:
```bash
python -m pytest tests/integration/
```

## API Documentation

See `docs/api.md` for detailed API documentation.
