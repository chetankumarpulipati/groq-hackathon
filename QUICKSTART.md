🏥 Multi-Agent Healthcare System - Quick Start Guide
==================================================

## 🚀 How to Run This Project (3 Simple Ways)

### Method 1: Quick Setup (Recommended for First-Time Users)
```bash
python setup.py
```
This will:
- Install all dependencies automatically
- Check your PostgreSQL connection
- Validate your API keys
- Start the system immediately

### Method 2: Manual Setup
```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Start the system
python main.py
```

### Method 3: Using Deploy Script (Linux/Mac/WSL)
```bash
# Make executable (first time only)
chmod +x deploy.sh

# Setup and start
./deploy.sh setup
./deploy.sh start
```

## ✅ What You Need (Already Have)
- ✅ Python 3.8+ (you have this)
- ✅ PostgreSQL database (you have this)
- ✅ API keys configured in .env (already set up)

## 🎯 What Happens When You Run It

1. **System starts on**: http://localhost:8000
2. **API Documentation**: http://localhost:8000/docs
3. **Health Check**: http://localhost:8000/health
4. **All 5 agents** will be ready (Voice, Vision, Validation, Diagnostic, Notification)
5. **8 AI models** available for maximum accuracy

## 🧪 Test the System

Once running, you can test it:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test voice processing
curl -X POST http://localhost:8000/agents/voice/process \
  -H "Content-Type: application/json" \
  -d '{"voice_input": "I have a headache and fever"}'

# Test complete healthcare workflow
curl -X POST http://localhost:8000/healthcare/process \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "patient_id": "TEST001",
      "demographics": {"age": 30, "gender": "female"},
      "symptoms": [{"name": "headache", "severity": "moderate"}]
    },
    "workflow_type": "comprehensive_diagnosis"
  }'
```

## 🎪 Run Demo (See All Features)
```bash
python demo.py
```
This shows all capabilities with sample data.

## 🛑 Stop the System
Press `Ctrl+C` in the terminal, or:
```bash
./deploy.sh stop
```

## ⚡ System Features Ready to Use

### 1. Voice Commands
- "Schedule appointment with cardiology"
- "I'm having chest pain"
- "Refill my medication"

### 2. Medical Image Analysis
- X-rays, CT scans, MRI images
- DICOM file support
- Clinical report generation

### 3. Data Validation
- HIPAA compliance checking
- Medical data integrity
- Critical value detection

### 4. AI-Powered Diagnosis
- Multi-modal data fusion
- 8 different AI models
- Emergency triage detection

### 5. Smart Notifications
- Email alerts for emergencies
- Appointment reminders
- Medication alerts

## 🔧 Troubleshooting

**Port 8000 already in use?**
```bash
# Find what's using port 8000
lsof -i :8000
# Kill the process
kill -9 <PID>
```

**Database connection issues?**
- Make sure PostgreSQL is running
- Check your DATABASE_URL in .env
- Ensure database user has proper permissions

**Import errors?**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## 📊 Your System Specs
- **Database**: PostgreSQL with JSONB support
- **Cache**: None (Redis disabled, system works perfectly without it)
- **AI Models**: 8 providers with intelligent selection
- **Performance**: Direct Python execution (no Docker overhead)

Ready to start? Run: `python setup.py`
