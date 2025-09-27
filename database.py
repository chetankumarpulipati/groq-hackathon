# Database configuration and models for Healthcare AI System

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./healthcare_system.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    medical_history = Column(Text)  # JSON string
    medications = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class VoiceAnalysis(Base):
    __tablename__ = "voice_analyses"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    transcription = Column(Text)
    sentiment = Column(String)
    sentiment_confidence = Column(Float)
    stress_level = Column(String)
    stress_confidence = Column(Float)
    medical_indicators = Column(Text)  # JSON string
    recommendations = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class VisionAnalysis(Base):
    __tablename__ = "vision_analyses"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    filename = Column(String)
    diagnosis = Column(String)
    confidence = Column(Float)
    findings = Column(Text)  # JSON string
    measurements = Column(Text)  # JSON string
    recommendations = Column(Text)
    model_used = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class DiagnosticTest(Base):
    __tablename__ = "diagnostic_tests"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    test_type = Column(String)
    results = Column(Text)  # JSON string
    confidence = Column(Float)
    diagnosis = Column(String)
    recommendations = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class AccuracyMetrics(Base):
    __tablename__ = "accuracy_metrics"

    id = Column(Integer, primary_key=True, index=True)
    overall_accuracy = Column(Float)
    total_cases = Column(Integer)
    correct_predictions = Column(Integer)
    category_accuracy = Column(Text)  # JSON string
    evaluation_timestamp = Column(DateTime, default=datetime.utcnow)

# Create all tables
def create_tables():
    Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions
def store_patient(db, patient_data):
    """Store patient information in database"""
    patient = Patient(
        patient_id=patient_data.get("patient_id"),
        name=patient_data.get("name", "Unknown"),
        age=patient_data.get("age"),
        gender=patient_data.get("gender"),
        medical_history=json.dumps(patient_data.get("medical_history", [])),
        medications=json.dumps(patient_data.get("medications", []))
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient

def store_voice_analysis(db, patient_id, analysis_data):
    """Store voice analysis results in database"""
    voice_analysis = VoiceAnalysis(
        patient_id=patient_id,
        transcription=analysis_data.get("transcription"),
        sentiment=analysis_data.get("sentiment"),
        sentiment_confidence=analysis_data.get("sentiment_confidence"),
        stress_level=analysis_data.get("stress_level"),
        stress_confidence=analysis_data.get("stress_confidence"),
        medical_indicators=json.dumps(analysis_data.get("medical_indicators", [])),
        recommendations=analysis_data.get("recommendations")
    )
    db.add(voice_analysis)
    db.commit()
    db.refresh(voice_analysis)
    return voice_analysis

def store_vision_analysis(db, patient_id, filename, analysis_data):
    """Store vision analysis results in database"""
    vision_analysis = VisionAnalysis(
        patient_id=patient_id,
        filename=filename,
        diagnosis=analysis_data.get("diagnosis"),
        confidence=analysis_data.get("confidence"),
        findings=json.dumps(analysis_data.get("findings", [])),
        measurements=json.dumps(analysis_data.get("measurements", {})),
        recommendations=analysis_data.get("recommendations"),
        model_used=analysis_data.get("model_used")
    )
    db.add(vision_analysis)
    db.commit()
    db.refresh(vision_analysis)
    return vision_analysis

def store_diagnostic_test(db, patient_id, test_data):
    """Store diagnostic test results in database"""
    diagnostic_test = DiagnosticTest(
        patient_id=patient_id,
        test_type=test_data.get("test_type"),
        results=json.dumps(test_data.get("results", {})),
        confidence=test_data.get("confidence"),
        diagnosis=test_data.get("diagnosis"),
        recommendations=test_data.get("recommendations")
    )
    db.add(diagnostic_test)
    db.commit()
    db.refresh(diagnostic_test)
    return diagnostic_test

def get_patient_history(db, patient_id):
    """Get complete patient history from database"""
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    voice_analyses = db.query(VoiceAnalysis).filter(VoiceAnalysis.patient_id == patient_id).all()
    vision_analyses = db.query(VisionAnalysis).filter(VisionAnalysis.patient_id == patient_id).all()
    diagnostic_tests = db.query(DiagnosticTest).filter(DiagnosticTest.patient_id == patient_id).all()

    return {
        "patient": patient,
        "voice_analyses": voice_analyses,
        "vision_analyses": vision_analyses,
        "diagnostic_tests": diagnostic_tests
    }
