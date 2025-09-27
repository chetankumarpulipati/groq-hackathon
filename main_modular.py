"""
Main entry point for the Multi-Agent Healthcare System.
Modular version with separated components for better maintainability.
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from pathlib import Path

# Import configuration and logging
from config.settings import config
from utils.logging import get_logger, setup_logging

# Import database setup
from database import create_tables

# Import orchestrators
from orchestrator_minimal import comprehensive_orchestrator as orchestrator

# Import route modules
from routes.voice_routes import router as voice_router
from routes.vision_routes import router as vision_router
from routes.accuracy_routes import router as accuracy_router
from routes.healthcare_routes import router as healthcare_router
from routes.patient_routes import router as patient_router

# Setup logging
setup_logging()
logger = get_logger("main_app")

# Create database tables
create_tables()

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Healthcare System",
    description="AI-powered healthcare system with voice processing and diagnosis capabilities",
    version="1.0.0 (Modular)",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for accuracy dashboard
static_path = Path("static")
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include all route modules
app.include_router(voice_router)
app.include_router(vision_router)
app.include_router(accuracy_router)
app.include_router(healthcare_router)
app.include_router(patient_router)


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    try:
        logger.info("🚀 Multi-Agent Healthcare System starting up (modular mode)...")
        logger.info("✅ System ready for healthcare processing")

    except Exception as e:
        logger.error(f"System startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Multi-Agent Healthcare System shutting down...")
    await orchestrator.shutdown()


@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "system": "Multi-Agent Healthcare System",
        "version": "1.0.0 (Modular Mode)",
        "status": orchestrator.system_status,
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": "/docs",
        "health_check": "/health",
        "message": "Healthcare system is ready for voice command processing!",
        "architecture": "Modular design with separated components"
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Enhanced health check with model accuracy information."""
    try:
        from routes.accuracy_routes import evaluator

        accuracy_info = "Not evaluated"
        if evaluator.metrics:
            accuracy_info = f"{evaluator.metrics['overall_accuracy']:.1%} accuracy"

        return {
            "status": "healthy",
            "service": "Multi-Agent Healthcare System",
            "version": "2.0.0 (Modular)",
            "model_accuracy": accuracy_info,
            "agents": ["voice", "diagnostic", "validation"],
            "timestamp": datetime.now().isoformat(),
            "architecture": "Modular with separated routes and services",
            "endpoints": {
                "accuracy_metrics": "/accuracy/metrics",
                "accuracy_report": "/accuracy/report",
                "accuracy_evaluation": "/accuracy/evaluate",
                "accuracy_visualizations": "/accuracy/visualizations",
                "voice_processing": "/voice/process",
                "vision_analysis": "/vision/analyze",
                "patient_history": "/patients/{patient_id}/history"
            }
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


if __name__ == "__main__":
    # Run the FastAPI application
    print("🏥 Starting Multi-Agent Healthcare System (Modular Mode)")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 Documentation: http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    print("🔧 Architecture: Modular design with separated components")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
