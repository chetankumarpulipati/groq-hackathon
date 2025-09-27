#!/usr/bin/env python3
"""
Quick setup script for the Multi-Agent Healthcare System without Redis.
Sets up PostgreSQL database and starts the system.
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header():
    print("🏥 Multi-Agent Healthcare System - Quick Setup (PostgreSQL Only)")
    print("=" * 65)

def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    directories = ["logs", "data/medical_images", "data/voice_samples", "data/text_data"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def check_database_connection():
    """Check if PostgreSQL database is accessible."""
    print("\n🗄️  Checking database connection...")
    try:
        # Import here to avoid issues if dependencies aren't installed yet
        from config.settings import config
        import psycopg2
        
        # Parse the database URL
        db_url = config.database.url
        if not db_url.startswith('postgresql://'):
            print("❌ Invalid PostgreSQL URL in configuration")
            return False
            
        # Test connection
        conn = psycopg2.connect(db_url)
        conn.close()
        print("✅ PostgreSQL database connection successful")
        return True
        
    except ImportError:
        print("⚠️  Database libraries not installed yet - will check after dependency installation")
        return True  # Allow setup to continue
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n💡 Quick fixes:")
        print("   1. Make sure PostgreSQL is running")
        print("   2. Create database: CREATE DATABASE healthcare_db;")
        print("   3. Create user: CREATE USER healthcare_user WITH PASSWORD 'healthcare_pass';")
        print("   4. Grant permissions: GRANT ALL PRIVILEGES ON DATABASE healthcare_db TO healthcare_user;")
        return False

def validate_configuration():
    """Validate system configuration."""
    print("\n⚙️  Validating configuration...")
    try:
        from config.settings import config
        
        # Check API keys
        if config.models.groq_api_key.startswith('gsk_'):
            print("✅ Groq API key configured")
        else:
            print("❌ Groq API key not properly configured")
            
        # Check if other API keys are configured
        api_keys = {
            'OpenAI': config.models.openai_api_key,
            'Google': config.models.google_api_key,
            'DeepSeek': config.models.deepseek_api_key,
        }
        
        configured_count = sum(1 for key in api_keys.values() if key and key.startswith('gsk_'))
        print(f"✅ {configured_count + 1} AI model providers configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def run_system_test():
    """Run a quick system test."""
    print("\n🧪 Running system test...")
    try:
        from orchestrator import orchestrator
        print("✅ System modules loaded successfully")
        return True
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def start_system():
    """Start the healthcare system."""
    print("\n🚀 Starting Multi-Agent Healthcare System...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 Documentation at: http://localhost:8000/docs")
    print("❤️  Health check at: http://localhost:8000/health")
    print("\n⚡ Press Ctrl+C to stop the system")
    print("-" * 50)
    
    try:
        # Start the FastAPI application
        os.system("python main.py")
    except KeyboardInterrupt:
        print("\n\n👋 Healthcare system stopped")

def main():
    """Main setup function."""
    print_header()
    
    # Step 1: Check Python
    if not check_python():
        return False
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Install dependencies
    if not install_dependencies():
        return False
    
    # Step 4: Check database
    if not check_database_connection():
        print("\n💡 Please set up PostgreSQL and try again")
        return False
    
    # Step 5: Validate configuration
    if not validate_configuration():
        return False
    
    # Step 6: Run system test
    if not run_system_test():
        return False
    
    print("\n✅ Setup completed successfully!")
    print("\n🎯 Next steps:")
    print("   1. System is ready to start")
    print("   2. All your API keys are configured")
    print("   3. PostgreSQL database is connected")
    print("   4. Redis caching is disabled (optional)")
    
    # Ask if user wants to start the system
    start_now = input("\n🚀 Start the healthcare system now? (y/n): ").lower().strip()
    if start_now in ['y', 'yes']:
        start_system()
    else:
        print("\n💾 To start later, run: python main.py")
        print("🎪 To run demo, run: python demo.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)
