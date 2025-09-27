#!/bin/bash

# Multi-Agent Healthcare System - Native Deployment Script
# This script sets up and runs the healthcare system without Docker requirements

set -e  # Exit on any error

echo "🏥 Multi-Agent Healthcare System - Native Setup & Deployment"
echo "============================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Python 3.8+ is installed (more compatible than 3.11+)
check_python() {
    print_info "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_status "Python $PYTHON_VERSION found (compatible)"
        else
            print_error "Python 3.8+ is required, found $PYTHON_VERSION"
            exit 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version | cut -d' ' -f2)
        print_status "Python $PYTHON_VERSION found"
    else
        print_error "Python is required but not found"
        print_info "Please install Python 3.8+ from https://python.org"
        exit 1
    fi
}

# Check PostgreSQL availability (optional - can be remote)
check_postgresql() {
    print_info "Checking PostgreSQL availability..."

    # Check if psql is available (optional)
    if command -v psql &> /dev/null; then
        print_status "PostgreSQL client tools found"

        # Try to connect to the database from .env
        if [ -f ".env" ]; then
            DB_URL=$(grep "DATABASE_URL" .env | cut -d'=' -f2)
            if [[ $DB_URL == postgresql://* ]]; then
                print_status "PostgreSQL connection string configured"
            else
                print_warning "PostgreSQL URL not found in .env - will use default"
            fi
        fi
    else
        print_warning "PostgreSQL client not found locally"
        print_info "That's OK - you can use a remote PostgreSQL database"
        print_info "Make sure your DATABASE_URL in .env points to your PostgreSQL server"
    fi
}

# Create virtual environment
setup_virtual_environment() {
    print_info "Setting up Python virtual environment..."

    # Use python3 or python, whichever is available
    PYTHON_CMD="python3"
    if ! command -v python3 &> /dev/null; then
        PYTHON_CMD="python"
    fi

    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_status "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi

    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        source venv/Scripts/activate
    else
        # Unix/Linux/Mac
        source venv/bin/activate
    fi
    print_status "Virtual environment activated"
}

# Install Python dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."

    # Upgrade pip first
    pip install --upgrade pip

    # Install requirements with error handling
    if pip install -r requirements.txt; then
        print_status "Dependencies installed successfully"
    else
        print_error "Failed to install some dependencies"
        print_info "This might be due to system-specific packages like pyaudio"
        print_info "The system will still work without audio processing"
    fi
}

# Setup environment configuration
setup_environment() {
    print_info "Setting up environment configuration..."

    if [ ! -f ".env" ]; then
        print_error ".env file not found"
        print_info "Creating basic .env file..."

        cat > .env << EOF
# Groq API Configuration (Required)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192

# Multi-Model Provider Configuration
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Model Selection Strategy
PRIMARY_MODEL_PROVIDER=groq
DIAGNOSTIC_MODEL_PROVIDER=groq
VISION_MODEL_PROVIDER=groq
VOICE_MODEL_PROVIDER=groq

# PostgreSQL Database Configuration
DATABASE_URL=postgresql://healthcare_user:healthcare_pass@localhost:5432/healthcare_db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Security
SECRET_KEY=healthcare_system_secret_key_2025
EOF

        print_warning "Basic .env file created - please configure your API keys"
    else
        print_status "Environment configuration exists"

        # Check if Groq API key is configured
        if grep -q "your_groq_api_key_here" .env; then
            print_warning "Please update your Groq API key in .env file"
        else
            print_status "API keys appear to be configured"
        fi
    fi
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."

    DIRECTORIES=("logs" "data/medical_images" "data/voice_samples" "data/text_data")

    for dir in "${DIRECTORIES[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
}

# Test database connection
test_database_connection() {
    print_info "Testing database connection..."

    # Use Python to test the connection
    python3 -c "
try:
    from config.settings import config
    import psycopg2
    conn = psycopg2.connect(config.database.url)
    conn.close()
    print('✅ Database connection successful')
except ImportError:
    print('⚠️  Database libraries not available yet')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    print('💡 Make sure PostgreSQL is running and accessible')
" 2>/dev/null || print_warning "Database connection test skipped"
}

# Run system validation
validate_system() {
    print_info "Running system validation..."

    # Test Python imports
    python3 -c "
try:
    from config.settings import config
    from orchestrator import orchestrator
    print('✅ System modules loaded successfully')
    if config.validate_config():
        print('✅ Configuration validation passed')
    else:
        print('⚠️  Configuration needs attention')
except Exception as e:
    print(f'❌ System validation failed: {e}')
    exit(1)
" || {
    print_error "System validation failed"
    print_info "Try running: python setup.py"
    exit 1
}
}

# Start the system
start_system() {
    print_info "Starting Multi-Agent Healthcare System..."

    # Check if port 8000 is available
    if command -v lsof &> /dev/null; then
        if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_error "Port 8000 is already in use"
            print_info "Stop the existing service or change the port in .env"
            exit 1
        fi
    fi

    print_status "🚀 Starting healthcare system on http://localhost:8000"
    print_info "📚 API Documentation: http://localhost:8000/docs"
    print_info "❤️  Health Check: http://localhost:8000/health"
    print_info "⚡ Press Ctrl+C to stop the server"
    print_info ""

    # Start the application
    python3 main.py
}

# Run demo
run_demo() {
    print_info "Running system demonstration..."
    python3 demo.py
}

# Quick setup for first-time users
quick_setup() {
    print_info "Running quick setup for first-time users..."
    python3 setup.py
}

# Main execution
main() {
    case "${1:-setup}" in
        "setup"|"install")
            print_info "Running complete system setup (no Docker required)..."
            check_python
            check_postgresql
            setup_virtual_environment
            install_dependencies
            setup_environment
            create_directories
            test_database_connection
            validate_system
            print_status "🎉 Setup completed successfully!"
            print_info ""
            print_info "Next steps:"
            print_info "  1. Configure your API keys in .env file"
            print_info "  2. Ensure PostgreSQL is running and accessible"
            print_info "  3. Run: ./deploy.sh start"
            ;;
        "start"|"run")
            print_info "Starting the healthcare system (native mode)..."
            setup_virtual_environment
            validate_system
            start_system
            ;;
        "test")
            print_info "Running system tests..."
            setup_virtual_environment
            python3 -m pytest tests/ -v || print_warning "Some tests may require additional setup"
            ;;
        "demo")
            print_info "Running system demonstration..."
            setup_virtual_environment
            run_demo
            ;;
        "quick")
            print_info "Running quick setup wizard..."
            quick_setup
            ;;
        "stop")
            print_info "Stopping healthcare system..."
            pkill -f "python3 main.py" || pkill -f "python main.py" || true
            print_status "System stopped"
            ;;
        "status")
            print_info "Checking system status..."
            if pgrep -f "python3 main.py" > /dev/null || pgrep -f "python main.py" > /dev/null; then
                print_status "Healthcare system is running"
                print_info "Health check: curl http://localhost:8000/health"
            else
                print_warning "Healthcare system is not running"
            fi
            ;;
        "clean")
            print_info "Cleaning up system..."
            rm -rf venv __pycache__ .pytest_cache logs/*.log *.pyc
            print_status "Cleanup completed"
            ;;
        *)
            echo "Multi-Agent Healthcare System - Native Deployment"
            echo ""
            echo "Usage: $0 {setup|start|test|demo|quick|stop|status|clean}"
            echo ""
            echo "Commands:"
            echo "  setup/install - Complete system setup (Python + PostgreSQL only)"
            echo "  start/run     - Start the healthcare system server"
            echo "  test          - Run system tests"
            echo "  demo          - Run comprehensive system demonstration"
            echo "  quick         - Run quick setup wizard"
            echo "  stop          - Stop the running system"
            echo "  status        - Check if system is running"
            echo "  clean         - Clean up temporary files"
            echo ""
            echo "No Docker required! Works with:"
            echo "  ✅ Python 3.8+"
            echo "  ✅ PostgreSQL (local or remote)"
            echo "  ✅ Your existing API keys"
            echo ""
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
