#!/bin/bash

# LANISTR Web Interface Startup Script
# Launches both Streamlit frontend and FastAPI backend

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
STREAMLIT_PORT=${STREAMLIT_PORT:-8501}
FASTAPI_PORT=${FASTAPI_PORT:-8000}
STREAMLIT_HOST=${STREAMLIT_HOST:-"0.0.0.0"}
FASTAPI_HOST=${FASTAPI_HOST:-"0.0.0.0"}

print_header() {
    echo "
╔══════════════════════════════════════════════════════════════╗
║                🚀 LANISTR Web Interface                     ║
║                                                              ║
║  Starting Streamlit frontend and FastAPI backend            ║
╚══════════════════════════════════════════════════════════════╝
"
}

check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [ ! -f "app.py" ] || [ ! -f "api.py" ]; then
        print_error "Please run this script from the web_interface directory"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if requirements are installed
    if ! python3 -c "import streamlit, fastapi" 2>/dev/null; then
        print_warning "Web interface dependencies not installed"
        print_info "Installing requirements..."
        pip3 install -r requirements.txt
    fi
    
    print_success "Prerequisites check passed"
}

start_services() {
    print_info "Starting services..."
    
    # Start FastAPI backend in background
    print_info "Starting FastAPI backend on http://$FASTAPI_HOST:$FASTAPI_PORT"
    python3 api.py &
    FASTAPI_PID=$!
    
    # Wait a moment for FastAPI to start
    sleep 3
    
    # Start Streamlit frontend
    print_info "Starting Streamlit frontend on http://$STREAMLIT_HOST:$STREAMLIT_PORT"
    print_info "API documentation available at http://$FASTAPI_HOST:$FASTAPI_PORT/docs"
    
    # Set environment variables for Streamlit
    export STREAMLIT_SERVER_PORT=$STREAMLIT_PORT
    export STREAMLIT_SERVER_ADDRESS=$STREAMLIT_HOST
    export STREAMLIT_SERVER_HEADLESS=true
    export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    
    # Start Streamlit
    streamlit run app.py --server.port $STREAMLIT_PORT --server.address $STREAMLIT_HOST
    
    # Cleanup on exit
    if [ ! -z "$FASTAPI_PID" ]; then
        print_info "Stopping FastAPI backend..."
        kill $FASTAPI_PID 2>/dev/null || true
    fi
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --streamlit-only    Start only Streamlit frontend"
    echo "  --api-only          Start only FastAPI backend"
    echo "  --port PORT         Set Streamlit port (default: 8501)"
    echo "  --api-port PORT     Set FastAPI port (default: 8000)"
    echo "  --host HOST         Set Streamlit host (default: 0.0.0.0)"
    echo "  --api-host HOST     Set FastAPI host (default: 0.0.0.0)"
    echo "  --help              Show this help message"
    echo
    echo "Examples:"
    echo "  $0                                    # Start both services"
    echo "  $0 --streamlit-only                   # Start only frontend"
    echo "  $0 --port 8502 --api-port 8001       # Custom ports"
    echo "  $0 --host localhost --api-host localhost  # Local only"
}

main() {
    print_header
    
    # Parse command line arguments
    STREAMLIT_ONLY=false
    API_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --streamlit-only)
                STREAMLIT_ONLY=true
                shift
                ;;
            --api-only)
                API_ONLY=true
                shift
                ;;
            --port)
                STREAMLIT_PORT="$2"
                shift 2
                ;;
            --api-port)
                FASTAPI_PORT="$2"
                shift 2
                ;;
            --host)
                STREAMLIT_HOST="$2"
                shift 2
                ;;
            --api-host)
                FASTAPI_HOST="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_prerequisites
    
    # Start services based on options
    if [ "$STREAMLIT_ONLY" = true ]; then
        print_info "Starting Streamlit frontend only..."
        export STREAMLIT_SERVER_PORT=$STREAMLIT_PORT
        export STREAMLIT_SERVER_ADDRESS=$STREAMLIT_HOST
        streamlit run app.py --server.port $STREAMLIT_PORT --server.address $STREAMLIT_HOST
    elif [ "$API_ONLY" = true ]; then
        print_info "Starting FastAPI backend only..."
        python3 api.py
    else
        start_services
    fi
}

# Handle Ctrl+C gracefully
trap 'print_info "Shutting down..."; exit 0' INT

# Run main function
main "$@" 