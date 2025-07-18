#!/bin/bash

# LANISTR Web Interface Status Check
# This script checks if the web interface services are running properly

echo "🔍 Checking LANISTR Web Interface Status..."
echo "=========================================="

# Check if services are running
echo "📊 Checking service processes..."

# Check Streamlit (Frontend)
if lsof -i :8501 > /dev/null 2>&1; then
    echo "✅ Streamlit Frontend: RUNNING on http://localhost:8501"
else
    echo "❌ Streamlit Frontend: NOT RUNNING"
fi

# Check FastAPI (Backend)
if lsof -i :8000 > /dev/null 2>&1; then
    echo "✅ FastAPI Backend: RUNNING on http://localhost:8000"
else
    echo "❌ FastAPI Backend: NOT RUNNING"
fi

echo ""
echo "🔗 Testing API endpoints..."

# Test API health
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API Health Check: PASSED"
    echo "   Health endpoint: http://localhost:8000/health"
else
    echo "❌ API Health Check: FAILED"
fi

# Test API docs
if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
    echo "✅ API Documentation: AVAILABLE at http://localhost:8000/docs"
else
    echo "❌ API Documentation: NOT AVAILABLE"
fi

# Test Streamlit frontend
if curl -s http://localhost:8501 > /dev/null 2>&1; then
    echo "✅ Streamlit Frontend: AVAILABLE at http://localhost:8501"
else
    echo "❌ Streamlit Frontend: NOT AVAILABLE"
fi

echo ""
echo "🎯 Quick Access URLs:"
echo "   Frontend: http://localhost:8501"
echo "   API Docs: http://localhost:8000/docs"
echo "   API Health: http://localhost:8000/health"
echo ""

echo "📋 Next Steps:"
echo "   1. Open http://localhost:8501 in your browser"
echo "   2. Configure your Google Cloud project"
echo "   3. Submit your first training job"
echo "   4. Monitor progress in real-time"
echo ""

echo "📖 For detailed instructions, see: WEB_INTERFACE_ACCESS.md" 