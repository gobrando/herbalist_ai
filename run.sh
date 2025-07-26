#!/bin/bash

# Ask an Herbalist - Run Script
echo "🌿 Starting Ask an Herbalist RAG Application..."

# Activate virtual environment
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Check if the vector database exists
if [ ! -d "chroma_db" ]; then
    echo "🔧 Vector database not found. Running setup..."
    python setup.py
    if [ $? -ne 0 ]; then
        echo "❌ Setup failed. Please check the error messages above."
        exit 1
    fi
fi

# Start the Streamlit application
echo "🚀 Starting Streamlit application..."
echo "📱 The app will open in your default browser at http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

streamlit run app.py --server.port 8501 --server.address 0.0.0.0