#!/bin/bash
# Quick Run Script for Multi-AI Chat Hub

echo "🚀 Starting Multi-AI Chat Hub..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. API keys will need to be entered manually."
    echo "💡 Tip: Run 'cp .env.example .env' and add your API keys for automatic loading."
fi

source multi-ai-chat/bin/activate
streamlit run main.py 