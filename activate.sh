#!/bin/bash
# UV Environment Activation Script for Multi-AI Chat Hub

echo "🚀 Activating Multi-AI Chat Hub UV Environment..."
source multi-ai-chat/bin/activate

echo "✅ Environment activated! Python version:"
python --version

echo ""
echo "📋 Available commands:"
echo "  streamlit run main.py  - Run the Multi-AI Chat Hub"
echo "  uv pip install <pkg>  - Install new packages"
echo "  uv pip list           - List installed packages"
echo "  deactivate            - Exit the environment"
echo ""
echo "🎯 Ready to chat with 21 AI models!" 