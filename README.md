# Perplexity AI Streamlit Webapp

A beautiful web interface for interacting with Perplexity AI's sonar models.

## Features

- 🤖 Interactive chat interface
- 🔄 Model selection across 2 providers (14 models total)
- 📊 Response details and token usage
- 🎨 Modern, user-friendly design

## Setup

### Option 1: UV Environment (Recommended - Faster!)

1. **Create and activate UV environment:**
```bash
# Quick setup
./activate.sh

# Or manual setup
uv venv multi-ai-chat
source multi-ai-chat/bin/activate
uv pip install -r requirements.txt
```

2. **Configure API Keys** (Optional but recommended):
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor
```

3. **Run the app:**
```bash
# Quick run
./run.sh

# Or manual run
streamlit run main.py
```

### Option 2: Traditional pip

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure API Keys** (Optional but recommended):
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor
```

3. **Run the app:**
```bash
streamlit run main.py
```

4. **Open your browser** and go to the URL shown in the terminal (usually `http://localhost:8501`)

## Usage

### **Basic Chat:**
1. Choose your preferred model from the sidebar (14 models available!)
2. Enter your question or prompt in the text area
3. Click "🚀 Send" to get your response
4. View response details by expanding the "📊 Response Details" section

## Models

**🔍 Perplexity Models:** R1-1776, Sonar Pro, Sonar Reasoning Pro, Sonar Reasoning, Sonar Deep Research, Sonar, Sonar Online

**⚡ Groq Models:** Llama 4 Maverick/Scout, DeepSeek R1, Llama 3.3 70B, Llama 3.1 8B, Gemma 2 9B, Qwen 3 32B, Kimi K2

**🔍 Perplexity Models:** R1-1776, Sonar Pro, Sonar Reasoning Pro, Sonar Reasoning, Sonar Deep Research, Sonar, Sonar Online

**⚡ Groq Models:** Llama 4 Maverick/Scout, DeepSeek R1, Llama 3.3 70B, Llama 3.1 8B, Gemma 2 9B, Qwen 3 32B, Kimi K2

Enjoy chatting with 14 AI models! 🚀

## 🔐 Environment Variables

### **API Key Security**
For better security, store your API keys in environment variables:

1. **Create `.env` file:**
```bash
cp .env.example .env
```

2. **Add your API keys:**
```bash
# Perplexity AI API Key
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Groq AI API Key  
GROQ_API_KEY=your_groq_api_key_here
```

3. **Benefits:**
- ✅ **Secure**: API keys not in code
- ✅ **Flexible**: Easy to change without editing code
- ✅ **Portable**: Works across different environments
- ✅ **Git-safe**: `.env` is in `.gitignore`

### **Getting API Keys**
- **Perplexity**: Get your key at [Perplexity API](https://www.perplexity.ai/settings/api)
- **Groq**: Get your key at [Groq Console](https://console.groq.com/keys)

## UV Environment Benefits

### Why UV?
- ⚡ **10-100x faster** than pip for installs and resolves
- 🔒 **Better dependency resolution** with conflict detection
- 📦 **Modern Python packaging** with pyproject.toml support
- 🛠️ **Built-in virtual environment** management

### UV Commands
```bash
# Environment management
uv venv multi-ai-chat        # Create environment
source multi-ai-chat/bin/activate  # Activate
deactivate                   # Deactivate

# Package management
uv pip install <package>     # Install package
uv pip install -r requirements.txt  # Install from requirements
uv pip list                  # List packages
uv pip freeze                # Export requirements

# Project management (with pyproject.toml)
uv sync                      # Sync dependencies
uv add <package>             # Add new dependency
uv remove <package>          # Remove dependency
```

### Performance Comparison
- **pip**: ~15-30 seconds to install dependencies
- **uv**: ~1-3 seconds to install dependencies 🚀

## 🔧 Troubleshooting

### Common API Issues



#### Perplexity/Groq Errors

**🔑 API Key Issues**
- Verify your API keys are correct
- Check provider documentation for key format
- Ensure keys have required permissions

**⚠️ Rate Limits**
- Wait a moment between requests
- Groq has generous free tier limits
- Perplexity provides excellent performance

### Quick Fixes

1. **No API Key Error**: Enter valid keys in the sidebar
2. **Model Not Working**: Try a different model from the same provider
3. **Slow Responses**: Try smaller/faster models (GPT-4o Mini, Llama 3.1 8B)
4. **OpenAI Issues**: Use Perplexity R1-1776 or Groq DeepSeek R1 for reasoning

### Best Practices

- **Start with free models**: Groq and Perplexity offer generous free tiers
- **Compare responses**: Try the same prompt across different providers
- **Use appropriate models**: Reasoning models for complex questions, fast models for simple queries
- **Monitor usage**: Check your API usage regularly 