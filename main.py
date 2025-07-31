import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-AI Chat - Perplexity & Groq",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# API configuration from environment variables
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY', '')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
PERPLEXITY_API_URL = os.getenv('PERPLEXITY_API_URL', 'https://api.perplexity.ai/chat/completions')
GROQ_API_URL = os.getenv('GROQ_API_URL', 'https://api.groq.com/openai/v1/chat/completions')

# Model configurations
MODELS = {
    # Perplexity Sonar Models
    'sonar-pro': {
        'provider': 'perplexity',
        'name': 'Sonar Pro',
        'description': '🚀 Premium model with advanced capabilities'
    },
    'sonar-reasoning-pro': {
        'provider': 'perplexity',
        'name': 'Sonar Reasoning Pro',
        'description': '🧠💎 Pro version with enhanced reasoning capabilities'
    },
    'sonar-reasoning': {
        'provider': 'perplexity',
        'name': 'Sonar Reasoning',
        'description': '🧠 Fast reasoning and logical analysis'
    },
    'sonar-deep-research': {
        'provider': 'perplexity', 
        'name': 'Sonar Deep Research',
        'description': '🔍 Comprehensive research and detailed analysis'
    },
    'sonar': {
        'provider': 'perplexity',
        'name': 'Sonar',
        'description': '⚡ Standard Sonar model for general queries'
    },
    'sonar-online': {
        'provider': 'perplexity',
        'name': 'Sonar Online',
        'description': '🌐 Real-time web search and current information'
    },
    'r1-1776': {
        'provider': 'perplexity',
        'name': 'R1-1776',
        'description': '🤖 Advanced reasoning model for complex philosophical analysis'
    },
    
    # Groq Production Models
    'llama-3.3-70b-versatile': {
        'provider': 'groq',
        'name': 'Llama 3.3 70B Versatile',
        'description': '🦙 Latest Llama model with enhanced capabilities'
    },
    'llama-3.1-8b-instant': {
        'provider': 'groq',
        'name': 'Llama 3.1 8B Instant',
        'description': '⚡ Ultra-fast responses with instant speed'
    },
    'gemma2-9b-it': {
        'provider': 'groq',
        'name': 'Gemma 2 9B',
        'description': '💎 Google\'s improved efficient model'
    },
    
    # Groq Preview Models (Latest & Most Advanced)
    'deepseek-r1-distill-llama-70b': {
        'provider': 'groq',
        'name': 'DeepSeek R1 Distill Llama 70B',
        'description': '🧠 Advanced reasoning model with step-by-step thinking'
    },
    'meta-llama/llama-4-maverick-17b-128e-instruct': {
        'provider': 'groq',
        'name': 'Llama 4 Maverick 17B',
        'description': '🚀 Latest Llama 4 with multimodal capabilities'
    },
    'meta-llama/llama-4-scout-17b-16e-instruct': {
        'provider': 'groq',
        'name': 'Llama 4 Scout 17B',
        'description': '👁️ Llama 4 with vision and image understanding'
    },
    'qwen/qwen3-32b': {
        'provider': 'groq',
        'name': 'Qwen 3 32B',
        'description': '🌟 Advanced Chinese & multilingual model'
    },
    'moonshotai/kimi-k2-instruct': {
        'provider': 'groq',
        'name': 'Kimi K2 Instruct',
        'description': '🤖 1T parameter MoE model for agentic intelligence'
    },
    

}

def call_ai_api(prompt, model, perplexity_key, groq_key, conversation_history):
    """Make API call to the appropriate AI provider"""
    model_config = MODELS.get(model)
    if not model_config:
        st.error(f"Unknown model: {model}")
        return None
    
    provider = model_config['provider']
    
    if provider == 'perplexity':
        if not perplexity_key:
            st.error("Please enter your Perplexity API key in the sidebar")
            return None
        return call_perplexity_api(prompt, model, perplexity_key, conversation_history)
    elif provider == 'groq':
        if not groq_key:
            st.error("Please enter your Groq API key in the sidebar")
            return None
        return call_groq_api(prompt, model, groq_key, conversation_history)
    else:
        st.error(f"Unsupported provider: {provider}")
        return None

def call_perplexity_api(prompt, model, api_key, conversation_history):
    """Make API call to Perplexity AI"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Use conversation history directly (user message already added)
    data = {
        'model': model,
        'messages': conversation_history
    }
    
    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Perplexity API Error: {str(e)}")
        return None

def call_groq_api(prompt, model, api_key, conversation_history):
    """Make API call to Groq AI"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Use conversation history directly (user message already added)
    data = {
        'model': model,
        'messages': conversation_history
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Groq API Error: {str(e)}")
        return None



# Main app
def main():
    st.title("🤖 Multi-AI Chat Hub")
    st.markdown("Ask questions and get intelligent responses using **14 cutting-edge AI models** from Perplexity and Groq!")
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Available Models", "14", help="7 Perplexity + 7 Groq models")
    with col2:
        st.metric("Providers", "2", help="Perplexity AI & Groq AI")
    with col3:
        st.metric("Latest Models", "R1-1776 + Llama 4", help="Including R1-1776 reasoning and Llama 4")
    
    st.markdown("---")
    
    # Sidebar for API keys and model selection
    with st.sidebar:
        st.header("🔑 API Keys")
        
        # API Key inputs
        perplexity_key = st.text_input(
            "Perplexity API Key:",
            value=PERPLEXITY_API_KEY,
            type="password",
            help="Enter your Perplexity API key (loaded from .env if available)"
        )
        
        groq_key = st.text_input(
            "Groq API Key:",
            value=GROQ_API_KEY,
            type="password", 
            help="Enter your Groq API key (loaded from .env if available)"
        )
        
        # Show environment status
        if PERPLEXITY_API_KEY or GROQ_API_KEY:
            st.success("✅ API keys loaded from environment variables")
        else:
            st.info("💡 Tip: Create a .env file to store your API keys securely")
        
        st.markdown("---")
        st.header("⚙️ Model Selection")
        
        # Provider status indicators
        st.markdown("### 🔍 Provider Status:")
        col1, col2 = st.columns(2)
        
        with col1:
            if perplexity_key:
                st.success("✅ Perplexity")
            else:
                st.warning("⚠️ Perplexity")
                
        with col2:
            if groq_key:
                st.success("✅ Groq") 
            else:
                st.warning("⚠️ Groq")
        
        st.markdown("---")
        
        # Group models by provider
        model_options = []
        model_labels = []
        
        for model_id, config in MODELS.items():
            provider = config['provider'].title()
            name = config['name']
            model_options.append(model_id)
            model_labels.append(f"{provider}: {name}")
        
        # Create selectbox with custom labels
        selected_index = st.selectbox(
            "Choose Model:",
            range(len(model_options)),
            format_func=lambda x: model_labels[x],
            help="Select the AI model for your query"
        )
        
        model = model_options[selected_index]
        model_config = MODELS[model]
        
        st.markdown("---")
        st.markdown("### 📋 Model Information")
        
        # Determine model category
        perplexity_models = ['sonar-pro', 'sonar-reasoning-pro', 'sonar-reasoning', 'sonar-deep-research', 'sonar', 'sonar-online', 'r1-1776']
        groq_production = ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'gemma2-9b-it']
        
        if model in perplexity_models + groq_production:
            model_category = "🟢 **Production Model** - Stable & Reliable"
            category_color = "success"
        else:
            model_category = "🟡 **Preview Model** - Latest & Experimental"
            category_color = "warning"
        
        st.info(f"**{model_config['name']}**\n\n{model_config['description']}")
        
        # Provider and category info
        provider = model_config['provider'].title()
        if provider == 'Perplexity':
            st.success("🔍 **Provider**: Perplexity AI")
        elif provider == 'Groq':
            st.success("⚡ **Provider**: Groq AI")
            
        # Model category
        if category_color == "success":
            st.success(model_category)
        else:
            st.warning(model_category)
            
        # Special callouts for advanced models
        if 'llama-4' in model:
            st.info("🚀 **NEW**: Latest Llama 4 model with cutting-edge capabilities!")
        elif 'deepseek-r1' in model:
            st.info("🧠 **REASONING**: Advanced model with step-by-step thinking!")
        elif 'kimi-k2' in model:
            st.info("🤖 **MASSIVE**: 1 trillion parameter mixture-of-experts model!")
        elif model == 'r1-1776':
            st.info("🤖 **ADVANCED REASONING**: Specialized for complex philosophical & analytical thinking!")
        elif model == 'sonar-reasoning-pro':
            st.info("🧠💎 **PRO REASONING**: Enhanced version with superior logical analysis!")
        elif model == 'sonar-pro':
            st.info("🚀 **PREMIUM**: Top-tier Sonar model with enhanced performance!")
        elif model == 'sonar-online':
            st.info("🌐 **REAL-TIME**: Access to current web information and live data!")
    
    # Display conversation history with expand/collapse
    if st.session_state.conversation_history:
        with st.expander("💬 Conversation History", expanded=True):
            for i, message in enumerate(st.session_state.conversation_history):
                if message['role'] == 'user':
                    st.markdown(f"**👤 You:** {message['content']}")
                else:
                    st.markdown(f"**🤖 AI:** {message['content']}")
        st.markdown("---")
    
    # Comparison mode toggle
    comparison_mode = st.checkbox("🔄 Enable Model Comparison", help="Compare two different models side by side")
    
    if comparison_mode:
        st.markdown("### 🔄 Model Comparison Mode")
        
        # Model selection for comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model 1 (Left Side):**")
            model1_index = st.selectbox(
                "Choose Model 1:",
                range(len(model_options)),
                format_func=lambda x: model_labels[x],
                key="model1_select"
            )
            model1 = model_options[model1_index]
        
        with col2:
            st.markdown("**Model 2 (Right Side):**")
            model2_index = st.selectbox(
                "Choose Model 2:",
                range(len(model_options)),
                format_func=lambda x: model_labels[x],
                key="model2_select"
            )
            model2 = model_options[model2_index]
        
        # Prompt mode selection
        prompt_mode = st.radio(
            "**Prompt Mode:**",
            ["Same Prompt", "Different Prompts"],
            help="Choose whether to use the same prompt for both models or different prompts"
        )
        
        st.markdown("---")
    
    # Main chat interface using form to prevent double-click issues
    with st.form("chat_form", clear_on_submit=True):
        if comparison_mode and prompt_mode == "Different Prompts":
            col1, col2 = st.columns(2)
            
            with col1:
                prompt1 = st.text_area(
                    "Prompt for Model 1:",
                    placeholder="Enter prompt for left model...",
                    height=120,
                    key="prompt1_input"
                )
            
            with col2:
                prompt2 = st.text_area(
                    "Prompt for Model 2:",
                    placeholder="Enter prompt for right model...",
                    height=120,
                    key="prompt2_input"
                )
        else:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                prompt = st.text_area(
                    "Enter your question or prompt:",
                    placeholder="What would you like to know?",
                    height=120,
                    key="prompt_input"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        
        # Single submit button for all modes
        if comparison_mode:
            submit_button = st.form_submit_button("🔄 Compare Models", type="primary", use_container_width=True)
        else:
            submit_button = st.form_submit_button("🚀 Send", type="primary", use_container_width=True)
    
    # Clear button outside form
    clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.conversation_history = []
        st.rerun()
    
    # Handle form submission
    if submit_button:
        if comparison_mode:
            # Comparison mode logic
            if prompt_mode == "Same Prompt":
                if not prompt:
                    st.warning("Please enter a prompt for comparison!")
                    return
                
                # Get model names
                model1_name = MODELS[model1]['name']
                model2_name = MODELS[model2]['name']
                
                st.markdown("---")
                st.markdown("### 🔄 Model Comparison Results")
                
                # Create two columns for side-by-side comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**🤖 {model1_name}**")
                    with st.spinner(f"Getting response from {model1_name}..."):
                        # Create messages array with the current prompt
                        messages1 = [{'role': 'user', 'content': prompt}]
                        response1 = call_ai_api(prompt, model1, perplexity_key, groq_key, messages1)
                        
                        if response1:
                            try:
                                content1 = response1['choices'][0]['message']['content']
                                st.markdown(content1)
                                
                                with st.expander(f"📊 {model1_name} Details"):
                                    st.write("**Model:**", response1.get('model', model1))
                                    st.write("**Provider:**", MODELS[model1]['provider'].title())
                                    if 'usage' in response1:
                                        st.write("**Tokens:**", response1['usage'].get('total_tokens', 'N/A'))
                            except (KeyError, IndexError) as e:
                                st.error(f"Error with {model1_name}")
                        else:
                            st.error(f"Failed to get response from {model1_name}")
                
                with col2:
                    st.markdown(f"**🤖 {model2_name}**")
                    with st.spinner(f"Getting response from {model2_name}..."):
                        # Create messages array with the current prompt
                        messages2 = [{'role': 'user', 'content': prompt}]
                        response2 = call_ai_api(prompt, model2, perplexity_key, groq_key, messages2)
                        
                        if response2:
                            try:
                                content2 = response2['choices'][0]['message']['content']
                                st.markdown(content2)
                                
                                with st.expander(f"📊 {model2_name} Details"):
                                    st.write("**Model:**", response2.get('model', model2))
                                    st.write("**Provider:**", MODELS[model2]['provider'].title())
                                    if 'usage' in response2:
                                        st.write("**Tokens:**", response2['usage'].get('total_tokens', 'N/A'))
                            except (KeyError, IndexError) as e:
                                st.error(f"Error with {model2_name}")
                        else:
                            st.error(f"Failed to get response from {model2_name}")
            
            else:  # Different Prompts
                if not prompt1 or not prompt2:
                    st.warning("Please enter prompts for both models!")
                    return
                
                # Get model names
                model1_name = MODELS[model1]['name']
                model2_name = MODELS[model2]['name']
                
                st.markdown("---")
                st.markdown("### 🔄 Model Comparison Results")
                
                # Create two columns for side-by-side comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**🤖 {model1_name}**")
                    st.markdown(f"**Prompt:** {prompt1}")
                    with st.spinner(f"Getting response from {model1_name}..."):
                        # Create messages array with the current prompt
                        messages1 = [{'role': 'user', 'content': prompt1}]
                        response1 = call_ai_api(prompt1, model1, perplexity_key, groq_key, messages1)
                        
                        if response1:
                            try:
                                content1 = response1['choices'][0]['message']['content']
                                st.markdown(content1)
                                
                                with st.expander(f"📊 {model1_name} Details"):
                                    st.write("**Model:**", response1.get('model', model1))
                                    st.write("**Provider:**", MODELS[model1]['provider'].title())
                                    if 'usage' in response1:
                                        st.write("**Tokens:**", response1['usage'].get('total_tokens', 'N/A'))
                            except (KeyError, IndexError) as e:
                                st.error(f"Error with {model1_name}")
                        else:
                            st.error(f"Failed to get response from {model1_name}")
                
                with col2:
                    st.markdown(f"**🤖 {model2_name}**")
                    st.markdown(f"**Prompt:** {prompt2}")
                    with st.spinner(f"Getting response from {model2_name}..."):
                        # Create messages array with the current prompt
                        messages2 = [{'role': 'user', 'content': prompt2}]
                        response2 = call_ai_api(prompt2, model2, perplexity_key, groq_key, messages2)
                        
                        if response2:
                            try:
                                content2 = response2['choices'][0]['message']['content']
                                st.markdown(content2)
                                
                                with st.expander(f"📊 {model2_name} Details"):
                                    st.write("**Model:**", response2.get('model', model2))
                                    st.write("**Provider:**", MODELS[model2]['provider'].title())
                                    if 'usage' in response2:
                                        st.write("**Tokens:**", response2['usage'].get('total_tokens', 'N/A'))
                            except (KeyError, IndexError) as e:
                                st.error(f"Error with {model2_name}")
                        else:
                            st.error(f"Failed to get response from {model2_name}")
        
        else:
            # Regular single model mode
            if not prompt:
                st.warning("Please enter a question or prompt!")
                return
                
            model_name = MODELS[model]['name']
            
            # Create a copy of conversation history and add current user message
            current_messages = st.session_state.conversation_history.copy()
            current_messages.append({
                'role': 'user',
                'content': prompt
            })
            
            with st.spinner(f"Getting response from {model_name}..."):
                # Make API call with the updated messages
                response = call_ai_api(prompt, model, perplexity_key, groq_key, current_messages)
                
                if response:
                    try:
                        content = response['choices'][0]['message']['content']
                        
                        # Add user message to conversation history
                        st.session_state.conversation_history.append({
                            'role': 'user',
                            'content': prompt
                        })
                        
                        # Add AI response to conversation history
                        st.session_state.conversation_history.append({
                            'role': 'assistant',
                            'content': content
                        })
                        
                        # Display response
                        st.markdown("---")
                        st.markdown("### 💬 Response:")
                        st.markdown(content)
                        
                        # Show additional info in expander
                        with st.expander("📊 Response Details"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write("**Model Used:**", response.get('model', model))
                            with col2:
                                st.write("**Provider:**", MODELS[model]['provider'].title())
                            with col3:
                                if 'usage' in response:
                                    st.write("**Tokens Used:**", response['usage'].get('total_tokens', 'N/A'))
                            
                            st.json(response)
                            
                    except (KeyError, IndexError) as e:
                        st.error("Error parsing response. Please try again.")
                        st.json(response)
    
    elif submit_button and not prompt:
        st.warning("Please enter a question or prompt!")
    
    # Footer
    st.markdown("---")
    
    # Model summary
    st.markdown("### 🎯 Available Models Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔍 Perplexity Models:**")
        st.markdown("• **R1-1776** - Advanced reasoning & philosophy")
        st.markdown("• **Sonar Pro** - Premium advanced capabilities")
        st.markdown("• **Sonar Reasoning Pro** - Enhanced reasoning")
        st.markdown("• **Sonar Reasoning** - Fast logical analysis")
        st.markdown("• **Sonar Deep Research** - Comprehensive research")
        st.markdown("• **Sonar** - Standard general queries")
        st.markdown("• **Sonar Online** - Real-time web search")
        
    with col2:
        st.markdown("**⚡ Groq Models:**")
        st.markdown("• **Llama 3.3 70B** - Latest production model")
        st.markdown("• **Llama 3.1 8B** - Ultra-fast responses")
        st.markdown("• **Gemma 2 9B** - Efficient Google model")
        st.markdown("• **DeepSeek R1** - Advanced reasoning")
        st.markdown("• **Llama 4 Maverick/Scout** - Multimodal & Vision")
        st.markdown("• **Qwen 3 32B** - Multilingual excellence")
        st.markdown("• **Kimi K2** - 1T parameter MoE")
        

    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
        "🚀 Powered by Perplexity AI & Groq AI | 14 AI Models | 7 Each Provider | Built with Streamlit"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()