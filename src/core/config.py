import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get port from environment
PORT = int(os.environ.get("PORT", 8080))

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Get preferred provider (default to openai)
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")

# List of OpenAI models - default
DEFAULT_OPENAI_MODELS = [
    "o3-mini", "o1", "o1-mini", "o1-pro",
    "gpt-4.5-preview", "gpt-4o", "gpt-4o-audio-preview",
    "chatgpt-4o-latest", "gpt-4o-mini", "gpt-4o-mini-audio-preview",
    "gpt-4.1", "gpt-4.1-mini" # Retaining defaults for fallback
]
# Load from environment or use default
OPENAI_MODELS_STR = os.environ.get("OPENAI_MODELS_CSV")
OPENAI_MODELS = [model.strip() for model in OPENAI_MODELS_STR.split(',')] if OPENAI_MODELS_STR else DEFAULT_OPENAI_MODELS

# List of Gemini models - default
DEFAULT_GEMINI_MODELS = [
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.0-flash"
]
# Load from environment or use default
GEMINI_MODELS_STR = os.environ.get("GEMINI_MODELS_CSV")
GEMINI_MODELS = [model.strip() for model in GEMINI_MODELS_STR.split(',')] if GEMINI_MODELS_STR else DEFAULT_GEMINI_MODELS
