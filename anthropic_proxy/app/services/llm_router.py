from app.core.config import OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY
import logging

logger = logging.getLogger(__name__)

def get_api_key_for_model(model_name: str) -> str:
    if model_name is None: # Added a check for None model_name
        logger.warning("model_name is None, defaulting to Anthropic API key")
        return ANTHROPIC_API_KEY
    if model_name.startswith("openai/"):
        logger.debug(f"Using OpenAI API key for model: {model_name}")
        return OPENAI_API_KEY
    elif model_name.startswith("gemini/"):
        logger.debug(f"Using Gemini API key for model: {model_name}")
        return GEMINI_API_KEY
    else: # Default or anthropic
        logger.debug(f"Using Anthropic API key for model: {model_name}")
        return ANTHROPIC_API_KEY
