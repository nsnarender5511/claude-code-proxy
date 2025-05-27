import logging
import json
from typing import Dict, Any, AsyncGenerator
from src.api.v1.schemas.anthropic_api import (
    AnthropicSSEError, # Updated import
    AnthropicSSEErrorContent # Updated import
)

logger = logging.getLogger(__name__)

def translate_openai_error_to_anthropic_format(
    error_details: Dict[str, Any],
) -> Dict[str, Any]:
    logger.debug(f'Translating error to Anthropic format: {error_details}')
    error_type = 'api_error' # Default error type
    error_message = 'An unexpected error occurred.'

    if isinstance(error_details, dict):
        # Look for OpenAI/LiteLLM like error structure first
        err_obj = error_details.get('error', error_details)
        
        # If it's a string, it might be a direct message (less common for structured errors)
        if isinstance(err_obj, str):
            error_message = err_obj
        elif isinstance(err_obj, dict):
            error_message = err_obj.get('message', error_message)
            raw_type = err_obj.get('type')
            if raw_type:
                raw_type_lower = raw_type.lower()
                if 'auth' in raw_type_lower or 'permission' in raw_type_lower or 'key' in raw_type_lower:
                    error_type = 'authentication_error'
                elif 'rate_limit' in raw_type_lower:
                    error_type = 'rate_limit_error'
                elif 'invalid_request' in raw_type_lower or 'validation' in raw_type_lower or 'bad_request' in raw_type_lower:
                    error_type = 'invalid_request_error'
                elif 'not_found' in raw_type_lower or 'model_not_found' in raw_type_lower:
                    error_type = 'not_found_error' # Or map to invalid_request_error if Anthropic prefers
                elif 'overloaded' in raw_type_lower or 'capacity' in raw_type_lower or 'unavailable' in raw_type_lower:
                    error_type = 'overloaded_error'
                # Add more specific mappings as needed
                else:
                    error_type = 'api_error' # Fallback for unmapped OpenAI types
    elif isinstance(error_details, Exception):
        error_message = str(error_details)
        # Attempt to classify common Python exceptions if helpful
        if isinstance(error_details, ValueError):
            error_type = 'invalid_request_error'
        elif isinstance(error_details, ConnectionError):
            error_type = 'api_connection_error'
            
    return {
        'type': 'error',
        'error': {
            'type': error_type,
            'message': error_message
        }
    }

async def generate_anthropic_error_sse_event(
    error_type_str: str, message: str
) -> AsyncGenerator[str, None]:
    logger.debug(f"Generating Anthropic SSE error event: type='{error_type_str}', message='{message}'")
    error_content = AnthropicSSEErrorContent(type=error_type_str, message=message)
    error_event_model = AnthropicSSEError(type='error', error=error_content)
    yield f'event: error\ndata: {error_event_model.model_dump_json()}\n\n' 