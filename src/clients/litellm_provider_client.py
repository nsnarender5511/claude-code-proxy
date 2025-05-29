import litellm
from loguru import logger
import asyncio
from typing import AsyncGenerator, Union, Optional, Any
from src.core.config import settings
from src.models.openai_provider_models import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChunk,
)

# Helper function to resolve provider details
def _resolve_provider_details(initial_model_in_payload: Optional[str], target_llm_provider_setting: str) -> tuple[Optional[str], Optional[str]]:
    target_model_to_use = initial_model_in_payload
    api_key_to_use = None

    if target_llm_provider_setting == 'openai':
        translated_model = settings.ANTHROPIC_TO_OPENAI_MAP.get(
            settings.OPENAI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY
        )
        if not translated_model:
            logger.error(
                f'Could not translate default OpenAI model key: {settings.OPENAI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY}'
            )
            raise ValueError('Default OpenAI model key not found in translation map.')
        target_model_to_use = translated_model
        api_key_to_use = settings.OPENAI_API_KEY
        if not api_key_to_use:
            logger.error('OpenAI is the target provider, but OPENAI_API_KEY is not set.')
            raise ValueError("OPENAI_API_KEY is not configured for the target provider 'openai'.")
        logger.debug(f"Resolved to OpenAI. Overriding model to '{target_model_to_use}'.")
    
    elif target_llm_provider_setting == 'gemini':
        translated_model = settings.ANTHROPIC_TO_GEMINI_MAP.get(
            settings.GEMINI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY
        )
        if not translated_model:
            logger.error(
                f'Could not translate default Gemini model key: {settings.GEMINI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY}'
            )
            raise ValueError('Default Gemini model key not found in translation map.')
        target_model_to_use = translated_model
        api_key_to_use = settings.GEMINI_API_KEY
        if not api_key_to_use:
            logger.error('Gemini is the target provider, but GEMINI_API_KEY is not set.')
            raise ValueError("GEMINI_API_KEY is not configured for the target provider 'gemini'.")
        logger.debug(f"Resolved to Gemini. Overriding model to '{target_model_to_use}'.")
    
    elif target_llm_provider_setting == 'anthropic':
        # For direct Anthropic, we use the model name as passed, assuming it's an Anthropic model.
        # No translation key needed here, just the API key.
        api_key_to_use = settings.ANTHROPIC_API_KEY
        if not api_key_to_use:
            logger.error('Anthropic is the target provider, but ANTHROPIC_API_KEY is not set.')
            raise ValueError(
                "ANTHROPIC_API_KEY is not configured for the target provider 'anthropic'."
            )
        logger.debug(f"Resolved to Anthropic. Using model '{target_model_to_use}'.")
    else:
        logger.warning(
            f"TARGET_LLM_PROVIDER '{target_llm_provider_setting}' not explicitly handled for model/key override. Using original model '{target_model_to_use}' and relying on global LiteLLM key config."
        )
    return target_model_to_use, api_key_to_use

# Helper function to process non-streaming responses
def _process_non_stream_response(litellm_completion: Any) -> OpenAIChatCompletionResponse:
    logger.debug('Processing LiteLLM non-stream response.')
    response_dict = litellm_completion.model_dump()
    if 'object' not in response_dict or response_dict['object'] != 'chat.completion':
        response_dict['object'] = 'chat.completion'
    return OpenAIChatCompletionResponse(**response_dict)

# Helper function to process streaming responses
async def _process_stream_response(litellm_stream_completion: Any) -> AsyncGenerator[OpenAIChatCompletionChunk, None]:
    logger.debug('Processing LiteLLM stream response.')
    async for chunk in litellm_stream_completion:
        chunk_dict = chunk.model_dump()
        if (
            'object' not in chunk_dict
            or chunk_dict['object'] != 'chat.completion.chunk'
        ):
            chunk_dict['object'] = 'chat.completion.chunk'
        yield OpenAIChatCompletionChunk(**chunk_dict)
    logger.debug('Finished processing LiteLLM stream.')

async def call_litellm_openai_chat_completions(
    request_data: OpenAIChatCompletionRequest,
) -> Union[OpenAIChatCompletionResponse, AsyncGenerator[OpenAIChatCompletionChunk, None]]:
    payload = request_data.model_dump(exclude_none=True)
    initial_model_in_payload = payload.get('model')

    final_target_model, api_key_to_use = _resolve_provider_details(
        initial_model_in_payload, 
        settings.TARGET_LLM_PROVIDER
    )
    
    payload['model'] = final_target_model

    logger.info(
        f'Sending request to LiteLLM: Model: {payload.get('model')}'
    )
    try:
        if request_data.stream:
            logger.debug(f"Initiating LiteLLM stream request for model: {payload.get('model')}")
        else:
            logger.debug(f"Initiating LiteLLM non-stream request for model: {payload.get('model')}")
        response = await litellm.acompletion(**payload, api_key=api_key_to_use)
        
        if not request_data.stream:
            return _process_non_stream_response(response)
        else:
            return _process_stream_response(response)
            
    except litellm.exceptions.APIError as e:
        logger.error(
            f'LiteLLM APIError: Status Code: {e.status_code}, Message: {e.message}, LLM Provider: {getattr(e, "llm_provider", "N/A")}'
        )
        raise Exception(
            f'LiteLLM API Error from {getattr(e, "llm_provider", "N/A")}: {e.message} (Status: {e.status_code})'
        ) from e
    except Exception as e:
        logger.exception('An unexpected error occurred while calling LiteLLM.')
        raise Exception(f'Unexpected error during LiteLLM operation: {str(e)}') from e 