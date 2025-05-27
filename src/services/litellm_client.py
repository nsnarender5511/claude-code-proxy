import litellm
import logging
import json
from typing import AsyncGenerator, Union
from src.core.config import settings
from src.api.models import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChunk,
)
from src.services.model_translator import ModelTranslationService

logger = logging.getLogger(__name__)
model_translator = ModelTranslationService()


async def call_litellm_openai_chat_completions(
    request_data: OpenAIChatCompletionRequest,
) -> Union[OpenAIChatCompletionResponse, AsyncGenerator[OpenAIChatCompletionChunk, None]]:
    payload = request_data.model_dump(exclude_none=True)
    target_model = payload.get('model')
    api_key_to_use = None
    if settings.TARGET_LLM_PROVIDER == 'openai':
        translated_model = model_translator.get_openai_equivalent(
            settings.OPENAI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY
        )
        if not translated_model:
            logger.error(
                f'Could not translate default OpenAI model key: {settings.OPENAI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY}'
            )
            raise ValueError('Default OpenAI model key not found in translation map.')
        target_model = translated_model
        api_key_to_use = settings.OPENAI_API_KEY
        if not api_key_to_use:
            logger.error('OpenAI is the target provider, but OPENAI_API_KEY is not set.')
            raise ValueError("OPENAI_API_KEY is not configured for the target provider 'openai'.")
        logger.info(f"Routing to OpenAI. Overriding model to '{target_model}'.")
    elif settings.TARGET_LLM_PROVIDER == 'gemini':
        translated_model = model_translator.get_gemini_equivalent(
            settings.GEMINI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY
        )
        if not translated_model:
            logger.error(
                f'Could not translate default Gemini model key: {settings.GEMINI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY}'
            )
            raise ValueError('Default Gemini model key not found in translation map.')
        target_model = translated_model
        api_key_to_use = settings.GEMINI_API_KEY
        if not api_key_to_use:
            logger.error('Gemini is the target provider, but GEMINI_API_KEY is not set.')
            raise ValueError("GEMINI_API_KEY is not configured for the target provider 'gemini'.")
        logger.info(f"Routing to Gemini. Overriding model to '{target_model}'.")
    elif settings.TARGET_LLM_PROVIDER == 'anthropic':
        api_key_to_use = settings.ANTHROPIC_API_KEY
        if not api_key_to_use:
            logger.error('Anthropic is the target provider, but ANTHROPIC_API_KEY is not set.')
            raise ValueError(
                "ANTHROPIC_API_KEY is not configured for the target provider 'anthropic'."
            )
        logger.info(f"Routing to Anthropic. Using model '{target_model}'.")
    else:
        logger.warning(
            f"TARGET_LLM_PROVIDER '{settings.TARGET_LLM_PROVIDER}' not explicitly handled for model/key override. Using original model '{target_model}' and relying on global LiteLLM key config."
        )
    payload['model'] = target_model
    logger.debug(
        f'Sending request to LiteLLM: Model: {payload.get('model')}, Stream: {payload.get('stream')}, Target Provider Setting: {settings.TARGET_LLM_PROVIDER}, API Key Provided: {bool(api_key_to_use)}'
    )
    try:
        response = await litellm.acompletion(**payload, api_key=api_key_to_use)
        if not request_data.stream:
            logger.info('LiteLLM non-stream request successful.')
            response_dict = response.model_dump()
            if 'object' not in response_dict or response_dict['object'] != 'chat.completion':
                response_dict['object'] = 'chat.completion'
            return OpenAIChatCompletionResponse(**response_dict)
        else:
            logger.info('LiteLLM stream request initiated.')

            async def stream_generator() -> AsyncGenerator[OpenAIChatCompletionChunk, None]:
                async for chunk in response:
                    chunk_dict = chunk.model_dump()
                    if (
                        'object' not in chunk_dict
                        or chunk_dict['object'] != 'chat.completion.chunk'
                    ):
                        chunk_dict['object'] = 'chat.completion.chunk'
                    yield OpenAIChatCompletionChunk(**chunk_dict)
                logger.info('LiteLLM stream finished.')

            return stream_generator()
    except litellm.exceptions.APIError as e:
        logger.error(
            f'LiteLLM APIError: Status Code: {e.status_code}, Message: {e.message}, LLM Provider: {getattr(e, 'llm_provider', 'N/A')}'
        )
        raise Exception(
            f'LiteLLM API Error from {getattr(e, 'llm_provider', 'N/A')}: {e.message} (Status: {e.status_code})'
        ) from e
    except Exception as e:
        logger.exception('An unexpected error occurred while calling LiteLLM.')
        raise Exception(f'Unexpected error during LiteLLM operation: {str(e)}') from e
