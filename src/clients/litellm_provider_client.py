import litellm
import logging
import asyncio
from typing import AsyncGenerator, Union, Optional, Any
from src.core.config import settings
from src.models.openai_provider_models import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChunk,
)

logger = logging.getLogger(__name__)


def _resolve_provider_details(
    initial_model_in_payload: Optional[str], target_llm_provider_setting: str
) -> tuple[Optional[str], Optional[str]]:
    target_model_to_use = initial_model_in_payload
    api_key_to_use = None
    if target_llm_provider_setting == "openai":
        if not initial_model_in_payload:
            logger.error("No model provided in payload for OpenAI provider.")
            return None, None
        translated_model = settings.ANTHROPIC_TO_OPENAI_MAP.get(
            initial_model_in_payload
        )
        if not translated_model:
            logger.error(
                f"Could not translate Anthropic model '{initial_model_in_payload}' to an OpenAI model. It's not found in ANTHROPIC_TO_OPENAI_MAP."
            )
            return None, None
        target_model_to_use = translated_model
        api_key_to_use = settings.OPENAI_API_KEY
        if not api_key_to_use:
            logger.error(
                "OpenAI is the target provider, but OPENAI_API_KEY is not set."
            )
            return None, None
        logger.debug(
            f"Resolved to OpenAI. Original model '{initial_model_in_payload}', translated to '{target_model_to_use}'."
        )
    elif target_llm_provider_setting == "gemini":
        if not initial_model_in_payload:
            logger.error("No model provided in payload for Gemini provider.")
            return None, None
        translated_model = settings.ANTHROPIC_TO_GEMINI_MAP.get(
            initial_model_in_payload
        )
        if not translated_model:
            logger.error(
                f"Could not translate Anthropic model '{initial_model_in_payload}' to a Gemini model. It's not found in ANTHROPIC_TO_GEMINI_MAP."
            )
            return None, None
        target_model_to_use = translated_model
        api_key_to_use = settings.GEMINI_API_KEY
        if not api_key_to_use:
            logger.error(
                "Gemini is the target provider, but GEMINI_API_KEY is not set."
            )
            return None, None
        logger.debug(
            f"Resolved to Gemini. Original model '{initial_model_in_payload}', translated to '{target_model_to_use}'."
        )
    elif target_llm_provider_setting == "anthropic":
        api_key_to_use = settings.ANTHROPIC_API_KEY
        if not api_key_to_use:
            logger.error(
                "Anthropic is the target provider, but ANTHROPIC_API_KEY is not set."
            )
            return None, None
        logger.debug(f"Resolved to Anthropic. Using model '{target_model_to_use}'.")
    else:
        logger.warning(
            f"TARGET_LLM_PROVIDER '{target_llm_provider_setting}' not explicitly handled for model/key override. Using original model '{target_model_to_use}' and relying on global LiteLLM key config."
        )
    return (target_model_to_use, api_key_to_use)


def _process_non_stream_response(
    litellm_completion: Any,
) -> OpenAIChatCompletionResponse:
    logger.debug("Processing LiteLLM non-stream response.")
    response_dict = litellm_completion.model_dump()
    if "object" not in response_dict or response_dict["object"] != "chat.completion":
        response_dict["object"] = "chat.completion"
    return OpenAIChatCompletionResponse(**response_dict)


async def _process_stream_response(
    litellm_stream_completion: Any,
) -> AsyncGenerator[OpenAIChatCompletionChunk, None]:
    logger.debug("Processing LiteLLM stream response.")
    async for chunk in litellm_stream_completion:
        chunk_dict = chunk.model_dump()
        if (
            "object" not in chunk_dict
            or chunk_dict["object"] != "chat.completion.chunk"
        ):
            chunk_dict["object"] = "chat.completion.chunk"
        yield OpenAIChatCompletionChunk(**chunk_dict)
    logger.debug("Finished processing LiteLLM stream.")


async def call_litellm_openai_chat_completions(
    request_data: OpenAIChatCompletionRequest,
) -> Union[
    tuple[OpenAIChatCompletionResponse, str],
    tuple[AsyncGenerator[OpenAIChatCompletionChunk, None], str],
]:
    payload = request_data.model_dump(exclude_none=True)
    initial_model_in_payload = payload.get("model")
    final_target_model, api_key_to_use = _resolve_provider_details(
        initial_model_in_payload, settings.TARGET_LLM_PROVIDER
    )
    if final_target_model is None or api_key_to_use is None:
        error_message = f"Failed to resolve provider details for model: {initial_model_in_payload} with provider: {settings.TARGET_LLM_PROVIDER}"
        logger.error(error_message)
        raise Exception(error_message)

    payload["model"] = final_target_model
    try:
        if request_data.stream:
            logger.debug(
                f"Initiating LiteLLM stream request for model: {payload.get('model')}"
            )
        else:
            logger.debug(
                f"Initiating LiteLLM non-stream request for model: {payload.get('model')}"
            )
        response = await litellm.acompletion(
            **payload, api_key=api_key_to_use, no_logs=True
        )
        if not request_data.stream:
            return _process_non_stream_response(response), final_target_model
        else:
            return _process_stream_response(response), final_target_model
    except litellm.exceptions.APIError as e:
        logger.error(
            f"LiteLLM APIError: Status Code: {e.status_code}, Message: {e.message}, LLM Provider: {getattr(e, 'llm_provider', 'N/A')}",
            exc_info=True,
        )
        raise Exception(
            f"LiteLLM API Error from {getattr(e, 'llm_provider', 'N/A')}: {e.message} (Status: {e.status_code})"
        ) from e
    except Exception as e:
        logger.exception("An unexpected error occurred while calling LiteLLM.")
        raise Exception(f"Unexpected error during LiteLLM operation: {str(e)}") from e
