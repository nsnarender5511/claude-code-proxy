import litellm # Changed from httpx
import logging
import json # For parsing error responses if necessary
from typing import AsyncGenerator, Union
from src.core.config import settings # Keep settings for TARGET_LLM_PROVIDER if used for logic
from src.api.models import OpenAIChatCompletionRequest, OpenAIChatCompletionResponse, OpenAIChatCompletionChunk
from src.services.model_translator import ModelTranslationService # Added import

logger = logging.getLogger(__name__)
model_translator = ModelTranslationService() # Instantiate the translator

async def call_litellm_openai_chat_completions( # Renamed function for clarity
    request_data: OpenAIChatCompletionRequest
) -> Union[OpenAIChatCompletionResponse, AsyncGenerator[OpenAIChatCompletionChunk, None]]:
    """
    Calls LiteLLM directly for chat completions.

    Args:
        request_data: The OpenAIChatCompletionRequest Pydantic model.

    Returns:
        If not streaming: OpenAIChatCompletionResponse Pydantic model.
        If streaming: An async generator yielding OpenAIChatCompletionChunk Pydantic models.
    
    Raises:
        litellm.exceptions.APIError: For API-related errors from LiteLLM.
        Exception: For other unexpected errors.
    """
    payload = request_data.model_dump(exclude_none=True)
    
    target_model = payload.get("model") # Original model from request
    api_key_to_use = None

    if settings.TARGET_LLM_PROVIDER == "openai":
        translated_model = model_translator.get_openai_equivalent(settings.OPENAI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY)
        if not translated_model:
            logger.error(f"Could not translate default OpenAI model key: {settings.OPENAI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY}")
            raise ValueError("Default OpenAI model key not found in translation map.")
        target_model = translated_model
        api_key_to_use = settings.OPENAI_API_KEY
        if not api_key_to_use:
            logger.error("OpenAI is the target provider, but OPENAI_API_KEY is not set.")
            raise ValueError("OPENAI_API_KEY is not configured for the target provider 'openai'.")
        logger.info(f"Routing to OpenAI. Overriding model to '{target_model}'.")
    elif settings.TARGET_LLM_PROVIDER == "gemini":
        translated_model = model_translator.get_gemini_equivalent(settings.GEMINI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY)
        if not translated_model:
            logger.error(f"Could not translate default Gemini model key: {settings.GEMINI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY}")
            raise ValueError("Default Gemini model key not found in translation map.")
        target_model = translated_model
        api_key_to_use = settings.GEMINI_API_KEY
        if not api_key_to_use:
            logger.error("Gemini is the target provider, but GEMINI_API_KEY is not set.")
            raise ValueError("GEMINI_API_KEY is not configured for the target provider 'gemini'.")
        logger.info(f"Routing to Gemini. Overriding model to '{target_model}'.")
    elif settings.TARGET_LLM_PROVIDER == "anthropic":
        # Keep original model if provider is anthropic, ensure key is set
        # The 'target_model' here is the original model from the request payload.
        api_key_to_use = settings.ANTHROPIC_API_KEY
        if not api_key_to_use:
            logger.error("Anthropic is the target provider, but ANTHROPIC_API_KEY is not set.")
            raise ValueError("ANTHROPIC_API_KEY is not configured for the target provider 'anthropic'.")
        logger.info(f"Routing to Anthropic. Using model '{target_model}'.")
    else:
        # Fallback: use original model and hope LiteLLM is configured globally for it
        # Or, raise an error if the provider is unsupported by this explicit logic.
        logger.warning(
            f"TARGET_LLM_PROVIDER '{settings.TARGET_LLM_PROVIDER}' not explicitly handled for model/key override. "
            f"Using original model '{target_model}' and relying on global LiteLLM key config."
        )
        # If you want to be strict:
        # raise ValueError(f"Unsupported TARGET_LLM_PROVIDER: {settings.TARGET_LLM_PROVIDER}")

    payload["model"] = target_model # Update payload with potentially overridden model

    logger.debug(
        f"Sending request to LiteLLM: Model: {payload.get('model')}, Stream: {payload.get('stream')}, "
        f"Target Provider Setting: {settings.TARGET_LLM_PROVIDER}, API Key Provided: {bool(api_key_to_use)}"
    )

    try:
        # Pass the selected API key to LiteLLM
        response = await litellm.acompletion(**payload, api_key=api_key_to_use)

        if not request_data.stream:
            # response is a litellm.ModelResponse object
            logger.info("LiteLLM non-stream request successful.")
            # LiteLLM's ModelResponse is designed to be OpenAI-compatible.
            # We convert it to a dictionary and then to our Pydantic model.
            response_dict = response.model_dump()
            
            # Ensure 'object' field is correctly set for our Pydantic model if not present or different
            if 'object' not in response_dict or response_dict['object'] != "chat.completion":
                response_dict['object'] = "chat.completion"
            
            return OpenAIChatCompletionResponse(**response_dict)
        else:
            # response is an async generator from LiteLLM when stream=True
            logger.info("LiteLLM stream request initiated.")
            
            async def stream_generator() -> AsyncGenerator[OpenAIChatCompletionChunk, None]:
                async for chunk in response:
                    # chunk is a litellm.ModelResponse (StreamingChoices) object
                    chunk_dict = chunk.model_dump()
                    
                    # Ensure 'object' field is correctly set for our Pydantic model
                    if 'object' not in chunk_dict or chunk_dict['object'] != "chat.completion.chunk":
                         chunk_dict['object'] = "chat.completion.chunk"
                    
                    # Ensure choices delta content is not None before creating OpenAIChatCompletionChunk
                    # This handles potential empty delta content in some chunks.
                    # LiteLLM chunks should generally be well-formed.
                    # Example of more detailed mapping if LiteLLM chunk structure differs significantly:
                    # mapped_choices = []
                    # for choice_chunk in chunk_dict.get("choices", []):
                    #    delta = choice_chunk.get("delta", {})
                    #    mapped_choices.append({
                    #        "delta": {
                    #            "role": delta.get("role"),
                    #            "content": delta.get("content"),
                    #            "tool_calls": delta.get("tool_calls")
                    #        },
                    #        "finish_reason": choice_chunk.get("finish_reason"),
                    #        "index": choice_chunk.get("index")
                    #    })
                    # final_chunk_data = {
                    #     "id": chunk_dict.get("id"),
                    #     "choices": mapped_choices,
                    #     "created": chunk_dict.get("created"),
                    #     "model": chunk_dict.get("model"),
                    #     "object": "chat.completion.chunk",
                    #     "system_fingerprint": chunk_dict.get("system_fingerprint")
                    # }
                    # yield OpenAIChatCompletionChunk(**final_chunk_data)
                    yield OpenAIChatCompletionChunk(**chunk_dict)
                logger.info("LiteLLM stream finished.")
            return stream_generator()

    except litellm.exceptions.APIError as e:
        logger.error(f"LiteLLM APIError: Status Code: {e.status_code}, Message: {e.message}, LLM Provider: {getattr(e, 'llm_provider', 'N/A')}")
        # It's important to translate this error into a format the calling endpoint expects.
        # For now, raising a generic exception. The endpoint should handle this.
        # Consider creating a custom application exception.
        raise Exception(f"LiteLLM API Error from {getattr(e, 'llm_provider', 'N/A')}: {e.message} (Status: {e.status_code})") from e
    except Exception as e:
        # Catch any other unexpected errors during LiteLLM call
        logger.exception("An unexpected error occurred while calling LiteLLM.")
        raise Exception(f"Unexpected error during LiteLLM operation: {str(e)}") from e
