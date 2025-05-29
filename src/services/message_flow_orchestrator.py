import logging
import json
import httpx
from typing import Union, AsyncGenerator, Dict, Any
from src.api.v1.schemas.anthropic_api import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from src.models.openai_provider_models import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChunk,
)
from src.services.request_translator_service import (
    translate_anthropic_to_openai_request,
)
from src.services.response_translator_service import (
    translate_openai_to_anthropic_response,
)
from src.services.anthropic_sse_builder_service import build_anthropic_sse_stream
from src.services.error_translator_service import (
    translate_openai_error_to_anthropic_format,
    generate_anthropic_error_sse_event,
)
from src.clients.litellm_provider_client import call_litellm_openai_chat_completions
from src.core.config import settings

logger = logging.getLogger(__name__)


async def _handle_streaming_response(
    litellm_sse_generator: AsyncGenerator[OpenAIChatCompletionChunk, None],
    anthropic_request_model: str,
) -> AsyncGenerator[str, None]:
    logger.debug("Orchestrator: Processing stream from LiteLLM via SSE builder.")
    first_chunk_id = "fallback_id_from_orchestrator"
    processed_litellm_stream: AsyncGenerator[OpenAIChatCompletionChunk, None]
    temp_litellm_iter = litellm_sse_generator.__aiter__()
    try:
        first_chunk: OpenAIChatCompletionChunk = await temp_litellm_iter.__anext__()
        if first_chunk and first_chunk.id:
            first_chunk_id = first_chunk.id

        async def combined_stream() -> AsyncGenerator[OpenAIChatCompletionChunk, None]:
            yield first_chunk
            async for chunk in temp_litellm_iter:
                yield chunk

        processed_litellm_stream = combined_stream()
    except StopAsyncIteration:
        logger.warning("Orchestrator: LiteLLM stream was empty.")

        async def empty_stream() -> AsyncGenerator[OpenAIChatCompletionChunk, None]:
            if False:
                yield

        processed_litellm_stream = empty_stream()
    
    try:
        async for sse_event_str in build_anthropic_sse_stream(
            processed_litellm_stream, anthropic_request_model, first_chunk_id
        ):
            yield sse_event_str
    except httpx.HTTPStatusError as e_stream_http:
        logger.error(
            f"Orchestrator Stream Handler: HTTPStatusError during SSE stream processing: {e_stream_http.response.status_code} - {e_stream_http.response.text}",
            exc_info=True,
        )
        try:
            error_payload = e_stream_http.response.json()
        except json.JSONDecodeError:
            error_payload = {
                "error": {
                    "type": "internal_server_error",
                    "message": e_stream_http.response.text,
                }
            }
        anthropic_fmt_error = translate_openai_error_to_anthropic_format(error_payload)
        async for error_event in generate_anthropic_error_sse_event(
            error_type_str=anthropic_fmt_error.get("error", {}).get(
                "type", "api_error"
            ),
            message=anthropic_fmt_error.get("error", {}).get(
                "message", "Streaming error."
            ),
        ):
            yield error_event
    except Exception as e_stream_generic:
        logger.exception(
            "Orchestrator Stream Handler: Exception during SSE stream processing by builder:"
        )
        async for error_event in generate_anthropic_error_sse_event(
            error_type_str="internal_server_error",
            message=f"Unexpected error during streaming orchestration: {str(e_stream_generic)}",
        ):
            yield error_event


async def orchestrate_message_proxy(
    anthropic_request: AnthropicMessagesRequest,
) -> Union[tuple[AnthropicMessagesResponse, str], tuple[AsyncGenerator[str, None], str], tuple[Dict[str, Any], str]]:
    logger.debug(
        f"Orchestrating message proxy. Model: {anthropic_request.model}, Stream: {anthropic_request.stream}"
    )
    target_model_name_for_logging = "unknown_target_model"
    try:
        openai_request: OpenAIChatCompletionRequest = (
            translate_anthropic_to_openai_request(anthropic_request)
        )
        logger.debug(
            f"Orchestrator: Translated to OpenAI. Model: {openai_request.model}, Messages count: {len(openai_request.messages)}, Stream: {openai_request.stream}"
        )
        litellm_response_or_generator, final_target_model = await call_litellm_openai_chat_completions(
            openai_request
        )
        target_model_name_for_logging = final_target_model

        if anthropic_request.stream:
            if not hasattr(litellm_response_or_generator, "__aiter__"):
                logger.error(
                    "Orchestrator: Expected an async generator for streaming, but did not receive one."
                )
                error_response = translate_openai_error_to_anthropic_format(
                    {
                        "error": {
                            "type": "internal_server_error",
                            "message": "Streaming requested but upstream did not provide a stream.",
                        }
                    }
                )
                return error_response, target_model_name_for_logging
            
            litellm_sse_generator = litellm_response_or_generator
            sse_event_generator = _handle_streaming_response(
                litellm_sse_generator, anthropic_request.model 
            )
            return sse_event_generator, target_model_name_for_logging
        else:
            if not isinstance(
                litellm_response_or_generator, OpenAIChatCompletionResponse
            ):
                logger.error(
                    "Orchestrator: Expected OpenAIChatCompletionResponse for non-streaming, got something else."
                )
                error_response = translate_openai_error_to_anthropic_format(
                    {
                        "error": {
                            "type": "internal_server_error",
                            "message": "Non-streaming request failed to return valid upstream response.",
                        }
                    }
                )
                return error_response, target_model_name_for_logging
            logger.debug("Orchestrator: Processing non-stream response from LiteLLM.")
            openai_response: OpenAIChatCompletionResponse = (
                litellm_response_or_generator
            )
            logger.debug(
                f"Orchestrator: Received OpenAI response. ID: {openai_response.id}, Model: {openai_response.model}, Choices: {len(openai_response.choices)}"
            )
            final_anthropic_response: AnthropicMessagesResponse = (
                translate_openai_to_anthropic_response(openai_response)
            )
            logger.debug(
                f"Orchestrator: Translated to Anthropic response. ID: {final_anthropic_response.id}, Model: {final_anthropic_response.model}, Content blocks: {len(final_anthropic_response.content)}"
            )
            return final_anthropic_response, target_model_name_for_logging
    except httpx.HTTPStatusError as e_http:
        logger.error(
            f"Orchestrator: HTTPStatusError from LiteLLM client: {e_http.response.status_code} - {e_http.response.text}",
            exc_info=True,
        )
        try:
            error_payload = e_http.response.json()
        except json.JSONDecodeError:
            error_payload = {
                "error": {"type": "upstream_error", "message": e_http.response.text}
            }
        return translate_openai_error_to_anthropic_format(error_payload), target_model_name_for_logging
    except httpx.RequestError as e_req:
        logger.error(
            f"Orchestrator: RequestError connecting to LiteLLM client: {str(e_req)}",
            exc_info=True,
        )
        return translate_openai_error_to_anthropic_format(
            {
                "error": {
                    "type": "connection_error",
                    "message": f"Could not connect to LiteLLM proxy: {str(e_req)}",
                }
            }
        ), target_model_name_for_logging
    except ValueError as e_val:
        logger.error(
            f"Orchestrator: ValueError during processing: {str(e_val)}", exc_info=True
        )
        return translate_openai_error_to_anthropic_format(
            {"error": {"type": "invalid_request_error", "message": str(e_val)}}
        ), target_model_name_for_logging
    except Exception as e_model_related:
        resolved_model_name = target_model_name_for_logging
        if "Failed to resolve provider details" in str(e_model_related):
            logger.error(f"Orchestrator: Model resolution failed: {str(e_model_related)}")
            return translate_openai_error_to_anthropic_format(
                {"error": {"type": "invalid_request_error", "message": str(e_model_related)}}
            ), resolved_model_name
        logger.exception("Orchestrator: Unhandled generic exception:")
        return translate_openai_error_to_anthropic_format(
            {
                "error": {
                    "type": "internal_server_error",
                    "message": f"Unexpected orchestration error: {str(e_model_related)}"
                }
            }
        ), resolved_model_name
