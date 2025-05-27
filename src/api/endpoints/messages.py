import logging
import json
import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from src.api.models import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChunk,
)
from src.services.anthropic_to_openai_translator import translate_anthropic_to_openai_request
from src.services.openai_to_anthropic_translator import (
    translate_openai_to_anthropic_response,
    translate_openai_to_anthropic_stream,
    translate_error_openai_to_anthropic,
    generate_anthropic_error_sse,
)
from src.services.litellm_client import (
    call_litellm_openai_chat_completions,
)  # MODIFIED Function name
from src.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


async def stream_anthropic_events(
    openai_sse_generator: httpx.Response, request_model_id: str
) -> StreamingResponse:
    try:
        async for anthropic_sse_event_str in translate_openai_to_anthropic_stream(
            openai_sse_generator, request_model_id
        ):
            yield anthropic_sse_event_str
    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTPStatusError during LiteLLM stream: {e.response.status_code} - {e.response.text}"
        )
        try:
            error_payload = e.response.json()
            anthropic_error = translate_error_openai_to_anthropic(error_payload)
        except json.JSONDecodeError:
            anthropic_error = translate_error_openai_to_anthropic(
                {"error": {"type": "internal_server_error", "message": e.response.text}}
            )

        async for error_event_str in generate_anthropic_error_sse(
            error_type_str=anthropic_error.get("error", {}).get("type", "api_error"),
            message=anthropic_error.get("error", {}).get(
                "message", "Streaming error from upstream."
            ),
        ):
            yield error_event_str

    except Exception as e:
        logger.exception("Exception during OpenAI to Anthropic SSE streaming:")
        async for error_event_str in generate_anthropic_error_sse(
            error_type_str="internal_server_error",
            message=f"An unexpected error occurred during streaming: {str(e)}",
        ):
            yield error_event_str


@router.post("/messages", response_model=None)
async def messages_endpoint(request: Request, anthropic_request: AnthropicMessagesRequest):
    logger.info(
        f"Received Anthropic messages request. Model: {anthropic_request.model}, Stream: {anthropic_request.stream}"
    )
    logger.debug(
        f"Anthropic request payload: {anthropic_request.model_dump_json(exclude_none=True)}"
    )

    try:
        # 1. Translate Anthropic request to OpenAI format
        openai_request: OpenAIChatCompletionRequest = translate_anthropic_to_openai_request(
            anthropic_request
        )
        logger.debug(
            f"Translated to OpenAI request: {openai_request.model_dump_json(exclude_none=True)}"
        )

        # 2. Call LiteLLM Proxy
        # The call_litellm_openai_chat_completions function returns either # MODIFIED Function name
        # OpenAIChatCompletionResponse (non-streaming) or an AsyncGenerator (streaming)
        litellm_response_or_generator = await call_litellm_openai_chat_completions(
            openai_request
        )  # MODIFIED Function name

        # 3. Handle response (streaming or non-streaming)
        if anthropic_request.stream:
            if not hasattr(litellm_response_or_generator, "__aiter__"):
                logger.error(
                    "Expected an async generator for streaming response from LiteLLM client, but did not receive one."
                )
                # This case should ideally be an error from litellm_client if stream=True but it couldn't stream.
                # For now, translate to an Anthropic error.
                anthropic_error = translate_error_openai_to_anthropic(
                    {
                        "error": {
                            "type": "internal_server_error",
                            "message": (
                                "Streaming was requested but upstream did not provide a stream."
                            ),
                        }
                    }
                )
                return JSONResponse(status_code=500, content=anthropic_error)

            logger.info("Processing stream from LiteLLM.")
            # translate_openai_to_anthropic_stream will yield Anthropic SSE formatted strings
            return StreamingResponse(
                stream_anthropic_events(litellm_response_or_generator, anthropic_request.model),
                media_type="text/event-stream",
            )
        else:
            if not isinstance(litellm_response_or_generator, OpenAIChatCompletionResponse):
                logger.error(
                    "Expected OpenAIChatCompletionResponse for non-streaming, but received something else."
                )
                anthropic_error = translate_error_openai_to_anthropic(
                    {
                        "error": {
                            "type": "internal_server_error",
                            "message": (
                                "Non-streaming request failed to return a valid upstream response."
                            ),
                        }
                    }
                )
                return JSONResponse(status_code=500, content=anthropic_error)

            logger.info("Processing non-stream response from LiteLLM.")
            openai_response: OpenAIChatCompletionResponse = litellm_response_or_generator
            logger.debug(
                f"Received OpenAI response from LiteLLM: {openai_response.model_dump_json(exclude_none=True)}"
            )

            # 4. Translate OpenAI response back to Anthropic format
            final_anthropic_response: AnthropicMessagesResponse = (
                translate_openai_to_anthropic_response(openai_response)
            )
            logger.debug(
                f"Translated to Anthropic response: {final_anthropic_response.model_dump_json(exclude_none=True)}"
            )
            return JSONResponse(content=final_anthropic_response.model_dump(exclude_none=True))

    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTPStatusError from LiteLLM proxy: {e.response.status_code} - {e.response.text}",
            exc_info=True,
        )
        try:
            error_payload = e.response.json()
        except json.JSONDecodeError:
            error_payload = {"error": {"type": "upstream_error", "message": e.response.text}}

        anthropic_error = translate_error_openai_to_anthropic(error_payload)
        return JSONResponse(status_code=e.response.status_code, content=anthropic_error)

    except httpx.RequestError as e:
        logger.error(f"RequestError connecting to LiteLLM proxy: {str(e)}", exc_info=True)
        anthropic_error = translate_error_openai_to_anthropic(
            {
                "error": {
                    "type": "connection_error",
                    "message": f"Could not connect to LiteLLM proxy: {str(e)}",
                }
            }
        )
        return JSONResponse(status_code=503, content=anthropic_error)  # Service Unavailable

    except ValueError as e:  # Catch ValueErrors from our translation logic if any
        logger.error(f"ValueError during processing: {str(e)}", exc_info=True)
        anthropic_error = translate_error_openai_to_anthropic(
            {"error": {"type": "invalid_request_error", "message": str(e)}}
        )
        return JSONResponse(status_code=400, content=anthropic_error)

    except Exception as e:
        logger.exception("An unexpected error occurred in /messages endpoint:")
        anthropic_error = translate_error_openai_to_anthropic(
            {
                "error": {
                    "type": "internal_server_error",
                    "message": f"An unexpected error occurred: {str(e)}",
                }
            }
        )
        return JSONResponse(status_code=500, content=anthropic_error)


@router.get("/health")
async def health_check():
    return {"status": "ok", "target_llm_provider": settings.TARGET_LLM_PROVIDER}
