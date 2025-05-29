import json  # For potential JSONResponse if error occurs before orchestrator
from typing import AsyncGenerator, Dict  # For error response type hint

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from src.api.v1.schemas.anthropic_api import (  # Updated import
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from src.services.error_translator_service import (
    translate_openai_error_to_anthropic_format,
)  # For immediate errors
from src.services.message_flow_orchestrator import orchestrate_message_proxy  # Updated import

router = APIRouter()


@router.post(
    "/messages", response_model=None
)  # response_model=None as orchestrator handles response types
async def messages_endpoint(request: Request, anthropic_request: AnthropicMessagesRequest):
    logger.debug(
        f"API Endpoint: Received Anthropic messages request. Model: {anthropic_request.model}"
    )
    try:
        orchestrator_response_or_stream = await orchestrate_message_proxy(anthropic_request)

        if anthropic_request.stream:
            if isinstance(orchestrator_response_or_stream, AsyncGenerator):
                logger.debug("API Endpoint: Returning SSE stream from orchestrator.")
                return StreamingResponse(
                    orchestrator_response_or_stream, media_type="text/event-stream"
                )
            else:
                logger.error(
                    "API Endpoint: Stream requested, but orchestrator returned unexpected non-streamable, non-error type."
                )
                final_error = translate_openai_error_to_anthropic_format(
                    {
                        "error": {
                            "type": "internal_server_error",
                            "message": "Internal error processing stream request.",
                        }
                    }
                )
                return JSONResponse(status_code=500, content=final_error)
        else:  # Non-streaming
            if isinstance(orchestrator_response_or_stream, AnthropicMessagesResponse):
                logger.debug("API Endpoint: Returning non-stream JSON response from orchestrator.")
                return JSONResponse(
                    content=orchestrator_response_or_stream.model_dump(exclude_none=True)
                )
            else:
                logger.error(
                    "API Endpoint: Non-stream requested, but orchestrator returned unexpected, non-JSON-error type."
                )
                final_error = translate_openai_error_to_anthropic_format(
                    {
                        "error": {
                            "type": "internal_server_error",
                            "message": "Internal error processing non-stream request.",
                        }
                    }
                )
                return JSONResponse(status_code=500, content=final_error)

    except Exception as e:
        logger.exception("API Endpoint: Unhandled exception in /messages endpoint:")
        # This is a fallback for truly unexpected errors in the endpoint itself.
        final_error = translate_openai_error_to_anthropic_format(
            {
                "error": {
                    "type": "internal_server_error",
                    "message": f"Unexpected API error: {str(e)}",
                }
            }
        )
        return JSONResponse(status_code=500, content=final_error)
