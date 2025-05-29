import json
import logging
from typing import AsyncGenerator, Dict
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from src.api.v1.schemas.anthropic_api import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from src.services.error_translator_service import (
    translate_openai_error_to_anthropic_format,
)
from src.services.message_flow_orchestrator import orchestrate_message_proxy

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/messages", response_model=None)
async def messages_endpoint(
    request: Request, anthropic_request: AnthropicMessagesRequest
):
    logger.info(
        f"#### API Endpoint: Received Anthropic messages request. Model: {anthropic_request.model}"
    )
    try:
        orchestrator_response_or_stream = await orchestrate_message_proxy(
            anthropic_request
        )
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
        elif isinstance(orchestrator_response_or_stream, AnthropicMessagesResponse):
            logger.debug(
                "API Endpoint: Returning non-stream JSON response from orchestrator."
            )
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
        final_error = translate_openai_error_to_anthropic_format(
            {
                "error": {
                    "type": "internal_server_error",
                    "message": f"Unexpected API error: {str(e)}",
                }
            }
        )
        return JSONResponse(status_code=500, content=final_error)
