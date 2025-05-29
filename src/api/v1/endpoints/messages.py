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
    target_model_name = "unknown_target_model" # Default
    try:
        # The orchestrate_message_proxy now returns a tuple
        orchestrator_result, target_model_name_from_orchestrator = await orchestrate_message_proxy(
            anthropic_request
        )
        target_model_name = target_model_name_from_orchestrator # Update with actual model name

        # Construct the new log message
        tool_count = len(anthropic_request.tools) if anthropic_request.tools else 0
        message_count = len(anthropic_request.messages)
        
        log_message_content = f"{anthropic_request.model} → {target_model_name} {tool_count} tools {message_count} messages"
        # The actual response handling will determine if it's an error or success for color, but the content is fixed.
        # We will log this message after response handling, but prepare it here.

        if anthropic_request.stream:
            # Ensure orchestrator_result is the async generator part of the tuple for streaming
            if isinstance(orchestrator_result, AsyncGenerator):
                logger.debug("API Endpoint: Returning SSE stream from orchestrator.")
                # Log *before* returning the response
                logger.info(log_message_content)
                return StreamingResponse(
                    orchestrator_result, media_type="text/event-stream"
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
                # Log error (though the content will be the same, the color might differ if we had level-based prefixes)
                logger.error(f"{log_message_content} - STREAM ERROR OCCURRED") # Or just log_message_content with error level
                return JSONResponse(status_code=500, content=final_error)
        elif isinstance(orchestrator_result, AnthropicMessagesResponse):
            logger.debug(
                "API Endpoint: Returning non-stream JSON response from orchestrator."
            )
            # Log *before* returning the response
            logger.info(log_message_content)
            return JSONResponse(
                content=orchestrator_result.model_dump(exclude_none=True)
            )
        # This handles cases where orchestrator_result is an error dictionary
        elif isinstance(orchestrator_result, dict) and "error" in orchestrator_result:
            logger.error(f"API Endpoint: Orchestrator returned an error structure. Logging: {log_message_content} - ORCHESTRATOR ERROR")
            # Assuming orchestrator_result is already in Anthropic error format if it's a dict with 'error'
            return JSONResponse(status_code=orchestrator_result.get("status_code", 500), content=orchestrator_result) # status_code might not be in all error dicts

        else: # Fallback for unexpected types from orchestrator after tuple unpacking
            logger.error(
                f"API Endpoint: Orchestrator returned unexpected type. Logging: {log_message_content} - UNEXPECTED ORCHESTRATOR RESPONSE TYPE"
            )
            final_error = translate_openai_error_to_anthropic_format(
                {
                    "error": {
                        "type": "internal_server_error",
                        "message": "Internal error processing request due to unexpected orchestrator response.",
                    }
                }
            )
            return JSONResponse(status_code=500, content=final_error)
    except Exception as e:
        # Log the constructed message even in case of an exception
        tool_count = len(anthropic_request.tools) if anthropic_request.tools else 0
        message_count = len(anthropic_request.messages)
        # Use the target_model_name if it was resolved before the exception, otherwise the default
        exception_log_message = f"{anthropic_request.model} → {target_model_name} {tool_count} tools {message_count} messages - EXCEPTION: {str(e)}"
        logger.exception(exception_log_message) # logger.exception includes stack trace and uses ERROR level
        
        final_error = translate_openai_error_to_anthropic_format(
            {
                "error": {
                    "type": "internal_server_error",
                    "message": f"Unexpected API error: {str(e)}",
                }
            }
        )
        return JSONResponse(status_code=500, content=final_error)
