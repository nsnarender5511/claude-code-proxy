from loguru import logger
import json # For potential JSONResponse if error occurs before orchestrator
from typing import AsyncGenerator, Dict # For error response type hint

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from src.api.v1.schemas.anthropic_api import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse
) # Updated import
from src.services.message_flow_orchestrator import orchestrate_message_proxy # Updated import
from src.services.error_translator_service import translate_openai_error_to_anthropic_format # For immediate errors

router = APIRouter()

@router.post("/messages", response_model=None) # response_model=None as orchestrator handles response types
async def messages_endpoint(request: Request, anthropic_request: AnthropicMessagesRequest):
    logger.info(
        f"API Endpoint: Received Anthropic messages request. Model: {anthropic_request.model}, Stream: {anthropic_request.stream}"
    )
    logger.debug(
        f"API Endpoint: Anthropic request payload: {anthropic_request.model_dump_json(exclude_none=True)}"
    )

    try:
        orchestrator_response_or_stream = await orchestrate_message_proxy(anthropic_request)

        if anthropic_request.stream:
            if isinstance(orchestrator_response_or_stream, AsyncGenerator):
                logger.info("API Endpoint: Returning SSE stream from orchestrator.")
                return StreamingResponse(orchestrator_response_or_stream, media_type="text/event-stream")
            elif isinstance(orchestrator_response_or_stream, dict) and 'error' in orchestrator_response_or_stream.get('type', ''):
                # Stream was requested, but orchestrator returned a JSON error (e.g., setup failed before streaming could start)
                logger.error(f"API Endpoint: Orchestrator returned JSON error for a stream request: {orchestrator_response_or_stream}")
                status_code = 500 # Default for pre-stream errors
                return JSONResponse(status_code=status_code, content=orchestrator_response_or_stream)
            else:
                logger.error("API Endpoint: Stream requested, but orchestrator returned unexpected non-streamable, non-error type.")
                final_error = translate_openai_error_to_anthropic_format({
                    "error": {"type": "internal_server_error", "message": "Internal error processing stream request."}
                })
                return JSONResponse(status_code=500, content=final_error)
        else: # Non-streaming
            if isinstance(orchestrator_response_or_stream, AnthropicMessagesResponse):
                logger.info("API Endpoint: Returning non-stream JSON response from orchestrator.")
                return JSONResponse(content=orchestrator_response_or_stream.model_dump(exclude_none=True))
            elif isinstance(orchestrator_response_or_stream, dict) and 'error' in orchestrator_response_or_stream.get('type', ''):
                logger.error(f"API Endpoint: Orchestrator returned JSON error for non-stream request: {orchestrator_response_or_stream}")
                status_code = 500 # Default
                # Example: if orchestrator_response_or_stream.get('error',{}).get('type') == 'invalid_request_error': status_code = 400
                # A more robust way to get status_code from the error object is needed.
                # For now, using a placeholder or relying on how `translate_openai_error_to_anthropic_format` might hint it.
                # The orchestrator should ideally return (status_code, error_payload) for JSON errors.
                # Assuming orchestrator returns a dict that IS the Anthropic error format.
                # The status code determination needs to be more robust based on error type.
                # Here, we are directly returning the error dict from the orchestrator.
                # It would be better if orchestrator errors were exceptions caught here, or a tuple (status, content)
                
                error_status_map = {
                    "authentication_error": 401,
                    "permission_error": 403,
                    "invalid_request_error": 400,
                    "not_found_error": 404,
                    "rate_limit_error": 429,
                    "api_connection_error": 503,
                    "overloaded_error": 503, # Added based on previous if/elif
                }
                err_type = orchestrator_response_or_stream.get("error", {}).get("type")
                final_status_code = error_status_map.get(err_type, 500) # Default to 500
                
                return JSONResponse(status_code=final_status_code, content=orchestrator_response_or_stream)
            else:
                logger.error("API Endpoint: Non-stream requested, but orchestrator returned unexpected, non-JSON-error type.")
                final_error = translate_openai_error_to_anthropic_format({
                    "error": {"type": "internal_server_error", "message": "Internal error processing non-stream request."}
                })
                return JSONResponse(status_code=500, content=final_error)

    except Exception as e:
        logger.exception("API Endpoint: Unhandled exception in /messages endpoint:")
        # This is a fallback for truly unexpected errors in the endpoint itself.
        final_error = translate_openai_error_to_anthropic_format({
            "error": {"type": "internal_server_error", "message": f"Unexpected API error: {str(e)}"}
        })
        return JSONResponse(status_code=500, content=final_error) 