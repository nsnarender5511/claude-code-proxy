import logging
import json # For potential JSONResponse if error occurs before orchestrator
from typing import AsyncGenerator, Dict, Any # For error response type hint

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from src.api.v1.schemas.anthropic_api import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse
) # Updated import
from src.services.message_flow_orchestrator import orchestrate_message_proxy # Updated import
from src.services.error_translator_service import translate_openai_error_to_anthropic_format # For immediate errors

logger = logging.getLogger(__name__)
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
                # Determine status code from error if possible, else default
                # This part needs careful thought on how orchestrator signals status for pre-stream errors
                status_code = 500 # Default
                # Example: if orchestrator_response_or_stream.get('error',{}).get('type') == 'invalid_request_error': status_code = 400
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
                final_status_code = 500 # default
                err_type = orchestrator_response_or_stream.get("error", {}).get("type")
                if err_type == "authentication_error": final_status_code = 401
                elif err_type == "permission_error": final_status_code = 403
                elif err_type == "invalid_request_error": final_status_code = 400
                elif err_type == "not_found_error": final_status_code = 404
                elif err_type == "rate_limit_error": final_status_code = 429
                elif err_type == "api_connection_error": final_status_code = 503
                elif err_type == "overloaded_error": final_status_code = 503
                
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