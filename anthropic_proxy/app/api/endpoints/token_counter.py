import logging
import json # For parsing raw_request if needed, though might be simplified

from fastapi import APIRouter, Request, HTTPException
import litellm

# Models
from app.api.models import TokenCountRequest, TokenCountResponse, MessagesRequest

# Services
from app.services.translation import convert_anthropic_to_litellm

# Utils
from app.utils.beautiful_log import log_request_beautifully

# Typing (Any is not used in this file)
# from typing import Any # Example, adjust as needed

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest, # Pydantic model for the request body
    raw_request: Request       # Raw FastAPI request object, for logging original model
):
    try:
        # Log the incoming token count request
        # We need raw_request to get the original model name if it was mapped by Pydantic
        # If Pydantic model 'request.original_model' is reliable, raw_request might not be needed here.
        # For now, assume it might be useful for logging or direct body access if required.
        
        original_model_for_logging = request.original_model or request.model # Prefer original_model from Pydantic
        
        display_model = original_model_for_logging
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        logger.debug(f"📊 COUNT TOKENS REQUEST: Original Model='{original_model_for_logging}', Validated Model='{request.model}'")

        # Convert the messages to a format LiteLLM can understand
        # The TokenCountRequest model is similar to MessagesRequest, so we adapt it
        # We need to construct a MessagesRequest-like object for convert_anthropic_to_litellm
        messages_request_for_conversion = MessagesRequest(
            model=request.model, # Pass the (potentially mapped) model
            max_tokens=1,  # Arbitrary, not used by token_counter
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            tool_choice=request.tool_choice,
            thinking=request.thinking,
            original_model=request.original_model # Ensure original_model is also passed
        )
        
        litellm_convertable_request = convert_anthropic_to_litellm(messages_request_for_conversion)
        
        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function - ensure litellm is updated if this fails
            from litellm import token_counter
            
            num_tools = len(request.tools) if request.tools else 0
            log_request_beautifully(
                "POST",
                str(raw_request.url.path), # Ensure path is string
                display_model,
                litellm_convertable_request.get('model'), # Log the model LiteLLM will use
                len(litellm_convertable_request.get('messages', [])),
                num_tools,
                200  # Assuming success at this point for logging
            )
            
            token_count = token_counter(
                model=litellm_convertable_request["model"],
                messages=litellm_convertable_request["messages"],
                # LiteLLM's token_counter might not support 'tools' or other params directly.
                # If it does, they should be passed from litellm_convertable_request.
            )
            
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("Could not import token_counter from litellm. Please ensure LiteLLM is up to date.")
            # Fallback to a simple approximation or error
            # For now, raising an error might be better than returning potentially incorrect data
            raise HTTPException(status_code=501, detail="Token counting feature is currently unavailable due to a server configuration issue.")
            # return TokenCountResponse(input_tokens=1000) # Default fallback if we choose to provide one
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        # Provide a generic error response
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while counting tokens: {str(e)}")
