import logging
import json # For parsing raw_request if needed, though might be simplified

from fastapi import APIRouter, Request, HTTPException
import litellm

# Models
from src.api.models import TokenCountRequest, TokenCountResponse, MessagesRequest

# Services
from src.services.translation import convert_anthropic_to_litellm

# Utils
from src.utils.beautiful_log import log_request_beautifully

# Typing (Any is not used in this file)
# from typing import Any # Example, adjust as needed

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest, 
    raw_request: Request 
):
    try:
        original_model_for_logging = request.original_model or request.model
        
        display_model_for_log = original_model_for_logging
        if "/" in display_model_for_log:
            display_model_for_log = display_model_for_log.split("/")[-1]
        
        logger.debug(f"📊 COUNT TOKENS REQUEST: Original Model='{original_model_for_logging}', Validated Model by Pydantic='{request.model}'")

        # Convert TokenCountRequest to a temporary MessagesRequest for consistent conversion
        # The model name in TokenCountRequest is already validated by its own Pydantic validator.
        # We pass this validated model (request.model) to MessagesRequest for conversion.
        # The original_model field in MessagesRequest will be populated by its validator if mapping occurs based on request.model.
        temp_messages_request = MessagesRequest(
            model=request.model, # Pass the already Pydantic-validated model from TokenCountRequest
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            tool_choice=request.tool_choice, # Pass tool_choice if present
            max_tokens=1 # Dummy value, not used by converter for token counting message prep
        )
        
        # convert_anthropic_to_litellm will use the model from temp_messages_request (which is request.model here)
        # and its internal logic will handle any further mapping if defined (e.g. adding prefixes)
        litellm_params = convert_anthropic_to_litellm(temp_messages_request)
        
        # Use LiteLLM's token_counter function
        try:
            from litellm import token_counter
            
            # Log before calling token_counter
            num_tools_for_log = len(litellm_params.get("tools", [])) if litellm_params.get("tools") else 0
            log_request_beautifully(
                "POST",
                str(raw_request.url.path),
                display_model_for_log, # Original model for display
                litellm_params["model"], # Model LiteLLM will use for counting (potentially re-mapped by convert_anthropic_to_litellm)
                len(litellm_params["messages"]),
                num_tools_for_log,
                200 
            )
            
            token_count = token_counter(
                model=litellm_params["model"],
                messages=litellm_params["messages"],
                tools=litellm_params.get("tools") # Pass tools if token_counter supports it and they are in litellm_params
            )
            
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("Could not import token_counter from litellm. Please ensure LiteLLM is up to date.")
            raise HTTPException(status_code=501, detail="Token counting feature is currently unavailable due to a server configuration issue.")
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while counting tokens: {str(e)}")
