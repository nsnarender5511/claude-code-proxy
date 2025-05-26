import asyncio
import json
import logging
import random
import string
from typing import AsyncGenerator, Dict, List, Optional, Union, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from litellm import acompletion
from pydantic import BaseModel

from src.api.models import MessagesRequest, SystemContent
from src.services.translation import convert_anthropic_to_litellm
from src.services.llm_router import get_api_key_for_model
from src.core.config import GEMINI_MODELS, OPENAI_MODELS
from src.utils.beautiful_log import log_request_beautifully
# from src.utils.tool_parser import parse_tool_calls # Not used in the new version

logger = logging.getLogger(__name__)
router = APIRouter()

# If MAX_TOKENS is not defined in models.py, define it here or adjust logic:
# MAX_TOKENS = {} # Example: Define as empty dict if not imported

async def simplified_handle_streaming(
    response_obj: MessagesRequest,
    litellm_response_chunks: AsyncGenerator,
) -> AsyncGenerator[str, None]:
    """
    Simplified SSE stream handler that adapts LiteLLM's OpenAI-compatible SSE chunks.
    """
    # model = response_obj.model # original_model or mapped model depending on context
    # The response_obj.model here will be the one after pydantic validation & mapping
    # We use response_obj.original_model for display purposes in the SSE message_start if needed
    # For LiteLLM call, we use the (potentially mapped) response_obj.model
    stream_id = ""
    first_chunk = True
    tool_calls_chunks: Dict[int, Dict[str, Union[str, int]]] = {} # Cache for tool call chunks
    fallback_stream_id = f"msg_ {''.join(random.choices(string.ascii_letters + string.digits, k=12))}"
    idx_counter = 0 # To keep track of content block indices

    async for chunk in litellm_response_chunks:
        chunk_dict = chunk.model_dump()
        if not stream_id and chunk_dict.get("id"):
            stream_id = chunk_dict["id"]
        
        current_event_stream_id = stream_id if stream_id else fallback_stream_id

        if first_chunk:
            first_chunk = False
            # Send message_start event
            # Ensure input_tokens are accurate if possible, or set to a placeholder like 0 or actual count if available
            # For now, we'll use a placeholder for input_tokens in message_start as LiteLLM might not provide it early.
            # It's often part of the final usage stats.
            # The main `messages` endpoint will log the request with `log_request_beautifully` which can give an idea of input tokens.
            # Actual input tokens as per LiteLLM will be in the final usage statistics.
            
            # Estimate input tokens for message_start event. This is a rough estimate.
            # Actual token count is available in the final response from LiteLLM.
            # We can import litellm.token_counter if we want a more precise estimate here.
            # For now, we'll omit input_tokens from message_start to avoid confusion,
            # as the final `usage` block will have the authoritative count.

            sse_message_start = {
                "type": "message_start",
                "message": {
                    "id": current_event_stream_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    # Use original_model for the message_start event as per Anthropic convention
                    "model": response_obj.original_model or response_obj.model, 
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}, # Placeholder, actual usage at the end
                },
            }
            yield f"data: {json.dumps(sse_message_start)}\n\n"

            # Send ping after message_start, before any content
            yield f"data: {json.dumps({'type': 'ping'})}\n\n"

        if not chunk.choices: # Handle chunks without choices (e.g. some metadata or error chunks)
            if hasattr(chunk, 'error') and chunk.error:
                logger.error(f"LiteLLM error chunk: {chunk_dict}")
                # Adapt to Anthropic error format if possible, or send as a custom error event
                error_event = {
                    "type": "error",
                    "error": {
                        "type": "provider_error", # Or more specific if determinable
                        "message": str(chunk.error)
                    }
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                # Also send a message_stop for graceful termination on client side
                yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
                return
            logger.debug(f"Skipping non-choice chunk: {chunk_dict}")
            continue # Skip if no choices and not an error we explicitly handle

        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        # Accumulate tool call chunks
        if delta.tool_calls:
            for tc_chunk in delta.tool_calls:
                tc_idx = tc_chunk.index # This is the litellm index for the tool_call
                # We need to map tc_idx to our content_block index if they differ.
                # For simplicity, let's assume one tool call corresponds to one content_block for now
                # And that we'll manage our own content_block indices.
                # Let's use `idx_counter` for content_block index for tools.

                current_tool_block_index = tc_idx # Use litellm's index for tool block continuity

                if current_tool_block_index not in tool_calls_chunks:
                    tool_calls_chunks[current_tool_block_index] = {"id": "", "name": "", "input": "", "type": "tool_use"}
                    # Send content_block_start for new tool
                    tool_start_event = {
                        "type": "content_block_start",
                        "index": current_tool_block_index,
                        "content_block": {"type": "tool_use", "id": "", "name": "", "input": {}}, # input will be {} initially
                    }
                    if tc_chunk.id: # id usually comes with the first part of the tool call
                         tool_start_event["content_block"]["id"] = tc_chunk.id
                         tool_calls_chunks[current_tool_block_index]["id"] = tc_chunk.id

                    yield f"data: {json.dumps(tool_start_event)}\n\n"

                # Update tool_calls_chunks with incoming data
                if tc_chunk.id and not tool_calls_chunks[current_tool_block_index]["id"]: # if ID was missing initially
                    tool_calls_chunks[current_tool_block_index]["id"] = tc_chunk.id
                    # Potentially update the already sent content_block_start if ID was missing.
                    # This is tricky with SSE. Simpler to ensure ID is there at start or client handles updates.
                    # For now, assume ID is part of the first chunk for a tool_call.

                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tool_calls_chunks[current_tool_block_index]["name"] = cast(str, tool_calls_chunks[current_tool_block_index]["name"]) + tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        current_args_chunk = tc_chunk.function.arguments
                        tool_calls_chunks[current_tool_block_index]["input"] = cast(str, tool_calls_chunks[current_tool_block_index]["input"]) + current_args_chunk
                        # Send input_json_delta
                        # The `input_json_delta` is an Anthropic specific event for streaming tool inputs.
                        # LiteLLM sends `arguments` which is a string representation of a JSON object.
                        input_delta_event = {
                            "type": "content_block_delta",
                            "index": current_tool_block_index,
                            "delta": {"type": "input_json_delta", "partial_json": current_args_chunk},
                        }
                        yield f"data: {json.dumps(input_delta_event)}\n\n"

        # Yield content delta for text
        if delta.content:
            # Assuming text content is always at index 0 for Claude-like responses from LiteLLM (OpenAI format)
            # If multiple content blocks of type text are possible, this needs adjustment.
            # For now, mapping all text to a single content_block at index 0.
            text_block_index = 0
            # Check if this is the first text delta for this block
            # This requires state, or make an assumption.
            # For simplicity, if it's the first text delta overall for the request, send content_block_start
            # This is not perfectly robust if text follows tools.
            # A better way would be to track content block types and indices.

            # Let's assume text always starts at index 0 if no tools preceded.
            # If tools preceded, text block index should be after the tool blocks.
            # Let's simplify: text is index 0, tools are 1, 2, ...
            # Or, use a running index for content blocks.

            # If this is the first non-tool content, assume it's text block 0
            # This needs to be more robust if mixing text and tools in complex ways.
            # For now, assume standard text response is one block at index 0.
            current_text_block_index = 0 # Default for simple text

            # Simplistic check: if this is the very first delta with content, send content_block_start
            # A more robust solution would be needed for complex sequences of text/tool blocks.
            # We can check if a content_block_start for index 0 has been sent.
            # To do this without complex state, we check if `idx_counter` is 0 and this is the first content.
            # This part of the logic needs refinement for true Anthropic multi-block compatibility.
            # For now, all text deltas go to index 0, and we send content_block_start once.

            if not hasattr(simplified_handle_streaming, f"_content_block_started_{current_text_block_index}"):
                content_start_event = {
                    "type": "content_block_start",
                    "index": current_text_block_index,
                    "content_block": {"type": "text", "text": ""}, # Initial text is empty
                }
                yield f"data: {json.dumps(content_start_event)}\n\n"
                setattr(simplified_handle_streaming, f"_content_block_started_{current_text_block_index}", True)

            content_delta_event = {
                "type": "content_block_delta",
                "index": current_text_block_index,
                "delta": {"type": "text_delta", "text": delta.content},
            }
            yield f"data: {json.dumps(content_delta_event)}\n\n"

        if finish_reason:
            # Stop events for content blocks
            # Stop text block (index 0)
            if hasattr(simplified_handle_streaming, f"_content_block_started_0"): # Check if text block was started
                text_stop_event = {
                    "type": "content_block_stop",
                    "index": 0
                }
                yield f"data: {json.dumps(text_stop_event)}\n\n"
                delattr(simplified_handle_streaming, f"_content_block_started_0")

            # Stop tool blocks
            if finish_reason == "tool_calls" or (delta.tool_calls and not delta.content): # Or if the last delta was only tool_calls related
                for tc_idx_stop in sorted(tool_calls_chunks.keys()):
                    # Check if already stopped or if it's the right time
                    tool_stop_event = {
                        "type": "content_block_stop",
                        "index": tc_idx_stop
                    }
                    yield f"data: {json.dumps(tool_stop_event)}\n\n"

            # Get usage from the final chunk (litellm often includes it here)
            final_usage = {"input_tokens": 0, "output_tokens": 0}
            if chunk.usage:
                final_usage["input_tokens"] = chunk.usage.prompt_tokens
                final_usage["output_tokens"] = chunk.usage.completion_tokens
            
            # Anthropic format: message_delta with stop_reason and usage
            message_delta_event = {
                "type": "message_delta",
                "delta": {"stop_reason": finish_reason if finish_reason != "tool_calls" else "tool_use"}, # Map "tool_calls" to "tool_use"
                "usage": final_usage,
            }
            yield f"data: {json.dumps(message_delta_event)}\n\n"

            # Final message_stop event
            message_stop_event = {"type": "message_stop", "amazon-bedrock-invocationMetrics": chunk.x_groq.get("usage") if chunk.x_groq else None} # Pass through provider specific metrics if available
            yield f"data: {json.dumps(message_stop_event)}\n\n"
            return # End of stream

    # Cleanup any stateful attributes on function for next call (if any were set)
    if hasattr(simplified_handle_streaming, "_content_block_started_0"):
        delattr(simplified_handle_streaming, "_content_block_started_0")
    # Add similar cleanup for other dynamic attributes if created


@router.post("/v1/messages", response_class=StreamingResponse)
async def messages(request: Request, response_obj: MessagesRequest) -> StreamingResponse:
    # Log the beautiful request
    # For streaming, input tokens are often calculated by the client or from a final usage block.
    # We log the request structure here.
    log_request_beautifully(response_obj.model_dump(exclude_none=True), role="user", request_type="anthropic_sdk")

    # mapped_model is now directly from response_obj.model after Pydantic validation
    mapped_model = response_obj.model 

    # convert_anthropic_to_litellm now handles messages, system_prompt, model, tools, and tool_choice
    # It returns a dictionary that can be directly used for LiteLLM, including processed tools.
    try:
        # Pass the entire response_obj to convert_anthropic_to_litellm
        # as it might need other fields like .tools, .tool_choice for full conversion
        final_request_params = convert_anthropic_to_litellm(response_obj)
    except Exception as e:
        logger.error(f"Error during convert_anthropic_to_litellm: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error processing request for LiteLLM conversion: {str(e)}")

    # Ensure the 'stream' parameter for acompletion matches the endpoint's stream behavior
    final_request_params["stream"] = response_obj.stream

    # Get the model name that LiteLLM will use (it might have been mapped by convert_anthropic_to_litellm)
    # or use the one from response_obj (which is already mapped by Pydantic)
    model_for_api_key = final_request_params.get("model", mapped_model)
    
    # Set the API key for LiteLLM
    # LiteLLM can also use environment variables, but explicit passing is safer.
    final_request_params["api_key"] = get_api_key_for_model(model_for_api_key)
    
    # Remove any top-level None values before calling acompletion
    # (convert_anthropic_to_litellm should ideally not include them, but this is a safeguard)
    final_request_params = {k: v for k, v in final_request_params.items() if v is not None}

    # Ensure 'messages' is present, as it's critical
    if "messages" not in final_request_params:
        logger.error(f"Critical error: 'messages' field is missing after conversion for model {model_for_api_key}.")
        raise HTTPException(status_code=500, detail="Internal error: Failed to prepare messages for LLM.")

    try:
        logger.info(f"Calling LiteLLM with model='{final_request_params.get("model")}', stream={response_obj.stream}")
        # logger.debug(f"LiteLLM Request Params: {json.dumps(final_request_params, indent=2)}") # Careful logging PII

        if response_obj.stream:
            litellm_response_chunks = await acompletion(**final_request_params) # type: ignore
            return StreamingResponse(
                simplified_handle_streaming(response_obj, litellm_response_chunks),
                media_type="text/event-stream",
            )
        else: # Non-streaming
            litellm_response = await acompletion(**final_request_params) # type: ignore

            if litellm_response is None:
                logger.error("Received None from litellm.acompletion (non-streaming)")
                raise HTTPException(status_code=500, detail="LLM call failed to return a response.")

            # Log the raw LiteLLM response before transformation
            # log_request_beautifully(litellm_response.model_dump(exclude_none=True), role="assistant_litellm_raw")

            assistant_content: List[Dict[str, any]] = []
            usage_data = { # Ensure we get valid usage data
                "input_tokens": litellm_response.usage.prompt_tokens if litellm_response.usage else 0,
                "output_tokens": litellm_response.usage.completion_tokens if litellm_response.usage else 0,
            }
            # Default stop_reason, will be overwritten if tool_calls or other reasons.
            stop_reason = litellm_response.choices[0].finish_reason if litellm_response.choices and litellm_response.choices[0].finish_reason else "unknown"


            if litellm_response.choices and litellm_response.choices[0].message.tool_calls:
                for tool_call in litellm_response.choices[0].message.tool_calls:
                    parsed_input = {}
                    try:
                        # Ensure arguments is a string before trying to parse
                        if isinstance(tool_call.function.arguments, str):
                            parsed_input = json.loads(tool_call.function.arguments)
                        elif isinstance(tool_call.function.arguments, dict): # Already a dict
                            parsed_input = tool_call.function.arguments
                        else: # Attempt to convert to string then parse, or handle error
                            logger.warning(f"Tool call arguments are not a string or dict: {type(tool_call.function.arguments)}. Attempting to parse as string.")
                            parsed_input = json.loads(str(tool_call.function.arguments))

                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse tool call arguments JSON: {tool_call.function.arguments}")
                        # Keep input as string if parsing fails, as per Anthropic's allowance for string content
                        parsed_input = {"_raw_arguments": tool_call.function.arguments}


                    assistant_content.append({
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": parsed_input,
                    })
                stop_reason = "tool_use" # Anthropic specific stop reason for tool usage
            elif litellm_response.choices and litellm_response.choices[0].message.content:
                assistant_content.append({"type": "text", "text": litellm_response.choices[0].message.content})
            
            # If content is empty but it's not a tool_call, check if it was a specific stop like 'length'
            if not assistant_content and stop_reason not in ["tool_use", "stop"]:
                # If litellm returns empty content with a specific finish_reason like 'length',
                # We should reflect that. For now, an empty text content block is added if no other content.
                 assistant_content.append({"type": "text", "text": ""})


            anthropic_response_dict = {
                "id": litellm_response.id if litellm_response.id else f"msg_{''.join(random.choices(string.ascii_letters + string.digits, k=12))}",
                "type": "message",
                "role": "assistant",
                "content": assistant_content,
                "model": response_obj.original_model or response_obj.model, # Return original model name if available
                "stop_reason": stop_reason,
                "stop_sequence": None, # LiteLLM might not directly provide this in the same way
                "usage": usage_data,
            }
            # Log the transformed Anthropic-like response
            log_request_beautifully(anthropic_response_dict, role="assistant_anthropic_sdk_final")

            return anthropic_response_dict # FastAPI will convert dict to JSONResponse

    except HTTPException: # Re-raise HTTPExceptions from validation or other upstream issues
        raise
    except asyncio.TimeoutError:
        logger.error(f"LiteLLM call timed out for model {final_request_params.get("model")}.")
        raise HTTPException(status_code=504, detail=f"Request to upstream LLM ({final_request_params.get("model")}) timed out.")
    except Exception as e:
        # This will catch errors from LiteLLM like APIConnectionError, AuthenticationError, etc.
        logger.exception(f"Error calling LiteLLM with model {final_request_params.get("model")}: {e}") # Log the full exception
        # Try to extract a more specific error message if it's a known LiteLLM exception type if possible
        # For now, a generic internal server error.
        # Example: if isinstance(e, litellm.exceptions.APIError):
        # raise HTTPException(status_code=e.status_code or 500, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error processing request with model {final_request_params.get("model")}: {str(e)}")


class Ping(BaseModel):
    type: str = "ping"

@router.get("/health")
async def health_check():
    return {"status": "ok"}

# Example of how the main app might add this router
# from fastapi import FastAPI
# app = FastAPI()
# app.include_router(router)
