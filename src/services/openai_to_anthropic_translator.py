import logging
import json
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from src.api.models import (
    AnthropicMessagesResponse, AnthropicMessage, AnthropicTool, AnthropicUsage,
    AnthropicContentBlockText, AnthropicContentBlockToolUse,
    OpenAIChatCompletionResponse, OpenAIChatCompletionChoiceMessage, OpenAIToolCall, OpenAICompletionUsage,
    OpenAIChatCompletionChunk, # For streaming
    AnthropicSSEMessageStart, AnthropicSSEContentBlockStart, AnthropicSSEContentBlockDelta,
    AnthropicSSEContentBlockStop, AnthropicSSEMessageDelta, AnthropicSSEMessageStop,
    AnthropicSSEPing, AnthropicSSEError, AnthropicSSEErrorContent # For streaming
)

logger = logging.getLogger(__name__)

def _translate_openai_message_content_to_anthropic(
    openai_message: OpenAIChatCompletionChoiceMessage,
    response_id: str,
    model_name: str,
    usage: OpenAICompletionUsage
) -> AnthropicMessagesResponse:
    """
    Translates the content part of an OpenAI ChatCompletion response message
    to an Anthropic MessagesResponse.
    """
    anthropic_content: List[Union[AnthropicContentBlockText, AnthropicContentBlockToolUse]] = []

    if openai_message.content:
        anthropic_content.append(AnthropicContentBlockText(type="text", text=openai_message.content))

    if openai_message.tool_calls:
        for tool_call in openai_message.tool_calls:
            anthropic_content.append(
                AnthropicContentBlockToolUse(
                    type="tool_use",
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                )
            )
    
    anthropic_usage = AnthropicUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens
    )

    # Determine stop_reason
    # OpenAI: "stop", "length", "tool_calls", "content_filter", "function_call" (legacy)
    # Anthropic: "end_turn", "max_tokens", "stop_sequence", "tool_use", "content_filtered"
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "content_filtered",
        "function_call": "tool_use" # map legacy openai to tool_use
    }
    
    # This mapping happens at the choice level, so we need to get it from there.
    # This function is called with a single message, assuming the choice selection is done prior.
    # For now, this function doesn't have direct access to choice.finish_reason.
    # This will be handled in the main translation function.

    return AnthropicMessagesResponse(
        id=response_id, # This should be the ID from the OpenAI response object
        model=model_name, # This should be the model from the OpenAI response object
        content=anthropic_content,
        usage=anthropic_usage,
        # stop_reason and stop_sequence will be set by the calling function
    )

def translate_openai_to_anthropic_response(
    openai_response: OpenAIChatCompletionResponse
) -> AnthropicMessagesResponse:
    """
    Translates a non-streaming OpenAI ChatCompletion response to an Anthropic MessagesResponse.
    """
    logger.info(f"Translating OpenAI response (ID: {openai_response.id}, Model: {openai_response.model}) to Anthropic format.")

    if not openai_response.choices:
        # Handle error case: No choices returned
        logger.error("OpenAI response contained no choices.")
        # This should ideally be converted to an Anthropic error response.
        # For now, raising an exception or returning a placeholder.
        # This part needs robust error handling.
        raise ValueError("OpenAI response had no choices.")

    # Assuming we always take the first choice, as is common.
    first_choice = openai_response.choices[0]
    
    anthropic_response = _translate_openai_message_content_to_anthropic(
        openai_message=first_choice.message,
        response_id=openai_response.id,
        model_name=openai_response.model,
        usage=openai_response.usage if openai_response.usage else OpenAICompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0) # Handle if usage is None
    )

    # Map finish_reason
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "content_filtered",
        "function_call": "tool_use" # map legacy openai to tool_use
    }
    if first_choice.finish_reason:
        anthropic_response.stop_reason = stop_reason_map.get(first_choice.finish_reason, first_choice.finish_reason) # type: ignore

    # stop_sequence is not directly available in OpenAI's ChatCompletion object's choice.
    # It's a request parameter. If LiteLLM passes it through, it might be in extensions.
    # For now, we leave it as None unless a clear mapping is found.

    return anthropic_response


async def translate_openai_to_anthropic_stream(
    openai_sse_generator: AsyncGenerator[OpenAIChatCompletionChunk, None],
    request_model_id: str # The model ID from the original Anthropic request
) -> AsyncGenerator[str, None]:
    """
    Translates an async generator of OpenAI SSE ChatCompletionChunk objects
    to an async generator of Anthropic SSE event strings.
    """
    logger.info("Starting OpenAI to Anthropic SSE stream translation.")
    
    # TODO: Implement detailed SSE chunk translation logic.
    # This is complex and needs to handle various event types and aggregate data.
    # - message_start: Sent once at the beginning. Needs OpenAI response ID, model, initial usage.
    # - content_block_start: When new text or tool_use block begins.
    # - content_block_delta: For text deltas or tool_use input deltas.
    # - content_block_stop: When a content block finishes.
    # - message_delta: For stop_reason and output_tokens usage updates.
    # - message_stop: Sent once at the end.
    # - ping: Can be forwarded if received, or generated.
    # - error: Translate OpenAI errors to Anthropic error events.

    # This is a simplified placeholder implementation.
    # A full implementation needs to manage state across chunks.
    
    # Send message_start (requires some info from the first chunk or pre-set)
    # This is tricky because the first OpenAI chunk might not have all info for Anthropic's message_start.
    # LiteLLM might provide a custom first event or we might need to buffer.
    
    # For now, a very basic pass-through of content, not correctly formatted as Anthropic SSE.
    # This will need significant work.
    
    # Example: Constructing a message_start (needs actual data)
    # initial_usage = AnthropicUsage(input_tokens=0, output_tokens=0) # Placeholder
    # message_start_event = AnthropicSSEMessageStart(
    #     type="message_start",
    #     message=AnthropicMessagesResponse(
    #         id="placeholder_id", # Should come from first OpenAI chunk
    #         type="message",
    #         role="assistant",
    #         model=request_model_id, # Echo back the requested model
    #         content=[], # Initially empty
    #         usage=initial_usage
    #     )
    # )
    # yield f"event: {message_start_event.type}\ndata: {message_start_event.model_dump_json()}\n\n"

    async for chunk in openai_sse_generator:
        logger.debug(f"Received OpenAI SSE chunk: {chunk.model_dump_json(exclude_none=True)}")
        # This is where the complex translation logic for each chunk type goes.
        # For now, just yielding a placeholder to show activity.
        # This will NOT be valid Anthropic SSE.
        
        # A more correct (but still incomplete) approach:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                # This would need to be wrapped in content_block_start, delta, stop
                # For simplicity, just sending a text delta (incorrectly formatted for Anthropic)
                # anthropic_delta = AnthropicContentBlockText(type="text", text=delta.content) # Incorrect usage for delta
                # yield f"event: content_block_delta\ndata: {{\"type\": \"text_delta\", \"text\": \"{delta.content}\"}}\n\n"
                
                # A slightly more structured attempt (still needs proper event sequence)
                # This is highly simplified and likely incorrect for full Anthropic compliance.
                if not hasattr(translate_openai_to_anthropic_stream, 'sent_message_start'):
                    # Simulate message_start (needs real data)
                    start_event_data = {
                        "type": "message_start",
                        "message": {
                            "id": chunk.id, "type": "message", "role": "assistant", "model": chunk.model,
                            "content": [], "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0} # Placeholder
                        }
                    }
                    yield f"event: message_start\ndata: {json.dumps(start_event_data)}\n\n"
                    
                    # Simulate content_block_start for the first text block
                    cb_start_data = {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
                    yield f"event: content_block_start\ndata: {json.dumps(cb_start_data)}\n\n"
                    setattr(translate_openai_to_anthropic_stream, 'sent_message_start', True)
                    setattr(translate_openai_to_anthropic_stream, 'current_block_index', 0)


                delta_data = {"type": "content_block_delta", "index": getattr(translate_openai_to_anthropic_stream, 'current_block_index', 0), "delta": {"type": "text_delta", "text": delta.content}}
                yield f"event: content_block_delta\ndata: {json.dumps(delta_data)}\n\n"

            if chunk.choices[0].finish_reason:
                # Simulate content_block_stop
                cb_stop_data = {"type": "content_block_stop", "index": getattr(translate_openai_to_anthropic_stream, 'current_block_index', 0)}
                yield f"event: content_block_stop\ndata: {json.dumps(cb_stop_data)}\n\n"

                # Simulate message_delta and message_stop
                msg_delta_data = {
                    "type": "message_delta",
                    "delta": {"stop_reason": chunk.choices[0].finish_reason},
                    "usage": {"output_tokens": chunk.usage.completion_tokens if chunk.usage else 0} # Placeholder
                }
                yield f"event: message_delta\ndata: {json.dumps(msg_delta_data)}\n\n"
                yield f"event: message_stop\ndata: {{\"type\": \"message_stop\"}}\n\n"
                # Reset state for potential next stream in a batch (if applicable, though usually one stream per request)
                if hasattr(translate_openai_to_anthropic_stream, 'sent_message_start'):
                    delattr(translate_openai_to_anthropic_stream, 'sent_message_start')


    logger.info("Finished OpenAI to Anthropic SSE stream translation.")

def translate_error_openai_to_anthropic(
    error_details: Dict[str, Any], # Could be an exception object or a dict from LiteLLM
    response_id: str = "error-id-unknown" # TODO: Generate or get a real ID
) -> Dict[str, Any]: # Returns a dict that can be used for Anthropic error JSON response
    """
    Translates an OpenAI/LiteLLM error structure to an Anthropic error structure.
    This is a basic version. LiteLLM might return errors in OpenAI format.
    Anthropic error format: {"type": "error", "error": {"type": "api_error", "message": "..."}}
    """
    logger.error(f"Translating error to Anthropic format: {error_details}")
    
    # Default error message
    error_type = "api_error"
    error_message = "An unexpected error occurred."

    # Attempt to parse LiteLLM/OpenAI style errors
    if isinstance(error_details, dict):
        err_obj = error_details.get("error", error_details) # LiteLLM might wrap it
        error_message = err_obj.get("message", error_message)
        # Try to map OpenAI error types to Anthropic ones if possible, or use a generic one
        # OpenAI types: invalid_request_error, api_error, rate_limit_error, authentication_error etc.
        # Anthropic types: invalid_request_error, authentication_error, permission_error, not_found_error,
        #                  rate_limit_error, api_error, overloaded_error
        raw_type = err_obj.get("type")
        if raw_type:
            if "auth" in raw_type.lower(): error_type = "authentication_error"
            elif "rate_limit" in raw_type.lower(): error_type = "rate_limit_error"
            elif "invalid_request" in raw_type.lower(): error_type = "invalid_request_error"
            # Add more mappings as needed
            else: error_type = "api_error" # Default fallback

    elif isinstance(error_details, Exception):
        error_message = str(error_details)

    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": error_message,
        },
        # "id": response_id, # Anthropic error responses don't typically include an 'id' at the top level
        # "model": "unknown", # Or the model that was attempted
        # "role": "assistant",
        # "stop_reason": "error" # This is not standard Anthropic error format
    }

# For SSE error streaming
async def generate_anthropic_error_sse(
    error_type_str: str,
    message: str
) -> AsyncGenerator[str, None]:
    """
    Generates Anthropic SSE error events.
    """
    error_content = AnthropicSSEErrorContent(type=error_type_str, message=message)
    error_event = AnthropicSSEError(type="error", error=error_content)
    yield f"event: error\ndata: {error_event.model_dump_json()}\n\n"