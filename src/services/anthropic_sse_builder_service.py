import json
from loguru import logger
from typing import AsyncGenerator, Dict, Any
from src.api.v1.schemas.anthropic_api import (
    AnthropicMessagesResponse, # For message_start event
    # SSE specific models are implicitly used by constructing dicts to match them
) # Updated import
from src.models.openai_provider_models import OpenAIChatCompletionChunk


def _build_message_start_event(initial_response_id: str, request_model_id: str) -> str:
    message_start_payload = AnthropicMessagesResponse(
        id=initial_response_id,
        type="message",
        role="assistant",
        model=request_model_id,
        content=[],
        stop_reason=None,
        stop_sequence=None,
        usage={"input_tokens": 0, "output_tokens": 0}
    )
    start_event_data = {
        "type": "message_start",
        "message": message_start_payload.model_dump(exclude_none=True)
    }
    return f'event: message_start\ndata: {json.dumps(start_event_data)}\n\n'


def _build_content_block_start_event(current_block_index: int) -> str:
    cb_start_data = {
        "type": "content_block_start",
        "index": current_block_index,
        "content_block": {"type": "text", "text": ""}
    }
    return f'event: content_block_start\ndata: {json.dumps(cb_start_data)}\n\n'


def _build_content_block_delta_event(current_block_index: int, text_content: str) -> str:
    delta_data = {
        "type": "content_block_delta",
        "index": current_block_index,
        "delta": {"type": "text_delta", "text": text_content}
    }
    return f'event: content_block_delta\ndata: {json.dumps(delta_data)}\n\n'


def _build_content_block_stop_event(current_block_index: int) -> str:
    cb_stop_data = {
        "type": "content_block_stop",
        "index": current_block_index
    }
    return f'event: content_block_stop\ndata: {json.dumps(cb_stop_data)}\n\n'


def _build_message_completion_events(finish_reason: str, chunk_usage: Dict[str, Any] | None) -> list[str]:
    events = []
    stop_reason_map = {
        'stop': 'end_turn',
        'length': 'max_tokens',
        'tool_calls': 'tool_use',
        'content_filter': 'content_filtered',
        'function_call': 'tool_use', # OpenAI often uses this for tools
    }
    anthropic_finish_reason = stop_reason_map.get(finish_reason, finish_reason)

    output_tokens = 1 # Default placeholder
    if chunk_usage:
        if isinstance(chunk_usage, dict):
            output_tokens = chunk_usage.get('completion_tokens', 1)
        else: # Assumes it's an object with attributes
            output_tokens = getattr(chunk_usage, 'completion_tokens', 1)

    msg_delta_data = {
        "type": "message_delta",
        "delta": {"stop_reason": anthropic_finish_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens}
    }
    events.append(f'event: message_delta\ndata: {json.dumps(msg_delta_data)}\n\n')
    events.append(f'event: message_stop\ndata: {json.dumps({"type": "message_stop"})}\n\n')
    return events


async def build_anthropic_sse_stream(
    openai_sse_generator: AsyncGenerator[OpenAIChatCompletionChunk, None],
    request_model_id: str, # Used for the initial message_start model field
    initial_response_id: str # Used for the initial message_start id field
) -> AsyncGenerator[str, None]:
    logger.debug('Starting OpenAI to Anthropic SSE stream translation for SSE builder service.')
    
    sent_message_start = False
    current_block_index = 0
    # Using a dict to track if content_block_start was sent for a given index
    # This is cleaner than dynamic attributes on the function itself.
    sent_content_block_starts: Dict[int, bool] = {}

    async for chunk in openai_sse_generator:
        logger.debug(f"SSE Builder: Received OpenAI SSE chunk. ID: {chunk.id}, Choices: {len(chunk.choices) if chunk.choices else 0}, Finish reason: {chunk.choices[0].finish_reason if chunk.choices and chunk.choices[0].finish_reason else 'N/A'}")
        
        if not sent_message_start:
            yield _build_message_start_event(initial_response_id, request_model_id)
            sent_message_start = True

        if chunk.choices:
            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            # Handling content_block_start and content_block_delta for text content
            if delta.content:
                # Send content_block_start if it's the beginning of new content for this block index
                # This simple version assumes one text block for now.
                # A more complex version would track if a tool_call_start needs to be sent.
                if not sent_content_block_starts.get(current_block_index):
                    yield _build_content_block_start_event(current_block_index)
                    sent_content_block_starts[current_block_index] = True

                yield _build_content_block_delta_event(current_block_index, delta.content)
            
            # Currently, this only handles text deltas.
            # TODO JIRA-1234: Need to handle comprehensive tool_call streaming here if OpenAI/LiteLLM sends tool_call deltas.
            # This involves:
            # 1. Detecting the start of a tool_call in a chunk (e.g., if chunk.choices[0].delta.tool_calls exists).
            # 2. If it's the first tool_call delta for a given tool_call_id and index, yield a content_block_start for tool_use.
            # 3. Yield content_block_delta events for the tool_use, accumulating the 'name', 'id', and 'input' parts.
            #    The input might be streamed as a JSON string, so careful accumulation is needed.
            # 4. When the tool_call is complete (either by a specific signal or end of stream for that tool_call), yield content_block_stop.
            # This is complex because tool_calls can be interleaved with text, and multiple tool_calls can appear.
            # The Anthropic equivalent is `content_block_start` (type: tool_use), `content_block_delta` (type: input_json_delta), `content_block_stop` (type: tool_use).
            if delta.tool_calls:
                # TODO: Implement tool_call handling
                pass

            if finish_reason:
                # Send content_block_stop for the current block
                if sent_content_block_starts.get(current_block_index):
                    yield _build_content_block_stop_event(current_block_index)
                    # No need to increment current_block_index here unless a tool_use block followed,
                    # which is not yet implemented.
                
                completion_events = _build_message_completion_events(finish_reason, chunk.usage)
                for event_str in completion_events:
                    yield event_str
                
                # Reset for next potential message in a session (if applicable, though typically one message per stream)
                sent_message_start = False
                current_block_index = 0 # Reset block index for a new message stream
                sent_content_block_starts.clear()

    logger.debug('Finished OpenAI to Anthropic SSE stream translation in SSE builder service.') 