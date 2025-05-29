import json
import logging
from typing import AsyncGenerator, Dict, Any
from src.api.v1.schemas.anthropic_api import AnthropicMessagesResponse
from src.models.openai_provider_models import OpenAIChatCompletionChunk

logger = logging.getLogger(__name__)


def _build_message_start_event(initial_response_id: str, request_model_id: str) -> str:
    message_start_payload = AnthropicMessagesResponse(
        id=initial_response_id,
        type="message",
        role="assistant",
        model=request_model_id,
        content=[],
        stop_reason=None,
        stop_sequence=None,
        usage={"input_tokens": 0, "output_tokens": 0},
    )
    start_event_data = {
        "type": "message_start",
        "message": message_start_payload.model_dump(exclude_none=True),
    }
    return f"event: message_start\ndata: {json.dumps(start_event_data)}\n\n"


def _build_content_block_start_event(current_block_index: int) -> str:
    cb_start_data = {
        "type": "content_block_start",
        "index": current_block_index,
        "content_block": {"type": "text", "text": ""},
    }
    return f"event: content_block_start\ndata: {json.dumps(cb_start_data)}\n\n"


def _build_content_block_delta_event(
    current_block_index: int, text_content: str
) -> str:
    delta_data = {
        "type": "content_block_delta",
        "index": current_block_index,
        "delta": {"type": "text_delta", "text": text_content},
    }
    return f"event: content_block_delta\ndata: {json.dumps(delta_data)}\n\n"


def _build_content_block_stop_event(current_block_index: int) -> str:
    cb_stop_data = {"type": "content_block_stop", "index": current_block_index}
    return f"event: content_block_stop\ndata: {json.dumps(cb_stop_data)}\n\n"


def _build_message_completion_events(
    finish_reason: str, chunk_usage: Dict[str, Any] | None
) -> list[str]:
    events = []
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "content_filtered",
        "function_call": "tool_use",
    }
    anthropic_finish_reason = stop_reason_map.get(finish_reason, finish_reason)
    output_tokens = 1
    if chunk_usage:
        if isinstance(chunk_usage, dict):
            output_tokens = chunk_usage.get("completion_tokens", 1)
        else:
            output_tokens = getattr(chunk_usage, "completion_tokens", 1)
    msg_delta_data = {
        "type": "message_delta",
        "delta": {"stop_reason": anthropic_finish_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }
    events.append(f"event: message_delta\ndata: {json.dumps(msg_delta_data)}\n\n")
    events.append(
        f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    )
    return events


async def build_anthropic_sse_stream(
    openai_sse_generator: AsyncGenerator[OpenAIChatCompletionChunk, None],
    request_model_id: str,
    initial_response_id: str,
) -> AsyncGenerator[str, None]:
    logger.debug(
        "Starting OpenAI to Anthropic SSE stream translation for SSE builder service."
    )
    sent_message_start = False
    current_block_index = 0
    sent_content_block_starts: Dict[int, bool] = {}
    async for chunk in openai_sse_generator:
        logger.debug(
            f"SSE Builder: Received OpenAI SSE chunk. ID: {chunk.id}, Choices: {(len(chunk.choices) if chunk.choices else 0)}, Finish reason: {(chunk.choices[0].finish_reason if chunk.choices and chunk.choices[0].finish_reason else 'N/A')}"
        )
        if not sent_message_start:
            yield _build_message_start_event(initial_response_id, request_model_id)
            sent_message_start = True
        if chunk.choices:
            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason
            if delta.content:
                if not sent_content_block_starts.get(current_block_index):
                    yield _build_content_block_start_event(current_block_index)
                    sent_content_block_starts[current_block_index] = True
                yield _build_content_block_delta_event(
                    current_block_index, delta.content
                )
            if delta.tool_calls:
                pass
            if finish_reason:
                if sent_content_block_starts.get(current_block_index):
                    yield _build_content_block_stop_event(current_block_index)
                completion_events = _build_message_completion_events(
                    finish_reason, chunk.usage
                )
                for event_str in completion_events:
                    yield event_str
                sent_message_start = False
                current_block_index = 0
                sent_content_block_starts.clear()
    logger.debug(
        "Finished OpenAI to Anthropic SSE stream translation in SSE builder service."
    )
