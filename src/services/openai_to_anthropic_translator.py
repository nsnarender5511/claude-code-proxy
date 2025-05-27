import logging
import json
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from src.api.models import (
    AnthropicMessagesResponse,
    AnthropicMessage,
    AnthropicTool,
    AnthropicUsage,
    AnthropicContentBlockText,
    AnthropicContentBlockToolUse,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChoiceMessage,
    OpenAIToolCall,
    OpenAICompletionUsage,
    OpenAIChatCompletionChunk,
    AnthropicSSEMessageStart,
    AnthropicSSEContentBlockStart,
    AnthropicSSEContentBlockDelta,
    AnthropicSSEContentBlockStop,
    AnthropicSSEMessageDelta,
    AnthropicSSEMessageStop,
    AnthropicSSEPing,
    AnthropicSSEError,
    AnthropicSSEErrorContent,
)

logger = logging.getLogger(__name__)


def _translate_openai_message_content_to_anthropic(
    openai_message: OpenAIChatCompletionChoiceMessage,
    response_id: str,
    model_name: str,
    usage: OpenAICompletionUsage,
) -> AnthropicMessagesResponse:
    anthropic_content: List[Union[AnthropicContentBlockText, AnthropicContentBlockToolUse]] = []
    if openai_message.content:
        anthropic_content.append(
            AnthropicContentBlockText(type='text', text=openai_message.content)
        )
    if openai_message.tool_calls:
        for tool_call in openai_message.tool_calls:
            anthropic_content.append(
                AnthropicContentBlockToolUse(
                    type='tool_use',
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=(
                        json.loads(tool_call.function.arguments)
                        if isinstance(tool_call.function.arguments, str)
                        else tool_call.function.arguments
                    ),
                )
            )
    anthropic_usage = AnthropicUsage(
        input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens
    )
    stop_reason_map = {
        'stop': 'end_turn',
        'length': 'max_tokens',
        'tool_calls': 'tool_use',
        'content_filter': 'content_filtered',
        'function_call': 'tool_use',
    }
    return AnthropicMessagesResponse(
        id=response_id, model=model_name, content=anthropic_content, usage=anthropic_usage
    )


def translate_openai_to_anthropic_response(
    openai_response: OpenAIChatCompletionResponse,
) -> AnthropicMessagesResponse:
    logger.info(
        f'Translating OpenAI response (ID: {openai_response.id}, Model: {openai_response.model}) to Anthropic format.'
    )
    if not openai_response.choices:
        logger.error('OpenAI response contained no choices.')
        raise ValueError('OpenAI response had no choices.')
    first_choice = openai_response.choices[0]
    anthropic_response = _translate_openai_message_content_to_anthropic(
        openai_message=first_choice.message,
        response_id=openai_response.id,
        model_name=openai_response.model,
        usage=(
            openai_response.usage
            if openai_response.usage
            else OpenAICompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        ),
    )
    stop_reason_map = {
        'stop': 'end_turn',
        'length': 'max_tokens',
        'tool_calls': 'tool_use',
        'content_filter': 'content_filtered',
        'function_call': 'tool_use',
    }
    if first_choice.finish_reason:
        anthropic_response.stop_reason = stop_reason_map.get(
            first_choice.finish_reason, first_choice.finish_reason
        )
    return anthropic_response


async def translate_openai_to_anthropic_stream(
    openai_sse_generator: AsyncGenerator[OpenAIChatCompletionChunk, None], request_model_id: str
) -> AsyncGenerator[str, None]:
    logger.info('Starting OpenAI to Anthropic SSE stream translation.')
    async for chunk in openai_sse_generator:
        logger.debug(f'Received OpenAI SSE chunk: {chunk.model_dump_json(exclude_none=True)}')
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                if not hasattr(translate_openai_to_anthropic_stream, 'sent_message_start'):
                    start_event_data = {
                        'type': 'message_start',
                        'message': {
                            'id': chunk.id,
                            'type': 'message',
                            'role': 'assistant',
                            'model': chunk.model,
                            'content': [],
                            'stop_reason': None,
                            'stop_sequence': None,
                            'usage': {'input_tokens': 0, 'output_tokens': 0},
                        },
                    }
                    yield f'event: message_start\ndata: {json.dumps(start_event_data)}\n\n'
                    cb_start_data = {
                        'type': 'content_block_start',
                        'index': 0,
                        'content_block': {'type': 'text', 'text': ''},
                    }
                    yield f'event: content_block_start\ndata: {json.dumps(cb_start_data)}\n\n'
                    setattr(translate_openai_to_anthropic_stream, 'sent_message_start', True)
                    setattr(translate_openai_to_anthropic_stream, 'current_block_index', 0)
                delta_data = {
                    'type': 'content_block_delta',
                    'index': getattr(
                        translate_openai_to_anthropic_stream, 'current_block_index', 0
                    ),
                    'delta': {'type': 'text_delta', 'text': delta.content},
                }
                yield f'event: content_block_delta\ndata: {json.dumps(delta_data)}\n\n'
            if chunk.choices[0].finish_reason:
                cb_stop_data = {
                    'type': 'content_block_stop',
                    'index': getattr(
                        translate_openai_to_anthropic_stream, 'current_block_index', 0
                    ),
                }
                yield f'event: content_block_stop\ndata: {json.dumps(cb_stop_data)}\n\n'
                msg_delta_data = {
                    'type': 'message_delta',
                    'delta': {'stop_reason': chunk.choices[0].finish_reason},
                    'usage': {'output_tokens': chunk.usage.completion_tokens if chunk.usage else 0},
                }
                yield f'event: message_delta\ndata: {json.dumps(msg_delta_data)}\n\n'
                yield f'event: message_stop\ndata: {{"type": "message_stop"}}\n\n'
                if hasattr(translate_openai_to_anthropic_stream, 'sent_message_start'):
                    delattr(translate_openai_to_anthropic_stream, 'sent_message_start')
    logger.info('Finished OpenAI to Anthropic SSE stream translation.')


def translate_error_openai_to_anthropic(
    error_details: Dict[str, Any], response_id: str = 'error-id-unknown'
) -> Dict[str, Any]:
    logger.error(f'Translating error to Anthropic format: {error_details}')
    error_type = 'api_error'
    error_message = 'An unexpected error occurred.'
    if isinstance(error_details, dict):
        err_obj = error_details.get('error', error_details)
        error_message = err_obj.get('message', error_message)
        raw_type = err_obj.get('type')
        if raw_type:
            if 'auth' in raw_type.lower():
                error_type = 'authentication_error'
            elif 'rate_limit' in raw_type.lower():
                error_type = 'rate_limit_error'
            elif 'invalid_request' in raw_type.lower():
                error_type = 'invalid_request_error'
            else:
                error_type = 'api_error'
    elif isinstance(error_details, Exception):
        error_message = str(error_details)
    return {'type': 'error', 'error': {'type': error_type, 'message': error_message}}


async def generate_anthropic_error_sse(
    error_type_str: str, message: str
) -> AsyncGenerator[str, None]:
    error_content = AnthropicSSEErrorContent(type=error_type_str, message=message)
    error_event = AnthropicSSEError(type='error', error=error_content)
    yield f'event: error\ndata: {error_event.model_dump_json()}\n\n'
