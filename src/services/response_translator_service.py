import logging
import json
from typing import List, Union
from src.api.v1.schemas.anthropic_api import (
    AnthropicMessagesResponse,
    AnthropicUsage,
    AnthropicContentBlockText,
    AnthropicContentBlockToolUse,
)
from src.models.openai_provider_models import (
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChoiceMessage,
    OpenAICompletionUsage,
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
    
    # stop_reason is handled by the main translate_openai_to_anthropic_response function
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
        'function_call': 'tool_use', # OpenAI uses this for function calling finish reason
    }
    if first_choice.finish_reason:
        anthropic_response.stop_reason = stop_reason_map.get(
            first_choice.finish_reason, first_choice.finish_reason # Default to original if not in map
        )
    return anthropic_response 