import logging
import json
from typing import List, Dict, Any, Optional, Union
from src.api.v1.schemas.anthropic_api import (
    AnthropicMessagesRequest,
    AnthropicMessage,
    AnthropicTool,
    AnthropicToolChoice,
    AnthropicContentBlockText,
    AnthropicContentBlockImage,
    AnthropicContentBlockToolUse,
    AnthropicContentBlockToolResult,
)
from src.models.openai_provider_models import (
    OpenAIChatCompletionRequest,
    OpenAIChatMessageSystem,
    OpenAIChatMessageUser,
    OpenAIChatMessageAssistant,
    OpenAIChatMessageTool,
    OpenAITool,
    OpenAIFunctionDefinition,
    OpenAIToolCall,
    OpenAIFunctionCall,
    OpenAIMessageContentPartText,
    OpenAIMessageContentPartImage,
    OpenAIMessageContentPartImageURL,
    OpenAIToolChoiceOption,
)
from src.core.config import settings

logger = logging.getLogger(__name__)


def _translate_anthropic_messages_to_openai(
    anthropic_messages: List[AnthropicMessage],
    anthropic_system_prompt: Optional[Union[str, List[AnthropicContentBlockText]]],
) -> List[
    Union[
        OpenAIChatMessageSystem,
        OpenAIChatMessageUser,
        OpenAIChatMessageAssistant,
        OpenAIChatMessageTool,
    ]
]:
    openai_messages: List[
        Union[
            OpenAIChatMessageSystem,
            OpenAIChatMessageUser,
            OpenAIChatMessageAssistant,
            OpenAIChatMessageTool,
        ]
    ] = []
    if anthropic_system_prompt:
        system_content = ""
        if isinstance(anthropic_system_prompt, str):
            system_content = anthropic_system_prompt
        elif isinstance(anthropic_system_prompt, list):
            system_content = "\n".join(
                (
                    block.text
                    for block in anthropic_system_prompt
                    if isinstance(block, AnthropicContentBlockText)
                )
            )
        if system_content and system_content.strip():
            openai_messages.append(
                OpenAIChatMessageSystem(role="system", content=system_content.strip())
            )
    for msg in anthropic_messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                openai_messages.append(
                    OpenAIChatMessageUser(role="user", content=msg.content)
                )
            elif isinstance(msg.content, list):
                current_message_text_image_parts: List[
                    Union[OpenAIMessageContentPartText, OpenAIMessageContentPartImage]
                ] = []
                current_message_tool_results: List[OpenAIChatMessageTool] = []
                for block in msg.content:
                    if isinstance(block, AnthropicContentBlockText):
                        current_message_text_image_parts.append(
                            OpenAIMessageContentPartText(type="text", text=block.text)
                        )
                    elif isinstance(block, AnthropicContentBlockImage):
                        media_type = block.source.media_type or "image/jpeg"
                        image_url = f"data:{media_type};base64,{block.source.data}"
                        current_message_text_image_parts.append(
                            OpenAIMessageContentPartImage(
                                type="image_url",
                                image_url=OpenAIMessageContentPartImageURL(
                                    url=image_url
                                ),
                            )
                        )
                    elif isinstance(block, AnthropicContentBlockToolResult):
                        try:
                            if isinstance(block.content, (list, dict)):
                                tool_content_str = json.dumps(block.content)
                            else:
                                tool_content_str = str(block.content)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Content for tool_call_id {block.tool_use_id} is a string but not valid JSON. Passing as raw string."
                            )
                            tool_content_str = str(block.content)
                        current_message_tool_results.append(
                            OpenAIChatMessageTool(
                                role="tool",
                                tool_call_id=block.tool_use_id,
                                content=tool_content_str,
                            )
                        )
                if current_message_tool_results:
                    openai_messages.extend(current_message_tool_results)
                if current_message_text_image_parts:
                    if (
                        len(current_message_text_image_parts) == 1
                        and current_message_text_image_parts[0].type == "text"
                    ):
                        openai_messages.append(
                            OpenAIChatMessageUser(
                                role="user",
                                content=current_message_text_image_parts[0].text,
                            )
                        )
                    else:
                        openai_messages.append(
                            OpenAIChatMessageUser(
                                role="user", content=current_message_text_image_parts
                            )
                        )
                elif (
                    not msg.content
                    and (not current_message_tool_results)
                    and (not current_message_text_image_parts)
                ):
                    openai_messages.append(
                        OpenAIChatMessageUser(role="user", content="")
                    )
        elif msg.role == "assistant":
            assistant_text_content_parts: List[str] = []
            assistant_tool_calls: List[OpenAIToolCall] = []
            if isinstance(msg.content, str):
                assistant_text_content_parts.append(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, AnthropicContentBlockText):
                        assistant_text_content_parts.append(block.text)
                    elif isinstance(block, AnthropicContentBlockToolUse):
                        assistant_tool_calls.append(
                            OpenAIToolCall(
                                id=block.id,
                                type="function",
                                function=OpenAIFunctionCall(
                                    name=block.name, arguments=json.dumps(block.input)
                                ),
                            )
                        )
            final_assistant_text_content: Optional[str] = (
                "\n".join(assistant_text_content_parts)
                if assistant_text_content_parts
                else None
            )
            final_tool_calls = assistant_tool_calls if assistant_tool_calls else None
            if final_assistant_text_content is None and (not final_tool_calls):
                openai_messages.append(
                    OpenAIChatMessageAssistant(role="assistant", content="")
                )
            else:
                openai_messages.append(
                    OpenAIChatMessageAssistant(
                        role="assistant",
                        content=final_assistant_text_content,
                        tool_calls=final_tool_calls,
                    )
                )
    return openai_messages


def _translate_anthropic_tools_to_openai(
    anthropic_tools: Optional[List[AnthropicTool]],
) -> Optional[List[OpenAITool]]:
    if anthropic_tools is None:
        return None
    if not anthropic_tools:
        return []
    openai_tools: List[OpenAITool] = []
    for tool in anthropic_tools:
        params_dict = tool.input_schema.model_dump(exclude_none=True)
        if "properties" in params_dict and isinstance(params_dict["properties"], dict):
            for prop_name, prop_schema in params_dict["properties"].items():
                if (
                    isinstance(prop_schema, dict)
                    and prop_schema.get("type") == "string"
                ):
                    if "format" in prop_schema and prop_schema["format"] != "date-time":
                        del prop_schema["format"]
        openai_tools.append(
            OpenAITool(
                type="function",
                function=OpenAIFunctionDefinition(
                    name=tool.name, description=tool.description, parameters=params_dict
                ),
            )
        )
    return openai_tools


def _translate_anthropic_tool_choice_to_openai(
    anthropic_tool_choice: Optional[AnthropicToolChoice],
) -> Optional[OpenAIToolChoiceOption]:
    if not anthropic_tool_choice:
        return None
    logger.debug(
        f"Attempting to translate anthropic_tool_choice: {anthropic_tool_choice}"
    )
    if hasattr(anthropic_tool_choice, "type"):
        choice_type = getattr(anthropic_tool_choice, "type")
        if choice_type == "auto":
            return "auto"
        elif choice_type == "any":
            logger.warning(
                "Anthropic 'tool_choice.type=any' is being mapped to OpenAI 'auto'. Behavior might differ slightly."
            )
            return "auto"
        elif choice_type == "tool" and hasattr(anthropic_tool_choice, "name"):
            tool_name = getattr(anthropic_tool_choice, "name")
            return {"type": "function", "function": {"name": tool_name}}
        else:
            logger.warning(
                f"Unhandled Anthropic tool_choice type: {choice_type}. Defaulting to 'auto'."
            )
            return "auto"
    elif isinstance(anthropic_tool_choice, str) and anthropic_tool_choice in [
        "auto",
        "any",
    ]:
        if anthropic_tool_choice == "any":
            logger.warning(
                "Anthropic 'tool_choice=any' (string) is being mapped to OpenAI 'auto'."
            )
            return "auto"
        return anthropic_tool_choice
    logger.warning(
        f"Could not translate Anthropic tool_choice: {anthropic_tool_choice}. Defaulting to 'auto'."
    )
    return "auto"


def translate_anthropic_to_openai_request(
    anthropic_request: AnthropicMessagesRequest,
) -> OpenAIChatCompletionRequest:
    logger.debug(
        f"Translating Anthropic request (model: {anthropic_request.model}) to OpenAI format."
    )
    openai_messages = _translate_anthropic_messages_to_openai(
        anthropic_messages=anthropic_request.messages,
        anthropic_system_prompt=anthropic_request.system,
    )
    openai_tools = _translate_anthropic_tools_to_openai(
        anthropic_tools=anthropic_request.tools
    )
    openai_tool_choice = _translate_anthropic_tool_choice_to_openai(
        anthropic_tool_choice=anthropic_request.tool_choice
    )
    request_dict = {
        "model": anthropic_request.model,
        "messages": openai_messages,
        "max_tokens": anthropic_request.max_tokens,
        "stream": anthropic_request.stream,
        "temperature": anthropic_request.temperature,
        "top_p": anthropic_request.top_p,
    }
    if openai_tools:
        request_dict["tools"] = openai_tools
    if openai_tool_choice:
        request_dict["tool_choice"] = openai_tool_choice
    if anthropic_request.stop_sequences:
        request_dict["stop"] = anthropic_request.stop_sequences
    final_request_dict = {k: v for k, v in request_dict.items() if v is not None}
    return OpenAIChatCompletionRequest(**final_request_dict)
