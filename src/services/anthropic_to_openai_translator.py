


import logging
import json # Added for serializing tool inputs/results
from typing import List, Dict, Any, Optional, Union
from src.api.models import (
    AnthropicMessagesRequest, AnthropicMessage, AnthropicTool, AnthropicToolChoice,
    AnthropicContentBlockText, AnthropicContentBlockImage, AnthropicContentBlockToolUse, AnthropicContentBlockToolResult,
    OpenAIChatCompletionRequest, OpenAIChatMessageSystem, OpenAIChatMessageUser,
    OpenAIChatMessageAssistant, OpenAIChatMessageTool, OpenAITool, OpenAIFunctionDefinition, OpenAIToolCall, OpenAIFunctionCall, # Added OpenAIToolCall
    OpenAIMessageContentPartText, OpenAIMessageContentPartImage, OpenAIMessageContentPartImageURL,
    OpenAIToolChoiceOption, OpenAIResponseFormat
)

logger = logging.getLogger(__name__)

def _translate_anthropic_messages_to_openai(
    anthropic_messages: List[AnthropicMessage],
    anthropic_system_prompt: Optional[Union[str, List[AnthropicContentBlockText]]]
) -> List[Union[OpenAIChatMessageSystem, OpenAIChatMessageUser, OpenAIChatMessageAssistant, OpenAIChatMessageTool]]:
    """
    Translates a list of Anthropic messages and an optional system prompt
    to a list of OpenAI ChatCompletion messages.
    """
    openai_messages: List[Union[OpenAIChatMessageSystem, OpenAIChatMessageUser, OpenAIChatMessageAssistant, OpenAIChatMessageTool]] = []

    if anthropic_system_prompt:
        system_content = ""
        if isinstance(anthropic_system_prompt, str):
            system_content = anthropic_system_prompt
        elif isinstance(anthropic_system_prompt, list):
            system_content = "\n".join(
                block.text for block in anthropic_system_prompt if isinstance(block, AnthropicContentBlockText)
            )
        
        if system_content and system_content.strip():
            openai_messages.append(OpenAIChatMessageSystem(role="system", content=system_content.strip()))

    for msg in anthropic_messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                # Simple case: content is a string
                openai_messages.append(OpenAIChatMessageUser(role="user", content=msg.content))
            elif isinstance(msg.content, list):
                # Temporary lists to hold parts for this specific Anthropic message
                current_message_text_image_parts: List[Union[OpenAIMessageContentPartText, OpenAIMessageContentPartImage]] = []
                current_message_tool_results: List[OpenAIChatMessageTool] = []

                for block in msg.content:
                    if isinstance(block, AnthropicContentBlockText):
                        current_message_text_image_parts.append(OpenAIMessageContentPartText(type="text", text=block.text))
                    elif isinstance(block, AnthropicContentBlockImage):
                        media_type = block.source.media_type or "image/jpeg"
                        image_url = f"data:{media_type};base64,{block.source.data}"
                        current_message_text_image_parts.append(
                            OpenAIMessageContentPartImage(
                                type="image_url",
                                image_url=OpenAIMessageContentPartImageURL(url=image_url)
                            )
                        )
                    elif isinstance(block, AnthropicContentBlockToolResult):
                        tool_content_str = ""
                        if isinstance(block.content, str):
                            tool_content_str = block.content
                        elif isinstance(block.content, list) or isinstance(block.content, dict): # Handle list or dict
                            tool_content_str = json.dumps(block.content)
                        else: # Fallback for other types (e.g. bool, number)
                            tool_content_str = str(block.content)
                        
                        current_message_tool_results.append(
                            OpenAIChatMessageTool(
                                role="tool",
                                tool_call_id=block.tool_use_id,
                                content=tool_content_str
                            )
                        )
                
                # After processing all blocks in the current Anthropic message:
                # If there were any tool results, add them to the main openai_messages list.
                if current_message_tool_results:
                    openai_messages.extend(current_message_tool_results)
                
                # If there were any text or image parts, they form a single OpenAIChatMessageUser.
                # This user message is added ONLY if there are text/image parts.
                # If the Anthropic message ONLY contained tool results, this block will not execute.
                if current_message_text_image_parts:
                    if len(current_message_text_image_parts) == 1 and current_message_text_image_parts[0].type == "text":
                        openai_messages.append(OpenAIChatMessageUser(role="user", content=current_message_text_image_parts[0].text))
                    else:
                        openai_messages.append(OpenAIChatMessageUser(role="user", content=current_message_text_image_parts))
                # If the Anthropic message content was an empty list from the start,
                # and thus no tool results or text/image parts were processed,
                # create a user message with empty string content.
                elif not msg.content and not current_message_tool_results and not current_message_text_image_parts: # Check if msg.content was originally empty list
                     openai_messages.append(OpenAIChatMessageUser(role="user", content=""))
            # Note: If msg.content is None or some other unexpected type, it won't be handled by above
            # and no user message will be added. This assumes msg.content is always str or list as per AnthropicMessage model.

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
                                type="function", # OpenAI's current tool type is 'function'
                                function=OpenAIFunctionCall(
                                    name=block.name,
                                    arguments=json.dumps(block.input) if isinstance(block.input, dict) else str(block.input)
                                )
                            )
                        )
            
            # Construct assistant message
            final_assistant_text_content: Optional[str] = "\n".join(assistant_text_content_parts) if assistant_text_content_parts else None
            final_tool_calls = assistant_tool_calls if assistant_tool_calls else None

            # OpenAI Assistant message requires 'content' if 'tool_calls' is not present.
            # If 'content' is an empty string and 'tool_calls' is None, it's a valid text message.
            # If 'content' is None and 'tool_calls' is present, it's a valid tool calling message.
            # If both are None (or empty equivalent), 'content' should default to ""
            if final_assistant_text_content is None and not final_tool_calls:
                openai_messages.append(OpenAIChatMessageAssistant(role="assistant", content=""))
            else:
                openai_messages.append(
                    OpenAIChatMessageAssistant(
                        role="assistant",
                        content=final_assistant_text_content, # This can be None if tool_calls are present
                        tool_calls=final_tool_calls
                    )
                )

    # No specific logging here, handled by the calling function if needed.
    return openai_messages

def _translate_anthropic_tools_to_openai(
    anthropic_tools: Optional[List[AnthropicTool]]
) -> Optional[List[OpenAITool]]:
    """
    Translates Anthropic tool definitions to OpenAI tool definitions,
    adjusting string parameter formats for compatibility.
    """
    if anthropic_tools is None: # Explicitly check for None
        return None
    if not anthropic_tools: # Handle empty list
        return []
    
    openai_tools: List[OpenAITool] = []
    
    for tool in anthropic_tools:
        # Convert input_schema to dict, excluding None values to keep schema clean
        params_dict = tool.input_schema.model_dump(exclude_none=True)
        
        # Process properties to remove unsupported string formats
        if "properties" in params_dict and isinstance(params_dict["properties"], dict):
            for prop_name, prop_schema in params_dict["properties"].items():
                if isinstance(prop_schema, dict) and prop_schema.get("type") == "string":
                    if "format" in prop_schema and prop_schema["format"] != "date-time":
                        # If format is not 'date-time', remove it.
                        # OpenAI/Vertex AI supports 'enum' as a keyword, not typically as a 'format' value for strings.
                        # The error message "only 'enum' and 'date-time' are supported for STRING type"
                        # likely means 'enum' as a keyword is fine, but format:"enum_value_itself" is not.
                        # Removing 'format' if it's not 'date-time' is the safest approach.
                        del prop_schema["format"]
        
        openai_tools.append(OpenAITool(
            type="function",
            function=OpenAIFunctionDefinition(
                name=tool.name,
                description=tool.description, # Will be None if not provided, which is fine
                parameters=params_dict
            )
        ))
        
    # Removed logger.warning, no specific info log needed for this sub-function
    return openai_tools

def _translate_anthropic_tool_choice_to_openai(
    anthropic_tool_choice: Optional[AnthropicToolChoice]
) -> Optional[OpenAIToolChoiceOption]:
    """
    Translates Anthropic tool_choice to OpenAI tool_choice.
    """
    if not anthropic_tool_choice:
        return None

    # TODO: Implement detailed tool_choice translation logic
    # - "auto" -> "auto"
    # - "any" -> ??? (OpenAI doesn't have a direct "any". Maybe "auto" or specific logic?)
    # - {"type": "tool", "name": "..."} -> {"type": "function", "function": {"name": "..."}}
    
    logger.warning("_translate_anthropic_tool_choice_to_openai: Detailed translation logic not yet implemented.")
    # Placeholder:
    if hasattr(anthropic_tool_choice, 'type'):
        if anthropic_tool_choice.type == "auto":
            return "auto"
        elif anthropic_tool_choice.type == "tool" and hasattr(anthropic_tool_choice, 'name'):
            return {"type": "function", "function": {"name": anthropic_tool_choice.name}}
    return "auto" # Default fallback

def translate_anthropic_to_openai_request(
    anthropic_request: AnthropicMessagesRequest
) -> OpenAIChatCompletionRequest:
    """
    Translates an Anthropic MessagesRequest to an OpenAI ChatCompletionRequest.
    """
    logger.info(f"Translating Anthropic request (model: {anthropic_request.model}) to OpenAI format.")

    openai_messages = _translate_anthropic_messages_to_openai(
        anthropic_messages=anthropic_request.messages,
        anthropic_system_prompt=anthropic_request.system
    )
    
    openai_tools = _translate_anthropic_tools_to_openai(
        anthropic_tools=anthropic_request.tools
    )
    
    openai_tool_choice = _translate_anthropic_tool_choice_to_openai(
        anthropic_tool_choice=anthropic_request.tool_choice
    )

    # Basic parameter mapping (more nuanced mapping might be needed)
    # Note: LiteLLM will handle some of this, but we pass common ones.
    # The 'model' field from anthropic_request is passed as is to LiteLLM.
    # LiteLLM uses this model identifier for routing.
    
    # OpenAI 'n' is for number of choices, Anthropic doesn't have a direct equivalent. Default to 1.
    # OpenAI 'response_format' could be mapped if Anthropic adds similar features.
    
    request_dict = {
        "model": anthropic_request.model, # Crucial: This is what LiteLLM uses for routing
        "messages": openai_messages,
        "max_tokens": anthropic_request.max_tokens,
        "stream": anthropic_request.stream,
        "temperature": anthropic_request.temperature,
        "top_p": anthropic_request.top_p,
        # "stop": anthropic_request.stop_sequences, # LiteLLM should handle this mapping
        # tools and tool_choice are handled below
    }

    if openai_tools:
        request_dict["tools"] = openai_tools
    if openai_tool_choice:
        request_dict["tool_choice"] = openai_tool_choice
    if anthropic_request.stop_sequences:
         request_dict["stop"] = anthropic_request.stop_sequences


    # Remove None values to keep the request clean, as OpenAI API might be strict
    # Pydantic models will exclude them by default if `exclude_none=True` on dump,
    # but explicit construction is safer here.
    final_request_dict = {k: v for k, v in request_dict.items() if v is not None}

    return OpenAIChatCompletionRequest(**final_request_dict)