import logging
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal

# Initialize a logger for this module
logger = logging.getLogger(__name__)

# --- Anthropic SDK Compatible Models ---

# Content Blocks for Anthropic Messages
class AnthropicContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class AnthropicContentBlockImageSource(BaseModel):
    type: Literal["base64"]
    media_type: str # e.g., "image/jpeg", "image/png"
    data: str

class AnthropicContentBlockImage(BaseModel):
    type: Literal["image"]
    source: AnthropicContentBlockImageSource

class AnthropicContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class AnthropicContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]]] # Simplified based on common usage, can be expanded
    is_error: Optional[bool] = None # Anthropic supports this

# Anthropic Message
class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[
        AnthropicContentBlockText,
        AnthropicContentBlockImage,
        AnthropicContentBlockToolUse,
        AnthropicContentBlockToolResult
    ]]]

# Anthropic Tool Definition
class AnthropicToolInputSchema(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, Any]
    required: Optional[List[str]] = None

class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: AnthropicToolInputSchema

# Anthropic Tool Choice
class AnthropicToolChoiceAuto(BaseModel):
    type: Literal["auto"]

class AnthropicToolChoiceAny(BaseModel):
    type: Literal["any"]

class AnthropicToolChoiceTool(BaseModel):
    type: Literal["tool"]
    name: str

AnthropicToolChoice = Union[AnthropicToolChoiceAuto, AnthropicToolChoiceAny, AnthropicToolChoiceTool]

class ThinkingConfig(BaseModel):
    enabled: bool
    budget_tokens: Optional[int] = None
    type: Optional[str] = None


# Anthropic Messages Request
class AnthropicMessagesRequest(BaseModel):
    model: str # This will be passed to LiteLLM, e.g., "openai/gpt-4o", "gemini/gemini-pro"
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[AnthropicContentBlockText]]] = None # Can be string or list of text blocks
    max_tokens: int
    metadata: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None # Default 1.0 by Anthropic
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    thinking: Optional[ThinkingConfig] = None # For advanced thinking configurations

# Anthropic Usage Information
class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int

# Anthropic Messages Response (Non-Streaming)
class AnthropicMessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str # Model that generated the response (echoed back)
    content: List[Union[AnthropicContentBlockText, AnthropicContentBlockToolUse]]
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "content_filtered"]] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage

# --- Placeholder for OpenAI Models (to be added next) ---

# --- Placeholder for SSE Event Models (to be added later) ---

# --- OpenAI SDK Compatible Models ---

# Based on OpenAI's Chat Completion API

# Message Content Parts (for vision-compatible models)
class OpenAIMessageContentPartText(BaseModel):
    type: Literal["text"]
    text: str

class OpenAIMessageContentPartImageURL(BaseModel):
    url: str # e.g., "data:image/jpeg;base64,{base64_image_data}" or "http://..."
    detail: Optional[Literal["auto", "low", "high"]] = "auto"

class OpenAIMessageContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: OpenAIMessageContentPartImageURL


# General Message (can be part of a list for 'content' field)
OpenAIMessageContent = Union[str, List[Union[OpenAIMessageContentPartText, OpenAIMessageContentPartImage]]]


# Tool Calls
class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: str # JSON string

class OpenAIToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall

# Messages for Request
class OpenAIChatMessageSystem(BaseModel):
    role: Literal["system"]
    content: str
    name: Optional[str] = None

class OpenAIChatMessageUser(BaseModel):
    role: Literal["user"]
    content: OpenAIMessageContent
    name: Optional[str] = None

class OpenAIChatMessageAssistant(BaseModel):
    role: Literal["assistant"]
    content: Optional[str] = None # Can be None if tool_calls are present
    name: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

class OpenAIChatMessageTool(BaseModel):
    role: Literal["tool"]
    content: str
    tool_call_id: str

OpenAIChatCompletionRequestMessage = Union[
    OpenAIChatMessageSystem,
    OpenAIChatMessageUser,
    OpenAIChatMessageAssistant,
    OpenAIChatMessageTool
]

# Tool Definition for Request
class OpenAIFunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] # JSON Schema object

class OpenAITool(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIFunctionDefinition

# Tool Choice for Request
OpenAIToolChoiceOption = Union[
    Literal["none", "auto"],
    Dict[Literal["type", "function"], Union[Literal["function"], Dict[Literal["name"], str]]]
] # e.g., {"type": "function", "function": {"name": "my_function"}}

class OpenAIResponseFormat(BaseModel):
    type: Optional[Literal["text", "json_object"]] = None


# OpenAI Chat Completion Request
class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIChatCompletionRequestMessage]
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1 # How many chat completion choices to generate for each input message.
    presence_penalty: Optional[float] = None
    response_format: Optional[OpenAIResponseFormat] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[OpenAIToolChoiceOption] = None
    user: Optional[str] = None # A unique identifier representing your end-user

# OpenAI Chat Completion Response (Non-Streaming)
class OpenAIChatCompletionChoiceDelta(BaseModel): # For streaming
    content: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None # Note: tool_calls in delta might be partial
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None

class OpenAIChatCompletionChoiceMessage(BaseModel): # For non-streaming response
    role: Literal["assistant"]
    content: Optional[str] = None # Null if tool_calls are present
    tool_calls: Optional[List[OpenAIToolCall]] = None

class OpenAIChatCompletionChoice(BaseModel):
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None # function_call is legacy
    index: int
    message: OpenAIChatCompletionChoiceMessage
    delta: Optional[OpenAIChatCompletionChoiceDelta] = None # Only for streaming
    logprobs: Optional[Any] = None # Complex object, define if needed

class OpenAICompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class OpenAIChatCompletionResponse(BaseModel):
    id: str
    choices: List[OpenAIChatCompletionChoice]
    created: int # Unix timestamp
    model: str # Model ID used
    system_fingerprint: Optional[str] = None
    object: Literal["chat.completion"] = "chat.completion"
    usage: Optional[OpenAICompletionUsage] = None

# --- SSE Event Models ---

# Anthropic SSE Stream Events
# Based on https://docs.anthropic.com/claude/reference/messages-streaming

class AnthropicSSEMessageStart(BaseModel):
    type: Literal["message_start"]
    message: AnthropicMessagesResponse # Contains id, type, role, model, usage

class AnthropicSSEContentBlockStart(BaseModel):
    type: Literal["content_block_start"]
    index: int
    content_block: Union[AnthropicContentBlockText, AnthropicContentBlockToolUse] # Only type and for tool_use: id, name

class AnthropicSSEContentBlockDelta(BaseModel):
    type: Literal["content_block_delta"]
    index: int
    delta: Union[AnthropicContentBlockText, Dict[Literal["type", "input"], str]] # For text: {"type": "text_delta", "text": "..."}; For tool_use input: {"type": "input_json_delta", "input": "..."}

class AnthropicSSEContentBlockStop(BaseModel):
    type: Literal["content_block_stop"]
    index: int

class AnthropicSSEMessageDelta(BaseModel):
    type: Literal["message_delta"]
    delta: Dict[Literal["stop_reason", "stop_sequence"], Any] # e.g. {"stop_reason": "max_tokens", "stop_sequence": null}
    usage: AnthropicUsage # Contains output_tokens for this delta

class AnthropicSSEMessageStop(BaseModel):
    type: Literal["message_stop"]

# General Ping event (Anthropic)
class AnthropicSSEPing(BaseModel):
    type: Literal["ping"]

# Error event (Anthropic)
class AnthropicSSEErrorContent(BaseModel):
    type: str # e.g., "overloaded_error"
    message: str

class AnthropicSSEError(BaseModel):
    type: Literal["error"]
    error: AnthropicSSEErrorContent


# OpenAI SSE Stream Events (Chat Completion Chunks)
# Based on OpenAI's API documentation for streaming

class OpenAIChatCompletionChunkChoiceDelta(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None # Note: tool_calls in delta might be partial, especially function.arguments
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    function_call: Optional[Dict[str, str]] = None # Legacy, for older models

class OpenAIChatCompletionChunkChoice(BaseModel):
    delta: OpenAIChatCompletionChunkChoiceDelta
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    index: int
    logprobs: Optional[Any] = None # Define if needed

class OpenAIChatCompletionChunk(BaseModel):
    id: str # Stream ID
    choices: List[OpenAIChatCompletionChunkChoice]
    created: int # Unix timestamp
    model: str # Model ID
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    system_fingerprint: Optional[str] = None
    usage: Optional[OpenAICompletionUsage] = None # Typically None until the final chunk in some implementations or present in Azure OpenAI
