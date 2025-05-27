import logging
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal

logger = logging.getLogger(__name__)


class AnthropicContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class AnthropicContentBlockImageSource(BaseModel):
    type: Literal["base64"]
    media_type: Optional[str] = None
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
    content: Union[str, List[Dict[str, Any]]]
    is_error: Optional[bool] = None


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                AnthropicContentBlockText,
                AnthropicContentBlockImage,
                AnthropicContentBlockToolUse,
                AnthropicContentBlockToolResult,
            ]
        ],
    ]


class AnthropicToolInputSchema(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: Optional[List[str]] = None


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: AnthropicToolInputSchema


class AnthropicToolChoiceAuto(BaseModel):
    type: Literal["auto"]


class AnthropicToolChoiceAny(BaseModel):
    type: Literal["any"]


class AnthropicToolChoiceTool(BaseModel):
    type: Literal["tool"]
    name: str


AnthropicToolChoice = Union[
    AnthropicToolChoiceAuto, AnthropicToolChoiceAny, AnthropicToolChoiceTool
]


class ThinkingConfig(BaseModel):
    enabled: bool
    budget_tokens: Optional[int] = None
    type: Optional[str] = None


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[AnthropicContentBlockText]]] = None
    max_tokens: int
    metadata: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    thinking: Optional[ThinkingConfig] = None


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class AnthropicMessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: List[Union[AnthropicContentBlockText, AnthropicContentBlockToolUse]]
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "content_filtered"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


class OpenAIMessageContentPartText(BaseModel):
    type: Literal["text"]
    text: str


class OpenAIMessageContentPartImageURL(BaseModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class OpenAIMessageContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: OpenAIMessageContentPartImageURL


OpenAIMessageContent = Union[
    str, List[Union[OpenAIMessageContentPartText, OpenAIMessageContentPartImage]]
]


class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: str


class OpenAIToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall


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
    content: Optional[str] = None
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
    OpenAIChatMessageTool,
]


class OpenAIFunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]


class OpenAITool(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIFunctionDefinition


OpenAIToolChoiceOption = Union[
    Literal["none", "auto"],
    Dict[Literal["type", "function"], Union[Literal["function"], Dict[Literal["name"], str]]],
]


class OpenAIResponseFormat(BaseModel):
    type: Optional[Literal["text", "json_object"]] = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIChatCompletionRequestMessage]
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = None
    response_format: Optional[OpenAIResponseFormat] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[OpenAIToolChoiceOption] = None
    user: Optional[str] = None


class OpenAIChatCompletionChoiceDelta(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None


class OpenAIChatCompletionChoiceMessage(BaseModel):
    role: Literal["assistant"]
    content: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None


class OpenAIChatCompletionChoice(BaseModel):
    finish_reason: Optional[
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    ] = None
    index: int
    message: OpenAIChatCompletionChoiceMessage
    delta: Optional[OpenAIChatCompletionChoiceDelta] = None
    logprobs: Optional[Any] = None


class OpenAICompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    choices: List[OpenAIChatCompletionChoice]
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    object: Literal["chat.completion"] = "chat.completion"
    usage: Optional[OpenAICompletionUsage] = None


class AnthropicSSEMessageStart(BaseModel):
    type: Literal["message_start"]
    message: AnthropicMessagesResponse


class AnthropicSSEContentBlockStart(BaseModel):
    type: Literal["content_block_start"]
    index: int
    content_block: Union[AnthropicContentBlockText, AnthropicContentBlockToolUse]


class AnthropicSSEContentBlockDelta(BaseModel):
    type: Literal["content_block_delta"]
    index: int
    delta: Union[AnthropicContentBlockText, Dict[Literal["type", "input"], str]]


class AnthropicSSEContentBlockStop(BaseModel):
    type: Literal["content_block_stop"]
    index: int


class AnthropicSSEMessageDelta(BaseModel):
    type: Literal["message_delta"]
    delta: Dict[Literal["stop_reason", "stop_sequence"], Any]
    usage: AnthropicUsage


class AnthropicSSEMessageStop(BaseModel):
    type: Literal["message_stop"]


class AnthropicSSEPing(BaseModel):
    type: Literal["ping"]


class AnthropicSSEErrorContent(BaseModel):
    type: str
    message: str


class AnthropicSSEError(BaseModel):
    type: Literal["error"]
    error: AnthropicSSEErrorContent


class OpenAIChatCompletionChunkChoiceDelta(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    function_call: Optional[Dict[str, str]] = None


class OpenAIChatCompletionChunkChoice(BaseModel):
    delta: OpenAIChatCompletionChunkChoiceDelta
    finish_reason: Optional[
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    ] = None
    index: int
    logprobs: Optional[Any] = None


class OpenAIChatCompletionChunk(BaseModel):
    id: str
    choices: List[OpenAIChatCompletionChunkChoice]
    created: int
    model: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    system_fingerprint: Optional[str] = None
    usage: Optional[OpenAICompletionUsage] = None
