from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional, Union, Literal


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

# SSE Models
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