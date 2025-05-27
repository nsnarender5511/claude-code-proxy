import time
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union, Literal

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