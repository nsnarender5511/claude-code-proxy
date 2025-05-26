import logging
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Union, Literal

# Import configuration variables used in validators
from src.core.config import (
    PREFERRED_PROVIDER, BIG_MODEL, SMALL_MODEL,
    GEMINI_MODELS, OPENAI_MODELS
)

# Initialize a logger for this module
logger = logging.getLogger(__name__)

# Shared model validation logic
def _validate_and_map_model_shared(v: str, field_values: Dict[str, Any], context_str: str = "") -> str:
    original_model = v
    new_model = v 

    logger.debug(f"📋 {context_str} MODEL VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

    clean_v = v
    if clean_v.startswith('anthropic/'):
        clean_v = clean_v[10:]
    elif clean_v.startswith('openai/'):
        clean_v = clean_v[7:]
    elif clean_v.startswith('gemini/'):
        clean_v = clean_v[7:]

    mapped = False
    if 'haiku' in clean_v.lower():
        if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
            new_model = f"gemini/{SMALL_MODEL}"
            mapped = True
        else:
            new_model = f"openai/{SMALL_MODEL}"
            mapped = True
    elif 'sonnet' in clean_v.lower():
        if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
            new_model = f"gemini/{BIG_MODEL}"
            mapped = True
        else:
            new_model = f"openai/{BIG_MODEL}"
            mapped = True
    elif not mapped:
        if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
            new_model = f"gemini/{clean_v}"
            mapped = True
        elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
            new_model = f"openai/{clean_v}"
            mapped = True
    
    if mapped:
        logger.debug(f"📌 {context_str} MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
        field_values['original_model'] = original_model 
    elif not v.startswith(('openai/', 'gemini/', 'anthropic/')):
        logger.warning(f"⚠️ No prefix or mapping rule for {context_str.lower()} model: '{original_model}'. Using as is.")
        field_values['original_model'] = original_model # Still store original if no mapping but also no prefix
    else:
        # If it has a prefix but no mapping rule applied, store it as original_model too
        # This ensures original_model is always populated if model field is present.
        field_values['original_model'] = original_model

    return new_model

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]]

# Moved ThinkingConfig before SystemContent
class ThinkingConfig(BaseModel):
    enabled: bool

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str
    tool_choice: Optional[Dict[str, Any]] = None

class Message(BaseModel):
    role: Literal["user", "assistant"] 
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @model_validator(mode='before')
    @classmethod
    def populate_original_model(cls, data: Any) -> Any:
        if isinstance(data, dict) and 'model' in data and 'original_model' not in data:
            data['original_model'] = data['model']
        return data

    @field_validator('model')
    @classmethod
    def validate_model_field(cls, v: str, info: Any) -> str:
        current_values = info.data if hasattr(info, 'data') else getattr(info, 'values', info if isinstance(info, dict) else {})
        if not isinstance(current_values, dict): current_values = {} # Ensure it's a dict
        return _validate_and_map_model_shared(v, current_values, "MESSAGES REQUEST")

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @model_validator(mode='before')
    @classmethod
    def populate_original_model_token(cls, data: Any) -> Any:
        if isinstance(data, dict) and 'model' in data and 'original_model' not in data:
            data['original_model'] = data['model']
        return data

    @field_validator('model')
    @classmethod
    def validate_model_token_count(cls, v: str, info: Any) -> str:
        current_values = info.data if hasattr(info, 'data') else getattr(info, 'values', info if isinstance(info, dict) else {})
        if not isinstance(current_values, dict): current_values = {} # Ensure it's a dict
        return _validate_and_map_model_shared(v, current_values, "TOKEN COUNT")

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage
