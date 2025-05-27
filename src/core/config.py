from typing import Optional, Dict, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

_DEFAULT_KEY_FOR_OPENAI_PROVIDER_MODEL = "anthropic/_internal_default_openai_provider_model"
_DEFAULT_KEY_FOR_GEMINI_PROVIDER_MODEL = "anthropic/_internal_default_gemini_provider_model"


class AnthropicModelInfo(BaseSettings):
    description: str = ""


class Settings(BaseSettings):
    PORT: int = 8080
    LOG_LEVEL: str = "INFO"
    TARGET_LLM_PROVIDER: str = "gemini"
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODELS_INFO: Dict[str, AnthropicModelInfo] = Field(
        default_factory=lambda: {
            "anthropic/claude-3-opus": AnthropicModelInfo(
                description="Anthropic's most powerful model, for highly complex tasks."
            ),
            "anthropic/claude-3.5-sonnet": AnthropicModelInfo(
                description="Anthropic's latest balanced model, excelling at intelligence and speed."
            ),
            "anthropic/claude-3-sonnet": AnthropicModelInfo(
                description="Anthropic's balanced model for intelligence and speed (previous generation to 3.5)."
            ),
            "anthropic/claude-3-haiku": AnthropicModelInfo(
                description="Anthropic's fastest and most compact model for near-instant responsiveness."
            ),
        }
    )

    ANTHROPIC_TO_OPENAI_MAP: Dict[str, str] = Field(
        default_factory=lambda: {
            "anthropic/claude-3-opus": "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet": "openai/gpt-4o",
            "anthropic/claude-3-sonnet": "openai/gpt-4-turbo",
            "anthropic/claude-3-haiku": "openai/gpt-4o-mini",
            _DEFAULT_KEY_FOR_OPENAI_PROVIDER_MODEL: "openai/gpt-3.5-turbo",
        }
    )

    ANTHROPIC_TO_GEMINI_MAP: Dict[str, str] = Field(
        default_factory=lambda: {
            "anthropic/claude-3-opus": "gemini/gemini-1.5-pro-latest",
            "anthropic/claude-3.5-sonnet": "gemini/gemini-1.5-pro-latest",
            "anthropic/claude-3-sonnet": "gemini/gemini-1.5-pro-latest",
            "anthropic/claude-3-haiku": "gemini/gemini-1.5-flash-latest",
            _DEFAULT_KEY_FOR_GEMINI_PROVIDER_MODEL: "gemini/gemini-1.5-pro-latest",
        }
    )

    OPENAI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY: str = _DEFAULT_KEY_FOR_OPENAI_PROVIDER_MODEL
    GEMINI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY: str = _DEFAULT_KEY_FOR_GEMINI_PROVIDER_MODEL

    MODEL_METADATA_SOURCES: List[str] = Field(
        default_factory=lambda: [
            "https://openrouter.ai/models?arch=Claude&fmt=table",
            "https://openrouter.ai/models?arch=GPT&fmt=table",
            "https://openrouter.ai/models?arch=Gemini&fmt=table",
            "Official Anthropic Documentation URL",
            "Official OpenAI Documentation URL",
            "Official Google AI Documentation URL",
        ]
    )
    MODEL_MAP_LAST_UPDATED: str = "2025-05-27"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
