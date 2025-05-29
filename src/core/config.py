from typing import Optional, Dict, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

_DEFAULT_KEY_FOR_OPENAI_PROVIDER_MODEL = (
    "anthropic/_internal_default_openai_provider_model"
)
_DEFAULT_KEY_FOR_GEMINI_PROVIDER_MODEL = (
    "anthropic/_internal_default_gemini_provider_model"
)


class AnthropicModelInfo(BaseSettings):
    description: str = ""


class Settings(BaseSettings):
    PORT: int = 8080
    LOG_LEVEL: str = "INFO"
    TARGET_LLM_PROVIDER: str = "gemini"
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_TO_OPENAI_MAP: Dict[str, str] = Field(
        default_factory=lambda: {
            "claude-3-5-haiku-20241022": "openai/gpt-o3",
            "anthropic/claude-3-opus": "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet": "openai/gpt-4o",
            "anthropic/claude-3-sonnet": "openai/gpt-4-turbo",
            "anthropic/claude-3-haiku": "openai/gpt-4o-mini",
            _DEFAULT_KEY_FOR_OPENAI_PROVIDER_MODEL: "openai/gpt-3.5-turbo",
        }
    )
    ANTHROPIC_TO_GEMINI_MAP: Dict[str, str] = Field(
        default_factory=lambda: {
            "claude-3-5-haiku-20241022": "gemini-2.5-flash-preview-04-17",

            "anthropic/claude-3-opus": "gemini/gemini-1.5-pro-latest",
            "anthropic/claude-3.5-sonnet": "gemini/gemini-1.5-pro-latest",
            "anthropic/claude-3-sonnet": "gemini/gemini-1.5-pro-latest",
            "anthropic/claude-3-haiku": "gemini/gemini-1.5-flash-latest",
            _DEFAULT_KEY_FOR_GEMINI_PROVIDER_MODEL: "gemini-2.5-pro-preview-03-25",
        }
    )
    OPENAI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY: str = (
        _DEFAULT_KEY_FOR_OPENAI_PROVIDER_MODEL
    )
    GEMINI_PROVIDER_DEFAULT_MODEL_TRANSLATION_KEY: str = (
        _DEFAULT_KEY_FOR_GEMINI_PROVIDER_MODEL
    )
    OTEL_SERVICE_NAME: str = "claude-code-proxy"
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None
    OTEL_EXPORTER_OTLP_PROTOCOL: str = "grpc"
    OTEL_EXPORTER_OTLP_HEADERS: Optional[str] = None
    OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED: bool = True
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
