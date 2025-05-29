import logging
from typing import Dict, Any, Optional, Union, AsyncGenerator
from src.api.v1.schemas.anthropic_api import AnthropicSSEError, AnthropicSSEErrorContent

logger = logging.getLogger(__name__)


def translate_openai_error_to_anthropic_format(
    error_details: Dict[str, Any],
) -> Dict[str, Any]:
    if isinstance(error_details, dict):
        err_obj_for_log = error_details.get("error", error_details)
        log_msg = f"Translating error to Anthropic format. Input error type: {err_obj_for_log.get('type')}, message: {err_obj_for_log.get('message')}, code: {err_obj_for_log.get('code')}"
        if "param" in err_obj_for_log:
            log_msg += f", param: {err_obj_for_log.get('param')}"
        logger.debug(log_msg)
    else:
        logger.debug(
            f"Translating error to Anthropic format (non-dict input): {type(error_details)}"
        )
    error_type = "api_error"
    error_message = "An unexpected error occurred."
    if isinstance(error_details, dict):
        err_obj = error_details.get("error", error_details)
        if isinstance(err_obj, str):
            error_message = err_obj
        elif isinstance(err_obj, dict):
            error_message = err_obj.get("message", error_message)
            raw_type = err_obj.get("type")
            if raw_type:
                raw_type_lower = raw_type.lower()
                if (
                    "auth" in raw_type_lower
                    or "permission" in raw_type_lower
                    or "key" in raw_type_lower
                ):
                    error_type = "authentication_error"
                elif "rate_limit" in raw_type_lower:
                    error_type = "rate_limit_error"
                elif (
                    "invalid_request" in raw_type_lower
                    or "validation" in raw_type_lower
                    or "bad_request" in raw_type_lower
                ):
                    error_type = "invalid_request_error"
                elif (
                    "not_found" in raw_type_lower or "model_not_found" in raw_type_lower
                ):
                    error_type = "not_found_error"
                elif (
                    "overloaded" in raw_type_lower
                    or "capacity" in raw_type_lower
                    or "unavailable" in raw_type_lower
                ):
                    error_type = "overloaded_error"
                else:
                    error_type = "api_error"
    elif isinstance(error_details, Exception):
        error_message = str(error_details)
        if isinstance(error_details, ValueError):
            error_type = "invalid_request_error"
        elif isinstance(error_details, ConnectionError):
            error_type = "api_connection_error"
    return {"type": "error", "error": {"type": error_type, "message": error_message}}


async def generate_anthropic_error_sse_event(
    error_type_str: str, message: str
) -> AsyncGenerator[str, None]:
    logger.debug(
        f"Generating Anthropic SSE error event: type='{error_type_str}', message='{message}'"
    )
    error_content = AnthropicSSEErrorContent(type=error_type_str, message=message)
    error_event_model = AnthropicSSEError(type="error", error=error_content)
    yield f"event: error\ndata: {error_event_model.model_dump_json()}\n\n"
