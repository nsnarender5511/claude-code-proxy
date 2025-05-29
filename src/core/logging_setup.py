import sys
from loguru import logger
from src.core.config import settings
import logging

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Only allow logs from the 'src' namespace to be processed by Loguru.
        # All other logs (e.g., 'litellm', 'some_other_library') will be dropped here.
        if not record.name.startswith('src'):
            return # Suppress the log

        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging():
    log_level_str = settings.LOG_LEVEL.upper()
    # Ensure log_level_str is a valid Loguru level, default to INFO
    valid_loguru_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    if log_level_str not in valid_loguru_levels:
        logger.warning(f"Invalid LOG_LEVEL '{log_level_str}' in settings, defaulting to INFO.")
        log_level_str = "INFO"

    logger.remove() # Remove default handler
    logger.add(
        sys.stdout,
        level=log_level_str,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    logger.info(f"Log level set to: {log_level_str}")

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logger.info("Standard logging intercepted and redirected to Loguru.")

    # Attempt to control LiteLLM's logger level
    logger.info("Attempting to forcefully disable all loggers starting with 'litellm'.")
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith('litellm'):
            current_logger = logging.getLogger(logger_name)
            current_logger.setLevel(logging.CRITICAL)
            for handler in list(current_logger.handlers): # Iterate over a copy
                current_logger.removeHandler(handler)
            current_logger.propagate = False # Stop it from propagating to root logger
            logger.debug(f"Disabled logger '{logger_name}' by setting level to CRITICAL, removing handlers, and disabling propagation.")

    # Also set the root 'litellm' logger just in case it's used directly or new sub-loggers are created later
    litellm_root_logger = logging.getLogger("litellm")
    litellm_root_logger.setLevel(logging.CRITICAL) # Set to highest level
    # Remove any existing handlers
    for handler in list(litellm_root_logger.handlers):
        litellm_root_logger.removeHandler(handler)
    # Add a NullHandler to prevent messages from reaching the root logger's handlers
    # or logging.lastResort if propagate is True for some reason on a child.
    litellm_root_logger.addHandler(logging.NullHandler())
    litellm_root_logger.propagate = False # Explicitly stop propagation
    logger.info(f"Forcefully disabled root 'litellm' logger by setting level to CRITICAL, removing other handlers, adding NullHandler, and disabling propagation.")

    # The loop for disabling other third-party library logs is removed,
    # as InterceptHandler now handles this dynamically.
    logger.info("Third-party log filtering is now handled by InterceptHandler.")

    return logger # Return the configured logger for convenience if needed 