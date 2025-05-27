import sys
from loguru import logger
from src.core.config import settings
import logging

class InterceptHandler(logging.Handler):
    def emit(self, record):
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

    return logger # Return the configured logger for convenience if needed 