import sys
import logging
from src.core.config import settings

# ANSI escape codes for colors
COLORS = {
    "WARNING": "\\033[93m",  # Yellow
    "INFO": "\\033[92m",  # Green
    "DEBUG": "\\033[94m",  # Blue
    "CRITICAL": "\\033[91m",  # Red
    "ERROR": "\\033[91m",  # Red
    "RESET": "\\033[0m",  # Reset color
}

SEPARATOR = " | "


class ColorizedFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None, style="%", validate=True):
        super().__init__(fmt, datefmt, style, validate)

    def format(self, record):
        log_message = super().format(record)
        level_color = COLORS.get(record.levelname, COLORS["RESET"])
        return f"{level_color}{log_message}{COLORS['RESET']}"


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        logger = logging.getLogger("src")
        level = record.levelno
        original_logger = logging.getLogger(record.name)
        if not record.name.startswith("src"):
            fn, lno, func, sinfo = ("(unknown file)", 0, "(unknown function)", None)
            if record.exc_info:
                sinfo = record.stack_info
            effective_logger = logging.getLogger("src.intercepted." + record.name)
            effectiv


class SrcFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith("src")


def setup_logging():
    log_level_str = settings.LOG_LEVEL.upper()
    numeric_log_level = getattr(logging, log_level_str, logging.INFO)
    src_logger = logging.getLogger("src")
    src_logger.setLevel(numeric_log_level)
    for handler in src_logger.handlers[:]:
        src_logger.removeHandler(handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_log_level)

    # Updated format string with more separators
    format_string = f"%(message)s"
    formatter = ColorizedFormatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler.setFormatter(formatter)
    console_handler.addFilter(SrcFilter())
    src_logger.addHandler(console_handler)
    src_logger.propagate = False
    src_logger.info(
        f"Logging setup for 'src' namespace complete. Log level: {log_level_str}"
    )
    logging.getLogger("litellm").setLevel(logging.WARNING)
    litellm_logger = logging.getLogger("litellm")
    litellm_logger.setLevel(logging.ERROR)
    src_logger.info("Standard logging for 'litellm' set to ERROR.")
