import logging
import sys
from src.core.config import settings # Import the new settings

def setup_logging():
    """
    Configures basic logging for the application.
    The log level is determined by the LOG_LEVEL setting in the configuration.
    """
    log_level_str = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout) # Changed to stdout for consistency
        ]
    )

    # Configure uvicorn loggers to be less verbose or match our app's level
    # This helps in reducing noise from uvicorn's own logging
    logging.getLogger("uvicorn").setLevel(max(log_level, logging.INFO))
    logging.getLogger("uvicorn.access").setLevel(max(log_level, logging.INFO)) # Access logs can be noisy
    logging.getLogger("uvicorn.error").setLevel(max(log_level, logging.INFO))

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level_str}")
    
    return logger

# If you want a globally accessible logger instance initialized by this module:
# logger = setup_logging()
# Otherwise, other modules can call `logging.getLogger(__name__)`
# after `setup_logging()` has been called once (e.g., in main.py).
