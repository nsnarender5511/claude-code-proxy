import logging
import sys
from src.core.config import settings


def setup_logging():
    log_level_str = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger('uvicorn').setLevel(max(log_level, logging.INFO))
    logging.getLogger('uvicorn.access').setLevel(max(log_level, logging.INFO))
    logging.getLogger('uvicorn.error').setLevel(max(log_level, logging.INFO))
    logger = logging.getLogger(__name__)
    logger.info(f'Logging configured with level: {log_level_str}')
    return logger 