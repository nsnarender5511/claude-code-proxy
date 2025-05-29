from fastapi import FastAPI, Request
import logging
from src.core.logging_setup import setup_logging
from src.api.v1.endpoints import messages as messages_v1
from src.api.v1.endpoints import health as health_v1
from src.core.config import settings
from src.core.opentelemetry_setup import setup_opentelemetry

logger = logging.getLogger(__name__)
setup_logging()
app = FastAPI(
    title="Anthropic SDK Facade for LiteLLM",
    version="1.1.0",
    description="Translates Anthropic SDK requests to LiteLLM's OpenAI-compatible endpoint.",
)


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    logger.debug(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.debug(f"Outgoing response: {response.status_code}")
    return response


app.include_router(messages_v1.router, prefix="/v1")
app.include_router(health_v1.router, prefix="/v1")


@app.get("/")
async def root():
    return {"message": "Anthropic SDK Facade for LiteLLM is running."}


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Uvicorn server on http://0.0.0.0:{settings.PORT}")

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(request_line)s ✓ %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.PORT,
        log_config=LOGGING_CONFIG,
    )
