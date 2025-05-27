from fastapi import FastAPI, Request
from src.core.logging_config import setup_logging
from src.api.endpoints import messages
from src.core.config import settings
import logging

setup_logging()
logger = logging.getLogger(__name__)
app = FastAPI(
    title='Anthropic SDK Facade for LiteLLM',
    version='1.1.0',
    description="Translates Anthropic SDK requests to LiteLLM's OpenAI-compatible endpoint.",
)


@app.middleware('http')
async def log_requests_middleware(request: Request, call_next):
    logger.debug(f'Incoming request: {request.method} {request.url.path}')
    response = await call_next(request)
    logger.debug(f'Outgoing response: {response.status_code}')
    return response


app.include_router(messages.router, prefix='/v1')


@app.get('/')
async def root():
    return {'message': 'Anthropic SDK Facade for LiteLLM is running.'}


if __name__ == '__main__':
    import uvicorn

    logger.info(f'Starting Uvicorn server on http://0.0.0.0:{settings.PORT}')
    uvicorn.run(app, host='0.0.0.0', port=settings.PORT, log_level=settings.LOG_LEVEL.lower())
