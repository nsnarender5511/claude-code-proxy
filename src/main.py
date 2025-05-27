from fastapi import FastAPI, Request
from src.core.logging_setup import setup_logging # Updated import
from src.api.v1.endpoints import messages as messages_v1 # Updated import
from src.api.v1.endpoints import health as health_v1   # Updated import
from src.core.config import settings # Updated import
from loguru import logger

# Initialize logging
setup_logging()

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

# Include v1 routers
app.include_router(messages_v1.router, prefix='/v1')
app.include_router(health_v1.router, prefix='/v1') # Typically health might be at / or /health, adjust if needed


@app.get('/')
async def root():
    return {'message': 'Anthropic SDK Facade for LiteLLM is running.'}


if __name__ == '__main__':
    import uvicorn

    logger.info(f'Starting Uvicorn server on http://0.0.0.0:{settings.PORT}')
    uvicorn.run(app, host='0.0.0.0', port=settings.PORT, log_level=settings.LOG_LEVEL.lower()) 