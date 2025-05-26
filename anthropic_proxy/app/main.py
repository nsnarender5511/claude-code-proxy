from fastapi import FastAPI, Request
from app.core.logging_config import setup_logging
from app.api.endpoints import messages, token_counter # Routers
import logging # For the middleware logger

# Apply logging configuration
setup_logging()

# Initialize logger for middleware if needed, or use a generic one
middleware_logger = logging.getLogger("app.middleware")

app = FastAPI(title="Anthropic Proxy for LiteLLM")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path
    
    # Log only basic request details at debug level
    middleware_logger.debug(f"Request: {method} {path}") # Changed to middleware_logger
    
    # Process the request and get the response
    response = await call_next(request)
    
    return response

# Include API routers
app.include_router(messages.router)
app.include_router(token_counter.router)

@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}
