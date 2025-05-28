from fastapi import FastAPI, Request
from src.core.logging_config import setup_logging
from src.api.endpoints import messages, token_counter # Routers
import logging # For the middleware logger
from src.core.config import PORT

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

if __name__ == "__main__":
    import uvicorn
    import sys

    # Note: The app instance is already defined above as 'app'
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("To run the server for development (with auto-reload):")
        print("  uvicorn src.main:app --reload --host 0.0.0.0 --port 8082")
        print("\nTo run the server directly using this script:")
        print("  python src/main.py")
        sys.exit(0)
    
    port = PORT # Default port

    print(f"Starting Uvicorn server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
