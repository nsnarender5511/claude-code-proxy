import uvicorn
import sys
from app.main import app # Import the FastAPI app instance

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn anthropic_proxy.app.main:app --reload --host 0.0.0.0 --port 8082")
        print("Or simply: python anthropic_proxy/run.py")
        sys.exit(0)
    
    # Get host and port from environment variables or use defaults
    # This is optional but good practice. For now, hardcoding is fine as in the original.
    host = "0.0.0.0"
    port = 8082
    
    print(f"Starting Uvicorn server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info") # Changed log_level from error to info for better default visibility
