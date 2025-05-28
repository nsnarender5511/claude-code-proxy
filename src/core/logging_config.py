import logging
import uvicorn
import sys # Needed for sys.stderr in setup_logging

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:", 
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record):
        if record.levelno == logging.DEBUG and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

def setup_logging():
    # Configure logging
    logging.basicConfig(
        level=logging.WARN,  # Change to INFO level to show more details
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    # Configure uvicorn to be quieter
    # Tell uvicorn's loggers to be quiet
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    # Apply the filter to the root logger to catch all messages
    root_logger = logging.getLogger()
    root_logger.addFilter(MessageFilter())

    # Apply custom formatter to console handler
    # Ensure there's a console handler to apply the formatter to
    console_handler_exists = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr: # Default stream for basicConfig
            handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))
            console_handler_exists = True
            break
    
    if not console_handler_exists:
        # If basicConfig didn't set up a handler (e.g. if root logger already had handlers)
        # or if we want to be absolutely sure, we can add one.
        # However, basicConfig should add a StreamHandler by default if no handlers are configured for the root logger.
        # For now, we'll assume basicConfig does its job.
        # If issues arise, one might add a handler explicitly:
        # console_handler = logging.StreamHandler(sys.stderr)
        # console_handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        # logger.addHandler(console_handler)
        # if logger is not root_logger: # If using a specific logger instance
        #     root_logger.addHandler(console_handler) # Also add to root if needed
        pass
    
    return logger

# Define ANSI color codes for terminal output (moved from server.py)
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"

# Moved from server.py
def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"
    
    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    # Format status code
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    
    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"
    
    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()

# Also, ensure the logger instance is available for other modules if they import it from here.
# Typically, modules will call `logging.getLogger(__name__)` themselves.
# The `setup_logging` function returns a logger instance, which can be used by the calling module (e.g. main.py)
# For now, we will rely on individual modules getting their own logger.
# If a global logger instance is needed, it can be initialized here after setup.
# logger = setup_logging() # This would make `logger` a global instance in this module.
# For now, keep it simple and let main.py get its own logger after calling setup_logging().
# The setup_logging() function will configure the root logger, which affects all loggers.
# The Colors class and log_request_beautifully function were moved to app.utils.beautiful_log.py.
# Their definitions were removed in a previous step.
