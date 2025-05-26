import json
import time
import logging
import uuid # Required for handle_streaming and convert_litellm_to_anthropic (which is used by create_message)

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import litellm

# Models
from app.api.models import MessagesRequest, MessagesResponse, Usage, ContentBlockText, ContentBlockToolUse

# Config API Keys are no longer directly imported here.
# They are accessed via get_api_key_for_model from llm_router.
# from app.core.config import ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY # REMOVED

# Services
from app.services.llm_router import get_api_key_for_model
from app.services.translation import convert_anthropic_to_litellm, convert_litellm_to_anthropic

# Utils
from app.utils.beautiful_log import log_request_beautifully
# clean_gemini_schema is used by convert_anthropic_to_litellm, so not directly here
# from app.utils.gemini_schema import clean_gemini_schema 

# Typing
from typing import Union, List, Dict, Any # Ensure all necessary typing imports are here

logger = logging.getLogger(__name__)
router = APIRouter()

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs
        
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.original_model or original_request.model, # Use original model
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0, # Will be updated at the end if available
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        
        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        
        tool_index = None # Anthropic's index for tool_use blocks (starts from 1 if text is also present)
        current_tool_call_litellm_index = None # LiteLLM's index for tool_calls
        tool_content = "" # Accumulated content for current tool's input JSON
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        anthropic_tool_block_idx_counter = 0 # Starts at 0 for text, then increments for each tool
        
        # Process each chunk
        async for chunk in response_generator:
            try:
                # Check for usage data in the chunk (sometimes at the end)
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens') and chunk.usage.prompt_tokens is not None:
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens') and chunk.usage.completion_tokens is not None:
                        output_tokens = chunk.usage.completion_tokens
                
                # Handle text content and tool calls from choices
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = getattr(choice, 'delta', {})
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    delta_content = getattr(delta, 'content', None)
                    if isinstance(delta, dict) and 'content' in delta and delta['content'] is not None: # Backup for dict delta
                        delta_content = delta['content']

                    if delta_content:
                        accumulated_text += delta_content
                        if current_tool_call_litellm_index is None and not text_block_closed: # Only send text if not in tool parsing mode
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"

                    delta_tool_calls = getattr(delta, 'tool_calls', None)
                    if isinstance(delta, dict) and 'tool_calls' in delta and delta['tool_calls'] is not None: # Backup for dict delta
                        delta_tool_calls = delta['tool_calls']

                    if delta_tool_calls:
                        if not text_block_closed and text_sent: # Close text block if it was open
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            text_block_closed = True
                        elif not text_block_closed and accumulated_text: # Text was accumulated but not sent (e.g. first chunk had tool)
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            text_sent = True
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            text_block_closed = True
                        elif not text_block_closed : # No text was ever sent or accumulated, but tool call is starting
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            text_block_closed = True


                        for tool_call_chunk in delta_tool_calls:
                            litellm_idx = getattr(tool_call_chunk, 'index', None)
                            if isinstance(tool_call_chunk, dict) and 'index' in tool_call_chunk: # Backup for dict
                                litellm_idx = tool_call_chunk['index']
                            
                            # New tool identified by LiteLLM
                            if litellm_idx is not None and (current_tool_call_litellm_index is None or current_tool_call_litellm_index != litellm_idx):
                                if current_tool_call_litellm_index is not None: # Stop previous tool block
                                     yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': anthropic_tool_block_idx_counter})}\n\n"

                                current_tool_call_litellm_index = litellm_idx
                                anthropic_tool_block_idx_counter += 1 # Increment for new Anthropic tool block
                                tool_content = "" # Reset for new tool

                                tool_id = getattr(tool_call_chunk, 'id', f"toolu_{uuid.uuid4().hex[:24]}")
                                if isinstance(tool_call_chunk, dict) and 'id' in tool_call_chunk: tool_id = tool_call_chunk['id']
                                
                                function_chunk = getattr(tool_call_chunk, 'function', {})
                                if isinstance(tool_call_chunk, dict) and 'function' in tool_call_chunk: function_chunk = tool_call_chunk['function']

                                name = getattr(function_chunk, 'name', '')
                                if isinstance(function_chunk, dict) and 'name' in function_chunk: name = function_chunk['name']
                                
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_block_idx_counter, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                            
                            # Accumulate arguments for the current tool
                            function_chunk = getattr(tool_call_chunk, 'function', {})
                            if isinstance(tool_call_chunk, dict) and 'function' in tool_call_chunk: function_chunk = tool_call_chunk['function']
                            
                            arguments_delta = getattr(function_chunk, 'arguments', '')
                            if isinstance(function_chunk, dict) and 'arguments' in function_chunk: arguments_delta = function_chunk['arguments']

                            if arguments_delta:
                                tool_content += arguments_delta
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_block_idx_counter, 'delta': {'type': 'input_json_delta', 'partial_json': arguments_delta}})}\n\n"
                    
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        
                        if current_tool_call_litellm_index is not None: # Stop any open tool block
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': anthropic_tool_block_idx_counter})}\n\n"
                        elif not text_block_closed : # If no tools, ensure text block is closed
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            text_block_closed = True

                        anthropic_stop_reason = "end_turn"
                        if finish_reason == "length": anthropic_stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls": anthropic_stop_reason = "tool_use"
                        
                        # Update usage with final counts if available
                        final_usage_data = {'output_tokens': output_tokens}
                        if input_tokens > 0: # Only include if we have a valid count
                            final_usage_data['input_tokens'] = input_tokens

                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': anthropic_stop_reason, 'stop_sequence': None}, 'usage': final_usage_data})}\n\n"
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        # yield "data: [DONE]\n\n" # LiteLLM's stream wrapper might send this or client might not need it
                        return
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)} - Chunk: {chunk}")
                continue
        
        # Fallback if stream ends without finish_reason
        if not has_sent_stop_reason:
            if current_tool_call_litellm_index is not None:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': anthropic_tool_block_idx_counter})}\n\n"
            if not text_block_closed:
                 yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

            final_usage_data = {'output_tokens': output_tokens}
            if input_tokens > 0:
                final_usage_data['input_tokens'] = input_tokens
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': final_usage_data})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    finally:
        # Ensure [DONE] is sent to signify end of stream to clients like curl
        # Some clients might not need this if they correctly handle message_stop
        yield "data: [DONE]\n\n"


@router.post("/v1/messages")
async def create_message(
    request: MessagesRequest, # This is the Pydantic model for the request body
    raw_request: Request      # This is the raw FastAPI request object
):
    try:
        # Get original model for logging purposes
        body = await raw_request.body()
        body_json = json.loads(body.decode('utf-8'))
        original_model_for_logging = body_json.get("model", "unknown")
        
        display_model = original_model_for_logging
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        logger.debug(f"📊 PROCESSING REQUEST: Original Model='{original_model_for_logging}', Validated Model='{request.model}', Stream={request.stream}")
        
        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Refactored API Key Logic
        litellm_request["api_key"] = get_api_key_for_model(request.model)
        
        # For OpenAI models - modify request format to work with limitations
        # This block might need further review if it's still necessary after robust translation
        if "openai" in litellm_request["model"] and "messages" in litellm_request:
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")
            for i, msg in enumerate(litellm_request["messages"]):
                if "content" in msg and isinstance(msg["content"], list):
                    # Simplified: convert complex content to simple string if it's a list
                    # This is a basic fallback; translation should ideally handle this.
                    text_content = ""
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_content += block.get("text", "") + "\n"
                        # Basic stringification for other complex types if any slip through
                        elif isinstance(block, dict):
                            text_content += json.dumps(block) + "\n"
                        elif isinstance(block, str):
                             text_content += block + "\n"

                    litellm_request["messages"][i]["content"] = text_content.strip() or "..."
                elif msg.get("content") is None:
                    litellm_request["messages"][i]["content"] = "..." 
                
                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        logger.warning(f"Removing unsupported field from message for OpenAI: {key}")
                        del msg[key]
        
        logger.debug(f"Request for LiteLLM model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")
        
        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST", 
            str(raw_request.url.path), # Ensure path is string
            display_model, 
            litellm_request.get('model'),
            len(litellm_request.get('messages', [])),
            num_tools,
            200  # Assuming success at this point for logging
        )

        if request.stream:
            response_generator = await litellm.acompletion(**litellm_request)
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            start_time = time.time()
            litellm_response = await litellm.acompletion(**litellm_request) # Use acompletion for consistency
            logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            return anthropic_response
                
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_details = {"error": str(e), "type": type(e).__name__, "traceback": error_traceback}
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr): error_details[attr] = getattr(e, attr)
        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__']: error_details[key] = str(value)
        
        logger.error(f"Error processing message request: {json.dumps(error_details, indent=2)}")
        status_code = error_details.get('status_code', 500)
        if not isinstance(status_code, int): status_code = 500 # Ensure status_code is an int
        
        # Construct a more detailed error message for the client
        client_error_message = f"Error: {error_details.get('type', 'UnknownError')}"
        if 'message' in error_details and error_details['message']:
            client_error_message += f" - {error_details['message']}"
        # Avoid sending full tracebacks or overly detailed internal info to client
        
        raise HTTPException(status_code=status_code, detail=client_error_message)
