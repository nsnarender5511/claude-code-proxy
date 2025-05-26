import pytest
from src.services.translation import convert_anthropic_to_litellm, convert_litellm_to_anthropic
from src.api.models import MessagesRequest, Message, ContentBlockText, ContentBlockToolUse, Usage, MessagesResponse
from unittest.mock import MagicMock # For mocking parts of LiteLLM response if needed

# Minimal valid MessagesRequest for testing conversion
@pytest.fixture
def sample_anthropic_request() -> MessagesRequest:
    return MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[
            Message(role="user", content="Hello")
        ]
    )

@pytest.fixture
def sample_anthropic_request_with_tools() -> MessagesRequest:
    return MessagesRequest(
        model="claude-3-opus-20240229",
        max_tokens=150,
        messages=[
            Message(role="user", content="What is 2+2 and the weather in SF?")
        ],
        tools=[
            {
                "name": "calculator", 
                "description": "desc", 
                "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}}
            }
        ]
    )

def test_convert_anthropic_to_litellm_basic(sample_anthropic_request):
    litellm_req = convert_anthropic_to_litellm(sample_anthropic_request)
    assert litellm_req["model"] == "openai/gpt-4.1-mini" # Assuming mapping for haiku
    assert len(litellm_req["messages"]) == 1
    assert litellm_req["messages"][0]["role"] == "user"
    assert litellm_req["messages"][0]["content"] == "Hello"

def test_convert_anthropic_to_litellm_with_tools(sample_anthropic_request_with_tools):
    litellm_req = convert_anthropic_to_litellm(sample_anthropic_request_with_tools)
    assert litellm_req["model"] == "openai/gpt-4.1" # Assuming mapping for opus via sonnet then to BIG_MODEL
    assert "tools" in litellm_req
    assert len(litellm_req["tools"]) == 1
    assert litellm_req["tools"][0]["type"] == "function"
    assert litellm_req["tools"][0]["function"]["name"] == "calculator"

@pytest.fixture
def sample_litellm_response_dict() -> dict:
    return {
        "id": "chatcmpl-xxxxxxxxxxxx",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The weather is sunny."
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    }

@pytest.fixture
def sample_litellm_response_tool_calls_dict() -> dict:
    return {
        "id": "chatcmpl-yyyyyyyyyyyy",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None, # OpenAI often has null content with tool_calls
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"San Francisco\"}"
                            }
                        }
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10}
    }

def test_convert_litellm_to_anthropic_basic(sample_litellm_response_dict, sample_anthropic_request):
    # sample_anthropic_request is used to pass the original model info
    anthropic_res = convert_litellm_to_anthropic(sample_litellm_response_dict, sample_anthropic_request)
    assert isinstance(anthropic_res, MessagesResponse)
    assert anthropic_res.role == "assistant"
    assert len(anthropic_res.content) == 1
    assert isinstance(anthropic_res.content[0], ContentBlockText)
    assert anthropic_res.content[0].text == "The weather is sunny."
    assert anthropic_res.stop_reason == "end_turn"
    assert anthropic_res.usage.input_tokens == 10
    assert anthropic_res.usage.output_tokens == 5

def test_convert_litellm_to_anthropic_tool_calls(sample_litellm_response_tool_calls_dict, sample_anthropic_request_with_tools):
    # Using _with_tools to simulate a request that might expect tools
    anthropic_res = convert_litellm_to_anthropic(sample_litellm_response_tool_calls_dict, sample_anthropic_request_with_tools)
    assert isinstance(anthropic_res, MessagesResponse)
    assert anthropic_res.role == "assistant"
    assert len(anthropic_res.content) > 0 # Could be 0 if content was None and no tool_calls processed, or 1 for tool_use
    
    found_tool_use = False
    for block in anthropic_res.content:
        if isinstance(block, ContentBlockToolUse):
            assert block.type == "tool_use"
            assert block.id == "call_abc123"
            assert block.name == "get_weather"
            assert block.input == {"location": "San Francisco"}
            found_tool_use = True
            break
    assert found_tool_use
    assert anthropic_res.stop_reason == "tool_use" 