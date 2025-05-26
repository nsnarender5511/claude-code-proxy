import pytest
from src.services.llm_router import get_api_key_for_model
from src.core import config # To access actual API key values for comparison if needed

# For these tests to be meaningful, you might need to set dummy API keys
# in your test environment or mock os.environ.get if config directly uses it.

@pytest.mark.parametrize(
    "model_name, expected_key_env_var",
    [
        ("openai/gpt-4o", "OPENAI_API_KEY"),
        ("gemini/gemini-1.5-pro", "GEMINI_API_KEY"),
        ("anthropic/claude-3-opus-20240229", "ANTHROPIC_API_KEY"),
        ("some-other-model", "ANTHROPIC_API_KEY"), # Default case
        (None, "ANTHROPIC_API_KEY") # Default for None
    ]
)
def test_get_api_key_for_model(model_name, expected_key_env_var, monkeypatch):
    # Mock environment variables directly accessed by config.py for isolated testing
    # If config.py is already imported, its values might be set. 
    # This approach is more robust for testing the router's logic itself.
    mock_keys = {
        "OPENAI_API_KEY": "test_openai_key",
        "GEMINI_API_KEY": "test_gemini_key",
        "ANTHROPIC_API_KEY": "test_anthropic_key"
    }
    
    # Monkeypatch the global config variables that llm_router imports
    monkeypatch.setattr(config, "OPENAI_API_KEY", mock_keys["OPENAI_API_KEY"])
    monkeypatch.setattr(config, "GEMINI_API_KEY", mock_keys["GEMINI_API_KEY"])
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", mock_keys["ANTHROPIC_API_KEY"])

    returned_key = get_api_key_for_model(model_name)
    assert returned_key == mock_keys[expected_key_env_var]

def test_get_api_key_for_model_no_env_keys(monkeypatch):
    """Test behavior when API keys might be None (e.g., not set)."""
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None) # Default key is also None

    assert get_api_key_for_model("openai/gpt-4") is None
    assert get_api_key_for_model("gemini/gemini-pro") is None
    assert get_api_key_for_model("anthropic/claude-2") is None # Anthropic is default, so returns its key (None)
    assert get_api_key_for_model("unknown-model") is None 