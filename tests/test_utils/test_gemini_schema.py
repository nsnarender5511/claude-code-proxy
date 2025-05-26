import pytest
from src.utils.gemini_schema import clean_gemini_schema

@pytest.mark.parametrize(
    "input_schema, expected_schema",
    [
        # Test case 1: Remove additionalProperties
        (
            {"type": "object", "properties": {"foo": {"type": "string"}}, "additionalProperties": False},
            {"type": "object", "properties": {"foo": {"type": "string"}}}
        ),
        # Test case 2: Remove default
        (
            {"type": "string", "default": "bar"},
            {"type": "string"}
        ),
        # Test case 3: Remove unsupported format for string type
        (
            {"type": "string", "format": "email"},
            {"type": "string"} # email format removed
        ),
        # Test case 4: Keep supported format (enum - though enum is not a format, but let's test date-time)
        (
            {"type": "string", "format": "date-time"},
            {"type": "string", "format": "date-time"}
        ),
        # Test case 5: Nested schema cleaning
        (
            {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {"name": {"type": "string", "default": "Anon"}},
                        "additionalProperties": True
                    }
                }
            },
            {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}}
                    }
                }
            }
        ),
        # Test case 6: List of schemas
        (
            [{"type": "string", "default": "a"}, {"type": "integer", "additionalProperties": False}],
            [{"type": "string"}, {"type": "integer"}]
        ),
        # Test case 7: Empty schema
        ({},{}),
        # Test case 8: No unsupported fields
        (
            {"type": "object", "properties": {"id": {"type": "integer"}}},
            {"type": "object", "properties": {"id": {"type": "integer"}}}
        )
    ]
)
def test_clean_gemini_schema(input_schema, expected_schema):
    cleaned = clean_gemini_schema(input_schema)
    assert cleaned == expected_schema 