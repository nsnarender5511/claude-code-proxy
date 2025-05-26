import pytest
import json
from src.utils.tool_parser import parse_tool_result_content
from src.api.models import ContentBlockText # Assuming this might be part of input

@pytest.mark.parametrize(
    "input_content, expected_output",
    [
        (None, "No content provided"),
        ("Simple string content", "Simple string content"),
        ([{"type": "text", "text": "First line."}, {"type": "text", "text": "Second line."}], "First line.\nSecond line."),
        (["Line one", "Line two"], "Line one\nLine two"),
        ([{"type": "text", "text": "Just one text block."}], "Just one text block."),
        ([{"type": "other", "data": "should be stringified"}, {"text": "some text"}], '{"type": "other", "data": "should be stringified"}\n{"text": "some text"}'),
        ({"type": "text", "text": "Dictionary text content"}, "Dictionary text content"),
        ({"type": "json_object", "data": {"key": "value"}}, '{"type": "json_object", "data": {"key": "value"}}'),
        (123, "123"),
        (True, "True"),
        # Edge case: list with mixed content including non-dict/non-string
        (["Text line", 123, {"type": "text", "text": "Another line"}], "Text line\n123\nAnother line"),
        # Edge case: A list of ContentBlockText (though models.py defines ContentBlockText, it is not directly used as input type hint)
        # For this test, we'll assume they are passed as dicts as per common API usage.
        ([ContentBlockText(type='text', text='ContentBlockText line').dict()], "ContentBlockText line"),
        # Test with a complex nested structure as dict
        ({"level1": {"level2": "deep text", "level2_list": [{"type":"text", "text":"item1"}, "item2"]}},
         '{"level1": {"level2": "deep text", "level2_list": [{"type": "text", "text": "item1"}, "item2"]}}')
    ]
)
def test_parse_tool_result_content(input_content, expected_output):
    assert parse_tool_result_content(input_content) == expected_output

def test_parse_tool_result_content_unparseable():
    class Unparseable:
        def __str__(self):
            raise TypeError("Cannot make this a string")
    assert parse_tool_result_content(Unparseable()) == "Unparseable content" 