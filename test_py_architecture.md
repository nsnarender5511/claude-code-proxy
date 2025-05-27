## Architecture Plan for `test.py`

## Architecture Overview
The `test.py` script will be designed as a modular test runner that uses the Anthropic Python SDK to send various requests to a local proxy server. It will focus on testing the server's handling of different message parameters (like `max_tokens`, `model`), tool use requests, and basic performance/reliability (response time, status codes). The script will be structured to allow easy addition of new test scenarios.

## System Structure
The test script will have the following conceptual components:

1.  **Test Runner Engine**:
    *   Responsibilities:
        *   Manages the execution of test scenarios.
        *   Collects and reports test results (success/failure, response times).
        *   Initializes the Anthropic client configured for the target server.
    *   Textual Diagram:
        ```
        [Test Runner Engine]
             |
             v
        [Anthropic Client (configured for http://127.0.0.1:3000/proxy)]
             |
             v
        [Scenario Executor] --> Executes --> [Test Scenario 1]
                                          --> [Test Scenario 2]
                                          --> ...
                                          --> [Test Scenario N]
             |
             v
        [Result Aggregator & Reporter]
        ```

2.  **Anthropic Client Configuration**:
    *   Responsibilities:
        *   Initialize the `Anthropic` client from the SDK.
        *   Set the `base_url` to `http://127.0.0.1:3000/proxy`.
        *   Since there's no auth, the `api_key` can be omitted or set to a dummy value if the SDK requires it.

3.  **Scenario Definitions**:
    *   Responsibilities:
        *   Define individual test cases. Each scenario will specify:
            *   Request parameters (messages, model, max_tokens, tools, etc.).
            *   Expected outcomes (e.g., status code 200).
            *   Validation logic.
    *   Examples:
        *   Scenario: Test `max_tokens` effect.
        *   Scenario: Test with `model_A` vs `model_B`.
        *   Scenario: Test basic tool use request.
        *   Scenario: Test function call request.

4.  **Request Executor**:
    *   Responsibilities:
        *   Takes a scenario definition.
        *   Uses the configured Anthropic client to send the request to the server.
        *   Measures response time.
        *   Captures the server's response (status code, headers, body).

5.  **Response Validator**:
    *   Responsibilities:
        *   Takes the actual response and the scenario's expected outcomes.
        *   Performs validation (e.g., checks status code is 200).
        *   Determines if the test scenario passed or failed.

6.  **Reporting Module**:
    *   Responsibilities:
        *   Prints a summary of test results.
        *   Indicates which scenarios passed/failed.
        *   Shows response times for each scenario.

## Technical Foundations
*   **Language**: Python 3.x
*   **Core Library**: `anthropic` (Anthropic Python SDK)
*   **HTTP Client (via Anthropic SDK)**: `httpx` (used internally by the SDK)
*   **Standard Libraries**: `time` (for response time), `json` (for handling tool/function call payloads).
*   **Architectural Pattern**: Test Case pattern.

## Component Specifications

1.  **`TestRunner` (Conceptual Class/Module)**
    *   `client`: Instance of `anthropic.Anthropic`.
    *   `scenarios`: List of scenario definitions.
    *   `run_tests()`: Iterates through scenarios, executes them, and collects results.
    *   `report_results()`: Prints the test summary.

2.  **`AnthropicClientFactory` (Conceptual Function/Class)**
    *   `create_client(base_url)`: Returns an `anthropic.Anthropic` instance configured with the given `base_url`.

3.  **`Scenario` (Conceptual Data Structure/Class)**
    *   `name`: String.
    *   `request_params`: Dictionary.
    *   `expected_status_code`: Integer.
    *   `validation_function`: (Optional) Custom validation function.

4.  **`TestExecutor` (Conceptual Function/Class)**
    *   `execute_scenario(client, scenario)`: Makes API call, measures time, returns `(response_object, response_time, error_if_any)`. Uses `with_raw_response` for status code.

## Quality Attribute Strategies
*   **Maintainability**: Modular design, clear naming, configuration at top.
*   **Usability**: Simple to run, clear output.
*   **Extensibility**: Easy to add new scenarios.

## Integration Patterns
*   Integrates with `http://127.0.0.1:3000/proxy` via HTTP using Anthropic SDK.

## Implementation Guidelines
1.  **Setup**: `pip install anthropic`.
2.  **Client Initialization**: `Anthropic(base_url="http://127.0.0.1:3000/proxy")`.
3.  **Scenario Definition**: List of dictionaries for `max_tokens`, `model`, `tool_use` tests.
    *   Tool use example:
        ```python
        tool_spec = {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
        scenario_tool_use = {
            "name": "Test basic tool use",
            "request_params": {
                "model": "claude-3-opus-20240229",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "What's the weather like in London?"}],
                "tools": [tool_spec]
            },
            "expected_status_code": 200
        }
        ```
4.  **Execution and Validation Loop**: Iterate scenarios, use `client.messages.with_raw_response.create(...)`, record time, check status code, print results.

## Evolution Strategy
*   **Iteration 1 (Current Plan)**: Basic tests for `max_tokens`, `model`, status code, response time, basic tool request forwarding.
*   **Iteration 2**: More sophisticated content validation, error condition testing, complex tool use.
*   **Iteration 3**: Integration with `pytest`, performance benchmarks.

## Architecture Decision Records
*   **ADR-001: Use Anthropic Python SDK**: Simplifies test script development.
*   **ADR-002: Focus on Interface Testing of the Proxy**: Keeps scope manageable.
*   **ADR-003: Use `with_raw_response` for Status Code Access**: Necessary for status code validation.

## Feedback Loop Status
*   Iteration 1: Initial plan based on user requirements.