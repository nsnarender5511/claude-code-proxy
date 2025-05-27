# Architecture Plan: Ultra-Lean Anthropic SDK Facade for LiteLLM (Version 1.1)

**Date:** 2025-05-27
**Status:** Finalized for Initial Implementation

## 1. Architecture Overview

This architecture defines an "ultra-lean" facade that allows clients using the Anthropic SDK to seamlessly interact with various LLMs (OpenAI, Gemini, Anthropic, etc.) managed by a backend LiteLLM Proxy Service. The core principle is to delegate almost all complex logic to LiteLLM, including model routing, API key management for downstream LLMs, and the primary request/response translation between different LLM paradigms.

This custom facade will be responsible for:

1.  Exposing an Anthropic-compatible API endpoint (`/v1/messages`).
2.  Translating incoming Anthropic SDK `MessagesRequest` objects into the OpenAI ChatCompletion request format.
3.  Forwarding these OpenAI-formatted requests to a configured LiteLLM Proxy Service (specifically its `/v1/chat/completions` endpoint).
4.  Receiving OpenAI-formatted responses (or SSE streams) from LiteLLM.
5.  Translating these OpenAI-formatted responses back into the Anthropic SDK `MessagesResponse` format (or corresponding SSE stream) for the client.

This approach, referred to as "Path B" during planning, was chosen based on LiteLLM documentation, which strongly supports LiteLLM's capability to translate OpenAI-formatted requests to various backends and return OpenAI-formatted responses. This minimizes custom translation complexity within the facade.

**Key Principles:**

*   **Maximize LiteLLM:** LiteLLM is the workhorse for all complex LLM interactions, translations to specific backends, and routing.
*   **Minimal Custom Code in Facade:** The facade contains the bare minimum code for the two-way translation between Anthropic and OpenAI formats.
*   **Stateless Translation:** Translations performed by the facade should be stateless and direct mappings between common API concepts.
*   **Configuration over Code:** Rely on LiteLLM's configuration for model definitions, backend API keys, and routing rules. The facade's configuration is minimal.

## 2. System Structure

The system consists of two main services:

1.  **Custom Anthropic Facade (This Application):** A thin FastAPI application.
2.  **LiteLLM Proxy Service:** A standard LiteLLM proxy instance, comprehensively configured with target models and their API keys.

**Textual Diagram: High-Level Interaction Flow (Path B)**

```text
+---------------------+      +--------------------------+      +------------------------+      +-----------------+
|   Anthropic SDK     |----->|  Custom Anthropic Facade |----->| LiteLLM Proxy Service  |----->| Target LLM APIs |
|      (Client)       |      |    (FastAPI App)         |      | (/v1/chat/completions) |      | (Anthropic,    |
+---------------------+      +--------------------------+      +------------------------+      |  OpenAI, Gemini)|
                             | - Receives Anthropic req |      | - Translates OpenAI req  |      +-----------------+
                             | - Translates to OpenAI   |      |   to Target LLM format   |
                             |   req format             |      | - Receives Target LLM resp|
                             | - Forwards to LiteLLM    |      | - Translates Target LLM  |
                             | - Receives OpenAI resp   |      |   resp to OpenAI format  |
                             | - Translates to Anthropic|      +------------------------+
                             |   resp format            |
                             | - Streams response back  |
                             +--------------------------+
```

**Components of the Custom Anthropic Facade:**

*   **A. API Layer (FastAPI):**
    *   Exposes Anthropic-compatible `/v1/messages` endpoint.
    *   Uses Pydantic models matching Anthropic SDK for request deserialization and OpenAI/Anthropic models for internal response handling.
*   **B. LiteLLM Forwarding Service (incorporates Anthropic-to-OpenAI Request Translation):**
    *   Takes the validated Anthropic `MessagesRequest` object.
    *   Performs a stateless translation to an OpenAI `ChatCompletionRequest` format. This includes mapping:
        *   Anthropic messages (including multi-block content and `tool_result`) to OpenAI messages (text content, `tool` role messages for results).
        *   Anthropic `system` prompt to OpenAI `system` message.
        *   Anthropic `tools` and `tool_choice` to OpenAI `tools` and `tool_choice`.
        *   Common parameters like `max_tokens`, `temperature`, `stream`.
    *   Uses an HTTP client (e.g., `httpx`) to call the LiteLLM Proxy Service's `/v1/chat/completions` endpoint with the translated OpenAI-formatted request. The `model` parameter in this request will be the one LiteLLM uses for routing (e.g., `openai/gpt-4o`, `gemini/gemini-pro`, `anthropic/claude-3...`).
*   **C. Response Handling Service (incorporates OpenAI-to-Anthropic Response Translation):**
    *   Receives the OpenAI-formatted response (streaming or non-streaming) from LiteLLM.
    *   Performs a stateless translation back to the Anthropic `MessagesResponse` format (or corresponding SSE stream for streaming). This includes mapping:
        *   OpenAI `choices[0].message.content` to Anthropic text content block.
        *   OpenAI `choices[0].message.tool_calls` to Anthropic `tool_use` content blocks.
        *   OpenAI `usage` to Anthropic `usage`.
        *   OpenAI `finish_reason` to Anthropic `stop_reason`.
        *   For streaming, OpenAI SSE events are translated chunk-by-chunk to Anthropic SSE events.
    *   Handles basic error mapping if LiteLLM returns errors (translating OpenAI error format to Anthropic error format).
*   **D. Configuration Management (Pydantic `BaseSettings`):**
    *   Manages: `PORT`, `LITELLM_PROXY_URL`, `LOG_LEVEL`.
    *   All LLM API keys for downstream models are managed solely by the LiteLLM Proxy Service's configuration.

## 3. Technical Foundations

*   **Programming Language:** Python 3.x
*   **Web Framework:** FastAPI
*   **Data Validation & Modeling:** Pydantic
*   **HTTP Client:** `httpx` (for asynchronous calls to LiteLLM)
*   **LiteLLM Proxy:** Standard LiteLLM installation, configured appropriately.
*   **Logging:** Python's standard `logging` module.
*   **Configuration:** Environment variables managed via Pydantic's `BaseSettings`.

## 4. Component Specifications (Details)

*   **API Endpoints (`messages.py` replacement):**
    *   `POST /v1/messages`:
        *   Accepts Anthropic SDK `MessagesRequest` (Pydantic model).
        *   Invokes `LiteLLM Forwarding Service` to translate and send to LiteLLM.
        *   Invokes `Response Handling Service` to translate and return/stream the response.
        *   Implements basic request/response metadata logging.

*   **LiteLLM Forwarding Service (`anthropic_to_openai_translator.py` + `litellm_client.py`):**
    *   **Input:** Anthropic `MessagesRequest` Pydantic model.
    *   **Output:** OpenAI `ChatCompletionRequest` Pydantic model (or dict).
    *   **Logic:** Contains specific mapping functions for each part of the request (messages, tools, parameters).
        *   `_translate_anthropic_messages_to_openai(...)`
        *   `_translate_anthropic_tools_to_openai(...)`
        *   `_translate_anthropic_tool_choice_to_openai(...)`
    *   Uses `httpx.AsyncClient` to send the translated request to `LITELLM_PROXY_URL/v1/chat/completions`.

*   **Response Handling Service (`openai_to_anthropic_translator.py` + streaming logic):**
    *   **Input (Non-streaming):** OpenAI `ChatCompletion` Pydantic model (or dict) from LiteLLM.
    *   **Output (Non-streaming):** Anthropic `MessagesResponse` Pydantic model.
    *   **Input (Streaming):** Async generator of OpenAI SSE data chunks from LiteLLM.
    *   **Output (Streaming):** Async generator of Anthropic SSE data chunks.
    *   **Logic:** Contains specific mapping functions.
        *   `_translate_openai_response_to_anthropic(...)`
        *   `_translate_openai_stream_chunk_to_anthropic(...)` (handles different event types like content delta, tool use delta, stop events).
    *   Error mapping: Translates OpenAI/LiteLLM error structures to Anthropic error structures.

*   **Pydantic Models (`models.py`):**
    *   Define Pydantic models for:
        *   Anthropic `MessagesRequest` and its sub-components (Content Blocks, Tool definitions, etc.) - mirroring the official Anthropic structures.
        *   Anthropic `MessagesResponse` and its sub-components.
        *   OpenAI `ChatCompletionRequest` and its sub-components (Messages, Tool definitions, etc.) - mirroring official OpenAI structures.
        *   OpenAI `ChatCompletion` (response) and its sub-components.
        *   Anthropic and OpenAI SSE event structures.

*   **Configuration Management (`config.py` replacement):**
    *   A single `Settings` class inheriting from Pydantic `BaseSettings`.
    *   Fields: `PORT: int`, `LITELLM_PROXY_URL: HttpUrl`, `LOG_LEVEL: str = "INFO"`.
    *   Loads from environment variables. Provides validation on load.

## 5. Quality Attribute Strategies

*   **Maintainability:** Significantly improved due to focused, stateless translation logic and delegation of complex tasks to LiteLLM. Clear separation between Anthropic-OpenAI mapping and LiteLLM interaction.
*   **Testability:** Each translation function (e.g., Anthropic message to OpenAI message) can be unit-tested independently. Integration tests will cover the full facade-to-LiteLLM flow.
*   **Reliability:** Relies on LiteLLM's reliability for backend interactions. Facade adds minimal points of failure. Robust configuration and error mapping are key.
*   **Performance:** Asynchronous operations (`FastAPI`, `httpx`) ensure good I/O performance. Translation overhead should be minimal for stateless mappings.
*   **Scalability:** Standard horizontal scaling for FastAPI applications. LiteLLM can also be scaled.
*   **Security:** Facade itself doesn't handle LLM API keys. If the facade needs its own auth, that would be a separate middleware. Standard secure coding practices for FastAPI.

## 6. Integration Patterns

*   **Custom Facade to LiteLLM Proxy:** Asynchronous HTTP/S calls to `LITELLM_PROXY_URL/v1/chat/completions`.
*   **Client to Custom Facade:** Standard Anthropic SDK calls to the facade's `/v1/messages` endpoint.

## 7. Data Architecture

*   No persistent data storage within the facade itself.

## 8. Implementation Guidelines

1.  **Set up and Configure LiteLLM Proxy:** Ensure LiteLLM is running and configured with all target backend models (Anthropic, OpenAI, Gemini, etc.) and their respective API keys. Test it directly using an OpenAI SDK client to confirm it routes correctly.
2.  **Define Pydantic Models:** Create comprehensive Pydantic models for Anthropic and OpenAI request/response/SSE structures.
3.  **Implement Configuration (`config.py` replacement):** Use Pydantic `BaseSettings`.
4.  **Implement Request Translation Logic (`LiteLLM Forwarding Service`):**
    *   Create functions to map Anthropic `MessagesRequest` fields to OpenAI `ChatCompletionRequest` fields.
    *   Pay close attention to message content (text, image if supported), tool definitions, and tool results (Anthropic `tool_result` block to OpenAI `tool` role message).
5.  **Implement Response Translation Logic (`Response Handling Service`):**
    *   Create functions to map OpenAI `ChatCompletion` (and its stream chunks) to Anthropic `MessagesResponse` (and its stream chunks).
    *   Handle `tool_calls` from OpenAI and map them to Anthropic `tool_use` blocks.
    *   Map `finish_reason` / `stop_reason`.
6.  **Implement FastAPI Endpoint (`messages.py` replacement):**
    *   Wire up the request translation, `httpx` call to LiteLLM, and response translation.
    *   Implement streaming using `StreamingResponse` and an async generator that consumes LiteLLM's stream and translates chunks.
7.  **Add Error Handling:** Implement try-except blocks around LiteLLM calls and translate potential errors from LiteLLM/OpenAI format to Anthropic error format.
8.  **Logging:** Implement basic request/response metadata logging. Add more detailed logging within translation services if needed for debugging.
9.  **Testing:**
    *   Unit tests for all translation functions (e.g., given an Anthropic message list, does it correctly convert to an OpenAI message list?).
    *   Unit tests for SSE chunk translations.
    *   Integration tests calling the facade's endpoint and mocking the LiteLLM proxy response to verify end-to-end translation within the facade.
    *   End-to-end tests with a real LiteLLM proxy instance.

## 9. Evolution Strategy

*   This ultra-lean facade provides a solid, maintainable base.
*   Future custom features (e.g., custom auth, rate limiting, specific audit logging) can be layered on top as distinct middleware or services without complicating the core translation path.
*   Continuously monitor LiteLLM's capabilities. If LiteLLM introduces more direct ways to handle Anthropic SDK requests for cross-vendor scenarios (improving "Path A"), the facade's translation logic could potentially be simplified even further or removed.

## 10. Architecture Decision Records (ADRs)

*   **ADR-001 (Revised): Delegate Complex Multi-Backend Translation and Routing to LiteLLM.** The facade will translate Anthropic SDK format to OpenAI format, send to LiteLLM, and translate OpenAI response format back to Anthropic SDK format.
*   **ADR-002: Pydantic for Configuration and API Data Modeling.** Use Pydantic `BaseSettings` for application configuration and Pydantic models for strict typing and validation of Anthropic and OpenAI API structures.
*   **ADR-003: Stateless Facade Translations.** All translations between Anthropic and OpenAI formats within the facade will be stateless, direct mappings to ensure simplicity and testability.

This architectural plan provides a clear path to building a lean, maintainable, and effective Anthropic SDK facade for LiteLLM, addressing the critical issues identified by the Roaster Agent and aligning with your stated goal.