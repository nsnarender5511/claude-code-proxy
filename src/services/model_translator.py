from typing import Optional, List, Dict
from src.core.config import settings, AnthropicModelInfo

class ModelTranslationService:
    """
    Service to provide translations and information for Anthropic models
    to their OpenAI and Gemini equivalents based on the application configuration.
    """

    def __init__(self):
        self._anthropic_models_info: Dict[str, AnthropicModelInfo] = settings.ANTHROPIC_MODELS_INFO
        self._anthropic_to_openai_map: Dict[str, str] = settings.ANTHROPIC_TO_OPENAI_MAP
        self._anthropic_to_gemini_map: Dict[str, str] = settings.ANTHROPIC_TO_GEMINI_MAP
        self._metadata_sources: List[str] = settings.MODEL_METADATA_SOURCES
        self._map_last_updated: str = settings.MODEL_MAP_LAST_UPDATED

    def get_anthropic_model_info(self, anthropic_model_id: str) -> Optional[AnthropicModelInfo]:
        """
        Retrieves descriptive information for a given Anthropic model ID.

        Args:
            anthropic_model_id: The ID of the Anthropic model (e.g., "anthropic/claude-3-opus").

        Returns:
            An AnthropicModelInfo object if the model is found, otherwise None.
        """
        return self._anthropic_models_info.get(anthropic_model_id)

    def get_openai_equivalent(self, anthropic_model_id: str) -> Optional[str]:
        """
        Retrieves the OpenAI equivalent model ID for a given Anthropic model ID.

        Args:
            anthropic_model_id: The ID of the Anthropic model.

        Returns:
            The corresponding OpenAI model ID as a string if a mapping exists, otherwise None.
        """
        return self._anthropic_to_openai_map.get(anthropic_model_id)

    def get_gemini_equivalent(self, anthropic_model_id: str) -> Optional[str]:
        """
        Retrieves the Gemini equivalent model ID for a given Anthropic model ID.

        Args:
            anthropic_model_id: The ID of the Anthropic model.

        Returns:
            The corresponding Gemini model ID as a string if a mapping exists, otherwise None.
        """
        return self._anthropic_to_gemini_map.get(anthropic_model_id)

    def get_metadata_sources(self) -> List[str]:
        """
        Returns the list of metadata sources used for the model mappings.
        """
        return self._metadata_sources

    def get_map_last_updated(self) -> str:
        """
        Returns the date when the model maps were last updated.
        """
        return self._map_last_updated

# Example usage (optional, for testing or demonstration):
if __name__ == "__main__":
    translator = ModelTranslationService()

    # Test getting model info
    opus_info = translator.get_anthropic_model_info("anthropic/claude-3-opus")
    if opus_info:
        print(f"Claude Opus Info: {opus_info.description}")
    else:
        print("Claude Opus info not found.")

    # Test getting OpenAI equivalent
    opus_to_openai = translator.get_openai_equivalent("anthropic/claude-3-opus")
    if opus_to_openai:
        print(f"Claude Opus OpenAI Equivalent: {opus_to_openai}")
    else:
        print("Claude Opus OpenAI equivalent not found.")

    # Test getting Gemini equivalent
    haiku_to_gemini = translator.get_gemini_equivalent("anthropic/claude-3-haiku")
    if haiku_to_gemini:
        print(f"Claude Haiku Gemini Equivalent: {haiku_to_gemini}")
    else:
        print("Claude Haiku Gemini equivalent not found.")
    
    non_existent_model = "anthropic/non-existent-model"
    print(f"Info for {non_existent_model}: {translator.get_anthropic_model_info(non_existent_model)}")
    print(f"OpenAI equivalent for {non_existent_model}: {translator.get_openai_equivalent(non_existent_model)}")
    print(f"Gemini equivalent for {non_existent_model}: {translator.get_gemini_equivalent(non_existent_model)}")

    print(f"\nMetadata Sources: {translator.get_metadata_sources()}")
    print(f"Map Last Updated: {translator.get_map_last_updated()}")