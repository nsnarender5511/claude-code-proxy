from typing import Optional, List, Dict
from src.core.config import settings, AnthropicModelInfo


class ModelTranslationService:

    def __init__(self):
        self._anthropic_models_info: Dict[str, AnthropicModelInfo] = settings.ANTHROPIC_MODELS_INFO
        self._anthropic_to_openai_map: Dict[str, str] = settings.ANTHROPIC_TO_OPENAI_MAP
        self._anthropic_to_gemini_map: Dict[str, str] = settings.ANTHROPIC_TO_GEMINI_MAP
        self._metadata_sources: List[str] = settings.MODEL_METADATA_SOURCES
        self._map_last_updated: str = settings.MODEL_MAP_LAST_UPDATED

    def get_anthropic_model_info(self, anthropic_model_id: str) -> Optional[AnthropicModelInfo]:
        return self._anthropic_models_info.get(anthropic_model_id)

    def get_openai_equivalent(self, anthropic_model_id: str) -> Optional[str]:
        return self._anthropic_to_openai_map.get(anthropic_model_id)

    def get_gemini_equivalent(self, anthropic_model_id: str) -> Optional[str]:
        return self._anthropic_to_gemini_map.get(anthropic_model_id)

    def get_metadata_sources(self) -> List[str]:
        return self._metadata_sources

    def get_map_last_updated(self) -> str:
        return self._map_last_updated


if __name__ == '__main__':
    translator = ModelTranslationService()
    opus_info = translator.get_anthropic_model_info('anthropic/claude-3-opus')
    if opus_info:
        print(f'Claude Opus Info: {opus_info.description}')
    else:
        print('Claude Opus info not found.')
    opus_to_openai = translator.get_openai_equivalent('anthropic/claude-3-opus')
    if opus_to_openai:
        print(f'Claude Opus OpenAI Equivalent: {opus_to_openai}')
    else:
        print('Claude Opus OpenAI equivalent not found.')
    haiku_to_gemini = translator.get_gemini_equivalent('anthropic/claude-3-haiku')
    if haiku_to_gemini:
        print(f'Claude Haiku Gemini Equivalent: {haiku_to_gemini}')
    else:
        print('Claude Haiku Gemini equivalent not found.')
    non_existent_model = 'anthropic/non-existent-model'
    print(
        f'Info for {non_existent_model}: {translator.get_anthropic_model_info(non_existent_model)}'
    )
    print(
        f'OpenAI equivalent for {non_existent_model}: {translator.get_openai_equivalent(non_existent_model)}'
    )
    print(
        f'Gemini equivalent for {non_existent_model}: {translator.get_gemini_equivalent(non_existent_model)}'
    )
    print(f'\nMetadata Sources: {translator.get_metadata_sources()}')
    print(f'Map Last Updated: {translator.get_map_last_updated()}')
