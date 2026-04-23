"""Factory helpers for grounded generators."""

from __future__ import annotations

from rag.config import GeneratorConfig
from rag.generation.openai_compatible import OpenAICompatibleGenerator


def build_generator(config: GeneratorConfig) -> OpenAICompatibleGenerator:
    return OpenAICompatibleGenerator(config=config)
