"""OpenAI-compatible grounded generation."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rag.config import GeneratorConfig
from rag.domain.models import RetrievedChunk
from rag.generation.prompt_builder import GroundedPromptBuilder
from rag.generation.response_cleaner import strip_reasoning_tags


@dataclass
class OpenAICompatibleGenerator:
    """OpenAI-compatible chat-completions generator."""

    config: GeneratorConfig
    prompt_builder: GroundedPromptBuilder = field(default_factory=GroundedPromptBuilder)
    _client: Any = field(default=None, init=False, repr=False)

    def generate(self, query: str, chunks: Sequence[RetrievedChunk]) -> str:
        response = self._client_instance().chat.completions.create(
            model=self.config.model_name,
            messages=self.prompt_builder.build_messages(query, chunks),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        content = response.choices[0].message.content
        return strip_reasoning_tags(content or "")

    def _client_instance(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=_api_key(),
                timeout=self.config.timeout_seconds,
            )
        return self._client


def _api_key() -> str:
    return os.getenv("API_KEY") or os.getenv("UTSA_API_KEY") or "EMPTY"
