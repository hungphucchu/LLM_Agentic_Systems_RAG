"""Normalization helpers for model responses."""

from __future__ import annotations

import re


def strip_reasoning_tags(text: str) -> str:
    """Remove visible Qwen-style reasoning tags from model output."""
    without_closed_blocks = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    if re.search(r"<think>", without_closed_blocks, flags=re.IGNORECASE):
        before_think = re.split(r"<think>", without_closed_blocks, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        return before_think
    return without_closed_blocks.strip()
