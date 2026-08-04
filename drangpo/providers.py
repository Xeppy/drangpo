"""Embedding and LLM providers.

Both are optional. Imports are lazy so the offline demo and tests run on bare
Python. Supply GEMINI_API_KEY for embeddings and ANTHROPIC_API_KEY for the
LLM-backed extractor and verifier. Any provider can be swapped: the rest of the
framework only depends on the two small function shapes at the bottom.
"""
from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Optional


class GeminiEmbedder:
    """Embeds queries. `task_type` must match how the corpus was embedded: a store
    built with RETRIEVAL_DOCUMENT is queried with RETRIEVAL_QUERY."""

    def __init__(self, model: str = "gemini-embedding-001", dims: int = 3072,
                 task_type: str = "RETRIEVAL_QUERY"):
        from google import genai  # lazy
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        self.dims = dims
        self.task_type = task_type

    def __call__(self, texts: List[str]) -> List[List[float]]:
        from google.genai import types
        r = self.client.models.embed_content(
            model=self.model, contents=texts,
            config=types.EmbedContentConfig(task_type=self.task_type))
        return [e.values for e in r.embeddings]


class ClaudeLLM:
    """Tool-forced structured output. Returns validated dicts, never free text."""

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 1500):
        import anthropic  # lazy
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.max_tokens = max_tokens

    def json(self, system: str, user: str, schema: Dict[str, Any],
             tool_name: str = "respond") -> Dict[str, Any]:
        tool = {"name": tool_name, "description": "Return the structured result.",
                "input_schema": schema}
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            tools=[tool], tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )
        for b in msg.content:
            if b.type == "tool_use":
                return b.input
        return {}


def default_llm() -> Optional[ClaudeLLM]:
    return ClaudeLLM() if os.environ.get("ANTHROPIC_API_KEY") else None


def default_embedder() -> Optional[GeminiEmbedder]:
    return GeminiEmbedder() if os.environ.get("GEMINI_API_KEY") else None
