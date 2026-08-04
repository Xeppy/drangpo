"""Retrieval with source weighting and a relevance floor.

The weighting encodes an opinion that matters for faithful persona answering:
the teacher's composed written word outranks a spoken aside, and a passage that
is really the students chanting in unison is not the teacher teaching. The
relevance floor is what makes honest abstention possible: if nothing close
enough comes back, the correct answer is to say so.
"""
from __future__ import annotations
from typing import List, Optional, Callable

from .types import Passage
from .config import Config


class Retriever:
    def __init__(self, store, cfg: Config, embed_fn: Optional[Callable] = None):
        self.store = store
        self.cfg = cfg
        self.embed_fn = embed_fn

    def _weight(self, p: Passage) -> float:
        d = p.distance
        if p.kind == "written":
            d -= self.cfg.written_boost
        elif p.kind == "practice":
            d += self.cfg.practice_penalty
        return d

    def search(self, query: str) -> List[Passage]:
        raw = self.store.search(query, self.cfg.top_k, self.embed_fn)
        for p in raw:
            p.distance = self._weight(p)
        raw.sort(key=lambda p: p.distance)
        return raw

    def is_relevant(self, passages: List[Passage]) -> bool:
        return bool(passages) and passages[0].distance <= self.cfg.relevance_floor
