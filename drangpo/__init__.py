"""drangpo (དྲང་པོ་, Tibetan: honest, upright, faithful) — provenance-guaranteed,
abstention-first answering in a real person's own words.

Public API:
    from drangpo import Config, Retriever, build, certify, render
    from drangpo.store import FixtureStore, SqliteVecStore
"""
from .config import Config
from .retrieve import Retriever
from .answer import build, certify
from .render import render
from .types import Answer, Certificate, Passage, Segment, Claim

__all__ = ["Config", "Retriever", "build", "certify", "render",
           "Answer", "Certificate", "Passage", "Segment", "Claim"]
__version__ = "0.1.0"
