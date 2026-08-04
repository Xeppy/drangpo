"""Text normalisation and verbatim matching.

This module answers one question with no LLM involved: does this span of text
actually appear in the source it claims to come from? It is the mechanical
backstop under the whole framework. An LLM can claim a quote is direct; this
checks it, deterministically, so a fabricated quote wearing quotation marks
cannot pass.
"""
from __future__ import annotations
import re
import difflib
from typing import Iterable, List, Tuple

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Comparison form only."""
    s = _PUNCT.sub(" ", (s or "").lower())
    return _WS.sub(" ", s).strip()


def sentences(text: str) -> List[str]:
    """Cheap sentence split that keeps the original surface form (with punctuation)."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def verbatim_coverage(span: str, source: str) -> float:
    """How much of `span` appears contiguously inside `source` (0..1).

    1.0 means the span is a clean substring of the source. Values below ~0.85
    mean the span has been paraphrased or invented, not quoted.
    """
    a, b = norm(span), norm(source)
    if not a:
        return 0.0
    if a in b:
        return 1.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    m = sm.find_longest_match(0, len(a), 0, len(b))
    return m.size / len(a)


def best_source(span: str, sources: Iterable[Tuple[str, str]]) -> Tuple[str, float]:
    """Return (source_id, coverage) for the source that best contains `span`."""
    best_id, best = "", 0.0
    for sid, text in sources:
        c = verbatim_coverage(span, text)
        if c > best:
            best_id, best = sid, c
    return best_id, best


_STOP = set(
    "the a an and or but if of to in on is are be for it as at by we he she his her "
    "my me us them they their this that these those with without within into onto from "
    "what when where which who whom why how do does did done have has had having will "
    "would could should can may might must shall you your yours i am was were been being "
    "not no nor so than then there here about above below over under again once only very "
    "just also too more most some any all both each few other such own same up down out off "
    "think thing things way ways like get got go going make made take see look feel felt "
    "want need know said say tell one two".split()
)


def _content(words):
    return {w for w in words if len(w) > 2 and w not in _STOP}


def keyword_overlap(query: str, text: str) -> float:
    """Fraction of distinct query content-words present in text. Lexical relevance.

    Common words are stripped so that overlap reflects topic, not grammar. This
    is only used by the offline FixtureStore; a real embedding store measures
    semantic distance directly and needs none of this.
    """
    q = _content(norm(query).split())
    if not q:
        return 0.0
    t = set(norm(text).split())
    return len(q & t) / len(q)
