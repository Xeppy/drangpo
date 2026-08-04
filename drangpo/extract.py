"""Extractive answering.

The reply is built, as far as possible, from the teacher's own verbatim
sentences, with the connective tissue kept to a minimum and marked as such. This
is the opposite of ordinary RAG, which paraphrases the sources into new prose.
Here the sources ARE the prose. Every span is then checked against its claimed
source, so a span presented as verbatim that was actually invented is caught
mechanically, before it can reach a reader.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

from .types import Passage, Segment
from .config import Config
from .text import sentences, keyword_overlap, verbatim_coverage, best_source


_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "abstain": {"type": "boolean",
                    "description": "true if the passages do not actually address the question"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": ["verbatim", "connective"]},
                    "source": {"type": "string",
                               "description": "passage id for verbatim spans, empty for connective"},
                },
                "required": ["text", "kind"],
            },
        },
    },
    "required": ["abstain", "segments"],
}

_EXTRACT_SYSTEM = (
    "You assemble an answer almost entirely from the speaker's own words. "
    "You are given numbered passages of what they actually said or wrote. "
    "Build the answer by copying whole sentences VERBATIM from those passages, in an order "
    "that answers the question. Mark each copied span kind='verbatim' with its passage id. "
    "Use kind='connective' only for the few linking words needed to read smoothly, and keep "
    "them minimal and content-free. Never introduce a claim, fact, name, or teaching that is "
    "not present in the passages. If the passages do not address the question, set abstain=true "
    "and return no segments. Do not paraphrase a verbatim span; copy it exactly."
)


def _render_passages(passages: List[Passage]) -> str:
    return "\n\n".join(f"[{p.id}] {p.text}" for p in passages)


def extract(query: str, passages: List[Passage], cfg: Config,
            llm=None) -> Tuple[List[Segment], bool]:
    if llm is not None:
        user = f"QUESTION:\n{query}\n\nPASSAGES:\n{_render_passages(passages)}"
        data = llm.json(_EXTRACT_SYSTEM, user, _EXTRACT_SCHEMA)
        if isinstance(data, dict) and data.get("abstain"):
            return [], True
        raw = data.get("segments", []) if isinstance(data, dict) else []
        segs = [Segment(text=s.get("text", ""), kind=s.get("kind", "connective"),
                        source=(s.get("source") or None))
                for s in raw if isinstance(s, dict) and s.get("text")]
        # if the model gave nothing usable, fall back to deterministic extraction
        # rather than silently abstaining on a question we did retrieve for.
        if not segs:
            return _extract_offline(query, passages, cfg), False
        return segs, False
    return _extract_offline(query, passages, cfg), False


def _extract_offline(query: str, passages: List[Passage], cfg: Config) -> List[Segment]:
    """Deterministic assembler: pick the source sentences most on-topic, in order."""
    ranked = []
    for p in passages:
        for s in sentences(p.text):
            score = keyword_overlap(query, s)
            if score > 0:
                ranked.append((score, p.id, s))
    ranked.sort(key=lambda t: -t[0])
    chosen = ranked[: cfg.max_sentences]
    # restore reading order by (passage order, position in passage)
    order = {p.id: i for i, p in enumerate(passages)}
    chosen.sort(key=lambda t: order.get(t[1], 999))
    segs: List[Segment] = []
    for i, (_, pid, s) in enumerate(chosen):
        if i > 0:
            segs.append(Segment(text=" ", kind="connective"))
        segs.append(Segment(text=s, kind="verbatim", source=pid))
    return segs


def check_provenance(segments: List[Segment], passages: List[Passage],
                     cfg: Config) -> float:
    """Verify every 'verbatim' span really appears in its source. Returns the
    verbatim ratio (share of answer characters that are genuine source words)."""
    by_id = {p.id: p.text for p in passages}
    verbatim_chars = 0
    total_chars = 0
    for seg in segments:
        total_chars += len(seg.text)
        if seg.kind != "verbatim":
            continue
        # prefer the claimed source, but fall back to searching all passages
        cov = verbatim_coverage(seg.text, by_id.get(seg.source or "", "")) if seg.source else 0.0
        if cov < cfg.verbatim_match_threshold:
            sid, cov2 = best_source(seg.text, [(p.id, p.text) for p in passages])
            if cov2 > cov:
                seg.source, cov = sid, cov2
        seg.coverage = cov
        seg.matched = cov >= cfg.verbatim_match_threshold
        if seg.matched:
            verbatim_chars += len(seg.text)
    return (verbatim_chars / total_chars) if total_chars else 0.0
