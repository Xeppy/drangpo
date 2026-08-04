"""The pipeline. retrieve -> abstain? -> extract -> check provenance -> verify ->
gate -> (self-repair) -> certify.

Nothing here trusts the previous step. Retrieval can decide there is nothing to
say. The extractor's verbatim claims are checked mechanically. The verifier
re-examines the assembled answer from scratch. When the gate fails on removable
grounds, self-repair excises the offending spans and re-checks. Only an answer
that clears the gate is marked shippable.
"""
from __future__ import annotations
from typing import List, Tuple

from .types import Answer, Certificate, Passage, Segment, Claim
from .config import Config
from .retrieve import Retriever
from .extract import extract, check_provenance
from .verify import verify, gate
from .repair import offending_indices, strip_segments, has_content


def _sources(passages: List[Passage], cited: set) -> list:
    out = []
    for p in passages:
        if p.id in cited:
            out.append({"id": p.id, "title": p.title, "source": p.source,
                        "kind": p.kind, "start": p.start,
                        "distance": round(p.distance, 4)})
    return out


def _abstain(query: str, passages: List[Passage], note: str) -> Answer:
    cert = Certificate(query=query, grounding="abstained", abstained=True, passed=True,
                       verbatim_ratio=0.0, notes=[note])
    text = ("I do not have a teaching from her that speaks to this directly, "
            "so I will not answer for her.")
    return Answer(text=text, segments=[Segment(text=text, kind="connective")],
                  claims=[], certificate=cert, passages=passages)


def _assess(text: str, segments: List[Segment], passages: List[Passage],
            cfg: Config, llm) -> Tuple[float, int, List[Claim], bool, List[str]]:
    """One full check: provenance, independent verification, and the gate."""
    verbatim_ratio = check_provenance(segments, passages, cfg)
    fabricated = sum(1 for s in segments if s.kind == "verbatim" and not s.matched)
    claims = verify(text, passages, cfg, llm)
    passed, reasons = gate(verbatim_ratio, fabricated, claims, cfg)
    return verbatim_ratio, fabricated, claims, passed, reasons


def certify(query: str, segments: List[Segment], passages: List[Passage],
            cfg: Config, llm=None) -> Answer:
    """Check a drafted set of segments, self-repairing if enabled, and issue the
    certificate. Shared by build() and by any caller assembling segments directly."""
    text = "".join(s.text for s in segments)
    vr, fab, claims, passed, reasons = _assess(text, segments, passages, cfg, llm)

    removed, rounds, repaired = 0, 0, False
    while (not passed) and cfg.self_repair and rounds < cfg.max_repair_rounds:
        drop = offending_indices(segments, claims, cfg)
        if not drop:
            break                                   # nothing removable; a real block
        segments = strip_segments(segments, drop)
        removed += len(drop)
        repaired = True
        rounds += 1
        if not has_content(segments):
            return _abstain(query, passages,
                            "nothing verifiable remained after removing unsupported material")
        text = "".join(s.text for s in segments)
        vr, fab, claims, passed, reasons = _assess(text, segments, passages, cfg, llm)

    grounding = ("repaired" if (passed and repaired) else "grounded" if passed else "blocked")
    notes = list(reasons)
    if passed and repaired:
        notes.insert(0, f"self-repair removed {removed} unsupported span(s)")

    cited = {s.source for s in segments if s.kind == "verbatim" and s.source}
    cert = Certificate(
        query=query, grounding=grounding, abstained=False, passed=passed,
        verbatim_ratio=vr,
        claims_supported=sum(1 for c in claims if c.verdict == "supported"),
        claims_extrapolated=sum(1 for c in claims if c.verdict == "extrapolated"),
        claims_unsupported=sum(1 for c in claims if c.verdict == "unsupported"),
        fabricated_quotes=fab, notes=notes, sources=_sources(passages, cited),
    )
    return Answer(text=text, segments=segments, claims=claims,
                  certificate=cert, passages=passages)


def build(query: str, retriever: Retriever, cfg: Config, llm=None) -> Answer:
    passages = retriever.search(query)
    if not retriever.is_relevant(passages):
        return _abstain(query, passages, "no sufficiently relevant teaching found")

    segments, abstain = extract(query, passages, cfg, llm)
    if abstain or not segments:
        return _abstain(query, passages, "the retrieved passages do not address this")

    return certify(query, segments, passages, cfg, llm)
