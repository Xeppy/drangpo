"""Self-repair.

When the faithfulness gate fails on removable grounds, do not throw the whole
answer away. Most failures are a single stray sentence: a fabricated quote, or a
claim the sources do not support, sitting inside an answer that is otherwise
sound. Self-repair excises exactly that material and re-verifies what remains. If
the remainder stands on its own it ships, marked as repaired. If nothing
verifiable is left, the system abstains. The guarantee is never weakened; the
blast radius of one bad sentence is.
"""
from __future__ import annotations
from typing import List, Set

from .types import Segment, Claim
from .text import verbatim_coverage
from .config import Config


def offending_indices(segments: List[Segment], claims: List[Claim], cfg: Config) -> Set[int]:
    """Segment indices to remove: fabricated verbatim spans, and the spans that
    carry each unsupported claim."""
    drop: Set[int] = set()

    # 1. any span presented as verbatim that did not actually match its source
    for i, s in enumerate(segments):
        if s.kind == "verbatim" and not s.matched:
            drop.add(i)

    # 2. the verbatim span that best carries each unsupported claim
    for c in claims:
        if c.verdict != "unsupported":
            continue
        best_i, best = -1, 0.0
        for i, s in enumerate(segments):
            if s.kind != "verbatim":
                continue
            cov = max(verbatim_coverage(c.claim, s.text), verbatim_coverage(s.text, c.claim))
            if cov > best:
                best, best_i = cov, i
        if best_i >= 0 and best >= cfg.repair_match_threshold:
            drop.add(best_i)

    return drop


def strip_segments(segments: List[Segment], drop: Set[int]) -> List[Segment]:
    """Remove the dropped segments and tidy the connective tissue left behind."""
    kept = [s for i, s in enumerate(segments) if i not in drop]
    out: List[Segment] = []
    for s in kept:
        if s.kind == "connective" and (not out or out[-1].kind == "connective"):
            continue                        # collapse doubled or leading connectives
        out.append(s)
    while out and out[-1].kind == "connective":
        out.pop()                           # trim a trailing connective
    return out


def has_content(segments: List[Segment]) -> bool:
    """Is there any verified verbatim material left to stand on?"""
    return any(s.kind == "verbatim" and s.matched for s in segments)
