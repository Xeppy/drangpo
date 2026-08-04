"""Core data types for the faithful pipeline.

The whole framework is organised around one artifact: the Certificate. Every
answer carries a machine-checkable record of where each of its words came from
and which of its claims are actually supported by the source. Nothing leaves the
pipeline without one.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class Passage:
    """One retrieved unit of the teacher's own recorded or written words."""
    id: str
    text: str
    title: str = ""
    source: str = ""            # video id, book id, article slug
    url: str = ""
    start: float = 0.0          # timestamp within a recording, if any
    kind: str = "spoken"        # spoken | written | practice
    distance: float = 1.0       # cosine distance from the query (lower is closer)


@dataclass
class Segment:
    """One span of a drafted answer, tagged by where it came from."""
    text: str
    kind: str                   # verbatim | connective
    source: Optional[str] = None  # passage id, for verbatim spans
    matched: bool = True        # did a 'verbatim' span actually appear in its source?
    coverage: float = 1.0       # how much of the span was found verbatim (0..1)


@dataclass
class Claim:
    """One atomic assertion extracted from the answer, and its verdict."""
    claim: str
    verdict: str                # supported | extrapolated | unsupported
    evidence: Optional[str] = None  # passage id backing it
    quote: str = ""


@dataclass
class Certificate:
    query: str
    grounding: str              # grounded | partial | abstained | blocked
    abstained: bool
    passed: bool                # did it clear the faithfulness gate?
    verbatim_ratio: float       # share of answer characters that are the teacher's own words
    claims_supported: int = 0
    claims_extrapolated: int = 0
    claims_unsupported: int = 0
    fabricated_quotes: int = 0  # spans presented as verbatim that did NOT match any source
    notes: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    text: str
    segments: List[Segment]
    claims: List[Claim]
    certificate: Certificate
    passages: List[Passage]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "segments": [asdict(s) for s in self.segments],
            "claims": [asdict(c) for c in self.claims],
            "certificate": self.certificate.to_dict(),
        }
