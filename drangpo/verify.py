"""Adversarial faithfulness verification.

A second, independent pass that treats the drafted answer as a suspect. It
decomposes the answer into atomic claims and, for each, asks whether the source
passages actually support it. This is deliberately separate from the extractor:
the thing that writes the answer is not the thing that is allowed to certify it.
Claims come back as supported, extrapolated (reasonable but beyond the sources),
or unsupported (not in the sources at all). Policy in config decides what to do.
"""
from __future__ import annotations
from typing import List, Tuple

from .types import Passage, Claim
from .config import Config
from .text import sentences, best_source


_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {"type": "string",
                                "enum": ["supported", "extrapolated", "unsupported"]},
                    "evidence": {"type": "string", "description": "passage id, or empty"},
                    "quote": {"type": "string", "description": "the exact supporting words, or empty"},
                },
                "required": ["claim", "verdict"],
            },
        }
    },
    "required": ["claims"],
}

_VERIFY_SYSTEM = (
    "You are a strict fact-checker. You are given an ANSWER and the PASSAGES it was supposedly "
    "built from. Break the answer into its atomic claims. For each claim, decide: 'supported' if "
    "a passage directly states it (give the passage id and the exact supporting quote); "
    "'extrapolated' if it is a reasonable extension but no passage states it; 'unsupported' if no "
    "passage backs it at all. Be adversarial: default to the weaker verdict when unsure. Do not "
    "give the answer the benefit of the doubt."
)


def verify(answer_text: str, passages: List[Passage], cfg: Config,
           llm=None) -> List[Claim]:
    if llm is not None:
        user = ("ANSWER:\n" + answer_text + "\n\nPASSAGES:\n"
                + "\n\n".join(f"[{p.id}] {p.text}" for p in passages))
        data = llm.json(_VERIFY_SYSTEM, user, _VERIFY_SCHEMA)
        raw = data.get("claims", []) if isinstance(data, dict) else []
        claims = [Claim(claim=c.get("claim", ""), verdict=c.get("verdict", "unsupported"),
                        evidence=(c.get("evidence") or None), quote=c.get("quote", ""))
                  for c in raw if isinstance(c, dict) and c.get("claim")]
        # malformed or empty model output must never mean "nothing to check":
        # fall back to the deterministic verbatim check rather than pass by default.
        if not claims:
            return _verify_offline(answer_text, passages, cfg)
        return claims
    return _verify_offline(answer_text, passages, cfg)


def _verify_offline(answer_text: str, passages: List[Passage], cfg: Config) -> List[Claim]:
    """Deterministic check: a claim is supported iff it appears verbatim in a passage."""
    claims: List[Claim] = []
    srcs = [(p.id, p.text) for p in passages]
    for s in sentences(answer_text):
        sid, cov = best_source(s, srcs)
        if cov >= cfg.verbatim_match_threshold:
            claims.append(Claim(claim=s, verdict="supported", evidence=sid, quote=s))
        else:
            claims.append(Claim(claim=s, verdict="unsupported"))
    return claims


def gate(verbatim_ratio: float, fabricated_quotes: int, claims: List[Claim],
         cfg: Config) -> Tuple[bool, List[str]]:
    """The faithfulness gate. Returns (passed, reasons_if_blocked)."""
    reasons: List[str] = []
    unsupported = sum(1 for c in claims if c.verdict == "unsupported")
    if cfg.block_on_fabricated_quote and fabricated_quotes > 0:
        reasons.append(f"{fabricated_quotes} span(s) presented as verbatim did not match any source")
    if cfg.block_on_unsupported and unsupported > 0:
        reasons.append(f"{unsupported} unsupported claim(s)")
    if verbatim_ratio < cfg.min_verbatim_ratio:
        reasons.append(f"verbatim ratio {verbatim_ratio:.0%} below floor {cfg.min_verbatim_ratio:.0%}")
    return (len(reasons) == 0), reasons
