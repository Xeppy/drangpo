"""Tunable policy. These are the knobs that make the framework opinionated."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Config:
    # retrieval
    top_k: int = 6
    relevance_floor: float = 0.62      # if the closest passage is farther than this, abstain
    written_boost: float = 0.05        # subtract from distance of written sources (books/articles)
    practice_penalty: float = 0.06     # add to distance of group-recitation passages

    # faithfulness gate
    min_verbatim_ratio: float = 0.50   # an answer must be at least this much the teacher's own words
    verbatim_match_threshold: float = 0.85  # a 'verbatim' span must reach this coverage to count as real
    block_on_unsupported: bool = True  # any unsupported claim blocks the answer
    block_on_fabricated_quote: bool = True  # any fabricated 'verbatim' span blocks the answer

    # self-repair
    self_repair: bool = True           # on a blockable failure, excise the offending spans and re-verify
    max_repair_rounds: int = 2         # how many excise-and-recheck passes before giving up
    repair_match_threshold: float = 0.5  # how strongly a span must carry an unsupported claim to be cut

    # extraction
    max_sentences: int = 5             # offline extractor: how many source sentences to assemble
