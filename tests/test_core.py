"""Deterministic tests for the parts that must never rely on a model's goodwill.
Run with `python tests/test_core.py` or `pytest`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drangpo import Config, Retriever, build, certify
from drangpo.store import FixtureStore
from drangpo.text import verbatim_coverage
from drangpo.extract import check_provenance
from drangpo.verify import gate, verify
from drangpo.types import Segment

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "fixtures", "corpus.json")


def _retriever(cfg=None):
    cfg = cfg or Config()
    return Retriever(FixtureStore.from_json(CORPUS), cfg), cfg


def test_verbatim_coverage_exact_vs_paraphrase():
    src = "Fear that is met with attention loses its grip."
    assert verbatim_coverage("met with attention loses its grip", src) == 1.0
    assert verbatim_coverage("fear disappears when you ignore it completely", src) < 0.85


def test_abstains_when_nothing_relevant():
    r, cfg = _retriever()
    ans = build("What is your opinion on stock market derivatives?", r, cfg)
    assert ans.certificate.abstained is True
    assert ans.certificate.grounding == "abstained"


def test_grounded_answer_is_mostly_verbatim():
    r, cfg = _retriever()
    ans = build("How should I work with fear?", r, cfg)
    assert ans.certificate.abstained is False
    assert ans.certificate.verbatim_ratio >= 0.5
    assert ans.certificate.passed is True
    assert ans.certificate.fabricated_quotes == 0


def test_fabricated_quote_is_caught_and_blocks_when_repair_off():
    r, cfg = _retriever(Config(self_repair=False))
    passages = r.search("How should I work with fear?")
    segs = [
        Segment(text="Do not push it away.", kind="verbatim", source="fear-1"),
        Segment(text="Fear should be conquered by force and never felt.",
                kind="verbatim", source="fear-1"),
    ]
    ratio = check_provenance(segs, passages, cfg)
    fabricated = sum(1 for s in segs if s.kind == "verbatim" and not s.matched)
    assert fabricated == 1
    text = "".join(s.text for s in segs)
    claims = verify(text, passages, cfg)
    passed, reasons = gate(ratio, fabricated, claims, cfg)
    assert passed is False
    assert any("verbatim" in x for x in reasons)


def test_self_repair_excises_offender_and_ships():
    r, cfg = _retriever(Config(self_repair=True))
    passages = r.search("How should I work with fear?")
    segs = [
        Segment(text="Do not push it away.", kind="verbatim", source="fear-1"),
        Segment(text=" ", kind="connective"),
        Segment(text="Fear should be conquered by force and never felt.",
                kind="verbatim", source="fear-1"),
    ]
    ans = certify("How should I work with fear?", segs, passages, cfg)
    assert ans.certificate.passed is True
    assert ans.certificate.grounding == "repaired"
    assert "conquered by force" not in ans.text
    assert "Do not push it away." in ans.text


def test_self_repair_abstains_when_nothing_verifiable_remains():
    r, cfg = _retriever(Config(self_repair=True))
    passages = r.search("How should I work with fear?")
    segs = [Segment(text="Fear should be conquered by force and never felt.",
                    kind="verbatim", source="fear-1")]
    ans = certify("How should I work with fear?", segs, passages, cfg)
    assert ans.certificate.abstained is True


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f" ERR  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
