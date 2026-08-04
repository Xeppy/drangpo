#!/usr/bin/env python3
"""End-to-end demo. Runs offline on the synthetic fixture corpus with no API keys
and no dependencies, so you can see the whole guarantee working in one command:

    python demo.py

Set ANTHROPIC_API_KEY (and point at a real sqlite-vec corpus) to run the same
pipeline with the LLM-backed extractor and verifier instead of the offline ones.
"""
import os
from drangpo import Config, Retriever, build, certify, render
from drangpo.store import FixtureStore
from drangpo.types import Segment

HERE = os.path.dirname(os.path.abspath(__file__))


def banner(t):
    print("\n" + "═" * 72 + f"\n  {t}\n" + "═" * 72)


def main():
    cfg = Config()
    store = FixtureStore.from_json(os.path.join(HERE, "fixtures", "corpus.json"))
    retriever = Retriever(store, cfg)          # offline: llm=None throughout

    banner("1 · A question she has answered  →  extractive, grounded")
    q = "How should I work with fear?"
    print(f"Q: {q}\n")
    print(render(build(q, retriever, cfg)))

    banner("2 · A question she never addressed  →  honest abstention")
    q = "What do you think about cryptocurrency investing?"
    print(f"Q: {q}\n")
    print(render(build(q, retriever, cfg)))

    # a draft where a model invented a sentence and dressed it as a direct quote
    q = "How should I work with fear?"
    passages = retriever.search(q)
    tainted = [
        Segment(text="Do not push it away.", kind="verbatim", source="fear-1"),
        Segment(text=" ", kind="connective"),
        Segment(text="Fear should be conquered by force and never felt.",
                kind="verbatim", source="fear-1"),      # not in the source at all
    ]

    banner("3 · A fabricated quote, repair OFF  →  blocked outright")
    print(f"Q: {q}\n(the model slipped in an invented sentence as a direct quote)\n")
    print(render(certify(q, list(tainted), passages, Config(self_repair=False))))

    banner("4 · Same draft, self-repair ON  →  offending span excised, ships repaired")
    print(f"Q: {q}\n(the fabricated span is removed; what stands on its own ships)\n")
    print(render(certify(q, list(tainted), passages, Config(self_repair=True))))
    print()


if __name__ == "__main__":
    main()
