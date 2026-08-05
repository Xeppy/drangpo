# drangpo

**Answering in a real person's own words, with provenance and abstention guarantees.**

*drangpo* (དྲང་པོ་) is Tibetan for *honest, straight, upright*. It is the property
this framework is built to enforce.

Most retrieval systems answer *about* a body of text. `drangpo` answers *as* a
person, under a hard constraint: the reply is built from what they actually said,
every span is checked against its source, every claim is verified independently,
and when the corpus is silent the system says so rather than improvising. When an
answer fails on a single stray sentence, that sentence is excised and the rest
re-checked, rather than the whole answer being lost. Nothing leaves the pipeline
without a machine-checkable certificate of where its words came from.

This is the opposite instinct to ordinary RAG. Ordinary RAG paraphrases sources
into fluent new prose, and a fluent paraphrase is exactly where a teacher's
meaning gets quietly bent. Here the sources are the prose, and the framework's
job is to prove it.

---

## Why this exists

If you are building a model of a real teacher, a scholar, a founder, anyone whose
words carry weight, three failures are unacceptable and ordinary systems commit
all three:

1. **Fabrication.** The model states something the person never said, in their
   voice. A confidence score does not catch this. A fluent sentence hides it.
2. **Silent paraphrase.** The model rewords a teaching into something subtly
   different and presents it as theirs.
3. **Answering the unanswerable.** Asked something the person never addressed,
   the model invents a plausible position rather than admitting the gap.

`drangpo` is built so all three are caught mechanically, not left to the goodwill
of the generating model.

## The guarantee, concretely

Every answer carries a **certificate**:

```
── faithfulness certificate ──
  status         ● GROUNDED
  verbatim        97% of the answer is her own words
  claims          16 supported · 0 extrapolated · 0 unsupported
  sources         [74615·spoken], [81629·spoken], [104985·spoken]
```

when a single bad sentence was found and removed:

```
  status         ● GROUNDED (self-repaired)
  note           self-repair removed 1 unsupported span(s)
```

when the model tried to invent a quote and nothing could be salvaged, or when the
person never addressed the question:

```
  status         ● BLOCKED (would not ship)          ● ABSTAINED (honest)
  fabricated     1 span(s) faked a quote             note   no relevant teaching found
```

## Architecture

```
   query
     │
     ▼
  ┌───────────┐   relevance floor + source weighting
  │ retrieve  │   (written word boosted, group recitation penalised)
  └─────┬─────┘
        │ nothing close enough? ─────────────► ABSTAIN
        ▼
  ┌───────────┐   assemble the answer from VERBATIM source sentences,
  │ extract   │   connective tissue minimal and marked
  └─────┬─────┘
        ▼
  ┌───────────┐   does every 'verbatim' span actually appear in its
  │ provenance│   source?  (deterministic, no LLM)  fake quote ► flagged
  └─────┬─────┘
        ▼
  ┌───────────┐   independent adversarial pass: decompose into claims,
  │ verify    │   label each supported / extrapolated / unsupported
  └─────┬─────┘
        ▼
  ┌───────────┐   verbatim floor · no unsupported claims · no fabricated quotes
  │ gate      │──────────────┐
  └─────┬─────┘              │ fails on removable grounds?
        │ passes             ▼
        │             ┌───────────┐  excise the offending spans, re-verify
        │             │ self-repair│  (drangpo/repair.py) ── nothing left? ► ABSTAIN
        │             └─────┬─────┘
        ▼                   ▼
   answer + certificate (grounded / repaired / blocked / abstained)
```

The thing that writes the answer is never the thing that certifies it. The
provenance check needs no model at all.

## Self-repair

See [`drangpo/repair.py`](drangpo/repair.py). Most gate failures are one stray
sentence inside an otherwise sound answer. Rather than discard the whole thing,
`drangpo` removes exactly the offending material, a fabricated quote or a claim
the sources do not support, and re-verifies what remains. If the remainder stands
on its own it ships, marked `● GROUNDED (self-repaired)`. If nothing verifiable is
left, it abstains. The guarantee is never weakened; only the blast radius of one
bad sentence is. Turn it off with `Config(self_repair=False)` to see the raw gate
block outright.

## Quickstart (offline, zero dependencies)

```bash
python demo.py          # grounded · abstained · blocked · self-repaired
python tests/test_core.py
```

Both run on the Python standard library against a small synthetic corpus in
`fixtures/`. No API keys, no installs.

## Pointing it at a real corpus

Swap the store and providers; the rest is unchanged.

```python
from drangpo import Config, Retriever, build, render
from drangpo.store import SqliteVecStore
from drangpo.providers import default_llm, default_embedder

cfg = Config()
store = SqliteVecStore("corpus.db")                 # any chunks + vec_chunks schema, opened read-only
embed = default_embedder()                          # needs GEMINI_API_KEY
retriever = Retriever(store, cfg, embed_fn=embed)
ans = build("How do I work with fear?", retriever, cfg, llm=default_llm())  # ANTHROPIC_API_KEY
print(render(ans))
print(ans.to_dict())                                # certificate as JSON
```

The expected schema is the common one a transcription pipeline produces:
`chunks(id, video_id, title, url, start, text, session_type)` plus a `vec_chunks`
virtual table (cosine). The query embedder's `task_type` must match how the
corpus was embedded. Providers and stores are small and swappable; nothing
downstream depends on Gemini or Claude specifically.

## Policy knobs (`drangpo/config.py`)

| knob | does |
|---|---|
| `relevance_floor` | how close a passage must be before the system will answer at all |
| `min_verbatim_ratio` | how much of an answer must be the person's own words |
| `verbatim_match_threshold` | how exact a "quote" must be to count as real |
| `block_on_unsupported` / `block_on_fabricated_quote` | what hard-fails the gate |
| `self_repair` / `max_repair_rounds` | excise offending spans and re-verify instead of blocking |
| `written_boost` / `practice_penalty` | privilege composed writing, discount group recitation |

## Status

This is the **core**, proven against a real 108k-passage corpus of transcribed
teaching: it produced grounded, mostly-verbatim answers for questions the teacher
had addressed, blocked answers where the model added an unsupported flourish, and
abstained honestly on questions she never touched. It is deliberately generic:
nothing here is tied to any particular teacher or community, and the demo runs on
a synthetic corpus. The surrounding pipeline (GPU transcription of accented
speech, a guarded correction pass, embedding, the vector index) is included,
sanitised, in [`project-khandro/`](project-khandro/) as a reusable scaffold
around this core. The reason to care about this project is the core, not the
scaffold.

## Licence

MIT. See [LICENSE](LICENSE).
