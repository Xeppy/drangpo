# project-khandro — the reference pipeline

The scaffold that feeds `drangpo`. These are the sanitised helper scripts used by
**Project Khandro** to turn a teacher's scattered recordings into a clean,
searchable corpus that `drangpo` then answers from, faithfully.

`drangpo` (one directory up) is the part that matters: it guarantees answers are
the teacher's own words, provenance-checked, with honest abstention. This folder
is the well-trodden plumbing around it, included so a new community has a working
starting point rather than a blank page. Nothing here is secret; every credential
is read from the environment, never hardcoded.

## The stages

```
recordings ──► transcribe ──► correct ──► index ──► [ drangpo answers ]
  (Vimeo,      (GPU whisper,   (guarded    (chunk +
   YouTube,     resumable)      regex map   Gemini embed
   files)                       + probe)    → sqlite-vec)
```

| folder | what it does |
|---|---|
| `transcribe/` | rent a GPU, transcribe accented speech with faster-whisper large-v3, POST results back. Resumable, bandwidth-free, never touches the app server. |
| `ingest/` | pull existing captions where they exist (Vimeo example), no ASR needed. |
| `corrections/` | the mapping layer: probe the corpus for ASR mishearings of Sanskrit/Tibetan terms, confirm in context, then apply a guarded regex map idempotently to both the live index and the source files. |
| `index/` | chunk transcripts, tag group-recitation sessions, embed with Gemini, write a `chunks` + `vec_chunks` sqlite-vec database — exactly the schema `drangpo.store.SqliteVecStore` reads. |

## Why a correction layer at all

Whisper mangles accented Sanskrit and Tibetan: *bodhicitta* becomes "body chitta",
*ngöndro* becomes "non dro", *torma* becomes "trauma". Character error rate hides
this because each miss is small; meaning does not survive it. The rule is never a
blind find-and-replace. `corrections/probe.py` finds candidates and pulls real
contexts so a human confirms each is a genuine mishearing (many look like real
English), then a guarded rule goes into the map, then `corrections/apply.py`
applies it everywhere, idempotently. `drangpo`'s verifier is the last line; a clean
corpus is the first.

## Handing off to drangpo

```python
from drangpo import Config, Retriever, build, render
from drangpo.store import SqliteVecStore
from drangpo.providers import default_llm, default_embedder

store = SqliteVecStore("corpus.db")          # the DB index/build_index.py wrote
retriever = Retriever(store, Config(), embed_fn=default_embedder())
print(render(build("How do I work with fear?", retriever, Config(), llm=default_llm())))
```

## Environment

Set what each stage needs; nothing is stored in the repo.

```
GEMINI_API_KEY      # embeddings (index/)
VIMEO_TOKEN         # ingest/ + transcribe/ source resolution
RUNPOD_KEY          # transcribe/ GPU rental
INGEST_URL          # where the GPU worker POSTs transcripts (your app)
INGEST_SECRET       # shared secret guarding that endpoint
APP_ORIGIN          # your app's public origin, for the pod bootstrap bundle
POD_ID              # transcribe/monitor.py, the running pod to watch
```
