"""Vector stores.

Two implementations, one interface. The FixtureStore runs on pure Python with no
dependencies and no API keys, so the framework can be demonstrated and tested
offline. The SqliteVecStore is the real path: it reads any sqlite-vec database
that follows the common `chunks` + `vec_chunks` schema (the same shape a
transcription pipeline would produce), so pointing this at a real corpus is a
one-line change.
"""
from __future__ import annotations
import json
import os
from typing import List, Optional, Callable

from .types import Passage
from .text import keyword_overlap


class FixtureStore:
    """In-memory lexical store for offline demos and tests.

    Distance is 1 - keyword_overlap, so it plugs into the same relevance-floor
    and weighting logic as a real vector store without needing embeddings.
    """

    def __init__(self, passages: List[Passage]):
        self.passages = passages

    @classmethod
    def from_json(cls, path: str) -> "FixtureStore":
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        return cls([Passage(**r) for r in rows])

    def search(self, query: str, k: int, embed_fn: Optional[Callable] = None) -> List[Passage]:
        scored = []
        for p in self.passages:
            dist = 1.0 - keyword_overlap(query, p.text)
            scored.append(Passage(**{**p.__dict__, "distance": dist}))
        scored.sort(key=lambda p: p.distance)
        return scored[: max(k, 1)]


class SqliteVecStore:
    """Reads a khandrobot-style sqlite-vec DB. Requires an embed_fn at query time.

    Expected schema:
        chunks(id, video_id, title, url, start, text, session_type)
        vec_chunks  -- virtual table, cosine, embedding column matches chunks.id via rowid
    """

    def __init__(self, db_path: str, dims: int = 3072, read_only: bool = True):
        import sqlite3
        import sqlite_vec  # lazy: only needed for the real path
        # read-only by default: this is an answering framework, it never writes the corpus
        self.db = (sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                   if read_only else sqlite3.connect(db_path))
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.dims = dims

    def search(self, query: str, k: int, embed_fn: Callable) -> List[Passage]:
        import struct
        vec = embed_fn([query])[0]
        packed = struct.pack(f"{len(vec)}f", *vec)
        rows = self.db.execute(
            """
            select c.id, c.text, c.title, c.video_id, c.url, c.start, c.session_type, v.distance
            from (select rowid, distance from vec_chunks where embedding match ? and k = ?) v
            join chunks c on c.id = v.rowid
            order by v.distance
            """,
            [packed, k],
        ).fetchall()
        out = []
        for cid, text, title, vid, url, start, stype, dist in rows:
            kind = "written" if str(vid).startswith(("book-", "art-")) else (
                "practice" if stype == "practice" else "spoken")
            out.append(Passage(id=str(cid), text=text, title=title or "", source=str(vid),
                               url=url or "", start=start or 0.0, kind=kind, distance=dist))
        return out
