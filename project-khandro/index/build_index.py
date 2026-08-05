#!/usr/bin/env python3
"""Chunk + embed transcripts into a sqlite-vec index.

Reads data/raw/*.json transcripts, splits each into passage-level chunks (keeping
the start timestamp + source id), embeds them with Gemini, and writes corpus.db
(a `chunks` metadata table + a `vec_chunks` vec0 virtual table). This is exactly
the schema drangpo.store.SqliteVecStore reads.

Env: GEMINI_API_KEY.
Usage: python3 build_index.py
"""
import glob
import json
import os
import re
import sqlite3
import struct
import time

# Group-practice / liturgy sessions are mostly student recitation, not the
# teacher teaching. Tag them so retrieval can prefer actual teaching. Substring
# matching, since titles are often concatenated (e.g. "VYChoTsok", "DC-Tsok").
PRACTICE_PAT = re.compile(
    r"tsok|tsog|ganachakra|gana ?chakra|puja|recitation|accumulation|"
    r"fire ?puja|smoke offering|torma|nyung ?ne|"
    r"self-?initiation|self-?empowerment|group practice", re.I)


def classify_session(title):
    return "practice" if PRACTICE_PAT.search(title or "") else "teaching"

import sqlite_vec
from google import genai
from google.genai import types

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIRS = [os.path.join(HERE, "data", "raw")]
DB = os.path.join(HERE, "data", "corpus.db")

EMBED_MODEL = "gemini-embedding-001"
DIM = 3072
CHUNK_CHARS = 900          # ~150-200 words per chunk
OVERLAP_CHARS = 150
BATCH = 20                 # embeddings per API call


def chunk_transcript(rec):
    """Yield {text,start} passage chunks from a transcript record."""
    segs = rec["segments"]
    buf, buf_start = [], None
    cur = 0
    for s in segs:
        txt = s["text"].replace("\n", " ").strip()
        if not txt:
            continue
        if buf_start is None:
            buf_start = s.get("start", 0)
        buf.append(txt)
        cur += len(txt) + 1
        if cur >= CHUNK_CHARS:
            yield {"text": " ".join(buf), "start": buf_start}
            tail, tlen = [], 0                       # overlap: keep the tail
            for t in reversed(buf):
                tail.insert(0, t); tlen += len(t) + 1
                if tlen >= OVERLAP_CHARS:
                    break
            buf, cur = tail, tlen
            buf_start = s.get("start", 0)
    if buf:
        yield {"text": " ".join(buf), "start": buf_start or 0}


def embed(client, texts, task_type):
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        for attempt in range(1, 5):
            try:
                r = client.models.embed_content(
                    model=EMBED_MODEL, contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type))
                out.extend(e.values for e in r.embeddings)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                print("  embed retry (%s) %s" % (attempt, str(e)[:80]))
                time.sleep(3 * attempt)
    return out


def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    files = sorted(f for d in RAW_DIRS for f in glob.glob(os.path.join(d, "*.json")))

    tmp = DB + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    db = sqlite3.connect(tmp)
    db.enable_load_extension(True); sqlite_vec.load(db); db.enable_load_extension(False)
    db.execute("""create table chunks(
        id integer primary key, video_id text, title text, url text,
        start real, text text, session_type text)""")
    db.execute(f"create virtual table vec_chunks using vec0(embedding float[{DIM}] distance_metric=cosine)")

    # Stream per file so we never hold more than one outer batch of chunks in
    # memory; a full in-memory chunk list can OOM the build at corpus scale.
    t0 = time.time()
    rid = [0]
    buf = []

    def flush():
        if not buf:
            return
        embs = embed(client, [r[4] for r in buf], "RETRIEVAL_DOCUMENT")
        for r, e in zip(buf, embs):
            rid[0] += 1
            db.execute("insert into chunks(id,video_id,title,url,start,text,session_type) values (?,?,?,?,?,?,?)",
                       (rid[0], r[0], r[1], r[2], r[3], r[4], r[5]))
            db.execute("insert into vec_chunks(rowid, embedding) values (?, ?)",
                       (rid[0], struct.pack(f"{DIM}f", *e)))
        db.commit()
        print("  indexed %d chunks (%.0fs)" % (rid[0], time.time() - t0))
        buf.clear()

    nfiles = 0
    for fp in files:
        rec = json.load(open(fp))
        nfiles += 1
        stype = classify_session(rec.get("title"))
        for c in chunk_transcript(rec):
            buf.append((rec["id"], rec["title"], rec["url"], c["start"], c["text"], stype))
            if len(buf) >= 128:
                flush()
    flush()
    db.close()
    os.replace(tmp, DB)   # atomic swap so a live reader never sees a half-built DB
    print("wrote %s with %d chunks from %d files" % (DB, rid[0], nfiles))


if __name__ == "__main__":
    main()
