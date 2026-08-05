#!/usr/bin/env python3
"""Incrementally embed any transcripts not yet in the index and APPEND them to
the live corpus.db (WAL mode, so a running reader keeps serving during writes).
Reuses the same chunking, session tagging, and Gemini embedding as the full
build, so newly transcribed recordings become searchable without a full rebuild."""
import glob, json, os, sqlite3, struct, time
import sqlite_vec
from google import genai
import build_index as B

DB = B.DB
DIM = B.DIM


def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    db = sqlite3.connect(DB, timeout=60)
    db.enable_load_extension(True); sqlite_vec.load(db); db.enable_load_extension(False)
    db.execute("PRAGMA journal_mode=WAL")   # concurrent reads + writes

    existing = {vid for (vid,) in db.execute("select distinct video_id from chunks")}
    rid = db.execute("select coalesce(max(id), 0) from chunks").fetchone()[0]
    files = sorted(f for d in B.RAW_DIRS for f in glob.glob(os.path.join(d, "*.json")))

    buf = []
    t0 = time.time()
    added_videos = 0

    def flush():
        nonlocal rid
        if not buf:
            return
        embs = B.embed(client, [r[4] for r in buf], "RETRIEVAL_DOCUMENT")
        for r, e in zip(buf, embs):
            rid += 1
            db.execute("insert into chunks(id,video_id,title,url,start,text,session_type) values (?,?,?,?,?,?,?)",
                       (rid, r[0], r[1], r[2], r[3], r[4], r[5]))
            db.execute("insert into vec_chunks(rowid, embedding) values (?, ?)",
                       (rid, struct.pack(f"{DIM}f", *e)))
        db.commit()
        print("  +%d chunks (total id %d, %.0fs)" % (len(buf), rid, time.time() - t0), flush=True)
        buf.clear()

    for fp in files:
        rec = json.load(open(fp))
        if rec["id"] in existing:
            continue
        added_videos += 1
        stype = B.classify_session(rec.get("title"))
        for c in B.chunk_transcript(rec):
            buf.append((rec["id"], rec["title"], rec["url"], c["start"], c["text"], stype))
            if len(buf) >= 128:
                flush()
    flush()
    n = db.execute("select count(*) from chunks").fetchone()[0]
    v = db.execute("select count(distinct video_id) from chunks").fetchone()[0]
    db.close()
    print("added %d new recordings | index now %d chunks / %d recordings" % (added_videos, n, v), flush=True)


if __name__ == "__main__":
    main()
