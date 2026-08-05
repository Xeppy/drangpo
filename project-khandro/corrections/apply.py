#!/usr/bin/env python3
"""Re-apply the corrections map to transcripts already ingested. Fixes the
searchable chunk text in the live index (so retrieval + display are correct) and
the raw source files (so a future re-index stays correct). Idempotent. WAL mode
means a running reader keeps serving during the update.

Confirm each rule with probe.py first. A rule is guarded (lookarounds, optional
case-sensitivity) so real words survive; see corrections.example.json.
"""
import glob, json, os, re, sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "index", "data", "corpus.db")
RAW = os.path.join(HERE, "..", "index", "data", "raw")
MAP = os.environ.get("CORRECTIONS", os.path.join(HERE, "corrections.example.json"))

RULES = [(re.compile(r["pat"], 0 if r.get("cs") else re.I), r["rep"])
         for r in json.load(open(MAP))["rules"]]


def fix(t):
    for rx, rep in RULES:
        t = rx.sub(rep, t or "")
    return t


def main():
    db = sqlite3.connect(DB, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    rows = db.execute("select id, text from chunks").fetchall()
    n = 0
    for cid, txt in rows:
        nt = fix(txt)
        if nt != txt:
            db.execute("update chunks set text=? where id=?", (nt, cid)); n += 1
    db.commit(); db.close()
    print("chunks fixed in index: %d / %d" % (n, len(rows)))

    fn = 0
    for f in glob.glob(os.path.join(RAW, "*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        changed = False
        for s in r.get("segments") or []:
            nt = fix(s.get("text", ""))
            if nt != s.get("text", ""):
                s["text"] = nt; changed = True
        if changed:
            json.dump(r, open(f, "w"), ensure_ascii=False); fn += 1
    print("source files fixed: %d" % fn)


if __name__ == "__main__":
    main()
