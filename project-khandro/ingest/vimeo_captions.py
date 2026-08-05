#!/usr/bin/env python3
"""Vimeo caption ingest (no ASR, no media download).

Pulls a channel's Vimeo auto-captions via the API (just the .vtt text, no
media, no transfer cost), applies the correction map, and writes one raw record
per recording to ../index/data/raw/<id>.json in the schema build_index.py reads.
Use this where captions already exist; use transcribe/ where they do not.

Env: VIMEO_TOKEN.
Usage:
  python3 vimeo_captions.py --list          # enumerate + dedupe, write catalog
  python3 vimeo_captions.py --limit 15      # ingest a small verification batch
  python3 vimeo_captions.py                 # ingest all not-yet-done
"""
import argparse
import json
import os
import re
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "index", "data", "raw")
CATALOG = os.path.join(HERE, "catalog.json")
CORR = os.environ.get("CORRECTIONS",
                      os.path.join(HERE, "..", "corrections", "corrections.example.json"))
API = "https://api.vimeo.com"
TOKEN = os.environ.get("VIMEO_TOKEN", "")

_corr = None


def corrections():
    global _corr
    if _corr is None:
        _corr = ([(re.compile(r["pat"], 0 if r.get("cs") else re.I), r["rep"])
                  for r in json.load(open(CORR))["rules"]] if os.path.exists(CORR) else [])
    return _corr


def apply_corrections(text):
    for rx, rep in corrections():
        text = rx.sub(rep, text)
    return text


def api_get(path):
    req = urllib.request.Request(API + path if path.startswith("/") else path,
                                 headers={"Authorization": "Bearer " + TOKEN,
                                          "Accept": "application/vnd.vimeo.*+json;version=3.4"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_url(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8", "ignore")


TS = re.compile(r"(\d\d):(\d\d):(\d\d)[.,](\d\d\d)\s*-->")


def _sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(vtt):
    segs, start, buf, last = [], None, [], ""
    for line in vtt.splitlines():
        m = TS.search(line)
        if m:
            if buf and start is not None:
                t = re.sub(r"<[^>]+>", "", " ".join(buf)).replace("&nbsp;", " ")
                t = re.sub(r"\s+", " ", t).strip()
                if t and t != last:
                    segs.append({"text": t, "start": round(start, 3), "duration": 0})
                    last = t
                buf = []
            start = _sec(*m.group(1, 2, 3, 4))
        elif line.strip() in ("", "WEBVTT") or line.startswith(("Kind:", "Language:", "NOTE")):
            continue
        elif re.match(r"^\d+$", line.strip()):
            continue
        else:
            buf.append(line.strip())
    if buf and start is not None:
        t = re.sub(r"<[^>]+>", "", " ".join(buf)).replace("&nbsp;", " ")
        t = re.sub(r"\s+", " ", t).strip()
        if t and t != last:
            segs.append({"text": t, "start": round(start, 3), "duration": 0})
    return segs


def enumerate_videos():
    """All recordings, Gallery-duplicates removed, written to CATALOG."""
    vids, page = [], 1
    while True:
        d = api_get("/me/videos?per_page=100&page=%d&fields=uri,name,duration&sort=date" % page)
        data = d.get("data") or []
        if not data:
            break
        for v in data:
            vids.append({"id": v["uri"].split("/")[-1], "name": v.get("name") or "",
                         "duration": v.get("duration")})
        if not (d.get("paging") or {}).get("next"):
            break
        page += 1
        time.sleep(0.2)
    names = {v["name"].replace("-Gallery", "").replace(" Gallery", "") for v in vids
             if "Gallery" not in v["name"]}
    kept = [v for v in vids if "Gallery" not in v["name"]
            or v["name"].replace("-Gallery", "").replace(" Gallery", "") not in names]
    json.dump({"total_raw": len(vids), "kept": len(kept), "videos": kept},
              open(CATALOG, "w"), ensure_ascii=False)
    print("enumerated %d, %d after Gallery-dedupe -> %s" % (len(vids), len(kept), CATALOG))
    return kept


def get_caption(vid):
    d = api_get("/videos/%s/texttracks" % vid)
    tracks = [t for t in (d.get("data") or []) if (t.get("language") or "").startswith("en")]
    if not tracks:
        return None, None
    tracks.sort(key=lambda t: 0 if "autogen" not in (t.get("language") or "") else 1)
    t = tracks[0]
    return fetch_url(t["link"]), ("autogen" in (t.get("language") or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--throttle", type=float, default=0.4)
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)

    if args.list or not os.path.exists(CATALOG):
        enumerate_videos()
        if args.list:
            return

    videos = json.load(open(CATALOG))["videos"]
    todo = [v for v in videos if not os.path.exists(os.path.join(RAW, v["id"] + ".json"))]
    if args.limit:
        todo = todo[: args.limit]
    print("catalog=%d have=%d todo=%d" % (len(videos), len(videos) - len(todo), len(todo)))

    counts = {"ok": 0, "nocap": 0, "err": 0}
    for i, v in enumerate(todo, 1):
        vid = v["id"]
        try:
            vtt, is_gen = get_caption(vid)
            if not vtt:
                counts["nocap"] += 1
                time.sleep(args.throttle)
                continue
            segs = parse_vtt(vtt)
            for s in segs:
                s["text"] = apply_corrections(s["text"])
            rec = {"id": vid, "title": v["name"],
                   "url": "https://vimeo.com/%s" % vid, "source": "vimeo",
                   "duration": v.get("duration"), "is_generated": bool(is_gen),
                   "n_segments": len(segs), "segments": segs}
            json.dump(rec, open(os.path.join(RAW, vid + ".json"), "w"), ensure_ascii=False)
            counts["ok"] += 1
            print("[%4d/%4d] ok %s %4d segs  %s" % (i, len(todo), vid, len(segs), v["name"][:42]))
        except Exception as e:
            counts["err"] += 1
            print("[%4d/%4d] err %s %s %s" % (i, len(todo), vid, type(e).__name__, str(e)[:60]))
        time.sleep(args.throttle)

    print("\n=== done ===", counts)


if __name__ == "__main__":
    main()
