#!/usr/bin/env python3
"""Periodic monitor for a GPU transcription run. Reports progress and cost, folds
new transcripts into the live index (detached, lock-guarded), and terminates the
pod when the run is complete. Intended to be fired on a cron.

Env: RUNPOD_KEY, POD_ID, and TARGET (how many recordings this run must finish),
plus BASELINE (transcripts already present before the run started, default 0).
"""
import glob, os, subprocess, sys, time
import requests

POD = os.environ["POD_ID"]
KEY = os.environ["RUNPOD_KEY"]
TARGET = int(os.environ.get("TARGET", "0"))
BASELINE = int(os.environ.get("BASELINE", "0"))
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "index", "data", "raw")
GQL = "https://api.runpod.io/graphql?api_key=" + KEY


def gql(q):
    try:
        return requests.post(GQL, json={"query": q}, timeout=30).json()
    except Exception as e:
        return {"error": str(e)}


def pod_info():
    r = gql('query { pod(input:{podId:"%s"}) { desiredStatus costPerHr runtime{uptimeInSeconds} } }' % POD)
    return (r.get("data") or {}).get("pod")


def terminate():
    gql('mutation { podTerminate(input:{podId:"%s"}) }' % POD)


def main():
    now = time.time()
    files = glob.glob(os.path.join(RAW, "*.json"))
    total = len(files)
    last_hr = sum(1 for f in files if now - os.path.getmtime(f) < 3600)
    done = total - BASELINE
    rem = (TARGET - done) if TARGET else 0

    info = pod_info()
    status = (info or {}).get("desiredStatus") or "GONE"
    up = ((info or {}).get("runtime") or {}).get("uptimeInSeconds") if info else None
    cost = (info or {}).get("costPerHr") or 0.69
    uph = (up / 3600) if up else 0

    # Fold the latest transcripts into the live index, DETACHED so this report
    # returns immediately. A lockfile prevents overlapping embeds.
    lock = "/tmp/corpus_embed.lock"
    if not (os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 3000):
        open(lock, "w").close()
        py = sys.executable
        idx = os.path.join(HERE, "..", "index")
        subprocess.Popen(
            ["bash", "-c",
             "%s embed_incremental.py >/tmp/corpus_embed.log 2>&1; rm -f %s" % (py, lock)],
            cwd=idx, start_new_session=True)

    print("REPORT")
    print("  last hour: %d transcribed" % last_hr)
    print("  processed: %d%s" % (done, (" / %d" % TARGET) if TARGET else ""))
    print("  pod: %s | up %.1fh | ~$%.2f spent" % (status, uph, uph * cost))
    if TARGET and rem > 0 and last_hr > 0:
        print("  ETA: ~%.1f h" % (rem / last_hr))

    if TARGET and rem <= 0:
        terminate()
        print("STATUS: COMPLETE — target reached, pod terminated.")
    elif not info or status != "RUNNING":
        print("STATUS: WARNING — pod not RUNNING (%s). Investigate." % status)
    else:
        print("STATUS: RUNNING")


if __name__ == "__main__":
    main()
