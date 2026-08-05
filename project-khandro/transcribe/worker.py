#!/usr/bin/env python3
"""GPU transcription worker (faster-whisper large-v3).

Runs on a rented GPU. For each remaining caption-less recording it:
  1. resolves the media URL (Vimeo example connector),
  2. streams it through ffmpeg to 16kHz mono WAV (video discarded, tiny on disk),
  3. transcribes with faster-whisper large-v3 (accent-robust, batched, ~30x realtime),
  4. applies the corrections map,
  5. POSTs the timestamped transcript back to the ingest endpoint,
  6. deletes the temp audio.
Resumable (skips ids already ingested). Bandwidth on the GPU host is free, so the
big downloads never cost anything and never touch the app server.

Env: VIMEO_TOKEN, INGEST_URL, INGEST_SECRET, WHISPER_MODEL (default large-v3).
Files alongside: todo.json (list of {id,title,duration}), corrections.json.
"""
import json, os, re, subprocess, tempfile, time, urllib.request
import requests
from faster_whisper import WhisperModel, BatchedInferencePipeline

HERE = os.path.dirname(os.path.abspath(__file__))
VIMEO_TOKEN = os.environ["VIMEO_TOKEN"]
INGEST_URL = os.environ["INGEST_URL"]            # your app: .../api/ingest_transcript
IDS_URL = INGEST_URL.rsplit("/", 1)[0] + "/ingested_ids"
INGEST_SECRET = os.environ["INGEST_SECRET"]
MODEL = os.environ.get("WHISPER_MODEL", "large-v3")

_corr_path = os.path.join(HERE, "corrections.json")
_corr = ([(re.compile(r["pat"], re.I), r["rep"])
          for r in json.load(open(_corr_path))["rules"]]
         if os.path.exists(_corr_path) else [])


def correct(t):
    for rx, rep in _corr:
        t = rx.sub(rep, t)
    return t


def vimeo(path):
    req = urllib.request.Request("https://api.vimeo.com" + path,
                                 headers={"Authorization": "Bearer " + VIMEO_TOKEN})
    return json.load(urllib.request.urlopen(req, timeout=30))


def media_url(vid):
    """Resolve a downloadable media URL for one recording. Swap this function to
    change source (YouTube, S3, local files); the rest of the worker is generic."""
    d = vimeo("/videos/%s?fields=download" % vid)
    dls = sorted((d.get("download") or []), key=lambda x: x.get("size") or 0)
    if not dls:
        return None
    rr = requests.get(dls[0]["link"], allow_redirects=False, stream=True, timeout=30)
    final = rr.headers.get("Location", dls[0]["link"])
    rr.close()
    return final


def extract_audio(url, wav):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", url,
                    "-vn", "-ac", "1", "-ar", "16000", wav],
                   check=True, timeout=3600)


def transcribe(pipe, wav):
    # batched inference: same large-v3 weights, ~3-4x faster on GPU, same accuracy
    segs, _ = pipe.transcribe(wav, language="en", batch_size=16, vad_filter=True)
    out = []
    for s in segs:
        txt = correct((s.text or "").strip())
        if txt:
            out.append({"text": txt, "start": round(s.start, 2),
                        "duration": round(s.end - s.start, 2)})
    return out


def already_done():
    try:
        r = requests.get(IDS_URL, headers={"x-ingest-secret": INGEST_SECRET}, timeout=30)
        return set(r.json().get("ids", []))
    except Exception:
        return set()


def main():
    todo = json.load(open(os.path.join(HERE, "todo.json")))
    done = already_done()
    todo = [v for v in todo if v["id"] not in done]
    print("to transcribe: %d (skipped %d already done)" % (len(todo), len(done)), flush=True)

    model = WhisperModel(MODEL, device="cuda", compute_type="float16")
    pipe = BatchedInferencePipeline(model=model)
    ok = err = 0
    t0 = time.time()
    for i, v in enumerate(todo, 1):
        vid = v["id"]
        wav = os.path.join(tempfile.gettempdir(), vid + ".wav")
        try:
            url = media_url(vid)
            if not url:
                err += 1; continue
            extract_audio(url, wav)
            segs = transcribe(pipe, wav)
            rec = {"id": vid, "title": v.get("title", ""),
                   "url": "https://vimeo.com/%s" % vid, "source": "asr",
                   "duration": v.get("duration"), "is_generated": True,
                   "n_segments": len(segs), "segments": segs}
            r = requests.post(INGEST_URL, headers={"x-ingest-secret": INGEST_SECRET},
                              json=rec, timeout=60)
            r.raise_for_status()
            ok += 1
            el = time.time() - t0
            print("[%d/%d] OK %s %dmin %d segs | %.0fmin elapsed, %.1f/hr" % (
                i, len(todo), vid, (v.get("duration") or 0) // 60, len(segs),
                el / 60, ok / (el / 3600 + 1e-9)), flush=True)
        except Exception as e:
            err += 1
            print("[%d/%d] ERR %s %s" % (i, len(todo), vid, str(e)[:90]), flush=True)
        finally:
            if os.path.exists(wav):
                os.remove(wav)
    print("=== done: ok=%d err=%d in %.0f min ===" % (ok, err, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
