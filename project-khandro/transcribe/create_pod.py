#!/usr/bin/env python3
"""Deploy a RunPod GPU pod that self-bootstraps the faster-whisper worker.

The pod pulls its worker bundle (worker.py, todo.json, corrections.json) from your
app over the ingest-secret-guarded bundle endpoint, then transcribes. Env:
  RUNPOD_KEY, VIMEO_TOKEN, INGEST_SECRET, APP_ORIGIN
  (optional) IMAGE, GPU
"""
import json, os
import requests

KEY = os.environ["RUNPOD_KEY"]
VIMEO_TOKEN = os.environ["VIMEO_TOKEN"]
INGEST_SECRET = os.environ["INGEST_SECRET"]
APP_ORIGIN = os.environ["APP_ORIGIN"].rstrip("/")          # e.g. https://your-app.example.com
BUNDLE_BASE = APP_ORIGIN + "/api/bundle"
INGEST_URL = APP_ORIGIN + "/api/ingest_transcript"
IMAGE = os.environ.get("IMAGE", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
GPU = os.environ.get("GPU", "NVIDIA GeForce RTX 4090")

docker_args = ("bash -c \"cd /workspace && curl -s "
               "$BUNDLE_BASE/$INGEST_SECRET/bootstrap.sh -o boot.sh && bash boot.sh\"")

inp = {
    "cloudType": "SECURE",
    "gpuCount": 1,
    "volumeInGb": 0,
    "containerDiskInGb": 40,
    "minVcpuCount": 4,
    "minMemoryInGb": 20,
    "gpuTypeId": GPU,
    "name": "corpus-whisper",
    "imageName": IMAGE,
    "dockerArgs": docker_args,
    "ports": "8888/http",
    "volumeMountPath": "/workspace",
    "env": [
        {"key": "VIMEO_TOKEN", "value": VIMEO_TOKEN},
        {"key": "INGEST_URL", "value": INGEST_URL},
        {"key": "INGEST_SECRET", "value": INGEST_SECRET},
        {"key": "BUNDLE_BASE", "value": BUNDLE_BASE},
    ],
}
query = """mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) { id imageName costPerHr desiredStatus }
}"""
r = requests.post("https://api.runpod.io/graphql?api_key=" + KEY,
                  json={"query": query, "variables": {"input": inp}}, timeout=60)
print(r.status_code)
print(json.dumps(r.json(), indent=1)[:900])
