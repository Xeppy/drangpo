#!/bin/bash
# Self-bootstrap for the GPU pod. Installs deps, fetches the worker bundle from
# your app, and runs faster-whisper transcription. Env expected:
# VIMEO_TOKEN, INGEST_URL, INGEST_SECRET, BUNDLE_BASE.
set -e
cd /workspace
pip install -q faster-whisper requests nvidia-cudnn-cu12 nvidia-cublas-cu12 2>&1 | tail -1 || true
apt-get update -qq && apt-get install -y -qq ffmpeg
# put pip-provided cuDNN/cuBLAS on the library path for ctranslate2
export LD_LIBRARY_PATH=$(python3 -c 'import os,nvidia.cudnn as c;print(os.path.dirname(c.__file__)+"/lib")'):$(python3 -c 'import os,nvidia.cublas as c;print(os.path.dirname(c.__file__)+"/lib")'):$LD_LIBRARY_PATH
B=$BUNDLE_BASE/$INGEST_SECRET
curl -s $B/worker.py -o worker.py
curl -s $B/todo.json -o todo.json
curl -s $B/corrections.json -o corrections.json
echo "bootstrap ready $(date -u) — starting worker"
python3 worker.py > worker.log 2>&1
echo "JOB-DONE $(date -u)" | tee -a worker.log
sleep 100000   # keep container alive for inspection; the monitor terminates the pod
