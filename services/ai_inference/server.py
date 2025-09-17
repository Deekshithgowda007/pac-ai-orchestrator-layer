# services/ai_inference/server.py
import os
import io
import time
import uuid
import json
import queue
import logging
import tempfile
import threading
import traceback
from typing import Optional, Dict, Any, List

import numpy as np
from flask import Flask, request, jsonify
import pydicom
from PIL import Image
import openai
import requests

# Transformers import is lazy (models loaded inside worker to reduce startup cost)
from transformers import pipeline as hf_pipeline
import torch

log = logging.getLogger("ai_inference")
logging.basicConfig(level=logging.INFO)

# Config from env
# === Hardcoded Config ===
# No more dependency on .env or os.getenv()

# Number of worker threads
AI_WORKERS = 2  # You can increase to 4 or 8 if you have CPU/GPU capacity

# HuggingFace token (if needed for private models)
HUGGINGFACE_TOKEN = "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"

# OpenAI API key and model (for summarization)
OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4"  # Change to "gpt-4" if you have access

# Set OpenAI globally
import openai
openai.api_key = OPENAI_API_KEY

# Worker timeout (seconds to wait for job to complete)
WORKER_TIMEOUT = 600  # 10 minutes

# Default engines per modality (hardcoded model identifiers)
DEFAULT_ENGINES = {
    "CT": "Salesforce/blip-image-captioning-large",  # <-- more descriptive captions
    "MR": "Salesforce/blip-image-captioning-large",
    "XRAY": "StanfordAIMI/chexcaption",              # <-- radiology-specific captions
    "PET": "Salesforce/blip-image-captioning-large",
    "ULTRASOUND": "Salesforce/blip-image-captioning-base",
}

# Convert model id -> pipeline instance cache (lazy)
MODEL_CACHE: Dict[str, Any] = {}
MODEL_LOCK = threading.Lock()

def load_model(model_id: str):
    with MODEL_LOCK:
        if model_id in MODEL_CACHE:
            return MODEL_CACHE[model_id]
        log.info("Loading HF pipeline for model %s ...", model_id)
        kwargs = {}
        if HUGGINGFACE_TOKEN:
            kwargs["token"] = HUGGINGFACE_TOKEN  # ✅ New param (use_auth_token is deprecated)
        device = 0 if torch.cuda.is_available() else -1
        # ✅ Explicitly specify "image-to-text"
        p = hf_pipeline("image-to-text", model=model_id, device=device, **kwargs)
        MODEL_CACHE[model_id] = p
        log.info("Model %s loaded successfully", model_id)
        return p

# Job queue: priority queue (lower number = higher priority)
JOB_QUEUE = queue.PriorityQueue()
JOB_RESULTS: Dict[str, Dict[str, Any]] = {}   # job_id -> result dict
JOB_EVENTS: Dict[str, threading.Event] = {}   # job_id -> event to notify sync waiters

app = Flask(__name__)

def dicom_bytes_to_image(file_bytes: bytes) -> np.ndarray:
    """
    Try to parse DICOM bytes into a 2D numpy array (grayscale).
    If multi-frame or a series, returns the first suitable 2D slice.
    """
    bio = io.BytesIO(file_bytes)
    try:
        ds = pydicom.dcmread(bio, force=True)  # force to be robust
    except Exception as e:
        raise RuntimeError(f"Failed to read DICOM: {e}")

    if not hasattr(ds, "PixelData"):
        raise RuntimeError("No PixelData in DICOM")

    try:
        arr = ds.pixel_array  # may raise if compressed and no handler
    except Exception as e:
        # try decode via pylibjpeg handlers (should be installed in container)
        raise RuntimeError(f"Could not decode PixelData: {e}")

    # If multi-frame (3D): pick middle slice for captioning
    if arr.ndim == 3:
        # common shape (frames, H, W) or (H, W, 3)
        if arr.shape[0] > 1 and (arr.ndim == 3 and arr.shape[0] != 3):
            slice_idx = arr.shape[0] // 2
            img = arr[slice_idx]
        elif arr.shape[-1] in (3, 4):
            # color image (H, W, C)
            img = arr
        else:
            # fallback to first frame
            img = arr[0]
    else:
        img = arr

    # Normalize to uint8
    img = img.astype(np.float32)
    # window/scale
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    img = (img * 255.0).astype(np.uint8)

    # If color mosaic or multi-channel handle
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        # if channels last and channels == 3 -> convert to rgb
        if img.shape[2] == 3:
            return img
        # else convert first channel to gray
        return img[..., 0]
    raise RuntimeError("Unsupported pixel array shape: %s" % (arr.shape,))

def bytes_to_pil_image(file_bytes: bytes) -> Image.Image:
    """Try DICOM first, then common image formats (PNG/JPEG)."""
    # try DICOM
    try:
        arr = dicom_bytes_to_image(file_bytes)
        # if array 2D -> create L mode image
        if arr.ndim == 2:
            im = Image.fromarray(arr).convert("L")
        else:
            im = Image.fromarray(arr)
        return im
    except Exception as e:
        # fallback to image
        try:
            return Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as e2:
            raise RuntimeError(f"Unsupported image/DICOM format: {e} / {e2}")

def detect_fracture_from_texts(texts: List[str], dicom_meta: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Simple heuristic-based fracture detector:
    scan generated texts for fracture keywords and try to produce a short location using DICOM tags.
    This is heuristic only — replace with a radiology model for production.
    """
    keywords = ["fracture", "fractured", "break", "displacement", "non-displaced", "open fracture", "lucency", "cortical"]
    found = []
    for t in texts:
        tlow = t.lower()
        for k in keywords:
            if k in tlow:
                found.append(k)
    present = len(found) > 0
    location = None
    if dicom_meta:
        for tag in ("BodyPartExamined", "ViewPosition", "SeriesDescription", "StudyDescription"):
            v = dicom_meta.get(tag)
            if v:
                location = v
                break
    return {
        "present": present,
        "keywords": list(set(found)),
        "suggested_location": location,
        "confidence": 0.8 if present else 0.1  # heuristic
    }

def build_structured_report(captions: List[str], dicom_meta: Optional[Dict[str,str]] = None) -> Dict[str,Any]:
    """Use OpenAI to compress/structure captions into Findings/Impression + fracture detection."""
    findings_text = "\n".join([f"- {c}" for c in captions])
    prompt = (
        "You are a radiology assistant. Given the following engine-generated descriptions of a single DICOM image or study slices, "
        "produce a structured radiology report in JSON with fields: findings (array of short sentences), impression (1-3 sentence summary), "
        "and probable_fracture (object with present:boolean, keywords:[], suggested_location:string, confidence:number).\n\n"
        "Engine outputs:\n" + findings_text + "\n\n"
        "Return strictly JSON."
    )

    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        content = resp.choices[0].message["content"]
        # try to parse JSON from the model output (it should return JSON)
        try:
            j = json.loads(content)
        except Exception:
            # fallback: simple construction based on prompt and heuristics
            fracture = detect_fracture_from_texts(captions, dicom_meta)
            j = {
                "findings": captions,
                "impression": " ".join(captions[:2]) if captions else "",
                "probable_fracture": fracture
            }
    except Exception as e:
        log.exception("OpenAI call failed: %s", e)
        fracture = detect_fracture_from_texts(captions, dicom_meta)
        j = {
            "findings": captions,
            "impression": " ".join(captions[:2]) if captions else "",
            "probable_fracture": fracture
        }
    return j

# Worker function
def worker_loop(worker_id: int):
    log.info("Worker %d started", worker_id)
    while True:
        try:
            prio, job = JOB_QUEUE.get()
            job_id = job["id"]
            log.info("Worker %d picked job %s (prio=%s)", worker_id, job_id, prio)
            result = {"ok": False, "error": None}
            start = time.time()
            try:
                file_bytes = job["file_bytes"]
                modality = job["modality"].upper()
                body_part = job.get("body_part")
                dicom_meta = job.get("dicom_meta", {})

                # convert bytes -> PIL image
                pil = bytes_to_pil_image(file_bytes)

                # Which HF models to run for this modality
                model_id = DEFAULT_ENGINES.get(modality, DEFAULT_ENGINES.get("CT"))
                captions = []
                if isinstance(model_id, list):
                    models_to_run = model_id
                else:
                    models_to_run = [model_id]

                # Run each model (load lazily)
                for i, m in enumerate(models_to_run):
                    try:
                        pipe = load_model(m)
                        # Call pipeline safely (remove num_beams if not supported)
                        try:
                            out = pipe(pil, return_text=True)  # ✅ only pass supported args
                        except TypeError as e:
                            log.warning("Retrying without extra kwargs due to: %s", e)
                            out = pipe(pil)  # fallback if pipeline rejects extra args

                        if isinstance(out, list):
                            t = out[0].get("generated_text") or out[0].get("caption") or str(out[0])
                        else:
                            t = str(out)
                        captions.append(f"[{m}] {t}")
                    except Exception as e:
                        log.exception("Engine %s failed: %s", m, e)
                        captions.append(f"[{m} ERROR] {e}")

                # Build structured report via OpenAI (or fallback heuristics)
                report = build_structured_report([c for c in captions], dicom_meta)

                result = {
                    "ok": True,
                    "summary": report.get("impression", ""),
                    "report": report,
                    "captions": captions,
                    "modality": modality,
                    "body_part": body_part,
                    "latency_s": time.time() - start
                }

            except Exception as e:
                log.exception("Job processing failed: %s", e)
                result = {"ok": False, "error": str(e), "trace": traceback.format_exc()}

            # Save result and notify any waiters
            JOB_RESULTS[job_id] = result
            evt = JOB_EVENTS.get(job_id)
            if evt:
                evt.set()

            JOB_QUEUE.task_done()
            log.info("Worker %d finished job %s", worker_id, job_id)
        except Exception as e:
            log.exception("Worker loop exception: %s", e)
            time.sleep(1)

# Start worker threads
for i in range(AI_WORKERS):
    t = threading.Thread(target=worker_loop, args=(i+1,), daemon=True)
    t.start()

@app.route("/upload", methods=["POST"])
def upload():
    """
    POST /upload
    form fields:
      - modality (CT/MR/XRAY/...)
      - body_part (optional)
      - priority (int, lower is higher priority)
      - sync (optional 'true'/'1' to wait till processed; default true)
      - file (multipart file)
    """
    try:
        modality = (request.form.get("modality") or "CT").upper()
        body_part = request.form.get("body_part", "")
        priority = int(request.form.get("priority", "50"))
        sync = request.form.get("sync", "true").lower() in ("1", "true", "yes")
        f = request.files.get("file")
        if not f:
            return jsonify({"ok": False, "error": "No file provided"}), 400
        file_bytes = f.read()
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "modality": modality,
            "body_part": body_part,
            "file_bytes": file_bytes,
            "created_at": time.time()
        }

        # metadata attempt (read DICOM tags without pixel decode)
        try:
            ds_meta = pydicom.dcmread(io.BytesIO(file_bytes), stop_before_pixels=True, force=True)
            meta = {}
            for tag in ("BodyPartExamined", "SeriesDescription", "StudyDescription", "ViewPosition", "Laterality"):
                if hasattr(ds_meta, tag):
                    meta[tag] = str(getattr(ds_meta, tag))
            job["dicom_meta"] = meta
        except Exception:
            job["dicom_meta"] = {}

        # register event for sync waiting
        if sync:
            evt = threading.Event()
            JOB_EVENTS[job_id] = evt

        JOB_QUEUE.put((priority, job))
        log.info("Enqueued job %s modality=%s priority=%s sync=%s", job_id, modality, priority, sync)

        if sync:
            # Wait up to WORKER_TIMEOUT seconds
            ev = JOB_EVENTS.get(job_id)
            finished = ev.wait(timeout=WORKER_TIMEOUT)
            # cleanup event
            JOB_EVENTS.pop(job_id, None)
            if not finished:
                return jsonify({"ok": False, "error": "Processing timeout", "job_id": job_id}), 504
            # return result
            res = JOB_RESULTS.get(job_id)
            if not res:
                return jsonify({"ok": False, "error": "No result found", "job_id": job_id}), 500
            # return structured result
            return jsonify(res)
        else:
            return jsonify({"ok": True, "job_id": job_id, "status": "queued"})
    except Exception as e:
        log.exception("Upload exception: %s", e)
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    r = JOB_RESULTS.get(job_id)
    if not r:
        return jsonify({"ok": False, "status": "unknown"}), 404
    return jsonify({"ok": True, "status": "done" if r.get("ok") else "failed", "result": r})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "workers": AI_WORKERS})

if __name__ == "__main__":
    # dev run
    app.run(host="0.0.0.0", port=8001, threaded=True)
