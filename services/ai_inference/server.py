# services/ai_inference/server.py
import os
import io
import time
import uuid
import threading
import logging
import traceback
import json
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify
import pydicom
from PIL import Image
import numpy as np
import openai
from transformers import pipeline as hf_pipeline
import torch
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from supabase import create_client, Client

# ----------------- LOGGING -----------------
log = logging.getLogger("ai_inference")
logging.basicConfig(level=logging.INFO)

# ----------------- CONFIG (hardcoded as requested) -----------------
AI_WORKERS = 2
HUGGINGFACE_TOKEN = "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"
OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4o"
WORKER_TIMEOUT = 600

SUPABASE_URL = "https://eposvgsqtvwmqtlrwpuw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVwb3N2Z3NxdHZ3bXF0bHJ3cHV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0NjgwNzgsImV4cCI6MjA2ODA0NDA3OH0.pdHXlAJFjcE1n4HoFSVWWOBG1Yhmqr_jXW0wqSdyhXg"

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# DEFAULT_ENGINES now holds lists (primary then fallback models) per modality
DEFAULT_ENGINES: Dict[str, List[str]] = {
    "CT": [
        "nlpconnect/vit-gpt2-image-captioning",
        "Salesforce/blip-image-captioning-large",
        "Salesforce/blip-image-captioning-base"
    ],
    "MR": [
        "nlpconnect/vit-gpt2-image-captioning",
        "Salesforce/blip-image-captioning-large"
    ],
    "XRAY": [
        "Salesforce/blip-image-captioning-large",
        "Salesforce/blip-image-captioning-base"
    ],
    "PET": [
        "nlpconnect/vit-gpt2-image-captioning",
        "Salesforce/blip-image-captioning-large"
    ],
    "ULTRASOUND": [
        "Salesforce/blip-image-captioning-base",
        "nlpconnect/vit-gpt2-image-captioning"
    ]
}

# ----------------- PROMETHEUS METRICS -----------------
INFERENCE_REQUESTS = Counter("inference_requests_total", "Total inference requests")
INFERENCE_ERRORS = Counter("inference_errors_total", "Total inference errors")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Latency for inference request")

# ----------------- SUPABASE CLIENT -----------------
supabase_client: Optional[Client] = None
try:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    log.info("✅ Connected to Supabase")
except Exception as e:
    log.error(f"❌ Failed to connect to Supabase: {e}")

# ----------------- MODEL CACHE -----------------
MODEL_CACHE: Dict[str, Any] = {}
MODEL_LOCK = threading.Lock()

def load_hf_pipeline(model_id: str):
    """Load (and cache) HF image->text pipeline for a given model id."""
    with MODEL_LOCK:
        if model_id in MODEL_CACHE:
            return MODEL_CACHE[model_id]
        log.info("Loading HF pipeline: %s", model_id)
        if HUGGINGFACE_TOKEN:
            os.environ["HF_HUB_TOKEN"] = HUGGINGFACE_TOKEN
        device = 0 if torch.cuda.is_available() else -1
        # use "image-to-text" which works for modern image->text models
        pipe = hf_pipeline("image-to-text", model=model_id, device=device)
        MODEL_CACHE[model_id] = pipe
        return pipe

# ----------------- DICOM HELPERS -----------------
def dicom_bytes_to_image(file_bytes: bytes) -> Image.Image:
    bio = io.BytesIO(file_bytes)
    ds = pydicom.dcmread(bio, force=True)
    if not hasattr(ds, "PixelData"):
        raise RuntimeError("No PixelData found in DICOM")
    arr = ds.pixel_array
    if arr.ndim == 3 and arr.shape[0] > 1:
        arr = arr[arr.shape[0] // 2]
    a = arr.astype("float32")
    a -= a.min()
    if a.max() > 0:
        a = a / a.max()
    img = (a * 255).astype("uint8")
    if img.ndim == 2:
        return Image.fromarray(img).convert("L").convert("RGB")
    if img.ndim == 3 and img.shape[2] == 3:
        return Image.fromarray(img).convert("RGB")
    return Image.fromarray(img[..., 0]).convert("RGB")

def extract_dicom_metadata(file_bytes: bytes) -> Dict[str, str]:
    meta: Dict[str,str] = {}
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), stop_before_pixels=True, force=True)
        standard_fields = [
            "PatientID","PatientName","PatientSex","PatientAge","PatientBirthDate",
            "Modality","BodyPartExamined","SeriesDescription","StudyDescription",
            "StudyInstanceUID","SeriesInstanceUID","SOPInstanceUID",
            "AccessionNumber","StudyDate","StudyTime","ReferringPhysicianName",
            "InstitutionName","Manufacturer","ProtocolName"
        ]
        for f in standard_fields:
            if hasattr(ds, f):
                meta[f] = str(getattr(ds, f))
        # also capture other keyworded tags
        for elem in ds:
            if elem.keyword and elem.keyword not in meta and elem.keyword != "PixelData":
                try:
                    meta[elem.keyword] = str(elem.value)
                except Exception:
                    meta[elem.keyword] = repr(elem.value)
    except Exception as e:
        log.warning("Metadata extraction failed: %s", e)
    return meta

# ----------------- CAPTIONING (multi-model fallback) -----------------
def caption_with_fallback(model_list: List[str], pil_img: Image.Image, max_attempts: int = 3) -> Tuple[str, str]:
    """
    Try each model in model_list sequentially until a caption is produced.
    Returns (used_model_id, caption_text).
    """
    last_exc = None
    for model_id in model_list:
        try:
            pipe = load_hf_pipeline(model_id)
            out = pipe(pil_img)
            # output normalization
            caption_text = None
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict):
                    caption_text = first.get("generated_text") or first.get("caption") or first.get("text")
                else:
                    caption_text = str(first)
            else:
                caption_text = str(out)
            if caption_text:
                caption_text = caption_text.strip()
                log.info("Caption produced by %s: %s", model_id, caption_text[:120])
                return model_id, caption_text
        except Exception as e:
            log.warning("Model %s failed for captioning: %s", model_id, e)
            last_exc = e
            continue
    # if none succeeded, raise last exception or return an explicit error text
    err_msg = f"No caption model produced output. Last error: {repr(last_exc)}"
    log.error(err_msg)
    return "none", err_msg

# ----------------- GPT / SUMMARY (kept robust) -----------------
def call_openai_for_json(captions: List[str], metadata: Dict[str,str], model_name: str) -> Optional[Dict[str,Any]]:
    prompt = (
        "You are a board-certified radiologist assistant. Produce STRICT JSON only (no prose outside JSON) "
        "with keys: findings (array of 3-4 short bullet sentences, <=15 words each), "
        "impression (4-5 line paragraph clinical summary and next steps), "
        "probable_pathology (object with present:boolean, keywords:array, confidence:float 0-1).\n\n"
        f"Modality: {metadata.get('Modality','UNKNOWN')}\nBody part: {metadata.get('BodyPartExamined','UNKNOWN')}\n"
        f"PatientID: {metadata.get('PatientID','N/A')}\n\n"
        "Engine captions:\n" + "\n".join(captions) + "\n\nReturn JSON only."
    )
    try:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role":"system","content":"You are a helpful, precise radiology assistant."},
                      {"role":"user","content":prompt}],
            max_tokens=700,
            temperature=0.0
        )
        text = resp.choices[0].message["content"].strip()
        try:
            return json.loads(text)
        except Exception:
            # try extract JSON block
            s = text.find("{"); e = text.rfind("}")
            if s != -1 and e != -1 and e > s:
                frag = text[s:e+1]
                try:
                    return json.loads(frag)
                except Exception:
                    log.debug("OpenAI returned non-JSON text that couldn't be parsed.")
                    return None
    except Exception as e:
        log.error("OpenAI call failed: %s", e)
    return None

def build_structured_report(captions: List[str], metadata: Dict[str,str]) -> Dict[str,Any]:
    # try openai strict json across a few models
    models_to_try = [OPENAI_MODEL, "gpt-4", "gpt-4o-mini", "gpt-4o"]
    for m in models_to_try:
        parsed = call_openai_for_json(captions, metadata, m)
        if parsed and isinstance(parsed, dict):
            findings = parsed.get("findings", parsed.get("Findings", []))
            impression = parsed.get("impression", parsed.get("Impression", parsed.get("summary", "")))
            probable = parsed.get("probable_pathology", parsed.get("probable_fracture", {}))
            return {"findings": findings, "impression": impression, "probable_pathology": probable}
    # fallback: plain-text impression in 4-5 lines
    try:
        prompt = (
            f"You are an expert radiologist. Based on captions below, write a clear 4-5 line professional impression.\n"
            f"Modality: {metadata.get('Modality','UNKNOWN')}, Body part: {metadata.get('BodyPartExamined','UNKNOWN')}\n\n"
            "Captions:\n" + "\n".join(captions)
        )
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"You are a radiology expert."}, {"role":"user","content":prompt}],
            temperature=0.15,
            max_tokens=250
        )
        text = resp.choices[0].message["content"].strip()
        findings = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if len(findings) < 4 and len(line.split()) <= 15:
                findings.append(line)
        if not findings:
            findings = [s.strip() for s in text.split(".") if s][:3]
        return {"findings": findings[:4], "impression": text, "probable_pathology": {"present": False, "keywords": [], "confidence": 0.0}}
    except Exception as e:
        log.error("Fallback OpenAI plain-text failed: %s", e)
        return {"findings": captions[:3], "impression": "Summary generation failed.", "probable_pathology": {"present": False, "keywords": [], "confidence": 0.0}}

# ----------------- SUPABASE WRAPPER -----------------
def safe_insert_supabase(table: str, row: Dict[str,Any]) -> Optional[Any]:
    if not supabase_client:
        log.warning("Supabase client not initialized — skipping logging.")
        return None
    try:
        row_copy = json.loads(json.dumps(row, default=str))
        # ensure ai_model_id (UUID) remains null to avoid uuid parse errors
        if "ai_model_id" in row_copy:
            row_copy["ai_model_id"] = None
        res = supabase_client.table(table).insert(row_copy).execute()
        return getattr(res, "data", None)
    except Exception as e:
        log.error("Supabase insert error: %s", e)
        return None

# ----------------- FLASK APP -----------------
app = Flask(__name__)

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/upload", methods=["POST"])
def upload():
    start_time = time.time()
    INFERENCE_REQUESTS.inc()
    try:
        f = request.files.get("file")
        if not f:
            return jsonify({"ok": False, "error": "No file provided"}), 400
        b = f.read()

        metadata = extract_dicom_metadata(b)
        modality = metadata.get("Modality", "CT")
        body_part = metadata.get("BodyPartExamined", "UNKNOWN")

        # pick list of caption models for modality
        model_candidates = DEFAULT_ENGINES.get(modality.upper(), DEFAULT_ENGINES["CT"])

        # generate caption using fallback chain
        captions: List[str] = []
        model_used = "none"
        try:
            pil = dicom_bytes_to_image(b)
            model_used, caption = caption_with_fallback(model_candidates, pil)
            captions.append(caption)
        except Exception as e:
            INFERENCE_ERRORS.inc()
            log.error("Caption generation pipeline failed: %s", e)
            captions.append(f"[ERROR] {e}")

        # structured report
        report = build_structured_report(captions, metadata)
        impression = report.get("impression") or "Summary generation failed."

        # build payload (avoid ai_model_id uuid insertion)
        payload = {
            "id": uuid.uuid4(),
            "ai_model_id": None,
            "model_id": model_used,
            "modality": modality,
            "body_part": body_part,
            "patient_id": metadata.get("PatientID"),
            "patient_name": metadata.get("PatientName"),
            "patient_sex": metadata.get("PatientSex"),
            "patient_age": metadata.get("PatientAge"),
            "study_instance_uid": metadata.get("StudyInstanceUID"),
            "series_instance_uid": metadata.get("SeriesInstanceUID"),
            "study_description": metadata.get("StudyDescription"),
            "series_description": metadata.get("SeriesDescription"),
            "accession_number": metadata.get("AccessionNumber"),
            "study_date": metadata.get("StudyDate"),
            "study_time": metadata.get("StudyTime"),
            "referring_physician_name": metadata.get("ReferringPhysicianName"),
            "captions": captions,
            "report": report,
            "summary": impression,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }

        sup_res = safe_insert_supabase("inference_logs", payload)
        if sup_res is None:
            log.warning("Supabase insert returned no data (check key/permissions).")

        INFERENCE_LATENCY.observe(time.time() - start_time)
        return jsonify({
            "ok": True,
            "model_id": model_used,
            "dicom_metadata": metadata,
            "captions": captions,
            "report": report,
            "summary": impression,
            "modality": modality,
            "body_part": body_part,
            "latency_sec": round(time.time() - start_time, 2)
        })
    except Exception as e:
        INFERENCE_ERRORS.inc()
        log.exception("Upload error: %s", e)
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "workers": AI_WORKERS})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, threaded=True)
