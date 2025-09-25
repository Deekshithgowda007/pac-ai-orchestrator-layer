# services/ai_inference/server.py
import os, io, time, uuid, threading, logging, traceback, json, base64
from typing import Any, Dict, List, Optional
from flask import Flask, request, jsonify
import pydicom
from PIL import Image
import numpy as np
import openai
import requests
from transformers import pipeline as hf_pipeline
import torch

log = logging.getLogger("ai_inference")
logging.basicConfig(level=logging.INFO)

AI_WORKERS = 2
HUGGINGFACE_TOKEN = "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"
OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4o"  # Use GPT-4o for better vision + reasoning
WORKER_TIMEOUT = 600

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

MODEL_CACHE: Dict[str, Any] = {}
MODEL_LOCK = threading.Lock()

DEFAULT_ENGINES = {
    "CT": "Salesforce/blip-image-captioning-large",
    "MR": "Salesforce/blip-image-captioning-large",
    "XRAY": "Salesforce/blip-image-captioning-large",
    "PET": "Salesforce/blip-image-captioning-large",
    "ULTRASOUND": "Salesforce/blip-image-captioning-base"
}

def load_model(model_id: str):
    with MODEL_LOCK:
        if model_id in MODEL_CACHE:
            return MODEL_CACHE[model_id]

        log.info("Loading HF pipeline %s", model_id)
        if HUGGINGFACE_TOKEN:
            os.environ["HF_HUB_TOKEN"] = HUGGINGFACE_TOKEN

        device = 0 if torch.cuda.is_available() else -1
        pipe = hf_pipeline("image-to-text", model=model_id, device=device)
        MODEL_CACHE[model_id] = pipe
        return pipe

def dicom_bytes_to_image(file_bytes: bytes) -> Image.Image:
    bio = io.BytesIO(file_bytes)
    ds = pydicom.dcmread(bio, force=True)
    if not hasattr(ds, "PixelData"):
        raise RuntimeError("No PixelData")
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

def build_structured_report(captions: List[str], modality: str, body_part: str, pil_image: Optional[Image.Image] = None):
    """
    Build a structured report using OpenAI GPT.
    Always produce at least 3-4 findings and a 4-5 line impression.
    """
    if not OPENAI_API_KEY:
        return {
            "findings": captions,
            "impression": " ".join(captions[:2]) if captions else "No caption",
            "probable_fracture": {"present": False, "keywords": [], "confidence": 0.0}
        }

    prompt = f"""
You are a board-certified radiologist assistant.
Analyze this medical image (modality: {modality}, body part: {body_part}) and engine-generated captions below.
Write a structured professional report with:
- "findings": 3-4 short bullet points of key observations (disease, fracture, abnormality)
- "impression": 4-5 line summary describing clinical significance
- "probable_fracture": object with present(boolean), keywords(list), confidence(0-1 float)

Engine-generated captions:
{chr(10).join(captions)}
Return strictly valid JSON.
"""

    image_payload = []
    if pil_image:
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_payload = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]

    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}] + image_payload}],
            max_tokens=700,
            temperature=0.1
        )
        content = resp.choices[0].message["content"]
        return json.loads(content)
    except Exception as e:
        log.exception("OpenAI error: %s", e)
        return {
            "findings": captions,
            "impression": " ".join(captions[:2]),
            "probable_fracture": {"present": False, "keywords": [], "confidence": 0.0}
        }

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        f = request.files.get("file")
        if not f:
            return jsonify({"ok": False, "error": "No file provided"}), 400
        b = f.read()

        # extract dicom meta
        try:
            ds_meta = pydicom.dcmread(io.BytesIO(b), stop_before_pixels=True, force=True)
            modality = getattr(ds_meta, "Modality", "CT") or "CT"
            body_part = getattr(ds_meta, "BodyPartExamined", "UNKNOWN") or "UNKNOWN"
        except Exception:
            modality, body_part = "CT", "UNKNOWN"

        pil = None
        captions = []
        try:
            pil = dicom_bytes_to_image(b)
            model_id = DEFAULT_ENGINES.get(modality.upper(), DEFAULT_ENGINES.get("CT"))
            pipe = load_model(model_id)
            out = pipe(pil)
            if isinstance(out, list) and len(out) > 0:
                text = out[0].get("generated_text") or out[0].get("caption") or str(out[0])
            else:
                text = str(out)
            captions.append(f"[{model_id}] {text}")
        except Exception as e:
            log.exception("Model inference failed: %s", e)
            captions.append(f"[ERROR] {e}")

        report = build_structured_report(captions, modality, body_part, pil_image=pil)

        res = {
            "ok": True,
            "summary": report.get("impression"),
            "report": report,
            "captions": captions,
            "modality": modality,
            "body_part": body_part
        }
        return jsonify(res)
    except Exception as e:
        log.exception("Upload error: %s", e)
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "workers": AI_WORKERS})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, threaded=True)
