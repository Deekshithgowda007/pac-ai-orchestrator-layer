# services/orchestrator/orchestrator_api.py
import os, uuid, time, json, requests
from typing import List, Tuple, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from supabase import create_client
from prometheus_client import make_asgi_app
from prometheus_middleware import metrics_middleware
from io import BytesIO
import pydicom
import openai  # NEW

load_dotenv()
app = FastAPI(title="PAC AI Orchestrator")

# Config
SUPABASE_URL = "https://eposvgsqtvwmqtlrwpuw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVwb3N2Z3NxdHZ3bXF0bHJ3cHV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0NjgwNzgsImV4cCI6MjA2ODA0NDA3OH0.pdHXlAJFjcE1n4HoFSVWWOBG1Yhmqr_jXW0wqSdyhXg"
AI_HOST = "ai_inference"
AI_PORT = 8001
AI_UPLOAD_TIMEOUT = 600
PACS_QIDO = "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs"
PACS_WADO = "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs"
PACS_STOW = "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs/studies"
OPENAI_API_KEY = "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"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY required in env")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# prometheus
metrics_app = make_asgi_app()
app.middleware("http")(metrics_middleware)
app.mount("/metrics", metrics_app)


def extract_modality_bodypart_from_dicom_bytes(b: bytes) -> Tuple[str, str, Dict[str, str]]:
    try:
        ds = pydicom.dcmread(BytesIO(b), force=True, stop_before_pixels=False)
    except Exception:
        ds = pydicom.dcmread(BytesIO(b), force=True, stop_before_pixels=True)
    modality = getattr(ds, "Modality", "UNK") or "UNK"
    body = getattr(ds, "BodyPartExamined", None) or getattr(ds, "StudyDescription", None) or "UNKNOWN"
    meta = {}
    for tag in ("BodyPartExamined", "SeriesDescription", "StudyDescription", "ViewPosition", "PatientID", "StudyInstanceUID"):
        if hasattr(ds, tag):
            meta[tag] = str(getattr(ds, tag))
    return modality.upper(), str(body), meta


def stow_to_pacs(dicom_bytes: bytes, filename: str) -> None:
    boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
    headers = {"Content-Type": f"multipart/related; type=application/dicom; boundary={boundary}"}
    body = (
        f"--{boundary}\r\n"
        "Content-Type: application/dicom\r\n"
        f"Content-Location: {filename}\r\n\r\n"
    ).encode("utf-8") + dicom_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    resp = requests.post(PACS_STOW, headers=headers, data=body, timeout=60)
    if resp.status_code not in (200, 202):
        raise RuntimeError(f"STOW failed with {resp.status_code}: {resp.text}")


def generate_medical_summary(captions: List[str], modality: str, body_part: str) -> str:
    """
    Calls OpenAI GPT to convert captions into a proper 4-5 line medical summary.
    """
    if not OPENAI_API_KEY or not captions:
        return "No summary available."

    prompt = (
        f"You are an expert radiologist. Based on the following image descriptions, "
        f"write a concise 4-5 line medical summary. "
        f"Include findings, possible fractures or lesions, and impression.\n\n"
        f"Modality: {modality}, Body part: {body_part}\n"
        "Image descriptions:\n" + "\n".join(captions)
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a radiology expert."},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250
        )
        summary = resp.choices[0].message["content"]
        return summary.strip()
    except Exception:
        return "Failed to generate summary."


@app.post("/trigger-inference-multi")
async def trigger_inference_multi(files: List[UploadFile] = File(...)):
    start_ts = time.time()
    log_id = str(uuid.uuid4())
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    file_results, errors = [], []

    for f in files:
        content = await f.read()

        # 1️⃣ STOW to PACS
        try:
            stow_to_pacs(content, f.filename)
        except Exception as e:
            errors.append(f"{f.filename}: STOW failed {e}")
            file_results.append({"filename": f.filename, "status": "failed", "error": str(e)})
            continue

        # 2️⃣ Metadata extraction
        try:
            modality, body_part, meta = extract_modality_bodypart_from_dicom_bytes(content)
        except Exception as e:
            errors.append(f"{f.filename}: invalid DICOM {e}")
            file_results.append({"filename": f.filename, "status": "failed", "error": str(e)})
            continue

        # 3️⃣ AI inference
        files_payload = {"file": (f.filename, content, "application/dicom")}
        try:
            r = requests.post(f"http://{AI_HOST}:{AI_PORT}/upload", files=files_payload, timeout=AI_UPLOAD_TIMEOUT)
            r.raise_for_status()
            out = r.json()
        except Exception as e:
            errors.append(f"{f.filename}: AI call failed {e}")
            file_results.append({"filename": f.filename, "status": "failed", "error": str(e)})
            continue

        if not out.get("ok"):
            errors.append(f"{f.filename}: {out.get('error')}")
            file_results.append({"filename": f.filename, "status": "failed", "error": out.get("error")})
        else:
            captions = out.get("captions", [])
            # ✅ Generate proper 4-5 line summary
            summary = generate_medical_summary(captions, modality, body_part)

            file_results.append({
                "filename": f.filename,
                "status": "completed",
                "summary": summary,
                "captions": captions
            })

    db_status = "completed" if all(r["status"] == "completed" for r in file_results) else "failed"

    try:
        sb.table("inference_logs").insert({
            "id": log_id,
            "study_uid": "N/A",
            "status": db_status,
            "error_message": "; ".join(errors) if errors else None,
            "latency_ms": int((time.time() - start_ts) * 1000)
        }).execute()
    except Exception:
        pass

    return {"id": log_id, "status": db_status, "files": file_results, "errors": errors}


@app.get("/health")
def health():
    return {"ok": True, "orchestrator": "up", "ai_host": AI_HOST, "ai_port": AI_PORT}