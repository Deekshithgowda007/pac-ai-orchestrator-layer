import os, uuid, time, json, requests, traceback
from typing import List, Optional, Tuple, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv
from supabase import create_client
from prometheus_client import make_asgi_app
from prometheus_middleware import metrics_middleware
from io import BytesIO
import pydicom
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from datetime import datetime

load_dotenv()
app = FastAPI(title="PAC AI Orchestrator")

# --- Hardcoded Config ---
SUPABASE_URL = "https://eposvgsqtvwmqtlrwpuw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVwb3N2Z3NxdHZ3bXF0bHJ3cHV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0NjgwNzgsImV4cCI6MjA2ODA0NDA3OH0.pdHXlAJFjcE1n4HoFSVWWOBG1Yhmqr_jXW0wqSdyhXg"
AI_HOST = "ai_inference"
AI_PORT = 8001
AI_UPLOAD_TIMEOUT = 600
PACS_QIDO = "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs"
PACS_WADO = "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs"
PACS_STOW = "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs/studies"
OPENAI_MODEL = "gpt-4"
OPENAI_API_KEY = "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"
AI_FALLBACK_ENGINES = ["default_engine"]
ELK_HTTP = "http://elk:9200"
ENGINE_INSTANCES_CONFIG = "default_engine:ai_inference:8001"
AI_CONCURRENCY_PER_ENGINE = 1  # hardcoded

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY required")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# Prometheus
metrics_app = make_asgi_app()
app.middleware("http")(metrics_middleware)
app.mount("/metrics", metrics_app)

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def parse_engine_config(cfg: str) -> Dict[Tuple[str,str], int]:
    out = {}
    if not cfg:
        return out
    parts = cfg.split(",")
    for p in parts:
        if "=" not in p:
            continue
        left,right = p.split("=")
        if ":" not in left:
            continue
        modality, body = left.split(":")
        try:
            out[(modality.strip().upper(), body.strip())] = int(right)
        except:
            pass
    return out

ENGINE_CONFIG = parse_engine_config(ENGINE_INSTANCES_CONFIG)

def extract_full_dicom_metadata(b: bytes) -> Dict[str, str]:
    ds = None
    try:
        ds = pydicom.dcmread(BytesIO(b), force=True, stop_before_pixels=False)
    except Exception:
        ds = pydicom.dcmread(BytesIO(b), force=True, stop_before_pixels=True)
    fields = [
        "Modality", "BodyPartExamined", "SeriesDescription", "StudyDescription",
        "PatientID", "PatientName", "PatientSex", "PatientAge",
        "AccessionNumber", "StudyInstanceUID", "SeriesInstanceUID",
        "StudyDate", "StudyTime", "ReferringPhysicianName"
    ]
    meta = {}
    for f in fields:
        if hasattr(ds, f):
            meta[f] = str(getattr(ds, f))
    return meta

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
        raise RuntimeError(f"STOW failed: {resp.status_code}: {resp.text}")

def ai_engine_endpoints() -> List[str]:
    out = [f"{AI_HOST}:{AI_PORT}"]
    out.extend(AI_FALLBACK_ENGINES)
    return out

def post_to_ai_engine(endpoint: str, dicom_bytes: bytes, filename: str, timeout: int=AI_UPLOAD_TIMEOUT) -> Dict[str,Any]:
    url = f"http://{endpoint}/upload" if not endpoint.startswith("http") else f"{endpoint}/upload"
    files_payload = {"file": (filename, dicom_bytes, "application/dicom")}
    r = requests.post(url, files=files_payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def generate_medical_summary_from_captions(captions: List[str], modality: str, body_part: str) -> str:
    if not OPENAI_API_KEY or not captions:
        return "No summary available."
    prompt = (
        f"You are a radiology expert. Provide a clear, concise medical report in 4-5 lines:\n"
        f"Modality: {modality}\nBody Part: {body_part}\n"
        f"Image descriptions:\n" + "\n".join(captions)
    )
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"You are a radiologist."},{"role":"user","content":prompt}],
            temperature=0.1,
            max_tokens=250
        )
        return resp.choices[0].message["content"].strip()
    except:
        return "Summary generation failed."

def send_to_elk(payload: Dict[str, Any]):
    if not ELK_HTTP:
        return
    try:
        requests.post(ELK_HTTP, json=payload, timeout=3)
    except:
        pass

def process_single_file(fname: str, content: bytes) -> Dict[str,Any]:
    result = {"filename": fname}
    try:
        stow_to_pacs(content, fname)
    except Exception as e:
        result.update({"status":"failed","error":f"STOW failed: {e}"})
        return result

    try:
        meta = extract_full_dicom_metadata(content)
        modality = meta.get("Modality","UNK")
        body_part = meta.get("BodyPartExamined") or meta.get("StudyDescription") or "UNKNOWN"
    except Exception as e:
        result.update({"status":"failed","error":f"invalid dicom: {e}"})
        return result

    endpoints = ai_engine_endpoints()
    out, last_err = None, None
    for ep in endpoints:
        try:
            out = post_to_ai_engine(ep, content, fname)
            break
        except Exception as e:
            last_err = e
            continue
    if out is None:
        result.update({"status":"failed","error":f"AI call failed: {last_err}"})
        return result

    captions = out.get("captions", [])
    summary = generate_medical_summary_from_captions(captions, modality, body_part)
    model_id = out.get("model_id") or "default_engine"

    # --- Insert to Supabase ---
    try:
        # inference_results
        sb.table("inference_results").insert({
            "queue_id": None,
            "ai_model_id": model_id,
            "summary": summary,
            "seg_sop_instance_uid": meta.get("SOPInstanceUID"),
            "seg_series_instance_uid": meta.get("SeriesInstanceUID"),
            "stored": True
        }).execute()

        # dicom_reports
        sb.table("dicom_reports").insert({
            "model_id": model_id,
            "modality": modality,
            "body_part": body_part,
            "summary": summary,
            "captions": json.dumps(captions),
            "dicom_metadata": meta,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        # results
        sb.table("results").insert({
            "filename": fname,
            "status": "completed",
            "model_id": model_id,
            "dicom_metadata": meta,
            "captions": captions,
            "findings": out.get("findings", {}),
            "impression": out.get("impression", ""),
            "probable_pathology": out.get("probable_pathology", []),
            "created_at": datetime.utcnow().isoformat()
        }).execute()

    except Exception as db_err:
        result.update({"db_error": str(db_err)})

    result.update({
        "status":"completed",
        "summary":summary,
        "captions":captions,
        "metadata":meta,
        "model_id":model_id
    })
    send_to_elk({**result,"timestamp":time.time()})
    return result

# --- Endpoints unchanged except they now use updated process_single_file() ---

@app.post("/trigger-inference-multi")
async def trigger_inference_multi(files: List[UploadFile] = File(...)):
    start_ts = time.time()
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    log_id = str(uuid.uuid4())
    pool_size = min(len(files), max(2, os.cpu_count() or 2))
    results, errors = [], []
    file_blobs = [(f.filename, await f.read()) for f in files]

    with ThreadPoolExecutor(max_workers=pool_size) as ex:
        futures = {ex.submit(process_single_file, fname, body): fname for fname, body in file_blobs}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if res.get("status") != "completed":
                errors.append(f"{res['filename']}: {res.get('error')}")

    db_status = "completed" if all(r["status"]=="completed" for r in results) else "failed"
    try:
        sb.table("inference_logs").insert({
            "id": log_id,
            "ai_model_id": results[0].get("model_id") if results else None,
            "study_uid": results[0].get("metadata",{}).get("StudyInstanceUID") if results else "N/A",
            "status": db_status,
            "error_message": "; ".join(errors) if errors else None,
            "latency_ms": int((time.time()-start_ts)*1000)
        }).execute()
    except:
        pass
    return {"id": log_id, "status": db_status, "files": results, "errors": errors}

@app.post("/trigger-inference-pacs")
def trigger_inference_pacs(study_uid: str = Form(...), series_uid: Optional[str] = Form(None)):
    start_ts = time.time()
    log_id = str(uuid.uuid4())
    try:
        url = f"{PACS_QIDO}/studies/{study_uid}/instances" if not series_uid \
              else f"{PACS_QIDO}/studies/{study_uid}/series/{series_uid}/instances"
        resp = requests.get(url, headers={"Accept":"application/dicom+json"}, timeout=30)
        resp.raise_for_status()
        instances = resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PACS query failed: {e}")

    results, errors = [], []
    for inst in instances:
        try:
            sop = inst.get("00080018", {}).get("Value", [None])[0]
            series_uid = inst.get("0020000E", {}).get("Value", [None])[0]
            if not sop or not series_uid:
                raise RuntimeError("invalid instance metadata")
            wado_url = f"{PACS_WADO}/studies/{study_uid}/series/{series_uid}/instances/{sop}"
            dresp = requests.get(wado_url, headers={"Accept":"application/dicom"}, timeout=60)
            dresp.raise_for_status()
            dicom_bytes = dresp.content
        except Exception as e:
            errors.append(f"fetch {sop} failed: {e}")
            results.append({"filename": f"{sop}.dcm", "status":"failed", "error": str(e)})
            continue

        res = process_single_file(f"{sop}.dcm", dicom_bytes)
        results.append(res)
        if res.get("status") != "completed":
            errors.append(f"{res.get('filename')}: {res.get('error')}")

    db_status = "completed" if all(r["status"]=="completed" for r in results) else "failed"
    try:
        sb.table("inference_logs").insert({
            "id": log_id,
            "ai_model_id": results[0].get("model_id") if results else None,
            "study_uid": study_uid,
            "status": db_status,
            "error_message": "; ".join(errors) if errors else None,
            "latency_ms": int((time.time()-start_ts)*1000)
        }).execute()
    except:
        pass
    return {"id": log_id, "status": db_status, "files": results, "errors": errors}

@app.get("/health")
def health():
    return {"ok": True, "orchestrator": "up", "ai_engines": ai_engine_endpoints()}
