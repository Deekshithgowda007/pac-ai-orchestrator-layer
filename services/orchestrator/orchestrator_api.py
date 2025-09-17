# services/orchestrator/orchestrator_api.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client
import os, uuid, time, requests
from prometheus_client import make_asgi_app
from prometheus_middleware import metrics_middleware

load_dotenv()
app = FastAPI(title="PAC AI Orchestrator")
# os.getenv("SUPABASE_URL") os.getenv("SUPABASE_KEY")
SUPABASE_URL = "https://eposvgsqtvwmqtlrwpuw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVwb3N2Z3NxdHZ3bXF0bHJ3cHV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0NjgwNzgsImV4cCI6MjA2ODA0NDA3OH0.pdHXlAJFjcE1n4HoFSVWWOBG1Yhmqr_jXW0wqSdyhXg"
AI_HOST   = "ai_inference"
AI_PORT   = 8001

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
metrics_app = make_asgi_app()
app.middleware("http")(metrics_middleware)
app.mount("/metrics", metrics_app)

class InferenceRequest(BaseModel):
    modality: str
    body_part: str

@app.post("/trigger-inference")
async def trigger_inference(
    modality: str = Form(...),
    body_part: str = Form(...),
    file: UploadFile = File(...)
):
    # 1) route lookup
    rr = sb.table("routing_rules").select("*").eq("modality", modality).eq("body_part", body_part).execute()
    if not rr.data:
        raise HTTPException(404, "No routing rule for this modality/body_part")
    ai_model_id = rr.data[0]["ai_model_id"]

    log_id = str(uuid.uuid4())
    sb.table("inference_logs").insert({
        "id": log_id,
        "ai_model_id": ai_model_id,
        "study_uid": "N/A",
        "status": "pending"
    }).execute()

    # 2) forward file to ai_inference /upload
    start = time.time()
    status, err, summary = "failed", None, None
    try:
        files = {
    "file": (file.filename, await file.read(), file.content_type or "application/dicom")
}
        data = {
    "modality": modality,
    "body_part": body_part,
    "sync": "true"
}
        r = requests.post(
    f"http://{AI_HOST}:{AI_PORT}/upload",
    files=files,
    data=data,
    timeout=600
)
        r.raise_for_status()
        out = r.json()
        status = "completed" if out.get("ok") else "failed"
        summary = out.get("summary")
    except Exception as e:
        err = str(e)

    sb.table("inference_logs").update({
        "status": status,
        "error_message": err,
        "latency_ms": int((time.time()-start)*1000)
    }).eq("id", log_id).execute()

    return {"id": log_id, "status": status, "summary": summary, "error": err}
