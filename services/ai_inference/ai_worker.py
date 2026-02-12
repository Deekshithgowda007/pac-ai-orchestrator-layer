import time
import os
from datetime import datetime
from supabase import create_client
from inference_engine import InferenceEngine
from dicom.sr_builder import build_ai_sr
from dicom.dicom_sender import send_sr_to_dcm4chee

# -----------------------
# CONFIG
# -----------------------
SUPABASE_URL = "https://eposvgsqtvwmqtlrwpuw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVwb3N2Z3NxdHZ3bXF0bHJ3cHV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0NjgwNzgsImV4cCI6MjA2ODA0NDA3OH0.pdHXlAJFjcE1n4HoFSVWWOBG1Yhmqr_jXW0wqSdyhXg"

WORKER_ID = "ai-worker-1"

POLL_INTERVAL = 5  # seconds

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
engine = InferenceEngine()

print(f"🤖 AI Worker started ({WORKER_ID})")


# ------------------------------------------------
# Claim job safely (prevents duplicate workers)
# ------------------------------------------------
def claim_job():
    res = (
        supabase
        .table("inference_queue")
        .select("*")
        .eq("status", "PENDING")
        .limit(1)
        .execute()
    )

    jobs = res.data or []

    if not jobs:
        return None

    job = jobs[0]

    # Attempt atomic update
    update = (
        supabase
        .table("inference_queue")
        .update({
            "status": "RUNNING",
            "worker_id": WORKER_ID,
            "updated_at": datetime.utcnow().isoformat()
        })
        .eq("id", job["id"])
        .eq("status", "PENDING")  # ensures no race condition
        .execute()
    )

    if update.data:
        return job

    return None


# ------------------------------------------------
# Process job
# ------------------------------------------------
def process_job(job):
    job_id = job["id"]
    study_uid = job["study_uid"]
    dicom_path = job.get("dicom_path")
    model_id = job["ai_model_id"]

    if not dicom_path:
        raise Exception("dicom_path missing in job")

    print(f"\n🚀 Processing Job: {job_id}")
    print(f"   Study: {study_uid}")
    print(f"   Model: {model_id}")

    start_time = time.time()

    # -----------------------
    # Run AI Model
    # -----------------------
    result = engine.run(model_id=model_id, dicom_path=dicom_path)

    print("🧠 AI Result:", result)

    # -----------------------
    # Store structured result
    # -----------------------
    supabase.table("ai_results").insert({
        "study_uid": study_uid,
        "ai_model_id": model_id,
        "result_json": result,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

    # -----------------------
    # Build DICOM SR
    # -----------------------
    sr_text = (
        f"Model: {result.get('model_name')}\n"
        f"Finding: {result.get('finding')}\n"
        f"Abnormal: {result.get('abnormal')}\n"
        f"Confidence: {result.get('confidence')}"
    )

    print("📄 Building DICOM SR")

    sr = build_ai_sr(
        study_uid=study_uid,
        patient_id="UNKNOWN",
        patient_name="ANONYMOUS",
        result_text=sr_text
    )

    # -----------------------
    # Send SR to PACS
    # -----------------------
    print("📤 Sending SR to dcm4chee")
    send_sr_to_dcm4chee(sr)

    duration = round(time.time() - start_time, 2)

    print(f"⏱ Inference completed in {duration} sec")

    return duration


# ------------------------------------------------
# Mark job completed
# ------------------------------------------------
def mark_completed(job_id, duration):
    supabase.table("inference_queue").update({
        "status": "COMPLETED",
        "duration_sec": duration,
        "updated_at": datetime.utcnow().isoformat()
    }).eq("id", job_id).execute()


# ------------------------------------------------
# Mark job failed
# ------------------------------------------------
def mark_failed(job_id, error_msg):
    supabase.table("inference_queue").update({
        "status": "FAILED",
        "error_message": str(error_msg),
        "updated_at": datetime.utcnow().isoformat()
    }).eq("id", job_id).execute()


# ------------------------------------------------
# MAIN LOOP
# ------------------------------------------------
while True:
    try:
        job = claim_job()

        if not job:
            print("⏳ No pending jobs")
            time.sleep(POLL_INTERVAL)
            continue

        job_id = job["id"]

        try:
            duration = process_job(job)
            mark_completed(job_id, duration)
            print(f"✅ Job completed: {job_id}")

        except Exception as job_error:
            print(f"🔥 Job failed: {job_error}")
            mark_failed(job_id, job_error)

    except Exception as e:
        print("🔥 Worker error:", str(e))

    time.sleep(POLL_INTERVAL)
