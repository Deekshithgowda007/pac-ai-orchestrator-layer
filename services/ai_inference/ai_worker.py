import time
import os
import base64
import requests
import json
import pydicom
from datetime import datetime
from supabase import create_client
from inference_engine import InferenceEngine
from dicom.sr_builder import build_ai_sr
from dicom.dicom_sender import send_sr_to_dcm4chee
from kafka import KafkaProducer

# -----------------------
# CONFIG
# -----------------------
SUPABASE_URL = "https://eposvgsqtvwmqtlrwpuw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVwb3N2Z3NxdHZ3bXF0bHJ3cHV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0NjgwNzgsImV4cCI6MjA2ODA0NDA3OH0.pdHXlAJFjcE1n4HoFSVWWOBG1Yhmqr_jXW0wqSdyhXg"

WEBHOOK_URL = "https://webhook.site/ce92a4b1-df0b-40f2-9ea2-fc542aec95bc"

# 🔥 Kafka Config (ASK YOUR FRIEND FOR THIS)
KAFKA_BOOTSTRAP_SERVERS = "172.16.16.39:9092"
KAFKA_TOPIC = "ai.results"                     # <-- CHANGE THIS

WORKER_ID = "ai-worker-1"
POLL_INTERVAL = 5

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
engine = InferenceEngine()

# -----------------------
# Kafka Producer
# -----------------------
def create_kafka_producer():
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                retries=5,
                linger_ms=10,
                request_timeout_ms=60000,
                max_request_size=20000000,
                retry_backoff_ms=3000,
                acks="all"
            )
            print("✅ Connected to Kafka:", KAFKA_BOOTSTRAP_SERVERS)
            return producer
        except Exception as e:
            print("⏳ Waiting for Kafka broker...", str(e))
            time.sleep(5)
            
producer = create_kafka_producer()

print(f"🤖 AI Worker started ({WORKER_ID})")


# ------------------------------------------------
# Extract DICOM metadata
# ------------------------------------------------
def extract_dicom_metadata(dicom_path):
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    metadata = {}

    for elem in ds:
        if elem.VR != "SQ":
            tag_name = elem.keyword or str(elem.tag)
            try:
                metadata[tag_name] = str(elem.value)
            except Exception:
                metadata[tag_name] = "Unsupported Value"

    return metadata


# ------------------------------------------------
# Encode DICOM to base64
# ------------------------------------------------
def encode_dicom_base64(dicom_path):
    with open(dicom_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ------------------------------------------------
# Send Webhook
# ------------------------------------------------
def send_to_webhook(payload):
    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=60)
        print("📡 Webhook response:", response.status_code)
    except Exception as e:
        print("🔥 Webhook failed:", str(e))


# ------------------------------------------------
# 🔥 Send to Kafka
# ------------------------------------------------
def send_to_kafka(payload):
    try:
        future = producer.send(KAFKA_TOPIC, value=payload)
        result = future.get(timeout=10)
        print("📨 Sent to Kafka topic:", KAFKA_TOPIC)
    except Exception as e:
        print("🔥 Kafka send failed:", str(e))


# ------------------------------------------------
# Claim job
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

    update = (
        supabase
        .table("inference_queue")
        .update({
            "status": "RUNNING",
            "worker_id": WORKER_ID,
            "updated_at": datetime.utcnow().isoformat()
        })
        .eq("id", job["id"])
        .eq("status", "PENDING")
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
    model_id = job.get("ai_model_id")

    if not dicom_path:
        raise Exception("dicom_path missing in job")

    print(f"\n🚀 Processing Job: {job_id}")
    start_time = time.time()

    # -----------------------
    # Run AI Model
    # -----------------------
    result = engine.run(dicom_path=dicom_path)
    print("🧠 AI Result:", result)

    # -----------------------
    # Store result
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

    sr = build_ai_sr(
        study_uid=study_uid,
        patient_id="UNKNOWN",
        patient_name="ANONYMOUS",
        result_text=sr_text
    )

    send_sr_to_dcm4chee(sr)

    # -----------------------
    # Prepare Payload
    # -----------------------
    if os.path.isdir(dicom_path):
        dicom_files = [
            os.path.join(dicom_path, f)
            for f in os.listdir(dicom_path)
            if f.lower().endswith(".dcm")
        ]
        dicom_file = dicom_files[0] if dicom_files else None
    else:
        dicom_file = dicom_path

    if dicom_file and os.path.exists(dicom_file):

        metadata = extract_dicom_metadata(dicom_file)
        dicom_base64 = encode_dicom_base64(dicom_file)

        payload = {
            "study_uid": study_uid,
            "job_id": job_id,
            "ai_model_id": model_id,
            "ai_result": result,
            "dicom_metadata": metadata,
            "dicom_file_base64": dicom_base64,
            "timestamp": datetime.utcnow().isoformat()
        }

        # ✅ Send to Webhook
        send_to_webhook(payload)

        # 🔥 Send to Kafka
        send_to_kafka(payload)

    duration = round(time.time() - start_time, 2)
    return duration


# ------------------------------------------------
# Mark completed
# ------------------------------------------------
def mark_completed(job_id, duration):
    supabase.table("inference_queue").update({
        "status": "COMPLETED",
        "duration_sec": duration,
        "updated_at": datetime.utcnow().isoformat()
    }).eq("id", job_id).execute()


# ------------------------------------------------
# Mark failed
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
