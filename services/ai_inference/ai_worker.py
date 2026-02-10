import time
import os
from datetime import datetime
import requests
from supabase import create_client

# -----------------------
# CONFIG
# -----------------------
SUPABASE_URL = "https://eposvgsqtvwmqtlrwpuw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVwb3N2Z3NxdHZ3bXF0bHJ3cHV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0NjgwNzgsImV4cCI6MjA2ODA0NDA3OH0.pdHXlAJFjcE1n4HoFSVWWOBG1Yhmqr_jXW0wqSdyhXg"

DCM4CHEE_WADO = "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs"
WORKER_ID = "ai-worker-1"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("🤖 AI Worker started")

# -----------------------
# MAIN LOOP
# -----------------------
while True:
    try:
        # 1️⃣ Get one pending job
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
            print("⏳ No pending jobs")
            time.sleep(5)
            continue

        job = jobs[0]
        job_id = job["id"]

        print(f"🚀 Picked job {job_id}")

        # 2️⃣ Mark RUNNING
        supabase.table("inference_queue").update({
            "status": "RUNNING",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", job_id).execute()

        # 3️⃣ Fetch DICOM from DCM4CHEE
        study_uid = job["study_uid"]
        print(f"📥 Fetching DICOM for {study_uid}")

        url = f"{DCM4CHEE_WADO}/studies/{study_uid}/instances"
        r = requests.get(url, headers={"Accept": "application/dicom+json"})
        r.raise_for_status()

        instances = r.json()
        print(f"📄 Instances fetched: {len(instances)}")

        # 4️⃣ Mock AI inference
        print("🧠 Running AI model (mock)")
        time.sleep(2)

        # 5️⃣ Mark COMPLETED
        supabase.table("inference_queue").update({
            "status": "COMPLETED",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", job_id).execute()

        print(f"✅ Job completed {job_id}")

    except Exception as e:
        print("🔥 ERROR:", str(e))
        time.sleep(5)
