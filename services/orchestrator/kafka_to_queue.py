import json
import time
import os
import traceback
from datetime import datetime

from kafka import KafkaConsumer
from supabase import create_client

# --------------------
# Config (ENV FIRST)
# --------------------
KAFKA_BROKERS = "kafka:29092"
KAFKA_TOPIC = "study.ingested"
KAFKA_GROUP = "orchestrator-router-v4"

SUPABASE_URL = "https://eposvgsqtvwmqtlrwpuw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVwb3N2Z3NxdHZ3bXF0bHJ3cHV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0NjgwNzgsImV4cCI6MjA2ODA0NDA3OH0.pdHXlAJFjcE1n4HoFSVWWOBG1Yhmqr_jXW0wqSdyhXg"

print("🚀 Kafka → Supabase Router starting", flush=True)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✅ Supabase connected", flush=True)

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BROKERS,
    group_id=KAFKA_GROUP,
    auto_offset_reset="earliest",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    enable_auto_commit=True,
)

print("🟢 Kafka subscribed", flush=True)


# --------------------
# MODEL RESOLUTION
# --------------------
def resolve_models(modality: str, body_part: str):
    res = (
        supabase
        .table("routing_rules")
        .select("ai_model_id")
        .eq("modality", modality)
        .eq("body_part", body_part)
        .eq("active", True)
        .execute()
    )

    return list({r["ai_model_id"] for r in (res.data or [])})


# --------------------
# MAIN LOOP
# --------------------
while True:
    try:
        records = consumer.poll(timeout_ms=5000)

        if not records:
            print("⏳ Kafka heartbeat", flush=True)
            continue

        for _, msgs in records.items():
            for msg in msgs:
                try:
                    s = msg.value

                    study_uid  = s.get("study_uid")
                    dicom_path = s.get("dicom_path")

                    # Normalize metadata (VERY IMPORTANT)
                    modality  = s.get("modality") or "UNKNOWN"
                    body_part = s.get("body_part") or "UNKNOWN"

                    if not study_uid or not dicom_path:
                        print("⚠️ Invalid payload (missing study_uid or dicom_path)", flush=True)
                        continue

                    print(
                        f"📥 Received study={study_uid} modality={modality} body_part={body_part}",
                        flush=True
                    )

                    model_ids = resolve_models(modality, body_part)

                    if not model_ids:
                        print("⚠️ No routing rules matched", flush=True)
                        continue

                    for model_id in model_ids:
                        job = {
                            "study_uid": study_uid,
                            "dicom_path": dicom_path,
                            "ai_model_id": model_id,
                            "modality": modality,
                            "body_part": body_part,
                            "status": "PENDING",
                            "priority": 5,
                            "retry_count": 0,
                            "created_at": datetime.utcnow().isoformat(),
                        }

                        try:
                            supabase.table("inference_queue").insert(job).execute()
                            print(f"📤 Job queued → model={model_id}", flush=True)
                        except Exception as db_err:
                            print("❌ Supabase insert failed", flush=True)
                            print(db_err, flush=True)

                except Exception:
                    print("❌ Error processing Kafka message", flush=True)
                    traceback.print_exc()

    except Exception:
        print("❌ Kafka consumer error", flush=True)
        traceback.print_exc()
        time.sleep(5)