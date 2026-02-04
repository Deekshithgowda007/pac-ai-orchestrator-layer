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
KAFKA_GROUP = "orchestrator-router-v3"

SUPABASE_URL = "https://eposvgsqtvwmqtlrwpuw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVwb3N2Z3NxdHZ3bXF0bHJ3cHV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0NjgwNzgsImV4cCI6MjA2ODA0NDA3OH0.pdHXlAJFjcE1n4HoFSVWWOBG1Yhmqr_jXW0wqSdyhXg"


# ============================================================
# Startup logs
# ============================================================
print("====================================", flush=True)
print("🚀 Kafka → Supabase Router starting", flush=True)
print(f"📡 Brokers : {KAFKA_BROKERS}", flush=True)
print(f"📥 Topic   : {KAFKA_TOPIC}", flush=True)
print(f"👥 Group   : {KAFKA_GROUP}", flush=True)
print("====================================", flush=True)

# ============================================================
# Validate config
# ============================================================
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("❌ Supabase credentials missing")

# ============================================================
# Supabase client
# ============================================================
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✅ Supabase connected", flush=True)

# ============================================================
# Kafka consumer
# ============================================================
consumer = KafkaConsumer(
    bootstrap_servers=KAFKA_BROKERS,
    group_id=KAFKA_GROUP,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    request_timeout_ms=30000,
    session_timeout_ms=10000,
)

consumer.subscribe([KAFKA_TOPIC])
print("🟢 Kafka consumer subscribed, waiting for messages...", flush=True)

# ============================================================
# Routing logic
# ============================================================
def resolve_models(modality: str, body_part: str):
    print(f"🔍 Resolving models | modality={modality}, body_part={body_part}", flush=True)

    res = (
        supabase
        .table("routing_rules")
        .select("ai_model_id")
        .eq("modality", modality)
        .eq("body_part", body_part)
        .eq("active", True)
        .execute()
    )

    rules = res.data or []
    model_ids = list({r["ai_model_id"] for r in rules})  # ✅ dedupe

    print(f"📜 Models resolved: {model_ids}", flush=True)
    return model_ids

# ============================================================
# Main loop
# ============================================================
while True:
    try:
        records = consumer.poll(timeout_ms=5000)

        if not records:
            print("⏳ Kafka heartbeat – no messages", flush=True)
            continue

        for _, msgs in records.items():
            for msg in msgs:
                study = msg.value

                print("====================================", flush=True)
                print(f"📥 Study received: {study}", flush=True)
                print("====================================", flush=True)

                # -----------------------------
                # Normalize payload
                # -----------------------------
                study_uid = study.get("study_uid")
                modality  = (study.get("modality") or "UNKNOWN").upper()
                body_part = (study.get("body_part") or "UNKNOWN").upper()

                if not study_uid:
                    print("⚠️ Missing study_uid — skipping", flush=True)
                    continue

                # -----------------------------
                # Resolve models
                # -----------------------------
                model_ids = resolve_models(modality, body_part)

                if not model_ids:
                    print("⏭️ No routing rules matched", flush=True)
                    continue

                # -----------------------------
                # Insert inference jobs
                # -----------------------------
                for model_id in model_ids:
                    job = {
                        "study_uid": study_uid,
                        "ai_model_id": model_id,
                        "modality": modality,
                        "body_part": body_part,
                        "status": "PENDING",
                        "priority": 5,
                        "retry_count": 0,
                    }

                    print(f"📦 Inserting job → {job}", flush=True)

                    supabase.table("inference_queue").insert(job).execute()
                    print(f"📤 Job queued → model={model_id}", flush=True)

    except Exception as e:
        print("🔥 ERROR in Kafka router", flush=True)
        print(str(e), flush=True)
        traceback.print_exc()
        time.sleep(5)
