import json
import time
from kafka import KafkaConsumer
from supabase import create_client
from datetime import datetime
import os

# --------------------
# Config
# --------------------
KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "study.ingested")
KAFKA_GROUP = os.getenv("KAFKA_GROUP", "orchestrator-router")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# --------------------
# Init clients
# --------------------
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BROKERS,
    group_id=KAFKA_GROUP,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("🚦 Kafka → Supabase Orchestrator started")

# --------------------
# Routing logic
# --------------------
def resolve_models(modality, body_part):
    """
    Resolve routing rules from DB
    """
    rules = (
        supabase.table("routing_rules")
        .select("*")
        .eq("modality", modality)
        .eq("body_part", body_part)
        .eq("active", True)
        .execute()
        .data
    )

    if not rules:
        print("⚠️ No routing rules found")
        return []

    return [r["ai_model_id"] for r in rules]

# --------------------
# Main loop
# --------------------
for msg in consumer:
    study = msg.value
    print(f"📥 Study received: {study['study_uid']}")

    model_ids = resolve_models(
        study.get("modality"),
        study.get("body_part"),
    )

    if not model_ids:
        continue

    for model_id in model_ids:
        job = {
            "study_uid": study["study_uid"],
            "ai_model_id": model_id,
            "status": "PENDING",
            "priority": 5,
            "retry_count": 0,
            "created_at": datetime.utcnow().isoformat(),
        }

        supabase.table("inference_queue").insert(job).execute()
        print(f"📤 Job queued for model {model_id}")
