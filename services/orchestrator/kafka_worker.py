import json, os, time
from kafka import KafkaConsumer
from supabase import create_client
from datetime import datetime

consumer = KafkaConsumer(
    "study.ingested",
    bootstrap_servers=os.getenv("KAFKA_BROKERS", "kafka:29092"),
    value_deserializer=lambda m: json.loads(m.decode()),
    group_id="orchestrator-main",
    auto_offset_reset="latest"
)

sb = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

print("🟢 Kafka orchestrator started")

for msg in consumer:
    study = msg.value
    print("📥 Received:", study["study_uid"])

    rules = (
        sb.table("routing_rules")
        .select("*")
        .eq("modality", study["modality"])
        .eq("body_part", study["body_part"])
        .eq("active", True)
        .execute()
        .data
    )

    for r in rules:
        sb.table("inference_queue").insert({
            "study_uid": study["study_uid"],
            "ai_model_id": r["ai_model_id"],
            "status": "PENDING",
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        print(f"📤 Queued model {r['ai_model_id']}")
