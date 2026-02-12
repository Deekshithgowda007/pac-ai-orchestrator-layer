import json
import os
from kafka import KafkaConsumer
from supabase import create_client
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
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


# -----------------------------
# ROUTING FUNCTION
# -----------------------------
def fetch_routing_rules(study):
    """
    Fetch matching routing rules ordered by priority
    """
    modality = study.get("modality")
    body_part = study.get("body_part")
    protocol = study.get("protocol_name")

    # Get all active rules
    rules = (
        sb.table("routing_rules")
        .select("*")
        .eq("active", True)
        .order("priority")
        .execute()
        .data
    )

    matched_rules = []

    for rule in rules:
        # Modality match
        if rule["modality"] not in [modality, "ANY"]:
            continue

        # Body part match
        if rule["body_part"] and body_part:
            if rule["body_part"].lower() != body_part.lower():
                continue

        # Protocol match
        if rule["protocol_contains"] and protocol:
            if rule["protocol_contains"].lower() not in protocol.lower():
                continue

        matched_rules.append(rule)

    return matched_rules


# -----------------------------
# MAIN LOOP
# -----------------------------
for msg in consumer:
    study = msg.value

    print(f"\n📥 Received study: {study['study_uid']}")
    print("   Modality:", study.get("modality"))
    print("   Body Part:", study.get("body_part"))

    matched_rules = fetch_routing_rules(study)

    if not matched_rules:
        print("⚠ No routing rules matched")
        continue

    for rule in matched_rules:
        sb.table("inference_queue").insert({
            "study_uid": study["study_uid"],
            "ai_model_id": rule["ai_model_id"],
            "status": "PENDING",
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        print(f"📤 Queued model {rule['ai_model_id']} (priority {rule['priority']})")
