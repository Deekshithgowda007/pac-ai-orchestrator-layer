import json
import os
from kafka import KafkaConsumer
from supabase import create_client

KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "kafka:29092")
TOPIC = os.getenv("KAFKA_TOPIC", "study.ingested")

SUPABASE_URL = "https://eposvgsqtvwmqtlrwpuw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVwb3N2Z3NxdHZ3bXF0bHJ3cHV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0NjgwNzgsImV4cCI6MjA2ODA0NDA3OH0.pdHXlAJFjcE1n4HoFSVWWOBG1Yhmqr_jXW0wqSdyhXg"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKERS,
    group_id="ai-orchestrator-group",
    auto_offset_reset="latest",   # 👈 CRITICAL CHANGE
    enable_auto_commit=True,
    value_deserializer=lambda v: v,  # 👈 raw bytes first
)

print("🟢 Kafka worker started, waiting for studies...")

for msg in consumer:
    try:
        raw = msg.value.decode("utf-8").strip()
        study = json.loads(raw)
    except Exception as e:
        print("❌ Invalid message, skipping:", msg.value)
        print("Reason:", e)
        continue   # 👈 DO NOT CRASH

    print(f"📥 Received study: {study}")

    supabase.table("inference_queue").insert({
        "study_uid": study["study_uid"],
        "modality": study.get("modality"),
        "body_part": study.get("body_part"),
        "payload": study,
        "status": "PENDING"
    }).execute()

    print(f"📤 Stored study {study['study_uid']} in Supabase")
