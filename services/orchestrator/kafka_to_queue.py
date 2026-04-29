import json
import os
import time
import traceback
from datetime import datetime
from typing import List, Optional

import requests
from kafka import KafkaConsumer
from supabase import create_client

# --------------------
# Config
# --------------------
KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "study.ingested")
KAFKA_GROUP = os.getenv("KAFKA_GROUP", "orchestrator-router-v4")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    or os.getenv("SUPABASE_KEY", "").strip()
)
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"
ORCHESTRATOR_API_URL = os.getenv(
    "ORCHESTRATOR_API_URL",
    "http://orchestrator:8000/trigger-inference-pacs",
).strip()
DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "default_engine")
ORCHESTRATOR_TRIGGER_TIMEOUT = int(os.getenv("ORCHESTRATOR_TRIGGER_TIMEOUT", "1800"))

print("Kafka router starting", flush=True)

supabase = None
if USE_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase connected", flush=True)
    except Exception as exc:
        supabase = None
        print(f"Supabase unavailable, using direct orchestrator routing: {exc}", flush=True)
else:
    print("Supabase disabled, using direct orchestrator routing", flush=True)

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BROKERS,
    group_id=KAFKA_GROUP,
    auto_offset_reset="earliest",
    value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
    enable_auto_commit=True,
)

print("Kafka subscribed", flush=True)


def resolve_models(modality: str, body_part: str) -> List[str]:
    if not supabase:
        return [DEFAULT_MODEL_ID]

    try:
        exact = (
            supabase.table("routing_rules")
            .select("ai_model_id")
            .eq("modality", modality)
            .eq("body_part", body_part)
            .execute()
        )
        exact_ids = list({row["ai_model_id"] for row in (exact.data or []) if row.get("ai_model_id")})
        if exact_ids:
            return exact_ids

        fallback = (
            supabase.table("routing_rules")
            .select("ai_model_id")
            .eq("modality", modality)
            .execute()
        )
        fallback_ids = list({row["ai_model_id"] for row in (fallback.data or []) if row.get("ai_model_id")})
        return fallback_ids or [DEFAULT_MODEL_ID]
    except Exception as exc:
        print(f"Routing rule lookup failed, using default route: {exc}", flush=True)
        return [DEFAULT_MODEL_ID]


def queue_job_in_supabase(job: dict) -> bool:
    if not supabase:
        return False
    try:
        supabase.table("inference_queue").insert(job).execute()
        return True
    except Exception as exc:
        print(f"Supabase insert failed, falling back to direct trigger: {exc}", flush=True)
        return False


def trigger_orchestrator(study_uid: str, series_uid: Optional[str] = None) -> None:
    payload = {"study_uid": study_uid}
    if series_uid:
        payload["series_uid"] = series_uid
    response = requests.post(ORCHESTRATOR_API_URL, data=payload, timeout=ORCHESTRATOR_TRIGGER_TIMEOUT)
    response.raise_for_status()
    print(f"Triggered orchestrator for study {study_uid}", flush=True)


while True:
    try:
        records = consumer.poll(timeout_ms=5000)
        if not records:
            print("Kafka heartbeat", flush=True)
            continue

        for _, messages in records.items():
            for message in messages:
                try:
                    event = message.value
                    study_uid = event.get("study_uid")
                    dicom_path = event.get("dicom_path")
                    modality = event.get("modality") or "UNKNOWN"
                    body_part = event.get("body_part") or "UNKNOWN"

                    if not study_uid:
                        print("Invalid payload: missing study_uid", flush=True)
                        continue

                    print(
                        f"Received study={study_uid} modality={modality} body_part={body_part}",
                        flush=True,
                    )

                    model_ids = resolve_models(modality, body_part)
                    queued_any = False

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
                        if queue_job_in_supabase(job):
                            queued_any = True
                            print(f"Queued job for model={model_id}", flush=True)

                    if not queued_any:
                        trigger_orchestrator(study_uid)

                except Exception:
                    print("Error processing Kafka message", flush=True)
                    traceback.print_exc()

    except Exception:
        print("Kafka consumer error", flush=True)
        traceback.print_exc()
        time.sleep(5)
