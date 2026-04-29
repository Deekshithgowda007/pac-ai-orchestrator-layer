import os
import json
import time
import requests
import email
from kafka import KafkaProducer

print("🟡 pacs_fetcher starting", flush=True)

# -------------------------------------------------
# ENV (INSIDE DOCKER → USE 8080)
# -------------------------------------------------
QIDO_BASE = "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs"

KAFKA_BROKERS = "kafka:29092"
TOPIC = "study.ingested"
ORCHESTRATOR_API_URL = os.getenv("ORCHESTRATOR_API_URL", "http://orchestrator:8000/trigger-inference-pacs")
ORCHESTRATOR_TRIGGER_TIMEOUT = int(os.getenv("ORCHESTRATOR_TRIGGER_TIMEOUT", "1800"))
ENABLE_DIRECT_TRIGGER = os.getenv("ENABLE_DIRECT_TRIGGER", "false").lower() == "true"

BASE_DICOM_DIR = "/data/dicom"
SEEN_FILE = "/tmp/seen.json"

os.makedirs(BASE_DICOM_DIR, exist_ok=True)


def _first_value(dataset: dict, tag: str, default: str = "") -> str:
    try:
        values = dataset.get(tag, {}).get("Value", [])
        if values:
            return str(values[0] or "").strip()
    except Exception:
        pass
    return default


def _best_body_part(study_record: dict, series_record: dict | None = None) -> str:
    candidates = [
        _first_value(study_record, "00180015"),
        _first_value(study_record, "00081030"),
    ]
    if series_record:
        candidates.extend(
            [
                _first_value(series_record, "00180015"),
                _first_value(series_record, "0008103E"),
                _first_value(series_record, "00181030"),
            ]
        )
    for candidate in candidates:
        if candidate and candidate.upper() != "UNKNOWN":
            return candidate
    return "UNKNOWN"

# -------------------------------------------------
# WAIT FOR DCM4CHEE
# -------------------------------------------------
def wait_for_pacs():
    print("⏳ Waiting for dcm4chee to become ready...", flush=True)
    while True:
        try:
            r = requests.get(
                f"{QIDO_BASE}/studies",
                headers={"Accept": "application/dicom+json"},
                timeout=10,
            )
            if r.status_code in (200, 204):
                print("✅ dcm4chee is ready", flush=True)
                return
        except Exception as e:
            print("⏳ PACS not ready:", e, flush=True)
        time.sleep(10)

wait_for_pacs()

# -------------------------------------------------
# KAFKA
# -------------------------------------------------
producer = None
while producer is None:
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        print("✅ Kafka connected", flush=True)
    except Exception as e:
        print("❌ Kafka not ready:", e, flush=True)
        time.sleep(5)

# -------------------------------------------------
# SEEN CACHE
# -------------------------------------------------
seen = set()
if os.path.exists(SEEN_FILE):
    try:
        seen = set(json.load(open(SEEN_FILE)))
    except Exception:
        seen = set()


def trigger_orchestrator_direct(study_uid):
    try:
        response = requests.post(
            ORCHESTRATOR_API_URL,
            data={"study_uid": study_uid},
            timeout=ORCHESTRATOR_TRIGGER_TIMEOUT,
        )
        response.raise_for_status()
        print(f"Directly triggered orchestrator for {study_uid}", flush=True)
    except Exception as exc:
        print(f"Direct orchestrator trigger failed for {study_uid}: {exc}", flush=True)

print("🟢 PACS fetcher running", flush=True)

# =================================================
# MAIN LOOP
# =================================================
while True:
    try:
        r = requests.get(
            f"{QIDO_BASE}/studies",
            headers={"Accept": "application/dicom+json"},
            timeout=30
        )

        if r.status_code == 204 or not r.content:
            print("⏳ No studies yet", flush=True)
            time.sleep(30)
            continue

        studies = r.json()

        for s in studies:
            study_uid = s["0020000D"]["Value"][0]

            if study_uid in seen:
                continue

            print(f"📚 New study {study_uid}", flush=True)

            modality = _first_value(s, "00080061", "UNKNOWN")
            body_part = _best_body_part(s)

            study_dir = f"{BASE_DICOM_DIR}/{study_uid}"
            os.makedirs(study_dir, exist_ok=True)

            # -------------------------------------------------
            # GET SERIES
            # -------------------------------------------------
            sr = requests.get(
                f"{QIDO_BASE}/studies/{study_uid}/series",
                headers={"Accept": "application/dicom+json"},
                timeout=30
            )
            if sr.status_code != 200:
                continue

            for series in sr.json():
                series_uid = series["0020000E"]["Value"][0]
                if body_part == "UNKNOWN":
                    body_part = _best_body_part(s, series)

                # -------------------------------------------------
                # GET INSTANCES
                # -------------------------------------------------
                ir = requests.get(
                    f"{QIDO_BASE}/studies/{study_uid}/series/{series_uid}/instances",
                    headers={"Accept": "application/dicom+json"},
                    timeout=30
                )
                if ir.status_code != 200:
                    continue

                for inst in ir.json():
                    sop_uid = inst["00080018"]["Value"][0]
                    out_path = f"{study_dir}/{sop_uid}.dcm"

                    if os.path.exists(out_path):
                        continue

                    print(f"📥 Downloading {sop_uid}", flush=True)

                    dicom = requests.get(
                        f"{QIDO_BASE}/studies/{study_uid}/series/{series_uid}/instances/{sop_uid}",
                        headers={
                            "Accept": "multipart/related; type=application/dicom"
                        },
                        timeout=120
                    )

                    dicom.raise_for_status()

                    # -------------------------------------------------
                    # PARSE MULTIPART RESPONSE CORRECTLY
                    # -------------------------------------------------
                    content_type = dicom.headers.get("Content-Type")

                    msg = email.message_from_bytes(
                        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + dicom.content
                    )

                    saved = False

                    for part in msg.walk():
                        if part.get_content_type() == "application/dicom":
                            with open(out_path, "wb") as f:
                                f.write(part.get_payload(decode=True))
                            saved = True
                            break

                    if not saved:
                        print("⚠️ No DICOM part found in multipart response", flush=True)

            # -------------------------------------------------
            # PUBLISH TO KAFKA
            # -------------------------------------------------
            payload = {
                "study_uid": study_uid,
                "modality": modality,
                "body_part": body_part,
                "dicom_path": study_dir
            }

            producer.send(TOPIC, payload)
            producer.flush()
            if ENABLE_DIRECT_TRIGGER:
                trigger_orchestrator_direct(study_uid)

            seen.add(study_uid)
            json.dump(list(seen), open(SEEN_FILE, "w"))

            print(f"📤 Published {study_uid}", flush=True)

    except Exception as e:
        print("❌ Error:", e, flush=True)

    time.sleep(50)
