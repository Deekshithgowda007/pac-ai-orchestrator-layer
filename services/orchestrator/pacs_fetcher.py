import os
import json
import time
import requests
from kafka import KafkaProducer

print("🟡 pacs_fetcher starting", flush=True)

# -------------------------------------------------
# ENV (INSIDE DOCKER → USE 8080)
# -------------------------------------------------
QIDO_BASE = os.getenv(
    "DCM4CHEE_BASE",
    "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs"
)

KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "kafka:29092")
TOPIC = "study.ingested"

BASE_DICOM_DIR = os.getenv("DICOM_STORE", "/data/dicom")
SEEN_FILE = "/tmp/seen.json"

os.makedirs(BASE_DICOM_DIR, exist_ok=True)

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

            modality = s.get("00080061", {}).get("Value", ["UNKNOWN"])[0]
            body_part = s.get("00180015", {}).get("Value", ["UNKNOWN"])[0]

            study_dir = f"{BASE_DICOM_DIR}/{study_uid}"
            os.makedirs(study_dir, exist_ok=True)

            sr = requests.get(
                f"{QIDO_BASE}/studies/{study_uid}/series",
                headers={"Accept": "application/dicom+json"},
                timeout=30
            )
            if sr.status_code != 200:
                continue

            for series in sr.json():
                series_uid = series["0020000E"]["Value"][0]

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

                    with open(out_path, "wb") as f:
                        f.write(dicom.content)

            payload = {
                "study_uid": study_uid,
                "modality": modality,
                "body_part": body_part,
                "dicom_path": study_dir
            }

            producer.send(TOPIC, payload)
            producer.flush()

            seen.add(study_uid)
            json.dump(list(seen), open(SEEN_FILE, "w"))

            print(f"📤 Published {study_uid}", flush=True)

    except Exception as e:
        print("❌ Error:", e, flush=True)

    time.sleep(50)
