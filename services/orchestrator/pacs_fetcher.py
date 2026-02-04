import requests, json, time, os
from kafka import KafkaProducer
from requests.exceptions import RequestException

print("🟡 pacs_fetcher process started", flush=True)

DCM4CHEE_BASE = os.getenv(
    "DCM4CHEE_BASE",
    "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs"
)

KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "kafka:29092")
TOPIC = "study.ingested"
SEEN_FILE = "/tmp/seen.json"

# --- Kafka connection ---
producer = None
while producer is None:
    try:
        print(f"⏳ Connecting to Kafka at {KAFKA_BROKERS}", flush=True)
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=10
        )
        print("✅ Connected to Kafka", flush=True)
    except Exception as e:
        print("❌ Kafka not ready:", repr(e), flush=True)
        time.sleep(5)

# --- Seen cache ---
seen = set()
if os.path.exists(SEEN_FILE):
    seen = set(json.load(open(SEEN_FILE)))

print("🟢 PACS fetcher started")

while True:
    try:
        print("🔎 Querying DCM4CHEE studies...", flush=True)

        r = requests.get(
            f"{DCM4CHEE_BASE}/studies",
            headers={"Accept": "application/dicom+json"},
            timeout=60   # 🔥 increased
        )

        print(f"📡 PACS HTTP {r.status_code}", flush=True)

        if r.status_code == 204:
            time.sleep(15)
            continue

        r.raise_for_status()
        studies = r.json()

        print(f"📚 Found {len(studies)} studies", flush=True)

        for s in studies:
            uid = s["0020000D"]["Value"][0]
            if uid in seen:
                continue

            payload = {
                "study_uid": uid,
                "modality": s.get("00080061", {}).get("Value", ["UNKNOWN"])[0],
                "body_part": s.get("00180015", {}).get("Value", ["UNKNOWN"])[0]
            }

            producer.send(TOPIC, payload)
            producer.flush()

            seen.add(uid)
            json.dump(list(seen), open(SEEN_FILE, "w"))

            print(f"📤 Published study {uid}", flush=True)

    except RequestException as e:
        print("⚠️ PACS not ready yet:", e, flush=True)

    except Exception as e:
        print("❌ Unexpected error:", e, flush=True)

    time.sleep(50)
