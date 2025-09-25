# services/orchestrator/pacs_fetcher.py
import os, time, requests
from dotenv import load_dotenv

load_dotenv()
DCM4CHEE_QIDO = os.getenv("DCM4CHEE_BASE") or os.getenv("PACS_QIDO_URL")
ORCH_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000/trigger-inference-pacs")
POLL = int(os.getenv("AI_POLL_INTERVAL_SECONDS", "5"))

seen = set()

def list_studies():
    url = f"{DCM4CHEE_QIDO}/studies"
    r = requests.get(url, headers={"Accept":"application/dicom+json"}, timeout=20)
    r.raise_for_status()
    return r.json()

def main_loop():
    global seen
    while True:
        try:
            studies = list_studies()
            for s in studies:
                suid = s.get("0020000D", {}).get("Value",[None])[0]
                if not suid or suid in seen:
                    continue
                # mark seen
                seen.add(suid)
                print("New study:", suid, "-> calling orchestrator")
                try:
                    resp = requests.post(ORCH_URL, data={"study_uid": suid}, timeout=60)
                    print("Orchestrator response:", resp.status_code, resp.text)
                except Exception as e:
                    print("Orchestrator call failed:", e)
        except Exception as e:
            print("Fetch error:", e)
        time.sleep(POLL)

if __name__ == "__main__":
    print("Starting pacs_fetcher poll loop...")
    main_loop()
