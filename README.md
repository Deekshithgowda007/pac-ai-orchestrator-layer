# PACS-AI-ORCHESTRATION-LAYER

## Prereqs
- Docker & Docker Compose
- Supabase project (or a Postgres you can talk to) with API key
- Replace secrets in `.env` (OPENAI_API_KEY, HF_API_KEY, SUPABASE_URL, SUPABASE_KEY)

## Bring up:
docker compose up --build -d

Services:
- dcm4chee UI: http://localhost:8080
- OHIF viewer: http://localhost:3001
- Orchestrator API: http://localhost:8000
- AI inference: http://localhost:8001 (internal)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Test flow (uploading .dcm locally using storescu)
1) Send a DICOM to PACS:
   storescu -c DCM4CHEE@localhost:11112 /path/to/file.dcm

2) poller pacs_fetcher will detect new study and call orchestrator -> orchestrator pulls instances and posts to ai_inference
   You can also call orchestrator directly:
   curl -X POST http://localhost:8000/trigger-inference-pacs -F "study_uid=1.2.3...."

3) Check orchestrator logs:
   docker compose logs -f orchestrator

4) Check Supabase inference_logs/inference_results
