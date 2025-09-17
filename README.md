## Run (dev)

1. Create `.env` with Supabase keys + OPENAI_API_KEY + HF_API_KEY. Example:
   SUPABASE_URL=...
   SUPABASE_KEY=...
   OPENAI_API_KEY=sk-...
   HF_API_KEY=hf-...
   ENGINE_INSTANCES_CONFIG=MRI:Head=2,CT:Chest=1

2. Ensure db schema includes the new tables. If using Supabase, run `db/supabase_schema.sql` with psql or Supabase SQL editor.

3. Build and start:
   docker compose up --build -d

   Optionally scale worker:
   docker compose up -d --scale ai_worker=2

4. Insert at least one ai_model and routing_rules entry into Supabase (see README notes):

   ai_models:
     - name: demo-otsu
       version: v0
       modality: CT
       body_part: Head
       route_label: demo-otsu

   routing_rules:
     - modality: CT
       body_part: Head
       ai_model_id: (id of demo-otsu)

5. Trigger inference:
   curl -X POST http://localhost:8000/trigger-inference \
     -H "Content-Type: application/json" \
     -d '{"modality":"CT","body_part":"Head","study_uid":"1.2.3.4.5"}'

6. Check queue item:
   curl http://localhost:8000/queue/<queue_id>

7. Open OHIF at http://localhost:3001 (or via your proxy) to view the study and overlays.
