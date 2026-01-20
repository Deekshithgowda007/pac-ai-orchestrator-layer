-- create extension if not exists pgcrypto;

-- create table if not exists ai_models (
--   id uuid primary key default gen_random_uuid(),
--   name text not null,
--   version text,
--   modality text not null,
--   body_part text,
--   route_label text,
--   is_active boolean default true,
--   created_at timestamptz default now()
-- );

-- create table if not exists routing_rules (
--   id uuid primary key default gen_random_uuid(),
--   modality text not null,
--   body_part text not null,
--   ai_model_id uuid references ai_models(id),
--   created_at timestamptz default now()
-- );

-- create table if not exists inference_logs (
--   id uuid primary key default gen_random_uuid(),
--   ai_model_id uuid references ai_models(id),
--   study_uid text not null,
--   status text not null check (status in ('pending','completed','failed')),
--   error_message text,
--   latency_ms integer,
--   timestamp timestamptz default now()
-- );

-- existing ai_models, routing_rules, inference_logs should already be present.
-- Add queue / engines / results

create table if not exists ai_engines (
  id uuid primary key default gen_random_uuid(),
  name text not null,               -- e.g. "MRI-engine-1"
  model_provider text not null,     -- "openai", "huggingface", "local"
  model_ref text,                   -- e.g. "gpt-4o-mini" or HF model id
  modality text not null,
  body_part text,
  instance_index integer default 0, -- 0..N-1
  is_active boolean default true,
  created_at timestamptz default now()
);

create table if not exists inference_queue (
  id uuid primary key default gen_random_uuid(),
  ai_model_id uuid references ai_models(id),
  modality text not null,
  body_part text not null,
  study_uid text not null,
  status text not null check(status in ('pending','processing','completed','failed')) default 'pending',
  priority integer default 100, -- lower number = higher priority
  assigned_engine uuid,         -- ai_engines.id
  error text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists inference_results (
  id uuid primary key default gen_random_uuid(),
  queue_id uuid references inference_queue(id),
  ai_model_id uuid references ai_models(id),
  summary text,
  seg_sop_instance_uid text,
  seg_series_instance_uid text,
  stored boolean default false,
  created_at timestamptz default now()
);

create table if not exists dicom_reports (
    id uuid primary key default gen_random_uuid(),
    model_id text,
    modality text,
    body_part text,
    summary text,
    captions jsonb,
    dicom_metadata jsonb,
    created_at timestamp default now()
);

create table if not exists results (
    id uuid primary key default gen_random_uuid(),
    inference_id uuid references inference_logs(id),
    filename text,
    status text,
    model_id text,
    dicom_metadata jsonb,
    captions jsonb,
    findings jsonb,
    impression text,
    probable_pathology jsonb,
    created_at timestamptz default now()
);

<<<<<<< HEAD
create table study_queue (
  id uuid primary key default gen_random_uuid(),
  study_uid text not null,
  payload jsonb,
  status text default 'PENDING',
  created_at timestamptz default now()
);

=======
>>>>>>> 848eaec7eeb0d9bfb49ce9f5ac4982789a55ee17
-- index to pick earliest high-priority pending jobs
create index if not exists idx_inference_queue_pending_priority on inference_queue(status, priority, created_at);
