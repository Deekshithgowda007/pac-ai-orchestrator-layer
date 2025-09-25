-- run this on your Supabase SQL editor once
create table if not exists ai_models (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  version text,
  modality text not null,
  body_part text,
  route_label text,
  is_active boolean default true,
  created_at timestamptz default now()
);

create table if not exists routing_rules (
  id uuid primary key default gen_random_uuid(),
  modality text not null,
  body_part text not null,
  ai_model_id uuid references ai_models(id),
  created_at timestamptz default now()
);

create table if not exists inference_logs (
  id uuid primary key default gen_random_uuid(),
  ai_model_id uuid references ai_models(id),
  study_uid text not null,
  status text not null check (status in ('pending','completed','failed')),
  error_message text,
  latency_ms integer,
  timestamp timestamptz default now()
);

create table if not exists ai_engines (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  model_provider text not null,
  model_ref text,
  modality text not null,
  body_part text,
  instance_index integer default 0,
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
  priority integer default 100,
  assigned_engine uuid,
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

create index if not exists idx_inference_queue_pending_priority on inference_queue(status, priority, created_at);
