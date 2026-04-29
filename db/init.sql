-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- CREATE TABLE hospitals(

-- id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
-- name TEXT,
-- city TEXT,
-- email TEXT UNIQUE,
-- password TEXT,
-- phone TEXT,

-- ae_title TEXT,
-- pacs_ip TEXT,
-- pacs_port INT,

-- modalities TEXT[],

-- created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

-- );

CREATE TABLE IF NOT EXISTS hospitals(
id TEXT PRIMARY KEY,
name TEXT,
city TEXT,
email TEXT,
password TEXT,
phone TEXT,
modalities TEXT
);

CREATE TABLE IF NOT EXISTS ai_reports(
id TEXT PRIMARY KEY,
study_uid TEXT,
series_uid TEXT,
sop_uid TEXT,
filename TEXT,
modality TEXT,
body_part TEXT,
status TEXT,
model_id TEXT,
summary TEXT,
report JSONB,
dicom_metadata JSONB,
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
