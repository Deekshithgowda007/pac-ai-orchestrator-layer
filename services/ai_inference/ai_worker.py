import time
import os
import base64
import requests
import json
import pydicom
import socket
import traceback
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client
from inference_engine import InferenceEngine
from dicom.sr_builder import build_ai_sr
from dicom.dicom_sender import send_sr_to_dcm4chee
from kafka import KafkaProducer

# -----------------------
# CONFIG
# -----------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
WEBHOOK_URL = os.getenv("RESULT_WEBHOOK_URL", "").strip()
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "172.16.16.25:9092").strip()
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "ai.results").strip()
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"

WORKER_ID = "ai-worker-1"
POLL_INTERVAL = 5

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if USE_SUPABASE and SUPABASE_URL and SUPABASE_KEY else None
engine = InferenceEngine()
producer = None


def extract_host(target):
    value = (target or "").strip()
    if "://" in value:
        value = value.split("://", 1)[1]
    value = value.split("/", 1)[0]
    value = value.split("@")[-1]
    return value.split(":", 1)[0].strip()


def debug_resolve(label, target):
    host = extract_host(target)
    if not host:
        print(f"⚠️ {label} host is empty: {target!r}")
        return
    try:
        resolved = socket.gethostbyname(host)
        print(f"✅ {label} host resolved: {host} -> {resolved}")
    except Exception as exc:
        print(f"🔥 {label} host resolution failed: {host} ({exc})")

# -----------------------
# Kafka Producer
# -----------------------
def create_kafka_producer():
    debug_resolve("Kafka", KAFKA_BOOTSTRAP_SERVERS)
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                retries=5,
                linger_ms=10,
                request_timeout_ms=60000,
                max_request_size=20000000,
                retry_backoff_ms=3000,
                acks="all"
            )
            print("✅ Connected to Kafka:", KAFKA_BOOTSTRAP_SERVERS)
            return producer
        except Exception as e:
            print("⏳ Waiting for Kafka broker...", str(e))
            time.sleep(5)
            
print(f"🤖 AI Worker started ({WORKER_ID})")
if USE_SUPABASE:
    debug_resolve("Supabase", SUPABASE_URL)
else:
    print("ℹ️ Supabase queue polling is disabled for ai_worker")
debug_resolve("Webhook", WEBHOOK_URL)


# ------------------------------------------------
# Extract DICOM metadata
# ------------------------------------------------
def extract_dicom_metadata(dicom_path):
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    metadata = {}

    for elem in ds:
        if elem.VR != "SQ":
            tag_name = elem.keyword or str(elem.tag)
            try:
                metadata[tag_name] = str(elem.value)
            except Exception:
                metadata[tag_name] = "Unsupported Value"

    return metadata


def build_metadata_summary(metadata):
    return {
        "patient": {
            "patient_id": metadata.get("PatientID"),
            "patient_name": metadata.get("PatientName"),
            "patient_sex": metadata.get("PatientSex"),
            "patient_age": metadata.get("PatientAge"),
            "patient_birth_date": metadata.get("PatientBirthDate"),
        },
        "study": {
            "study_instance_uid": metadata.get("StudyInstanceUID"),
            "study_date": metadata.get("StudyDate"),
            "study_time": metadata.get("StudyTime"),
            "study_description": metadata.get("StudyDescription"),
            "accession_number": metadata.get("AccessionNumber"),
            "institution_name": metadata.get("InstitutionName"),
            "manufacturer": metadata.get("Manufacturer"),
            "manufacturer_model_name": metadata.get("ManufacturerModelName"),
        },
        "series": {
            "series_instance_uid": metadata.get("SeriesInstanceUID"),
            "series_description": metadata.get("SeriesDescription"),
            "protocol_name": metadata.get("ProtocolName"),
            "modality": metadata.get("Modality"),
            "body_part_examined": metadata.get("BodyPartExamined"),
            "series_number": metadata.get("SeriesNumber"),
            "instance_number": metadata.get("InstanceNumber"),
        },
        "acquisition": {
            "image_type": metadata.get("ImageType"),
            "mr_acquisition_type": metadata.get("MRAcquisitionType"),
            "magnetic_field_strength": metadata.get("MagneticFieldStrength"),
            "slice_thickness": metadata.get("SliceThickness"),
            "spacing_between_slices": metadata.get("SpacingBetweenSlices"),
            "pixel_spacing": metadata.get("PixelSpacing"),
            "rows": metadata.get("Rows"),
            "columns": metadata.get("Columns"),
            "number_of_frames": metadata.get("NumberOfFrames"),
            "repetition_time": metadata.get("RepetitionTime"),
            "echo_time": metadata.get("EchoTime"),
            "flip_angle": metadata.get("FlipAngle"),
        },
    }


def estimate_frame_count(metadata):
    value = metadata.get("NumberOfFrames")
    if value:
        try:
            return int(float(value))
        except Exception:
            pass
    return 1


def build_clinical_summary(result, metadata):
    modality = (metadata.get("Modality") or "").upper() or "UNKNOWN"
    body_part = metadata.get("BodyPartExamined") or metadata.get("StudyDescription") or metadata.get("SeriesDescription") or "UNKNOWN"
    report_type = result.get("report_type") or "unknown"
    metrics = result.get("metrics") or {}
    frame_count = int(metrics.get("slice_count") or metrics.get("frame_count") or estimate_frame_count(metadata))

    exam = f"{modality} {body_part}".strip()
    technique = (
        f"Primary sequence {metadata.get('SeriesDescription') or metadata.get('SeriesInstanceUID') or 'selected series'}; "
        f"estimated slices/frames {frame_count}."
    )
    findings = result.get("finding") or "No automated findings available."
    impression = result.get("finding") or "No automated conclusion available."

    if report_type == "preliminary-echocardiography-lv-function":
        exam = metadata.get("StudyDescription") or body_part or "Echocardiogram"
        if not str(exam).upper().startswith("US"):
            exam = f"US {exam}".strip()
        clip_label = metadata.get("SeriesDescription") or metadata.get("ProtocolName") or "selected echocardiographic cine clip"
        technique_parts = [f"Representative echocardiographic cine clip reviewed from the {clip_label}"]
        if frame_count:
            technique_parts.append(f"approximately {frame_count} frame(s) analyzed")
        frame_time_ms = metrics.get("frame_time_ms")
        if frame_time_ms:
            technique_parts.append(f"nominal frame time {float(frame_time_ms):.1f} ms")
        technique = "; ".join(technique_parts) + "."
        ef_percent = metrics.get("estimated_ef_percent")
        fractional_area_change = metrics.get("fractional_area_change")
        function_text = result.get("finding") or "left ventricular systolic function estimate unavailable"
        findings_parts = []
        if ef_percent is not None:
            findings_parts.append(
                f"Automated echocardiographic cine analysis estimated left ventricular ejection fraction at approximately {float(ef_percent):.1f}%."
            )
        if fractional_area_change is not None:
            findings_parts.append(
                f"Proxy cavity contraction between end-diastolic and end-systolic frames was about {float(fractional_area_change) * 100.0:.1f}%."
            )
        findings_parts.append(function_text)
        findings = " ".join(findings_parts)
        impression = result.get("finding") or "Formal cardiology review is required."
    elif report_type == "preliminary-lung-nodule-screening":
        exam = "CT chest"
        ct_series_label = metadata.get("SeriesDescription") or "selected chest CT series"
        technique = f"Axial chest CT reviewed from the {ct_series_label}; approximately {frame_count} slices analyzed."
        findings = result.get("finding") or "Chest CT screening did not identify a dominant pulmonary nodule candidate."
        impression = result.get("finding") or "Radiologist review is required."
    elif report_type == "preliminary-stenosis-screening":
        exam = "XA biplane angiography" if "BIPLANE" in str(metadata.get("ImageType") or "").upper() else "XA angiography"
        xa_label = metadata.get("SeriesDescription") or "selected angiographic run"
        technique = f"Dynamic angiographic run reviewed from the {xa_label}; {frame_count} frame(s) analyzed."
        findings = result.get("finding") or "No dominant focal luminal narrowing pattern was identified on screening."
        impression = result.get("finding") or "Formal angiographic interpretation is required."
    elif report_type == "preliminary-brain-tumor-segmentation":
        exam = "MR brain"
        technique = f"Multisequence brain MRI processed; approximately {frame_count} slices analyzed."
        findings = result.get("finding") or "Brain MRI segmentation screening did not mark a dominant candidate lesion burden."
        impression = result.get("finding") or "Neuroradiology review is required."
    elif report_type == "non-diagnostic-anatomy" and modality in {"MR", "MRI"}:
        exam = f"MR {body_part}".strip()
        technique = (
            f"MR anatomy-oriented AI review of the selected series; approximately {frame_count} slices analyzed."
        )
        findings = (
            "An anatomy-focused MR AI model completed structural analysis only. "
            "No pathology-specific diagnostic interpretation was available for this examination."
        )
        impression = "Anatomy-only MR AI support was generated. Formal radiologist interpretation is required."

    recommendation = result.get("recommendation") or "Clinical review required."
    summary = (
        f"Exam: {exam}\n"
        f"Study UID: {metadata.get('StudyInstanceUID') or 'UNKNOWN'}\n"
        f"Technique: {technique}\n"
        f"Findings: {findings}\n"
        f"Impression: {impression}\n"
        f"Recommendation: {recommendation}"
    )
    return {
        "exam": exam,
        "technique": technique,
        "findings": findings,
        "impression": impression,
        "summary": summary,
    }


# ------------------------------------------------
# Encode DICOM to base64
# ------------------------------------------------
def encode_dicom_base64(dicom_path):
    with open(dicom_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ------------------------------------------------
# Send Webhook
# ------------------------------------------------
def send_to_webhook(payload):
    print("Webhook delivery disabled; skipping webhook send")
    return
    debug_resolve("Webhook", WEBHOOK_URL)
    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=60)
        print("📡 Webhook response:", response.status_code)
    except Exception as e:
        print("🔥 Webhook failed:", str(e))


# ------------------------------------------------
# 🔥 Send to Kafka
# ------------------------------------------------
def send_to_kafka(payload):
    global producer
    if producer is None:
        producer = create_kafka_producer()
    try:
        future = producer.send(KAFKA_TOPIC, value=payload)
        result = future.get(timeout=10)
        print("📨 Sent to Kafka topic:", KAFKA_TOPIC)
    except Exception as e:
        print("🔥 Kafka send failed:", str(e))


# ------------------------------------------------
# Claim job
# ------------------------------------------------
def claim_job():
    if not USE_SUPABASE:
        return None
    if not supabase:
        raise Exception("SUPABASE_URL or SUPABASE_KEY is not configured for ai_worker")
    debug_resolve("Supabase", SUPABASE_URL)
    res = (
        supabase
        .table("inference_queue")
        .select("*")
        .eq("status", "PENDING")
        .limit(1)
        .execute()
    )

    jobs = res.data or []
    if not jobs:
        return None

    job = jobs[0]

    update = (
        supabase
        .table("inference_queue")
        .update({
            "status": "RUNNING",
            "worker_id": WORKER_ID,
            "updated_at": datetime.utcnow().isoformat()
        })
        .eq("id", job["id"])
        .eq("status", "PENDING")
        .execute()
    )

    if update.data:
        return job

    return None


# ------------------------------------------------
# Process job
# ------------------------------------------------
def process_job(job):

    job_id = job["id"]
    study_uid = job["study_uid"]
    dicom_path = job.get("dicom_path")
    model_id = job.get("ai_model_id")

    if not dicom_path:
        raise Exception("dicom_path missing in job")

    print(f"\n🚀 Processing Job: {job_id}")
    start_time = time.time()

    # -----------------------
    # Run AI Model
    # -----------------------
    result = engine.run(dicom_path=dicom_path)
    print("🧠 AI Result:", result)

    # -----------------------
    # Store result
    # -----------------------
    supabase.table("ai_results").insert({
        "study_uid": study_uid,
        "ai_model_id": model_id,
        "result_json": result,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

    # -----------------------
    # Build DICOM SR
    # -----------------------
    sr_text = (
        f"Model: {result.get('model_name')}\n"
        f"Finding: {result.get('finding')}\n"
        f"Abnormal: {result.get('abnormal')}\n"
        f"Confidence: {result.get('confidence')}"
    )

    sr = build_ai_sr(
        study_uid=study_uid,
        patient_id="UNKNOWN",
        patient_name="ANONYMOUS",
        result_text=sr_text
    )

    send_sr_to_dcm4chee(sr)

    # -----------------------
    # Prepare Payload
    # -----------------------
    if os.path.isdir(dicom_path):
        dicom_files = [
            os.path.join(dicom_path, f)
            for f in os.listdir(dicom_path)
            if f.lower().endswith(".dcm")
        ]
        dicom_file = dicom_files[0] if dicom_files else None
    else:
        dicom_file = dicom_path

    if dicom_file and os.path.exists(dicom_file):

        metadata = extract_dicom_metadata(dicom_file)
        dicom_base64 = encode_dicom_base64(dicom_file)

        clinical_summary = build_clinical_summary(result, metadata)
        ai_result_payload = {
            "model_name": result.get("model_name"),
            "finding": clinical_summary["impression"],
            "exam": clinical_summary["exam"],
            "technique": clinical_summary["technique"],
            "findings": clinical_summary["findings"],
            "impression": clinical_summary["impression"],
            "abnormal": result.get("abnormal"),
            "confidence": result.get("confidence"),
            "diagnostic_support": result.get("diagnostic_support"),
            "diagnostic_available": (result.get("support_matrix") or {}).get("diagnostic_available", False),
            "report_type": result.get("report_type"),
            "summary": clinical_summary["summary"],
            "recommendation": result.get("recommendation"),
            "limitations": result.get("limitations", []),
            "routing_decision": result.get("routing_decision", {}),
            "support_matrix": result.get("support_matrix", {}),
            "model_registry": result.get("support_matrix", {}),
        }

        payload = {
            "study_uid": study_uid,
            "job_id": job_id,
            "ai_model_id": model_id,
            "ai_result": ai_result_payload,
            "dicom_metadata": {
                "metadata_summary": build_metadata_summary(metadata),
                "metadata": metadata,
            },
            "dicom_file_base64": dicom_base64,
            "timestamp": datetime.utcnow().isoformat()
        }

        # ✅ Send to Webhook
        send_to_webhook(payload)

        # 🔥 Send to Kafka
        send_to_kafka(payload)

    duration = round(time.time() - start_time, 2)
    return duration


# ------------------------------------------------
# Mark completed
# ------------------------------------------------
def mark_completed(job_id, duration):
    supabase.table("inference_queue").update({
        "status": "COMPLETED",
        "duration_sec": duration,
        "updated_at": datetime.utcnow().isoformat()
    }).eq("id", job_id).execute()


# ------------------------------------------------
# Mark failed
# ------------------------------------------------
def mark_failed(job_id, error_msg):
    supabase.table("inference_queue").update({
        "status": "FAILED",
        "error_message": str(error_msg),
        "updated_at": datetime.utcnow().isoformat()
    }).eq("id", job_id).execute()


# ------------------------------------------------
# MAIN LOOP
# ------------------------------------------------
while True:
    try:
        if not USE_SUPABASE:
            print("⏳ ai_worker idle: USE_SUPABASE=false, no queue polling configured")
            time.sleep(POLL_INTERVAL)
            continue

        job = claim_job()

        if not job:
            print("⏳ No pending jobs")
            time.sleep(POLL_INTERVAL)
            continue

        job_id = job["id"]

        try:
            duration = process_job(job)
            mark_completed(job_id, duration)
            print(f"✅ Job completed: {job_id}")

        except Exception as job_error:
            print(f"🔥 Job failed: {job_error}")
            mark_failed(job_id, job_error)

    except Exception as e:
        print("🔥 Worker error:", str(e))
        print(traceback.format_exc())

    time.sleep(POLL_INTERVAL)
