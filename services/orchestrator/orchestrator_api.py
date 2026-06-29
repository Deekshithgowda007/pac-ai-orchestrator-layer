import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import email
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pydicom
from pydicom.sequence import Sequence as DicomSequence
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from kafka import KafkaProducer
from prometheus_client import Counter, Histogram, make_asgi_app
from supabase import create_client

from prometheus_middleware import metrics_middleware

load_dotenv()

app = FastAPI(title="PAC AI Orchestrator")

USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
AI_HOST = os.getenv("AI_HOST", "ai_inference")
AI_PORT = int(os.getenv("AI_PORT", "8001"))
AI_ENGINE_ENDPOINTS = os.getenv("AI_ENGINE_ENDPOINTS", "").strip()
AI_UPLOAD_TIMEOUT = int(os.getenv("AI_UPLOAD_TIMEOUT", "600"))
PACS_QIDO = os.getenv("PACS_QIDO_URL", "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs")
PACS_WADO = os.getenv("PACS_WADO_URL", "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs")
PACS_STOW = os.getenv("PACS_STOW_URL", "http://dcm4chee-arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs/studies")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELK_HTTP = os.getenv("ELK_HTTP", "").strip()
RESULT_WEBHOOK_URL = os.getenv("RESULT_WEBHOOK_URL", "").strip()
HOSPITAL_RESULT_URL = os.getenv("HOSPITAL_RESULT_URL", "http://hospital-service:8001/results").strip()
KAFKA_RESULT_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").strip()
KAFKA_RESULT_TOPIC = os.getenv("KAFKA_TOPIC", "ai.results").strip()

sb = create_client(SUPABASE_URL, SUPABASE_KEY) if USE_SUPABASE and SUPABASE_URL and SUPABASE_KEY else None
kafka_result_producer: Optional[KafkaProducer] = None
metrics_app = make_asgi_app()
app.middleware("http")(metrics_middleware)
app.mount("/metrics", metrics_app)

CHAIN_SELECTION_TOTAL = Counter(
    "orchestrator_chain_selection_total",
    "Total second-stage chain selection decisions",
    ["route_name", "next_stage", "decision"],
)
CHAIN_BLOCK_REASON_TOTAL = Counter(
    "orchestrator_chain_block_reason_total",
    "Total second-stage chain blocking reasons",
    ["route_name", "reason"],
)
CHAIN_EXECUTION_TOTAL = Counter(
    "orchestrator_chain_execution_total",
    "Total second-stage chain execution outcomes",
    ["route_name", "stage_name", "status"],
)
CHAIN_MERGE_TOTAL = Counter(
    "orchestrator_chain_merge_total",
    "Total second-stage chain merge outcomes",
    ["route_name", "merge_mode", "applied"],
)
CHAIN_STAGE_DURATION_SECONDS = Histogram(
    "orchestrator_chain_stage_duration_seconds",
    "Second-stage chain execution duration in seconds",
    ["route_name", "stage_name"],
)


def get_kafka_result_producer() -> Optional[KafkaProducer]:
    global kafka_result_producer
    if not KAFKA_RESULT_BOOTSTRAP_SERVERS:
        return None
    if kafka_result_producer is not None:
        return kafka_result_producer
    try:
        kafka_result_producer = KafkaProducer(
            bootstrap_servers=KAFKA_RESULT_BOOTSTRAP_SERVERS,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            retries=5,
            linger_ms=10,
            request_timeout_ms=60000,
            max_request_size=20000000,
            retry_backoff_ms=3000,
            acks="all",
        )
        print(
            f"Kafka result producer connected to {KAFKA_RESULT_BOOTSTRAP_SERVERS} topic={KAFKA_RESULT_TOPIC}",
            flush=True,
        )
    except Exception as exc:
        kafka_result_producer = None
        print(f"Kafka result producer unavailable: {exc}", flush=True)
    return kafka_result_producer


def extract_full_dicom_metadata(dicom_bytes: bytes) -> Dict[str, str]:
    try:
        ds = pydicom.dcmread(BytesIO(dicom_bytes), force=True, stop_before_pixels=True)
    except Exception:
        ds = pydicom.dcmread(BytesIO(dicom_bytes), force=True)
    metadata: Dict[str, str] = {}
    for elem in ds:
        if elem.keyword and elem.keyword != "PixelData":
            try:
                metadata[elem.keyword] = summarize_metadata_value(elem.keyword, elem.value)
            except Exception:
                metadata[elem.keyword] = repr(elem.value)
    return metadata


def summarize_metadata_value(keyword: str, value: Any) -> str:
    if isinstance(value, DicomSequence):
        return f"<Sequence with {len(value)} item(s)>"

    if isinstance(value, (list, tuple)):
        preview = ", ".join(str(item) for item in list(value)[:8])
        if len(value) > 8:
            preview += f", ... ({len(value)} total)"
        return f"[{preview}]"

    rendered = str(value)
    if len(rendered) > 300:
        return f"{rendered[:300]}... [truncated {len(rendered)} chars]"
    return rendered


def build_metadata_summary(metadata: Dict[str, str]) -> Dict[str, Any]:
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


def validate_required_metadata(
    metadata: Dict[str, str],
    required_fields: Optional[List[str]] = None,
) -> Tuple[bool, str, str]:
    required = required_fields or ["StudyInstanceUID", "SeriesInstanceUID", "Modality"]
    missing = [field for field in required if not str(metadata.get(field) or "").strip()]
    if missing:
        return (
            False,
            f"DICOM metadata is missing required field(s): {', '.join(missing)}.",
            "invalid-dicom-metadata",
        )
    return True, "", ""


def build_radiology_style_summary(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    report = item.get("report") or {}
    metrics = report.get("metrics") or {}

    series_description = metadata.get("SeriesDescription") or item.get("series_uid") or "Unnamed series"
    modality = metadata.get("Modality") or item.get("modality") or "UNKNOWN"
    body_part = metadata.get("BodyPartExamined") or item.get("body_part") or "UNKNOWN"
    protocol = metadata.get("ProtocolName") or "Not provided"
    slice_count = metrics.get("slice_count") or estimate_frame_count(metadata, item.get("instance_count", 1))
    findings = report.get("observations") or []
    segmented = metrics.get("top_segmented_structures_ml") or {}

    lines = [
        f"Exam: {modality} {body_part}",
        f"Series: {series_description}",
        f"Technique: Protocol {protocol}; estimated slices/frames {slice_count}.",
    ]

    if segmented:
        segmented_names = ", ".join(list(segmented.keys())[:5])
        lines.append(f"Automated anatomy structures identified: {segmented_names}.")

    if findings and report.get("analysis_type") != "sequence-metadata-only":
        lines.append(f"Automated findings: {findings[0]}")
    elif report.get("analysis_type") == "sequence-metadata-only":
        lines.append("Sequence role: Supporting metadata-only sequence.")

    conclusion = report.get("conclusion") or "No automated conclusion available."
    recommendation = report.get("recommendation") or "Radiologist review is still required."
    lines.append(f"Impression: {conclusion}")
    lines.append(f"Recommendation: {recommendation}")
    return "\n".join(lines)


def extract_dicom_from_multipart(response: requests.Response) -> bytes:
    content_type = response.headers.get("Content-Type", "")
    message = email.message_from_bytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + response.content
    )
    for part in message.walk():
        if part.get_content_type() == "application/dicom":
            payload = part.get_payload(decode=True)
            if payload:
                return payload
    raise RuntimeError("No DICOM part found in PACS multipart response")


def stow_to_pacs(dicom_bytes: bytes, filename: str) -> None:
    boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
    headers = {"Content-Type": f"multipart/related; type=application/dicom; boundary={boundary}"}
    body = (
        f"--{boundary}\r\n"
        "Content-Type: application/dicom\r\n"
        f"Content-Location: {filename}\r\n\r\n"
    ).encode("utf-8") + dicom_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    response = requests.post(PACS_STOW, headers=headers, data=body, timeout=60)
    if response.status_code not in (200, 202):
        raise RuntimeError(f"STOW failed: {response.status_code}: {response.text}")


def ai_engine_candidates() -> List[str]:
    endpoints: List[str] = []
    if AI_ENGINE_ENDPOINTS:
        endpoints.extend([item.strip() for item in AI_ENGINE_ENDPOINTS.split(",") if item.strip()])
    if not endpoints:
        endpoints.append(f"{AI_HOST}:{AI_PORT}")
    return endpoints


def post_to_ai_engine(dicom_bytes: bytes, filename: str, timeout: int = AI_UPLOAD_TIMEOUT) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for endpoint in ai_engine_candidates():
        url = endpoint if endpoint.startswith("http") else f"http://{endpoint}"
        try:
            response = requests.post(
                f"{url}/upload",
                data=dicom_bytes,
                headers={
                    "Content-Type": "application/dicom",
                    "X-Filename": filename,
                    "Host": "localhost",
                },
                timeout=timeout,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"{response.status_code} from {endpoint}: {response.text}")
            payload = response.json()
            payload["engine_endpoint"] = endpoint
            return payload
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"All AI engines failed. Last error: {last_error}")


def post_series_to_ai_engine(series_files: List[Dict[str, Any]], timeout: int = AI_UPLOAD_TIMEOUT) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    multipart_files = [
        ("files", (item["filename"], item["content"], "application/dicom"))
        for item in series_files
    ]
    for endpoint in ai_engine_candidates():
        url = endpoint if endpoint.startswith("http") else f"http://{endpoint}"
        try:
            response = requests.post(
                f"{url}/upload",
                files=multipart_files,
                headers={"Host": "localhost"},
                timeout=timeout,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"{response.status_code} from {endpoint}: {response.text}")
            payload = response.json()
            payload["engine_endpoint"] = endpoint
            return payload
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"All AI engines failed. Last error: {last_error}")


def generate_summary_from_report(report: Dict[str, Any], metadata: Dict[str, str]) -> str:
    conclusion = report.get("conclusion", "No automated conclusion available.")
    observations = report.get("observations") or []
    summary_lines = [
        f"Modality: {metadata.get('Modality', 'UNKNOWN')}",
        f"Body part: {metadata.get('BodyPartExamined') or metadata.get('StudyDescription') or metadata.get('SeriesDescription') or 'UNKNOWN'}",
        f"Analysis type: {report.get('analysis_type', 'unknown')}",
        f"Status: {report.get('abnormality_status', 'unknown')}",
        f"Region: {report.get('region_of_interest', 'diffuse/undetermined')}",
    ]
    if observations:
        summary_lines.append(f"Observation: {observations[0]}")
    summary_lines.append(f"Impact: {report.get('impact', 'No impact assessment available.')}")
    summary_lines.append(f"Conclusion: {conclusion}")
    summary_lines.append(f"Recommendation: {report.get('recommendation', 'No recommendation available.')}")
    return "\n".join(summary_lines)


def send_to_elk(payload: Dict[str, Any]) -> None:
    if not ELK_HTTP:
        return
    try:
        requests.post(ELK_HTTP, json=payload, timeout=3)
    except Exception:
        pass


def build_webhook_observer_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    reduced_payload: Dict[str, Any] = {}
    for key in [
        "study_uid",
        "job_id",
        "status",
        "inference_status",
        "created_at",
        "latency_ms",
        "errors",
    ]:
        if key in payload:
            reduced_payload[key] = payload[key]

    ai_result = payload.get("ai_result") or {}
    reduced_ai_result: Dict[str, Any] = {}
    for key in [
        "finding",
        "exam",
        "technique",
        "findings",
        "impression",
        "abnormal",
        "confidence",
        "confidence_band",
        "diagnostic_support",
        "diagnostic_available",
        "report_type",
        "summary",
        "recommendation",
        "limitations",
    ]:
        if key in ai_result:
            reduced_ai_result[key] = ai_result[key]
    reduced_payload["ai_result"] = reduced_ai_result
    reduced_payload["dicom_metadata"] = payload.get("dicom_metadata") or {}
    return reduced_payload


def send_to_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not RESULT_WEBHOOK_URL:
        return {
            "target": "webhook",
            "enabled": False,
            "delivered": False,
            "status": "disabled",
            "detail": "Webhook URL is empty.",
        }
    try:
        observer_payload = build_webhook_observer_payload(payload)
        response = requests.post(RESULT_WEBHOOK_URL, json=observer_payload, timeout=30)
        response.raise_for_status()
        return {
            "target": "webhook",
            "enabled": True,
            "delivered": True,
            "status": "delivered",
            "detail": f"HTTP {response.status_code}",
            "url": RESULT_WEBHOOK_URL,
        }
    except Exception as exc:
        print(f"Webhook delivery failed: {exc}", flush=True)
        return {
            "target": "webhook",
            "enabled": True,
            "delivered": False,
            "status": "failed",
            "detail": str(exc),
            "url": RESULT_WEBHOOK_URL,
        }


def send_to_kafka(payload: Dict[str, Any]) -> Dict[str, Any]:
    producer = get_kafka_result_producer()
    if not producer:
        detail = "Kafka producer is unavailable."
        print("Kafka result publish skipped because producer is unavailable", flush=True)
        return {
            "target": "kafka",
            "enabled": bool(KAFKA_RESULT_BOOTSTRAP_SERVERS),
            "delivered": False,
            "status": "unavailable",
            "detail": detail,
            "bootstrap_servers": KAFKA_RESULT_BOOTSTRAP_SERVERS,
            "topic": KAFKA_RESULT_TOPIC,
        }
    try:
        producer.send(KAFKA_RESULT_TOPIC, value=payload).get(timeout=30)
        print(
            f"Kafka result published to {KAFKA_RESULT_BOOTSTRAP_SERVERS} topic={KAFKA_RESULT_TOPIC}",
            flush=True,
        )
        return {
            "target": "kafka",
            "enabled": True,
            "delivered": True,
            "status": "delivered",
            "detail": "Kafka publish succeeded.",
            "bootstrap_servers": KAFKA_RESULT_BOOTSTRAP_SERVERS,
            "topic": KAFKA_RESULT_TOPIC,
        }
    except Exception as exc:
        print(f"Kafka result publish failed: {exc}", flush=True)
        return {
            "target": "kafka",
            "enabled": True,
            "delivered": False,
            "status": "failed",
            "detail": str(exc),
            "bootstrap_servers": KAFKA_RESULT_BOOTSTRAP_SERVERS,
            "topic": KAFKA_RESULT_TOPIC,
        }


def send_to_hospital_service(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not HOSPITAL_RESULT_URL:
        return {
            "target": "hospital_service",
            "enabled": False,
            "delivered": False,
            "status": "disabled",
            "detail": "Hospital result URL is empty.",
        }
    try:
        response = requests.post(HOSPITAL_RESULT_URL, json=payload, timeout=30)
        response.raise_for_status()
        return {
            "target": "hospital_service",
            "enabled": True,
            "delivered": True,
            "status": "delivered",
            "detail": f"HTTP {response.status_code}",
            "url": HOSPITAL_RESULT_URL,
        }
    except Exception as exc:
        print(f"Hospital service delivery failed: {exc}", flush=True)
        return {
            "target": "hospital_service",
            "enabled": True,
            "delivered": False,
            "status": "failed",
            "detail": str(exc),
            "url": HOSPITAL_RESULT_URL,
        }


def publish_result_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    delivery_status = {
        "hospital_service": send_to_hospital_service(payload),
        "webhook": send_to_webhook(payload),
        "kafka": send_to_kafka(payload),
    }
    payload["delivery_status"] = delivery_status
    payload["delivery_summary"] = {
        "all_enabled_targets_delivered": all(
            (not target_status.get("enabled")) or target_status.get("delivered")
            for target_status in delivery_status.values()
        ),
        "failed_targets": [
            name
            for name, target_status in delivery_status.items()
            if target_status.get("enabled") and not target_status.get("delivered")
        ],
    }
    return payload


def derive_inference_status(results: List[Dict[str, Any]], errors: List[str]) -> str:
    if not results:
        return "failed"
    completed = [item for item in results if item.get("status") == "completed"]
    if completed and not errors and len(completed) == len(results):
        return "completed"
    if completed:
        return "partial"
    return "failed"


def derive_delivery_status(payload: Dict[str, Any]) -> str:
    summary = payload.get("delivery_summary") or {}
    delivery_status = payload.get("delivery_status") or {}
    enabled_targets = [
        target_status
        for target_status in delivery_status.values()
        if target_status.get("enabled")
    ]
    if not enabled_targets:
        return "not-configured"
    if summary.get("all_enabled_targets_delivered"):
        return "delivered"
    if any(target_status.get("delivered") for target_status in enabled_targets):
        return "partial"
    return "failed"


def derive_operation_status(inference_status: str, delivery_status: str) -> str:
    if inference_status == "failed":
        return "failed"
    if inference_status == "partial":
        return "partial"
    if delivery_status in {"failed", "partial"}:
        return "degraded"
    return "completed"


def build_failed_result(
    *,
    filename: str,
    series_uid: Optional[str] = None,
    study_uid: Optional[str] = None,
    modality: Optional[str] = None,
    body_part: Optional[str] = None,
    error: str,
    failure_stage: str,
    error_category: str,
) -> Dict[str, Any]:
    return {
        "filename": filename,
        "series_uid": series_uid,
        "study_uid": study_uid,
        "modality": modality,
        "body_part": body_part,
        "status": "failed",
        "error": error,
        "failure_stage": failure_stage,
        "error_category": error_category,
        "manual_review_required": True,
    }


def categorize_failure(failure_stage: str, error: str) -> str:
    message = (error or "").lower()
    if failure_stage == "input-validation":
        return "invalid-input"
    if "timeout" in message or "timed out" in message:
        return "network-timeout"
    if "all ai engines failed" in message or "ai call failed" in message:
        return "engine-unavailable"
    if "404" in message or "connection refused" in message or "name or service not known" in message:
        return "network-unavailable"
    if "stow failed" in message or failure_stage in {"stow", "pacs-fetch"}:
        return "delivery"
    return "unexpected-error"


def summarize_failures(series_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    failed_results = [item for item in series_results if item.get("status") == "failed"]
    by_stage: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    for item in failed_results:
        stage = item.get("failure_stage") or "unknown"
        category = item.get("error_category") or "unexpected-error"
        by_stage[stage] = by_stage.get(stage, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1
    return {
        "failed_series_count": len(failed_results),
        "manual_review_required": bool(failed_results),
        "by_stage": by_stage,
        "by_category": by_category,
    }


def sanitize_clinical_text(text: str) -> str:
    sanitized = str(text or "")
    replacements = {
        "confirmed": "suggests",
        "definitive": "apparent",
        "diagnostic": "screening",
        "normal study": "no dominant abnormality flagged on screening",
        "normal examination": "no dominant abnormality flagged on screening",
        "ruled out": "not flagged on screening",
        "rule out": "screen for",
        "rules out": "screens for",
        "definitely": "apparently",
        "certainly": "apparently",
        "proves": "supports",
        "guarantees": "supports",
    }
    for source, target in replacements.items():
        sanitized = sanitized.replace(source, target).replace(source.title(), target.capitalize())
    return sanitized.strip()


def confidence_guardrail_threshold(report: Dict[str, Any]) -> Optional[float]:
    report_type = str(report.get("report_type") or "").strip()
    thresholds = {
        "preliminary-lung-nodule-screening": 0.85,
        "preliminary-echocardiography-lv-function": 0.65,
        "preliminary-screening": 0.75,
        "preliminary-2d-screening": 0.75,
    }
    return thresholds.get(report_type)


def requires_confidence_softening(report: Dict[str, Any]) -> bool:
    threshold = confidence_guardrail_threshold(report)
    if threshold is None:
        return False
    confidence = report.get("confidence")
    try:
        return confidence is not None and float(confidence) < threshold
    except Exception:
        return False


def has_minimum_screening_evidence(report: Dict[str, Any]) -> bool:
    report_type = str(report.get("report_type") or "").strip()
    metrics = report.get("metrics") or {}
    observations = report.get("observations") or []
    abnormalities = report.get("abnormalities") or []
    conclusion = str(report.get("conclusion") or "").strip()

    if report_type == "preliminary-lung-nodule-screening":
        return bool(
            int(metrics.get("candidate_count") or 0) > 0
            or metrics.get("top_boxes_xyzxyz")
            or metrics.get("top_scores")
            or conclusion
        )
    if report_type == "preliminary-stenosis-screening":
        return bool(
            int(metrics.get("frame_count") or 0) > 0
            or int(metrics.get("positive_frames") or 0) > 0
            or float(metrics.get("max_positive_pixel_ratio") or 0.0) > 0.0
            or conclusion
        )
    if report_type == "preliminary-echocardiography-lv-function":
        return bool(
            metrics.get("estimated_ef_percent") is not None
            or metrics.get("fractional_area_change") is not None
            or conclusion
        )
    if report_type == "preliminary-brain-tumor-segmentation":
        return bool(
            int(metrics.get("whole_tumor_voxels") or 0) > 0
            or float(metrics.get("whole_tumor_ratio") or 0.0) > 0.0
            or conclusion
        )
    if report_type in {"preliminary-screening", "preliminary-2d-screening"}:
        return bool(observations or abnormalities or conclusion)
    return True


def evaluate_modality_guardrails(report: Dict[str, Any]) -> Dict[str, Any]:
    report_type = str(report.get("report_type") or "").strip()
    diagnostic_support = str(report.get("diagnostic_support") or "").strip()
    metrics = report.get("metrics") or {}
    observations = report.get("observations") or []
    abnormalities = report.get("abnormalities") or []

    result = {"downgrade": False, "reasons": []}

    if diagnostic_support not in {"screening-only", "anatomy-only", "not-supported"}:
        result["reasons"].append("unknown-diagnostic-support")

    if report_type == "preliminary-lung-nodule-screening":
        if int(metrics.get("candidate_count") or 0) <= 0 and not metrics.get("top_boxes_xyzxyz"):
            result["downgrade"] = True
            result["reasons"].append("ct-missing-candidate-evidence")
    elif report_type == "preliminary-stenosis-screening":
        if int(metrics.get("frame_count") or 0) <= 0 or int(metrics.get("positive_frames") or 0) <= 0:
            result["downgrade"] = True
            result["reasons"].append("xa-missing-frame-burden")
    elif report_type == "preliminary-echocardiography-lv-function":
        if metrics.get("estimated_ef_percent") is None and metrics.get("fractional_area_change") is None:
            result["downgrade"] = True
            result["reasons"].append("us-missing-functional-evidence")
    elif report_type == "preliminary-brain-tumor-segmentation":
        if int(metrics.get("whole_tumor_voxels") or 0) <= 0 and float(metrics.get("whole_tumor_ratio") or 0.0) <= 0.0:
            result["downgrade"] = True
            result["reasons"].append("mr-missing-lesion-burden")
    elif report_type in {"preliminary-screening", "preliminary-2d-screening"}:
        if not observations and not abnormalities and not str(report.get("conclusion") or "").strip():
            result["downgrade"] = True
            result["reasons"].append("xr-us-missing-observation-evidence")
    elif diagnostic_support == "screening-only" and not has_minimum_screening_evidence(report):
        result["downgrade"] = True
        result["reasons"].append("insufficient-structured-evidence")

    return result


def apply_output_guardrails(
    report: Dict[str, Any],
    clinical_summary: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    summary = dict(clinical_summary)
    guardrails = {"applied": False, "reasons": []}
    diagnostic_support = str(report.get("diagnostic_support") or "").strip()
    report_type = str(report.get("report_type") or "").strip()
    modality_guardrails = evaluate_modality_guardrails(report)

    for key in ("findings", "impression", "recommendation"):
        summary[key] = sanitize_clinical_text(summary.get(key) or "")

    if modality_guardrails.get("downgrade"):
        guardrails["applied"] = True
        guardrails["reasons"].extend(modality_guardrails.get("reasons", []))
        summary["findings"] = (
            "Automated screening output did not provide enough structured evidence for a reliable modality-specific summary."
        )
        summary["impression"] = (
            "Automated screening output is non-diagnostic after guardrail review. Formal specialist interpretation is required."
        )
        summary["recommendation"] = "Treat this result as non-diagnostic and escalate for specialist review."
        return summary, guardrails

    if diagnostic_support == "screening-only" and requires_confidence_softening(report):
        guardrails["applied"] = True
        guardrails["reasons"].append("low-confidence-wording-softened")
        summary["findings"] = sanitize_clinical_text(summary.get("findings") or "")
        summary["impression"] = (
            "Low-confidence automated screening pattern detected. Formal specialist interpretation is required for confirmation."
        )
        summary["recommendation"] = (
            "Treat this as low-confidence screening support only and correlate with specialist review."
        )

    if diagnostic_support in {"anatomy-only", "not-supported"} or report_type in {
        "non-diagnostic",
        "non-diagnostic-anatomy",
        "analysis-failed-manual-review-required",
        "manual-review-required",
    }:
        if "Formal" not in summary.get("impression", "") and "required" not in summary.get("impression", ""):
            guardrails["applied"] = True
            guardrails["reasons"].append("forced-manual-review-language")
            summary["impression"] = (
                "Automated AI support is non-diagnostic for this study. Formal specialist interpretation is required."
            )
    return summary, guardrails


def apply_modality_wording_templates(
    report: Dict[str, Any],
    clinical_summary: Dict[str, Any],
) -> Dict[str, Any]:
    summary = dict(clinical_summary)
    report_type = str(report.get("report_type") or "").strip()
    diagnostic_support = str(report.get("diagnostic_support") or "").strip()

    findings = sanitize_clinical_text(summary.get("findings") or "")
    impression = sanitize_clinical_text(summary.get("impression") or "")
    recommendation = sanitize_clinical_text(summary.get("recommendation") or "")

    if report_type == "preliminary-lung-nodule-screening":
        summary["findings"] = (
            findings
            if findings.startswith("Chest CT screening")
            else f"Chest CT screening findings: {findings}"
        )
        summary["impression"] = (
            impression
            if impression.startswith("Screening chest CT") or impression.startswith("No dominant pulmonary nodule")
            else f"Screening chest CT impression: {impression}"
        )
    elif report_type == "preliminary-stenosis-screening":
        summary["findings"] = (
            findings
            if findings.startswith("Angiographic screening")
            else f"Angiographic screening findings: {findings}"
        )
        summary["impression"] = (
            impression
            if impression.startswith("Preliminary angiographic AI screening") or impression.startswith("No dominant focal")
            else f"Preliminary angiographic AI screening impression: {impression}"
        )
    elif report_type == "preliminary-echocardiography-lv-function":
        summary["findings"] = (
            findings
            if findings.startswith("Automated echocardiographic cine analysis")
            else f"Automated echocardiographic screening findings: {findings}"
        )
        summary["impression"] = (
            impression
            if impression.startswith("Automated screening suggests")
            else f"Automated echocardiographic screening impression: {impression}"
        )
    elif report_type == "preliminary-brain-tumor-segmentation":
        summary["findings"] = (
            findings
            if findings.startswith("Brain MRI segmentation screening")
            else f"Brain MRI segmentation screening findings: {findings}"
        )
        summary["impression"] = (
            impression
            if impression.startswith("Screening brain MRI segmentation") or impression.startswith("No dominant candidate tumor")
            else f"Brain MRI segmentation screening impression: {impression}"
        )
    elif report_type in {"non-diagnostic", "non-diagnostic-anatomy", "analysis-failed-manual-review-required", "manual-review-required"} or diagnostic_support in {"anatomy-only", "not-supported"}:
        summary["impression"] = (
            impression
            if "required" in impression.lower()
            else "Automated AI support is non-diagnostic for this study. Formal specialist interpretation is required."
        )

    summary["findings"] = summary.get("findings") or findings
    summary["impression"] = summary.get("impression") or impression
    summary["recommendation"] = recommendation or "Radiologist review required."
    return summary


def derive_result_policy(report: Dict[str, Any]) -> Dict[str, Any]:
    report_type = str(report.get("report_type") or "").strip()
    diagnostic_support = str(report.get("diagnostic_support") or "").strip()

    policy = {
        "can_set_abnormal": True,
        "can_expose_confidence": True,
        "can_claim_specific_structure": True,
        "must_force_manual_review": False,
        "must_hide_confidence": False,
        "must_hide_abnormal": False,
    }

    if diagnostic_support in {"anatomy-only", "not-supported"} or report_type in {
        "non-diagnostic",
        "non-diagnostic-anatomy",
        "analysis-failed-manual-review-required",
        "manual-review-required",
    }:
        policy.update(
            {
                "can_set_abnormal": False,
                "can_expose_confidence": False,
                "must_force_manual_review": True,
                "must_hide_confidence": True,
                "must_hide_abnormal": True,
            }
        )
        return policy

    if report_type == "preliminary-stenosis-screening":
        policy.update(
            {
                "can_expose_confidence": False,
                "must_hide_confidence": True,
            }
        )
    elif report_type in {"preliminary-screening", "preliminary-2d-screening"}:
        policy.update(
            {
                "can_claim_specific_structure": False,
            }
        )
    elif report_type == "preliminary-echocardiography-lv-function":
        policy.update(
            {
                "can_claim_specific_structure": True,
            }
        )

    return policy


def derive_recommendation_policy(
    report: Dict[str, Any],
    result_policy: Dict[str, Any],
    output_guardrails: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report_type = str(report.get("report_type") or "").strip()
    diagnostic_support = str(report.get("diagnostic_support") or "").strip()
    confidence = report.get("confidence")
    guardrail_reasons = set((output_guardrails or {}).get("reasons", []))

    policy = {
        "can_recommend_treatment_change": False,
        "can_recommend_discharge": False,
        "can_recommend_no_follow_up": False,
        "can_recommend_urgent_escalation": False,
        "can_recommend_specialist_review": True,
        "can_recommend_followup_imaging": False,
        "must_include_manual_review": False,
        "force_screening_only_language": False,
        "force_non_diagnostic_language": False,
    }

    if result_policy.get("must_force_manual_review") or diagnostic_support in {"anatomy-only", "not-supported"}:
        policy.update(
            {
                "must_include_manual_review": True,
                "force_non_diagnostic_language": True,
            }
        )
        return policy

    if diagnostic_support == "screening-only":
        policy["force_screening_only_language"] = True
        policy["can_recommend_followup_imaging"] = True

    if report_type in {
        "preliminary-lung-nodule-screening",
        "preliminary-stenosis-screening",
        "preliminary-brain-tumor-segmentation",
        "preliminary-echocardiography-lv-function",
        "preliminary-screening",
        "preliminary-2d-screening",
    }:
        policy["must_include_manual_review"] = True

    if report_type in {
        "preliminary-stenosis-screening",
        "preliminary-brain-tumor-segmentation",
    }:
        policy["can_recommend_urgent_escalation"] = True

    if "low-confidence-wording-softened" in guardrail_reasons:
        policy.update(
            {
                "must_include_manual_review": True,
                "force_screening_only_language": True,
                "can_recommend_urgent_escalation": False,
            }
        )

    threshold = confidence_guardrail_threshold(report)
    if isinstance(confidence, (int, float)) and isinstance(threshold, (int, float)) and confidence < threshold:
        policy["force_screening_only_language"] = True

    return policy


def apply_recommendation_policy(
    report: Dict[str, Any],
    clinical_summary: Dict[str, Any],
    result_policy: Dict[str, Any],
    output_guardrails: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    summary = dict(clinical_summary)
    policy = derive_recommendation_policy(report, result_policy, output_guardrails)
    recommendation = sanitize_clinical_text(summary.get("recommendation") or "")
    lowered = recommendation.lower()

    unsafe_treatment_terms = (
        "start ",
        "stop ",
        "increase ",
        "decrease ",
        "surgery",
        "operate",
        "biopsy",
        "chemotherapy",
        "radiation therapy",
        "thrombolysis",
        "anticoagulation",
        "stent",
        "cabg",
        "discharge",
    )
    unsafe_no_followup_terms = (
        "no follow-up",
        "no further workup",
        "no further action",
        "discharge",
        "reassurance only",
    )
    urgent_terms = ("urgent", "immediate", "emergent", "emergency", "stat")
    recommendation_rewritten = False

    if any(term in lowered for term in unsafe_treatment_terms) and not policy["can_recommend_treatment_change"]:
        recommendation = "Use this result for screening support only and obtain specialist review before any treatment decision."
        recommendation_rewritten = True

    if any(term in lowered for term in unsafe_no_followup_terms) and not policy["can_recommend_no_follow_up"]:
        recommendation = "Do not use this result to stop follow-up. Correlate with formal specialist review."
        recommendation_rewritten = True

    if not recommendation_rewritten and any(term in lowered for term in urgent_terms) and not policy["can_recommend_urgent_escalation"]:
        recommendation = "Prompt specialist review is recommended to interpret this screening result in clinical context."

    if policy["force_non_diagnostic_language"]:
        if str(report.get("diagnostic_support") or "").strip() == "anatomy-only":
            recommendation = "Use this result as anatomy context only. Formal specialist review is required."
        else:
            recommendation = "Use this result as non-diagnostic support only. Formal specialist review is required."
    elif policy["must_include_manual_review"]:
        if "review" not in recommendation.lower() and "interpretation" not in recommendation.lower():
            recommendation = "Formal specialist review is required before acting on this result."
        elif "screening" not in recommendation.lower() and policy["force_screening_only_language"]:
            recommendation = f"{recommendation.rstrip('.')} and treat this as screening support only."
    elif not recommendation:
        recommendation = "Radiologist review required."

    summary["recommendation"] = recommendation
    return summary, policy


def derive_claim_scope_policy(report: Dict[str, Any], result_policy: Dict[str, Any]) -> Dict[str, Any]:
    report_type = str(report.get("report_type") or "").strip()
    diagnostic_support = str(report.get("diagnostic_support") or "").strip()

    policy = {
        "can_claim_pathology": True,
        "can_claim_specific_structure": bool(result_policy.get("can_claim_specific_structure", True)),
        "allowed_claim_family": "general",
        "must_use_non_diagnostic_claims": False,
    }

    if diagnostic_support in {"anatomy-only", "not-supported"} or report_type in {
        "non-diagnostic",
        "non-diagnostic-anatomy",
        "analysis-failed-manual-review-required",
        "manual-review-required",
    }:
        policy.update(
            {
                "can_claim_pathology": False,
                "can_claim_specific_structure": bool(result_policy.get("can_claim_specific_structure", True)),
                "allowed_claim_family": "non-diagnostic",
                "must_use_non_diagnostic_claims": True,
            }
        )
        return policy

    if report_type == "preliminary-lung-nodule-screening":
        policy["allowed_claim_family"] = "pulmonary-nodule"
    elif report_type == "preliminary-stenosis-screening":
        policy["allowed_claim_family"] = "vascular-narrowing"
    elif report_type == "preliminary-echocardiography-lv-function":
        policy["allowed_claim_family"] = "lv-function"
    elif report_type == "preliminary-brain-tumor-segmentation":
        policy["allowed_claim_family"] = "brain-lesion-burden"
    elif report_type in {"preliminary-screening", "preliminary-2d-screening"}:
        policy.update(
            {
                "allowed_claim_family": "generic-screening",
                "can_claim_specific_structure": False,
            }
        )

    return policy


def apply_claim_scope_guardrails(
    report: Dict[str, Any],
    clinical_summary: Dict[str, Any],
    result_policy: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    summary = dict(clinical_summary)
    policy = derive_claim_scope_policy(report, result_policy)
    guardrails = {"applied": False, "reasons": [], "policy": policy}

    findings = sanitize_clinical_text(summary.get("findings") or "")
    impression = sanitize_clinical_text(summary.get("impression") or "")

    if policy["must_use_non_diagnostic_claims"]:
        diagnostic_support = str(report.get("diagnostic_support") or "").strip()
        guardrails["applied"] = True
        guardrails["reasons"].append("claim-scope-non-diagnostic")
        if diagnostic_support == "anatomy-only":
            summary["findings"] = (
                "An anatomy-focused AI model completed structural analysis only. "
                "No pathology-specific diagnostic interpretation was available for this examination."
            )
            summary["impression"] = "Anatomy-only AI support was generated. Formal specialist interpretation is required."
        else:
            summary["findings"] = (
                "Automated AI support could not provide a pathology-specific diagnostic claim for this examination."
            )
            summary["impression"] = "Automated AI support is non-diagnostic for this study. Formal specialist interpretation is required."
        return summary, guardrails

    if not policy["can_claim_specific_structure"]:
        overly_specific_terms = (
            "left lower lobe",
            "right upper lobe",
            "coronary artery",
            "left ventricle",
            "frontal lobe",
            "temporal lobe",
            "basal ganglia",
            "hippocampus",
        )
        findings_lower = findings.lower()
        impression_lower = impression.lower()
        if any(term in findings_lower for term in overly_specific_terms) or any(term in impression_lower for term in overly_specific_terms):
            guardrails["applied"] = True
            guardrails["reasons"].append("claim-scope-specific-structure-softened")
            summary["findings"] = "Automated screening flagged suspicious image patterns, but route-level guardrails limited structure-specific claim wording."
            summary["impression"] = "Automated screening detected a suspicious pattern. Formal specialist interpretation is required for structure-specific localization."
            return summary, guardrails

    return summary, guardrails


def band_confidence_value(confidence: Optional[float]) -> Optional[str]:
    if not isinstance(confidence, (int, float)):
        return None
    if confidence < 0.6:
        return "low"
    if confidence < 0.8:
        return "moderate"
    return "high"


def derive_confidence_disclosure_policy(
    report: Dict[str, Any],
    result_policy: Dict[str, Any],
    output_guardrails: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report_type = str(report.get("report_type") or "").strip()
    diagnostic_support = str(report.get("diagnostic_support") or "").strip()
    guardrail_reasons = set((output_guardrails or {}).get("reasons", []))
    minimum_threshold = confidence_guardrail_threshold(report)

    policy = {
        "expose_mode": "rounded_numeric",
        "can_expose_numeric": True,
        "can_expose_qualitative": True,
        "numeric_precision": 2,
        "minimum_confidence_to_expose": minimum_threshold,
        "band_thresholds": {"low": 0.6, "moderate": 0.8},
    }

    if result_policy.get("must_hide_confidence") or diagnostic_support in {"anatomy-only", "not-supported"} or report_type in {
        "non-diagnostic",
        "non-diagnostic-anatomy",
        "analysis-failed-manual-review-required",
        "manual-review-required",
    }:
        policy.update(
            {
                "expose_mode": "hidden",
                "can_expose_numeric": False,
                "can_expose_qualitative": False,
            }
        )
        return policy

    if report_type in {
        "preliminary-lung-nodule-screening",
        "preliminary-brain-tumor-segmentation",
        "preliminary-screening",
        "preliminary-2d-screening",
    }:
        policy.update(
            {
                "expose_mode": "qualitative_band",
                "can_expose_numeric": False,
                "can_expose_qualitative": True,
            }
        )
    elif report_type == "preliminary-echocardiography-lv-function":
        policy.update(
            {
                "expose_mode": "rounded_numeric",
                "can_expose_numeric": True,
                "can_expose_qualitative": True,
                "numeric_precision": 2,
            }
        )

    if "low-confidence-wording-softened" in guardrail_reasons and policy["expose_mode"] != "hidden":
        policy.update(
            {
                "expose_mode": "qualitative_band",
                "can_expose_numeric": False,
                "can_expose_qualitative": True,
            }
        )

    return policy


def normalize_confidence_disclosure(
    report: Dict[str, Any],
    result_policy: Dict[str, Any],
    output_guardrails: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Optional[str], Dict[str, Any]]:
    policy = derive_confidence_disclosure_policy(report, result_policy, output_guardrails)
    raw_confidence = report.get("confidence")
    try:
        confidence_value = float(raw_confidence) if raw_confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None

    if policy["expose_mode"] == "hidden" or confidence_value is None:
        return None, None, policy

    minimum_threshold = policy.get("minimum_confidence_to_expose")
    if isinstance(minimum_threshold, (int, float)) and confidence_value < minimum_threshold:
        if policy["can_expose_qualitative"]:
            return None, band_confidence_value(confidence_value), policy
        return None, None, policy

    if policy["expose_mode"] == "rounded_numeric" and policy["can_expose_numeric"]:
        precision = int(policy.get("numeric_precision") or 2)
        return round(confidence_value, precision), band_confidence_value(confidence_value), policy

    if policy["expose_mode"] == "qualitative_band" and policy["can_expose_qualitative"]:
        return None, band_confidence_value(confidence_value), policy

    return None, None, policy


def derive_limitation_policy(
    report: Dict[str, Any],
    result_policy: Dict[str, Any],
    output_guardrails: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report_type = str(report.get("report_type") or "").strip()
    diagnostic_support = str(report.get("diagnostic_support") or "").strip()
    guardrail_reasons = set((output_guardrails or {}).get("reasons", []))

    required_limitations: List[str] = []

    if report_type == "preliminary-lung-nodule-screening":
        required_limitations.extend(
            [
                "This route provides preliminary chest CT nodule screening support only and is not a final diagnosis.",
                "Pulmonary nodule candidates require radiologist confirmation on the full examination.",
            ]
        )
    elif report_type == "preliminary-stenosis-screening":
        required_limitations.extend(
            [
                "This route provides preliminary angiographic stenosis screening support only and is not a final diagnosis.",
                "Angiographic narrowing patterns require formal interventional or radiology interpretation.",
            ]
        )
    elif report_type == "preliminary-echocardiography-lv-function":
        required_limitations.extend(
            [
                "This route provides preliminary echocardiographic screening support only and is not a final cardiology interpretation.",
                "Estimated ventricular function should be correlated with the full echocardiographic examination.",
            ]
        )
    elif report_type == "preliminary-brain-tumor-segmentation":
        required_limitations.extend(
            [
                "This route provides preliminary brain MRI segmentation screening support only and is not a final diagnosis.",
                "Segmentation-derived lesion burden requires neuroradiology correlation on the full examination.",
            ]
        )

    if diagnostic_support == "anatomy-only" or report_type == "non-diagnostic-anatomy":
        required_limitations.extend(
            [
                "This route provides anatomy-focused AI support only and does not provide pathology-specific diagnosis.",
                "Formal specialist interpretation is required for pathology assessment.",
            ]
        )

    if diagnostic_support == "not-supported" or report_type in {
        "non-diagnostic",
        "analysis-failed-manual-review-required",
        "manual-review-required",
    }:
        required_limitations.extend(
            [
                "Automated analysis did not produce a diagnostic-grade study result for this examination.",
                "Manual specialist review is required before clinical use.",
            ]
        )

    if "low-confidence-wording-softened" in guardrail_reasons:
        required_limitations.append(
            "Low-confidence screening output was softened by guardrails and should not be used without specialist confirmation."
        )

    if any(
        reason in guardrail_reasons
        for reason in {
            "insufficient-structured-evidence",
            "ct-missing-candidate-evidence",
            "xa-missing-frame-burden",
            "us-missing-functional-evidence",
            "mr-missing-lesion-burden",
            "xr-us-missing-observation-evidence",
        }
    ):
        required_limitations.append(
            "Structured evidence was insufficient for a reliable modality-specific summary, so the result was downgraded."
        )

    return {
        "must_include_route_limitations": True,
        "must_include_manual_review_limitations": result_policy.get("must_force_manual_review", False),
        "required_limitations": required_limitations,
    }


def apply_limitation_policy(
    report: Dict[str, Any],
    result_policy: Dict[str, Any],
    output_guardrails: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    policy = derive_limitation_policy(report, result_policy, output_guardrails)
    existing_limitations = [str(item).strip() for item in (report.get("limitations") or []) if str(item).strip()]
    final_limitations: List[str] = []

    for limitation in existing_limitations + policy.get("required_limitations", []):
        if limitation and limitation not in final_limitations:
            final_limitations.append(limitation)

    return final_limitations, policy


def build_final_ai_result_summary(
    study_uid: Optional[str],
    status: str,
    ai_result: Dict[str, Any],
    dicom_metadata: Dict[str, Any],
    supporting_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    metadata_summary = (dicom_metadata or {}).get("metadata_summary") or {}
    series_summary = metadata_summary.get("series") or {}
    exam = ai_result.get("exam") or "Unknown exam"
    modality = series_summary.get("modality") or (metadata_summary.get("study") or {}).get("modality") or "unknown"
    body_part = series_summary.get("body_part_examined") or "unknown"
    primary_series = (
        series_summary.get("series_description")
        or series_summary.get("series_instance_uid")
        or "unknown"
    )

    lines = [
        f"Exam: {exam}",
        f"Study UID: {study_uid or 'unknown'}",
        f"Status: {status}",
        f"Modalities: {modality}",
        f"Body parts: {body_part}",
        f"Primary analyzed series: {primary_series}",
    ]

    if ai_result.get("technique"):
        lines.append(f"Technique: {ai_result['technique']}")
    if ai_result.get("findings"):
        lines.append(f"Findings: {ai_result['findings']}")
    if ai_result.get("impression"):
        lines.append(f"Impression: {ai_result['impression']}")
    if ai_result.get("recommendation"):
        lines.append(f"Recommendation: {ai_result['recommendation']}")

    limitations = ai_result.get("limitations") or []
    if limitations:
        lines.append("Limitations: " + " ".join(limitations[:2]))

    supporting_sequence_labels: List[str] = []
    for result in supporting_results or []:
        metadata = result.get("metadata_summary") or {}
        series = metadata.get("series") or {}
        label = series.get("series_description") or series.get("series_instance_uid")
        if label and label not in supporting_sequence_labels:
            supporting_sequence_labels.append(label)
    if supporting_sequence_labels:
        lines.append("Supporting sequences: " + ", ".join(supporting_sequence_labels[:6]))

    return "\n".join(lines)


def apply_summary_consistency_guardrails(
    ai_result: Dict[str, Any],
    dicom_metadata: Dict[str, Any],
    status: str,
    study_uid: Optional[str],
    supporting_results: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    result = dict(ai_result)
    guardrails = {"applied": False, "reasons": []}

    if result.get("impression"):
        result["finding"] = result["impression"]
        guardrails["applied"] = True
        guardrails["reasons"].append("finding-aligned-to-impression")
    elif result.get("findings"):
        result["finding"] = result["findings"]
        guardrails["applied"] = True
        guardrails["reasons"].append("finding-aligned-to-findings")

    result["summary"] = build_final_ai_result_summary(
        study_uid=study_uid,
        status=status,
        ai_result=result,
        dicom_metadata=dicom_metadata,
        supporting_results=supporting_results,
    )
    guardrails["applied"] = True
    guardrails["reasons"].append("summary-rebuilt-from-final-fields")

    return result, guardrails


def derive_model_chain_contract(
    report: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report_type = str(report.get("report_type") or "").strip()
    diagnostic_support = str(report.get("diagnostic_support") or "").strip()
    metrics = report.get("metrics") or {}
    observations = report.get("observations") or []
    metadata = metadata or {}

    contract = {
        "chain_ready": False,
        "chain_stage": "single-stage",
        "next_stage": None,
        "required_artifacts": [],
        "available_artifacts": [],
        "missing_artifacts": [],
        "structured_evidence": {},
    }

    if diagnostic_support in {"anatomy-only", "not-supported"} or report_type in {
        "non-diagnostic",
        "non-diagnostic-anatomy",
        "analysis-failed-manual-review-required",
        "manual-review-required",
    }:
        contract["chain_stage"] = "non-diagnostic"
        return contract

    if report_type == "preliminary-lung-nodule-screening":
        candidate_regions = describe_ct_chest_candidate_regions(metrics, metadata)
        candidate_locations = report.get("candidate_locations") or report.get("candidate_regions") or []
        preferred_targets = candidate_locations or candidate_regions
        contract.update(
            {
                "chain_stage": "detector-output",
                "next_stage": "candidate-triage",
                "required_artifacts": ["candidate_count", "candidate_boxes_or_regions"],
                "available_artifacts": [
                    artifact
                    for artifact, present in {
                        "candidate_count": int(metrics.get("candidate_count") or 0) > 0,
                        "candidate_boxes": bool(metrics.get("top_boxes_xyzxyz")),
                        "candidate_scores": bool(metrics.get("top_scores")) or metrics.get("top_score") is not None,
                        "candidate_regions": bool(preferred_targets),
                    }.items()
                    if present
                ],
                "structured_evidence": {
                    "candidate_count": int(metrics.get("candidate_count") or 0),
                    "candidate_regions": preferred_targets,
                    "top_score": metrics.get("top_score"),
                },
            }
        )
        contract["chain_ready"] = bool(
            int(metrics.get("candidate_count") or 0) > 0
            and (metrics.get("top_boxes_xyzxyz") or preferred_targets)
        )
    elif report_type == "preliminary-stenosis-screening":
        contract.update(
            {
                "chain_stage": "frame-detector-output",
                "next_stage": "run-summarizer",
                "required_artifacts": ["frame_count", "positive_frames"],
                "available_artifacts": [
                    artifact
                    for artifact, present in {
                        "frame_count": int(metrics.get("frame_count") or 0) > 0,
                        "positive_frames": int(metrics.get("positive_frames") or 0) > 0,
                        "max_positive_pixel_ratio": float(metrics.get("max_positive_pixel_ratio") or 0.0) > 0.0,
                    }.items()
                    if present
                ],
                "structured_evidence": {
                    "frame_count": int(metrics.get("frame_count") or 0),
                    "positive_frames": int(metrics.get("positive_frames") or 0),
                    "max_positive_pixel_ratio": float(metrics.get("max_positive_pixel_ratio") or 0.0),
                },
            }
        )
        contract["chain_ready"] = (
            int(metrics.get("frame_count") or 0) > 0 and int(metrics.get("positive_frames") or 0) > 0
        )
    elif report_type == "preliminary-echocardiography-lv-function":
        contract.update(
            {
                "chain_stage": "functional-metrics",
                "next_stage": "cardiac-summary-classifier",
                "required_artifacts": ["estimated_ef_percent_or_fractional_area_change"],
                "available_artifacts": [
                    artifact
                    for artifact, present in {
                        "estimated_ef_percent": metrics.get("estimated_ef_percent") is not None,
                        "fractional_area_change": metrics.get("fractional_area_change") is not None,
                        "frame_time_ms": metrics.get("frame_time_ms") is not None,
                    }.items()
                    if present
                ],
                "structured_evidence": {
                    "estimated_ef_percent": metrics.get("estimated_ef_percent"),
                    "fractional_area_change": metrics.get("fractional_area_change"),
                    "frame_time_ms": metrics.get("frame_time_ms"),
                },
            }
        )
        contract["chain_ready"] = (
            metrics.get("estimated_ef_percent") is not None or metrics.get("fractional_area_change") is not None
        )
    elif report_type == "preliminary-brain-tumor-segmentation":
        contract.update(
            {
                "chain_stage": "segmentation-metrics",
                "next_stage": "lesion-burden-interpreter",
                "required_artifacts": ["whole_tumor_voxels_or_ratio"],
                "available_artifacts": [
                    artifact
                    for artifact, present in {
                        "whole_tumor_voxels": int(metrics.get("whole_tumor_voxels") or 0) > 0,
                        "whole_tumor_ratio": float(metrics.get("whole_tumor_ratio") or 0.0) > 0.0,
                        "tumor_core_voxels": int(metrics.get("tumor_core_voxels") or 0) > 0,
                        "enhancing_tumor_voxels": int(metrics.get("enhancing_tumor_voxels") or 0) > 0,
                    }.items()
                    if present
                ],
                "structured_evidence": {
                    "whole_tumor_voxels": int(metrics.get("whole_tumor_voxels") or 0),
                    "whole_tumor_ratio": float(metrics.get("whole_tumor_ratio") or 0.0),
                    "tumor_core_voxels": int(metrics.get("tumor_core_voxels") or 0),
                    "enhancing_tumor_voxels": int(metrics.get("enhancing_tumor_voxels") or 0),
                },
            }
        )
        contract["chain_ready"] = (
            int(metrics.get("whole_tumor_voxels") or 0) > 0 or float(metrics.get("whole_tumor_ratio") or 0.0) > 0.0
        )
    elif report_type in {"preliminary-screening", "preliminary-2d-screening"}:
        contract.update(
            {
                "chain_stage": "screening-observation",
                "next_stage": "rule-based-triage",
                "required_artifacts": ["observations_or_abnormalities"],
                "available_artifacts": [
                    artifact
                    for artifact, present in {
                        "observations": bool(observations),
                        "abnormalities": bool(report.get("abnormalities") or []),
                        "conclusion": bool(str(report.get("conclusion") or "").strip()),
                    }.items()
                    if present
                ],
                "structured_evidence": {
                    "observations_count": len(observations) if isinstance(observations, list) else 0,
                    "abnormalities_count": len(report.get("abnormalities") or []),
                    "has_conclusion": bool(str(report.get("conclusion") or "").strip()),
                },
            }
        )
        contract["chain_ready"] = bool(observations or report.get("abnormalities") or str(report.get("conclusion") or "").strip())

    for artifact in contract["required_artifacts"]:
        if artifact not in contract["available_artifacts"] and not any(
            artifact.startswith(prefix) and any(item.startswith(prefix) for item in contract["available_artifacts"])
            for prefix in ("candidate_", "estimated_", "whole_tumor_", "observations_", "frame_")
        ):
            contract["missing_artifacts"].append(artifact)

    return contract


def normalize_chain_evidence(
    report: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    report_type = str(report.get("report_type") or "").strip()
    diagnostic_support = str(report.get("diagnostic_support") or "").strip()
    metrics = report.get("metrics") or {}
    contract = derive_model_chain_contract(report, metadata)

    bundle = {
        "route_name": (report.get("routing_decision") or {}).get("route_name"),
        "report_type": report_type,
        "diagnostic_support": diagnostic_support,
        "chain_ready": contract.get("chain_ready", False),
        "next_stage": contract.get("next_stage"),
        "evidence_type": "none",
        "targets": [],
        "measurements": {},
        "localization": {},
        "confidence_context": {
            "raw_confidence": report.get("confidence"),
            "top_score": metrics.get("top_score"),
        },
        "source_artifacts": contract.get("available_artifacts", []),
    }

    if report_type == "preliminary-lung-nodule-screening":
        candidate_regions = describe_ct_chest_candidate_regions(metrics, metadata)
        candidate_locations = report.get("candidate_locations") or report.get("candidate_regions") or []
        bundle.update(
            {
                "evidence_type": "candidate-detection",
                "targets": candidate_locations or candidate_regions,
                "measurements": {
                    "candidate_count": int(metrics.get("candidate_count") or 0),
                    "top_score": metrics.get("top_score"),
                    "top_scores": metrics.get("top_scores") or [],
                },
                "localization": {
                    "boxes_xyzxyz": metrics.get("top_boxes_xyzxyz") or [],
                },
            }
        )
    elif report_type == "preliminary-stenosis-screening":
        bundle.update(
            {
                "evidence_type": "frame-burden",
                "measurements": {
                    "frame_count": int(metrics.get("frame_count") or 0),
                    "positive_frames": int(metrics.get("positive_frames") or 0),
                    "max_positive_pixel_ratio": float(metrics.get("max_positive_pixel_ratio") or 0.0),
                },
            }
        )
    elif report_type == "preliminary-echocardiography-lv-function":
        bundle.update(
            {
                "evidence_type": "functional-metrics",
                "measurements": {
                    "estimated_ef_percent": metrics.get("estimated_ef_percent"),
                    "fractional_area_change": metrics.get("fractional_area_change"),
                    "frame_time_ms": metrics.get("frame_time_ms"),
                },
            }
        )
    elif report_type == "preliminary-brain-tumor-segmentation":
        bundle.update(
            {
                "evidence_type": "segmentation-burden",
                "measurements": {
                    "whole_tumor_voxels": int(metrics.get("whole_tumor_voxels") or 0),
                    "whole_tumor_ratio": float(metrics.get("whole_tumor_ratio") or 0.0),
                    "tumor_core_voxels": int(metrics.get("tumor_core_voxels") or 0),
                    "enhancing_tumor_voxels": int(metrics.get("enhancing_tumor_voxels") or 0),
                },
            }
        )
    elif report_type in {"preliminary-screening", "preliminary-2d-screening"}:
        bundle.update(
            {
                "evidence_type": "observation-screening",
                "targets": report.get("abnormalities") or [],
                "measurements": {
                    "observations_count": len(report.get("observations") or []),
                    "abnormalities_count": len(report.get("abnormalities") or []),
                },
            }
        )

    return bundle


def derive_second_stage_selection_policy(
    report: Dict[str, Any],
    result_policy: Dict[str, Any],
    output_guardrails: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    contract = derive_model_chain_contract(report, metadata)
    evidence = normalize_chain_evidence(report, metadata)
    diagnostic_support = str(report.get("diagnostic_support") or "").strip()
    report_type = str(report.get("report_type") or "").strip()
    guardrail_reasons = list((output_guardrails or {}).get("reasons", []))

    policy = {
        "should_invoke": False,
        "decision": "blocked",
        "next_stage": contract.get("next_stage"),
        "reason": "chain-not-ready",
        "blocking_reasons": [],
        "required_preconditions": list(contract.get("required_artifacts", [])),
    }

    if diagnostic_support in {"anatomy-only", "not-supported"} or report_type in {
        "non-diagnostic",
        "non-diagnostic-anatomy",
        "analysis-failed-manual-review-required",
        "manual-review-required",
    }:
        policy.update(
            {
                "decision": "blocked",
                "reason": "non-diagnostic-route",
                "blocking_reasons": ["non-diagnostic-route"],
            }
        )
        return policy

    if result_policy.get("must_force_manual_review"):
        policy.update(
            {
                "decision": "blocked",
                "reason": "manual-review-required",
                "blocking_reasons": ["manual-review-required"],
            }
        )
        return policy

    if guardrail_reasons:
        blocking_reasons = [
            reason
            for reason in guardrail_reasons
            if reason
            in {
                "insufficient-structured-evidence",
                "ct-missing-candidate-evidence",
                "xa-missing-frame-burden",
                "us-missing-functional-evidence",
                "mr-missing-lesion-burden",
                "xr-us-missing-observation-evidence",
            }
        ]
        if blocking_reasons:
            policy.update(
                {
                    "decision": "blocked",
                    "reason": "guardrail-blocked",
                    "blocking_reasons": blocking_reasons,
                }
            )
            return policy

    if not contract.get("chain_ready"):
        policy.update(
            {
                "decision": "deferred",
                "reason": "missing-chain-artifacts",
                "blocking_reasons": list(contract.get("missing_artifacts", [])),
            }
        )
        return policy

    evidence_type = evidence.get("evidence_type")
    if evidence_type == "none":
        policy.update(
            {
                "decision": "deferred",
                "reason": "no-normalized-evidence",
                "blocking_reasons": ["no-normalized-evidence"],
            }
        )
        return policy

    policy.update(
        {
            "should_invoke": True,
            "decision": "invoke",
            "reason": "chain-ready",
            "blocking_reasons": [],
        }
    )
    return policy


def build_second_stage_input_payload(
    study_uid: Optional[str],
    report: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    result_policy: Optional[Dict[str, Any]] = None,
    output_guardrails: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    result_policy = result_policy or derive_result_policy(report)
    normalized_evidence = normalize_chain_evidence(report, metadata)
    chain_contract = derive_model_chain_contract(report, metadata)
    selection_policy = derive_second_stage_selection_policy(
        report,
        result_policy,
        output_guardrails,
        metadata,
    )

    metadata_summary = build_metadata_summary(metadata) if metadata else {}

    return {
        "study_uid": study_uid,
        "series_uid": metadata.get("SeriesInstanceUID"),
        "modality": metadata.get("Modality"),
        "body_part": metadata.get("BodyPartExamined"),
        "route_name": (report.get("routing_decision") or {}).get("route_name"),
        "source_model_name": report.get("model_name") or report.get("model"),
        "report_type": report.get("report_type"),
        "diagnostic_support": report.get("diagnostic_support"),
        "chain_invocation": {
            "should_invoke": selection_policy.get("should_invoke", False),
            "decision": selection_policy.get("decision"),
            "next_stage": selection_policy.get("next_stage"),
            "reason": selection_policy.get("reason"),
        },
        "chain_contract": chain_contract,
        "normalized_evidence": normalized_evidence,
        "result_policy": result_policy,
        "output_guardrails": output_guardrails or {"applied": False, "reasons": []},
        "metadata_summary": metadata_summary,
    }


def derive_second_stage_merge_policy(
    ai_result: Dict[str, Any],
    second_stage_selection_policy: Dict[str, Any],
) -> Dict[str, Any]:
    report_type = str(ai_result.get("report_type") or "").strip()
    diagnostic_support = str(ai_result.get("diagnostic_support") or "").strip()
    decision = str(second_stage_selection_policy.get("decision") or "").strip()

    policy = {
        "can_merge": False,
        "merge_mode": "blocked",
        "allowed_fields": [],
        "blocked_fields": ["report_type", "diagnostic_support", "result_policy"],
        "require_consistency_rebuild": True,
        "reason": "merge-blocked",
    }

    if decision != "invoke":
        policy["reason"] = "second-stage-not-invoked"
        return policy

    if diagnostic_support in {"anatomy-only", "not-supported"} or report_type in {
        "non-diagnostic",
        "non-diagnostic-anatomy",
        "analysis-failed-manual-review-required",
        "manual-review-required",
    }:
        policy["reason"] = "non-diagnostic-base-result"
        return policy

    policy.update(
        {
            "can_merge": True,
            "merge_mode": "augment",
            "allowed_fields": [
                "finding",
                "findings",
                "impression",
                "recommendation",
                "limitations",
                "confidence_band",
                "normalized_chain_evidence",
            ],
            "reason": "merge-allowed",
        }
    )
    return policy


def merge_second_stage_result(
    ai_result: Dict[str, Any],
    second_stage_result: Dict[str, Any],
    merge_policy: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    merged = dict(ai_result)
    merge_summary = {
        "applied": False,
        "mode": merge_policy.get("merge_mode", "blocked"),
        "merged_fields": [],
        "blocked_fields": list(merge_policy.get("blocked_fields", [])),
        "reason": merge_policy.get("reason", "merge-blocked"),
    }

    if not merge_policy.get("can_merge"):
        return merged, merge_summary

    allowed_fields = set(merge_policy.get("allowed_fields", []))

    for field in ("finding", "findings", "impression", "recommendation", "confidence_band"):
        value = second_stage_result.get(field)
        if field in allowed_fields and value not in (None, "", []):
            merged[field] = value
            merge_summary["merged_fields"].append(field)

    if "limitations" in allowed_fields:
        merged_limitations = list(merged.get("limitations") or [])
        for limitation in second_stage_result.get("limitations") or []:
            if limitation and limitation not in merged_limitations:
                merged_limitations.append(limitation)
        merged["limitations"] = merged_limitations
        if second_stage_result.get("limitations"):
            merge_summary["merged_fields"].append("limitations")

    if "normalized_chain_evidence" in allowed_fields and second_stage_result.get("normalized_chain_evidence"):
        merged["normalized_chain_evidence"] = second_stage_result["normalized_chain_evidence"]
        merge_summary["merged_fields"].append("normalized_chain_evidence")

    merge_summary["applied"] = bool(merge_summary["merged_fields"])
    return merged, merge_summary


def build_chain_observability(
    ai_result: Dict[str, Any],
    primary_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    selection_policy = ai_result.get("second_stage_selection_policy") or {}
    execution = ai_result.get("second_stage_execution") or {}
    merge_summary = ai_result.get("second_stage_merge_summary") or {}
    chain_contract = ai_result.get("model_chain_contract") or {}
    normalized_evidence = ai_result.get("normalized_chain_evidence") or {}
    output_guardrails = ai_result.get("output_guardrails") or {}

    source_model = None
    if primary_result:
        source_model = primary_result.get("model_id") or primary_result.get("model_name")
    if not source_model:
        source_model = ai_result.get("model_name")

    return {
        "first_stage_model": source_model,
        "first_stage_report_type": ai_result.get("report_type"),
        "first_stage_diagnostic_support": ai_result.get("diagnostic_support"),
        "chain_ready": chain_contract.get("chain_ready", False),
        "requested_second_stage": selection_policy.get("next_stage"),
        "selection_decision": selection_policy.get("decision"),
        "selection_reason": selection_policy.get("reason"),
        "selection_blocking_reasons": selection_policy.get("blocking_reasons", []),
        "second_stage_invoked": execution.get("invoked", False),
        "second_stage_status": execution.get("status"),
        "second_stage_name": execution.get("stage_name"),
        "second_stage_model": (execution.get("result") or {}).get("stage_model_name"),
        "merge_applied": merge_summary.get("applied", False),
        "merge_mode": merge_summary.get("mode"),
        "merge_reason": merge_summary.get("reason"),
        "merged_fields": merge_summary.get("merged_fields", []),
        "evidence_type": normalized_evidence.get("evidence_type"),
        "guardrail_reasons": output_guardrails.get("reasons", []),
    }


def record_chain_metrics(ai_result: Dict[str, Any]) -> None:
    routing_decision = ai_result.get("routing_decision") or {}
    route_name = str(routing_decision.get("route_name") or "unknown-route")

    selection = ai_result.get("second_stage_selection_policy") or {}
    next_stage = str(selection.get("next_stage") or "none")
    decision = str(selection.get("decision") or "unknown")
    CHAIN_SELECTION_TOTAL.labels(route_name, next_stage, decision).inc()

    for reason in selection.get("blocking_reasons") or []:
        CHAIN_BLOCK_REASON_TOTAL.labels(route_name, str(reason)).inc()

    execution = ai_result.get("second_stage_execution") or {}
    stage_name = str(execution.get("stage_name") or next_stage or "none")
    status = str(execution.get("status") or "unknown")
    CHAIN_EXECUTION_TOTAL.labels(route_name, stage_name, status).inc()

    merge_summary = ai_result.get("second_stage_merge_summary") or {}
    merge_mode = str(merge_summary.get("mode") or "unknown")
    applied = "true" if merge_summary.get("applied") else "false"
    CHAIN_MERGE_TOTAL.labels(route_name, merge_mode, applied).inc()


def run_ct_candidate_triage_second_stage(second_stage_input_payload: Dict[str, Any]) -> Dict[str, Any]:
    evidence = second_stage_input_payload.get("normalized_evidence") or {}
    measurements = evidence.get("measurements") or {}
    confidence_context = evidence.get("confidence_context") or {}
    candidate_count = int(measurements.get("candidate_count") or 0)
    top_score = measurements.get("top_score")
    if top_score is None:
        top_score = confidence_context.get("top_score")
    if top_score is None:
        top_score = confidence_context.get("raw_confidence")
    targets = evidence.get("targets") or []
    target_text = ", ".join(targets[:2]) if targets else "the flagged lung region"

    if candidate_count <= 1:
        burden = "limited"
    elif candidate_count <= 3:
        burden = "focal"
    else:
        burden = "multifocal"

    confidence_band = band_confidence_value(top_score if isinstance(top_score, (int, float)) else None) or "indeterminate"
    findings = (
        f"Second-stage CT triage classified the candidate burden as {burden}, with the dominant screening target in {target_text}."
    )
    impression = (
        f"Second-stage CT triage supports a {burden} pulmonary nodule candidate pattern. "
        "Radiologist confirmation on the full examination remains required."
    )
    recommendation = (
        "Use the second-stage triage output for prioritization only, treat this as screening support only, and correlate with formal radiologist review."
    )

    refined_evidence = dict(evidence)
    refined_evidence["triage_summary"] = {
        "candidate_burden": burden,
        "dominant_targets": targets[:2],
        "confidence_band": confidence_band,
    }

    return {
        "finding": impression,
        "findings": findings,
        "impression": impression,
        "recommendation": recommendation,
        "confidence_band": confidence_band,
        "limitations": [
            "Second-stage CT triage refines screening output only and is not a substitute for radiologist diagnosis."
        ],
        "normalized_chain_evidence": refined_evidence,
        "stage_name": "candidate-triage",
        "stage_status": "completed",
        "stage_model_name": "ct-candidate-triage-rule-engine",
    }


def run_xa_run_summarizer_second_stage(second_stage_input_payload: Dict[str, Any]) -> Dict[str, Any]:
    evidence = second_stage_input_payload.get("normalized_evidence") or {}
    measurements = evidence.get("measurements") or {}
    frame_count = int(measurements.get("frame_count") or 0)
    positive_frames = int(measurements.get("positive_frames") or 0)
    max_ratio = float(measurements.get("max_positive_pixel_ratio") or 0.0)

    if frame_count <= 0 or positive_frames <= 0:
        burden = "indeterminate"
    else:
        fraction = positive_frames / float(frame_count)
        if fraction >= 0.75 or max_ratio >= 0.03:
            burden = "high-burden"
        elif fraction >= 0.4 or max_ratio >= 0.01:
            burden = "moderate-burden"
        else:
            burden = "limited-burden"

    findings = (
        f"Second-stage XA run summarization classified the angiographic narrowing pattern as {burden}, "
        f"with suspicious frames in {positive_frames} of {frame_count} reviewed frame(s)."
    )
    impression = (
        f"Second-stage XA run summarization supports a {burden} narrowing pattern on screening review. "
        "Formal interventional or radiology interpretation remains required."
    )
    recommendation = (
        "Use this second-stage XA run summary for prioritization only and correlate with formal angiographic interpretation."
    )

    refined_evidence = dict(evidence)
    refined_evidence["triage_summary"] = {
        "run_burden": burden,
        "positive_frame_fraction": (positive_frames / float(frame_count)) if frame_count > 0 else None,
        "max_positive_pixel_ratio": max_ratio,
    }

    return {
        "finding": impression,
        "findings": findings,
        "impression": impression,
        "recommendation": recommendation,
        "limitations": [
            "Second-stage XA run summarization refines screening output only and is not a substitute for formal angiographic interpretation."
        ],
        "normalized_chain_evidence": refined_evidence,
        "stage_name": "run-summarizer",
        "stage_status": "completed",
        "stage_model_name": "xa-run-summarizer-rule-engine",
    }


def run_us_cardiac_summary_second_stage(second_stage_input_payload: Dict[str, Any]) -> Dict[str, Any]:
    evidence = second_stage_input_payload.get("normalized_evidence") or {}
    measurements = evidence.get("measurements") or {}
    ef_percent = measurements.get("estimated_ef_percent")
    fac = measurements.get("fractional_area_change")

    numeric_value = None
    metric_name = None
    if isinstance(ef_percent, (int, float)):
        numeric_value = float(ef_percent)
        metric_name = "ef"
    elif isinstance(fac, (int, float)):
        numeric_value = float(fac)
        metric_name = "fac"

    if metric_name == "ef":
        if numeric_value >= 50:
            function_class = "preserved systolic function"
        elif numeric_value >= 40:
            function_class = "mildly reduced systolic function"
        elif numeric_value >= 30:
            function_class = "moderately reduced systolic function"
        else:
            function_class = "severely reduced systolic function"
        metric_phrase = f"estimated EF of {numeric_value:.1f}%"
    elif metric_name == "fac":
        if numeric_value >= 35:
            function_class = "preserved systolic function"
        elif numeric_value >= 25:
            function_class = "mildly reduced systolic function"
        elif numeric_value >= 18:
            function_class = "moderately reduced systolic function"
        else:
            function_class = "severely reduced systolic function"
        metric_phrase = f"fractional area change of {numeric_value:.1f}%"
    else:
        function_class = "indeterminate systolic function"
        metric_phrase = "limited quantitative metrics"

    findings = (
        f"Second-stage cardiac summary classified the screening metrics as {function_class}, based on {metric_phrase}."
    )
    impression = (
        f"Second-stage cardiac summary supports {function_class} on screening review. "
        "Formal cardiology interpretation remains required."
    )
    recommendation = (
        "Use this second-stage cardiac summary for prioritization only, treat this as screening support only, and correlate with formal cardiology review."
    )

    refined_evidence = dict(evidence)
    refined_evidence["triage_summary"] = {
        "function_class": function_class,
        "metric_name": metric_name,
        "metric_value": numeric_value,
    }

    return {
        "finding": impression,
        "findings": findings,
        "impression": impression,
        "recommendation": recommendation,
        "limitations": [
            "Second-stage cardiac summary refines screening output only and is not a substitute for formal echocardiographic interpretation."
        ],
        "normalized_chain_evidence": refined_evidence,
        "stage_name": "cardiac-summary-classifier",
        "stage_status": "completed",
        "stage_model_name": "us-cardiac-summary-rule-engine",
    }


def run_mr_lesion_burden_second_stage(second_stage_input_payload: Dict[str, Any]) -> Dict[str, Any]:
    evidence = second_stage_input_payload.get("normalized_evidence") or {}
    measurements = evidence.get("measurements") or {}
    whole_ratio = float(measurements.get("whole_tumor_ratio") or 0.0)
    whole_voxels = int(measurements.get("whole_tumor_voxels") or 0)
    core_voxels = int(measurements.get("tumor_core_voxels") or 0)
    enhancing_voxels = int(measurements.get("enhancing_tumor_voxels") or 0)

    if whole_ratio >= 0.08 or whole_voxels >= 120000:
        burden = "substantial lesion burden"
    elif whole_ratio >= 0.03 or whole_voxels >= 30000:
        burden = "moderate lesion burden"
    else:
        burden = "limited lesion burden"

    component_notes: List[str] = []
    if core_voxels > 0:
        component_notes.append("core component present")
    if enhancing_voxels > 0:
        component_notes.append("enhancing component present")
    if not component_notes:
        component_notes.append("no dominant high-risk component characterized on screening metrics")

    findings = (
        f"Second-stage brain MRI burden interpretation classified the screening segmentation as {burden}, "
        f"with {', '.join(component_notes)}."
    )
    impression = (
        f"Second-stage brain MRI burden interpretation supports {burden} on segmentation screening review. "
        "Formal neuroradiology interpretation remains required."
    )
    recommendation = (
        "Use this second-stage brain MRI burden summary for prioritization only, treat this as screening support only, and correlate with formal neuroradiology review."
    )

    refined_evidence = dict(evidence)
    refined_evidence["triage_summary"] = {
        "lesion_burden_class": burden,
        "whole_tumor_ratio": whole_ratio,
        "whole_tumor_voxels": whole_voxels,
        "core_present": core_voxels > 0,
        "enhancing_present": enhancing_voxels > 0,
    }

    return {
        "finding": impression,
        "findings": findings,
        "impression": impression,
        "recommendation": recommendation,
        "limitations": [
            "Second-stage brain MRI burden interpretation refines screening segmentation only and is not a substitute for formal neuroradiology diagnosis."
        ],
        "normalized_chain_evidence": refined_evidence,
        "stage_name": "lesion-burden-interpreter",
        "stage_status": "completed",
        "stage_model_name": "mr-lesion-burden-rule-engine",
    }


def run_second_stage_pipeline(second_stage_input_payload: Dict[str, Any]) -> Dict[str, Any]:
    invocation = second_stage_input_payload.get("chain_invocation") or {}
    if not invocation.get("should_invoke"):
        return {
            "invoked": False,
            "status": "skipped",
            "reason": invocation.get("reason") or "second-stage-not-invoked",
            "result": {},
        }

    next_stage = invocation.get("next_stage")
    route_name = str(second_stage_input_payload.get("route_name") or "unknown-route")
    if next_stage == "candidate-triage":
        started = time.time()
        result = run_ct_candidate_triage_second_stage(second_stage_input_payload)
        CHAIN_STAGE_DURATION_SECONDS.labels(route_name, next_stage).observe(time.time() - started)
        return {
            "invoked": True,
            "status": "completed",
            "stage_name": next_stage,
            "result": result,
        }
    if next_stage == "run-summarizer":
        started = time.time()
        result = run_xa_run_summarizer_second_stage(second_stage_input_payload)
        CHAIN_STAGE_DURATION_SECONDS.labels(route_name, next_stage).observe(time.time() - started)
        return {
            "invoked": True,
            "status": "completed",
            "stage_name": next_stage,
            "result": result,
        }
    if next_stage == "cardiac-summary-classifier":
        started = time.time()
        result = run_us_cardiac_summary_second_stage(second_stage_input_payload)
        CHAIN_STAGE_DURATION_SECONDS.labels(route_name, next_stage).observe(time.time() - started)
        return {
            "invoked": True,
            "status": "completed",
            "stage_name": next_stage,
            "result": result,
        }
    if next_stage == "lesion-burden-interpreter":
        started = time.time()
        result = run_mr_lesion_burden_second_stage(second_stage_input_payload)
        CHAIN_STAGE_DURATION_SECONDS.labels(route_name, next_stage).observe(time.time() - started)
        return {
            "invoked": True,
            "status": "completed",
            "stage_name": next_stage,
            "result": result,
        }

    return {
        "invoked": False,
        "status": "unsupported",
        "reason": f"unsupported-second-stage:{next_stage}",
        "result": {},
    }


def validate_ai_response(ai_response: Any) -> Tuple[bool, str, str]:
    if not isinstance(ai_response, dict):
        return False, "AI engine returned a non-dictionary response.", "invalid-engine-response"
    report = ai_response.get("report")
    if not isinstance(report, dict):
        return False, "AI engine response did not include a valid report object.", "invalid-engine-response"

    model_id = ai_response.get("model_id")
    if not model_id:
        return False, "AI engine response did not include a model identifier.", "invalid-engine-response"

    analysis_type = str(report.get("analysis_type") or "").strip()
    conclusion = str(report.get("conclusion") or "").strip()
    observations = report.get("observations")
    if observations is None:
        observations_list: List[str] = []
    elif isinstance(observations, list):
        observations_list = [str(item).strip() for item in observations if str(item).strip()]
    else:
        return False, "AI engine response observations were not a list.", "invalid-engine-response"

    report_type = str(report.get("report_type") or "").strip()
    diagnostic_support = str(report.get("diagnostic_support") or "").strip()

    if not any([analysis_type, conclusion, observations_list, report_type, diagnostic_support]):
        return False, "AI engine response did not include any usable clinical content.", "empty-engine-response"

    return True, "", ""


def estimate_frame_count(metadata: Dict[str, str], instance_count: int = 1) -> int:
    value = metadata.get("NumberOfFrames")
    if value:
        try:
            return int(float(value))
        except Exception:
            pass
    return max(instance_count, 1)


def describe_ct_chest_candidate_regions(metrics: Dict[str, Any], metadata: Dict[str, str]) -> List[str]:
    boxes = metrics.get("top_boxes_xyzxyz") or []
    if not isinstance(boxes, list) or not boxes:
        return []

    try:
        width = float(metadata.get("Columns") or 512)
    except Exception:
        width = 512.0
    try:
        depth = float(metrics.get("slice_count") or metadata.get("NumberOfFrames") or 1)
    except Exception:
        depth = 1.0

    width = width if width > 0 else 512.0
    depth = depth if depth > 0 else 1.0

    region_labels: List[str] = []
    for raw_box in boxes[:3]:
        if not isinstance(raw_box, list) or len(raw_box) < 6:
            continue
        try:
            x1, _, z1, x2, _, z2 = [float(v) for v in raw_box[:6]]
        except Exception:
            continue
        center_x = (x1 + x2) / 2.0
        center_z = (z1 + z2) / 2.0
        side = "right lung" if center_x < (width / 2.0) else "left lung"
        depth_ratio = center_z / depth if depth else 0.5
        if depth_ratio < 0.34:
            vertical_region = "upper"
        elif depth_ratio < 0.67:
            vertical_region = "mid"
        else:
            vertical_region = "lower"
        label = f"{vertical_region} {side}"
        if label not in region_labels:
            region_labels.append(label)
    return region_labels


def summarize_ct_chest_top_candidates(metrics: Dict[str, Any], metadata: Dict[str, str]) -> List[str]:
    boxes = metrics.get("top_boxes_xyzxyz") or []
    scores = metrics.get("top_scores") or []
    if not isinstance(boxes, list) or not boxes:
        return []

    # RetinaNet preprocessing resamples to this fixed spacing before inference.
    sx = 0.703125
    sy = 0.703125
    sz = 1.25

    try:
        width = float(metadata.get("Columns") or 512)
    except Exception:
        width = 512.0
    try:
        depth = float(metrics.get("slice_count") or metadata.get("NumberOfFrames") or 1)
    except Exception:
        depth = 1.0
    width = width if width > 0 else 512.0
    depth = depth if depth > 0 else 1.0

    descriptions: List[str] = []
    for index, raw_box in enumerate(boxes[:3]):
        if not isinstance(raw_box, list) or len(raw_box) < 6:
            continue
        try:
            x1, y1, z1, x2, y2, z2 = [float(v) for v in raw_box[:6]]
        except Exception:
            continue
        center_x = (x1 + x2) / 2.0
        center_z = (z1 + z2) / 2.0
        side = "right lung" if center_x < (width / 2.0) else "left lung"
        depth_ratio = center_z / depth if depth else 0.5
        if depth_ratio < 0.34:
            vertical_region = "upper"
        elif depth_ratio < 0.67:
            vertical_region = "mid"
        else:
            vertical_region = "lower"
        region = f"{vertical_region} {side}"

        dx = max(x2 - x1, 0.0) * sx
        dy = max(y2 - y1, 0.0) * sy
        dz = max(z2 - z1, 0.0) * sz
        score = scores[index] if index < len(scores) else None
        size_phrase = f"approximately {dx:.1f} x {dy:.1f} x {dz:.1f} mm"
        if score is not None:
            descriptions.append(
                f"{region}: suspicious pulmonary nodule candidate measuring {size_phrase}"
            )
        else:
            descriptions.append(
                f"{region}: suspicious pulmonary nodule candidate measuring {size_phrase}"
            )
    return descriptions


def describe_ct_candidate_burden(metrics: Dict[str, Any]) -> str:
    candidate_count = int(metrics.get("candidate_count") or 0)
    top_score = float(metrics.get("top_score") or 0.0)
    if candidate_count <= 0:
        return "no dominant pulmonary nodule candidate"
    if candidate_count == 1 and top_score < 0.35:
        return "a low-volume indeterminate pulmonary nodule candidate"
    if candidate_count <= 3 and top_score < 0.65:
        return "a limited pulmonary nodule candidate burden"
    if candidate_count <= 5:
        return "a focal pulmonary nodule candidate burden"
    return "multiple pulmonary nodule candidates"


def describe_xa_screening_burden(metrics: Dict[str, Any]) -> Tuple[str, str]:
    frame_count = int(metrics.get("frame_count") or 0)
    positive_frames = int(metrics.get("positive_frames") or 0)
    max_ratio = float(metrics.get("max_positive_pixel_ratio") or 0.0)
    if frame_count <= 0 or positive_frames <= 0:
        return (
            "No dominant focal luminal narrowing pattern was identified across the analyzed angiographic run.",
            "No dominant focal luminal narrowing pattern was identified on this angiographic run, but formal angiographic interpretation remains required.",
        )

    frame_fraction = positive_frames / float(frame_count)
    if frame_fraction >= 0.5 or max_ratio >= 0.015:
        burden = "multiframe suspicious narrowing pattern"
    elif frame_fraction >= 0.2 or max_ratio >= 0.006:
        burden = "multiframe moderate suspicious narrowing pattern"
    else:
        burden = "limited focal suspicious narrowing pattern"

    findings = (
        f"Angiographic screening marked a {burden} involving {positive_frames} of {frame_count} frame(s). "
        "The most involved frame showed a small focal narrowing pattern on screening review."
    )
    impression = (
        f"Preliminary angiographic AI screening suggests a {burden}. "
        "Correlation with the full angiographic series and formal interventional/cardiology interpretation is required."
    )
    return findings, impression


def describe_mr_tumor_burden(metrics: Dict[str, Any]) -> Tuple[str, str]:
    whole_voxels = int(metrics.get("whole_tumor_voxels") or 0)
    core_voxels = int(metrics.get("tumor_core_voxels") or 0)
    enhancing_voxels = int(metrics.get("enhancing_tumor_voxels") or 0)
    whole_ratio = float(metrics.get("whole_tumor_ratio") or 0.0)

    if whole_voxels <= 0:
        return (
            "Brain MRI segmentation screening did not mark a dominant candidate tumor burden within the analyzed volume.",
            "No dominant candidate tumor region was marked on screening brain MRI, but neuroradiology review remains required.",
        )

    if whole_ratio >= 0.03:
        burden_phrase = "substantial candidate tumor burden"
    elif whole_ratio >= 0.01:
        burden_phrase = "moderate candidate tumor burden"
    else:
        burden_phrase = "limited candidate tumor burden"

    region_parts = [f"whole-tumor voxels approximately {whole_voxels}"]
    if core_voxels > 0:
        region_parts.append(f"tumor-core voxels approximately {core_voxels}")
    if enhancing_voxels > 0:
        region_parts.append(f"enhancing-tumor voxels approximately {enhancing_voxels}")

    findings = (
        f"Brain MRI segmentation screening marked a {burden_phrase}, involving {whole_ratio * 100.0:.2f}% of the analyzed brain volume; "
        + "; ".join(region_parts)
        + "."
    )
    impression = (
        f"Screening brain MRI segmentation suggests a {burden_phrase} with candidate lesion components requiring neuroradiology correlation."
    )
    return findings, impression


def select_primary_series(series_candidates: List[Dict[str, Any]]) -> Optional[str]:
    if not series_candidates:
        return None

    def score_item(item: Dict[str, Any]) -> Tuple[int, int, int, int]:
        metadata = item.get("metadata") or {}
        description = (metadata.get("SeriesDescription") or "").lower()
        protocol = (metadata.get("ProtocolName") or "").lower()
        combined = f"{description} {protocol}"
        frame_count = estimate_frame_count(metadata, item.get("instance_count", 1))
        score = 0

        if "flair" in combined:
            score += 70
        if "t2" in combined:
            score += 60
        if "t1" in combined:
            score += 40
        if any(term in combined for term in ["ax ", " ax", "tra", "trans"]):
            score += 20
        if "sag" in combined:
            score -= 5
        if any(term in combined for term in ["tof", "angio", "mra"]):
            score -= 30
        if "3d" in combined:
            score -= 10
        if any(term in combined for term in ["localizer", "scout"]):
            score -= 100

        # Prefer moderate-length diagnostic sequences over very long angiographic volumes.
        if 16 <= frame_count <= 40:
            score += 20
        elif frame_count > 100:
            score -= 20

        return (score, -abs(frame_count - 24), frame_count, -len(description))

    chosen = max(series_candidates, key=score_item)
    return chosen.get("series_uid")


def build_study_clinical_summary(
    primary_result: Optional[Dict[str, Any]],
    supporting_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not primary_result:
        return {
            "exam": "Unknown",
            "technique": "No successful series available.",
            "findings": "No automated findings available.",
            "impression": "No successful series were analyzed.",
            "recommendation": "Radiologist review required.",
        }

    metadata = primary_result.get("metadata") or {}
    report = primary_result.get("report") or {}
    metrics = report.get("metrics") or {}
    supporting_names = [
        (item.get("metadata") or {}).get("SeriesDescription") or item.get("series_uid") or "Unknown series"
        for item in supporting_results
    ]
    modality = metadata.get("Modality") or primary_result.get("modality") or "UNKNOWN"
    body_part = metadata.get("BodyPartExamined") or primary_result.get("body_part") or "UNKNOWN"
    study_desc = metadata.get("StudyDescription") or body_part
    report_type = report.get("report_type")
    exam = f"{modality} {study_desc}".strip()
    frame_or_slice_count = metrics.get("slice_count") or estimate_frame_count(metadata, primary_result.get("instance_count", 1))
    technique = (
        f"Primary sequence {metadata.get('SeriesDescription') or primary_result.get('series_uid')}; "
        f"protocol {metadata.get('ProtocolName') or 'not provided'}; "
        f"estimated slices/frames {frame_or_slice_count}."
    )
    if modality == "XA":
        image_type = str(metadata.get("ImageType") or "")
        rows = metadata.get("Rows")
        columns = metadata.get("Columns")
        plane_hint = "biplane " if "BIPLANE" in image_type.upper() else ""
        exam = f"XA {plane_hint}angiography".strip()
        technique_parts = [f"{int(frame_or_slice_count)}-frame XA run"]
        cleaned_image_type = image_type.replace("[", "").replace("]", "").replace("'", "").strip()
        if cleaned_image_type:
            technique_parts.append(cleaned_image_type)
        if rows and columns:
            technique_parts.append(f"{rows} x {columns}")
        technique = "; ".join(technique_parts) + "."
    if supporting_names:
        technique += f" Supporting sequences: {', '.join(supporting_names)}."

    findings = report.get("conclusion") or "No automated findings available."
    impression = report.get("conclusion") or "No automated conclusion available."
    if report_type == "preliminary-stenosis-screening":
        frame_count = int(metrics.get("frame_count") or estimate_frame_count(metadata, primary_result.get("instance_count", 1)))
        positive_frames = int(metrics.get("positive_frames") or 0)
        findings, impression = describe_xa_screening_burden(metrics)
        exam = metadata.get("StudyDescription") or body_part or "XA angiographic run"
        if not exam.upper().startswith("XA"):
            exam = f"XA {exam}".strip()
        xa_series_label = metadata.get("SeriesDescription") or metadata.get("ProtocolName") or "selected angiographic run"
        technique_parts = [f"Dynamic angiographic run reviewed from the {xa_series_label}"]
        if frame_count:
            technique_parts.append(f"{frame_count} frame(s) analyzed")
        technique = "; ".join(technique_parts) + "."
    elif report_type == "preliminary-echocardiography-lv-function":
        exam = metadata.get("StudyDescription") or body_part or "Echocardiogram"
        if not exam.upper().startswith("US"):
            exam = f"US {exam}".strip()
        clip_label = metadata.get("SeriesDescription") or metadata.get("ProtocolName") or "selected echocardiographic cine clip"
        technique_parts = [f"Representative echocardiographic cine clip reviewed from the {clip_label}"]
        if frame_or_slice_count:
            technique_parts.append(f"approximately {int(float(frame_or_slice_count))} frame(s) analyzed")
        frame_time_ms = metrics.get("frame_time_ms")
        if frame_time_ms:
            technique_parts.append(f"nominal frame time {float(frame_time_ms):.1f} ms")
        technique = "; ".join(technique_parts) + "."
        ef_percent = metrics.get("estimated_ef_percent")
        fractional_area_change = metrics.get("fractional_area_change")
        function_phrase = "screening estimate unavailable"
        conclusion_text = str(report.get("conclusion") or "").lower()
        if "severely reduced" in conclusion_text:
            function_phrase = "severely reduced left ventricular systolic function"
        elif "moderately reduced" in conclusion_text:
            function_phrase = "moderately reduced left ventricular systolic function"
        elif "mildly reduced" in conclusion_text:
            function_phrase = "mildly reduced left ventricular systolic function"
        elif "preserved" in conclusion_text:
            function_phrase = "preserved left ventricular systolic function"

        findings_parts = []
        if ef_percent is not None:
            findings_parts.append(
                f"Automated echocardiographic cine analysis estimated left ventricular ejection fraction at approximately {float(ef_percent):.1f}%."
            )
        if fractional_area_change is not None:
            findings_parts.append(
                f"Proxy cavity contraction between end-diastolic and end-systolic frames was about {float(fractional_area_change) * 100.0:.1f}%."
            )
        findings_parts.append(
            f"Clip-based screening pattern was most consistent with {function_phrase}."
        )
        findings = " ".join(findings_parts)
        if report.get("abnormality_status") == "abnormal":
            impression = (
                f"Automated screening suggests {function_phrase}. "
                "Formal cardiology review of the complete echocardiographic study is required for confirmation."
            )
        else:
            impression = (
                "Automated screening suggests preserved left ventricular systolic function on the analyzed clip. "
                "Formal cardiology review of the complete echocardiographic study is still required."
            )
    elif report_type in {"preliminary-screening", "preliminary-2d-screening"}:
        if modality in {"CR", "DX", "MG"}:
            exam = "Chest radiograph"
            projection_label = metadata.get("ViewPosition") or metadata.get("SeriesDescription") or "selected projection image"
            technique = (
                f"Single projection image reviewed from the {projection_label}"
                + (f"; approximately {int(float(frame_or_slice_count))} frame(s) analyzed." if frame_or_slice_count else ".")
            )
        elif modality == "US":
            exam = metadata.get("StudyDescription") or body_part or "Ultrasound examination"
            if not exam.upper().startswith("US"):
                exam = f"US {exam}".strip()
            projection_label = metadata.get("SeriesDescription") or metadata.get("ProtocolName") or "selected ultrasound image"
            technique = (
                f"Single ultrasound image reviewed from the {projection_label}"
                + (f"; approximately {int(float(frame_or_slice_count))} frame(s) analyzed." if frame_or_slice_count else ".")
            )
        else:
            exam = f"{modality} projection image".strip()
            projection_label = metadata.get("ViewPosition") or metadata.get("SeriesDescription") or "selected projection image"
            technique = (
                f"Single projection image reviewed from the {projection_label}"
                + (f"; approximately {int(float(frame_or_slice_count))} frame(s) analyzed." if frame_or_slice_count else ".")
            )
        sorted_metric_items = [
            (str(label), float(score))
            for label, score in metrics.items()
            if isinstance(label, str) and isinstance(score, (int, float))
            and label not in {"slice_count", "abnormal_threshold"}
        ]
        sorted_metric_items.sort(key=lambda item: item[1], reverse=True)
        top_labels = [label for label, _ in sorted_metric_items[:3] if label and label != "No Finding"]
        if modality == "US" and "text" in str(report.get("analysis_type") or ""):
            generated_summary = (
                report.get("conclusion")
                or report.get("finding")
                or ""
            ).strip()
            if (
                not generated_summary
                or generated_summary == "Descriptive AI captioning of the representative medical image was non-informative."
            ):
                findings = (
                    "Representative echocardiographic/ultrasound frame was rendered for descriptive AI review, "
                    "but the returned caption was non-informative and did not support reliable chamber-level or lesion-level characterization."
                )
                impression = (
                    "Non-diagnostic descriptive AI result for this ultrasound examination. "
                    "Specialist review of the full study is required."
                )
            else:
                findings = (
                    "Descriptive AI review of the representative ultrasound frame reported: "
                    f"{generated_summary}"
                )
                impression = (
                    "Descriptive AI summary generated from a rendered ultrasound image. "
                    "This may provide supportive context only and requires specialist confirmation."
                )
        elif top_labels:
            if modality == "US":
                findings = (
                    "AI ultrasound image screening flagged descriptive label patterns involving "
                    + ", ".join(top_labels)
                    + "."
                )
            else:
                findings = (
                    "AI projection-image screening flagged abnormal label patterns involving "
                    + ", ".join(top_labels)
                    + "."
                )
            impression = (
                (
                    "Abnormal descriptive pattern detected on the ultrasound image. Clinical review is required for confirmation and interpretation."
                    if modality == "US"
                    else "Abnormal screening pattern detected on the projection image. Radiologist review is required for confirmation and clinical interpretation."
                )
                if primary_result.get("report", {}).get("abnormality_status") == "abnormal"
                else (
                    "No dominant abnormal descriptive label was flagged on the ultrasound image, but clinician review remains required."
                    if modality == "US"
                    else "No dominant abnormal screening label was flagged, but radiologist review remains required."
                )
            )
        else:
            findings = (
                "AI ultrasound image screening did not flag a dominant abnormal descriptive label pattern."
                if modality == "US"
                else "AI projection-image screening did not flag a dominant abnormal label pattern."
            )
            impression = (
                "No dominant abnormal descriptive label was flagged on the ultrasound image, but clinician review remains required."
                if modality == "US"
                else "No dominant abnormal screening label was flagged, but radiologist review remains required."
            )
    elif report_type == "preliminary-lung-nodule-screening":
        candidate_count = int(metrics.get("candidate_count") or 0)
        top_score = metrics.get("top_score")
        candidate_regions = describe_ct_chest_candidate_regions(metrics, metadata)
        candidate_details = summarize_ct_chest_top_candidates(metrics, metadata)
        burden_phrase = describe_ct_candidate_burden(metrics)
        exam = "CT chest"
        ct_series_label = metadata.get("SeriesDescription") or "selected chest CT series"
        protocol_name = metadata.get("ProtocolName")
        technique_parts = [f"Axial chest CT reviewed from the {ct_series_label}"]
        if protocol_name:
            technique_parts.append(f"protocol {protocol_name}")
        if frame_or_slice_count:
            technique_parts.append(f"approximately {int(float(frame_or_slice_count))} slices analyzed")
        technique = "; ".join(technique_parts) + "."
        if candidate_count > 0:
            findings = (
                f"Chest CT screening marked {burden_phrase} within the analyzed series."
            )
            if candidate_details:
                findings += " Dominant candidate description: " + "; ".join(candidate_details[:2]) + "."
            elif candidate_regions:
                findings += f" Approximate candidate distribution includes the {', '.join(candidate_regions)}."
            elif top_score is not None:
                confidence_band = band_confidence_value(top_score) or "uncertain"
                findings += f" The most suspicious candidate had {confidence_band} screening confidence."
        else:
            findings = (
                "Chest CT screening did not identify a dominant pulmonary nodule candidate "
                "within the analyzed series."
            )
        impression = (
            f"Screening chest CT is concerning for {burden_phrase}. "
            "Radiologist review of the full examination is required to confirm or exclude a true pulmonary nodule."
            if candidate_count > 0
            else "No dominant pulmonary nodule candidate was flagged on AI screening, but formal radiologist review remains required."
        )
    elif report_type == "preliminary-brain-tumor-segmentation":
        exam = "MR brain"
        technique = (
            "Multisequence brain MRI processed with T1c, T1, T2, and FLAIR inputs"
            + (f"; approximately {int(float(frame_or_slice_count))} slices analyzed." if frame_or_slice_count else ".")
        )
        findings, impression = describe_mr_tumor_burden(metrics)
    elif report_type == "non-diagnostic-anatomy":
        if modality == "CT":
            exam = f"CT {body_part}".strip()
            technique = (
                f"CT anatomy-oriented AI review of the selected series"
                + (f"; approximately {int(float(frame_or_slice_count))} slices analyzed." if frame_or_slice_count else ".")
            )
            findings = (
                "An anatomy-focused CT AI model completed structural analysis only. "
                "No pathology-specific diagnostic interpretation was available for this examination."
            )
            impression = "Anatomy-only CT AI support was generated. Formal radiologist interpretation is required."
        elif modality == "MR":
            exam = f"MR {body_part}".strip()
            technique = (
                f"MR anatomy-oriented AI review of the selected series"
                + (f"; approximately {int(float(frame_or_slice_count))} slices analyzed." if frame_or_slice_count else ".")
            )
            findings = (
                "An anatomy-focused MR AI model completed structural analysis only. "
                "No pathology-specific diagnostic interpretation was available for this examination."
            )
            impression = "Anatomy-only MR AI support was generated. Formal radiologist interpretation is required."
    elif report_type in {"non-diagnostic", "analysis-failed-manual-review-required", "manual-review-required"}:
        impression = report.get("conclusion") or "Automated interpretation is non-diagnostic for this study."
    if supporting_names and "Supporting sequences:" not in technique:
        technique += f" Supporting sequences: {', '.join(supporting_names)}."
    return {
        "exam": exam,
        "technique": technique,
        "findings": findings,
        "impression": impression,
        "recommendation": report.get("recommendation") or "Radiologist review required.",
    }


def build_series_metadata_only_result(series_entry: Dict[str, Any], study_uid: str) -> Dict[str, Any]:
    metadata = series_entry["metadata"]
    series_uid = metadata.get("SeriesInstanceUID") or series_entry["series_uid"]
    description = metadata.get("SeriesDescription") or "Unnamed series"
    slice_count = estimate_frame_count(metadata, series_entry.get("instance_count", 1))
    return {
        "filename": f"{series_uid}.dcm",
        "study_uid": study_uid,
        "series_uid": series_uid,
        "sop_uid": metadata.get("SOPInstanceUID"),
        "modality": metadata.get("Modality"),
        "body_part": metadata.get("BodyPartExamined") or metadata.get("StudyDescription"),
        "summary": (
            f"Exam: {metadata.get('Modality') or 'UNKNOWN'} {metadata.get('BodyPartExamined') or 'UNKNOWN'}\n"
            f"Series: {description}\n"
            f"Technique: Supporting sequence, estimated slices/frames {slice_count}.\n"
            "Impression: This series was cataloged as supporting study context only.\n"
            "Recommendation: Review with the primary analyzed series during radiologist interpretation."
        ),
        "status": "completed",
        "model_id": "metadata-only-nonprimary-series",
        "engine_endpoint": "orchestrator-metadata-only",
        "report": {
            "analysis_type": "sequence-metadata-only",
            "abnormality_status": "not-selected-for-primary-inference",
            "conclusion": (
                "This sequence was included for metadata context; intensive model inference was reserved "
                "for the primary representative series to reduce runtime."
            ),
            "recommendation": "Review this sequence as supporting context alongside the primary analyzed series.",
            "observations": [
                f"Series description: {description}.",
                f"Estimated slices/frames: {slice_count}.",
            ],
            "metrics": {"slice_count": slice_count},
        },
        "metadata": metadata,
        "metadata_summary": build_metadata_summary(metadata),
        "instance_count": series_entry.get("instance_count", 1),
        "created_at": datetime.utcnow().isoformat(),
    }


def summarize_series_for_study(item: Dict[str, Any]) -> Dict[str, Any]:
    report = item.get("report") or {}
    metadata = item.get("metadata") or {}
    metrics = report.get("metrics") or {}
    top_structures = (metrics.get("top_segmented_structures_ml") or {})
    return {
        "series_uid": item.get("series_uid"),
        "series_description": metadata.get("SeriesDescription"),
        "modality": item.get("modality"),
        "body_part": item.get("body_part"),
        "model_id": item.get("model_id"),
        "analysis_type": report.get("analysis_type"),
        "analysis_status": report.get("abnormality_status"),
        "conclusion": report.get("conclusion"),
        "recommendation": report.get("recommendation"),
        "slice_count": metrics.get("slice_count"),
        "segmented_structure_count": metrics.get("segmented_structure_count"),
        "top_segmented_structures_ml": top_structures,
        "observations": report.get("observations", []),
        "radiology_style_summary": build_radiology_style_summary(item),
    }


def build_consolidated_study_summary(
    study_uid: str,
    completed_results: List[Dict[str, Any]],
    errors: List[str],
    modalities: List[str],
    body_parts: List[str],
) -> str:
    primary_result = next(
        (item for item in completed_results if (item.get("report") or {}).get("analysis_type") != "sequence-metadata-only"),
        completed_results[0] if completed_results else None,
    )
    supporting_results = [item for item in completed_results if item is not primary_result]
    clinical_summary = build_study_clinical_summary(primary_result, supporting_results)
    primary_metadata = (primary_result or {}).get("metadata") or {}
    primary_modality = primary_metadata.get("Modality") or (primary_result or {}).get("modality")
    lines = [
        f"Exam: {clinical_summary['exam']}",
        f"Study UID: {study_uid}",
        f"Status: {'completed' if completed_results and not errors else 'partial' if completed_results else 'failed'}",
    ]
    if modalities:
        lines.append(f"Modalities: {', '.join(modalities)}")
    if body_parts:
        normalized_body_parts = body_parts
        if primary_modality == "XA":
            normalized_body_parts = ["angiographic acquisition"]
        lines.append(f"Body parts: {', '.join(normalized_body_parts)}")

    if primary_result:
        primary_report = primary_result.get("report") or {}
        primary_series_label = primary_metadata.get("SeriesDescription")
        if not primary_series_label and primary_modality == "XA":
            primary_series_label = "XA primary run"
        lines.append(
            f"Primary analyzed series: {primary_series_label or primary_result.get('series_uid') or 'Unknown series'}"
        )
        lines.append(f"Technique: {clinical_summary['technique']}")
        lines.append(f"Findings: {clinical_summary['findings']}")
        lines.append(f"Impression: {clinical_summary['impression']}")
        lines.append(f"Recommendation: {primary_report.get('recommendation') or 'Radiologist review required.'}")

    supporting_series = []
    for item in supporting_results:
        metadata = item.get("metadata") or {}
        description = metadata.get("SeriesDescription") or item.get("series_uid") or "Unknown series"
        supporting_series.append(description)
    if supporting_series:
        lines.append(f"Supporting sequences: {', '.join(supporting_series)}")

    if errors:
        lines.append(f"Errors: {'; '.join(errors[:5])}")

    return "\n".join(lines)


def build_study_webhook_payload(
    study_uid: str,
    series_results: List[Dict[str, Any]],
    errors: List[str],
    started_at: float,
    job_id: str,
) -> Dict[str, Any]:
    completed_results = [item for item in series_results if item.get("status") == "completed"]
    failure_summary = summarize_failures(series_results)
    primary_result = next(
        (item for item in completed_results if (item.get("report") or {}).get("analysis_type") != "sequence-metadata-only"),
        completed_results[0] if completed_results else None,
    )
    supporting_results = [item for item in completed_results if item is not primary_result]
    modalities = sorted({item.get("modality", "UNKNOWN") for item in completed_results if item.get("modality")})
    body_parts = sorted({item.get("body_part", "UNKNOWN") for item in completed_results if item.get("body_part")})
    series_summaries = [summarize_series_for_study(item) for item in completed_results]
    inference_status = derive_inference_status(series_results, errors)
    consolidated_summary = build_consolidated_study_summary(
        study_uid=study_uid,
        completed_results=completed_results,
        errors=errors,
        modalities=modalities,
        body_parts=body_parts,
    )
    study_limitations: List[str] = []
    study_recommendations: List[str] = []
    for item in completed_results:
        report = item.get("report") or {}
        for limitation in report.get("limitations", []):
            if limitation not in study_limitations:
                study_limitations.append(limitation)
        recommendation = report.get("recommendation")
        if recommendation and recommendation not in study_recommendations:
            study_recommendations.append(recommendation)

    if primary_result:
        primary_report = primary_result.get("report") or {}
        primary_metadata = primary_result.get("metadata") or {}
        clinical_summary = build_study_clinical_summary(primary_result, supporting_results)
        clinical_summary, output_guardrails = apply_output_guardrails(primary_report, clinical_summary)
        clinical_summary = apply_modality_wording_templates(primary_report, clinical_summary)
        result_policy = derive_result_policy(primary_report)
        clinical_summary, claim_scope_guardrails = apply_claim_scope_guardrails(
            primary_report,
            clinical_summary,
            result_policy,
        )
        clinical_summary, recommendation_policy = apply_recommendation_policy(
            primary_report,
            clinical_summary,
            result_policy,
            output_guardrails,
        )
        limitations_value, limitation_policy = apply_limitation_policy(
            primary_report,
            result_policy,
            output_guardrails,
        )
        finding = (
            clinical_summary.get("impression")
            or clinical_summary.get("findings")
            or primary_report.get("conclusion")
            or "No automated finding available."
        )
        diagnostic_support = primary_report.get("diagnostic_support")
        report_type = primary_report.get("report_type")
        abnormal_value = primary_report.get("abnormality_status") == "abnormal"
        confidence_value, confidence_band, confidence_policy = normalize_confidence_disclosure(
            primary_report,
            result_policy,
            output_guardrails,
        )
        if output_guardrails.get("applied") and any(
            reason in output_guardrails.get("reasons", [])
            for reason in {
                "insufficient-structured-evidence",
                "ct-missing-candidate-evidence",
                "xa-missing-frame-burden",
                "us-missing-functional-evidence",
                "mr-missing-lesion-burden",
                "xr-us-missing-observation-evidence",
            }
        ):
            diagnostic_support = "not-supported"
            report_type = "manual-review-required"
            abnormal_value = None
            result_policy = derive_result_policy(
                {
                    **primary_report,
                    "diagnostic_support": diagnostic_support,
                    "report_type": report_type,
                }
            )
            clinical_summary, claim_scope_guardrails = apply_claim_scope_guardrails(
                {
                    **primary_report,
                    "diagnostic_support": diagnostic_support,
                    "report_type": report_type,
                },
                clinical_summary,
                result_policy,
            )
            clinical_summary, recommendation_policy = apply_recommendation_policy(
                {
                    **primary_report,
                    "diagnostic_support": diagnostic_support,
                    "report_type": report_type,
                    "confidence": None,
                },
                clinical_summary,
                result_policy,
                output_guardrails,
            )
            confidence_value, confidence_band, confidence_policy = normalize_confidence_disclosure(
                {
                    **primary_report,
                    "diagnostic_support": diagnostic_support,
                    "report_type": report_type,
                    "confidence": None,
                },
                result_policy,
                output_guardrails,
            )
            limitations_value, limitation_policy = apply_limitation_policy(
                {
                    **primary_report,
                    "diagnostic_support": diagnostic_support,
                    "report_type": report_type,
                },
                result_policy,
                output_guardrails,
            )
        if diagnostic_support in {"anatomy-only", "not-supported"} or report_type in {
            "anatomy-only",
            "non-diagnostic",
            "non-diagnostic-anatomy",
            "analysis-failed-manual-review-required",
            "manual-review-required",
        }:
            abnormal_value = None
            confidence_value = None
        if result_policy.get("must_hide_abnormal") or not result_policy.get("can_set_abnormal"):
            abnormal_value = None
        if result_policy.get("must_hide_confidence") or not result_policy.get("can_expose_confidence"):
            confidence_value = None
            confidence_band = None
        if confidence_policy.get("expose_mode") == "hidden":
            confidence_value = None
            confidence_band = None
        second_stage_selection_policy = derive_second_stage_selection_policy(
            primary_report,
            result_policy,
            output_guardrails,
            primary_metadata,
        )
        second_stage_input_payload = build_second_stage_input_payload(
            study_uid,
            primary_report,
            primary_metadata,
            result_policy,
            output_guardrails,
        )
        ai_result = {
            "model_name": primary_result.get("model_id"),
            "finding": finding,
            "exam": clinical_summary.get("exam"),
            "technique": clinical_summary.get("technique"),
            "findings": clinical_summary.get("findings"),
            "impression": clinical_summary.get("impression"),
            "abnormal": abnormal_value,
            "confidence": confidence_value,
            "confidence_band": confidence_band,
            "diagnostic_support": diagnostic_support,
            "diagnostic_available": (primary_report.get("support_matrix", {}) or {}).get("diagnostic_available", False),
            "report_type": report_type,
            "summary": consolidated_summary,
            "recommendation": clinical_summary.get("recommendation"),
            "limitations": limitations_value,
            "routing_decision": primary_report.get("routing_decision", {}),
            "support_matrix": primary_report.get("support_matrix", {}),
            "model_registry": primary_report.get("support_matrix", {}),
            "output_guardrails": output_guardrails,
            "result_policy": result_policy,
            "claim_scope_guardrails": claim_scope_guardrails,
            "recommendation_policy": recommendation_policy,
            "confidence_policy": confidence_policy,
            "limitation_policy": limitation_policy,
            "model_chain_contract": derive_model_chain_contract(primary_report, primary_metadata),
            "normalized_chain_evidence": normalize_chain_evidence(primary_report, primary_metadata),
            "second_stage_selection_policy": second_stage_selection_policy,
            "second_stage_input_payload": second_stage_input_payload,
            "second_stage_merge_policy": {},
            "second_stage_execution": {},
        }
        ai_result["second_stage_merge_policy"] = derive_second_stage_merge_policy(
            ai_result,
            second_stage_selection_policy,
        )
        second_stage_execution = run_second_stage_pipeline(second_stage_input_payload)
        ai_result["second_stage_execution"] = second_stage_execution
        if second_stage_execution.get("invoked") and second_stage_execution.get("status") == "completed":
            merged_ai_result, merge_summary = merge_second_stage_result(
                ai_result,
                second_stage_execution.get("result") or {},
                ai_result["second_stage_merge_policy"],
            )
            merged_ai_result["second_stage_merge_summary"] = merge_summary
            ai_result = merged_ai_result
        else:
            ai_result["second_stage_merge_summary"] = {
                "applied": False,
                "mode": ai_result["second_stage_merge_policy"].get("merge_mode", "blocked"),
                "merged_fields": [],
                "blocked_fields": list(ai_result["second_stage_merge_policy"].get("blocked_fields", [])),
                "reason": second_stage_execution.get("reason") or ai_result["second_stage_merge_policy"].get("reason", "merge-blocked"),
            }
        dicom_metadata = {
            "metadata_summary": primary_result.get("metadata_summary") or build_metadata_summary(primary_metadata),
            "metadata": primary_metadata,
        }
        ai_result, summary_consistency_guardrails = apply_summary_consistency_guardrails(
            ai_result,
            dicom_metadata,
            inference_status,
            study_uid,
            supporting_results,
        )
        ai_result["summary_consistency_guardrails"] = summary_consistency_guardrails
        ai_result["chain_observability"] = build_chain_observability(ai_result, primary_result)
        record_chain_metrics(ai_result)
        ai_model_id = primary_result.get("model_id")
    else:
        output_guardrails = {"applied": False, "reasons": []}
        result_policy = derive_result_policy(
            {
                "diagnostic_support": "not-supported",
                "report_type": "analysis-failed-manual-review-required",
            }
        )
        claim_scope_guardrails = {
            "applied": True,
            "reasons": ["claim-scope-non-diagnostic"],
            "policy": derive_claim_scope_policy(
                {
                    "diagnostic_support": "not-supported",
                    "report_type": "analysis-failed-manual-review-required",
                },
                result_policy,
            ),
        }
        recommendation_policy = derive_recommendation_policy(
            {
                "diagnostic_support": "not-supported",
                "report_type": "analysis-failed-manual-review-required",
            },
            result_policy,
            output_guardrails,
        )
        limitations_value, limitation_policy = apply_limitation_policy(
            {
                "diagnostic_support": "not-supported",
                "report_type": "analysis-failed-manual-review-required",
                "limitations": ["No study series completed automated analysis successfully."],
            },
            result_policy,
            output_guardrails,
        )
        second_stage_selection_policy = derive_second_stage_selection_policy(
            {
                "diagnostic_support": "not-supported",
                "report_type": "analysis-failed-manual-review-required",
            },
            result_policy,
            output_guardrails,
            {},
        )
        second_stage_input_payload = build_second_stage_input_payload(
            study_uid,
            {
                "diagnostic_support": "not-supported",
                "report_type": "analysis-failed-manual-review-required",
            },
            {},
            result_policy,
            output_guardrails,
        )
        ai_result = {
            "model_name": None,
            "finding": "Automated analysis could not complete on any study series. Manual specialist review is required.",
            "exam": None,
            "technique": None,
            "findings": "No study series completed AI analysis successfully.",
            "impression": "Automated analysis failed at the study level. Manual specialist review is required.",
            "abnormal": None,
            "confidence": None,
            "confidence_band": None,
            "diagnostic_support": "not-supported",
            "diagnostic_available": False,
            "report_type": "analysis-failed-manual-review-required",
            "summary": consolidated_summary,
            "recommendation": "Radiologist review required.",
            "limitations": limitations_value,
            "routing_decision": {},
            "support_matrix": {},
            "model_registry": {},
            "output_guardrails": output_guardrails,
            "result_policy": result_policy,
            "claim_scope_guardrails": claim_scope_guardrails,
            "recommendation_policy": recommendation_policy,
            "confidence_policy": derive_confidence_disclosure_policy(
                {
                    "diagnostic_support": "not-supported",
                    "report_type": "analysis-failed-manual-review-required",
                },
                result_policy,
                output_guardrails,
            ),
            "limitation_policy": limitation_policy,
            "model_chain_contract": derive_model_chain_contract(
                {
                    "diagnostic_support": "not-supported",
                    "report_type": "analysis-failed-manual-review-required",
                },
                {},
            ),
            "normalized_chain_evidence": normalize_chain_evidence(
                {
                    "diagnostic_support": "not-supported",
                    "report_type": "analysis-failed-manual-review-required",
                },
                {},
            ),
            "second_stage_selection_policy": second_stage_selection_policy,
            "second_stage_input_payload": second_stage_input_payload,
            "second_stage_merge_policy": {},
            "second_stage_execution": {},
        }
        ai_result["second_stage_merge_policy"] = derive_second_stage_merge_policy(
            ai_result,
            second_stage_selection_policy,
        )
        ai_result["second_stage_execution"] = run_second_stage_pipeline(second_stage_input_payload)
        ai_result["second_stage_merge_summary"] = {
            "applied": False,
            "mode": ai_result["second_stage_merge_policy"].get("merge_mode", "blocked"),
            "merged_fields": [],
            "blocked_fields": list(ai_result["second_stage_merge_policy"].get("blocked_fields", [])),
            "reason": ai_result["second_stage_execution"].get("reason") or ai_result["second_stage_merge_policy"].get("reason", "merge-blocked"),
        }
        dicom_metadata = {"metadata_summary": {}, "metadata": {}}
        ai_result, summary_consistency_guardrails = apply_summary_consistency_guardrails(
            ai_result,
            dicom_metadata,
            inference_status,
            study_uid,
            supporting_results,
        )
        ai_result["summary_consistency_guardrails"] = summary_consistency_guardrails
        ai_result["chain_observability"] = build_chain_observability(ai_result, None)
        record_chain_metrics(ai_result)
        ai_model_id = None

    return {
        "study_uid": study_uid,
        "job_id": job_id,
        "ai_model_id": ai_model_id,
        "ai_result": ai_result,
        "dicom_metadata": dicom_metadata,
        "status": inference_status,
        "inference_status": inference_status,
        "failure_summary": failure_summary,
        "created_at": datetime.utcnow().isoformat(),
        "latency_ms": int((time.time() - started_at) * 1000),
        "errors": errors,
    }


def save_result_records(result: Dict[str, Any]) -> None:
    if not sb:
        return
    metadata = result.get("metadata", {})
    report = result.get("report", {})
    try:
        sb.table("dicom_reports").insert(
            {
                "model_id": result.get("model_id"),
                "modality": metadata.get("Modality"),
                "body_part": metadata.get("BodyPartExamined") or metadata.get("StudyDescription"),
                "summary": result.get("summary"),
                "captions": [],
                "dicom_metadata": metadata,
                "created_at": datetime.utcnow().isoformat(),
            }
        ).execute()
        sb.table("results").insert(
            {
                "filename": result.get("filename"),
                "status": result.get("status"),
                "model_id": result.get("model_id"),
                "dicom_metadata": metadata,
                "captions": [],
                "findings": report.get("observations", []),
                "impression": report.get("conclusion"),
                "probable_pathology": {
                    "present": report.get("abnormality_status") == "abnormal",
                    "keywords": report.get("abnormalities", []),
                    "confidence": report.get("confidence"),
                },
                "created_at": datetime.utcnow().isoformat(),
            }
        ).execute()
    except Exception as exc:
        result["db_error"] = str(exc)


def process_single_file(fname: str, content: bytes, source: str = "upload") -> Dict[str, Any]:
    result = {"filename": fname}
    metadata = extract_full_dicom_metadata(content)
    metadata_valid, metadata_error, metadata_category = validate_required_metadata(metadata)
    study_uid = metadata.get("StudyInstanceUID")
    series_uid = metadata.get("SeriesInstanceUID")
    modality = metadata.get("Modality")
    body_part = metadata.get("BodyPartExamined") or metadata.get("StudyDescription")

    if not metadata_valid:
        result.update(
            build_failed_result(
                filename=fname,
                series_uid=series_uid,
                study_uid=study_uid,
                modality=modality,
                body_part=body_part,
                error=metadata_error,
                failure_stage="input-validation",
                error_category=metadata_category,
            )
        )
        return result

    if source == "upload":
        try:
            stow_to_pacs(content, fname)
        except Exception as exc:
            error_text = f"STOW failed: {exc}"
            result.update(
                build_failed_result(
                    filename=fname,
                    series_uid=series_uid,
                    study_uid=study_uid,
                    modality=modality,
                    body_part=body_part,
                    error=error_text,
                    failure_stage="stow",
                    error_category=categorize_failure("stow", error_text),
                )
            )
            return result

    try:
        ai_response = post_to_ai_engine(content, fname)
    except Exception as exc:
        error_text = f"AI call failed: {exc}"
        result.update(
            build_failed_result(
                filename=fname,
                series_uid=series_uid,
                study_uid=study_uid,
                modality=modality,
                body_part=body_part,
                error=error_text,
                failure_stage="inference",
                error_category=categorize_failure("inference", error_text),
            )
        )
        return result

    is_valid, validation_error, validation_category = validate_ai_response(ai_response)
    if not is_valid:
        result.update(
            build_failed_result(
                filename=fname,
                series_uid=series_uid,
                study_uid=study_uid,
                modality=modality,
                body_part=body_part,
                error=validation_error,
                failure_stage="response-validation",
                error_category=validation_category,
            )
        )
        return result

    report = ai_response.get("report", {})
    summary = ai_response.get("summary") or generate_summary_from_report(report, metadata)
    payload = {
        "study_uid": metadata.get("StudyInstanceUID"),
        "series_uid": metadata.get("SeriesInstanceUID"),
        "sop_uid": metadata.get("SOPInstanceUID"),
        "filename": fname,
        "modality": metadata.get("Modality"),
        "body_part": metadata.get("BodyPartExamined") or metadata.get("StudyDescription"),
        "summary": summary,
        "status": "completed",
        "model_id": ai_response.get("model_id"),
        "engine_endpoint": ai_response.get("engine_endpoint"),
        "report": report,
        "metadata": metadata,
        "metadata_summary": build_metadata_summary(metadata),
        "created_at": datetime.utcnow().isoformat(),
    }

    save_result_records(payload)
    send_to_elk({**payload, "timestamp": time.time()})

    result.update(payload)
    return result


def process_series(series_files: List[Dict[str, Any]], source: str = "pacs") -> Dict[str, Any]:
    if not series_files:
        error_text = "No series files supplied"
        return build_failed_result(
            filename="unknown-series",
            error=error_text,
            failure_stage="input-validation",
            error_category=categorize_failure("input-validation", error_text),
        )

    first_item = series_files[0]
    first_metadata = extract_full_dicom_metadata(first_item["content"])
    metadata_valid, metadata_error, metadata_category = validate_required_metadata(first_metadata)
    result = {
        "filename": first_item["filename"],
        "study_uid": first_metadata.get("StudyInstanceUID"),
        "series_uid": first_metadata.get("SeriesInstanceUID"),
        "sop_uid": first_metadata.get("SOPInstanceUID"),
        "modality": first_metadata.get("Modality"),
        "body_part": first_metadata.get("BodyPartExamined") or first_metadata.get("StudyDescription"),
        "instance_count": len(series_files),
    }

    if not metadata_valid:
        result.update(
            build_failed_result(
                filename=first_item["filename"],
                series_uid=first_metadata.get("SeriesInstanceUID"),
                study_uid=first_metadata.get("StudyInstanceUID"),
                modality=first_metadata.get("Modality"),
                body_part=first_metadata.get("BodyPartExamined") or first_metadata.get("StudyDescription"),
                error=metadata_error,
                failure_stage="input-validation",
                error_category=metadata_category,
            )
        )
        return result

    if source == "upload":
        for item in series_files:
            try:
                stow_to_pacs(item["content"], item["filename"])
            except Exception as exc:
                error_text = f"STOW failed for {item['filename']}: {exc}"
                result.update(
                    build_failed_result(
                        filename=first_item["filename"],
                        series_uid=first_metadata.get("SeriesInstanceUID"),
                        study_uid=first_metadata.get("StudyInstanceUID"),
                        modality=first_metadata.get("Modality"),
                        body_part=first_metadata.get("BodyPartExamined") or first_metadata.get("StudyDescription"),
                        error=error_text,
                        failure_stage="stow",
                        error_category=categorize_failure("stow", error_text),
                    )
                )
                return result

    try:
        ai_response = post_series_to_ai_engine(series_files)
    except Exception as exc:
        error_text = f"AI call failed: {exc}"
        result.update(
            build_failed_result(
                filename=first_item["filename"],
                series_uid=first_metadata.get("SeriesInstanceUID"),
                study_uid=first_metadata.get("StudyInstanceUID"),
                modality=first_metadata.get("Modality"),
                body_part=first_metadata.get("BodyPartExamined") or first_metadata.get("StudyDescription"),
                error=error_text,
                failure_stage="inference",
                error_category=categorize_failure("inference", error_text),
            )
        )
        return result

    is_valid, validation_error, validation_category = validate_ai_response(ai_response)
    if not is_valid:
        result.update(
            build_failed_result(
                filename=first_item["filename"],
                series_uid=first_metadata.get("SeriesInstanceUID"),
                study_uid=first_metadata.get("StudyInstanceUID"),
                modality=first_metadata.get("Modality"),
                body_part=first_metadata.get("BodyPartExamined") or first_metadata.get("StudyDescription"),
                error=validation_error,
                failure_stage="response-validation",
                error_category=validation_category,
            )
        )
        return result

    report = ai_response.get("report", {})
    summary = build_radiology_style_summary(
        {
            "series_uid": first_metadata.get("SeriesInstanceUID"),
            "modality": first_metadata.get("Modality"),
            "body_part": first_metadata.get("BodyPartExamined") or first_metadata.get("StudyDescription"),
            "metadata": first_metadata,
            "report": report,
            "instance_count": len(series_files),
        }
    )
    payload = {
        "study_uid": first_metadata.get("StudyInstanceUID"),
        "series_uid": first_metadata.get("SeriesInstanceUID"),
        "sop_uid": first_metadata.get("SOPInstanceUID"),
        "filename": first_item["filename"],
        "modality": first_metadata.get("Modality"),
        "body_part": first_metadata.get("BodyPartExamined") or first_metadata.get("StudyDescription"),
        "summary": summary,
        "status": "completed",
        "model_id": ai_response.get("model_id"),
        "engine_endpoint": ai_response.get("engine_endpoint"),
        "report": report,
        "metadata": first_metadata,
        "metadata_summary": build_metadata_summary(first_metadata),
        "instance_count": len(series_files),
        "created_at": datetime.utcnow().isoformat(),
    }

    save_result_records(payload)
    send_to_elk({**payload, "timestamp": time.time()})

    result.update(payload)
    return result


def collect_series_from_upload(file_blobs: List[Tuple[str, bytes]]) -> List[Dict[str, Any]]:
    grouped_series: Dict[str, List[Dict[str, Any]]] = {}
    for index, (fname, body) in enumerate(file_blobs):
        metadata = extract_full_dicom_metadata(body)
        series_key = metadata.get("SeriesInstanceUID") or f"upload-series-{index}"
        grouped_series.setdefault(series_key, []).append(
            {"filename": fname or f"instance-{index:04d}.dcm", "content": body, "metadata": metadata}
        )

    series_entries: List[Dict[str, Any]] = []
    for series_uid, items in grouped_series.items():
        first_metadata = items[0]["metadata"]
        series_entries.append(
            {
                "series_uid": series_uid,
                "study_uid": first_metadata.get("StudyInstanceUID"),
                "metadata": first_metadata,
                "instance_count": len(items),
                "files": items,
            }
        )
    return series_entries


@app.post("/trigger-inference-multi")
async def trigger_inference_multi(files: List[UploadFile] = File(...)):
    start_ts = time.time()
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    log_id = str(uuid.uuid4())
    file_blobs = [(file.filename, await file.read()) for file in files]
    series_entries = collect_series_from_upload(file_blobs)
    primary_series_uid = select_primary_series(series_entries)
    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    study_uid = series_entries[0]["study_uid"] if series_entries else log_id

    for series_entry in series_entries:
        try:
            if series_entry["series_uid"] == primary_series_uid:
                item = process_series(series_entry["files"], "upload")
            else:
                item = build_series_metadata_only_result(series_entry, study_uid)
        except Exception as exc:
            error_text = str(exc)
            item = build_failed_result(
                filename=f"{series_entry['series_uid']}.dcm",
                series_uid=series_entry["series_uid"],
                study_uid=study_uid,
                modality=(series_entry.get("metadata") or {}).get("Modality"),
                body_part=(series_entry.get("metadata") or {}).get("BodyPartExamined")
                or (series_entry.get("metadata") or {}).get("StudyDescription"),
                error=error_text,
                failure_stage="series-processing",
                error_category=categorize_failure("series-processing", error_text),
            )
        results.append(item)
        if item.get("status") != "completed":
            errors.append(f"{item.get('series_uid', item['filename'])}: {item.get('error')}")

    inference_status = derive_inference_status(results, errors)
    if sb:
        try:
            sb.table("inference_logs").insert(
                {
                    "id": log_id,
                    "study_uid": results[0].get("study_uid") if results else "N/A",
                    "status": inference_status,
                    "error_message": "; ".join(errors) if errors else None,
                    "latency_ms": int((time.time() - start_ts) * 1000),
                }
            ).execute()
        except Exception:
            pass

    webhook_payload = build_study_webhook_payload(
        results[0].get("study_uid", log_id) if results else log_id,
        results,
        errors,
        start_ts,
        log_id,
    )
    webhook_payload = publish_result_payload(webhook_payload)
    delivery_status = derive_delivery_status(webhook_payload)
    operation_status = derive_operation_status(inference_status, delivery_status)

    return {
        "id": log_id,
        "status": operation_status,
        "inference_status": inference_status,
        "delivery_status": delivery_status,
        "files": results,
        "errors": errors,
        "webhook_payload": webhook_payload,
    }


@app.post("/trigger-inference-pacs")
def trigger_inference_pacs(study_uid: str = Form(...), series_uid: Optional[str] = Form(None)):
    start_ts = time.time()
    log_id = str(uuid.uuid4())

    try:
        url = (
            f"{PACS_QIDO}/studies/{study_uid}/instances"
            if not series_uid else
            f"{PACS_QIDO}/studies/{study_uid}/series/{series_uid}/instances"
        )
        response = requests.get(url, headers={"Accept": "application/dicom+json"}, timeout=30)
        response.raise_for_status()
        instances = response.json()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PACS query failed: {exc}")

    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    series_map: Dict[str, List[Dict[str, Any]]] = {}
    for instance in instances:
        sop_uid = instance.get("00080018", {}).get("Value", [None])[0]
        current_series_uid = instance.get("0020000E", {}).get("Value", [None])[0]
        sop_class_uid = instance.get("00080016", {}).get("Value", [None])[0]
        modality_value = instance.get("00080060", {}).get("Value", [None])[0]
        if not sop_uid or not current_series_uid:
            errors.append("invalid instance metadata")
            continue
        if modality_value == "SR" or sop_class_uid == "1.2.840.10008.5.1.4.1.1.88.33":
            continue
        series_map.setdefault(current_series_uid, []).append(
            {"sop_uid": sop_uid, "series_uid": current_series_uid}
        )

    series_entries: List[Dict[str, Any]] = []
    for current_series_uid, series_instances in series_map.items():
        series_files: List[Dict[str, Any]] = []
        first_metadata: Optional[Dict[str, str]] = None
        try:
            for series_instance in series_instances:
                sop_uid = series_instance["sop_uid"]
                wado_url = f"{PACS_WADO}/studies/{study_uid}/series/{current_series_uid}/instances/{sop_uid}"
                dicom_response = requests.get(
                    wado_url,
                    headers={"Accept": "multipart/related; type=application/dicom"},
                    timeout=60,
                )
                dicom_response.raise_for_status()
                dicom_bytes = extract_dicom_from_multipart(dicom_response)
                metadata = extract_full_dicom_metadata(dicom_bytes)
                if first_metadata is None:
                    first_metadata = metadata
                series_files.append({"filename": f"{sop_uid}.dcm", "content": dicom_bytes})
            if first_metadata:
                series_entries.append(
                    {
                        "series_uid": current_series_uid,
                        "study_uid": study_uid,
                        "metadata": first_metadata,
                        "instance_count": len(series_files),
                        "files": series_files,
                    }
                )
        except Exception as exc:
            error_text = str(exc)
            item = build_failed_result(
                filename=f"{current_series_uid}.dcm",
                series_uid=current_series_uid,
                study_uid=study_uid,
                error=error_text,
                failure_stage="pacs-fetch",
                error_category=categorize_failure("pacs-fetch", error_text),
            )
            results.append(item)
            errors.append(f"{item.get('series_uid', item['filename'])}: {item.get('error')}")

    primary_series_uid = select_primary_series(series_entries)
    for series_entry in series_entries:
        try:
            if series_entry["series_uid"] == primary_series_uid:
                item = process_series(series_entry["files"], "pacs")
            else:
                item = build_series_metadata_only_result(series_entry, study_uid)
        except Exception as exc:
            error_text = str(exc)
            item = build_failed_result(
                filename=f"{series_entry['series_uid']}.dcm",
                series_uid=series_entry["series_uid"],
                study_uid=study_uid,
                modality=(series_entry.get("metadata") or {}).get("Modality"),
                body_part=(series_entry.get("metadata") or {}).get("BodyPartExamined")
                or (series_entry.get("metadata") or {}).get("StudyDescription"),
                error=error_text,
                failure_stage="series-processing",
                error_category=categorize_failure("series-processing", error_text),
            )
        results.append(item)
        if item.get("status") != "completed":
            errors.append(f"{item.get('series_uid', item['filename'])}: {item.get('error')}")

    inference_status = derive_inference_status(results, errors)
    if sb:
        try:
            sb.table("inference_logs").insert(
                {
                    "id": log_id,
                    "study_uid": study_uid,
                    "status": inference_status,
                    "error_message": "; ".join(errors) if errors else None,
                    "latency_ms": int((time.time() - start_ts) * 1000),
                }
            ).execute()
        except Exception:
            pass

    webhook_payload = build_study_webhook_payload(study_uid, results, errors, start_ts, log_id)
    webhook_payload = publish_result_payload(webhook_payload)
    delivery_status = derive_delivery_status(webhook_payload)
    operation_status = derive_operation_status(inference_status, delivery_status)

    return {
        "id": log_id,
        "status": operation_status,
        "inference_status": inference_status,
        "delivery_status": delivery_status,
        "files": results,
        "errors": errors,
        "webhook_payload": webhook_payload,
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "orchestrator": "up",
        "ai_endpoints": ai_engine_candidates(),
        "supabase_enabled": bool(sb),
    }
