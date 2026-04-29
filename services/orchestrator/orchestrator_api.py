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
from prometheus_client import make_asgi_app
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


def send_to_webhook(payload: Dict[str, Any]) -> None:
    if not RESULT_WEBHOOK_URL:
        print("Webhook skipped because RESULT_WEBHOOK_URL is empty", flush=True)
        return
    try:
        response = requests.post(RESULT_WEBHOOK_URL, json=payload, timeout=30)
        print(f"Webhook delivered to {RESULT_WEBHOOK_URL} with status {response.status_code}", flush=True)
    except Exception as exc:
        print(f"Webhook delivery failed: {exc}", flush=True)


def send_to_kafka(payload: Dict[str, Any]) -> None:
    producer = get_kafka_result_producer()
    if not producer:
        print("Kafka result publish skipped because producer is unavailable", flush=True)
        return
    try:
        producer.send(KAFKA_RESULT_TOPIC, value=payload).get(timeout=30)
        print(
            f"Kafka result published to {KAFKA_RESULT_BOOTSTRAP_SERVERS} topic={KAFKA_RESULT_TOPIC}",
            flush=True,
        )
    except Exception as exc:
        print(f"Kafka result publish failed: {exc}", flush=True)


def send_to_hospital_service(payload: Dict[str, Any]) -> None:
    if not HOSPITAL_RESULT_URL:
        return
    try:
        requests.post(HOSPITAL_RESULT_URL, json=payload, timeout=30)
    except Exception:
        pass


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
                findings += f" The most suspicious candidate had high screening confidence ({top_score})."
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
    primary_result = next(
        (item for item in completed_results if (item.get("report") or {}).get("analysis_type") != "sequence-metadata-only"),
        completed_results[0] if completed_results else None,
    )
    supporting_results = [item for item in completed_results if item is not primary_result]
    modalities = sorted({item.get("modality", "UNKNOWN") for item in completed_results if item.get("modality")})
    body_parts = sorted({item.get("body_part", "UNKNOWN") for item in completed_results if item.get("body_part")})
    series_summaries = [summarize_series_for_study(item) for item in completed_results]
    overall_status = "completed" if completed_results and not errors else "failed" if errors and not completed_results else "partial"
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
        finding = (
            clinical_summary.get("impression")
            or clinical_summary.get("findings")
            or primary_report.get("conclusion")
            or "No automated finding available."
        )
        diagnostic_support = primary_report.get("diagnostic_support")
        report_type = primary_report.get("report_type")
        abnormal_value = primary_report.get("abnormality_status") == "abnormal"
        confidence_value = primary_report.get("confidence")
        if diagnostic_support in {"anatomy-only", "not-supported"} or report_type in {
            "anatomy-only",
            "non-diagnostic",
            "non-diagnostic-anatomy",
            "analysis-failed-manual-review-required",
            "manual-review-required",
        }:
            abnormal_value = None
            confidence_value = None
        if report_type == "preliminary-stenosis-screening":
            confidence_value = None
        ai_result = {
            "model_name": primary_result.get("model_id"),
            "finding": finding,
            "exam": clinical_summary.get("exam"),
            "technique": clinical_summary.get("technique"),
            "findings": clinical_summary.get("findings"),
            "impression": clinical_summary.get("impression"),
            "abnormal": abnormal_value,
            "confidence": confidence_value,
            "diagnostic_support": diagnostic_support,
            "diagnostic_available": (primary_report.get("support_matrix", {}) or {}).get("diagnostic_available", False),
            "report_type": report_type,
            "summary": consolidated_summary,
            "recommendation": primary_report.get("recommendation"),
            "limitations": primary_report.get("limitations", []),
            "routing_decision": primary_report.get("routing_decision", {}),
            "support_matrix": primary_report.get("support_matrix", {}),
            "model_registry": primary_report.get("support_matrix", {}),
        }
        dicom_metadata = {
            "metadata_summary": primary_result.get("metadata_summary") or build_metadata_summary(primary_metadata),
            "metadata": primary_metadata,
        }
        ai_model_id = primary_result.get("model_id")
    else:
        ai_result = {
            "model_name": None,
            "finding": "No successful series were analyzed.",
            "abnormal": None,
            "confidence": None,
            "diagnostic_available": False,
            "summary": consolidated_summary,
            "recommendation": "Radiologist review required.",
            "limitations": [],
        }
        dicom_metadata = {"metadata_summary": {}, "metadata": {}}
        ai_model_id = None

    return {
        "study_uid": study_uid,
        "job_id": job_id,
        "ai_model_id": ai_model_id,
        "ai_result": ai_result,
        "dicom_metadata": dicom_metadata,
        "status": overall_status,
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

    if source == "upload":
        try:
            stow_to_pacs(content, fname)
        except Exception as exc:
            result.update({"status": "failed", "error": f"STOW failed: {exc}"})
            return result

    try:
        ai_response = post_to_ai_engine(content, fname)
    except Exception as exc:
        result.update({"status": "failed", "error": f"AI call failed: {exc}"})
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
    send_to_hospital_service(payload)
    send_to_webhook(payload)
    send_to_elk({**payload, "timestamp": time.time()})

    result.update(payload)
    return result


def process_series(series_files: List[Dict[str, Any]], source: str = "pacs") -> Dict[str, Any]:
    if not series_files:
        return {"filename": "unknown-series", "status": "failed", "error": "No series files supplied"}

    first_item = series_files[0]
    first_metadata = extract_full_dicom_metadata(first_item["content"])
    result = {
        "filename": first_item["filename"],
        "study_uid": first_metadata.get("StudyInstanceUID"),
        "series_uid": first_metadata.get("SeriesInstanceUID"),
        "sop_uid": first_metadata.get("SOPInstanceUID"),
        "modality": first_metadata.get("Modality"),
        "body_part": first_metadata.get("BodyPartExamined") or first_metadata.get("StudyDescription"),
        "instance_count": len(series_files),
    }

    if source == "upload":
        for item in series_files:
            try:
                stow_to_pacs(item["content"], item["filename"])
            except Exception as exc:
                result.update({"status": "failed", "error": f"STOW failed for {item['filename']}: {exc}"})
                return result

    try:
        ai_response = post_series_to_ai_engine(series_files)
    except Exception as exc:
        result.update({"status": "failed", "error": f"AI call failed: {exc}"})
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
    send_to_hospital_service(payload)
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
            item = {
                "filename": f"{series_entry['series_uid']}.dcm",
                "series_uid": series_entry["series_uid"],
                "status": "failed",
                "error": str(exc),
            }
        results.append(item)
        if item.get("status") != "completed":
            errors.append(f"{item.get('series_uid', item['filename'])}: {item.get('error')}")

    status = "completed" if all(item["status"] == "completed" for item in results) else "failed"
    if sb:
        try:
            sb.table("inference_logs").insert(
                {
                    "id": log_id,
                    "study_uid": results[0].get("study_uid") if results else "N/A",
                    "status": status,
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
    send_to_webhook(webhook_payload)

    return {"id": log_id, "status": status, "files": results, "errors": errors, "webhook_payload": webhook_payload}


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
            item = {"filename": f"{current_series_uid}.dcm", "series_uid": current_series_uid, "status": "failed", "error": str(exc)}
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
            item = {"filename": f"{series_entry['series_uid']}.dcm", "series_uid": series_entry["series_uid"], "status": "failed", "error": str(exc)}
        results.append(item)
        if item.get("status") != "completed":
            errors.append(f"{item.get('series_uid', item['filename'])}: {item.get('error')}")

    status = "completed" if all(item["status"] == "completed" for item in results) else "failed"
    if sb:
        try:
            sb.table("inference_logs").insert(
                {
                    "id": log_id,
                    "study_uid": study_uid,
                    "status": status,
                    "error_message": "; ".join(errors) if errors else None,
                    "latency_ms": int((time.time() - start_ts) * 1000),
                }
            ).execute()
        except Exception:
            pass

    webhook_payload = build_study_webhook_payload(study_uid, results, errors, start_ts, log_id)
    send_to_webhook(webhook_payload)
    send_to_kafka(webhook_payload)

    return {"id": log_id, "status": status, "files": results, "errors": errors, "webhook_payload": webhook_payload}


@app.get("/health")
def health():
    return {
        "ok": True,
        "orchestrator": "up",
        "ai_endpoints": ai_engine_candidates(),
        "supabase_enabled": bool(sb),
    }
