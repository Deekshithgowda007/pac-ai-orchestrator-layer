import io
import json
import logging
import os
import shutil
import tempfile
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import pydicom
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from supabase import Client, create_client

from dicom.dicom_sender import send_sr_to_dcm4chee
from dicom.sr_builder import build_ai_sr
from inference_engine import InferenceEngine

load_dotenv()

log = logging.getLogger("ai_inference")
logging.basicConfig(level=logging.INFO)

AI_WORKERS = int(os.getenv("AI_WORKERS", "2"))
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
RESULT_WEBHOOK_URL = os.getenv("RESULT_WEBHOOK_URL", "").strip()
STORE_DICOM_SR = os.getenv("STORE_DICOM_SR", "true").lower() == "true"

INFERENCE_REQUESTS = Counter("inference_requests_total", "Total inference requests")
INFERENCE_ERRORS = Counter("inference_errors_total", "Total inference errors")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Latency for inference request")

supabase_client: Optional[Client] = None
if USE_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        log.info("Supabase client initialized")
    except Exception as exc:
        log.warning("Supabase initialization failed: %s", exc)

engine = InferenceEngine()
app = Flask(__name__)
app.config["TRUSTED_HOSTS"] = ["*", "ai_inference", "ai_inference_backup", "localhost", "127.0.0.1"]


def extract_dicom_metadata(file_bytes: bytes) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    ds = pydicom.dcmread(io.BytesIO(file_bytes), stop_before_pixels=True, force=True)
    for elem in ds:
        if elem.keyword and elem.keyword != "PixelData":
            try:
                metadata[elem.keyword] = str(elem.value)
            except Exception:
                metadata[elem.keyword] = repr(elem.value)
    return metadata


def build_report(engine_result: Dict[str, Any], metadata: Dict[str, str]) -> Dict[str, Any]:
    modality = metadata.get("Modality", "UNKNOWN")
    body_part = (
        metadata.get("BodyPartExamined")
        or metadata.get("StudyDescription")
        or metadata.get("SeriesDescription")
        or "UNKNOWN"
    )
    abnormal_value = engine_result.get("abnormal")
    if abnormal_value is None:
        abnormal = None
    else:
        abnormal = bool(abnormal_value)

    confidence_value = engine_result.get("confidence")
    if confidence_value is None:
        confidence = None
    else:
        confidence = float(confidence_value)
    region = engine_result.get("region", "diffuse/undetermined")
    observations = engine_result.get("observations", [])
    analysis_type = engine_result.get("analysis_type", "preliminary-rule-based")
    explicit_status = engine_result.get("analysis_status")

    if explicit_status:
        status = explicit_status
    elif analysis_type == "insufficient-series":
        status = "non-diagnostic-insufficient-series"
    else:
        if abnormal is None:
            status = "non-diagnostic"
        else:
            status = "abnormal" if abnormal else "no-dominant-abnormality-detected"

    abnormalities = engine_result.get("abnormalities") or (observations[-2:] if abnormal else [])
    conclusion = engine_result.get("finding", "No automated conclusion available.")
    impact = engine_result.get("impact", "No impact assessment available.")
    if engine_result.get("recommendation"):
        recommendation = engine_result["recommendation"]
    elif analysis_type == "unsupported-modality":
        recommendation = "Route to radiologist/manual review."
    elif analysis_type == "insufficient-series":
        recommendation = (
            f"Upload the complete {modality} DICOM series or full study before attempting automated "
            "volume analysis, then send the case for radiologist review."
        )
    else:
        recommendation = (
            "Escalate for prompt radiologist review and correlate clinically."
            if abnormal else
            "Proceed with standard radiologist review and correlate with symptoms."
        )

    return {
        "analysis_type": analysis_type,
        "model_name": engine_result.get("model_name", "unknown"),
        "diagnostic_support": engine_result.get("diagnostic_support", "not-supported"),
        "diagnostic_available": engine_result.get("support_matrix", {}).get("diagnostic_available", False),
        "report_type": engine_result.get("report_type", "non-diagnostic"),
        "modality": modality,
        "body_part": body_part,
        "anatomy_involved": engine_result.get("anatomy_involved", body_part),
        "abnormality_status": status,
        "confidence": confidence,
        "region_of_interest": region,
        "observations": observations,
        "abnormalities": abnormalities,
        "impact": impact,
        "conclusion": conclusion,
        "recommendation": recommendation,
        "limitations": engine_result.get("limitations", []),
        "metrics": engine_result.get("metrics", {}),
        "routing_decision": engine_result.get("routing_decision", {}),
        "support_matrix": engine_result.get("support_matrix", {}),
        "model_registry": engine_result.get("support_matrix", {}),
        "metadata_summary": {
            "study_uid": metadata.get("StudyInstanceUID"),
            "series_uid": metadata.get("SeriesInstanceUID"),
            "sop_uid": metadata.get("SOPInstanceUID"),
            "series_description": metadata.get("SeriesDescription"),
            "study_description": metadata.get("StudyDescription"),
            "study_date": metadata.get("StudyDate"),
        },
    }


def refine_report_with_openai(report: Dict[str, Any], metadata: Dict[str, str]) -> Dict[str, Any]:
    return report


def build_summary(report: Dict[str, Any]) -> str:
    observations = report.get("observations") or []
    abnormalities = report.get("abnormalities") or []
    summary_lines = [
        f"Modality: {report.get('modality', 'UNKNOWN')}",
        f"Body part: {report.get('body_part', 'UNKNOWN')}",
        f"Analysis type: {report.get('analysis_type', 'unknown')}",
        f"Status: {report.get('abnormality_status', 'unknown')}",
        f"Region: {report.get('region_of_interest', 'diffuse/undetermined')}",
    ]
    if observations:
        summary_lines.append(f"Observation: {observations[0]}")
    if len(observations) > 1:
        summary_lines.append(f"Finding detail: {observations[1]}")
    if abnormalities:
        summary_lines.append(f"Abnormalities: {', '.join(str(item) for item in abnormalities)}")
    summary_lines.append(f"Impact: {report.get('impact', 'No impact assessment available.')}")
    summary_lines.append(f"Conclusion: {report.get('conclusion', 'No automated conclusion available.')}")
    summary_lines.append(f"Recommendation: {report.get('recommendation', 'No recommendation available.')}")
    return "\n".join(summary_lines)


def safe_insert_supabase(table: str, row: Dict[str, Any]) -> None:
    if not supabase_client:
        return
    try:
        supabase_client.table(table).insert(json.loads(json.dumps(row, default=str))).execute()
    except Exception as exc:
        log.warning("Supabase insert into %s failed: %s", table, exc)


def send_webhook(payload: Dict[str, Any]) -> None:
    if not RESULT_WEBHOOK_URL:
        log.info("Webhook skipped because RESULT_WEBHOOK_URL is empty")
        return
    try:
        import requests

        resp = requests.post(RESULT_WEBHOOK_URL, json=payload, timeout=30)
        log.info("Webhook delivered to %s with status %s", RESULT_WEBHOOK_URL, resp.status_code)
    except Exception as exc:
        log.warning("Webhook delivery failed: %s", exc)


def store_structured_report(study_uid: Optional[str], metadata: Dict[str, str], summary: str, report: Dict[str, Any]) -> None:
    if not STORE_DICOM_SR or not study_uid:
        return
    try:
        sr_text = json.dumps(
            {
                "summary": summary,
                "conclusion": report.get("conclusion"),
                "impact": report.get("impact"),
                "recommendation": report.get("recommendation"),
            },
            default=str,
        )
        sr = build_ai_sr(
            study_uid=study_uid,
            patient_id=metadata.get("PatientID", "UNKNOWN"),
            patient_name=metadata.get("PatientName", "ANONYMOUS"),
            result_text=sr_text,
        )
        send_sr_to_dcm4chee(sr)
    except Exception as exc:
        log.warning("DICOM SR store failed: %s", exc)


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/upload", methods=["POST"])
def upload():
    start_time = time.time()
    INFERENCE_REQUESTS.inc()
    temp_path: Optional[str] = None
    try:
        raw_body = request.get_data(cache=True)
        uploaded_files = request.files.getlist("files") or request.files.getlist("file")
        input_instance_count = 1
        if uploaded_files:
            temp_path = tempfile.mkdtemp(prefix="dicom-series-")
            first_file_bytes: Optional[bytes] = None
            input_instance_count = 0
            for index, uploaded_file in enumerate(uploaded_files):
                file_bytes = uploaded_file.read()
                if not file_bytes:
                    continue
                input_instance_count += 1
                if first_file_bytes is None:
                    first_file_bytes = file_bytes
                filename = uploaded_file.filename or f"instance-{index:04d}.dcm"
                if not filename.lower().endswith(".dcm"):
                    filename = f"{filename}.dcm"
                with open(os.path.join(temp_path, filename), "wb") as handle:
                    handle.write(file_bytes)
            if not first_file_bytes or input_instance_count == 0:
                return jsonify(
                    {
                        "ok": False,
                        "error": "No DICOM payload provided",
                        "content_type": request.content_type,
                        "form_keys": list(request.form.keys()),
                        "file_keys": list(request.files.keys()),
                    }
                ), 400
            metadata = extract_dicom_metadata(first_file_bytes)
        else:
            file_bytes = raw_body
            if not file_bytes:
                return jsonify(
                    {
                        "ok": False,
                        "error": "No DICOM payload provided",
                        "content_type": request.content_type,
                        "form_keys": list(request.form.keys()),
                        "file_keys": list(request.files.keys()),
                    }
                ), 400
            metadata = extract_dicom_metadata(file_bytes)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as handle:
                handle.write(file_bytes)
                temp_path = handle.name

        engine_result = engine.run(temp_path)
        report = build_report(engine_result, metadata)
        report = refine_report_with_openai(report, metadata)
        summary = build_summary(report)

        response_payload = {
            "ok": True,
            "id": str(uuid.uuid4()),
            "model_id": report["model_name"],
            "engine_instance": os.getenv("ENGINE_INSTANCE_NAME", "primary"),
            "dicom_metadata": metadata,
            "report": report,
            "summary": summary,
            "modality": report["modality"],
            "body_part": report["body_part"],
            "input_instance_count": input_instance_count,
            "latency_sec": round(time.time() - start_time, 2),
        }

        safe_insert_supabase(
            "dicom_reports",
            {
                "model_id": report["model_name"],
                "modality": report["modality"],
                "body_part": report["body_part"],
                "summary": summary,
                "captions": [],
                "dicom_metadata": metadata,
                "created_at": datetime.utcnow().isoformat(),
            },
        )
        store_structured_report(metadata.get("StudyInstanceUID"), metadata, summary, report)
        INFERENCE_LATENCY.observe(time.time() - start_time)
        return jsonify(response_payload)
    except Exception as exc:
        INFERENCE_ERRORS.inc()
        log.exception("Upload error: %s", exc)
        return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            if os.path.isdir(temp_path):
                shutil.rmtree(temp_path, ignore_errors=True)
            else:
                os.unlink(temp_path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "workers": AI_WORKERS, "supabase_enabled": bool(supabase_client)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, threaded=True)
