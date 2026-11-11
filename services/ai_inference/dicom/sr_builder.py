"""The DICOM Structured Report written back to the PACS for an AI result.

The SR is the system of record for an AI result inside imaging: it lands in the study a
radiologist already opens, and it survives this service. Two things it must therefore say
plainly (SDD-SCANSURE-AI-002 T4):

* **which model produced it** -- ``ManufacturerModelName`` / ``SoftwareVersions`` carry the
  manifest ``model_id`` / ``version`` of the model that ran, and the series description names
  it, so a viewer's series list already shows ``monai-lung-nodule-retinanet@1.0.0`` or
  ``study-descriptor@1.0.0``;
* **whether it is a reading at all** -- a descriptive summary (ladder rung 3) or an
  unavailable analysis is labelled non-diagnostic in the series description and in the body,
  and ``abnormal`` / ``confidence`` stay ``null`` there. Nothing here invents a value the
  engine did not produce: a missing stamp leaves the attribute absent, never "unknown".
"""
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import (
    generate_uid,
    ExplicitVRLittleEndian,
    ComprehensiveSRStorage
)

#: analysis_status values that are NOT a diagnostic reading. The same closed set the
#: monorepo's pac_adapter.NON_DIAGNOSTIC_STATUSES maps to ANALYSIS_UNAVAILABLE / text-only.
NON_DIAGNOSTIC_STATUSES = frozenset({
    "non-diagnostic-insufficient-series",
    "unsupported-for-diagnostic-inference",
    "descriptive-non-diagnostic",
    "non-diagnostic",
})
NON_DIAGNOSTIC_SUPPORT = frozenset({"descriptive-only", "not-supported"})

#: Closed vocabularies for reading ``abnormal`` back out of ``abnormality_status``.
ABNORMAL_STATUSES = frozenset({"abnormal", "positive", "suspicious"})
NORMAL_STATUSES = frozenset({"normal", "negative", "no-dominant-abnormality-detected", "no-abnormality-detected"})

SERIES_DESCRIPTION_MAX = 64  # LO
DEFAULT_SERIES_DESCRIPTION = "AI Analysis Result"


def abnormal_from_status(report: Dict[str, Any]) -> Optional[bool]:
    """``abnormal`` as the report's status vocabulary states it, and only from it. A status
    outside both closed sets (a failure marker, a descriptive status) is unknown, never False."""
    status = str(report.get("abnormality_status") or "").lower()
    if status in ABNORMAL_STATUSES:
        return True
    if status in NORMAL_STATUSES:
        return False
    return None


def is_diagnostic(report: Dict[str, Any]) -> bool:
    """True only when the report is a reading: not descriptive-only / unsupported, a status
    outside the non-diagnostic set, and an abnormality verdict or a confidence actually present."""
    if str(report.get("diagnostic_support") or "").lower() in NON_DIAGNOSTIC_SUPPORT:
        return False
    if str(report.get("abnormality_status") or "").lower() in NON_DIAGNOSTIC_STATUSES:
        return False
    return abnormal_from_status(report) is not None or report.get("confidence") is not None


def series_description(model_id: Optional[str], model_version: Optional[str], diagnostic: Optional[bool]) -> str:
    """``AI Analysis Result - <model>@<version>``; ``AI descriptive summary (non-diagnostic) - ...``
    when the result is not a reading; the bare legacy label when nothing is known."""
    head = "AI descriptive summary (non-diagnostic)" if diagnostic is False else DEFAULT_SERIES_DESCRIPTION
    if model_id:
        who = f"{model_id}@{model_version}" if model_version else str(model_id)
        head = f"{head} - {who}"
    return head[:SERIES_DESCRIPTION_MAX]


def compact_ladder(ladder: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The ladder rows an SR reader needs: who, what happened, why. Nothing else."""
    rows = []
    for step in ladder or []:
        if not isinstance(step, dict):
            continue
        rows.append({
            "model": f"{step.get('model_id')}@{step.get('version')}" if step.get("model_id") else step.get("route_name"),
            "tier": step.get("tier"),
            "outcome": step.get("outcome"),
            "detail": step.get("detail"),
        })
    return rows


def sr_payload(report: Dict[str, Any], summary: str) -> Dict[str, Any]:
    """The JSON body stored as the SR's text value.

    Pure function of the report so it can be tested without the Flask app. ``abnormal`` and
    ``confidence`` are copied as they are -- ``None`` stays ``null``.
    """
    diagnostic = is_diagnostic(report)
    body: Dict[str, Any] = {
        "model_id": report.get("model_id") or report.get("model_name"),
        "model_version": report.get("model_version"),
        "tier": report.get("tier"),
        "diagnostic": diagnostic,
        "analysis_status": report.get("abnormality_status"),
        "diagnostic_support": report.get("diagnostic_support"),
        "abnormal": abnormal_from_status(report) if diagnostic else None,
        "confidence": report.get("confidence") if diagnostic else None,
        "summary": summary,
        "conclusion": report.get("conclusion"),
        "impact": report.get("impact"),
        "recommendation": report.get("recommendation"),
        "models_considered": compact_ladder(report.get("ladder")),
    }
    if not diagnostic:
        body["notice"] = (
            "Non-diagnostic AI output: no model produced a diagnostic interpretation of this study. "
            "The text describes the study and the models considered. Radiologist interpretation is required."
        )
    return body


def build_ai_sr(study_uid, patient_id, patient_name, result_text, *,
                model_id: Optional[str] = None, model_version: Optional[str] = None,
                tier: Optional[str] = None, diagnostic: Optional[bool] = None):
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = ComprehensiveSRStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        None,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128
    )

    # Patient
    ds.PatientID = patient_id
    ds.PatientName = patient_name

    # Study / Series
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    ds.Modality = "SR"
    ds.SeriesDescription = series_description(model_id, model_version, diagnostic)
    ds.Manufacturer = "PAC-AI"
    # The model stamp, in the attributes DICOM has for exactly this. Absent when unknown:
    # a reader must be able to tell "no model named" from a model called "unknown".
    if model_id:
        ds.ManufacturerModelName = str(model_id)[:64]
    if model_version:
        ds.SoftwareVersions = str(model_version)[:64]
    if tier:
        ds.DeviceDescription = f"manifest tier {tier}"[:64]

    # Time
    now = datetime.now()
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")

    # Minimal SR text
    ds.ValueType = "TEXT"
    ds.TextValue = result_text

    return ds
