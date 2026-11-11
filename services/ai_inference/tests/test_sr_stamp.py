"""SDD-SCANSURE-AI-002 T4: the DICOM SR names the model that ran and says whether it is a reading.

Runs on pydicom alone (no Flask, no torch): ``sr_payload`` is a pure function of the report
``server.build_report`` produces, and ``build_ai_sr`` is the dataset that goes to dcm4chee.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dicom.sr_builder import (  # noqa: E402
    DEFAULT_SERIES_DESCRIPTION,
    SERIES_DESCRIPTION_MAX,
    build_ai_sr,
    compact_ladder,
    is_diagnostic,
    series_description,
    sr_payload,
)

LADDER = [
    {"route_name": "ct-chest-nodule", "model_id": "monai-lung-nodule-retinanet", "version": "1.0.0", "tier": "T2",
     "outcome": "weights_unavailable", "detail": "CT_MONAI_CHECKPOINT_PATH"},
    {"route_name": "ct-segmentation", "model_id": "totalsegmentator-total", "version": "1.0.0", "tier": "T2",
     "outcome": "declined", "detail": None},
    {"route_name": "fallback", "model_id": "heuristic-fallback", "version": "1.0.0", "tier": "T3",
     "outcome": "model_disabled", "detail": None},
]

# What server.build_report hands store_structured_report for the three shapes that matter.
DESCRIPTIVE = {
    "model_name": "study-descriptor", "model_id": "study-descriptor", "model_version": "1.0.0", "tier": None,
    "diagnostic_support": "descriptive-only", "report_type": "descriptive-summary",
    "abnormality_status": "descriptive-non-diagnostic", "confidence": None,
    "conclusion": "Descriptive AI summary only: no diagnostic AI interpretation was produced for this chest CT. Radiologist interpretation is required.",
    "impact": "No automated interpretation; the summary describes the study for the reader.",
    "recommendation": "Radiologist review required. The AI described the study and did not interpret it.",
    "ladder": LADDER,
}
READING = {
    "model_name": "monai-lung-nodule-retinanet", "model_id": "monai-lung-nodule-retinanet", "model_version": "1.0.0",
    "tier": "T2", "diagnostic_support": "supported", "report_type": "preliminary",
    "abnormality_status": "abnormal", "confidence": 0.81,
    "conclusion": "Two pulmonary nodules, largest 9 mm, right upper lobe.",
    "impact": "x", "recommendation": "y", "ladder": [dict(LADDER[0], outcome="ran", detail=None)],
}
UNAVAILABLE = {
    "model_name": "none", "model_id": None, "model_version": None, "tier": None,
    "diagnostic_support": "not-supported", "report_type": "manual-review-required",
    "abnormality_status": "unsupported-for-diagnostic-inference", "confidence": None,
    "conclusion": None, "impact": None, "recommendation": "Route to radiologist/manual review.", "ladder": [],
}
LEGACY = {  # an engine from before manifests: no stamp at all, a normal reading
    "model_name": "torchxrayvision-densenet121", "abnormality_status": "no-dominant-abnormality-detected",
    "confidence": 0.93, "conclusion": "No acute cardiopulmonary abnormality.",
}


def test_descriptive_summary_is_labelled_non_diagnostic_and_invents_nothing():
    body = sr_payload(DESCRIPTIVE, "summary text")
    assert body["diagnostic"] is False
    assert body["model_id"] == "study-descriptor" and body["model_version"] == "1.0.0" and body["tier"] is None
    assert body["abnormal"] is None and body["confidence"] is None
    assert body["analysis_status"] == "descriptive-non-diagnostic"
    assert "Non-diagnostic AI output" in body["notice"] and "Radiologist interpretation is required" in body["notice"]
    assert [row["outcome"] for row in body["models_considered"]] == ["weights_unavailable", "declined", "model_disabled"]
    assert body["models_considered"][0] == {
        "model": "monai-lung-nodule-retinanet@1.0.0", "tier": "T2",
        "outcome": "weights_unavailable", "detail": "CT_MONAI_CHECKPOINT_PATH",
    }
    # The body survives the JSON round-trip the SR stores, nulls as nulls.
    stored = json.loads(json.dumps(body, default=str))
    assert stored["abnormal"] is None and stored["confidence"] is None


def test_reading_is_diagnostic_with_its_verdict_confidence_and_stamp():
    body = sr_payload(READING, "s")
    assert body["diagnostic"] is True
    assert body["abnormal"] is True and body["confidence"] == 0.81
    assert (body["model_id"], body["model_version"], body["tier"]) == ("monai-lung-nodule-retinanet", "1.0.0", "T2")
    assert "notice" not in body
    assert body["models_considered"] == [{"model": "monai-lung-nodule-retinanet@1.0.0", "tier": "T2", "outcome": "ran", "detail": None}]


def test_unavailable_analysis_carries_no_model_and_no_verdict():
    body = sr_payload(UNAVAILABLE, "s")
    assert body["diagnostic"] is False
    assert body["model_id"] in (None, "none") and body["model_version"] is None
    assert body["abnormal"] is None and body["confidence"] is None
    assert "notice" in body


def test_legacy_engine_without_a_stamp_is_still_a_reading_with_nulls_for_the_stamp():
    body = sr_payload(LEGACY, "s")
    assert body["diagnostic"] is True and body["abnormal"] is False and body["confidence"] == 0.93
    assert body["model_id"] == "torchxrayvision-densenet121"
    assert body["model_version"] is None and body["tier"] is None   # not "unknown"
    assert body["models_considered"] == []


def test_failure_marker_status_is_never_read_as_normal():
    """model-backend-failed is in neither closed vocabulary: not a reading, abnormal unknown."""
    body = sr_payload(dict(LEGACY, abnormality_status="model-backend-failed", confidence=None), "s")
    assert body["diagnostic"] is False and body["abnormal"] is None
    assert is_diagnostic({"abnormality_status": "model-backend-failed", "confidence": None}) is False


@pytest.mark.parametrize("model_id, version, diagnostic, expected", [
    (None, None, None, DEFAULT_SERIES_DESCRIPTION),
    ("monai-lung-nodule-retinanet", "1.0.0", True, "AI Analysis Result - monai-lung-nodule-retinanet@1.0.0"),
    ("study-descriptor", "1.0.0", False, "AI descriptive summary (non-diagnostic) - study-descriptor@1.0.0"),
    ("torchxrayvision-densenet121", None, True, "AI Analysis Result - torchxrayvision-densenet121"),
])
def test_series_description_names_the_model_and_the_non_diagnostic_case(model_id, version, diagnostic, expected):
    text = series_description(model_id, version, diagnostic)
    assert text == expected
    assert len(text) <= SERIES_DESCRIPTION_MAX


def test_series_description_is_truncated_to_the_lo_limit():
    text = series_description("x" * 80, "1.0.0", False)
    assert len(text) == SERIES_DESCRIPTION_MAX


def test_sr_dataset_carries_the_stamp_in_dicom_attributes_and_omits_them_when_unknown():
    stamped = build_ai_sr("1.2.3", "P1", "ANON", "{}", model_id="study-descriptor", model_version="1.0.0",
                          tier=None, diagnostic=False)
    assert stamped.Modality == "SR"
    assert stamped.ManufacturerModelName == "study-descriptor"
    assert stamped.SoftwareVersions == "1.0.0"
    assert "DeviceDescription" not in stamped                      # tier None -> absent, not "manifest tier None"
    assert stamped.SeriesDescription == "AI descriptive summary (non-diagnostic) - study-descriptor@1.0.0"
    assert stamped.StudyInstanceUID == "1.2.3" and stamped.TextValue == "{}"

    tiered = build_ai_sr("1.2.3", "P1", "ANON", "{}", model_id="monai-lung-nodule-retinanet", model_version="1.0.0",
                         tier="T2", diagnostic=True)
    assert tiered.DeviceDescription == "manifest tier T2"
    assert tiered.SeriesDescription == "AI Analysis Result - monai-lung-nodule-retinanet@1.0.0"

    bare = build_ai_sr("1.2.3", "P1", "ANON", "text")                # the pre-T4 call shape still works
    assert bare.SeriesDescription == DEFAULT_SERIES_DESCRIPTION
    for keyword in ("ManufacturerModelName", "SoftwareVersions", "DeviceDescription"):
        assert keyword not in bare, keyword


def test_compact_ladder_ignores_junk_rows_and_names_the_registry_row():
    rows = compact_ladder([None, "x", {"route_name": "registry", "outcome": "profile_incomplete", "detail": "modality"}])
    assert rows == [{"model": "registry", "tier": None, "outcome": "profile_incomplete", "detail": "modality"}]
