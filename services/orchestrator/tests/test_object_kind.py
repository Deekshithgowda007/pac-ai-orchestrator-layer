"""Non-image objects stay out of series assembly (SDD-SCANSURE-AI-001 T7).

One test per excluded kind: SR, SEG, PR, KO, RT, PDF, waveform, and an unknown SOP class.
The structured-report exclusion that existed before this change is the first of them.
"""
import io
import pathlib
import sys
from unittest.mock import patch

import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ORCHESTRATOR_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_DIR))

import object_kind as kinds  # noqa: E402
import orchestrator_api  # noqa: E402

S = "1.2.840.10008.5.1.4.1.1."
CT = S + "2"
# kind -> (SOP class, modality an archive would report for it)
EXCLUDED = {
    "sr": (S + "88.33", "SR"),                 # Comprehensive SR: the exclusion that existed before
    "seg": (S + "66.4", "SEG"),
    "pr": (S + "11.1", "PR"),
    "ko": (S + "88.59", "KO"),
    "rt": (S + "481.3", "RTSTRUCT"),
    "pdf": (S + "104.1", "DOC"),
    "waveform": (S + "9.1.1", "ECG"),
    "other": ("1.2.840.10008.5.1.4.1.1.999.1", "OT"),
}


# --- the helper -------------------------------------------------------------------------------

@pytest.mark.parametrize("sop_class, expected", [
    (S + "1", "image"), (S + "1.1", "image"), (S + "1.2", "image"), (S + "1.3", "image"),
    (S + "2", "image"), (S + "4", "image"), (S + "6.1", "image"), (S + "7", "image"),
    (S + "12.1", "image"), (S + "12.2", "image"), (S + "20", "image"), (S + "128", "image"),
    (S + "77.1.5.4", "image"), (S + "77.1.6", "image"),
    (S + "3.1", "multiframe"), (S + "7.2", "multiframe"), (S + "77.1.4.1", "multiframe"),
    (S + "2.1", "enhanced"), (S + "4.1", "enhanced"), (S + "13.1.3", "enhanced"),
    (S + "88.11", "sr"), (S + "88.22", "sr"), (S + "88.33", "sr"), (S + "88.67", "sr"),
    (S + "88.59", "ko"),
    (S + "66.4", "seg"), (S + "66.5", "seg"), (S + "66.7", "seg"),
    (S + "11.1", "pr"), (S + "11.2", "pr"),
    (S + "481.1", "rt"), (S + "481.2", "rt"), (S + "481.3", "rt"), (S + "481.5", "rt"),
    (S + "104.1", "pdf"),
    (S + "9.1.1", "waveform"), (S + "9.2.1", "waveform"), (S + "9.4.1", "waveform"),
    ("1.2.840.10008.5.1.4.1.1.999.1", "other"), ("1.3.6.1.4.1.5962.301.1", "other"),
    (None, None), ("", None), ("   ", None),
])
def test_object_kind_table(sop_class, expected):
    assert kinds.object_kind(sop_class) == expected


def test_a_single_class_image_with_several_frames_is_multiframe():
    assert kinds.object_kind(CT, frames=1) == "image"
    assert kinds.object_kind(S + "7", frames=12) == "multiframe"


@pytest.mark.parametrize("kind", sorted(EXCLUDED))
def test_every_non_image_kind_is_excluded_and_named(kind):
    sop_class, modality = EXCLUDED[kind]
    assert kinds.excluded_kind(sop_class, modality) == kind
    assert kinds.excluded_kind(sop_class) == kind              # SOP class alone decides


def test_images_are_kept_and_an_unclassifiable_object_is_kept_unless_it_is_sr():
    for sop_class in (CT, S + "2.1", S + "3.1"):
        assert kinds.excluded_kind(sop_class, "CT") is None
    assert kinds.excluded_kind(None, "CT") is None             # no SOP class to judge by
    assert kinds.excluded_kind("", None) is None
    assert kinds.excluded_kind(None, "SR") == "sr"             # the rule that was there before
    assert kinds.excluded_kind(CT, "SR") == "sr"


@pytest.mark.parametrize("modalities, expected", [
    (["CT"], "CT"), (["SR", "CT"], "CT"), (["KO", "PR", "MR", "SR"], "MR"),
    (["SR"], "UNKNOWN"), (["SEG", "RTSTRUCT", "DOC"], "UNKNOWN"), ([], "UNKNOWN"), (None, "UNKNOWN"),
    ([" ct "], "CT"), ([None, "", "US"], "US"),
])
def test_the_modality_hint_is_the_first_image_modality(modalities, expected):
    assert kinds.first_image_modality(modalities) == expected


# --- /trigger-inference-pacs: series assembly from the archive --------------------------------

def qido_instance(sop_uid, series_uid, sop_class, modality):
    item = {"00080018": {"vr": "UI", "Value": [sop_uid]}, "0020000E": {"vr": "UI", "Value": [series_uid]}}
    if sop_class is not None:
        item["00080016"] = {"vr": "UI", "Value": [sop_class]}
    if modality is not None:
        item["00080060"] = {"vr": "CS", "Value": [modality]}
    return item


class FakeResponse:
    def __init__(self, payload=None):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def run_pacs_trigger(instances):
    """Call the route with the archive answering `instances`. Returns (response, fetched UIDs,
    the series handed to processing)."""
    fetched, processed = [], []

    def fake_get(url, headers=None, timeout=None, **_):
        if url.endswith("/instances"):
            return FakeResponse(instances)
        fetched.append(url.rsplit("/", 1)[1])
        return FakeResponse()

    def fake_process(files, source):
        processed.append([item["filename"] for item in files])
        return {"status": "completed", "series_uid": "series-ct", "filename": "x.dcm", "modality": "CT"}

    with patch.object(orchestrator_api.requests, "get", side_effect=fake_get), \
            patch.object(orchestrator_api, "extract_dicom_from_multipart", return_value=b"dicom"), \
            patch.object(orchestrator_api, "extract_full_dicom_metadata",
                         return_value={"Modality": "CT", "SeriesDescription": "AX"}), \
            patch.object(orchestrator_api, "process_series", side_effect=fake_process), \
            patch.object(orchestrator_api, "build_series_metadata_only_result",
                         side_effect=lambda entry, study: {"status": "completed", "series_uid": entry["series_uid"],
                                                           "filename": "m.dcm"}), \
            patch.object(orchestrator_api, "build_study_webhook_payload",
                         return_value={"study_uid": "1.2.3", "dicom_metadata": {"metadata": {"Modality": "CT"}}}), \
            patch.object(orchestrator_api, "publish_result_payload", side_effect=lambda payload: payload), \
            patch.object(orchestrator_api, "derive_delivery_status", return_value={}), \
            patch.object(orchestrator_api, "derive_operation_status", return_value="completed"), \
            patch.object(orchestrator_api, "sb", None):
        response = orchestrator_api.trigger_inference_pacs(study_uid="1.2.3", series_uid=None)
    return response, fetched, processed


CT_INSTANCES = [qido_instance("ct-1", "series-ct", CT, "CT"), qido_instance("ct-2", "series-ct", CT, "CT")]


@pytest.mark.parametrize("kind", sorted(EXCLUDED))
def test_a_non_image_object_is_not_fetched_not_assembled_and_is_listed(kind):
    sop_class, modality = EXCLUDED[kind]
    extra = qido_instance(f"{kind}-1", f"series-{kind}", sop_class, modality)

    response, fetched, processed = run_pacs_trigger(CT_INSTANCES + [extra])

    assert fetched == ["ct-1", "ct-2"]                         # the non-image object is never retrieved
    assert processed == [["ct-1.dcm", "ct-2.dcm"]]             # one series, the images
    assert [item.get("series_uid") for item in response["files"]] == ["series-ct"]
    metadata = response["webhook_payload"]["dicom_metadata"]
    assert metadata["attachments"] == [{"kind": kind, "sop_instance_uid": f"{kind}-1",
                                        "series_uid": f"series-{kind}", "sop_class_uid": sop_class}]
    assert metadata["metadata"] == {"Modality": "CT"}          # what was there is still there


def test_a_non_image_series_cannot_become_the_primary_series():
    """Before this change a segmentation series went through select_primary_series."""
    extras = [qido_instance(f"{kind}-1", f"series-{kind}", *EXCLUDED[kind]) for kind in sorted(EXCLUDED)]
    seen = []
    real = orchestrator_api.select_primary_series

    def spy(entries):
        seen.extend(entry["series_uid"] for entry in entries)
        return real(entries)

    with patch.object(orchestrator_api, "select_primary_series", side_effect=spy):
        response, fetched, _ = run_pacs_trigger(extras + CT_INSTANCES)

    assert seen == ["series-ct"] and fetched == ["ct-1", "ct-2"]
    kinds_listed = [item["kind"] for item in response["webhook_payload"]["dicom_metadata"]["attachments"]]
    assert sorted(kinds_listed) == sorted(EXCLUDED)


def test_a_study_of_images_only_lists_no_attachments():
    response, fetched, _ = run_pacs_trigger(CT_INSTANCES)

    assert fetched == ["ct-1", "ct-2"]
    assert response["webhook_payload"]["dicom_metadata"]["attachments"] == []


def test_a_study_with_no_image_at_all_says_so_instead_of_looking_empty():
    only_reports = [qido_instance("sr-1", "series-sr", *EXCLUDED["sr"]),
                    qido_instance("seg-1", "series-seg", *EXCLUDED["seg"])]

    response, fetched, processed = run_pacs_trigger(only_reports)

    assert fetched == [] and processed == []
    assert "no image objects to analyse: 2 non-image object(s) skipped" in response["errors"]
    assert len(response["webhook_payload"]["dicom_metadata"]["attachments"]) == 2


def test_an_instance_whose_sop_class_the_archive_does_not_report_is_still_assembled():
    """Unchanged behaviour: nothing to classify it by. Recorded as a limit, not hidden."""
    response, fetched, _ = run_pacs_trigger([qido_instance("x-1", "series-ct", None, "CT")])

    assert fetched == ["x-1"]
    assert response["webhook_payload"]["dicom_metadata"]["attachments"] == []


# --- /trigger-inference-multi: series assembly from uploaded files ----------------------------

def dicom_bytes(sop_class, modality, series_uid):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = sop_class
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = "1.2.3", series_uid
    ds.Modality, ds.PatientID = modality, "SYNTH-0001"
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


@pytest.mark.parametrize("kind", sorted(EXCLUDED))
def test_uploaded_non_image_objects_are_left_out_of_the_series_and_listed(kind):
    sop_class, modality = EXCLUDED[kind]
    blobs = [("ct.dcm", dicom_bytes(CT, "CT", "1.2.3.1")), (f"{kind}.dcm", dicom_bytes(sop_class, modality, "1.2.3.2"))]
    attachments = []

    series = orchestrator_api.collect_series_from_upload(blobs, attachments)

    assert [entry["series_uid"] for entry in series] == ["1.2.3.1"]
    assert [entry["instance_count"] for entry in series] == [1]
    assert len(attachments) == 1
    assert attachments[0]["kind"] == kind and attachments[0]["sop_class_uid"] == sop_class
    assert attachments[0]["series_uid"] == "1.2.3.2" and attachments[0]["sop_instance_uid"]


def test_collect_series_still_works_for_callers_that_do_not_ask_for_the_list():
    blobs = [("ct.dcm", dicom_bytes(CT, "CT", "1.2.3.1")), ("sr.dcm", dicom_bytes(S + "88.33", "SR", "1.2.3.2"))]

    assert [entry["series_uid"] for entry in orchestrator_api.collect_series_from_upload(blobs)] == ["1.2.3.1"]


def test_attach_skipped_objects_adds_the_list_without_dropping_existing_metadata():
    payload = {"dicom_metadata": {"metadata_summary": {"a": 1}, "metadata": {"Modality": "CT"}}}
    out = orchestrator_api.attach_skipped_objects(payload, [{"kind": "sr"}])

    assert out["dicom_metadata"] == {"metadata_summary": {"a": 1}, "metadata": {"Modality": "CT"},
                                     "attachments": [{"kind": "sr"}]}
    assert orchestrator_api.attach_skipped_objects({}, [])["dicom_metadata"] == {"attachments": []}
    # and the list survives the reduction applied before a webhook is sent
    reduced = orchestrator_api.build_webhook_observer_payload({**out, "ai_result": {}})
    assert reduced["dicom_metadata"]["attachments"] == [{"kind": "sr"}]


# --- the fetcher ------------------------------------------------------------------------------

def test_the_fetcher_uses_the_helper_and_its_image_carries_it():
    """pacs_fetcher.py runs its polling loop at import, so it cannot be imported here. What
    can be checked: it calls the helper for the instance skip and the modality hint, and its
    image copies object_kind.py. The loop itself is not exercised by any test."""
    source = (ORCHESTRATOR_DIR / "pacs_fetcher.py").read_text(encoding="utf-8")
    dockerfile = (ORCHESTRATOR_DIR / "Dockerfile.pacs_fetcher").read_text(encoding="utf-8")

    assert "from object_kind import excluded_kind, first_image_modality" in source
    assert source.count("excluded_kind(") == 1 and source.count("first_image_modality(") == 1
    assert source.index("excluded_kind(") < source.index('print(f"📥 Downloading')   # skip before download
    assert "object_kind.py" in dockerfile
