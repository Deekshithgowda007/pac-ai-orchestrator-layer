"""Routing through the manifests; the heuristic fallback is no longer on the path (SDD-002 T3).

Runs without torch: the adapters import their frameworks lazily, and the studies here are
tiny synthetic CT volumes no model has weights for, so every route ends in "unavailable".
That is the point -- the result must SAY so instead of producing a verdict from pixels.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

# The only eager framework use in the adapters is `__import__("torch").cuda.is_available()` for
# device selection; a CPU test box has no GPU to choose, so skip it and keep torch out of the suite.
os.environ.setdefault("ENABLE_GPU_INFERENCE", "false")

AI_INFERENCE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(AI_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_INFERENCE_DIR))

from inference_engine import InferenceEngine  # noqa: E402
from models import manifest_registry as mr  # noqa: E402
from models import model_registry  # noqa: E402
from models.modality_adapters import AdapterRegistry  # noqa: E402
from models.study_context import StudyContext, collect_study_context  # noqa: E402

HEURISTIC_SENTENCES = ("Preliminary automated review", "does not detect a dominant", "abnormal signal burden")


def write_volume(folder: str, n: int = 4, *, modality="CT", sop_class=CTImageStorage, body_part="CHEST", **extra) -> None:
    study_uid, series_uid = generate_uid(), generate_uid()
    rng = np.random.default_rng(7)
    for i in range(n):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = sop_class
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.SOPClassUID, ds.SOPInstanceUID = sop_class, ds.file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.Modality = modality
        ds.PatientName, ds.PatientID = "SYNTH^ROUTING", "SYNTH-ROUTE-1"
        ds.PatientAge = "040Y"
        if body_part:
            ds.BodyPartExamined = body_part
        ds.SeriesDescription = "AX 1.0"
        ds.SliceThickness = "1.0"
        ds.PixelSpacing = ["0.7", "0.7"]
        ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
        ds.ImagePositionPatient = ["0", "0", str(i)]
        ds.Rows = ds.Columns = 16
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        ds.RescaleIntercept, ds.RescaleSlope = "-1024", "1"
        ds.PixelData = rng.integers(0, 4000, (16, 16), dtype=np.uint16).tobytes()
        for k, v in extra.items():
            setattr(ds, k, v)
        ds.save_as(os.path.join(folder, f"{i:03d}.dcm"), enforce_file_format=True)


NO_WEIGHTS = {"CT_MONAI_CHECKPOINT_PATH": "", "MR_MONAI_CHECKPOINT_PATH": "", "STENUNET_CHECKPOINT_PATH": "",
              "AI_AUTHORIZED_ONLY": "false"}


def assert_unavailable(test: unittest.TestCase, result: dict) -> None:
    test.assertIsNone(result["abnormal"], "absence of a reading must not become 'not abnormal'")
    test.assertIsNone(result["confidence"], "no model ran, so there is no confidence")
    test.assertEqual(result["analysis_status"], "unsupported-for-diagnostic-inference")
    test.assertEqual(result["model_name"], "none")
    rendered = json.dumps(result)
    for sentence in HEURISTIC_SENTENCES:
        test.assertNotIn(sentence, rendered, "a heuristic sentence reached the result")


class DuplicatesStayInStep(unittest.TestCase):
    def test_object_kind_copy_identical_below_docstring(self):
        here = AI_INFERENCE_DIR / "models" / "object_kind.py"
        orchestrator = AI_INFERENCE_DIR.parent / "orchestrator" / "object_kind.py"
        if not orchestrator.exists():
            self.skipTest("orchestrator copy not in this checkout")
        body = lambda p: p.read_text(encoding="utf-8").split("from __future__", 1)[1]
        self.assertEqual(body(here), body(orchestrator))


class DefaultLockRouting(unittest.TestCase):
    def setUp(self):
        model_registry.reset_cache()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()
        model_registry.reset_cache()

    def test_context_carries_a_profile_from_every_header(self):
        write_volume(self.tmp.name)
        context = collect_study_context(self.tmp.name)
        self.assertEqual(context.profile["modality"], "CT")
        self.assertEqual(context.profile["protocol_family"], "ct_chest")
        self.assertEqual(context.profile["dimensionality"], "3d")
        self.assertEqual(context.profile["series"][0]["instances"], 4)
        self.assertEqual(context.profile["patient_age_band"], "adult")
        self.assertNotIn("SYNTH", json.dumps(context.profile))

    def test_chest_ct_without_weights_is_unavailable_not_a_heuristic_verdict(self):
        """The refusal T3 exists for: before, this study got abnormal/not-abnormal with a
        confidence of 0.35-0.90 from voxel statistics."""
        write_volume(self.tmp.name)
        with patch.dict(os.environ, NO_WEIGHTS):
            result = InferenceEngine().run(self.tmp.name)
        assert_unavailable(self, result)
        decision = result["routing_decision"]
        self.assertIn(decision["route_name"], {"no-matching-manifest", "adapter-failed"})
        self.assertIn(decision["no_match_reason"], {"weights_unavailable", "adapter_failed"})
        self.assertTrue(result["limitations"], "the reason(s) are listed, not hidden")

    def test_unknown_modality_is_profile_incomplete(self):
        write_volume(self.tmp.name, modality="", body_part=None)
        with patch.dict(os.environ, NO_WEIGHTS):
            result = InferenceEngine().run(self.tmp.name)
        assert_unavailable(self, result)
        self.assertEqual(result["routing_decision"]["no_match_reason"], "profile_incomplete")

    def test_other_modality_names_the_disabled_catch_all(self):
        write_volume(self.tmp.name, modality="OT", sop_class=SecondaryCaptureImageStorage, body_part=None)
        with patch.dict(os.environ, NO_WEIGHTS):
            result = InferenceEngine().run(self.tmp.name)
        assert_unavailable(self, result)
        self.assertEqual(result["routing_decision"]["no_match_reason"], "model_disabled")
        self.assertIn("heuristic-fallback", result["routing_decision"]["reason"] + json.dumps(result["limitations"]))

    def test_missing_path_and_empty_folder(self):
        result = InferenceEngine().run("/nowhere/at/all")
        assert_unavailable(self, result)
        self.assertEqual(result["routing_decision"]["no_match_reason"], "input_missing")
        result = InferenceEngine().run(self.tmp.name)
        assert_unavailable(self, result)
        self.assertEqual(result["routing_decision"]["no_match_reason"], "no_dicom_files")

    def test_context_without_profile_is_incomplete(self):
        context = StudyContext(dicom_path="x", files=[], metadata={}, modality="CT", body_part="CHEST",
                               series_uid="s", study_uid="u", profile=None)
        result = AdapterRegistry().analyze(context)
        assert_unavailable(self, result)
        self.assertEqual(result["routing_decision"]["no_match_reason"], "profile_incomplete")

    def test_heuristic_runs_only_when_a_deployment_enables_it(self):
        write_volume(self.tmp.name)
        lock = pathlib.Path(self.tmp.name) / "lock.json"
        lock.write_text(json.dumps({"schema": mr.LOCK_SCHEMA,
                                    "enabled": [{"model_id": "heuristic-fallback", "version": "1.0.0"}]}))
        with patch.dict(os.environ, {**NO_WEIGHTS, "AI_REGISTRY_LOCK_PATH": str(lock)}):
            model_registry.reset_cache()
            result = InferenceEngine().run(self.tmp.name)
        self.assertEqual(result["routing_decision"]["route_name"], "heuristic-fallback")
        self.assertIsNotNone(result["confidence"], "enabled on purpose, the heuristic does run")
        self.assertIn("heuristic-fallback", result["routing_decision"]["reason"])

    def test_authorized_only_with_no_t1_is_radiologist_only(self):
        write_volume(self.tmp.name)
        with patch.dict(os.environ, {**NO_WEIGHTS, "AI_AUTHORIZED_ONLY": "true"}):
            result = InferenceEngine().run(self.tmp.name)
        assert_unavailable(self, result)
        self.assertEqual(result["routing_decision"]["no_match_reason"], "not_authorized")


if __name__ == "__main__":
    unittest.main()
