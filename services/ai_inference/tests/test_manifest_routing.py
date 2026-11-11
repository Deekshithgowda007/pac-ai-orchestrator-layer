"""Routing through the manifests and the three-rung ladder (SDD-002 T3 / T3b).

Runs without torch: the adapters import their frameworks lazily, and the studies here are
tiny synthetic volumes no model has weights for. So rung 1 never produces a reading, rung 2
declines (no TotalSegmentator, too few slices), and every study must still come back with a
rung-3 description that makes no clinical claim -- never a verdict from pixel statistics.
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
from models.study_description import ANALYSIS_STATUS, DESCRIPTOR_VERSION, MODEL_NAME  # noqa: E402

HEURISTIC_SENTENCES = ("Preliminary automated review", "does not detect a dominant", "abnormal signal burden",
                       "rule-based-fallback")
CLAIM_WORDS = ("no abnormality", "not abnormal", "normal study", "no dominant")
NAME, PID = "SYNTH^ROUTING", "SYNTH-ROUTE-1"


def write_volume(folder: str, n: int = 4, *, modality="CT", sop_class=CTImageStorage, body_part="CHEST",
                 gap_at: int | None = None, **extra) -> None:
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
        ds.PatientName, ds.PatientID, ds.PatientAge = NAME, PID, "040Y"
        if body_part:
            ds.BodyPartExamined = body_part
        ds.SeriesDescription = "AX 1.0"
        ds.SliceThickness = "1.0"
        ds.PixelSpacing = ["0.7", "0.7"]
        ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
        z = i if gap_at is None or i < gap_at else i + 3
        ds.ImagePositionPatient = ["0", "0", str(z)]
        ds.Rows = ds.Columns = 16
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        ds.RescaleIntercept, ds.RescaleSlope = "-1024", "1"
        ds.PixelData = rng.integers(0, 4000, (16, 16), dtype=np.uint16).tobytes()
        for k, v in extra.items():
            setattr(ds, k, v)
        ds.save_as(os.path.join(folder, f"{i:03d}.dcm"), enforce_file_format=True)


NO_WEIGHTS = {"CT_MONAI_CHECKPOINT_PATH": "", "MR_MONAI_CHECKPOINT_PATH": "", "STENUNET_CHECKPOINT_PATH": "",
              "AI_AUTHORIZED_ONLY": "false", "ENABLE_TOTALSEGMENTATOR": "true"}


def assert_no_claim(test: unittest.TestCase, result: dict) -> None:
    test.assertIsNone(result["abnormal"], "absence of a reading must not become 'not abnormal'")
    test.assertIsNone(result["confidence"], "no model ran, so there is no confidence")
    rendered = json.dumps(result).lower()
    for sentence in HEURISTIC_SENTENCES:
        test.assertNotIn(sentence.lower(), rendered, "a heuristic sentence reached the result")
    for words in CLAIM_WORDS:
        test.assertNotIn(words, rendered, f"a clinical claim ({words!r}) reached a descriptive result")
    for value in (NAME, PID, "040Y"):
        test.assertNotIn(value, json.dumps(result))


def assert_descriptive(test: unittest.TestCase, result: dict) -> None:
    assert_no_claim(test, result)
    test.assertEqual(result["model_name"], MODEL_NAME)
    test.assertEqual((result["model_id"], result["model_version"]), (MODEL_NAME, DESCRIPTOR_VERSION))
    test.assertEqual(result["analysis_status"], ANALYSIS_STATUS)
    test.assertEqual(result["diagnostic_support"], "descriptive-only")
    test.assertEqual(result["report_type"], "descriptive-summary")
    test.assertTrue(result["finding"].startswith("Descriptive AI summary only"))
    lines = result["observations"]
    test.assertTrue(any(l.startswith("Technique:") for l in lines), lines)
    test.assertTrue(any(l.startswith("Series inventory:") for l in lines), lines)
    test.assertTrue(any(l.startswith("Image quality:") for l in lines), lines)
    test.assertTrue(any(l.startswith("AI models considered:") for l in lines), lines)
    test.assertIsInstance(result["ladder"], list)
    test.assertEqual(result["routing_decision"]["route_name"], "study-descriptor")


def ladder_outcomes(result: dict) -> dict:
    return {row["route_name"]: row["outcome"] for row in result["ladder"]}


class DuplicatesStayInStep(unittest.TestCase):
    def test_object_kind_copy_identical_below_docstring(self):
        here = AI_INFERENCE_DIR / "models" / "object_kind.py"
        orchestrator = AI_INFERENCE_DIR.parent / "orchestrator" / "object_kind.py"
        if not orchestrator.exists():
            self.skipTest("orchestrator copy not in this checkout")
        body = lambda p: p.read_text(encoding="utf-8").split("from __future__", 1)[1]
        self.assertEqual(body(here), body(orchestrator))


class Ladder(unittest.TestCase):
    def setUp(self):
        model_registry.reset_cache()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()
        model_registry.reset_cache()

    def run_engine(self, **env):
        with patch.dict(os.environ, {**NO_WEIGHTS, **env}):
            model_registry.reset_cache()
            return InferenceEngine().run(self.tmp.name)

    def test_context_carries_a_profile_from_every_header(self):
        write_volume(self.tmp.name)
        context = collect_study_context(self.tmp.name)
        self.assertEqual((context.profile["modality"], context.profile["protocol_family"], context.profile["dimensionality"]),
                         ("CT", "ct_chest", "3d"))
        self.assertEqual(context.profile["series"][0]["instances"], 4)
        self.assertNotIn("SYNTH", json.dumps(context.profile))

    def test_chest_ct_without_weights_gets_a_description_not_a_verdict(self):
        """The refusal this ladder exists for: before, this study came back 'not abnormal, 0.9'."""
        write_volume(self.tmp.name)
        result = self.run_engine()
        assert_descriptive(self, result)
        outcomes = ladder_outcomes(result)
        self.assertEqual(outcomes["ct-monai-screening"], "weights_unavailable")
        self.assertEqual(outcomes["ct-anatomy-segmentation"], "declined", "4 slices, no TotalSegmentator: rung 2 declines")
        self.assertEqual(outcomes["heuristic-fallback"], "model_disabled")
        self.assertEqual(outcomes["chest-radiograph-screening"], "mismatch")
        considered = [l for l in result["observations"] if l.startswith("AI models considered:")]
        self.assertTrue(any("monai-lung-nodule-retinanet@1.0.0 - not run: weights not present" in l for l in considered), considered)

    def test_description_content(self):
        write_volume(self.tmp.name, n=6, gap_at=3)
        result = self.run_engine()
        technique = next(l for l in result["observations"] if l.startswith("Technique:"))
        self.assertIn("chest CT", technique)
        self.assertIn("volumetric stack", technique)
        self.assertIn("1 image series, 6 instance(s)", technique)
        self.assertIn("slice thickness 1.0 mm", technique)
        self.assertIn("adult patient", technique)
        self.assertIn("contrast not stated in the header", technique)
        inventory = next(l for l in result["observations"] if l.startswith("Series inventory:"))
        self.assertIn("AX 1.0 - 6 image(s), CT, Explicit VR LE", inventory)
        quality = [l for l in result["observations"] if l.startswith("Image quality:")]
        self.assertTrue(any("1 gap(s) in slice positions" in l for l in quality), quality)
        self.assertIn("noise_estimate", result["metrics"])

    def test_unknown_modality_is_still_described(self):
        write_volume(self.tmp.name, modality="", body_part=None)
        result = self.run_engine()
        assert_descriptive(self, result)
        self.assertTrue(all(o in {"incomplete", "model_disabled", "mismatch"} for r, o in ladder_outcomes(result).items() if r != "registry"))
        self.assertEqual(ladder_outcomes(result)["registry"], "profile_incomplete")

    def test_other_modality_is_described_and_names_the_disabled_catch_all(self):
        write_volume(self.tmp.name, modality="OT", sop_class=SecondaryCaptureImageStorage, body_part=None)
        result = self.run_engine()
        assert_descriptive(self, result)
        self.assertEqual(ladder_outcomes(result)["heuristic-fallback"], "model_disabled")
        self.assertEqual(ladder_outcomes(result)["registry"], "model_disabled")

    def test_missing_path_and_empty_folder_are_unavailable(self):
        result = InferenceEngine().run("/nowhere/at/all")
        assert_no_claim(self, result)
        self.assertEqual(result["routing_decision"]["no_match_reason"], "input_missing")
        result = InferenceEngine().run(self.tmp.name)
        assert_no_claim(self, result)
        self.assertEqual(result["routing_decision"]["no_match_reason"], "no_dicom_files")

    def test_heuristic_runs_only_when_a_deployment_enables_it_and_is_then_the_primary(self):
        write_volume(self.tmp.name)
        lock = pathlib.Path(self.tmp.name) / "lock.json"
        lock.write_text(json.dumps({"schema": mr.LOCK_SCHEMA,
                                    "enabled": [{"model_id": "heuristic-fallback", "version": "1.0.0"}]}))
        result = self.run_engine(AI_REGISTRY_LOCK_PATH=str(lock))
        self.assertEqual(result["routing_decision"]["route_name"], "heuristic-fallback")
        self.assertEqual((result["model_id"], result["tier"]), ("heuristic-fallback", "T3"))
        self.assertIsNotNone(result["confidence"], "enabled on purpose, the heuristic does run")
        self.assertEqual(ladder_outcomes(result)["heuristic-fallback"], "ran")
        # rung 3 still rides along as a supporting result
        self.assertEqual(result["supporting_results"][-1]["model_name"], MODEL_NAME)

    def test_authorized_only_with_no_t1_still_describes(self):
        write_volume(self.tmp.name)
        result = self.run_engine(AI_AUTHORIZED_ONLY="true")
        assert_descriptive(self, result)
        self.assertEqual(ladder_outcomes(result)["ct-anatomy-segmentation"], "not_authorized")
        self.assertEqual(ladder_outcomes(result)["registry"], "not_authorized")

    def test_context_without_profile_is_unavailable(self):
        context = StudyContext(dicom_path="x", files=[], metadata={}, modality="CT", body_part="CHEST",
                               series_uid="s", study_uid="u", profile=None)
        with patch.dict(os.environ, NO_WEIGHTS):
            result = AdapterRegistry().analyze(context)
        assert_no_claim(self, result)
        self.assertEqual(result["routing_decision"]["route_name"], "study-descriptor")
        self.assertTrue(any("pixel data could not be read" in l for l in result["observations"]))

    def test_primary_reading_is_stamped_and_keeps_the_description(self):
        """A model that does run is the result; the description rides along, and the stamp names the manifest."""
        write_volume(self.tmp.name)
        registry = AdapterRegistry()
        fake = registry.adapters["ct_monai"]
        with patch.object(type(fake), "supports", return_value=True), \
             patch.object(type(fake), "analyze", return_value={"model_name": "monai-lung-nodule-retinanet", "finding": "one 6 mm nodule candidate",
                                                               "abnormal": True, "confidence": 0.71, "observations": ["one 6 mm nodule candidate"],
                                                               "diagnostic_support": "screening-only", "report_type": "preliminary-lung-nodule-screening"}), \
             tempfile.NamedTemporaryFile(suffix=".pt") as weights, \
             patch.dict(os.environ, {**NO_WEIGHTS, "CT_MONAI_CHECKPOINT_PATH": weights.name}):
            model_registry.reset_cache()
            result = registry.analyze(collect_study_context(self.tmp.name))
        self.assertEqual(result["routing_decision"]["route_name"], "ct-monai-screening")
        self.assertEqual((result["model_id"], result["model_version"], result["tier"]), ("monai-lung-nodule-retinanet", "1.0.0", "T3"))
        self.assertEqual(result["confidence"], 0.71)
        self.assertEqual(ladder_outcomes(result)["ct-monai-screening"], "ran")
        self.assertEqual(result["supporting_results"][-1]["model_name"], MODEL_NAME)


if __name__ == "__main__":
    unittest.main()
