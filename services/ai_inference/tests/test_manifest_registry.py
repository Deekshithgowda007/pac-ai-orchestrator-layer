"""Manifests, lock and matching rule (SDD-SCANSURE-AI-002 T2).

Dependency-free on purpose: imports only models.manifest_registry / models.profile_vocab /
models.model_registry, never an adapter, so the suite runs without torch.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

AI_INFERENCE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(AI_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_INFERENCE_DIR))

from models import manifest_registry as mr  # noqa: E402
from models import model_registry  # noqa: E402
from models import profile_vocab as vocab  # noqa: E402

COMMITTED = mr.load_registry()


def profile(**over):
    base = {
        "schema": vocab.PROFILE_SCHEMA, "modality": "CT",
        "body_part": {"raw": "CHEST", "normalized": "chest", "source": "BodyPartExamined"},
        "dimensionality": "3d", "protocol_family": "ct_chest", "contrast": None, "laterality": None,
        "patient_age_band": "adult", "patient_sex": None, "series": [], "excluded_objects": [],
        "derivation": {"version": "1.0.0", "warnings": []},
    }
    base.update(over)
    return base


def manifest(**over):
    base = {
        "schema": mr.MANIFEST_SCHEMA, "model_id": "m", "version": "1.0.0", "route_name": "r", "display_name": "M", "tier": "T3",
        "intended_use": {"modalities": ["CT"], "protocol_families": ["ct_chest"], "dimensionality": ["3d"],
                         "age_bands": "any", "body_parts": "any", "contrast": "any"},
        "license": {"code": "Apache-2.0", "authorization": None, "dpa_reference": None},
        "outputs": {"kind": "classification", "fields": ["abnormal"], "produces_seg": False, "produces_sr": True},
        "runtime": {"adapter": "x", "weights_env": None, "weights_required": False, "weights_sha256": None, "device": "cpu", "timeout_s": 60},
        "labels": {"cdsco_text": None, "screening_only": True},
    }
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def registry(*manifests, enabled=None):
    manifests = [mr.validate_manifest(m) for m in manifests]
    if enabled is None:
        enabled = manifests
    return mr.Registry(manifests=list(manifests), enabled=list(enabled))


class CommittedManifests(unittest.TestCase):
    def test_sixteen_manifests_validate_and_nine_are_enabled(self):
        # 10 from T2 + 6 rung-1 candidates (CANDIDATES.md), all six disabled; pinned in detail by
        # tests/test_candidate_manifests.py
        self.assertEqual(len(COMMITTED.manifests), 16)
        self.assertEqual(len(COMMITTED.enabled), 9)
        self.assertNotIn("heuristic-fallback", [m["route_name"] for m in COMMITTED.enabled], "D2: disabled by default")
        self.assertIsNotNone(COMMITTED.by_route("heuristic-fallback"), "but it exists, so a deployment may enable it knowingly")

    def test_every_legacy_route_has_a_manifest(self):
        for route in ("ct-monai-screening", "mr-brats-segmentation", "us-echo-lite-lv-function", "us-hf-2d-screening",
                      "xa-stenosis-screening", "chest-radiograph-screening", "hf-2d-screening",
                      "ct-anatomy-segmentation", "mr-anatomy-segmentation"):
            self.assertIsNotNone(COMMITTED.by_route(route), route)

    def test_no_committed_manifest_is_t1_or_claims_more_than_screening(self):
        for m in COMMITTED.manifests:
            self.assertEqual(m["tier"], "T3", m["model_id"])
            self.assertIsNone(m["license"]["authorization"])
            self.assertTrue(m["labels"]["screening_only"])
            self.assertIsNone(m["labels"]["cdsco_text"], "D5: no label text until the lawyer's wording")

    def test_vocabulary_matches_the_profiler(self):
        # The monorepo profiler is the source of these sets; keep the duplicate honest.
        self.assertIn("ct_chest", vocab.PROTOCOL_FAMILIES)
        self.assertIn("unknown", vocab.PROTOCOL_FAMILIES)
        self.assertEqual(vocab.DIMENSIONALITIES, {"2d", "2d_multiframe", "3d", "4d", "cine", "unknown"})
        self.assertEqual(vocab.AGE_BANDS, {"neonate", "child", "adolescent", "adult", "unknown"})


class Validation(unittest.TestCase):
    def test_unknown_top_level_key_refused(self):
        with self.assertRaisesRegex(mr.ManifestError, "unknown keys"):
            mr.validate_manifest(manifest(intended_us={}))

    def test_vocabulary_outside_profile_refused(self):
        with self.assertRaisesRegex(mr.ManifestError, "not in the StudyProfile vocabulary"):
            mr.validate_manifest(manifest(intended_use={"protocol_families": ["ct_thorax"]}))

    def test_unknown_may_never_be_accepted(self):
        with self.assertRaisesRegex(mr.ManifestError, '"unknown" may never'):
            mr.validate_manifest(manifest(intended_use={"protocol_families": ["unknown"]}))

    def test_empty_constraint_list_refused(self):
        with self.assertRaisesRegex(mr.ManifestError, "non-empty list"):
            mr.validate_manifest(manifest(intended_use={"modalities": []}))

    def test_t1_requires_authorization(self):
        with self.assertRaisesRegex(mr.ManifestError, "T1 manifest must carry"):
            mr.validate_manifest(manifest(tier="T1"))
        ok = manifest(tier="T1", license={"authorization": {"jurisdiction": "IN", "authority": "CDSCO", "reference": "X", "valid_until": "2030-01-01"}})
        self.assertEqual(mr.validate_manifest(ok)["tier"], "T1")

    def test_non_t1_cannot_claim_more_than_screening(self):
        with self.assertRaisesRegex(mr.ManifestError, "only a T1 manifest"):
            mr.validate_manifest(manifest(labels={"screening_only": False}))

    def test_bad_date_refused(self):
        with self.assertRaisesRegex(mr.ManifestError, "ISO date"):
            mr.validate_manifest(manifest(tier="T1", license={"authorization": {"jurisdiction": "IN", "authority": "CDSCO", "reference": "X", "valid_until": "soon"}}))

    def test_loader_refuses_misfiled_and_duplicate_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "m").mkdir()
            (root / "m" / "2.0.0.json").write_text(json.dumps(manifest()))   # version says 1.0.0
            with self.assertRaisesRegex(mr.ManifestError, "named <version>.json"):
                mr.load_manifests(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "m").mkdir(); (root / "other").mkdir()
            (root / "m" / "1.0.0.json").write_text(json.dumps(manifest()))
            (root / "other" / "1.0.0.json").write_text(json.dumps(manifest(model_id="other")))   # same route_name "r"
            with self.assertRaisesRegex(mr.ManifestError, "route_name 'r' already used"):
                mr.load_manifests(root)

    def test_lock_naming_an_absent_manifest_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "m").mkdir()
            (root / "m" / "1.0.0.json").write_text(json.dumps(manifest()))
            lock = root / "lock.json"
            lock.write_text(json.dumps({"schema": mr.LOCK_SCHEMA, "enabled": [{"model_id": "ghost", "version": "1.0.0"}]}))
            with self.assertRaisesRegex(mr.ManifestError, "has no manifest"):
                mr.load_registry(root, lock)
            lock.write_text(json.dumps({"schema": mr.LOCK_SCHEMA, "enabled": []}))
            self.assertEqual(mr.load_registry(root, lock).enabled, [], "an empty lock is legal: radiologist-only")


class Matching(unittest.TestCase):
    def test_match_and_shape(self):
        result = mr.select_models(profile(), registry(manifest()), authorized_only=False)
        self.assertIsInstance(result, list)
        self.assertEqual((result[0].route_name, result[0].tier, result[0].weights_present), ("r", "T3", False))

    def test_unknown_never_matches_a_constraint(self):
        reg = registry(manifest())
        for field, value in (("protocol_family", "unknown"), ("dimensionality", "unknown"), ("modality", None), ("modality", "MIXED")):
            result = mr.select_models(profile(**{field: value}), reg, authorized_only=False)
            self.assertIsInstance(result, mr.NoMatch, field)
            self.assertEqual(result.reason, "profile_incomplete", field)
            self.assertIn(field, result.detail)

    def test_any_leaves_a_dimension_open_even_when_unknown(self):
        reg = registry(manifest(intended_use={"protocol_families": "any", "dimensionality": "any"}))
        result = mr.select_models(profile(protocol_family="unknown", dimensionality="unknown"), reg, authorized_only=False)
        self.assertIsInstance(result, list)

    def test_mismatch_is_no_manifest_for_profile(self):
        result = mr.select_models(profile(protocol_family="ct_head"), registry(manifest()), authorized_only=False)
        self.assertEqual(result.reason, "no_manifest_for_profile")

    def test_body_part_read_from_normalized(self):
        reg = registry(manifest(intended_use={"body_parts": ["chest"]}))
        self.assertIsInstance(mr.select_models(profile(), reg, authorized_only=False), list)
        p = profile(body_part={"raw": None, "normalized": None, "source": None})
        self.assertEqual(mr.select_models(p, reg, authorized_only=False).reason, "profile_incomplete")

    def test_contrast_constraint(self):
        reg = registry(manifest(intended_use={"contrast": False}))
        self.assertEqual(mr.select_models(profile(contrast=None), reg, authorized_only=False).reason, "profile_incomplete")
        self.assertEqual(mr.select_models(profile(contrast=True), reg, authorized_only=False).reason, "no_manifest_for_profile")
        self.assertIsInstance(mr.select_models(profile(contrast=False), reg, authorized_only=False), list)

    def test_disabled_manifest_is_model_disabled(self):
        m = manifest()
        result = mr.select_models(profile(), registry(m, enabled=[]), authorized_only=False)
        self.assertEqual((result.reason, result.candidates), ("model_disabled", ("r",)))

    def test_authorized_only_rejects_t3_and_expired_t1(self):
        t3 = manifest()
        t1_expired = manifest(model_id="v", route_name="v", tier="T1",
                              license={"authorization": {"jurisdiction": "IN", "authority": "CDSCO", "reference": "A", "valid_until": "2020-01-01"}})
        t1_ok = manifest(model_id="w", route_name="w", tier="T1",
                         license={"authorization": {"jurisdiction": "IN", "authority": "CDSCO", "reference": "B", "valid_until": "2030-01-01"}})
        result = mr.select_models(profile(), registry(t3, t1_expired), authorized_only=True, today=date(2026, 11, 11))
        self.assertEqual(result.reason, "not_authorized")
        result = mr.select_models(profile(), registry(t3, t1_expired, t1_ok), authorized_only=True, today=date(2026, 11, 11))
        self.assertEqual([m.route_name for m in result], ["w"])
        # off: all three match, T1 first, then lock order
        result = mr.select_models(profile(), registry(t3, t1_expired, t1_ok), authorized_only=False)
        self.assertEqual([m.route_name for m in result], ["v", "w", "r"])

    def test_authorized_only_read_from_env(self):
        with patch.dict("os.environ", {"AI_AUTHORIZED_ONLY": "true"}):
            self.assertTrue(mr.authorized_only_enabled())
            self.assertEqual(mr.select_models(profile(), registry(manifest())).reason, "not_authorized")
        with patch.dict("os.environ", {"AI_AUTHORIZED_ONLY": "false"}):
            self.assertIsInstance(mr.select_models(profile(), registry(manifest())), list)

    def test_weights_required_but_absent(self):
        m = manifest(runtime={"weights_env": "T2_TEST_WEIGHTS", "weights_required": True})
        with patch.dict("os.environ", {"T2_TEST_WEIGHTS": "/nowhere/weights.pt"}):
            result = mr.select_models(profile(), registry(m), authorized_only=False)
        self.assertEqual((result.reason, result.candidates), ("weights_unavailable", ("r",)))
        with tempfile.NamedTemporaryFile(suffix=".pt") as weights, patch.dict("os.environ", {"T2_TEST_WEIGHTS": weights.name}):
            result = mr.select_models(profile(), registry(m), authorized_only=False)
        self.assertEqual((result[0].weights_present, result[0].weights_path), (True, weights.name))

    def test_no_profile_is_incomplete(self):
        self.assertEqual(mr.select_models({}, registry(manifest()), authorized_only=False).reason, "profile_incomplete")
        self.assertEqual(mr.select_models(None, registry(manifest()), authorized_only=False).reason, "profile_incomplete")

    def test_unknown_nomatch_reason_cannot_be_constructed(self):
        with self.assertRaises(ValueError):
            mr.NoMatch("because", "x")


class CommittedRouting(unittest.TestCase):
    """What the committed manifests do with real-shaped profiles (authorized-only off, weights absent)."""

    def route(self, **over):
        with patch.dict("os.environ", {"CT_MONAI_CHECKPOINT_PATH": "", "MR_MONAI_CHECKPOINT_PATH": "", "STENUNET_CHECKPOINT_PATH": ""}):
            return mr.select_models(profile(**over), COMMITTED, authorized_only=False)

    def test_chest_ct_volume_without_weights_still_has_anatomy(self):
        result = self.route()
        self.assertEqual([m.route_name for m in result], ["ct-anatomy-segmentation"], "lung-nodule needs its checkpoint")

    def test_adult_chest_radiograph(self):
        result = self.route(modality="CR", protocol_family="cr_chest", dimensionality="2d")
        self.assertEqual([m.route_name for m in result], ["chest-radiograph-screening", "hf-2d-screening"])

    def test_child_chest_radiograph_is_not_screened_by_the_adult_model(self):
        result = self.route(modality="CR", protocol_family="cr_chest", dimensionality="2d", patient_age_band="child")
        self.assertNotIn("chest-radiograph-screening", [m.route_name for m in result])

    def test_mammogram_is_not_a_chest_radiograph(self):
        result = self.route(modality="MG", protocol_family="dx_mammo", dimensionality="2d",
                            body_part={"raw": "BREAST", "normalized": "breast", "source": "BodyPartExamined"})
        self.assertNotIn("chest-radiograph-screening", [m.route_name for m in result])

    def test_obstetric_ultrasound_has_no_model(self):
        result = self.route(modality="US", protocol_family="us_obstetric", dimensionality="2d",
                            body_part={"raw": None, "normalized": None, "source": None})
        # only the two generic image-to-text routes accept it; with those disabled the study is radiologist-only.
        # No manifest names us_obstetric, which is what the PC-PNDT rule (Phase 4) will rely on.
        self.assertEqual([m.route_name for m in result], ["us-hf-2d-screening", "hf-2d-screening"])
        self.assertFalse(any("us_obstetric" in (m["intended_use"]["protocol_families"] or []) for m in COMMITTED.manifests
                             if m["intended_use"]["protocol_families"] != "any"))

    def test_brain_mr_without_all_sequences_is_structural_and_gets_anatomy_only(self):
        result = self.route(modality="MR", protocol_family="mr_brain_structural",
                            body_part={"raw": "BRAIN", "normalized": "head", "source": "BodyPartExamined"})
        self.assertEqual([m.route_name for m in result], ["mr-anatomy-segmentation"])

    def test_unknown_everything_is_profile_incomplete_not_a_heuristic(self):
        result = self.route(modality=None, protocol_family="unknown", dimensionality="unknown",
                            body_part={"raw": None, "normalized": None, "source": None}, patient_age_band="unknown")
        self.assertIsInstance(result, mr.NoMatch)
        self.assertEqual(result.reason, "profile_incomplete")

    def test_cbct_has_no_model(self):
        result = self.route(protocol_family="cbct_dental", body_part={"raw": "JAW", "normalized": "jaw", "source": "BodyPartExamined"})
        self.assertEqual([m.route_name for m in result], ["ct-anatomy-segmentation"])


class LegacyShape(unittest.TestCase):
    def test_get_model_entry_keeps_its_keys(self):
        entry = model_registry.get_model_entry("chest-radiograph-screening")
        for key in ("route_name", "modalities", "body_parts", "model_name", "model_family", "diagnostic_support", "report_type",
                    "diagnostic_available", "weights_required", "weights_env", "weights_path", "weights_present", "runtime", "notes"):
            self.assertIn(key, entry)
        self.assertEqual(entry["model_name"], "torchxrayvision-densenet121")
        self.assertEqual(entry["body_parts"], ["CHEST", "LUNG"])
        self.assertEqual((entry["model_id"], entry["model_version"], entry["tier"]), ("torchxrayvision-densenet121", "1.0.0", "T3"))

    def test_unsupported_and_unknown_routes(self):
        self.assertEqual(model_registry.get_model_entry("unsupported-modalities")["report_type"], "manual-review-required")
        self.assertEqual(model_registry.get_model_entry("nope")["notes"], "No model registry entry found for this route.")

    def test_weights_required_route_is_not_diagnostic_available_without_weights(self):
        with patch.dict("os.environ", {"CT_MONAI_CHECKPOINT_PATH": "/nowhere.pt"}):
            self.assertFalse(model_registry.get_model_entry("ct-monai-screening")["diagnostic_available"])

    def test_model_registry_list_iterates_every_manifest_plus_unsupported(self):
        # 16 manifests (10 from T2 + 6 disabled rung-1 candidates) + the synthetic unsupported entry
        self.assertEqual(len(list(model_registry.MODEL_REGISTRY)), 17)


if __name__ == "__main__":
    unittest.main()
