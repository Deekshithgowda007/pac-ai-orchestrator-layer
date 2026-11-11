"""Candidate manifests are committed but may not be enabled before they have earned it.

SDD-SCANSURE-AI-002 rung-1 widening (see models/manifests/CANDIDATES.md). Three guards, each of
which REFUSES a lock that enables a candidate too early -- and each is shown refusing, because a
guard that has never failed is of unknown value (CLAUDE.md, BUG-025 rule):

* an enabled manifest must name an adapter the image actually has;
* an enabled manifest outside the grandfathered nine must carry a validation report, a reviewed
  licence (never the ``UNVERIFIED`` sentinel) and a weights checksum;
* routing must now say ``model_disabled`` for a study a candidate would accept, and still say
  ``no_manifest_for_profile`` where no candidate exists -- the widening changes the first and must
  not change the second.
"""
from __future__ import annotations

import os
import pathlib
import sys
import unittest

os.environ.setdefault("ENABLE_GPU_INFERENCE", "false")

AI_INFERENCE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(AI_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_INFERENCE_DIR))

from models import manifest_registry as mr  # noqa: E402
from models import profile_vocab as vocab  # noqa: E402
from models.modality_adapters import AdapterRegistry  # noqa: E402

COMMITTED = mr.load_registry()

#: The routes that were enabled before the widening. They predate the validation-report rule and
#: are NOT exempt from it forever: each is to receive a report (CANDIDATES.md). Nothing may be
#: added to this set -- a new model earns its lock entry through the rule, not through this list.
GRANDFATHERED_ROUTES = frozenset({
    "ct-monai-screening", "mr-brats-segmentation", "us-echo-lite-lv-function", "us-hf-2d-screening",
    "xa-stenosis-screening", "chest-radiograph-screening", "hf-2d-screening",
    "ct-anatomy-segmentation", "mr-anatomy-segmentation",
    "heuristic-fallback",   # disabled by D2 but enable-able knowingly; not a model, has no report
})

CANDIDATES = {
    "rsna-ich-ct-head": "ct-head-haemorrhage-screening",
    "mrnet-knee-mr": "mr-knee-abnormality-screening",
    "mura-msk-radiograph": "cr-msk-abnormality-screening",
    "monai-vista3d": "ct-anatomy-vista3d",
    "toothfairy-cbct-dental": "cbct-dental-structure-segmentation",
    "medgemma-4b-it": "vlm-radiograph-description",
}

UNVERIFIED = "UNVERIFIED"


def enabled_without_adapter(registry: mr.Registry, adapters) -> list:
    return sorted(m["model_id"] for m in registry.enabled if m["runtime"]["adapter"] not in adapters)


def enabled_before_earned(registry: mr.Registry) -> dict:
    """model_id -> the reasons it may not be enabled yet. Empty when the lock is legitimate."""
    problems = {}
    for m in registry.enabled:
        if m["route_name"] in GRANDFATHERED_ROUTES:
            continue
        why = []
        if not m["validation"].get("report"):
            why.append("no validation report")
        if m["license"]["code"] == UNVERIFIED:
            why.append("licence not reviewed")
        if m["runtime"].get("weights_required") and not m["runtime"].get("weights_sha256"):
            why.append("no weights checksum")
        if why:
            problems[m["model_id"]] = why
    return problems


def profile(**over):
    base = {
        "schema": vocab.PROFILE_SCHEMA, "modality": "CT",
        "body_part": {"raw": "HEAD", "normalized": "head", "source": "BodyPartExamined"},
        "dimensionality": "3d", "protocol_family": "ct_head", "contrast": False, "laterality": None,
        "patient_age_band": "adult", "patient_sex": None, "series": [], "excluded_objects": [],
        "derivation": {"version": "1.0.0", "warnings": []},
    }
    base.update(over)
    return base


def with_enabled(*model_ids: str) -> mr.Registry:
    extra = [m for m in COMMITTED.manifests if m["model_id"] in model_ids]
    assert len(extra) == len(model_ids), model_ids
    return mr.Registry(manifests=list(COMMITTED.manifests), enabled=list(COMMITTED.enabled) + extra,
                       lock_path="<test>", manifest_dir=COMMITTED.manifest_dir)


class CommittedState(unittest.TestCase):
    def test_sixteen_manifests_nine_enabled_six_candidates_disabled(self):
        self.assertEqual(len(COMMITTED.manifests), 16)
        self.assertEqual(len(COMMITTED.enabled), 9)
        enabled_ids = {m["model_id"] for m in COMMITTED.enabled}
        for model_id, route in CANDIDATES.items():
            self.assertNotIn(model_id, enabled_ids, f"{model_id} must not be enabled yet")
            self.assertEqual(COMMITTED.by_route(route)["model_id"], model_id)

    def test_every_candidate_is_t3_screening_only_with_nothing_invented(self):
        for model_id in CANDIDATES:
            m = next(m for m in COMMITTED.manifests if m["model_id"] == model_id)
            self.assertEqual(m["tier"], "T3", model_id)
            self.assertIsNone(m["license"]["authorization"], model_id)
            self.assertTrue(m["labels"]["screening_only"], model_id)
            self.assertIsNone(m["validation"]["report"], model_id)
            self.assertIsNone(m["runtime"]["weights_sha256"], model_id)
            self.assertTrue(m["runtime"]["weights_required"], f"{model_id}: a candidate always needs staged weights")

    def test_narrative_candidates_do_not_promise_a_verdict(self):
        """A language model's sentence is not a reading: no abnormal, no confidence in its fields."""
        m = next(m for m in COMMITTED.manifests if m["model_id"] == "medgemma-4b-it")
        self.assertEqual(m["outputs"]["kind"], "narrative")
        self.assertNotIn("abnormal", m["outputs"]["fields"])
        self.assertNotIn("confidence", m["outputs"]["fields"])

    def test_grandfathered_set_is_exactly_the_pre_widening_routes(self):
        self.assertEqual(len(GRANDFATHERED_ROUTES), 10)
        for route in GRANDFATHERED_ROUTES:
            self.assertIsNotNone(COMMITTED.by_route(route), route)
        self.assertFalse(GRANDFATHERED_ROUTES & set(CANDIDATES.values()))


class AdapterGuard(unittest.TestCase):
    def setUp(self):
        self.adapters = AdapterRegistry().adapters

    def test_every_enabled_manifest_has_its_adapter_in_this_image(self):
        self.assertEqual(enabled_without_adapter(COMMITTED, self.adapters), [])

    def test_guard_refuses_a_candidate_enabled_before_its_adapter_exists(self):
        for model_id in CANDIDATES:
            self.assertEqual(enabled_without_adapter(with_enabled(model_id), self.adapters), [model_id])


class EarnedGuard(unittest.TestCase):
    def test_committed_lock_is_legitimate(self):
        self.assertEqual(enabled_before_earned(COMMITTED), {})

    def test_guard_refuses_each_candidate_and_says_why(self):
        for model_id in CANDIDATES:
            problems = enabled_before_earned(with_enabled(model_id))
            self.assertEqual(set(problems), {model_id})
            self.assertIn("no validation report", problems[model_id])
            self.assertIn("no weights checksum", problems[model_id])
        self.assertIn("licence not reviewed", enabled_before_earned(with_enabled("rsna-ich-ct-head"))["rsna-ich-ct-head"])
        # the two with a named licence are still refused, on the report and the checksum
        self.assertNotIn("licence not reviewed", enabled_before_earned(with_enabled("monai-vista3d"))["monai-vista3d"])

    def test_a_report_alone_is_not_enough(self):
        reg = with_enabled("monai-vista3d")
        m = next(m for m in reg.enabled if m["model_id"] == "monai-vista3d")
        m = {**m, "validation": {**m["validation"], "report": "docs/validation/monai-vista3d-1.0.0.md"}}
        reg = mr.Registry(manifests=reg.manifests, enabled=[x for x in reg.enabled if x["model_id"] != "monai-vista3d"] + [m],
                          lock_path="<test>", manifest_dir=reg.manifest_dir)
        self.assertEqual(enabled_before_earned(reg), {"monai-vista3d": ["no weights checksum"]})


PROFILES = {
    "head_ct": profile(),
    "knee_mr": profile(modality="MR", protocol_family="mr_msk",
                       body_part={"raw": "KNEE", "normalized": "extremity", "source": "BodyPartExamined"}),
    "hand_cr": profile(modality="CR", protocol_family="cr_extremity", dimensionality="2d", contrast=None,
                       body_part={"raw": "HAND", "normalized": "extremity", "source": "BodyPartExamined"}),
    "dental_cbct": profile(protocol_family="cbct_dental", body_part={"raw": "JAW", "normalized": "jaw", "source": "BodyPartExamined"}),
    "chest_ct": profile(protocol_family="ct_chest", contrast=None,
                        body_part={"raw": "CHEST", "normalized": "chest", "source": "BodyPartExamined"}),
    "obstetric_us": profile(modality="US", protocol_family="us_obstetric", dimensionality="2d", contrast=None,
                            body_part={"raw": "FETUS", "normalized": "fetus", "source": "BodyPartExamined"}),
}


def outcome(result) -> tuple:
    if isinstance(result, mr.NoMatch):
        return ("nomatch", result.reason, result.candidates)
    return ("match", tuple((m.route_name, m.weights_present) for m in result))


WITHOUT_CANDIDATES = mr.Registry(manifests=[m for m in COMMITTED.manifests if m["model_id"] not in CANDIDATES],
                                 enabled=list(COMMITTED.enabled), lock_path="<test>", manifest_dir=COMMITTED.manifest_dir)


class RoutingAfterWidening(unittest.TestCase):
    """Committing disabled manifests changes NO routing outcome. In the shipped lock the anatomy
    routes accept every CT/MR volume and the generic Hugging Face routes every 2D image, so a
    disabled candidate is shadowed by an enabled catch-all; it surfaces as ``model_disabled`` only
    in a lock without those catch-alls. Both facts are pinned so the document can state them."""

    def test_every_profile_routes_exactly_as_before_the_widening(self):
        for name, prof in PROFILES.items():
            self.assertEqual(outcome(mr.select_models(prof, COMMITTED, authorized_only=False)),
                             outcome(mr.select_models(prof, WITHOUT_CANDIDATES, authorized_only=False)), name)

    def test_no_candidate_ever_appears_as_a_match(self):
        for name, prof in PROFILES.items():
            result = mr.select_models(prof, COMMITTED, authorized_only=False)
            if isinstance(result, list):
                self.assertFalse({m.route_name for m in result} & set(CANDIDATES.values()), name)

    def test_shadowing_is_real_catch_alls_accept_these_studies(self):
        """What a radiologist sees for these study types today: anatomy or a generic description."""
        expect = {"head_ct": "ct-anatomy-segmentation", "knee_mr": "mr-anatomy-segmentation", "hand_cr": "hf-2d-screening",
                  "dental_cbct": "ct-anatomy-segmentation", "obstetric_us": "us-hf-2d-screening"}
        for name, route in expect.items():
            result = mr.select_models(PROFILES[name], COMMITTED, authorized_only=False)
            self.assertIsInstance(result, list, name)
            self.assertIn(route, [m.route_name for m in result], name)

    def test_candidates_surface_as_model_disabled_only_without_the_catch_alls(self):
        """A radiologist-only lock (empty) still lets routing name the model that exists for the study."""
        radiologist_only = mr.Registry(manifests=list(COMMITTED.manifests), enabled=[], lock_path="<test>",
                                       manifest_dir=COMMITTED.manifest_dir)
        expect = {"head_ct": {"ct-head-haemorrhage-screening", "ct-anatomy-vista3d"},
                  "knee_mr": {"mr-knee-abnormality-screening"}, "hand_cr": {"cr-msk-abnormality-screening", "vlm-radiograph-description"},
                  "dental_cbct": {"cbct-dental-structure-segmentation"}}   # VISTA-3D does not list cbct_dental
        for name, routes in expect.items():
            result = mr.select_models(PROFILES[name], radiologist_only, authorized_only=False)
            self.assertIsInstance(result, mr.NoMatch, name)
            self.assertEqual(result.reason, "model_disabled", name)
            self.assertTrue(routes <= set(result.candidates), (name, result.candidates))
        # and where no candidate exists either, the reason stays the honest one
        result = mr.select_models(PROFILES["obstetric_us"], radiologist_only, authorized_only=False)
        self.assertEqual(result.reason, "model_disabled")   # the generic US route exists, disabled in this lock
        self.assertNotIn("vlm-radiograph-description", result.candidates, "MedGemma does not accept ultrasound")


if __name__ == "__main__":
    unittest.main()
