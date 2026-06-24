import importlib
import pathlib
import sys
import unittest
from unittest.mock import Mock, patch


ORCHESTRATOR_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_DIR))

orchestrator_api = importlib.import_module("orchestrator_api")


class PublishResultPayloadTests(unittest.TestCase):
    def test_send_to_webhook_succeeds_when_configured(self) -> None:
        payload = {"study_uid": "study-webhook-ok"}
        response = Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        with patch.object(orchestrator_api, "RESULT_WEBHOOK_URL", "https://webhook.site/test"), patch.object(
            orchestrator_api.requests,
            "post",
            return_value=response,
        ) as mocked_post:
            result = orchestrator_api.send_to_webhook(payload)

        mocked_post.assert_called_once()
        self.assertTrue(result["enabled"])
        self.assertTrue(result["delivered"])
        self.assertEqual(result["status"], "delivered")

    def test_send_to_webhook_failure_is_non_blocking_status(self) -> None:
        payload = {"study_uid": "study-webhook-fail"}
        with patch.object(orchestrator_api, "RESULT_WEBHOOK_URL", "https://webhook.site/test"), patch.object(
            orchestrator_api.requests,
            "post",
            side_effect=RuntimeError("expired webhook"),
        ):
            result = orchestrator_api.send_to_webhook(payload)

        self.assertTrue(result["enabled"])
        self.assertFalse(result["delivered"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("expired webhook", result["detail"])

    def test_publish_result_payload_marks_all_enabled_targets_delivered(self) -> None:
        payload = {"study_uid": "study-1"}
        hospital_status = {
            "target": "hospital_service",
            "enabled": True,
            "delivered": True,
            "status": "delivered",
            "detail": "HTTP 200",
        }
        webhook_status = {
            "target": "webhook",
            "enabled": True,
            "delivered": True,
            "status": "delivered",
            "detail": "HTTP 200",
        }
        kafka_status = {
            "target": "kafka",
            "enabled": True,
            "delivered": True,
            "status": "delivered",
            "detail": "Kafka publish succeeded.",
        }

        with patch.object(orchestrator_api, "send_to_hospital_service", return_value=hospital_status), patch.object(
            orchestrator_api,
            "send_to_webhook",
            return_value=webhook_status,
        ), patch.object(orchestrator_api, "send_to_kafka", return_value=kafka_status):
            result = orchestrator_api.publish_result_payload(payload)

        self.assertTrue(result["delivery_summary"]["all_enabled_targets_delivered"])
        self.assertEqual(result["delivery_summary"]["failed_targets"], [])
        self.assertEqual(result["delivery_status"]["hospital_service"], hospital_status)
        self.assertEqual(result["delivery_status"]["kafka"], kafka_status)

    def test_publish_result_payload_collects_failed_targets(self) -> None:
        payload = {"study_uid": "study-2"}
        hospital_status = {
            "target": "hospital_service",
            "enabled": True,
            "delivered": True,
            "status": "delivered",
            "detail": "HTTP 200",
        }
        webhook_status = {
            "target": "webhook",
            "enabled": True,
            "delivered": False,
            "status": "failed",
            "detail": "expired webhook",
        }
        kafka_status = {
            "target": "kafka",
            "enabled": True,
            "delivered": False,
            "status": "failed",
            "detail": "timeout",
        }

        with patch.object(orchestrator_api, "send_to_hospital_service", return_value=hospital_status), patch.object(
            orchestrator_api,
            "send_to_webhook",
            return_value=webhook_status,
        ), patch.object(orchestrator_api, "send_to_kafka", return_value=kafka_status):
            result = orchestrator_api.publish_result_payload(payload)

        self.assertFalse(result["delivery_summary"]["all_enabled_targets_delivered"])
        self.assertEqual(result["delivery_summary"]["failed_targets"], ["webhook", "kafka"])


class StatusDerivationTests(unittest.TestCase):
    def test_build_failed_result_marks_manual_review_required(self) -> None:
        result = orchestrator_api.build_failed_result(
            filename="series-1.dcm",
            series_uid="series-1",
            study_uid="study-1",
            modality="MR",
            body_part="BRAIN",
            error="AI call failed: engine unavailable",
            failure_stage="inference",
            error_category="engine-unavailable",
        )
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["manual_review_required"])
        self.assertEqual(result["failure_stage"], "inference")
        self.assertEqual(result["error_category"], "engine-unavailable")

    def test_derive_inference_status_completed(self) -> None:
        results = [{"status": "completed"}, {"status": "completed"}]
        self.assertEqual(orchestrator_api.derive_inference_status(results, []), "completed")

    def test_derive_inference_status_partial(self) -> None:
        results = [{"status": "completed"}, {"status": "failed"}]
        errors = ["series-2 failed"]
        self.assertEqual(orchestrator_api.derive_inference_status(results, errors), "partial")

    def test_derive_inference_status_failed_when_no_completed_results(self) -> None:
        results = [{"status": "failed"}]
        errors = ["series-1 failed"]
        self.assertEqual(orchestrator_api.derive_inference_status(results, errors), "failed")

    def test_derive_delivery_status_not_configured(self) -> None:
        payload = {
            "delivery_status": {
                "webhook": {"enabled": False, "delivered": False},
                "kafka": {"enabled": False, "delivered": False},
            },
            "delivery_summary": {"all_enabled_targets_delivered": True, "failed_targets": []},
        }
        self.assertEqual(orchestrator_api.derive_delivery_status(payload), "not-configured")

    def test_derive_delivery_status_partial(self) -> None:
        payload = {
            "delivery_status": {
                "hospital_service": {"enabled": True, "delivered": True},
                "kafka": {"enabled": True, "delivered": False},
            },
            "delivery_summary": {"all_enabled_targets_delivered": False, "failed_targets": ["kafka"]},
        }
        self.assertEqual(orchestrator_api.derive_delivery_status(payload), "partial")

    def test_derive_operation_status_degraded_when_delivery_fails(self) -> None:
        self.assertEqual(orchestrator_api.derive_operation_status("completed", "failed"), "degraded")

    def test_categorize_failure_inference_timeout(self) -> None:
        category = orchestrator_api.categorize_failure(
            "inference",
            "AI call failed: request timed out after 30 seconds",
        )
        self.assertEqual(category, "network-timeout")

    def test_summarize_failures_groups_stage_and_category(self) -> None:
        results = [
            orchestrator_api.build_failed_result(
                filename="series-1.dcm",
                series_uid="series-1",
                study_uid="study-1",
                error="AI call failed: all ai engines failed",
                failure_stage="inference",
                error_category="engine-unavailable",
            ),
            orchestrator_api.build_failed_result(
                filename="series-2.dcm",
                series_uid="series-2",
                study_uid="study-1",
                error="STOW failed: connection refused",
                failure_stage="stow",
                error_category="delivery",
            ),
            {"status": "completed", "series_uid": "series-3"},
        ]
        summary = orchestrator_api.summarize_failures(results)
        self.assertEqual(summary["failed_series_count"], 2)
        self.assertTrue(summary["manual_review_required"])
        self.assertEqual(summary["by_stage"]["inference"], 1)
        self.assertEqual(summary["by_stage"]["stow"], 1)
        self.assertEqual(summary["by_category"]["engine-unavailable"], 1)
        self.assertEqual(summary["by_category"]["delivery"], 1)

    def test_validate_ai_response_rejects_non_dict(self) -> None:
        is_valid, message, category = orchestrator_api.validate_ai_response("bad-response")
        self.assertFalse(is_valid)
        self.assertEqual(category, "invalid-engine-response")
        self.assertIn("non-dictionary", message)

    def test_validate_ai_response_rejects_empty_clinical_content(self) -> None:
        ai_response = {
            "model_id": "demo-model",
            "report": {
                "analysis_type": "",
                "conclusion": "",
                "observations": [],
                "report_type": "",
                "diagnostic_support": "",
            },
        }
        is_valid, message, category = orchestrator_api.validate_ai_response(ai_response)
        self.assertFalse(is_valid)
        self.assertEqual(category, "empty-engine-response")
        self.assertIn("usable clinical content", message)

    def test_validate_ai_response_accepts_minimal_valid_response(self) -> None:
        ai_response = {
            "model_id": "demo-model",
            "report": {
                "analysis_type": "screening",
                "conclusion": "No acute abnormality.",
                "observations": [],
            },
        }
        is_valid, message, category = orchestrator_api.validate_ai_response(ai_response)
        self.assertTrue(is_valid)
        self.assertEqual(message, "")
        self.assertEqual(category, "")

    def test_validate_required_metadata_rejects_missing_fields(self) -> None:
        metadata = {"StudyInstanceUID": "study-1", "Modality": "CT"}
        is_valid, message, category = orchestrator_api.validate_required_metadata(metadata)
        self.assertFalse(is_valid)
        self.assertEqual(category, "invalid-dicom-metadata")
        self.assertIn("SeriesInstanceUID", message)

    def test_sanitize_clinical_text_softens_overclaim_language(self) -> None:
        text = "Confirmed diagnostic normal study."
        sanitized = orchestrator_api.sanitize_clinical_text(text)
        self.assertIn("suggests", sanitized.lower())
        self.assertNotIn("confirmed", sanitized.lower())

    def test_sanitize_clinical_text_blocks_unsafe_certainty_phrases(self) -> None:
        text = "This definitely rules out disease and proves normal examination."
        sanitized = orchestrator_api.sanitize_clinical_text(text)
        self.assertNotIn("definitely", sanitized.lower())
        self.assertNotIn("proves", sanitized.lower())
        self.assertNotIn("rules out", sanitized.lower())

    def test_requires_confidence_softening_for_low_confidence_screening(self) -> None:
        report = {
            "report_type": "preliminary-2d-screening",
            "confidence": 0.51,
        }
        self.assertTrue(orchestrator_api.requires_confidence_softening(report))

    def test_apply_output_guardrails_downgrades_screening_without_evidence(self) -> None:
        report = {
            "diagnostic_support": "screening-only",
            "report_type": "preliminary-2d-screening",
            "conclusion": "",
            "observations": [],
            "abnormalities": [],
            "metrics": {},
        }
        summary = {
            "exam": "CR chest",
            "technique": "Single projection image.",
            "findings": "Possible finding.",
            "impression": "Abnormal screening pattern detected.",
            "recommendation": "Escalate for review.",
        }
        guarded_summary, guardrails = orchestrator_api.apply_output_guardrails(report, summary)
        self.assertTrue(guardrails["applied"])
        self.assertIn("xr-us-missing-observation-evidence", guardrails["reasons"])
        self.assertIn("non-diagnostic", guarded_summary["impression"].lower())

    def test_apply_output_guardrails_softens_low_confidence_screening_language(self) -> None:
        report = {
            "diagnostic_support": "screening-only",
            "report_type": "preliminary-2d-screening",
            "confidence": 0.51,
            "conclusion": "Possible finding.",
            "observations": ["Possible opacity"],
            "abnormalities": ["opacity"],
            "metrics": {},
        }
        summary = {
            "exam": "CR chest",
            "technique": "Single projection image.",
            "findings": "Possible opacity.",
            "impression": "Abnormal screening pattern detected.",
            "recommendation": "Escalate for review.",
        }
        guarded_summary, guardrails = orchestrator_api.apply_output_guardrails(report, summary)
        self.assertTrue(guardrails["applied"])
        self.assertIn("low-confidence-wording-softened", guardrails["reasons"])
        self.assertIn("low-confidence", guarded_summary["impression"].lower())

    def test_evaluate_modality_guardrails_ct_requires_candidate_evidence(self) -> None:
        report = {
            "diagnostic_support": "screening-only",
            "report_type": "preliminary-lung-nodule-screening",
            "metrics": {"candidate_count": 0},
        }
        result = orchestrator_api.evaluate_modality_guardrails(report)
        self.assertTrue(result["downgrade"])
        self.assertIn("ct-missing-candidate-evidence", result["reasons"])

    def test_evaluate_modality_guardrails_xa_requires_frame_burden(self) -> None:
        report = {
            "diagnostic_support": "screening-only",
            "report_type": "preliminary-stenosis-screening",
            "metrics": {"frame_count": 0, "positive_frames": 0},
        }
        result = orchestrator_api.evaluate_modality_guardrails(report)
        self.assertTrue(result["downgrade"])
        self.assertIn("xa-missing-frame-burden", result["reasons"])

    def test_evaluate_modality_guardrails_us_requires_functional_metrics(self) -> None:
        report = {
            "diagnostic_support": "screening-only",
            "report_type": "preliminary-echocardiography-lv-function",
            "metrics": {},
        }
        result = orchestrator_api.evaluate_modality_guardrails(report)
        self.assertTrue(result["downgrade"])
        self.assertIn("us-missing-functional-evidence", result["reasons"])

    def test_apply_modality_wording_templates_ct_prefixes_summary(self) -> None:
        report = {
            "report_type": "preliminary-lung-nodule-screening",
            "diagnostic_support": "screening-only",
        }
        summary = {
            "findings": "Multiple suspicious pulmonary nodule candidates are present.",
            "impression": "Radiologist review is required.",
            "recommendation": "Review with radiologist.",
        }
        templated = orchestrator_api.apply_modality_wording_templates(report, summary)
        self.assertTrue(templated["findings"].startswith("Chest CT screening"))
        self.assertTrue(templated["impression"].startswith("Screening chest CT"))

    def test_apply_modality_wording_templates_xa_prefixes_summary(self) -> None:
        report = {
            "report_type": "preliminary-stenosis-screening",
            "diagnostic_support": "screening-only",
        }
        summary = {
            "findings": "Multiframe suspicious narrowing pattern identified.",
            "impression": "Formal angiographic interpretation is required.",
            "recommendation": "Correlate with cardiology review.",
        }
        templated = orchestrator_api.apply_modality_wording_templates(report, summary)
        self.assertTrue(templated["findings"].startswith("Angiographic screening"))
        self.assertTrue(templated["impression"].startswith("Preliminary angiographic AI screening"))

    def test_derive_result_policy_hides_confidence_for_xa(self) -> None:
        policy = orchestrator_api.derive_result_policy(
            {
                "report_type": "preliminary-stenosis-screening",
                "diagnostic_support": "screening-only",
            }
        )
        self.assertFalse(policy["can_expose_confidence"])
        self.assertTrue(policy["must_hide_confidence"])

    def test_derive_result_policy_for_anatomy_only_forces_manual_review(self) -> None:
        policy = orchestrator_api.derive_result_policy(
            {
                "report_type": "non-diagnostic-anatomy",
                "diagnostic_support": "anatomy-only",
            }
        )
        self.assertFalse(policy["can_set_abnormal"])
        self.assertFalse(policy["can_expose_confidence"])
        self.assertTrue(policy["must_force_manual_review"])

    def test_derive_result_policy_limited_structure_claims_for_2d_screening(self) -> None:
        policy = orchestrator_api.derive_result_policy(
            {
                "report_type": "preliminary-2d-screening",
                "diagnostic_support": "screening-only",
            }
        )
        self.assertFalse(policy["can_claim_specific_structure"])

    def test_derive_confidence_disclosure_policy_hides_xa_numeric_confidence(self) -> None:
        result_policy = orchestrator_api.derive_result_policy(
            {
                "report_type": "preliminary-stenosis-screening",
                "diagnostic_support": "screening-only",
            }
        )
        policy = orchestrator_api.derive_confidence_disclosure_policy(
            {
                "report_type": "preliminary-stenosis-screening",
                "diagnostic_support": "screening-only",
                "confidence": 0.91,
            },
            result_policy,
        )
        self.assertEqual(policy["expose_mode"], "hidden")
        self.assertFalse(policy["can_expose_numeric"])

    def test_derive_confidence_disclosure_policy_uses_qualitative_band_for_ct(self) -> None:
        result_policy = orchestrator_api.derive_result_policy(
            {
                "report_type": "preliminary-lung-nodule-screening",
                "diagnostic_support": "screening-only",
            }
        )
        policy = orchestrator_api.derive_confidence_disclosure_policy(
            {
                "report_type": "preliminary-lung-nodule-screening",
                "diagnostic_support": "screening-only",
                "confidence": 0.88,
            },
            result_policy,
        )
        self.assertEqual(policy["expose_mode"], "qualitative_band")
        self.assertFalse(policy["can_expose_numeric"])

    def test_normalize_confidence_disclosure_rounds_us_confidence(self) -> None:
        result_policy = orchestrator_api.derive_result_policy(
            {
                "report_type": "preliminary-echocardiography-lv-function",
                "diagnostic_support": "screening-only",
            }
        )
        confidence_value, confidence_band, policy = orchestrator_api.normalize_confidence_disclosure(
            {
                "report_type": "preliminary-echocardiography-lv-function",
                "diagnostic_support": "screening-only",
                "confidence": 0.8349,
            },
            result_policy,
        )
        self.assertEqual(confidence_value, 0.83)
        self.assertEqual(confidence_band, "high")
        self.assertEqual(policy["expose_mode"], "rounded_numeric")

    def test_apply_limitation_policy_adds_required_anatomy_only_limitations(self) -> None:
        result_policy = orchestrator_api.derive_result_policy(
            {
                "report_type": "non-diagnostic-anatomy",
                "diagnostic_support": "anatomy-only",
            }
        )
        limitations, policy = orchestrator_api.apply_limitation_policy(
            {
                "report_type": "non-diagnostic-anatomy",
                "diagnostic_support": "anatomy-only",
                "limitations": ["Model-specific note."],
            },
            result_policy,
            {"applied": False, "reasons": []},
        )
        self.assertIn("Model-specific note.", limitations)
        self.assertTrue(any("anatomy-focused ai support only" in item.lower() for item in limitations))
        self.assertTrue(policy["must_include_manual_review_limitations"])

    def test_apply_limitation_policy_adds_downgrade_limitation(self) -> None:
        result_policy = orchestrator_api.derive_result_policy(
            {
                "report_type": "manual-review-required",
                "diagnostic_support": "not-supported",
            }
        )
        limitations, _ = orchestrator_api.apply_limitation_policy(
            {
                "report_type": "manual-review-required",
                "diagnostic_support": "not-supported",
                "limitations": [],
            },
            result_policy,
            {"applied": True, "reasons": ["insufficient-structured-evidence"]},
        )
        self.assertTrue(any("structured evidence was insufficient" in item.lower() for item in limitations))

    def test_apply_summary_consistency_guardrails_rebuilds_summary_from_final_fields(self) -> None:
        ai_result, guardrails = orchestrator_api.apply_summary_consistency_guardrails(
            {
                "finding": "Old finding.",
                "exam": "CT chest",
                "technique": "Axial CT reviewed.",
                "findings": "Chest CT screening findings: candidate present.",
                "impression": "Screening chest CT impression: specialist confirmation required.",
                "recommendation": "Review with radiologist and treat this as screening support only.",
                "limitations": ["Route limitation one.", "Route limitation two."],
            },
            {
                "metadata_summary": {
                    "series": {
                        "modality": "CT",
                        "body_part_examined": "CHEST",
                        "series_description": "Axial chest series",
                    }
                }
            },
            "completed",
            "study-summary-1",
            [],
        )
        self.assertEqual(ai_result["finding"], ai_result["impression"])
        self.assertIn("Recommendation: Review with radiologist", ai_result["summary"])
        self.assertIn("Limitations: Route limitation one. Route limitation two.", ai_result["summary"])
        self.assertIn("summary-rebuilt-from-final-fields", guardrails["reasons"])

    def test_derive_model_chain_contract_for_ct_is_chain_ready(self) -> None:
        contract = orchestrator_api.derive_model_chain_contract(
            {
                "report_type": "preliminary-lung-nodule-screening",
                "diagnostic_support": "screening-only",
                "metrics": {
                    "candidate_count": 2,
                    "top_boxes_xyzxyz": [[1, 2, 3, 4, 5, 6]],
                    "top_score": 0.88,
                },
            },
            {"NumberOfFrames": "12"},
        )
        self.assertTrue(contract["chain_ready"])
        self.assertEqual(contract["chain_stage"], "detector-output")
        self.assertEqual(contract["next_stage"], "candidate-triage")
        self.assertIn("candidate_count", contract["available_artifacts"])

    def test_derive_model_chain_contract_for_anatomy_only_is_not_chain_ready(self) -> None:
        contract = orchestrator_api.derive_model_chain_contract(
            {
                "report_type": "non-diagnostic-anatomy",
                "diagnostic_support": "anatomy-only",
            },
            {},
        )
        self.assertFalse(contract["chain_ready"])
        self.assertEqual(contract["chain_stage"], "non-diagnostic")
        self.assertIsNone(contract["next_stage"])

    def test_normalize_chain_evidence_for_ct_exposes_candidate_bundle(self) -> None:
        evidence = orchestrator_api.normalize_chain_evidence(
            {
                "report_type": "preliminary-lung-nodule-screening",
                "diagnostic_support": "screening-only",
                "confidence": 0.88,
                "metrics": {
                    "candidate_count": 2,
                    "top_score": 0.88,
                    "top_boxes_xyzxyz": [[1, 2, 3, 4, 5, 6]],
                },
                "candidate_locations": ["left lower lobe"],
            },
            {"NumberOfFrames": "12"},
        )
        self.assertEqual(evidence["evidence_type"], "candidate-detection")
        self.assertTrue(evidence["chain_ready"])
        self.assertEqual(evidence["measurements"]["candidate_count"], 2)
        self.assertIn("left lower lobe", evidence["targets"])

    def test_normalize_chain_evidence_for_anatomy_only_is_non_chainable(self) -> None:
        evidence = orchestrator_api.normalize_chain_evidence(
            {
                "report_type": "non-diagnostic-anatomy",
                "diagnostic_support": "anatomy-only",
            },
            {},
        )
        self.assertEqual(evidence["evidence_type"], "none")
        self.assertFalse(evidence["chain_ready"])
        self.assertIsNone(evidence["next_stage"])

    def test_derive_second_stage_selection_policy_invokes_for_chain_ready_ct(self) -> None:
        result_policy = orchestrator_api.derive_result_policy(
            {
                "report_type": "preliminary-lung-nodule-screening",
                "diagnostic_support": "screening-only",
            }
        )
        policy = orchestrator_api.derive_second_stage_selection_policy(
            {
                "report_type": "preliminary-lung-nodule-screening",
                "diagnostic_support": "screening-only",
                "metrics": {
                    "candidate_count": 2,
                    "top_boxes_xyzxyz": [[1, 2, 3, 4, 5, 6]],
                    "top_score": 0.88,
                },
                "candidate_locations": ["left lower lobe"],
            },
            result_policy,
            {"applied": False, "reasons": []},
            {"NumberOfFrames": "12"},
        )
        self.assertTrue(policy["should_invoke"])
        self.assertEqual(policy["decision"], "invoke")
        self.assertEqual(policy["next_stage"], "candidate-triage")

    def test_derive_second_stage_selection_policy_blocks_anatomy_only(self) -> None:
        result_policy = orchestrator_api.derive_result_policy(
            {
                "report_type": "non-diagnostic-anatomy",
                "diagnostic_support": "anatomy-only",
            }
        )
        policy = orchestrator_api.derive_second_stage_selection_policy(
            {
                "report_type": "non-diagnostic-anatomy",
                "diagnostic_support": "anatomy-only",
            },
            result_policy,
            {"applied": False, "reasons": []},
            {},
        )
        self.assertFalse(policy["should_invoke"])
        self.assertEqual(policy["decision"], "blocked")
        self.assertEqual(policy["reason"], "non-diagnostic-route")

    def test_derive_second_stage_selection_policy_blocks_guardrail_downgraded_screening(self) -> None:
        result_policy = orchestrator_api.derive_result_policy(
            {
                "report_type": "preliminary-lung-nodule-screening",
                "diagnostic_support": "screening-only",
            }
        )
        policy = orchestrator_api.derive_second_stage_selection_policy(
            {
                "report_type": "preliminary-lung-nodule-screening",
                "diagnostic_support": "screening-only",
                "metrics": {"candidate_count": 0},
            },
            result_policy,
            {"applied": True, "reasons": ["ct-missing-candidate-evidence"]},
            {},
        )
        self.assertFalse(policy["should_invoke"])
        self.assertEqual(policy["decision"], "blocked")
        self.assertEqual(policy["reason"], "guardrail-blocked")

    def test_build_second_stage_input_payload_for_ct_is_invokable(self) -> None:
        report = {
            "report_type": "preliminary-lung-nodule-screening",
            "diagnostic_support": "screening-only",
            "routing_decision": {"route_name": "ct-monai-screening"},
            "metrics": {
                "candidate_count": 2,
                "top_boxes_xyzxyz": [[1, 2, 3, 4, 5, 6]],
                "top_score": 0.88,
            },
            "candidate_locations": ["left lower lobe"],
        }
        metadata = {
            "StudyInstanceUID": "study-ct-stage2",
            "SeriesInstanceUID": "series-ct-stage2",
            "Modality": "CT",
            "BodyPartExamined": "CHEST",
        }
        payload = orchestrator_api.build_second_stage_input_payload(
            "study-ct-stage2",
            report,
            metadata,
            orchestrator_api.derive_result_policy(report),
            {"applied": False, "reasons": []},
        )
        self.assertTrue(payload["chain_invocation"]["should_invoke"])
        self.assertEqual(payload["chain_invocation"]["next_stage"], "candidate-triage")
        self.assertEqual(payload["normalized_evidence"]["evidence_type"], "candidate-detection")
        self.assertEqual(payload["metadata_summary"]["series"]["modality"], "CT")

    def test_build_second_stage_input_payload_for_anatomy_only_mr_is_blocked(self) -> None:
        report = {
            "report_type": "non-diagnostic-anatomy",
            "diagnostic_support": "anatomy-only",
            "routing_decision": {"route_name": "mr-anatomy-segmentation"},
        }
        metadata = {
            "StudyInstanceUID": "study-mr-stage2",
            "SeriesInstanceUID": "series-mr-stage2",
            "Modality": "MR",
            "BodyPartExamined": "BRAIN",
        }
        payload = orchestrator_api.build_second_stage_input_payload(
            "study-mr-stage2",
            report,
            metadata,
            orchestrator_api.derive_result_policy(report),
            {"applied": False, "reasons": []},
        )
        self.assertFalse(payload["chain_invocation"]["should_invoke"])
        self.assertEqual(payload["chain_invocation"]["decision"], "blocked")
        self.assertEqual(payload["normalized_evidence"]["evidence_type"], "none")

    def test_derive_second_stage_merge_policy_allows_ct_augmentation(self) -> None:
        ai_result = {
            "report_type": "preliminary-lung-nodule-screening",
            "diagnostic_support": "screening-only",
        }
        selection_policy = {
            "should_invoke": True,
            "decision": "invoke",
            "next_stage": "candidate-triage",
        }
        policy = orchestrator_api.derive_second_stage_merge_policy(ai_result, selection_policy)
        self.assertTrue(policy["can_merge"])
        self.assertEqual(policy["merge_mode"], "augment")
        self.assertIn("impression", policy["allowed_fields"])

    def test_derive_second_stage_merge_policy_blocks_anatomy_only_mr(self) -> None:
        ai_result = {
            "report_type": "non-diagnostic-anatomy",
            "diagnostic_support": "anatomy-only",
        }
        selection_policy = {
            "should_invoke": False,
            "decision": "blocked",
            "next_stage": None,
        }
        policy = orchestrator_api.derive_second_stage_merge_policy(ai_result, selection_policy)
        self.assertFalse(policy["can_merge"])
        self.assertEqual(policy["merge_mode"], "blocked")

    def test_merge_second_stage_result_augments_allowed_fields_only(self) -> None:
        ai_result = {
            "finding": "Old finding",
            "findings": "Old findings",
            "impression": "Old impression",
            "recommendation": "Old recommendation",
            "limitations": ["Base limitation"],
            "report_type": "preliminary-lung-nodule-screening",
            "diagnostic_support": "screening-only",
        }
        merge_policy = {
            "can_merge": True,
            "merge_mode": "augment",
            "allowed_fields": ["finding", "impression", "recommendation", "limitations"],
            "blocked_fields": ["report_type", "diagnostic_support"],
            "reason": "merge-allowed",
        }
        merged, merge_summary = orchestrator_api.merge_second_stage_result(
            ai_result,
            {
                "finding": "New refined finding",
                "impression": "New refined impression",
                "recommendation": "New refined recommendation",
                "limitations": ["Second-stage limitation"],
                "report_type": "manual-review-required",
                "diagnostic_support": "not-supported",
            },
            merge_policy,
        )
        self.assertEqual(merged["finding"], "New refined finding")
        self.assertEqual(merged["impression"], "New refined impression")
        self.assertEqual(merged["recommendation"], "New refined recommendation")
        self.assertIn("Second-stage limitation", merged["limitations"])
        self.assertEqual(merged["report_type"], "preliminary-lung-nodule-screening")
        self.assertEqual(merged["diagnostic_support"], "screening-only")
        self.assertTrue(merge_summary["applied"])

    def test_run_second_stage_pipeline_executes_ct_candidate_triage(self) -> None:
        payload = {
            "chain_invocation": {
                "should_invoke": True,
                "decision": "invoke",
                "next_stage": "candidate-triage",
                "reason": "chain-ready",
            },
            "normalized_evidence": {
                "evidence_type": "candidate-detection",
                "targets": ["left lower lobe"],
                "measurements": {"candidate_count": 2, "top_score": 0.88},
            },
        }
        result = orchestrator_api.run_second_stage_pipeline(payload)
        self.assertTrue(result["invoked"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stage_name"], "candidate-triage")
        self.assertIn("pulmonary nodule candidate pattern", result["result"]["impression"].lower())

    def test_run_second_stage_pipeline_executes_xa_run_summarizer(self) -> None:
        payload = {
            "chain_invocation": {
                "should_invoke": True,
                "decision": "invoke",
                "next_stage": "run-summarizer",
                "reason": "chain-ready",
            },
            "normalized_evidence": {
                "evidence_type": "frame-burden",
                "measurements": {
                    "frame_count": 10,
                    "positive_frames": 6,
                    "max_positive_pixel_ratio": 0.02,
                },
            },
        }
        result = orchestrator_api.run_second_stage_pipeline(payload)
        self.assertTrue(result["invoked"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stage_name"], "run-summarizer")
        self.assertIn("narrowing pattern", result["result"]["impression"].lower())

    def test_run_second_stage_pipeline_executes_us_cardiac_summary(self) -> None:
        payload = {
            "chain_invocation": {
                "should_invoke": True,
                "decision": "invoke",
                "next_stage": "cardiac-summary-classifier",
                "reason": "chain-ready",
            },
            "normalized_evidence": {
                "evidence_type": "functional-metrics",
                "measurements": {
                    "estimated_ef_percent": 35.0,
                    "fractional_area_change": None,
                },
            },
        }
        result = orchestrator_api.run_second_stage_pipeline(payload)
        self.assertTrue(result["invoked"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stage_name"], "cardiac-summary-classifier")
        self.assertIn("systolic function", result["result"]["impression"].lower())

    def test_run_second_stage_pipeline_executes_mr_lesion_burden_interpreter(self) -> None:
        payload = {
            "chain_invocation": {
                "should_invoke": True,
                "decision": "invoke",
                "next_stage": "lesion-burden-interpreter",
                "reason": "chain-ready",
            },
            "normalized_evidence": {
                "evidence_type": "segmentation-burden",
                "measurements": {
                    "whole_tumor_voxels": 45000,
                    "whole_tumor_ratio": 0.04,
                    "tumor_core_voxels": 12000,
                    "enhancing_tumor_voxels": 3000,
                },
            },
        }
        result = orchestrator_api.run_second_stage_pipeline(payload)
        self.assertTrue(result["invoked"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stage_name"], "lesion-burden-interpreter")
        self.assertIn("lesion burden", result["result"]["impression"].lower())

    def test_build_chain_observability_for_successful_chain(self) -> None:
        ai_result = {
            "model_name": "monai-lung-nodule-retinanet",
            "report_type": "preliminary-lung-nodule-screening",
            "diagnostic_support": "screening-only",
            "model_chain_contract": {"chain_ready": True},
            "normalized_chain_evidence": {"evidence_type": "candidate-detection"},
            "second_stage_selection_policy": {
                "next_stage": "candidate-triage",
                "decision": "invoke",
                "reason": "chain-ready",
                "blocking_reasons": [],
            },
            "second_stage_execution": {
                "invoked": True,
                "status": "completed",
                "stage_name": "candidate-triage",
                "result": {"stage_model_name": "ct-candidate-triage-rule-engine"},
            },
            "second_stage_merge_summary": {
                "applied": True,
                "mode": "augment",
                "reason": "merge-allowed",
                "merged_fields": ["findings", "impression"],
            },
            "output_guardrails": {"reasons": []},
        }
        observability = orchestrator_api.build_chain_observability(
            ai_result,
            {"model_id": "monai-lung-nodule-retinanet"},
        )
        self.assertTrue(observability["chain_ready"])
        self.assertTrue(observability["second_stage_invoked"])
        self.assertTrue(observability["merge_applied"])
        self.assertEqual(observability["second_stage_name"], "candidate-triage")

    def test_build_chain_observability_for_blocked_chain(self) -> None:
        ai_result = {
            "model_name": "totalsegmentator-total_mr",
            "report_type": "non-diagnostic-anatomy",
            "diagnostic_support": "anatomy-only",
            "model_chain_contract": {"chain_ready": False},
            "normalized_chain_evidence": {"evidence_type": "none"},
            "second_stage_selection_policy": {
                "next_stage": None,
                "decision": "blocked",
                "reason": "non-diagnostic-route",
                "blocking_reasons": ["non-diagnostic-route"],
            },
            "second_stage_execution": {
                "invoked": False,
                "status": "skipped",
                "reason": "non-diagnostic-route",
            },
            "second_stage_merge_summary": {
                "applied": False,
                "mode": "blocked",
                "reason": "non-diagnostic-route",
                "merged_fields": [],
            },
            "output_guardrails": {"reasons": []},
        }
        observability = orchestrator_api.build_chain_observability(ai_result, None)
        self.assertFalse(observability["chain_ready"])
        self.assertFalse(observability["second_stage_invoked"])
        self.assertFalse(observability["merge_applied"])
        self.assertEqual(observability["selection_decision"], "blocked")

    def test_record_chain_metrics_for_invoked_chain(self) -> None:
        ai_result = {
            "routing_decision": {"route_name": "ct-monai-screening"},
            "second_stage_selection_policy": {
                "next_stage": "candidate-triage",
                "decision": "invoke",
                "blocking_reasons": [],
            },
            "second_stage_execution": {
                "stage_name": "candidate-triage",
                "status": "completed",
            },
            "second_stage_merge_summary": {
                "mode": "augment",
                "applied": True,
            },
        }
        before_selection = orchestrator_api.CHAIN_SELECTION_TOTAL.labels(
            "ct-monai-screening", "candidate-triage", "invoke"
        )._value.get()
        before_execution = orchestrator_api.CHAIN_EXECUTION_TOTAL.labels(
            "ct-monai-screening", "candidate-triage", "completed"
        )._value.get()
        before_merge = orchestrator_api.CHAIN_MERGE_TOTAL.labels(
            "ct-monai-screening", "augment", "true"
        )._value.get()

        orchestrator_api.record_chain_metrics(ai_result)

        self.assertEqual(
            orchestrator_api.CHAIN_SELECTION_TOTAL.labels(
                "ct-monai-screening", "candidate-triage", "invoke"
            )._value.get(),
            before_selection + 1,
        )
        self.assertEqual(
            orchestrator_api.CHAIN_EXECUTION_TOTAL.labels(
                "ct-monai-screening", "candidate-triage", "completed"
            )._value.get(),
            before_execution + 1,
        )
        self.assertEqual(
            orchestrator_api.CHAIN_MERGE_TOTAL.labels(
                "ct-monai-screening", "augment", "true"
            )._value.get(),
            before_merge + 1,
        )

    def test_record_chain_metrics_for_blocked_chain(self) -> None:
        ai_result = {
            "routing_decision": {"route_name": "mr-anatomy-segmentation"},
            "second_stage_selection_policy": {
                "next_stage": None,
                "decision": "blocked",
                "blocking_reasons": ["non-diagnostic-route"],
            },
            "second_stage_execution": {
                "status": "skipped",
            },
            "second_stage_merge_summary": {
                "mode": "blocked",
                "applied": False,
            },
        }
        before_block_reason = orchestrator_api.CHAIN_BLOCK_REASON_TOTAL.labels(
            "mr-anatomy-segmentation", "non-diagnostic-route"
        )._value.get()

        orchestrator_api.record_chain_metrics(ai_result)

        self.assertEqual(
            orchestrator_api.CHAIN_BLOCK_REASON_TOTAL.labels(
                "mr-anatomy-segmentation", "non-diagnostic-route"
            )._value.get(),
            before_block_reason + 1,
        )

    def test_run_second_stage_pipeline_records_duration_metric(self) -> None:
        payload = {
            "route_name": "ct-monai-screening",
            "chain_invocation": {
                "should_invoke": True,
                "decision": "invoke",
                "next_stage": "candidate-triage",
                "reason": "chain-ready",
            },
            "normalized_evidence": {
                "evidence_type": "candidate-detection",
                "targets": ["left lower lobe"],
                "measurements": {"candidate_count": 2, "top_score": 0.88},
            },
        }
        histogram = orchestrator_api.CHAIN_STAGE_DURATION_SECONDS.labels(
            "ct-monai-screening", "candidate-triage"
        )
        def sample_value(sample_name: str) -> float:
            for metric in histogram.collect():
                for sample in metric.samples:
                    if sample.name == sample_name:
                        return float(sample.value)
            self.fail(f"Histogram sample {sample_name} was not found")

        before_count = sample_value("orchestrator_chain_stage_duration_seconds_count")

        result = orchestrator_api.run_second_stage_pipeline(payload)

        self.assertTrue(result["invoked"])
        self.assertEqual(
            sample_value("orchestrator_chain_stage_duration_seconds_count"),
            before_count + 1,
        )

    def test_derive_recommendation_policy_for_anatomy_only_forces_manual_review(self) -> None:
        result_policy = orchestrator_api.derive_result_policy(
            {
                "report_type": "non-diagnostic-anatomy",
                "diagnostic_support": "anatomy-only",
            }
        )
        policy = orchestrator_api.derive_recommendation_policy(
            {
                "report_type": "non-diagnostic-anatomy",
                "diagnostic_support": "anatomy-only",
            },
            result_policy,
        )
        self.assertTrue(policy["must_include_manual_review"])
        self.assertTrue(policy["force_non_diagnostic_language"])
        self.assertFalse(policy["can_recommend_treatment_change"])

    def test_apply_recommendation_policy_blocks_treatment_change_language(self) -> None:
        summary, policy = orchestrator_api.apply_recommendation_policy(
            {
                "report_type": "preliminary-lung-nodule-screening",
                "diagnostic_support": "screening-only",
                "confidence": 0.94,
            },
            {
                "recommendation": "Start chemotherapy and urgent biopsy based on this result.",
            },
            orchestrator_api.derive_result_policy(
                {
                    "report_type": "preliminary-lung-nodule-screening",
                    "diagnostic_support": "screening-only",
                }
            ),
            {"applied": False, "reasons": []},
        )
        self.assertIn("screening support only", summary["recommendation"].lower())
        self.assertFalse(policy["can_recommend_treatment_change"])

    def test_apply_recommendation_policy_forces_manual_review_language(self) -> None:
        summary, policy = orchestrator_api.apply_recommendation_policy(
            {
                "report_type": "manual-review-required",
                "diagnostic_support": "not-supported",
            },
            {
                "recommendation": "Follow up as needed.",
            },
            orchestrator_api.derive_result_policy(
                {
                    "report_type": "manual-review-required",
                    "diagnostic_support": "not-supported",
                }
            ),
            {"applied": False, "reasons": []},
        )
        self.assertIn("formal specialist review is required", summary["recommendation"].lower())
        self.assertTrue(policy["must_include_manual_review"])

    def test_apply_recommendation_policy_keeps_anatomy_only_language_for_mr(self) -> None:
        summary, _ = orchestrator_api.apply_recommendation_policy(
            {
                "report_type": "non-diagnostic-anatomy",
                "diagnostic_support": "anatomy-only",
            },
            {
                "recommendation": "Use this result as anatomy context only. Pathology interpretation requires radiologist review.",
            },
            orchestrator_api.derive_result_policy(
                {
                    "report_type": "non-diagnostic-anatomy",
                    "diagnostic_support": "anatomy-only",
                }
            ),
            {"applied": False, "reasons": []},
        )
        self.assertIn("anatomy context only", summary["recommendation"].lower())
        self.assertNotIn("screening support only", summary["recommendation"].lower())

    def test_apply_claim_scope_guardrails_forces_non_diagnostic_mr_anatomy_claims(self) -> None:
        summary, guardrails = orchestrator_api.apply_claim_scope_guardrails(
            {
                "report_type": "non-diagnostic-anatomy",
                "diagnostic_support": "anatomy-only",
            },
            {
                "findings": "Possible tumor in the frontal lobe.",
                "impression": "Likely intracranial neoplasm.",
            },
            orchestrator_api.derive_result_policy(
                {
                    "report_type": "non-diagnostic-anatomy",
                    "diagnostic_support": "anatomy-only",
                }
            ),
        )
        self.assertTrue(guardrails["applied"])
        self.assertIn("claim-scope-non-diagnostic", guardrails["reasons"])
        self.assertIn("structural analysis only", summary["findings"].lower())
        self.assertIn("formal specialist interpretation is required", summary["impression"].lower())

    def test_apply_claim_scope_guardrails_softens_generic_screening_specific_structure_claims(self) -> None:
        summary, guardrails = orchestrator_api.apply_claim_scope_guardrails(
            {
                "report_type": "preliminary-2d-screening",
                "diagnostic_support": "screening-only",
            },
            {
                "findings": "Suspicious opacity in the left lower lobe.",
                "impression": "Left lower lobe lesion identified.",
            },
            orchestrator_api.derive_result_policy(
                {
                    "report_type": "preliminary-2d-screening",
                    "diagnostic_support": "screening-only",
                }
            ),
        )
        self.assertTrue(guardrails["applied"])
        self.assertIn("claim-scope-specific-structure-softened", guardrails["reasons"])
        self.assertIn("route-level guardrails limited structure-specific claim wording", summary["findings"].lower())


class BuildStudyWebhookPayloadTests(unittest.TestCase):
    def test_build_study_webhook_payload_sets_inference_status(self) -> None:
        started_at = 0.0
        series_results = [
            {
                "status": "completed",
                "model_id": "demo-model",
                "modality": "CT",
                "body_part": "CHEST",
                "report": {
                    "analysis_type": "screening",
                    "conclusion": "Finding present.",
                    "diagnostic_support": "screening-only",
                    "report_type": "preliminary",
                    "support_matrix": {"diagnostic_available": True},
                    "limitations": [],
                },
                "metadata": {"Modality": "CT", "BodyPartExamined": "CHEST"},
                "metadata_summary": {"study": {"study_instance_uid": "study-1"}},
            },
            {
                "status": "failed",
                "series_uid": "series-2",
                "error": "secondary failure",
            },
        ]
        errors = ["series-2: secondary failure"]

        with patch.object(orchestrator_api, "time") as mocked_time:
            mocked_time.time.return_value = 5.0
            payload = orchestrator_api.build_study_webhook_payload(
                "study-1",
                series_results,
                errors,
                started_at,
                "job-1",
            )

        self.assertEqual(payload["status"], "partial")
        self.assertEqual(payload["inference_status"], "partial")
        self.assertEqual(payload["job_id"], "job-1")

    def test_build_study_webhook_payload_sets_manual_review_when_all_series_fail(self) -> None:
        started_at = 0.0
        series_results = [
            orchestrator_api.build_failed_result(
                filename="series-1.dcm",
                series_uid="series-1",
                study_uid="study-1",
                modality="CT",
                body_part="CHEST",
                error="AI call failed: engine unavailable",
                failure_stage="inference",
                error_category="engine-unavailable",
            )
        ]
        errors = ["series-1: AI call failed: engine unavailable"]

        with patch.object(orchestrator_api, "time") as mocked_time:
            mocked_time.time.return_value = 5.0
            payload = orchestrator_api.build_study_webhook_payload(
                "study-1",
                series_results,
                errors,
                started_at,
                "job-2",
            )

        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["inference_status"], "failed")
        self.assertEqual(
            payload["ai_result"]["report_type"],
            "analysis-failed-manual-review-required",
        )
        self.assertEqual(payload["ai_result"]["diagnostic_support"], "not-supported")
        self.assertIn("Manual specialist review is required", payload["ai_result"]["impression"])
        self.assertEqual(payload["failure_summary"]["failed_series_count"], 1)
        self.assertEqual(payload["failure_summary"]["by_stage"]["inference"], 1)

    def test_build_study_webhook_payload_downgrades_guardrailed_screening_result(self) -> None:
        started_at = 0.0
        series_results = [
            {
                "status": "completed",
                "model_id": "demo-screening-model",
                "modality": "CR",
                "body_part": "CHEST",
                "report": {
                    "analysis_type": "screening",
                    "conclusion": "",
                    "diagnostic_support": "screening-only",
                    "report_type": "preliminary-2d-screening",
                    "support_matrix": {"diagnostic_available": True},
                    "limitations": [],
                    "observations": [],
                    "abnormalities": [],
                    "metrics": {},
                    "abnormality_status": "abnormal",
                    "confidence": 0.91,
                },
                "metadata": {"Modality": "CR", "BodyPartExamined": "CHEST"},
                "metadata_summary": {"study": {"study_instance_uid": "study-1"}},
            }
        ]

        with patch.object(orchestrator_api, "time") as mocked_time:
            mocked_time.time.return_value = 5.0
            payload = orchestrator_api.build_study_webhook_payload(
                "study-1",
                series_results,
                [],
                started_at,
                "job-3",
            )

        self.assertEqual(payload["ai_result"]["report_type"], "manual-review-required")
        self.assertEqual(payload["ai_result"]["diagnostic_support"], "not-supported")
        self.assertIsNone(payload["ai_result"]["abnormal"])
        self.assertIsNone(payload["ai_result"]["confidence"])
        self.assertTrue(payload["ai_result"]["output_guardrails"]["applied"])

    def test_build_study_webhook_payload_hides_confidence_for_xa_policy(self) -> None:
        started_at = 0.0
        series_results = [
            {
                "status": "completed",
                "model_id": "stenunet-xa",
                "modality": "XA",
                "body_part": "HEART",
                "report": {
                    "analysis_type": "screening",
                    "conclusion": "Preliminary angiographic AI screening suggests a suspicious narrowing pattern.",
                    "diagnostic_support": "screening-only",
                    "report_type": "preliminary-stenosis-screening",
                    "support_matrix": {"diagnostic_available": True},
                    "limitations": [],
                    "metrics": {"frame_count": 10, "positive_frames": 4, "max_positive_pixel_ratio": 0.01},
                    "abnormality_status": "abnormal",
                    "confidence": 0.92,
                },
                "metadata": {"Modality": "XA", "BodyPartExamined": "HEART", "SeriesDescription": "XA run"},
                "metadata_summary": {"study": {"study_instance_uid": "study-xa"}},
            }
        ]

        with patch.object(orchestrator_api, "time") as mocked_time:
            mocked_time.time.return_value = 5.0
            payload = orchestrator_api.build_study_webhook_payload(
                "study-xa",
                series_results,
                [],
                started_at,
                "job-xa",
            )

        self.assertIsNone(payload["ai_result"]["confidence"])
        self.assertIsNone(payload["ai_result"]["confidence_band"])
        self.assertFalse(payload["ai_result"]["result_policy"]["can_expose_confidence"])
        self.assertEqual(payload["ai_result"]["confidence_policy"]["expose_mode"], "hidden")
        self.assertTrue(payload["ai_result"]["second_stage_selection_policy"]["should_invoke"])
        self.assertEqual(payload["ai_result"]["second_stage_selection_policy"]["next_stage"], "run-summarizer")
        self.assertTrue(payload["ai_result"]["second_stage_execution"]["invoked"])
        self.assertEqual(payload["ai_result"]["second_stage_execution"]["stage_name"], "run-summarizer")
        self.assertTrue(payload["ai_result"]["second_stage_merge_summary"]["applied"])
        self.assertIn("second-stage xa run summarization", payload["ai_result"]["findings"].lower())

    def test_build_study_webhook_payload_applies_recommendation_policy_to_final_payload(self) -> None:
        started_at = 0.0
        series_results = [
            {
                "status": "completed",
                "model_id": "monai-lung-nodule-retinanet",
                "modality": "CT",
                "body_part": "CHEST",
                "report": {
                    "analysis_type": "screening",
                    "conclusion": "Suspicious pulmonary nodule candidate detected.",
                    "diagnostic_support": "screening-only",
                    "report_type": "preliminary-lung-nodule-screening",
                    "support_matrix": {"diagnostic_available": True},
                    "limitations": [],
                    "metrics": {"candidate_count": 2},
                    "candidate_locations": ["left lower lobe"],
                    "recommendation": "Start chemotherapy immediately.",
                    "abnormality_status": "abnormal",
                    "confidence": 0.95,
                },
                "metadata": {"Modality": "CT", "BodyPartExamined": "CHEST"},
                "metadata_summary": {"study": {"study_instance_uid": "study-ct"}},
            }
        ]

        with patch.object(orchestrator_api, "time") as mocked_time:
            mocked_time.time.return_value = 5.0
            payload = orchestrator_api.build_study_webhook_payload(
                "study-ct",
                series_results,
                [],
                started_at,
                "job-ct",
            )

        self.assertIn("screening support only", payload["ai_result"]["recommendation"].lower())
        self.assertFalse(payload["ai_result"]["recommendation_policy"]["can_recommend_treatment_change"])
        self.assertIn(payload["ai_result"]["recommendation"], payload["ai_result"]["summary"])

    def test_build_study_webhook_payload_exposes_claim_scope_guardrails_for_mr_anatomy(self) -> None:
        started_at = 0.0
        series_results = [
            {
                "status": "completed",
                "model_id": "totalsegmentator-total_mr",
                "modality": "MR",
                "body_part": "BRAIN",
                "report": {
                    "analysis_type": "segmentation",
                    "conclusion": "Possible frontal lobe tumor.",
                    "diagnostic_support": "anatomy-only",
                    "report_type": "non-diagnostic-anatomy",
                    "support_matrix": {"diagnostic_available": False},
                    "limitations": [],
                    "recommendation": "Use this result as anatomy context only.",
                    "abnormality_status": None,
                    "confidence": None,
                },
                "metadata": {"Modality": "MR", "BodyPartExamined": "BRAIN"},
                "metadata_summary": {"study": {"study_instance_uid": "study-mr"}},
            }
        ]

        with patch.object(orchestrator_api, "time") as mocked_time:
            mocked_time.time.return_value = 5.0
            payload = orchestrator_api.build_study_webhook_payload(
                "study-mr",
                series_results,
                [],
                started_at,
                "job-mr",
            )

        self.assertTrue(payload["ai_result"]["claim_scope_guardrails"]["applied"])
        self.assertIn("claim-scope-non-diagnostic", payload["ai_result"]["claim_scope_guardrails"]["reasons"])
        self.assertIn("structural analysis only", payload["ai_result"]["findings"].lower())
        self.assertTrue(any("anatomy-focused ai support only" in item.lower() for item in payload["ai_result"]["limitations"]))
        self.assertTrue(payload["ai_result"]["limitation_policy"]["must_include_manual_review_limitations"])
        self.assertEqual(payload["ai_result"]["finding"], payload["ai_result"]["impression"])
        self.assertIn(payload["ai_result"]["impression"], payload["ai_result"]["summary"])

    def test_build_study_webhook_payload_uses_confidence_band_for_ct(self) -> None:
        started_at = 0.0
        series_results = [
            {
                "status": "completed",
                "model_id": "monai-lung-nodule-retinanet",
                "modality": "CT",
                "body_part": "CHEST",
                "report": {
                    "analysis_type": "screening",
                    "conclusion": "Suspicious pulmonary nodule candidate detected.",
                    "diagnostic_support": "screening-only",
                    "report_type": "preliminary-lung-nodule-screening",
                    "support_matrix": {"diagnostic_available": True},
                    "limitations": [],
                    "metrics": {"candidate_count": 1},
                    "candidate_locations": ["left lower lobe"],
                    "recommendation": "Review with radiologist.",
                    "abnormality_status": "abnormal",
                    "confidence": 0.88,
                },
                "metadata": {"Modality": "CT", "BodyPartExamined": "CHEST"},
                "metadata_summary": {"study": {"study_instance_uid": "study-ct-band"}},
            }
        ]

        with patch.object(orchestrator_api, "time") as mocked_time:
            mocked_time.time.return_value = 5.0
            payload = orchestrator_api.build_study_webhook_payload(
                "study-ct-band",
                series_results,
                [],
                started_at,
                "job-ct-band",
            )

        self.assertIsNone(payload["ai_result"]["confidence"])
        self.assertEqual(payload["ai_result"]["confidence_band"], "high")
        self.assertEqual(payload["ai_result"]["confidence_policy"]["expose_mode"], "qualitative_band")
        self.assertTrue(payload["ai_result"]["model_chain_contract"]["chain_ready"])
        self.assertEqual(payload["ai_result"]["model_chain_contract"]["next_stage"], "candidate-triage")
        self.assertEqual(payload["ai_result"]["normalized_chain_evidence"]["evidence_type"], "candidate-detection")
        self.assertTrue(payload["ai_result"]["normalized_chain_evidence"]["chain_ready"])
        self.assertTrue(payload["ai_result"]["second_stage_selection_policy"]["should_invoke"])
        self.assertEqual(payload["ai_result"]["second_stage_selection_policy"]["decision"], "invoke")
        self.assertTrue(payload["ai_result"]["second_stage_input_payload"]["chain_invocation"]["should_invoke"])
        self.assertEqual(payload["ai_result"]["second_stage_input_payload"]["normalized_evidence"]["evidence_type"], "candidate-detection")
        self.assertTrue(payload["ai_result"]["second_stage_merge_policy"]["can_merge"])
        self.assertEqual(payload["ai_result"]["second_stage_merge_policy"]["merge_mode"], "augment")
        self.assertTrue(payload["ai_result"]["second_stage_execution"]["invoked"])
        self.assertEqual(payload["ai_result"]["second_stage_execution"]["stage_name"], "candidate-triage")
        self.assertTrue(payload["ai_result"]["second_stage_merge_summary"]["applied"])
        self.assertIn("second-stage ct triage", payload["ai_result"]["findings"].lower())

    def test_build_study_webhook_payload_enforces_route_limitations_for_ct(self) -> None:
        started_at = 0.0
        series_results = [
            {
                "status": "completed",
                "model_id": "monai-lung-nodule-retinanet",
                "modality": "CT",
                "body_part": "CHEST",
                "report": {
                    "analysis_type": "screening",
                    "conclusion": "Suspicious pulmonary nodule candidate detected.",
                    "diagnostic_support": "screening-only",
                    "report_type": "preliminary-lung-nodule-screening",
                    "support_matrix": {"diagnostic_available": True},
                    "limitations": ["Bundle-specific note."],
                    "metrics": {"candidate_count": 1},
                    "candidate_locations": ["left lower lobe"],
                    "recommendation": "Review with radiologist.",
                    "abnormality_status": "abnormal",
                    "confidence": 0.88,
                },
                "metadata": {"Modality": "CT", "BodyPartExamined": "CHEST"},
                "metadata_summary": {"study": {"study_instance_uid": "study-ct-limit"}},
            }
        ]

        with patch.object(orchestrator_api, "time") as mocked_time:
            mocked_time.time.return_value = 5.0
            payload = orchestrator_api.build_study_webhook_payload(
                "study-ct-limit",
                series_results,
                [],
                started_at,
                "job-ct-limit",
            )

        self.assertIn("Bundle-specific note.", payload["ai_result"]["limitations"])
        self.assertTrue(any("preliminary chest ct nodule screening support only" in item.lower() for item in payload["ai_result"]["limitations"]))
        self.assertTrue(payload["ai_result"]["limitation_policy"]["must_include_route_limitations"])

    def test_build_study_webhook_payload_marks_anatomy_only_mr_as_not_chain_ready(self) -> None:
        started_at = 0.0
        series_results = [
            {
                "status": "completed",
                "model_id": "totalsegmentator-total_mr",
                "modality": "MR",
                "body_part": "BRAIN",
                "report": {
                    "analysis_type": "segmentation",
                    "conclusion": "Structural anatomy analysis completed.",
                    "diagnostic_support": "anatomy-only",
                    "report_type": "non-diagnostic-anatomy",
                    "support_matrix": {"diagnostic_available": False},
                    "limitations": [],
                    "recommendation": "Use this result as anatomy context only.",
                    "abnormality_status": None,
                    "confidence": None,
                },
                "metadata": {"Modality": "MR", "BodyPartExamined": "BRAIN"},
                "metadata_summary": {"study": {"study_instance_uid": "study-mr-chain"}},
            }
        ]

        with patch.object(orchestrator_api, "time") as mocked_time:
            mocked_time.time.return_value = 5.0
            payload = orchestrator_api.build_study_webhook_payload(
                "study-mr-chain",
                series_results,
                [],
                started_at,
                "job-mr-chain",
            )

        self.assertFalse(payload["ai_result"]["model_chain_contract"]["chain_ready"])
        self.assertEqual(payload["ai_result"]["model_chain_contract"]["chain_stage"], "non-diagnostic")
        self.assertEqual(payload["ai_result"]["normalized_chain_evidence"]["evidence_type"], "none")
        self.assertFalse(payload["ai_result"]["normalized_chain_evidence"]["chain_ready"])
        self.assertFalse(payload["ai_result"]["second_stage_selection_policy"]["should_invoke"])
        self.assertEqual(payload["ai_result"]["second_stage_selection_policy"]["decision"], "blocked")
        self.assertFalse(payload["ai_result"]["second_stage_input_payload"]["chain_invocation"]["should_invoke"])
        self.assertEqual(payload["ai_result"]["second_stage_input_payload"]["normalized_evidence"]["evidence_type"], "none")
        self.assertFalse(payload["ai_result"]["second_stage_merge_policy"]["can_merge"])
        self.assertEqual(payload["ai_result"]["second_stage_merge_policy"]["merge_mode"], "blocked")
        self.assertFalse(payload["ai_result"]["second_stage_execution"]["invoked"])
        self.assertFalse(payload["ai_result"]["second_stage_merge_summary"]["applied"])

    def test_build_study_webhook_payload_executes_us_second_stage(self) -> None:
        started_at = 0.0
        series_results = [
            {
                "status": "completed",
                "model_id": "us-echo-lite-lv-function",
                "modality": "US",
                "body_part": "HEART",
                "report": {
                    "analysis_type": "screening",
                    "conclusion": "Automated screening suggests reduced left ventricular systolic function.",
                    "diagnostic_support": "screening-only",
                    "report_type": "preliminary-echocardiography-lv-function",
                    "support_matrix": {"diagnostic_available": True},
                    "limitations": [],
                    "metrics": {
                        "estimated_ef_percent": 35.0,
                        "fractional_area_change": None,
                        "frame_time_ms": 33.0,
                    },
                    "recommendation": "Correlate with cardiology review.",
                    "abnormality_status": "abnormal",
                    "confidence": 0.84,
                },
                "metadata": {"Modality": "US", "BodyPartExamined": "HEART", "SeriesDescription": "Echo cine"},
                "metadata_summary": {"study": {"study_instance_uid": "study-us-chain"}},
            }
        ]

        with patch.object(orchestrator_api, "time") as mocked_time:
            mocked_time.time.return_value = 5.0
            payload = orchestrator_api.build_study_webhook_payload(
                "study-us-chain",
                series_results,
                [],
                started_at,
                "job-us-chain",
            )

        self.assertTrue(payload["ai_result"]["second_stage_selection_policy"]["should_invoke"])
        self.assertEqual(payload["ai_result"]["second_stage_selection_policy"]["next_stage"], "cardiac-summary-classifier")
        self.assertTrue(payload["ai_result"]["second_stage_execution"]["invoked"])
        self.assertEqual(payload["ai_result"]["second_stage_execution"]["stage_name"], "cardiac-summary-classifier")
        self.assertTrue(payload["ai_result"]["second_stage_merge_summary"]["applied"])
        self.assertIn("second-stage cardiac summary", payload["ai_result"]["findings"].lower())

    def test_build_study_webhook_payload_executes_mr_tumor_second_stage(self) -> None:
        started_at = 0.0
        series_results = [
            {
                "status": "completed",
                "model_id": "mr-brats-segmentation",
                "modality": "MR",
                "body_part": "BRAIN",
                "report": {
                    "analysis_type": "segmentation",
                    "conclusion": "Screening brain MRI segmentation suggests a moderate candidate tumor burden.",
                    "diagnostic_support": "screening-only",
                    "report_type": "preliminary-brain-tumor-segmentation",
                    "support_matrix": {"diagnostic_available": True},
                    "limitations": [],
                    "metrics": {
                        "whole_tumor_voxels": 45000,
                        "whole_tumor_ratio": 0.04,
                        "tumor_core_voxels": 12000,
                        "enhancing_tumor_voxels": 3000,
                    },
                    "recommendation": "Correlate with neuroradiology review.",
                    "abnormality_status": "abnormal",
                    "confidence": 0.87,
                    "routing_decision": {"route_name": "mr-brats-segmentation"},
                },
                "metadata": {"Modality": "MR", "BodyPartExamined": "BRAIN", "SeriesDescription": "Brain MRI"},
                "metadata_summary": {"study": {"study_instance_uid": "study-mr-tumor-chain"}},
            }
        ]

        with patch.object(orchestrator_api, "time") as mocked_time:
            mocked_time.time.return_value = 5.0
            payload = orchestrator_api.build_study_webhook_payload(
                "study-mr-tumor-chain",
                series_results,
                [],
                started_at,
                "job-mr-tumor-chain",
            )

        self.assertTrue(payload["ai_result"]["second_stage_selection_policy"]["should_invoke"])
        self.assertEqual(payload["ai_result"]["second_stage_selection_policy"]["next_stage"], "lesion-burden-interpreter")
        self.assertTrue(payload["ai_result"]["second_stage_execution"]["invoked"])
        self.assertEqual(payload["ai_result"]["second_stage_execution"]["stage_name"], "lesion-burden-interpreter")
        self.assertTrue(payload["ai_result"]["second_stage_merge_summary"]["applied"])
        self.assertIn("second-stage brain mri burden interpretation", payload["ai_result"]["findings"].lower())
        self.assertTrue(payload["ai_result"]["chain_observability"]["second_stage_invoked"])
        self.assertTrue(payload["ai_result"]["chain_observability"]["merge_applied"])
        self.assertEqual(payload["ai_result"]["chain_observability"]["second_stage_name"], "lesion-burden-interpreter")


class ProcessingValidationTests(unittest.TestCase):
    def test_process_single_file_downgrades_missing_required_metadata(self) -> None:
        dicom_bytes = b"fake-dicom"
        metadata = {
            "StudyInstanceUID": "study-1",
            "Modality": "CT",
        }
        with patch.object(orchestrator_api, "extract_full_dicom_metadata", return_value=metadata):
            result = orchestrator_api.process_single_file("file-1.dcm", dicom_bytes, source="pacs")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_stage"], "input-validation")
        self.assertEqual(result["error_category"], "invalid-dicom-metadata")
        self.assertTrue(result["manual_review_required"])

    def test_process_single_file_downgrades_invalid_engine_response(self) -> None:
        dicom_bytes = b"fake-dicom"
        metadata = {
            "StudyInstanceUID": "study-1",
            "SeriesInstanceUID": "series-1",
            "Modality": "CT",
            "BodyPartExamined": "CHEST",
        }
        with patch.object(orchestrator_api, "extract_full_dicom_metadata", return_value=metadata), patch.object(
            orchestrator_api,
            "post_to_ai_engine",
            return_value={"model_id": "demo-model", "report": {"analysis_type": "", "conclusion": "", "observations": []}},
        ):
            result = orchestrator_api.process_single_file("file-1.dcm", dicom_bytes, source="pacs")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_stage"], "response-validation")
        self.assertEqual(result["error_category"], "empty-engine-response")
        self.assertTrue(result["manual_review_required"])

    def test_process_series_downgrades_missing_required_metadata(self) -> None:
        metadata = {
            "StudyInstanceUID": "study-1",
            "SOPInstanceUID": "sop-1",
            "BodyPartExamined": "HEART",
        }
        series_files = [{"filename": "frame-1.dcm", "content": b"fake-dicom"}]
        with patch.object(orchestrator_api, "extract_full_dicom_metadata", return_value=metadata):
            result = orchestrator_api.process_series(series_files, source="pacs")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_stage"], "input-validation")
        self.assertEqual(result["error_category"], "invalid-dicom-metadata")

    def test_process_series_downgrades_non_dict_engine_response(self) -> None:
        metadata = {
            "StudyInstanceUID": "study-1",
            "SeriesInstanceUID": "series-1",
            "SOPInstanceUID": "sop-1",
            "Modality": "XA",
            "BodyPartExamined": "HEART",
        }
        series_files = [{"filename": "frame-1.dcm", "content": b"fake-dicom"}]
        with patch.object(orchestrator_api, "extract_full_dicom_metadata", return_value=metadata), patch.object(
            orchestrator_api,
            "post_series_to_ai_engine",
            return_value="bad-response",
        ):
            result = orchestrator_api.process_series(series_files, source="pacs")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_stage"], "response-validation")
        self.assertEqual(result["error_category"], "invalid-engine-response")


if __name__ == "__main__":
    unittest.main()
