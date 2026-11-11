# Candidate manifests — committed, disabled, and what each needs before it may be enabled

SDD-SCANSURE-AI-002, rung-1 widening. The product requirement is that every study gets useful AI
output; the engineering position is that no verdict or confidence ever comes from a model that
did not run. Rung 3 (the study descriptor) already satisfies the first for every readable study.
Widening rung 1 means adding *models*, and a model enters this deployment in three steps that
are each visible in Git:

1. **Manifest committed, not in `registry.lock.json`** (this document). This changes **no routing
   outcome** in the shipped lock, and `tests/test_candidate_manifests.py` pins that: the anatomy
   routes accept every CT/MR volume and the generic Hugging Face routes every 2D image, so a
   disabled candidate is shadowed by an enabled catch-all. It surfaces as `model_disabled` only in a
   lock without those catch-alls (an empty, radiologist-only lock), where routing then names the
   model that exists for the study instead of saying no model exists. Making the ladder list
   "would accept, disabled" rows alongside the enabled catch-alls is a registered follow-up, not
   part of this step.
2. **Weights staged and the licence reviewed.** `runtime.weights_sha256` filled, `license.code`
   changed from `UNVERIFIED` to the reviewed SPDX identifier (or the vendor's terms, named), a
   `dpa_reference` where the terms require one.
3. **Validation report committed** (`validation.report` points at it; `datasets`, `metrics`,
   `internal_concordance` filled from it), the adapter written and in `AdapterRegistry.adapters`,
   then the lock entry. `tests/test_candidate_manifests.py` refuses the lock entry until 2 and 3
   are true, and refuses an enabled manifest whose adapter the image does not have.

None of this changes a model's tier. Every candidate is T3 and `screening_only`; T1 needs a
regulator authorisation on the manifest (`license.authorization`), which no open model has.

## Vocabulary note

`license.code: "UNVERIFIED"` is a deliberate sentinel: it means *nobody has yet read the terms
the weights come under*. It is refused in the lock by test. It is not a licence.

## The candidates

| model_id | Study it would accept | Output kind | Source family | Licence state | Weights env | Adapter to write | Blockers before enablement |
|---|---|---|---|---|---|---|---|
| `rsna-ich-ct-head` | non-contrast adult head CT (`ct_head`, 3d / multiframe) | classification: abnormal, confidence, subtypes | RSNA Intracranial Hemorrhage Detection challenge family (2019) | **UNVERIFIED** — the challenge data are non-commercial; open re-implementations carry their own, differing licences | `CT_HEAD_ICH_CHECKPOINT_PATH` | `ct_head_ich` | licence of a specific weight release; validation on an Indian non-contrast head CT sample with radiologist ground truth; a "window/level and slice-ordering" check in the adapter, since ICH models are sensitive to both |
| `mrnet-knee-mr` | adult / adolescent knee MR (`mr_msk`, 3d, extremity) | classification: abnormal, confidence (abnormality, ACL, meniscus) | Stanford MRNet (2018) | **UNVERIFIED** — Stanford research-use agreement on the data; no vendor weights | `MR_KNEE_MRNET_CHECKPOINT_PATH` | `mr_knee_mrnet` | weights under a reviewed licence; sequence selection (sagittal/coronal/axial) in the adapter; validation report |
| `mura-msk-radiograph` | adult extremity radiograph (`cr_extremity`, 2d) | classification: abnormal, confidence | Stanford MURA (2017) | **UNVERIFIED** — research-use agreement | `CR_MSK_MURA_CHECKPOINT_PATH` | `cr_msk_mura` | as above; per-region performance (MURA performance differs markedly by body region, the report must say so per region) |
| `monai-vista3d` | adult CT, any protocol family, 3d | anatomy: structures, organ volumes (no verdict) | MONAI model zoo VISTA-3D | Apache-2.0 (to be confirmed against the checkpoint release notes) | `CT_VISTA3D_CHECKPOINT_PATH` | `ct_vista3d` | GPU on the inference host (owner decision outstanding, see D-GPU below); checkpoint staged with sha256; concordance against TotalSegmentator on the same studies |
| `toothfairy-cbct-dental` | dental CBCT (`cbct_dental`, 3d, jaw) | anatomy: teeth / canal / jaw structures (no verdict) | ToothFairy challenge family (MICCAI) | **UNVERIFIED** — challenge submissions vary; some are non-commercial | `CBCT_DENTAL_CHECKPOINT_PATH` | `cbct_dental_seg` | licence of the chosen submission; CBCT field-of-view handling; validation report |
| `medgemma-4b-it` | adult 2D CR/DX/CT/MR images | narrative: findings text only | Google MedGemma 4B (instruction-tuned) | `HAI-DEF-terms` — Health AI Developer Foundations terms, acceptance by the owner required | `MEDGEMMA_MODEL_DIR` | `vlm_medgemma` | owner accepts the terms; GPU; a prompt and output contract that produces *description* and never a diagnosis sentence; validation of the text against radiologist reports (hallucination rate is the metric that matters here, not AUC) |

What the table does **not** contain: OPG (panoramic dental radiograph) and obstetric ultrasound.
No open model with a reviewable licence was identified for either; both keep returning
`no_manifest_for_profile`, which is the honest state, and rung 3 still describes them.

## Why the mixed-licence candidates are committed at all

Because the alternative is a verbal list. A manifest that is in Git, validated at load time against
the profiler's vocabulary, and visible in the ladder as `model_disabled` is a tracked decision with a
test guarding its enablement. A model named only in a chat or a document is neither.

## Outstanding owner decisions this widening depends on

- **D-GPU** — whether the pilot inference host gets a GPU. `monai-vista3d` and `medgemma-4b-it`
  are `runtime.device: gpu`; on CPU they would time out or be unusable. The current pilot host has
  none (`docs/analysis/20-migration-log.md`, finding B2/F12 in the monorepo).
- **D-MedGemma** — acceptance of the Health AI Developer Foundations terms in the owner's name.
- **D-Validation-set** — which de-identified study sets with radiologist ground truth the validation
  reports are produced on. Nothing identifiable goes into Git or the corpus bucket; the report
  carries counts and metrics, never studies.

## How to enable one, when its day comes

1. Stage the weights on the host; put the path in the env var the manifest names; compute the
   sha256 and put it in `runtime.weights_sha256`.
2. Replace `license.code: "UNVERIFIED"` with the reviewed identifier; fill `dpa_reference` if the
   terms need one.
3. Commit the validation report under `docs/validation/<model_id>-<version>.md` and point
   `validation.report` at it; fill `datasets`, `metrics`, `internal_concordance`.
4. Write the adapter, register it under the manifest's `runtime.adapter` name.
5. Add the lock entry. Run `python -m pytest -q tests`: `test_candidate_manifests.py` is what
   must now pass that previously would have refused.
6. Run the monorepo's `scripts/e2e/run-integration2.sh` with a study of that type and confirm the
   ladder row reads `ran` and the SR's series description names `<model_id>@<version>`.
