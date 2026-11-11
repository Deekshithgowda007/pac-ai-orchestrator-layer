"""The StudyProfile vocabulary a manifest may constrain on (SDD-SCANSURE-AI-002 §2.3, §3.3).

Deliberate duplicate of the closed sets in the ScanSureX monorepo's
``shared/python/study_profile/study_profile.py`` (owner decision, as for ``object_kind``: no
shared package across repositories). The two must stay in step; ``tests/test_manifest_registry.py``
pins every member a committed manifest uses, so a manifest naming a value the profiler can
never produce is refused at load time rather than silently matching nothing.
"""
from __future__ import annotations

PROFILE_SCHEMA = "scansurex.study_profile/1"

PROTOCOL_FAMILIES = frozenset({
    "cr_chest", "cr_abdomen", "cr_extremity", "cr_spine", "cr_other",
    "dx_mammo",
    "ct_head", "ct_chest", "ct_abdomen_pelvis", "ct_spine", "ct_extremity", "ct_angio",
    "cbct_dental",
    "mr_brain_structural", "mr_brain_multisequence", "mr_spine", "mr_msk", "mr_angio",
    "us_echo", "us_abdomen", "us_obstetric",
    "xa_coronary",
    "nm_pet_ct",
    "opg",
    "unknown",
})

DIMENSIONALITIES = frozenset({"2d", "2d_multiframe", "3d", "4d", "cine", "unknown"})
AGE_BANDS = frozenset({"neonate", "child", "adolescent", "adult", "unknown"})
BODY_PARTS = frozenset({
    "chest", "head", "neck", "abdomen", "pelvis", "abdomen_pelvis", "spine", "extremity",
    "breast", "heart", "jaw", "whole_body", "fetus", "other",
})
MODALITIES = frozenset({
    "CR", "DX", "MG", "IO", "PX", "CT", "MR", "US", "XA", "RF", "NM", "PT", "OT", "SC", "ES", "XC", "OP", "OCT",
})

#: The value of a profile field that means "the headers did not say". It never satisfies a
#: constraint: absence of signal is not a match (CLAUDE.md invariant, SDD-002 §3.3 rule 1).
UNKNOWN = frozenset({"unknown", None})
