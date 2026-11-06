"""What kind of DICOM object a SOP class is (SDD-SCANSURE-AI-001 T7).

Series assembly, primary-series selection and the fetcher consider IMAGES only. A study also
holds structured reports, segmentations, presentation states, key-object selections,
radiotherapy objects, encapsulated PDFs and waveforms; sent to an image model those are
analysed as if they were pictures, or fail in a way that reads like a broken study.

    object_kind(sop_class_uid) -> image / multiframe / enhanced
                                  sr / seg / pr / ko / rt / pdf / waveform / other
                                  None when there is no SOP class to classify

This file is a deliberate duplicate of shared/python/object_kind/object_kind.py in the
ScanSureX monorepo (owner decision: duplicate, do not share a package across repositories).
The two must stay in step; each carries the same table-driven test.
"""
from __future__ import annotations

from typing import Iterable, Optional

_STORAGE = "1.2.840.10008.5.1.4.1.1."
ENHANCED = {_STORAGE + suffix for suffix in (
    "2.1", "2.2",               # Enhanced CT, Legacy Converted Enhanced CT
    "4.1", "4.3", "4.4",        # Enhanced MR, Enhanced MR Color, Legacy Converted Enhanced MR
    "130", "128.1",             # Enhanced PET, Legacy Converted Enhanced PET
    "6.2",                      # Enhanced US Volume
    "12.1.1", "12.2.1",         # Enhanced XA, Enhanced XRF
    "13.1.1", "13.1.2", "13.1.3",  # X-Ray 3D angiographic / craniofacial, Breast Tomosynthesis
)}
MULTIFRAME = {_STORAGE + suffix for suffix in (
    "3.1",                      # Ultrasound Multi-frame
    "7.1", "7.2", "7.3", "7.4",  # Multi-frame Secondary Capture (1-bit, grey byte/word, colour)
    "77.1.1.1", "77.1.2.1", "77.1.4.1",  # Video endoscopic / microscopic / photographic
)}
IMAGE = {_STORAGE + suffix for suffix in (
    "1", "1.1", "1.1.1",        # CR, DX presentation / processing
    "1.2", "1.2.1",             # Mammography
    "1.3", "1.3.1",             # Intra-oral (dental)
    "2", "4",                   # CT, MR
    "6.1",                      # Ultrasound
    "7",                        # Secondary Capture
    "12.1", "12.2",             # XA, RF
    "20",                       # Nuclear Medicine
    "128",                      # PET
    "77.1.1", "77.1.2", "77.1.3", "77.1.4",  # VL endoscopic, microscopic, slide, photographic
    "77.1.5.1", "77.1.5.2", "77.1.5.4",      # Ophthalmic photography 8/16 bit, OPT
    "77.1.6",                   # VL Whole Slide Microscopy
)}
KEY_OBJECT = _STORAGE + "88.59"
ENCAPSULATED_PDF = _STORAGE + "104.1"
SEGMENTATION = {_STORAGE + "66.4", _STORAGE + "66.5", _STORAGE + "66.7"}

IMAGE_KINDS = frozenset({"image", "multiframe", "enhanced"})

# Modality codes of objects that are not images. Used only to pick a modality HINT from a
# study's ModalitiesInStudy; what is analysed is decided per instance by SOP class.
NON_IMAGE_MODALITIES = frozenset({
    "SR", "KO", "PR", "SEG", "DOC", "REG", "FID", "PLAN", "M3D",
    "RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD", "RTIMAGE",
    "ECG", "HD", "AU", "EPS", "RESP", "BDUS",
})


def object_kind(sop_class_uid: Optional[str], frames: Optional[int] = None) -> Optional[str]:
    """image / multiframe / enhanced / sr / seg / pr / ko / rt / pdf / waveform / other.

    None when there is no SOP class to classify: unknown is not "other".
    """
    uid = str(sop_class_uid or "").strip()
    if not uid:
        return None
    if uid == KEY_OBJECT:
        return "ko"
    if uid.startswith(_STORAGE + "88."):
        return "sr"
    if uid in SEGMENTATION:
        return "seg"
    if uid.startswith(_STORAGE + "11."):
        return "pr"
    if uid.startswith(_STORAGE + "481."):
        return "rt"
    if uid == ENCAPSULATED_PDF:
        return "pdf"
    if uid.startswith(_STORAGE + "9."):
        return "waveform"
    if uid in ENHANCED:
        return "enhanced"
    if uid in MULTIFRAME:
        return "multiframe"
    if uid in IMAGE:
        return "multiframe" if (frames or 0) > 1 else "image"
    return "other"


def excluded_kind(sop_class_uid: Optional[str], modality: Optional[str] = None) -> Optional[str]:
    """The kind to record when this object must stay out of series assembly, else None.

    An object is kept only when its SOP class is a known image class. A SOP class this table
    does not know is excluded as "other" and listed, never analysed on a guess. The one case
    kept without a classification is an object whose SOP class the caller could not supply
    at all: there is nothing to judge it by, and dropping it would empty every study from an
    archive that omits the attribute. The SR modality check is the rule that was here before
    this helper and still applies in that case.
    """
    kind = object_kind(sop_class_uid)
    if str(modality or "").strip().upper() == "SR":
        return kind if kind and kind not in IMAGE_KINDS else "sr"
    if kind is None or kind in IMAGE_KINDS:
        return None
    return kind


def first_image_modality(modalities: Optional[Iterable], default: str = "UNKNOWN") -> str:
    """The first modality in ModalitiesInStudy that is an image modality.

    A study listed as ["SR", "CT"] is a CT study with a report in it, not an SR study.
    """
    for value in modalities or []:
        code = str(value or "").strip().upper()
        if code and code not in NON_IMAGE_MODALITIES:
            return code
    return default
