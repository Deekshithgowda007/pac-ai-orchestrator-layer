"""StudyProfile: one header-derived description of a study (SDD-SCANSURE-AI-002 T1, Pillar B).

Routing a study to an AI model is a decision about what the study IS: its modality, body
part, dimensionality, protocol family, laterality, contrast, and which of its series are
images at all. Today those facts are re-derived in several places from one instance's header
(or from a filename, or from the upload form). This module derives them once, from the
headers of every instance, and never reads pixel data.

    build_study_profile(datasets) -> dict        # pydicom Datasets read with stop_before_pixels
    profile_from_bytes(blobs)     -> dict        # convenience for services holding raw instances

Three rules, each pinned by a test:

1. A fact the headers do not carry is ``None`` / ``"unknown"``. Nothing is guessed from a
   filename, the upload form, or a default. Where a value IS inferred from header keywords
   (protocol_family from SeriesDescription), ``derivation.warnings`` says so.
2. Nothing identifying leaves the header: no name, identifier, date, accession or
   institution. Age is carried as a band, never as a value or a birth date. Sex is carried
   only because the PC-PNDT rule (Phase 4) needs it.
3. The series inventory classifies objects with ``object_kind``; non-image objects (reports,
   segmentations, presentation states ...) are listed as ``excluded_objects`` and contribute
   nothing to modality, body part or dimensionality.

Deviation from SDD-002 section 2.3, recorded here and in the task log: ``contrast`` is its own
field and is NOT folded into the protocol family (``ct_chest`` rather than
``ct_chest_noncontrast`` / ``ct_chest_contrast``). ContrastBolusAgent is routinely absent on
non-contrast acquisitions, so a combined family would read ``unknown`` for most chest CTs and
a manifest could never match them; two orthogonal dimensions say the same thing exactly.

Deliberate duplicate of the ScanSureX monorepo's ``shared/python/study_profile/study_profile.py``
(owner decision, as for ``object_kind``: no shared package across repositories). Identical below
the docstring except the import line; keep the two in step.
"""
from __future__ import annotations

import io
import re
from collections import OrderedDict
from datetime import date
from typing import Any, Iterable, Optional

import pydicom

from models.object_kind import IMAGE_KINDS, excluded_kind, object_kind

SCHEMA = "scansurex.study_profile/1"
DERIVATION_VERSION = "1.0.0"

#: The closed set a manifest may name in ``intended_use.protocol_families``. Extending it is a
#: change to this file, never a string in a config. ``unknown`` never matches a constraint.
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

DIMENSIONALITIES = ("2d", "2d_multiframe", "3d", "4d", "cine", "unknown")
AGE_BANDS = ("neonate", "child", "adolescent", "adult", "unknown")

# BodyPartExamined (0018,0015) defined terms and the common free-text variants seen in the
# Phase 1 corpus work, to a small closed vocabulary. Anything else is "other" with a warning
# and the raw value kept, so a reader can extend the table rather than guess.
_BODY_PART_TABLE: dict[str, str] = {
    "CHEST": "chest", "THORAX": "chest", "LUNG": "chest", "LUNGS": "chest", "CHESTABDOMEN": "chest",
    "HEAD": "head", "BRAIN": "head", "SKULL": "head", "CRANIUM": "head",
    "NECK": "neck", "CERVICAL": "neck",
    "ABDOMEN": "abdomen", "LIVER": "abdomen", "KIDNEY": "abdomen", "PANCREAS": "abdomen",
    "PELVIS": "pelvis", "ABDOMENPELVIS": "abdomen_pelvis", "ABDOMEN PELVIS": "abdomen_pelvis",
    "SPINE": "spine", "CSPINE": "spine", "TSPINE": "spine", "LSPINE": "spine", "SSPINE": "spine",
    "LSSPINE": "spine", "WHOLESPINE": "spine", "CERVICAL SPINE": "spine", "LUMBAR SPINE": "spine",
    "EXTREMITY": "extremity", "KNEE": "extremity", "HAND": "extremity", "FOOT": "extremity",
    "SHOULDER": "extremity", "HIP": "extremity", "ELBOW": "extremity", "WRIST": "extremity",
    "ANKLE": "extremity", "LEG": "extremity", "ARM": "extremity", "FEMUR": "extremity",
    "TIBIA": "extremity", "HUMERUS": "extremity", "CLAVICLE": "extremity", "FINGER": "extremity",
    "TOE": "extremity", "LOWER LIMB": "extremity", "UPPER LIMB": "extremity",
    "BREAST": "breast",
    "HEART": "heart", "CARDIAC": "heart",
    "JAW": "jaw", "MANDIBLE": "jaw", "MAXILLA": "jaw", "TEETH": "jaw", "TOOTH": "jaw",
    "DENTAL": "jaw", "TMJ": "jaw", "ORAL": "jaw",
    "WHOLEBODY": "whole_body", "WHOLE BODY": "whole_body",
    "UTERUS": "pelvis", "PROSTATE": "pelvis", "BLADDER": "pelvis",
    "FETUS": "fetus", "FETAL": "fetus",
}

_MULTISEQUENCE_TOKENS = ("t1", "t2", "flair")
_ANGIO_TOKENS = ("angio", "cta", "mra", "tof", "angiogram")
_CARDIAC_TOKENS = ("echo", "cardiac", "heart", "lv ", "ventric")
_OBSTETRIC_TOKENS = ("obstet", "fetal", "fetus", "pregnan", "gestat", "ob ", "anomaly scan", "nt scan")
_CORONARY_TOKENS = ("coronar", "lca", "rca", "lad", "cath")
_CBCT_TOKENS = ("cbct", "cone beam", "conebeam", "dental", "i-cat", "icat", "planmeca", "carestream cs")
_OPG_TOKENS = ("pano", "opg", "orthopan")
_CONTRAST_YES = ("contrast", "c+", "with iv", "post con", "postcon", "arterial", "venous", "portal", "delayed")
_CONTRAST_NO = ("non-contrast", "noncontrast", "non contrast", "w/o", "without contrast", "unenhanced", "c-", "nc ")

_LATERALITY = {"L": "L", "R": "R", "B": "B", "LEFT": "L", "RIGHT": "R", "BOTH": "B", "BILATERAL": "B"}


def _s(value: Any) -> Optional[str]:
    """A stripped string, or None for absent / empty."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _get(ds: pydicom.Dataset, keyword: str) -> Any:
    try:
        value = getattr(ds, keyword, None)
    except Exception:  # a malformed element must not take the profile down
        return None
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return list(value) if len(value) else None
    return value if _s(value) is not None else None


def _float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_body_part(raw: Optional[str]) -> Optional[str]:
    """``None`` when absent; a vocabulary term when known; ``"other"`` when not."""
    text = _s(raw)
    if text is None:
        return None
    key = re.sub(r"[^A-Z ]", "", text.upper()).strip()
    compact = key.replace(" ", "")
    return _BODY_PART_TABLE.get(key) or _BODY_PART_TABLE.get(compact) or "other"


def orientation_of(iop: Any) -> Optional[str]:
    """axial / coronal / sagittal / oblique from ImageOrientationPatient; None when absent."""
    if not iop or len(iop) != 6:
        return None
    try:
        row = [float(v) for v in iop[:3]]
        col = [float(v) for v in iop[3:]]
    except (TypeError, ValueError):
        return None
    normal = [
        row[1] * col[2] - row[2] * col[1],
        row[2] * col[0] - row[0] * col[2],
        row[0] * col[1] - row[1] * col[0],
    ]
    magnitudes = [abs(component) for component in normal]
    axis = magnitudes.index(max(magnitudes))
    if max(magnitudes) < 0.8:          # no axis dominates: an oblique acquisition
        return "oblique"
    return ("sagittal", "coronal", "axial")[axis]


def age_band(ds: pydicom.Dataset, warnings: list[str]) -> str:
    """A band from PatientAge, or from PatientBirthDate + StudyDate; the dates are not kept."""
    age = _s(_get(ds, "PatientAge"))
    years: Optional[float] = None
    if age:
        match = re.fullmatch(r"(\d{1,3})([DWMY])", age.upper())
        if match:
            amount, unit = int(match.group(1)), match.group(2)
            years = {"D": amount / 365.25, "W": amount * 7 / 365.25, "M": amount / 12, "Y": float(amount)}[unit]
        else:
            warnings.append("PatientAge present but not in DICOM AS form; age band unknown")
    if years is None:
        birth, study = _s(_get(ds, "PatientBirthDate")), _s(_get(ds, "StudyDate"))
        if birth and study:
            try:
                b = date(int(birth[:4]), int(birth[4:6]), int(birth[6:8]))
                s = date(int(study[:4]), int(study[4:6]), int(study[6:8]))
                years = (s - b).days / 365.25
            except ValueError:
                warnings.append("PatientBirthDate/StudyDate unparseable; age band unknown")
    if years is None:
        return "unknown"
    if years < 0:
        warnings.append("age computes negative; age band unknown")
        return "unknown"
    if years * 365.25 < 28:
        return "neonate"
    if years < 12:
        return "child"
    if years < 18:
        return "adolescent"
    return "adult"


def _contrast(ds_list: list[pydicom.Dataset], descriptions: str, warnings: list[str]) -> Optional[bool]:
    for ds in ds_list:
        if _s(_get(ds, "ContrastBolusAgent")):
            return True
    if any(token in descriptions for token in _CONTRAST_NO):
        warnings.append("contrast=false inferred from series/protocol description")
        return False
    if any(token in descriptions for token in _CONTRAST_YES):
        warnings.append("contrast=true inferred from series/protocol description")
        return True
    return None


def _series_entry(series_uid: Optional[str], members: list[pydicom.Dataset], warnings: list[str]) -> dict[str, Any]:
    first = members[0]
    sop_class = _s(_get(first, "SOPClassUID"))
    frames_each = [_int(_get(ds, "NumberOfFrames")) or 1 for ds in members]
    kind = object_kind(sop_class, frames_each[0])
    modality = _s(_get(first, "Modality"))
    modality = modality.upper() if modality else None
    kinds = {object_kind(_s(_get(ds, "SOPClassUID")), f) for ds, f in zip(members, frames_each)}
    if len(kinds) > 1:
        warnings.append(f"series {series_uid or '<no uid>'} mixes object kinds {sorted(k or 'none' for k in kinds)}")
    excluded = excluded_kind(sop_class, modality)
    spacing = _get(first, "PixelSpacing")
    spacing_mm = None
    if isinstance(spacing, list) and len(spacing) == 2:
        parsed = [_float(v) for v in spacing]
        spacing_mm = parsed if None not in parsed else None
    ts = None
    file_meta = getattr(first, "file_meta", None)
    if file_meta is not None:
        ts = _s(getattr(file_meta, "TransferSyntaxUID", None))
    return {
        "series_uid": series_uid,
        "modality": modality,
        "object_kind": kind,
        "excluded_as": excluded,
        "instances": len(members),
        "frames": sum(frames_each),
        "transfer_syntax": ts,
        "description": _s(_get(first, "SeriesDescription")),
        "protocol_name": _s(_get(first, "ProtocolName")),
        "slice_thickness_mm": _float(_get(first, "SliceThickness")),
        "pixel_spacing_mm": spacing_mm,
        "orientation": orientation_of(_get(first, "ImageOrientationPatient")),
        "temporal_positions": _int(_get(first, "NumberOfTemporalPositions")),
        "distinct_positions": len({tuple(_s(v) for v in (_get(ds, "ImagePositionPatient") or [])) for ds in members
                                   if _get(ds, "ImagePositionPatient")}),
    }


def _dimensionality(image_series: list[dict[str, Any]]) -> str:
    if not image_series:
        return "unknown"
    modalities = {s["modality"] for s in image_series}
    max_frames = max(s["frames"] for s in image_series)
    if any((s["temporal_positions"] or 0) > 1 for s in image_series):
        return "4d"
    if modalities & {"US", "XA", "RF"} and max_frames > 1:
        return "cine"
    for s in image_series:
        if s["distinct_positions"] >= 3:
            return "3d"
        if s["object_kind"] == "enhanced" and s["frames"] >= 3 and s["modality"] in {"CT", "MR", "PT", "NM"}:
            return "3d"
        if s["instances"] >= 3 and s["modality"] in {"CT", "MR", "PT", "NM"} and s["slice_thickness_mm"]:
            return "3d"
    if max_frames > 1:
        return "2d_multiframe"
    return "2d"


def _protocol_family(modality: Optional[str], body: Optional[str], image_series: list[dict[str, Any]],
                     text: str, warnings: list[str]) -> str:
    def inferred(family: str, why: str) -> str:
        warnings.append(f"protocol_family={family} inferred from {why}")
        return family

    if modality is None or modality == "MIXED":
        return "unknown"
    if modality == "PX":
        return "opg"
    if modality in {"CR", "DX", "IO"} and any(t in text for t in _OPG_TOKENS):
        return inferred("opg", "series description")
    if modality in {"CR", "DX"}:
        if body == "breast":
            return "dx_mammo"
        if body in {"chest", "abdomen", "extremity", "spine"}:
            return f"cr_{body}"
        if body is not None:
            return "cr_other"
        return "unknown"
    if modality == "MG":
        return "dx_mammo"
    if modality == "CT":
        if body == "jaw":
            return "cbct_dental"
        if any(t in text for t in _CBCT_TOKENS):
            return inferred("cbct_dental", "description/manufacturer tokens")
        if any(t in text for t in _ANGIO_TOKENS):
            return inferred("ct_angio", "series description")
        return {"head": "ct_head", "chest": "ct_chest", "abdomen": "ct_abdomen_pelvis", "pelvis": "ct_abdomen_pelvis",
                "abdomen_pelvis": "ct_abdomen_pelvis", "spine": "ct_spine", "extremity": "ct_extremity"}.get(body or "", "unknown")
    if modality == "MR":
        if any(t in text for t in _ANGIO_TOKENS):
            return inferred("mr_angio", "series description")
        if body == "head":
            present = {tok for tok in _MULTISEQUENCE_TOKENS if tok in text}
            if present == set(_MULTISEQUENCE_TOKENS) and len(image_series) >= 3:
                return inferred("mr_brain_multisequence", "T1, T2 and FLAIR named across series descriptions")
            return "mr_brain_structural"
        if body == "spine":
            return "mr_spine"
        if body == "extremity":
            return "mr_msk"
        return "unknown"
    if modality == "US":
        if body in {"heart", "fetus"}:
            return "us_echo" if body == "heart" else "us_obstetric"
        if any(t in text for t in _OBSTETRIC_TOKENS):
            return inferred("us_obstetric", "series description")
        if any(t in text for t in _CARDIAC_TOKENS):
            return inferred("us_echo", "series description")
        if body == "abdomen":
            return "us_abdomen"
        return "unknown"
    if modality == "XA":
        if body == "heart" or any(t in text for t in _CORONARY_TOKENS):
            return "xa_coronary" if body == "heart" else inferred("xa_coronary", "series description")
        return "unknown"
    if modality == "PT":
        return "nm_pet_ct"
    return "unknown"


def build_study_profile(datasets: Iterable[pydicom.Dataset]) -> dict[str, Any]:
    """The StudyProfile for the instances given. Never raises on header content: a fact that
    cannot be derived is null/unknown and ``derivation.warnings`` says why."""
    warnings: list[str] = []
    grouped: "OrderedDict[Optional[str], list[pydicom.Dataset]]" = OrderedDict()
    for ds in datasets:
        grouped.setdefault(_s(_get(ds, "SeriesInstanceUID")), []).append(ds)
    if None in grouped:
        warnings.append(f"{len(grouped[None])} instance(s) carry no SeriesInstanceUID; grouped as one unnamed series")

    series = [_series_entry(uid, members, warnings) for uid, members in grouped.items()]
    image_series = [s for s in series if s["excluded_as"] is None]
    excluded = [{"series_uid": s["series_uid"], "object_kind": s["excluded_as"], "instances": s["instances"]}
                for s in series if s["excluded_as"] is not None]

    modalities = sorted({s["modality"] for s in image_series if s["modality"]})
    if not image_series:
        modality = None
        warnings.append("no image series: modality, body part and dimensionality are unknown")
    elif len(modalities) == 1:
        modality = modalities[0]
    elif not modalities:
        modality = None
        warnings.append("image series carry no Modality")
    else:
        modality = "MIXED"
        warnings.append(f"image series span several modalities {modalities}")

    image_datasets = [ds for uid, members in grouped.items() for ds in members
                      if any(s["series_uid"] == uid and s["excluded_as"] is None for s in series)]
    raw_body = next((_s(_get(ds, "BodyPartExamined")) for ds in image_datasets if _s(_get(ds, "BodyPartExamined"))), None)
    normalized = normalize_body_part(raw_body)
    if raw_body is None and image_series:
        warnings.append("BodyPartExamined absent; body_part is null (not inferred)")
    elif normalized == "other":
        warnings.append(f"BodyPartExamined {raw_body!r} not in the vocabulary; normalized as other")

    text = " ".join(filter(None, [
        *(s["description"] for s in image_series), *(s["protocol_name"] for s in image_series),
        *(_s(_get(ds, "StudyDescription")) for ds in image_datasets[:1]),
        *(_s(_get(ds, "ManufacturerModelName")) for ds in image_datasets[:1]),
    ])).lower()

    lat = None
    for ds in image_datasets:
        value = _s(_get(ds, "ImageLaterality")) or _s(_get(ds, "Laterality"))
        if value:
            lat = _LATERALITY.get(value.upper())
            break

    sex = _s(_get(image_datasets[0], "PatientSex")) if image_datasets else None
    sex = sex.upper() if sex and sex.upper() in {"M", "F", "O"} else None

    profile = {
        "schema": SCHEMA,
        "modality": modality,
        "body_part": {"raw": raw_body, "normalized": normalized, "source": "BodyPartExamined" if raw_body else None},
        "dimensionality": _dimensionality(image_series),
        "protocol_family": _protocol_family(modality, normalized, image_series, text, warnings),
        "contrast": _contrast(image_datasets, text, warnings) if image_datasets else None,
        "laterality": lat,
        "patient_age_band": age_band(image_datasets[0], warnings) if image_datasets else "unknown",
        "patient_sex": sex,
        "series": [{k: v for k, v in s.items() if k not in {"excluded_as", "temporal_positions", "distinct_positions", "protocol_name"}}
                   for s in image_series],
        "excluded_objects": excluded,
        "derivation": {"version": DERIVATION_VERSION, "warnings": warnings},
    }
    assert profile["protocol_family"] in PROTOCOL_FAMILIES
    assert profile["dimensionality"] in DIMENSIONALITIES
    assert profile["patient_age_band"] in AGE_BANDS
    return profile


def profile_from_bytes(blobs: Iterable[bytes]) -> dict[str, Any]:
    """Build the profile from raw instances; unreadable blobs are counted in a warning."""
    datasets: list[pydicom.Dataset] = []
    unreadable = 0
    for blob in blobs:
        try:
            ds = pydicom.dcmread(io.BytesIO(blob), stop_before_pixels=True, force=True)
        except Exception:
            unreadable += 1
            continue
        # force=True does not refuse garbage; it returns a dataset with nothing a study is made
        # of. Such a blob is unreadable, not an image series with every field unknown.
        if not any(keyword in ds for keyword in ("SOPClassUID", "SOPInstanceUID", "SeriesInstanceUID", "Modality")):
            unreadable += 1
            continue
        datasets.append(ds)
    profile = build_study_profile(datasets)
    if unreadable:
        profile["derivation"]["warnings"].append(f"{unreadable} instance(s) could not be read as DICOM and are not profiled")
    return profile


#: Keys that must never appear in a profile; the privacy test and callers check against this.
FORBIDDEN_KEYWORDS = frozenset({
    "PatientName", "PatientID", "OtherPatientIDs", "PatientBirthDate", "PatientAge", "StudyDate",
    "SeriesDate", "AcquisitionDate", "ContentDate", "AccessionNumber", "InstitutionName",
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName", "StudyID",
    "PatientAddress", "PatientTelephoneNumbers",
})
