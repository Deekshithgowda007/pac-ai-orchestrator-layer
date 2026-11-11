"""Rung 3 of the routing ladder: the descriptive summary every study gets (SDD-SCANSURE-AI-002 T3b).

A study no pathology model may read must still leave the engine with something a radiologist
can use and the hospital can see. This rung describes the study -- technique, series inventory,
anatomy covered (when rung 2 produced it), image-quality facts, what else is in the study, and
exactly which models were considered and why each did or did not run. It is written from the
StudyProfile and from pixel STATISTICS; it is not a model, it has no opinion about pathology,
and it never says "normal" or "abnormal": ``abnormal`` and ``confidence`` are None and
``diagnostic_support`` is ``descriptive-only``, which orchestration already treats like
``anatomy-only`` (a real text for the radiologist, with no diagnostic claim attached).

It runs for every study whose files could be read, regardless of the registry lock and of
AI_AUTHORIZED_ONLY, because it makes no claim a licence governs. The result is stamped
``study-descriptor@DESCRIPTOR_VERSION`` so it can be told apart from a model everywhere.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pydicom

from models.study_context import StudyContext

DESCRIPTOR_VERSION = "1.0.0"
MODEL_NAME = "study-descriptor"
ANALYSIS_STATUS = "descriptive-non-diagnostic"
MAX_SAMPLED_INSTANCES = 48       # enough for noise / gap statistics, bounded for a 2000-slice study

_FAMILY_LABEL = {
    "cr_chest": "chest radiograph", "cr_abdomen": "abdominal radiograph", "cr_extremity": "extremity radiograph",
    "cr_spine": "spine radiograph", "cr_other": "radiograph", "dx_mammo": "mammogram",
    "ct_head": "head CT", "ct_chest": "chest CT", "ct_abdomen_pelvis": "abdomen/pelvis CT", "ct_spine": "spine CT",
    "ct_extremity": "extremity CT", "ct_angio": "CT angiogram", "cbct_dental": "dental cone-beam CT",
    "mr_brain_structural": "brain MR", "mr_brain_multisequence": "multisequence brain MR", "mr_spine": "spine MR",
    "mr_msk": "musculoskeletal MR", "mr_angio": "MR angiogram",
    "us_echo": "echocardiogram", "us_abdomen": "abdominal ultrasound", "us_obstetric": "obstetric ultrasound",
    "xa_coronary": "coronary angiogram", "nm_pet_ct": "PET-CT", "opg": "panoramic dental radiograph",
}
_DIM_LABEL = {"2d": "single-image", "2d_multiframe": "multi-frame", "3d": "volumetric stack", "4d": "time-resolved volume",
              "cine": "cine loop", "unknown": "undetermined dimensionality"}
_TS_LABEL = {
    "1.2.840.10008.1.2": "Implicit VR LE", "1.2.840.10008.1.2.1": "Explicit VR LE", "1.2.840.10008.1.2.2": "Explicit VR BE",
    "1.2.840.10008.1.2.1.99": "Deflated", "1.2.840.10008.1.2.5": "RLE", "1.2.840.10008.1.2.4.50": "JPEG baseline",
    "1.2.840.10008.1.2.4.57": "JPEG lossless", "1.2.840.10008.1.2.4.70": "JPEG lossless SV1",
    "1.2.840.10008.1.2.4.80": "JPEG-LS lossless", "1.2.840.10008.1.2.4.81": "JPEG-LS near-lossless",
    "1.2.840.10008.1.2.4.90": "JPEG 2000 lossless", "1.2.840.10008.1.2.4.91": "JPEG 2000",
    "1.2.840.10008.1.2.4.201": "HTJ2K lossless", "1.2.840.10008.1.2.4.203": "HTJ2K",
}


def _label(mapping: Dict[str, str], key: Optional[str], default: str) -> str:
    return mapping.get(key or "", default)


def _fmt(value: Any, unit: str = "", digits: int = 1) -> Optional[str]:
    try:
        return f"{float(value):.{digits}f}{unit}"
    except (TypeError, ValueError):
        return None


class StudyDescriptor:
    """Builds the rung-3 result. Pure numpy/pydicom; safe to run on any readable study."""

    def describe(self, context: StudyContext, *, anatomy: Optional[List[Dict[str, Any]]] = None,
                 ladder: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        profile = context.profile or {}
        modality = profile.get("modality") or (context.modality or None)
        family = profile.get("protocol_family")
        body = (profile.get("body_part") or {}).get("normalized")
        body_raw = (profile.get("body_part") or {}).get("raw")
        exam = _label(_FAMILY_LABEL, family, " ".join(x for x in (modality, body_raw) if x) or "imaging study")

        observations: List[str] = [self._technique(profile, exam)]
        observations.extend(self._inventory(profile))
        quality, metrics = self._image_quality(context, profile)
        observations.extend(quality)
        for result in anatomy or []:
            for line in (result.get("observations") or [])[:12]:
                observations.append(f"Anatomy ({result.get('model_name', 'anatomy model')}): {line}")
        excluded = profile.get("excluded_objects") or []
        if excluded:
            kinds = ", ".join(f"{e['instances']} {e['object_kind']}" for e in excluded)
            observations.append(f"Also in the study, not analysed as images: {kinds}.")
        for step in ladder or []:
            observations.append(self._ladder_line(step))

        conclusion = (f"Descriptive AI summary only: no diagnostic AI interpretation was produced for this {exam}. "
                      f"Radiologist interpretation is required.")
        return {
            "model_name": MODEL_NAME,
            "model_id": MODEL_NAME,
            "model_version": DESCRIPTOR_VERSION,
            "tier": None,
            "finding": conclusion,
            "impression_summary": f"Descriptive AI summary only; no diagnostic AI reading for this {exam}.",
            "abnormal": None,
            "confidence": None,
            "diagnostic_support": "descriptive-only",
            "report_type": "descriptive-summary",
            "analysis_type": "descriptive-summary",
            "analysis_status": ANALYSIS_STATUS,
            "anatomy_involved": body_raw or context.body_part,
            "region": "not-applicable",
            "observations": observations,
            "abnormalities": [],
            "impact": "No automated interpretation; the summary describes the study for the reader.",
            "recommendation": "Radiologist review required. The AI described the study and did not interpret it.",
            "limitations": [
                "Descriptive only: this text states what is in the study and makes no statement about normality or disease.",
            ],
            "metrics": metrics,
            "adapter_name": "StudyDescriptor",
        }

    # ------------------------------------------------------------------ sections

    @staticmethod
    def _technique(profile: Dict[str, Any], exam: str) -> str:
        series = profile.get("series") or []
        n_inst = sum(s.get("instances") or 0 for s in series)
        parts = [f"Technique: {exam}", _label(_DIM_LABEL, profile.get("dimensionality"), "undetermined dimensionality"),
                 f"{len(series)} image series, {n_inst} instance(s)"]
        thick = next((s.get("slice_thickness_mm") for s in series if s.get("slice_thickness_mm")), None)
        if thick:
            parts.append(f"slice thickness {_fmt(thick, ' mm')}")
        spacing = next((s.get("pixel_spacing_mm") for s in series if s.get("pixel_spacing_mm")), None)
        if spacing:
            parts.append(f"pixel spacing {_fmt(spacing[0], '', 2)} x {_fmt(spacing[1], ' mm', 2)}")
        orientations = sorted({s.get("orientation") for s in series if s.get("orientation")})
        if orientations:
            parts.append("orientation " + "/".join(orientations))
        contrast = profile.get("contrast")
        parts.append("contrast given" if contrast is True else "no contrast recorded" if contrast is False else "contrast not stated in the header")
        if profile.get("laterality"):
            parts.append(f"laterality {profile['laterality']}")
        band = profile.get("patient_age_band")
        if band and band != "unknown":
            parts.append(f"{band} patient")
        return "; ".join(parts) + "."

    @staticmethod
    def _inventory(profile: Dict[str, Any]) -> List[str]:
        lines = []
        for s in (profile.get("series") or [])[:8]:
            desc = s.get("description") or "(no series description)"
            ts = _label(_TS_LABEL, s.get("transfer_syntax"), s.get("transfer_syntax") or "transfer syntax not stated")
            frames = f", {s['frames']} frames" if (s.get("frames") or 0) > (s.get("instances") or 0) else ""
            lines.append(f"Series inventory: {desc} - {s.get('instances')} image(s){frames}, {s.get('modality') or '?'}, {ts}.")
        extra = len(profile.get("series") or []) - 8
        if extra > 0:
            lines.append(f"Series inventory: {extra} further series not listed.")
        return lines

    def _image_quality(self, context: StudyContext, profile: Dict[str, Any]) -> tuple:
        """Facts about the pixels, not an opinion about the patient."""
        metrics: Dict[str, float] = {}
        lines: List[str] = []
        series_map = self._load_sampled(context)
        if not series_map:
            return ["Image quality: pixel data could not be read; no image statistics."], metrics
        series = max(series_map.values(), key=len)
        frames = []
        for ds in series:
            try:
                arr = np.asarray(ds.pixel_array, dtype=np.float32)
                arr = np.squeeze(arr)
                if arr.ndim == 3:
                    arr = arr[arr.shape[0] // 2] if arr.shape[0] > 4 else arr[..., 0]
                if arr.ndim == 2:
                    frames.append(arr)
            except Exception:
                continue
        if not frames:
            return ["Image quality: pixel data could not be decoded; no image statistics."], metrics
        stack = np.stack([f for f in frames if f.shape == frames[0].shape]) if frames else None
        lo, hi = float(np.percentile(stack, 1)), float(np.percentile(stack, 99))
        metrics["intensity_p1"], metrics["intensity_p99"] = lo, hi
        # noise: median absolute deviation of the horizontal first difference, robust to anatomy
        diff = np.diff(stack, axis=-1)
        noise = float(np.median(np.abs(diff - np.median(diff))) * 1.4826)
        metrics["noise_estimate"] = noise
        rel = noise / (hi - lo) if hi > lo else math.nan
        metrics["noise_relative"] = rel if not math.isnan(rel) else -1.0
        lines.append(f"Image quality: sampled {len(frames)} image(s) of the largest series; intensity 1st-99th percentile "
                     f"{lo:.0f} to {hi:.0f}; relative noise {'%.3f' % rel if not math.isnan(rel) else 'n/a'}.")
        sat = float(np.mean(stack >= stack.max())) if stack.size else 0.0
        if sat > 0.02:
            lines.append(f"Image quality: {sat * 100:.1f}% of pixels at the maximum value (possible saturation or burned-in overlay).")
        gaps = self._position_gaps(series)
        if gaps is not None:
            metrics["slice_position_gaps"] = float(gaps)
            lines.append("Image quality: slice positions are evenly spaced; no missing slices detected in the sample."
                         if gaps == 0 else f"Image quality: {gaps} gap(s) in slice positions in the sample (possible missing slices).")
        return lines, metrics

    @staticmethod
    def _position_gaps(series: List[pydicom.Dataset]) -> Optional[int]:
        positions = []
        for ds in series:
            try:
                positions.append(float(ds.ImagePositionPatient[2]))
            except Exception:
                continue
        if len(positions) < 3:
            return None
        positions.sort()
        steps = np.diff(positions)
        median = float(np.median(steps))
        if median <= 0:
            return None
        return int(np.sum(steps > 1.5 * median))

    @staticmethod
    def _load_sampled(context: StudyContext) -> Dict[str, List[pydicom.Dataset]]:
        files = list(context.files)
        if len(files) > MAX_SAMPLED_INSTANCES:
            idx = np.linspace(0, len(files) - 1, MAX_SAMPLED_INSTANCES).astype(int)
            files = [files[i] for i in idx]
        series_map: Dict[str, List[pydicom.Dataset]] = {}
        for path in files:
            try:
                ds = pydicom.dcmread(path, force=True)
            except Exception:
                continue
            if "PixelData" in ds:
                series_map.setdefault(str(getattr(ds, "SeriesInstanceUID", "unknown")), []).append(ds)
        return series_map

    @staticmethod
    def _ladder_line(step: Dict[str, Any]) -> str:
        who = f"{step.get('model_id')}@{step.get('version')}" if step.get("model_id") else step.get("route_name", "model")
        outcome = step.get("outcome")
        detail = f" ({step['detail']})" if step.get("detail") else ""
        words = {
            "ran": "produced a result", "declined": "declined the study", "failed": "failed",
            "weights_unavailable": "not run: weights not present", "not_authorized": "not run: not authorized for this deployment",
            "model_disabled": "not run: disabled in the registry lock", "mismatch": "not applicable to this study",
            "incomplete": "not run: the study header lacks a field the model requires",
        }
        return f"AI models considered: {who} - {words.get(outcome, outcome)}{detail}."
