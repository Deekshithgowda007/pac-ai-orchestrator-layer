from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def _split_csv_env(name: str) -> List[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


MODEL_REGISTRY: List[Dict[str, Any]] = [
    {
        "route_name": "ct-monai-screening",
        "modalities": ["CT"],
        "body_parts": ["CHEST", "LUNG", "THORAX"],
        "model_name": "monai-lung-nodule-retinanet",
        "model_family": "monai-retinanet",
        "diagnostic_support": "screening-only",
        "report_type": "preliminary-lung-nodule-screening",
        "diagnostic_available": True,
        "weights_required": True,
        "weights_env": "CT_MONAI_CHECKPOINT_PATH",
        "weights_path": None,
        "runtime": {
            "device": "cpu",
            "gpu_recommended": True,
        },
        "notes": "MONAI lung_nodule_ct_detection RetinaNet route for chest CT pulmonary nodule screening. Requires the external bundle checkpoint and should be treated as preliminary screening support only.",
    },
    {
        "route_name": "mr-brats-segmentation",
        "modalities": ["MR", "MRI"],
        "body_parts": ["BRAIN"],
        "model_name": "brats-mri-segresnet",
        "model_family": "monai",
        "diagnostic_support": "screening-only",
        "report_type": "preliminary-brain-tumor-segmentation",
        "diagnostic_available": True,
        "weights_required": True,
        "weights_env": "MR_MONAI_CHECKPOINT_PATH",
        "weights_path": None,
        "runtime": {
            "device": "cpu",
            "gpu_recommended": True,
        },
        "notes": "BraTS SegResNet brain MRI segmentation route. Requires a full T1c/T1/T2/FLAIR brain MRI input set and should be treated as preliminary screening support.",
    },

    {
        "route_name": "us-echo-lite-lv-function",
        "modalities": ["US"],
        "body_parts": [],
        "model_name": "echonet-dynamic-style-lv-function-lite",
        "model_family": "heuristic-echocardiography",
        "diagnostic_support": "screening-only",
        "report_type": "preliminary-echocardiography-lv-function",
        "diagnostic_available": True,
        "weights_required": False,
        "weights_env": None,
        "weights_path": None,
        "runtime": {
            "device": "cpu",
            "gpu_recommended": False,
        },
        "notes": "Lightweight multi-frame echocardiography screening route that estimates left ventricular systolic function and approximate ejection fraction from cine-loop motion without external model downloads.",
    },
    {
        "route_name": "us-hf-2d-screening",
        "modalities": ["US"],
        "body_parts": [],
        "model_name": "huggingface-us-image-to-text",
        "model_family": "huggingface-transformers",
        "diagnostic_support": "screening-only",
        "report_type": "preliminary-2d-screening",
        "diagnostic_available": True,
        "weights_required": False,
        "weights_env": None,
        "weights_path": None,
        "runtime": {
            "device": "cpu",
            "gpu_recommended": False,
        },
        "notes": "Uses a US-specific Hugging Face image-to-text fallback on a representative PNG rendered from ultrasound DICOM. Intended for descriptive support only, not diagnosis.",
    },
    {
        "route_name": "xa-stenosis-screening",
        "modalities": ["XA"],
        "body_parts": [],
        "model_name": "stenunet-xa",
        "model_family": "stenunet",
        "diagnostic_support": "screening-only",
        "report_type": "preliminary-stenosis-screening",
        "diagnostic_available": True,
        "weights_required": True,
        "weights_env": "STENUNET_CHECKPOINT_PATH",
        "weights_path": None,
        "runtime": {
            "device": "cpu",
            "gpu_recommended": True,
        },
        "notes": "Research XA stenosis screening model. Requires the StenUNet checkpoint and should be treated as preliminary AI support, not a final diagnosis.",
    },
    {
        "route_name": "chest-radiograph-screening",
        "modalities": ["CR", "DX", "MG"],
        "body_parts": ["CHEST", "LUNG"],
        "model_name": "torchxrayvision-densenet121",
        "model_family": "torchxrayvision",
        "diagnostic_support": "screening-only",
        "report_type": "preliminary-screening",
        "diagnostic_available": True,
        "weights_required": False,
        "weights_env": None,
        "weights_path": None,
        "runtime": {
            "device": "cpu",
            "gpu_recommended": False,
        },
        "notes": "Provides chest radiograph screening labels only; not a final radiology diagnosis.",
    },
    {
        "route_name": "hf-2d-screening",
        "modalities": ["CR", "DX", "MG", "CT", "MR", "MRI", "US", "RF", "XA"],
        "body_parts": [],
        "model_name": "huggingface-2d",
        "model_family": "huggingface-transformers",
        "diagnostic_support": "screening-only",
        "report_type": "preliminary-2d-screening",
        "diagnostic_available": True,
        "weights_required": False,
        "weights_env": None,
        "weights_path": None,
        "runtime": {
            "device": "cpu",
            "gpu_recommended": False,
        },
        "notes": "Uses a configured Hugging Face 2D pipeline on a medically rendered PNG exported from DICOM. Intended for projection radiographs and other single-frame studies only.",
    },
    {
        "route_name": "ct-anatomy-segmentation",
        "modalities": ["CT"],
        "body_parts": [],
        "model_name": "totalsegmentator-total",
        "model_family": "totalsegmentator",
        "diagnostic_support": "anatomy-only",
        "report_type": "non-diagnostic-anatomy",
        "diagnostic_available": False,
        "weights_required": False,
        "weights_env": None,
        "weights_path": None,
        "runtime": {
            "device": "cpu",
            "gpu_recommended": True,
        },
        "notes": "Provides CT anatomy segmentation only unless a trained pathology model is added.",
    },
    {
        "route_name": "mr-anatomy-segmentation",
        "modalities": ["MR", "MRI"],
        "body_parts": [],
        "model_name": "totalsegmentator-total_mr",
        "model_family": "totalsegmentator",
        "diagnostic_support": "anatomy-only",
        "report_type": "non-diagnostic-anatomy",
        "diagnostic_available": False,
        "weights_required": False,
        "weights_env": None,
        "weights_path": None,
        "runtime": {
            "device": "cpu",
            "gpu_recommended": True,
        },
        "notes": "Provides MR anatomy segmentation only unless a trained pathology model is added.",
    },
    {
        "route_name": "unsupported-modalities",
        "modalities": ["XA", "RF", "US", "NM", "PT", "OCT"],
        "body_parts": [],
        "model_name": "manual-review",
        "model_family": "none",
        "diagnostic_support": "not-supported",
        "report_type": "manual-review-required",
        "diagnostic_available": False,
        "weights_required": False,
        "weights_env": None,
        "weights_path": None,
        "runtime": {
            "device": "n/a",
            "gpu_recommended": False,
        },
        "notes": "No validated modality-specific model is currently configured for these modalities.",
    },
]


def _resolve_weights(entry: Dict[str, Any]) -> Dict[str, Any]:
    weights_env = entry.get("weights_env")
    weights_path: Optional[str] = None
    weights_present = False
    if weights_env:
        candidates = _split_csv_env(weights_env)
        for candidate in candidates:
            if os.path.exists(candidate):
                weights_path = candidate
                weights_present = True
                break
        if not weights_path and candidates:
            weights_path = candidates[0]

    enriched = dict(entry)
    enriched["weights_path"] = weights_path
    enriched["weights_present"] = weights_present
    if enriched.get("weights_required") and not weights_present:
        enriched["diagnostic_available"] = False
    return enriched


def get_model_entry(route_name: str) -> Dict[str, Any]:
    for entry in MODEL_REGISTRY:
        if entry["route_name"] == route_name:
            return _resolve_weights(entry)
    return {
        "route_name": route_name,
        "modalities": [],
        "body_parts": [],
        "model_name": None,
        "model_family": "unknown",
        "diagnostic_support": "unknown",
        "report_type": "unknown",
        "diagnostic_available": False,
        "weights_required": False,
        "weights_env": None,
        "weights_path": None,
        "weights_present": False,
        "runtime": {"device": "unknown", "gpu_recommended": False},
        "notes": "No model registry entry found for this route.",
    }

