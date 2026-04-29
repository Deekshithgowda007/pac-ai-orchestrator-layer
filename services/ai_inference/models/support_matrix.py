from __future__ import annotations

from typing import Any, Dict

from models.model_registry import get_model_entry


def describe_support(route_name: str) -> Dict[str, Any]:
    entry = get_model_entry(route_name)
    return {
        "route_name": entry["route_name"],
        "modalities": entry["modalities"],
        "body_parts": entry["body_parts"],
        "model_name": entry["model_name"],
        "model_family": entry["model_family"],
        "diagnostic_support": entry["diagnostic_support"],
        "report_type": entry["report_type"],
        "diagnostic_available": entry["diagnostic_available"],
        "weights_required": entry["weights_required"],
        "weights_path": entry["weights_path"],
        "weights_present": entry["weights_present"],
        "runtime": entry["runtime"],
        "notes": entry["notes"],
    }
