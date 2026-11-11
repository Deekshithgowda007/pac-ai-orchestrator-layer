"""The model registry, read from manifests (SDD-SCANSURE-AI-002 T2).

Until T2 this file held a hard-coded list of ten route dicts. It now derives the same dicts
from ``models/manifests/<model_id>/<version>.json`` through ``manifest_registry``, so there
is one description of each model and ``get_model_entry(route_name)`` keeps its shape for the
adapters and ``support_matrix`` that read it. New keys ride alongside the old ones
(``model_id``, ``model_version``, ``tier``, ``weights_sha256``).

``unsupported-modalities`` is not a model and has no manifest; it stays here as the one
synthetic entry because ``UnsupportedModalityAdapter``'s result names that route.
"""
from __future__ import annotations

from typing import Any, Dict, List

from models import manifest_registry

_UNSUPPORTED_ENTRY: Dict[str, Any] = {
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
    "weights_present": False,
    "runtime": {"device": "n/a", "gpu_recommended": False},
    "notes": "No validated modality-specific model is currently configured for these modalities.",
}

_registry: manifest_registry.Registry | None = None


def registry() -> manifest_registry.Registry:
    """The validated registry, loaded once. A bad manifest fails the first call loudly."""
    global _registry
    if _registry is None:
        _registry = manifest_registry.load_registry()
    return _registry


def reset_cache() -> None:
    global _registry
    _registry = None


def _entries() -> List[Dict[str, Any]]:
    return [manifest_registry.legacy_entry(m) for m in registry().manifests] + [dict(_UNSUPPORTED_ENTRY)]


class _LazyRegistryList(list):
    """``MODEL_REGISTRY`` kept importable for callers that iterate it; populated on first use."""

    def _load(self) -> None:
        if not super().__len__():
            super().extend(_entries())

    def __iter__(self):
        self._load()
        return super().__iter__()

    def __len__(self) -> int:
        self._load()
        return super().__len__()

    def __getitem__(self, item):
        self._load()
        return super().__getitem__(item)


MODEL_REGISTRY: List[Dict[str, Any]] = _LazyRegistryList()


def get_model_entry(route_name: str) -> Dict[str, Any]:
    for entry in _entries():
        if entry["route_name"] == route_name:
            return entry
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
