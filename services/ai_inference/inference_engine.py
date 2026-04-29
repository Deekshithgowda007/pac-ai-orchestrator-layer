import os
from typing import Dict

from models.modality_adapters import AdapterRegistry
from models.heuristic_fallback import HeuristicFallbackEngine
from models.study_context import collect_study_context


class InferenceEngine:
    """
    Rule-based fallback inference engine.

    This is not a diagnostic model. It produces deterministic, metadata-grounded
    preliminary observations so the orchestration layer behaves consistently until
    trained modality-specific models are plugged in.
    """

    def __init__(self, development_mode: bool = True):
        self.development_mode = development_mode
        self.adapter_registry = AdapterRegistry()
        self.fallback = HeuristicFallbackEngine(development_mode=development_mode)

    def run(self, dicom_path: str) -> Dict:
        if not os.path.exists(dicom_path):
            return self._fallback("Path not found")

        context = collect_study_context(dicom_path)
        if context:
            try:
                return self.adapter_registry.analyze(context)
            except Exception:
                pass

        return self.fallback.run(dicom_path)

    def _fallback(self, message: str) -> Dict:
        return self.fallback._fallback(message)
