import os
from typing import Dict

from models.modality_adapters import AdapterRegistry
from models.study_context import StudyContext, collect_study_context


class InferenceEngine:
    """Routes a study to the models whose manifests accept it (SDD-SCANSURE-AI-002 T3).

    Until T3 this class was "a rule-based fallback inference engine": when no adapter
    succeeded, or when the path was missing, it ran HeuristicFallbackEngine over the pixels and
    returned abnormal/not-abnormal with a confidence between 0.35 and 0.90 -- a verdict from no
    model, indistinguishable downstream from a reading. That path is gone. A study nothing can
    read now returns a result whose clinical fields are explicitly absent (abnormal None,
    confidence None, analysis_status unsupported-for-diagnostic-inference) and whose
    routing_decision says why. The heuristic survives only as a manifest (heuristic-fallback)
    that the registry lock does not enable by default.
    """

    def __init__(self, development_mode: bool = True):
        self.development_mode = development_mode
        self.adapter_registry = AdapterRegistry()

    def run(self, dicom_path: str) -> Dict:
        if not os.path.exists(dicom_path):
            return self._unavailable(dicom_path, "input_missing", "the study path does not exist")
        context = collect_study_context(dicom_path)
        if context is None:
            return self._unavailable(dicom_path, "no_dicom_files", "no .dcm file was found at the study path")
        return self.adapter_registry.analyze(context)

    @staticmethod
    def _unavailable(dicom_path: str, status: str, detail: str) -> Dict:
        empty = StudyContext(dicom_path=dicom_path, files=[], metadata={}, modality="", body_part="UNKNOWN",
                             series_uid="unknown", study_uid="unknown", profile=None)
        return AdapterRegistry.unavailable(empty, status, detail, route_name="input-unavailable")
