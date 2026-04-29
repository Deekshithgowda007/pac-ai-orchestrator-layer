from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from models.hf_utils import resolve_hf_bundle_dir
from models.heuristic_fallback import HeuristicFallbackEngine
from models.study_context import StudyContext
from models.support_matrix import describe_support

log = logging.getLogger("ai_inference.adapters")


class ModalityAdapter(ABC):
    @abstractmethod
    def supports(self, context: StudyContext) -> bool:
        raise NotImplementedError

    @abstractmethod
    def analyze(self, context: StudyContext) -> Dict[str, Any]:
        raise NotImplementedError


class UnsupportedModalityAdapter(ModalityAdapter):
    def __init__(self) -> None:
        self.unsupported_modalities = {"XA", "RF", "US", "NM", "PT", "OCT"}

    def supports(self, context: StudyContext) -> bool:
        return context.modality.upper() in self.unsupported_modalities

    def analyze(self, context: StudyContext) -> Dict[str, Any]:
        modality = context.modality.upper() or "UNKNOWN"
        return {
            "model_name": f"{modality.lower()}-unsupported",
            "finding": f"No validated open-source diagnostic inference model is configured for {modality} studies in this deployment.",
            "abnormal": None,
            "confidence": None,
            "diagnostic_support": "not-supported",
            "report_type": "manual-review-required",
            "analysis_type": "unsupported-modality",
            "analysis_status": "unsupported-for-diagnostic-inference",
            "anatomy_involved": context.body_part,
            "region": "not-applicable",
            "observations": [
                f"Study modality is {modality}.",
                f"{modality} is currently routed to manual review because no validated modality-specific model is configured.",
            ],
            "abnormalities": [],
            "impact": "Automated diagnostic interpretation was intentionally skipped for safety.",
            "recommendation": "Route this study to radiologist/manual review or configure a validated modality-specific model before enabling automated reporting.",
            "limitations": [
                f"{modality} is not supported for diagnostic inference in the current open-source model stack.",
            ],
            "metrics": {},
        }


class FallbackAdapter(ModalityAdapter):
    def __init__(self) -> None:
        self.engine = HeuristicFallbackEngine()

    def supports(self, context: StudyContext) -> bool:
        return True

    def analyze(self, context: StudyContext) -> Dict[str, Any]:
        return self.engine.run(context.dicom_path)


class XRayVisionAdapter(ModalityAdapter):
    def __init__(self) -> None:
        self.enabled = os.getenv("ENABLE_TORCHXRAYVISION", "true").lower() == "true"
        self.threshold = float(os.getenv("XRAY_ABNORMAL_THRESHOLD", "0.55"))
        self.model = None
        self.xrv = None
        self.transforms = None

    def supports(self, context: StudyContext) -> bool:
        if not self.enabled:
            return False
        modality = context.modality.upper()
        body = context.body_part.upper()
        return modality in {"CR", "DX", "MG"} or "CHEST" in body or "LUNG" in body

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        import torch
        import torchvision
        import torchxrayvision as xrv

        self.xrv = xrv
        self.transforms = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )
        self.model = xrv.models.DenseNet(weights=os.getenv("XRAY_MODEL_WEIGHTS", "densenet121-res224-all"))
        self.model.eval()
        self.torch = torch

    def analyze(self, context: StudyContext) -> Dict[str, Any]:
        self._ensure_loaded()
        image = context.load_projection_image()
        if image is None:
            raise RuntimeError("Unable to read projection image")

        if image.ndim != 2:
            image = image.squeeze()
        image = self.xrv.datasets.normalize((image * 255).astype(np.float32), 255)
        image = image[None, ...]
        image = self.transforms(image)
        tensor = self.torch.from_numpy(image).float()[None, ...]

        with self.torch.no_grad():
            outputs = self.model(tensor)[0].detach().cpu().numpy()

        raw_preds = dict(zip(self.model.pathologies, [float(v) for v in outputs]))
        sorted_preds = sorted(raw_preds.items(), key=lambda item: item[1], reverse=True)
        findings = [name for name, score in sorted_preds if score >= self.threshold][:5]
        abnormal = bool(findings)
        top_name, top_score = sorted_preds[0]

        observations = [
            f"Open-source chest X-ray classifier evaluated {len(raw_preds)} pathology labels.",
            f"Highest scoring label was {top_name} with score {top_score:.3f}.",
        ]
        if findings:
            observations.append(f"Labels above threshold: {', '.join(findings)}.")
        else:
            observations.append("No pathology label crossed the configured abnormality threshold.")

        return {
            "model_name": "torchxrayvision-densenet121",
            "finding": (
                f"Open-source chest X-ray model flagged possible findings: {', '.join(findings)}."
                if findings else
                "Open-source chest X-ray model did not flag a dominant pathology label."
            ),
            "abnormal": abnormal,
            "confidence": round(float(top_score), 3),
            "diagnostic_support": "screening-only",
            "report_type": "preliminary-screening",
            "analysis_type": "open-source-pathology-classification",
            "anatomy_involved": context.body_part,
            "region": "thorax/chest",
            "observations": observations,
            "abnormalities": findings,
            "impact": (
                "Prompt radiologist review is advised because the chest X-ray classifier flagged candidate pathology labels."
                if abnormal else
                "No dominant classifier label was flagged, but radiologist review is still required."
            ),
            "metrics": {key: round(value, 4) for key, value in sorted_preds[:8]},
            "limitations": [
                "TorchXRayVision is a pathology classifier for chest radiographs, not a final diagnosis engine.",
                "Predictions should be clinically correlated and reviewed by a radiologist.",
            ],
        }


class HuggingFace2DAdapter(ModalityAdapter):
    def __init__(self) -> None:
        self.enabled = os.getenv("ENABLE_HF_2D_INFERENCE", "true").lower() == "true"
        self.model_id = os.getenv("HF_2D_MODEL_ID", "").strip()
        self.task = os.getenv("HF_2D_TASK", "image-classification").strip().lower()
        self.threshold = float(os.getenv("HF_2D_ABNORMAL_THRESHOLD", "0.5"))
        self.allow_non_xray = os.getenv("HF_2D_ALLOW_NON_XRAY", "false").lower() == "true"
        self.us_model_enabled = os.getenv("ENABLE_US_HF_INFERENCE", "true").lower() == "true"
        self.us_model_id = os.getenv("US_HF_MODEL_ID", "").strip()
        self._pipeline = None

    @staticmethod
    def _extract_generated_text(outputs: Any) -> str:
        if isinstance(outputs, dict):
            for key in ("generated_text", "text", "caption"):
                value = outputs.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""
        if isinstance(outputs, list):
            parts: List[str] = []
            for item in outputs:
                text = HuggingFace2DAdapter._extract_generated_text(item)
                if text:
                    parts.append(text)
            return " ".join(parts).strip()
        return ""

    @staticmethod
    def _is_low_signal_caption(text: str) -> bool:
        cleaned = " ".join(str(text or "").strip().split()).lower()
        if not cleaned:
            return True
        if cleaned.endswith(".png") or cleaned.endswith(".jpg") or cleaned.endswith(".jpeg"):
            return True
        if "/tmp/" in cleaned or "\\tmp\\" in cleaned or "input.png" in cleaned:
            return True
        if cleaned.startswith("/") or cleaned.startswith("c:\\") or cleaned.startswith("d:\\"):
            return True
        if cleaned in {"unknown", "none", "normal", "image", "medical image", "ultrasound"}:
            return True
        if len(cleaned) < 12:
            return True
        return False

    def supports(self, context: StudyContext) -> bool:
        if not self.enabled or not self.model_id:
            return False
        modality = context.modality.upper()
        if modality in {"CR", "DX", "MG"}:
            return True
        if modality == "US" and self.us_model_enabled and self.us_model_id:
            return False
        if not self.allow_non_xray:
            return False
        series, series_stats = context.load_primary_series()
        slice_count = int(series_stats.get("slice_count", 0))
        if modality == "US":
            return bool(series) and context.load_projection_image() is not None
        return bool(series) and slice_count <= 1 and context.load_projection_image() is not None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline

        kwargs: Dict[str, Any] = {"task": self.task, "model": self.model_id, "device": -1}
        token = os.getenv("HF_API_KEY", "").strip()
        if token:
            kwargs["token"] = token
        self._pipeline = pipeline(**kwargs)

    def analyze(self, context: StudyContext) -> Dict[str, Any]:
        self._ensure_loaded()
        series, series_stats = context.load_primary_series()
        slice_count = int(series_stats.get("slice_count", 0))

        with tempfile.TemporaryDirectory(prefix="hf-2d-") as temp_dir:
            png_path = os.path.join(temp_dir, "input.png")
            if not context.export_projection_png(png_path):
                raise RuntimeError("Unable to export the input DICOM image into PNG for Hugging Face 2D inference.")
            outputs = self._pipeline(png_path)

        if "text" in self.task:
            generated_text = self._extract_generated_text(outputs)
            low_signal = self._is_low_signal_caption(generated_text)
            abnormal = None if low_signal else False
            confidence = None
            finding = (
                generated_text
                if not low_signal
                else "Descriptive AI captioning of the representative medical image was non-informative."
            )
            observations = [
                f"Hugging Face model {self.model_id} generated a text description from a DICOM-derived PNG.",
                f"Primary series frame count: {slice_count}.",
            ]
            if generated_text:
                observations.append(f"Raw generated caption: {generated_text}")
            if low_signal:
                observations.append("Generated caption was too low-signal for a reliable descriptive summary.")
            metrics = {"slice_count": float(slice_count)}
            abnormalities = []
        else:
            predictions = outputs if isinstance(outputs, list) else [outputs]
            normalized = []
            for item in predictions:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "unknown"))
                score = float(item.get("score", 0.0))
                normalized.append((label, score))
            normalized.sort(key=lambda item: item[1], reverse=True)
            if not normalized:
                raise RuntimeError("Hugging Face 2D model did not return any classification scores.")
            top_label, top_score = normalized[0]
            flagged = [label for label, score in normalized if score >= self.threshold][:5]
            abnormal = bool(flagged)
            confidence = round(float(top_score), 4)
            finding = (
                f"Hugging Face 2D model flagged candidate labels: {', '.join(flagged)}."
                if abnormal
                else f"Hugging Face 2D model did not flag a label above threshold; top label was {top_label}."
            )
            observations = [
                f"Hugging Face model {self.model_id} analyzed a DICOM-derived PNG using task {self.task}.",
                f"Top label was {top_label} with score {top_score:.3f}.",
                f"Primary series frame count: {slice_count}.",
            ]
            metrics = {label: round(score, 4) for label, score in normalized[:8]}
            metrics["slice_count"] = float(slice_count)
            metrics["abnormal_threshold"] = round(self.threshold, 4)
            abnormalities = flagged

        return {
            "model_name": f"huggingface-2d:{self.model_id}",
            "finding": finding,
            "abnormal": abnormal,
            "confidence": confidence,
            "diagnostic_support": "screening-only",
            "report_type": "preliminary-2d-screening",
            "analysis_type": f"huggingface-2d-{self.task}",
            "analysis_status": (
                "non-diagnostic"
                if abnormal is None
                else "abnormal" if abnormal else "no-dominant-abnormality-detected"
            ),
            "anatomy_involved": context.body_part,
            "region": "single-frame/projection-image",
            "observations": observations,
            "abnormalities": abnormalities,
            "impact": (
                "A Hugging Face 2D model flagged candidate findings that should be reviewed by a clinician."
                if abnormal
                else "This 2D screening output is non-diagnostic and still requires clinical review."
            ),
            "recommendation": (
                "Review the generated 2D model output together with the original DICOM image and clinical context."
            ),
            "metrics": metrics,
            "limitations": [
                "This Hugging Face 2D path uses a PNG rendered from DICOM pixel data and is not a final diagnosis engine.",
                "2D image models do not preserve the full 3D medical context available in complete CT/MR/XA studies.",
            ],
        }


class EchoLiteLVFunctionAdapter(ModalityAdapter):
    def __init__(self) -> None:
        self.enabled = os.getenv("ENABLE_US_ECHO_LITE", "true").lower() == "true"
        self.min_frames = int(os.getenv("US_ECHO_LITE_MIN_FRAMES", "6"))
        self.max_frames = int(os.getenv("US_ECHO_LITE_MAX_FRAMES", "48"))
        self.motion_percentile = float(os.getenv("US_ECHO_LITE_MOTION_PERCENTILE", "82"))
        self.cavity_percentile = float(os.getenv("US_ECHO_LITE_CAVITY_PERCENTILE", "35"))
        self.center_crop_ratio = float(os.getenv("US_ECHO_LITE_CENTER_CROP_RATIO", "0.72"))

    def supports(self, context: StudyContext) -> bool:
        if not self.enabled or context.modality.upper() != "US":
            return False
        cine, cine_stats = context.load_primary_cine_frames(max_frames=self.max_frames)
        frame_count = int(cine_stats.get("frame_count", 0)) if cine_stats else 0
        return cine is not None and frame_count >= self.min_frames

    @staticmethod
    def _classify_lv_function(ef_percent: float) -> Tuple[str, str]:
        if ef_percent < 30:
            return "severely reduced", "Marked reduction in left ventricular pump function is suggested on automated screening."
        if ef_percent < 40:
            return "moderately reduced", "Moderate reduction in left ventricular pump function is suggested on automated screening."
        if ef_percent < 50:
            return "mildly reduced", "Mild reduction in left ventricular pump function is suggested on automated screening."
        return "preserved", "Preserved left ventricular systolic function is suggested on automated screening."

    @staticmethod
    def _bounded(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def analyze(self, context: StudyContext) -> Dict[str, Any]:
        cine, cine_stats = context.load_primary_cine_frames(max_frames=self.max_frames)
        if cine is None or cine.shape[0] < self.min_frames:
            raise RuntimeError("Ultrasound cine clip did not provide enough frames for LV function estimation.")

        frame_count, height, width = cine.shape
        crop_ratio = self._bounded(self.center_crop_ratio, 0.45, 1.0)
        crop_h = max(int(height * crop_ratio), 32)
        crop_w = max(int(width * crop_ratio), 32)
        y0 = max((height - crop_h) // 2, 0)
        x0 = max((width - crop_w) // 2, 0)
        crop = cine[:, y0:y0 + crop_h, x0:x0 + crop_w]

        temporal_std = np.std(crop, axis=0)
        motion_threshold = float(np.percentile(temporal_std, self.motion_percentile))
        motion_mask = temporal_std >= motion_threshold
        mask_fraction = float(np.mean(motion_mask))
        if mask_fraction < 0.015:
            inner_y0 = crop_h // 4
            inner_y1 = crop_h - inner_y0
            inner_x0 = crop_w // 4
            inner_x1 = crop_w - inner_x0
            motion_mask = np.zeros((crop_h, crop_w), dtype=bool)
            motion_mask[inner_y0:inner_y1, inner_x0:inner_x1] = True
            mask_fraction = float(np.mean(motion_mask))

        cavity_areas: List[float] = []
        frame_means: List[float] = []
        for frame in crop:
            masked_values = frame[motion_mask]
            if masked_values.size == 0:
                masked_values = frame.reshape(-1)
            cavity_threshold = float(np.percentile(masked_values, self.cavity_percentile))
            cavity_mask = np.logical_and(frame <= cavity_threshold, motion_mask)
            cavity_areas.append(float(np.sum(cavity_mask)))
            frame_means.append(float(np.mean(masked_values)))

        area_curve = np.asarray(cavity_areas, dtype=np.float32)
        if area_curve.size >= 3:
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            area_curve = np.convolve(area_curve, kernel, mode="same")

        end_diastolic_area = float(np.max(area_curve))
        end_systolic_area = float(np.min(area_curve))
        contraction_fraction = 0.0
        if end_diastolic_area > 0:
            contraction_fraction = (end_diastolic_area - end_systolic_area) / end_diastolic_area

        ef_percent = round(self._bounded(contraction_fraction * 100.0, 10.0, 80.0), 1)
        function_class, impact = self._classify_lv_function(ef_percent)

        frame_time_ms = float(cine_stats.get("frame_time_ms", 0.0) or 0.0)
        confidence = self._bounded(
            0.42
            + min(frame_count / 40.0, 1.0) * 0.14
            + min(max(contraction_fraction, 0.0), 0.6) * 0.35
            + min(mask_fraction / 0.18, 1.0) * 0.09,
            0.45,
            0.84,
        )

        abnormal = ef_percent < 50.0
        series_label = context.metadata.get("SeriesDescription") or context.series_uid or "selected echo clip"
        observations = [
            f"Lightweight echocardiography cine analysis reviewed {frame_count} frame(s) from {series_label}.",
            (
                "Estimated left ventricular cavity contraction proxy suggests "
                f"an ejection fraction of approximately {ef_percent:.1f}%."
            ),
            (
                "Automated screening classification is "
                f"{function_class} left ventricular systolic function."
            ),
        ]
        if frame_time_ms > 0:
            observations.append(f"Nominal frame time was {frame_time_ms:.1f} ms.")

        return {
            "model_name": "echonet-dynamic-style-lv-function-lite",
            "finding": (
                f"Automated echocardiography screening estimated left ventricular ejection fraction at approximately "
                f"{ef_percent:.1f}% with {function_class} systolic function."
            ),
            "abnormal": abnormal,
            "confidence": round(float(confidence), 4),
            "diagnostic_support": "screening-only",
            "report_type": "preliminary-echocardiography-lv-function",
            "analysis_type": "echocardiography-lv-function-lite",
            "analysis_status": "abnormal" if abnormal else "no-dominant-abnormality-detected",
            "anatomy_involved": "Left ventricle",
            "region": "central cardiac chamber / left ventricular cavity proxy",
            "observations": observations,
            "abnormalities": [f"{function_class} left ventricular systolic function"] if abnormal else [],
            "impact": impact,
            "recommendation": (
                "Correlate the automated LV function estimate with the full echocardiographic study and formal cardiology interpretation."
            ),
            "metrics": {
                "slice_count": float(frame_count),
                "frame_count": float(frame_count),
                "frame_time_ms": round(frame_time_ms, 4),
                "estimated_ef_percent": ef_percent,
                "lv_function_class_score": round(float(confidence), 4),
                "end_diastolic_area_px": round(end_diastolic_area, 1),
                "end_systolic_area_px": round(end_systolic_area, 1),
                "fractional_area_change": round(float(contraction_fraction), 4),
                "motion_mask_fraction": round(mask_fraction, 4),
                "mean_motion_region_intensity": round(float(np.mean(frame_means)), 4),
            },
            "limitations": [
                "This is a lightweight clip-based LV function estimate derived from a representative ultrasound cine loop, not a full echocardiography workstation measurement.",
                "Valve disease, regional wall-motion abnormalities, chamber dimensions, and pericardial findings require formal echocardiographic review.",
            ],
        }


class UltrasoundHuggingFaceAdapter(HuggingFace2DAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = os.getenv("ENABLE_US_HF_INFERENCE", "true").lower() == "true"
        self.model_id = os.getenv("US_HF_MODEL_ID", "").strip()
        self.task = os.getenv("US_HF_TASK", "image-text-to-text").strip().lower()
        self.allow_non_xray = True
        self._pipeline = None

    def supports(self, context: StudyContext) -> bool:
        if not self.enabled or not self.model_id:
            return False
        if context.modality.upper() != "US":
            return False
        series, _ = context.load_primary_series()
        return bool(series) and context.load_projection_image() is not None


class XAStenUNetAdapter(ModalityAdapter):
    def __init__(self) -> None:
        self.enabled = os.getenv("ENABLE_STENUNET_XA", "true").lower() == "true"
        self.repo_path = os.getenv("STENUNET_REPO_PATH", "/opt/external/StenUNet")
        self.checkpoint_path = os.getenv("STENUNET_CHECKPOINT_PATH", "/opt/model_weights/stenunet_shared_weights.pth")
        self.pixel_ratio_threshold = float(os.getenv("STENUNET_PIXEL_RATIO_THRESHOLD", "0.001"))

    def supports(self, context: StudyContext) -> bool:
        return (
            self.enabled
            and context.modality.upper() == "XA"
            and os.path.isdir(self.repo_path)
            and os.path.isfile(self.checkpoint_path)
        )

    @staticmethod
    def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame, dtype=np.float32)
        arr = np.squeeze(arr)
        arr -= float(arr.min())
        max_val = float(arr.max())
        if max_val > 0:
            arr /= max_val
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    def _export_xa_frames(self, context: StudyContext, raw_dir: str) -> int:
        import cv2
        import pydicom

        exported = 0
        for path in context.files:
            ds = pydicom.dcmread(path, force=True)
            if not hasattr(ds, "PixelData"):
                continue
            pixels = np.asarray(ds.pixel_array)
            pixels = np.squeeze(pixels)
            frames: List[np.ndarray]
            if pixels.ndim == 2:
                frames = [pixels]
            elif pixels.ndim == 3:
                if pixels.shape[-1] <= 4 and pixels.shape[0] > 4 and pixels.shape[1] > 4:
                    frames = [pixels[..., index] for index in range(pixels.shape[-1])]
                else:
                    frames = [np.squeeze(frame) for frame in pixels]
            else:
                flattened = pixels.reshape((-1, pixels.shape[-2], pixels.shape[-1]))
                frames = [np.squeeze(frame) for frame in flattened]

            for frame in frames:
                out_name = f"xa_{exported:04d}_0000.png"
                cv2.imwrite(os.path.join(raw_dir, out_name), self._frame_to_uint8(frame))
                exported += 1
        return exported

    def analyze(self, context: StudyContext) -> Dict[str, Any]:
        import cv2
        import torch

        # StenUNet/nnU-Net checkpoints rely on full torch checkpoint loading.
        # PyTorch 2.6 changed torch.load default behavior to weights_only=True,
        # which breaks these research checkpoints unless explicitly disabled.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)

        from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data as predict
        from post_process.remove_small_segments import remove_small_segments
        from pre_process.preprocess import preprocess

        with tempfile.TemporaryDirectory(prefix="stenunet-xa-") as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            pre_dir = os.path.join(temp_dir, "preprocessed")
            pred_dir = os.path.join(temp_dir, "raw_prediction")
            post_dir = os.path.join(temp_dir, "post_prediction")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(pre_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(post_dir, exist_ok=True)

            frame_count = self._export_xa_frames(context, raw_dir)
            if frame_count == 0:
                raise RuntimeError("No XA frames could be extracted from the DICOM study.")

            for file_name in os.listdir(raw_dir):
                raw_image = cv2.imread(os.path.join(raw_dir, file_name), cv2.IMREAD_GRAYSCALE)
                pre_image = preprocess(raw_image)
                cv2.imwrite(os.path.join(pre_dir, file_name), pre_image)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            predict(
                list_of_lists_or_source_folder=pre_dir,
                output_folder=pred_dir,
                model_training_output_dir=os.path.join(self.repo_path, "model_folder"),
                use_folds=[0],
                checkpoint_name=self.checkpoint_path,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
                device=device,
            )
            # StenUNet post-processing concatenates folder + filename directly,
            # so provide a trailing separator to avoid malformed paths.
            remove_small_segments(pred_dir + os.sep, post_dir + os.sep, threshold=600)

            positive_ratios: List[float] = []
            positive_frames = 0
            for file_name in os.listdir(post_dir):
                mask = cv2.imread(os.path.join(post_dir, file_name), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                ratio = float(np.count_nonzero(mask > 0)) / float(mask.size or 1)
                positive_ratios.append(ratio)
                if ratio >= self.pixel_ratio_threshold:
                    positive_frames += 1

            max_ratio = max(positive_ratios) if positive_ratios else 0.0
            mean_ratio = float(np.mean(positive_ratios)) if positive_ratios else 0.0
            abnormal = positive_frames > 0

            if abnormal:
                finding = (
                    f"Candidate stenosis-like regions were marked on XA screening in "
                    f"{positive_frames} of {frame_count} frame(s)."
                )
                recommendation = "Escalate for interventional/radiologist review and correlate with angiographic interpretation."
            else:
                finding = f"XA screening did not mark a dominant candidate stenosis region across {frame_count} frame(s)."
                recommendation = "Review angiography with a qualified reader; this screening result is not a final diagnosis."

            return {
                "model_name": "stenunet-xa",
                "finding": finding,
                "abnormal": abnormal,
                "confidence": None,
                "diagnostic_support": "screening-only",
                "report_type": "preliminary-stenosis-screening",
                "analysis_type": "open-source-xa-stenosis-segmentation",
                "analysis_status": "abnormal" if abnormal else "no-dominant-abnormality-detected",
                "anatomy_involved": context.body_part,
                "region": "coronary-angiography-frame-space",
                "observations": [
                    f"StenUNet processed {frame_count} XA frame(s).",
                    f"Frames with candidate segmented narrowing regions: {positive_frames}.",
                    f"Maximum positive-mask ratio across frames: {round(max_ratio, 6)}.",
                ],
                "abnormalities": (
                    ["candidate stenosis region(s) detected on XA screening"] if abnormal else []
                ),
                "impact": (
                    "Candidate stenosis regions were marked by the research XA model and should be reviewed on the angiographic run."
                    if abnormal else
                    "No dominant screening region was marked, but angiographic interpretation still requires a qualified clinician."
                ),
                "recommendation": recommendation,
                "metrics": {
                    "frame_count": float(frame_count),
                    "positive_frames": float(positive_frames),
                    "max_positive_pixel_ratio": round(max_ratio, 6),
                    "mean_positive_pixel_ratio": round(mean_ratio, 6),
                },
                "limitations": [
                    "StenUNet is a research XA stenosis screening model, not a validated final diagnosis engine.",
                    "Results depend on checkpoint availability and matching preprocessing assumptions.",
                ],
            }


class MRBraTSBundleAdapter(ModalityAdapter):
    def __init__(self) -> None:
        self.bundle_dir = os.getenv("MR_MONAI_BUNDLE_DIR", "/opt/external/monai_bundles/brats_mri_segmentation")
        self.hf_repo_id = os.getenv("MR_MONAI_HF_REPO_ID", "").strip()
        self.checkpoint_path = os.getenv("MR_MONAI_CHECKPOINT_PATH", "").strip()
        self._model = None
        self._loaded_path: Optional[str] = None
        self._sequence_labels = ("T1c", "T1", "T2", "FLAIR")

    def _resolve_bundle(self) -> Tuple[str, str]:
        bundle_dir = resolve_hf_bundle_dir(
            local_dir=self.bundle_dir,
            repo_id=self.hf_repo_id,
            required_paths=("configs/inference.json", "models/model.pt"),
        )
        checkpoint_path = self.checkpoint_path or os.path.join(bundle_dir, "models", "model.pt")
        return bundle_dir, checkpoint_path

    @staticmethod
    def _series_text(series: List[Any]) -> str:
        first = series[0]
        series_desc = str(getattr(first, "SeriesDescription", "") or "")
        protocol = str(getattr(first, "ProtocolName", "") or "")
        return f"{series_desc} {protocol}".strip().lower()

    def supports(self, context: StudyContext) -> bool:
        if context.modality.upper() not in {"MR", "MRI"}:
            return False
        if "BRAIN" not in context.body_part.upper():
            return False
        self.bundle_dir, self.checkpoint_path = self._resolve_bundle()
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            return False
        if not os.path.isdir(self.bundle_dir):
            return False
        sequence_map = self._collect_brats_sequences(context)
        return all(label in sequence_map for label in self._sequence_labels)

    def _ensure_model(self):
        if self._model is not None and self._loaded_path == self.checkpoint_path:
            return
        import torch
        from monai.networks.nets import SegResNet

        device = "cuda" if os.getenv("ENABLE_GPU_INFERENCE", "true").lower() == "true" and torch.cuda.is_available() else "cpu"
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        ).to(device)

        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
        else:
            state_dict = checkpoint
        cleaned = {}
        for key, value in state_dict.items():
            resolved_key = key[7:] if isinstance(key, str) and key.startswith("module.") else key
            cleaned[resolved_key] = value
        model.load_state_dict(cleaned, strict=True)
        model.eval()

        self._model = (model, device)
        self._loaded_path = self.checkpoint_path

    def _classify_sequence(self, text: str) -> Optional[str]:
        if "flair" in text:
            return "FLAIR"
        if "t2" in text and "flair" not in text:
            return "T2"
        if "t1" in text:
            if any(token in text for token in ("t1c", "post", "ce", "contrast", "gd", "gad")):
                return "T1c"
            return "T1"
        return None

    def _collect_brats_sequences(self, context: StudyContext) -> Dict[str, List[Any]]:
        series_map = context.load_series_map()
        labeled: Dict[str, List[Tuple[int, List[Any]]]] = defaultdict(list)
        for series in series_map.values():
            if not series:
                continue
            label = self._classify_sequence(self._series_text(series))
            if not label:
                continue
            frame_count = sum(context.dataset_frame_count(ds) for ds in series)
            labeled[label].append((frame_count, series))

        selected: Dict[str, List[Any]] = {}
        for label in self._sequence_labels:
            if labeled.get(label):
                selected[label] = max(labeled[label], key=lambda item: item[0])[1]
        return selected

    @staticmethod
    def _resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
        return resized.squeeze(0).squeeze(0).cpu().numpy()

    def analyze(self, context: StudyContext) -> Dict[str, Any]:
        import torch
        from monai.inferers import sliding_window_inference

        sequence_map = self._collect_brats_sequences(context)
        missing = [label for label in self._sequence_labels if label not in sequence_map]
        if missing:
            raise RuntimeError(
                f"BraTS MRI bundle requires T1c, T1, T2, and FLAIR series. Missing: {', '.join(missing)}."
            )

        volumes: Dict[str, np.ndarray] = {}
        metrics: Dict[str, Any] = {}
        target_shape: Optional[Tuple[int, int, int]] = None
        for label in self._sequence_labels:
            volume, vol_stats = context.build_volume(sequence_map[label])
            if volume is None:
                raise RuntimeError(f"Could not construct {label} volume for BraTS MRI bundle.")
            if target_shape is None:
                target_shape = tuple(int(dim) for dim in volume.shape)
            volumes[label] = volume
            metrics[f"{label.lower()}_slice_count"] = float(vol_stats.get("slice_count", volume.shape[0]))

        assert target_shape is not None
        stacked = []
        for label in self._sequence_labels:
            volume = volumes[label]
            if tuple(int(dim) for dim in volume.shape) != target_shape:
                volume = self._resize_volume(volume, target_shape)
            stacked.append(volume)

        self._ensure_model()
        model, device = self._model

        input_tensor = torch.from_numpy(np.stack(stacked, axis=0)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )
            probs = torch.sigmoid(logits)
            pred = probs > 0.5

        tumor_core = pred[0, 0].detach().cpu().numpy().astype(np.uint8)
        whole_tumor = pred[0, 1].detach().cpu().numpy().astype(np.uint8)
        enhancing_tumor = pred[0, 2].detach().cpu().numpy().astype(np.uint8)
        whole_voxels = int(np.count_nonzero(whole_tumor))
        core_voxels = int(np.count_nonzero(tumor_core))
        enhancing_voxels = int(np.count_nonzero(enhancing_tumor))
        total_voxels = int(np.prod(whole_tumor.shape))
        whole_ratio = float(whole_voxels) / float(total_voxels or 1)
        abnormal = whole_voxels > 0

        if abnormal:
            finding = (
                f"BraTS-style brain MRI segmentation marked candidate tumor regions across "
                f"{whole_voxels} voxels ({whole_ratio:.4%} of the analyzed volume)."
            )
            recommendation = "Escalate for neuroradiology review and correlate with the full multiparametric brain MRI exam."
        else:
            finding = "BraTS-style brain MRI segmentation did not mark a dominant tumor region in the analyzed volume."
            recommendation = "Proceed with neuroradiology review; this screening result is not a final diagnosis."

        abnormalities: List[str] = []
        if whole_voxels > 0:
            abnormalities.append("candidate whole-tumor region")
        if core_voxels > 0:
            abnormalities.append("candidate tumor-core region")
        if enhancing_voxels > 0:
            abnormalities.append("candidate enhancing-tumor region")

        return {
            "model_name": "brats-mri-segresnet",
            "finding": finding,
            "abnormal": abnormal,
            "confidence": None,
            "diagnostic_support": "screening-only",
            "report_type": "preliminary-brain-tumor-segmentation",
            "analysis_type": "monai-brats-mri-segmentation",
            "analysis_status": "abnormal" if abnormal else "no-dominant-abnormality-detected",
            "anatomy_involved": "BRAIN",
            "region": "brain-volume",
            "observations": [
                "MONAI BraTS SegResNet processed a 4-sequence brain MRI input (T1c, T1, T2, FLAIR).",
                f"Whole-tumor voxels: {whole_voxels}, tumor-core voxels: {core_voxels}, enhancing-tumor voxels: {enhancing_voxels}.",
                f"Input sequences used: {', '.join(self._sequence_labels)}.",
            ],
            "abnormalities": abnormalities,
            "impact": (
                "Candidate tumor regions were marked by the BraTS segmentation model and require neuroradiologist review."
                if abnormal else
                "No dominant tumor region was marked, but neuroradiologist review remains required."
            ),
            "recommendation": recommendation,
            "metrics": {
                **metrics,
                "whole_tumor_voxels": float(whole_voxels),
                "tumor_core_voxels": float(core_voxels),
                "enhancing_tumor_voxels": float(enhancing_voxels),
                "whole_tumor_ratio": round(whole_ratio, 6),
                "target_depth": float(target_shape[0]),
                "target_height": float(target_shape[1]),
                "target_width": float(target_shape[2]),
            },
            "limitations": [
                "BraTS MRI segmentation is a research brain tumor segmentation model, not a final diagnosis engine.",
                "This route requires a full four-sequence brain MRI set with T1c, T1, T2, and FLAIR inputs.",
            ],
        }


class MRSynthSegAdapter(ModalityAdapter):
    def __init__(self) -> None:
        self.endpoint = os.getenv("SYNTHSEG_URL", "http://synthseg_inference:8001").rstrip("/")
        self.timeout = int(os.getenv("SYNTHSEG_TIMEOUT", "900"))

    def supports(self, context: StudyContext) -> bool:
        return bool(self.endpoint) and context.modality.upper() in {"MR", "MRI"} and "BRAIN" in context.body_part.upper()

    def analyze(self, context: StudyContext) -> Dict[str, Any]:
        files = []
        handles = []
        try:
            for path in context.files:
                handle = open(path, "rb")
                handles.append(handle)
                files.append(("files", (os.path.basename(path), handle, "application/dicom")))

            response = requests.post(f"{self.endpoint}/upload", files=files, timeout=self.timeout)
            if response.status_code >= 400:
                detail = response.text.strip()
                if len(detail) > 1200:
                    detail = detail[:1200] + "...<truncated>"
                raise RuntimeError(
                    f"SynthSeg service returned HTTP {response.status_code}: {detail or 'no response body'}"
                )
            payload = response.json()
            if not payload.get("ok"):
                raise RuntimeError(payload.get("error") or "SynthSeg service returned an unsuccessful response.")
            report = payload.get("report") or {}
            if not report:
                raise RuntimeError("SynthSeg service did not return a report payload.")
            return report
        finally:
            for handle in handles:
                try:
                    handle.close()
                except Exception:
                    pass


class CTLungNoduleBundleAdapter(ModalityAdapter):
    def __init__(self) -> None:
        self.bundle_dir = os.getenv("CT_MONAI_BUNDLE_DIR", "/opt/external/monai_bundles/lung_nodule_ct_detection")
        self.hf_repo_id = os.getenv("CT_MONAI_HF_REPO_ID", "").strip()
        self.checkpoint_path = os.getenv(
            "CT_MONAI_CHECKPOINT_PATH",
            os.path.join(self.bundle_dir, "models", "model.pt"),
        ).strip()
        self.device_name = (
            "cuda"
            if os.getenv("ENABLE_GPU_INFERENCE", "true").lower() == "true" and __import__("torch").cuda.is_available()
            else "cpu"
        )
        self._runtime: Optional[Dict[str, Any]] = None
        self._loaded_path: Optional[str] = None
        self.score_threshold = float(os.getenv("CT_LUNG_NODULE_SCORE_THRESHOLD", "0.15"))
        self._chest_keywords = ("chest", "lung", "thorax", "pulm", "nodule")

    def _resolve_bundle(self) -> Tuple[str, str]:
        bundle_dir = resolve_hf_bundle_dir(
            local_dir=self.bundle_dir,
            repo_id=self.hf_repo_id,
            required_paths=("configs/inference.json", "models/model.pt"),
        )
        checkpoint_path = self.checkpoint_path or os.path.join(bundle_dir, "models", "model.pt")
        return bundle_dir, checkpoint_path

    def _context_text(self, context: StudyContext) -> str:
        parts = [context.body_part, context.metadata.get("StudyDescription", ""), context.metadata.get("SeriesDescription", ""), context.metadata.get("ProtocolName", "")]
        return " ".join(str(part or "") for part in parts).lower()

    def supports(self, context: StudyContext) -> bool:
        if context.modality.upper() != "CT":
            return False
        self.bundle_dir, self.checkpoint_path = self._resolve_bundle()
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            return False
        if not os.path.isdir(self.bundle_dir):
            return False
        return any(keyword in self._context_text(context) for keyword in self._chest_keywords)

    def _ensure_runtime(self) -> None:
        if self._runtime is not None and self._loaded_path == self.checkpoint_path:
            return

        import torch
        from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
        from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
        from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
        from monai.networks.nets.resnet import resnet50
        from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd

        device = torch.device(self.device_name)
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[1, 2, 4],
            base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]],
        )
        backbone = resnet50(
            spatial_dims=3,
            n_input_channels=1,
            conv1_t_stride=[2, 2, 1],
            conv1_t_size=[7, 7, 7],
        )
        feature_extractor = resnet_fpn_feature_extractor(backbone, 3, False, [1, 2], None)
        network = RetinaNet(
            spatial_dims=3,
            num_classes=1,
            num_anchors=3,
            feature_extractor=feature_extractor,
            size_divisible=[16, 16, 8],
            use_list_output=False,
        ).to(device)

        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        network.load_state_dict(state_dict, strict=True)
        network.eval()

        detector = RetinaNetDetector(
            network=network,
            anchor_generator=anchor_generator,
            debug=False,
            spatial_dims=3,
            num_classes=1,
            size_divisible=[16, 16, 8],
        )
        detector.set_target_keys(box_key="box", label_key="label")
        detector.set_box_selector_parameters(
            score_thresh=0.02,
            topk_candidates_per_level=1000,
            nms_thresh=0.22,
            detections_per_img=300,
        )
        detector.set_sliding_window_inferer(
            roi_size=[512, 512, 192],
            overlap=0.25,
            sw_batch_size=1,
            mode="constant",
            device="cpu",
        )

        preprocessing = Compose(
            [
                LoadImaged(keys="image"),
                EnsureChannelFirstd(keys="image"),
                Orientationd(keys="image", axcodes="RAS"),
                Spacingd(keys="image", pixdim=[0.703125, 0.703125, 1.25]),
                ScaleIntensityRanged(keys="image", a_min=-1024.0, a_max=300.0, b_min=0.0, b_max=1.0, clip=True),
                EnsureTyped(keys="image"),
            ]
        )

        self._runtime = {
            "device": device,
            "network": network,
            "detector": detector,
            "preprocessing": preprocessing,
            "sliding_window_size": int(np.prod([512, 512, 192])),
        }
        self._loaded_path = self.checkpoint_path

    def analyze(self, context: StudyContext) -> Dict[str, Any]:
        import nibabel as nib
        import torch

        self._ensure_runtime()
        runtime = self._runtime or {}
        series, series_stats = context.load_primary_series()
        volume_hu, affine, hu_stats = context.build_hu_volume(series)
        if volume_hu is None:
            raise RuntimeError("No usable CT volume could be constructed for lung nodule detection.")

        with tempfile.TemporaryDirectory(prefix="ct-lung-bundle-") as temp_dir:
            nifti_path = os.path.join(temp_dir, "ct_series.nii.gz")
            nib.save(nib.Nifti1Image(volume_hu.astype(np.float32), affine), nifti_path)
            data = runtime["preprocessing"]({"image": nifti_path})
            image = data["image"].to(runtime["device"])
            inputs = [image]
            detector = runtime["detector"]
            detector.network = runtime["network"]
            detector.training = detector.network.training
            use_inferer = not all(inp[0, ...].numel() < runtime["sliding_window_size"] for inp in inputs)

            with torch.no_grad():
                outputs = detector(inputs, use_inferer=use_inferer)

        prediction = outputs[0] if isinstance(outputs, list) and outputs else outputs
        boxes = prediction.get("box") if isinstance(prediction, dict) else None
        scores = prediction.get("label_scores") if isinstance(prediction, dict) else None
        if boxes is None or scores is None:
            raise RuntimeError("Lung nodule detector did not return detection boxes and scores.")

        boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
        scores_np = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
        if scores_np.ndim > 1:
            scores_np = np.squeeze(scores_np, axis=-1)

        keep_mask = scores_np >= self.score_threshold
        kept_scores = scores_np[keep_mask]
        kept_boxes = boxes_np[keep_mask] if len(boxes_np) == len(scores_np) else boxes_np[: len(kept_scores)]
        order = np.argsort(-kept_scores) if kept_scores.size else np.array([], dtype=int)
        top_scores = [round(float(kept_scores[idx]), 4) for idx in order[:5]]
        top_boxes = []
        for idx in order[:3]:
            box = kept_boxes[idx]
            if len(box) >= 6:
                top_boxes.append([round(float(v), 2) for v in box[:6]])

        candidate_count = int(kept_scores.size)
        abnormal = candidate_count > 0
        top_score = float(top_scores[0]) if top_scores else 0.0

        finding = (
            f"Chest CT lung nodule screening detected {candidate_count} candidate nodule(s); highest score {top_score:.3f}."
            if abnormal
            else "Chest CT lung nodule screening did not detect a candidate pulmonary nodule above the configured threshold."
        )

        recommendation = (
            "Review the flagged chest CT findings with a radiologist and correlate with the full examination."
            if abnormal
            else "Continue radiologist review; this screening result does not replace clinical interpretation."
        )

        return {
            "model_name": "monai-lung-nodule-retinanet",
            "finding": finding,
            "abnormal": abnormal,
            "confidence": round(top_score, 4) if abnormal else None,
            "diagnostic_support": "screening-only",
            "report_type": "preliminary-lung-nodule-screening",
            "analysis_type": "monai-lung-nodule-detection",
            "analysis_status": "abnormal" if abnormal else "no-dominant-abnormality-detected",
            "anatomy_involved": "CHEST",
            "region": "lungs/chest-ct-volume",
            "observations": [
                f"MONAI RetinaNet processed a chest CT volume with {int(series_stats.get('slice_count', 0) or hu_stats.get('slice_count', 0) or 0)} slice(s).",
                f"Candidate nodules above threshold: {candidate_count}.",
                f"Top detection scores: {', '.join(str(score) for score in top_scores)}." if top_scores else "No detection scores exceeded the configured reporting threshold.",
            ],
            "abnormalities": (["candidate pulmonary nodule"] if abnormal else []),
            "impact": (
                "Candidate pulmonary nodules were marked by the chest CT detection model and should be reviewed by a radiologist."
                if abnormal
                else "No candidate pulmonary nodule was marked above threshold, but radiologist review remains required."
            ),
            "recommendation": recommendation,
            "metrics": {
                **hu_stats,
                "candidate_count": float(candidate_count),
                "top_score": round(top_score, 4),
                "score_threshold": round(self.score_threshold, 4),
                "top_scores": top_scores,
                "top_boxes_xyzxyz": top_boxes,
            },
            "limitations": [
                "This MONAI bundle is a LUNA16-trained chest CT pulmonary nodule detector and is not a final diagnosis engine.",
                "The route is intended for chest CT studies and may not generalize to non-chest CT protocols.",
            ],
        }


class TotalSegmentatorAdapter(ModalityAdapter):
    def __init__(self, modality: str) -> None:
        self.modality = modality
        self.enabled = os.getenv("ENABLE_TOTALSEGMENTATOR", "true").lower() == "true"
        self.fast_mode = os.getenv("TOTALSEGMENTATOR_FAST", "true").lower() == "true"
        self.fallback = HeuristicFallbackEngine()

    def supports(self, context: StudyContext) -> bool:
        if not self.enabled:
            return False
        series, series_stats = context.load_primary_series()
        slice_count = int(series_stats.get("slice_count", 0))
        if not series or slice_count < 8:
            return False
        if self.modality == "CT":
            return context.modality == "CT"
        return context.modality in {"MR", "MRI"}

    def _task_name(self) -> str:
        return "total" if self.modality == "CT" else "total_mr"

    def _run_totalsegmentator(self, context: StudyContext) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        import nibabel as nib

        output_dir = tempfile.mkdtemp(prefix=f"totalseg-{self.modality.lower()}-")
        command = [
            "TotalSegmentator",
            "-i",
            context.dicom_path,
            "-o",
            output_dir,
            "--task",
            self._task_name(),
            "--statistics",
        ]
        if self.fast_mode:
            command.append("--fast")

        completed = subprocess.run(command, capture_output=True, text=True, check=True)

        anatomy_volumes: List[Tuple[str, float]] = []
        for filename in os.listdir(output_dir):
            lower = filename.lower()
            if not (lower.endswith(".nii.gz") or lower.endswith(".nii")):
                continue
            path = os.path.join(output_dir, filename)
            try:
                image = nib.load(path)
                data = image.get_fdata()
                voxels = float(np.count_nonzero(data > 0))
                if voxels <= 0:
                    continue
                zooms = image.header.get_zooms()[:3]
                voxel_volume_ml = float(np.prod(zooms)) / 1000.0 if zooms else 0.0
                anatomy_volumes.append((filename.replace(".nii.gz", "").replace(".nii", ""), round(voxels * voxel_volume_ml, 2)))
            except Exception:
                continue

        anatomy_volumes.sort(key=lambda item: item[1], reverse=True)
        stats = {
            "task": self._task_name(),
            "segmented_structure_count": float(len(anatomy_volumes)),
            "stdout_tail": completed.stdout[-1000:],
        }
        return anatomy_volumes[:10], stats

    def analyze(self, context: StudyContext) -> Dict[str, Any]:
        series, series_stats = context.load_primary_series()
        volume, vol_stats = context.build_volume(series)

        try:
            structures, seg_stats = self._run_totalsegmentator(context)
            top_structures = [name for name, _ in structures[:5]]
            neutral_stats: List[str] = []
            slice_count = int(series_stats.get("slice_count", 0))
            std_value = float(vol_stats.get("std", 0.0))
            mean_value = float(vol_stats.get("mean", 0.0))

            if slice_count:
                neutral_stats.append(f"Series contained {slice_count} slices.")
            if std_value:
                spread = "low"
                if std_value >= 0.12:
                    spread = "moderate"
                if std_value >= 0.22:
                    spread = "high"
                neutral_stats.append(f"Image intensity spread was {spread} for this series.")
            if mean_value:
                neutral_stats.append(f"Normalized mean intensity was {mean_value:.3f}.")

            observations = [
                f"TotalSegmentator completed {self._task_name()} inference for a {self.modality} series.",
                "Anatomy segmentation completed successfully for this series.",
            ]
            observations.extend(neutral_stats[:2])
            if top_structures:
                observations.append(f"Largest segmented structures: {', '.join(top_structures)}.")

            report = {
                "model_name": f"totalsegmentator-{self._task_name()}",
                "finding": (
                    f"Non-diagnostic {self.modality} anatomy segmentation completed successfully. "
                    "No validated pathology classifier is configured for this modality/protocol in the current deployment."
                ),
                "abnormal": None,
                "confidence": None,
                "diagnostic_support": "anatomy-only",
                "report_type": "non-diagnostic-anatomy",
                "analysis_type": "open-source-anatomy-segmentation",
                "analysis_status": "anatomy-segmentation-completed",
                "anatomy_involved": context.body_part,
                "region": "segmented-anatomy-available",
                "observations": observations,
                "abnormalities": [],
                "impact": (
                    "Anatomy segmentation is available for review, but pathology interpretation still requires "
                    "a radiologist or a validated modality-specific pathology model."
                ),
                "recommendation": (
                    "Use this result as anatomy context only. Pathology interpretation requires radiologist review."
                ),
                "metrics": {**vol_stats, **seg_stats, "top_segmented_structures_ml": dict(structures)},
                "limitations": [
                    "TotalSegmentator is an anatomy segmentation model, not a comprehensive pathology detector.",
                    "No validated pathology classifier is configured here for this modality/protocol.",
                ],
            }
            return report
        except Exception as exc:
            log.warning("TotalSegmentator adapter failed for %s: %s", self.modality, exc)
            return {
                "model_name": f"{self.modality.lower()}-analysis-unavailable",
                "finding": (
                    f"Automated {self.modality} analysis could not be completed because the "
                    f"{self._task_name()} backend failed. No diagnostic conclusion is available."
                ),
                "abnormal": None,
                "confidence": None,
                "diagnostic_support": "not-supported",
                "report_type": "analysis-failed-manual-review-required",
                "analysis_type": "analysis-failed",
                "analysis_status": "model-backend-failed",
                "anatomy_involved": context.body_part,
                "region": "not-available",
                "observations": [
                    f"Primary {self.modality} analysis backend {self._task_name()} failed for this study.",
                    "No reliable fallback diagnostic model is configured for this modality/protocol.",
                ],
                "abnormalities": [],
                "impact": "Automated interpretation is unavailable for this study.",
                "recommendation": "Send for radiologist/manual review.",
                "metrics": {},
                "limitations": [
                    f"TotalSegmentator {self._task_name()} backend was unavailable: {exc}",
                    f"No validated fallback diagnostic model is configured for {self.modality}.",
                ],
            }


class AdapterRegistry:
    def __init__(self) -> None:
        self.adapters = {
            "ct_monai": CTLungNoduleBundleAdapter(),
            "mr_brats": MRBraTSBundleAdapter(),
            "mr_synthseg": MRSynthSegAdapter(),
            "xa_stenunet": XAStenUNetAdapter(),
            "us_echo_lite": EchoLiteLVFunctionAdapter(),
            "us_hf_2d": UltrasoundHuggingFaceAdapter(),
            "unsupported": UnsupportedModalityAdapter(),
            "xrayvision": XRayVisionAdapter(),
            "hf_2d": HuggingFace2DAdapter(),
            "totalseg_ct": TotalSegmentatorAdapter("CT"),
            "totalseg_mr": TotalSegmentatorAdapter("MR"),
            "fallback": FallbackAdapter(),
        }
        self.routing_rules = [
            {
                "name": "ct-monai-screening",
                "modalities": {"CT"},
                "adapter": "ct_monai",
                "reason": "Chest CT studies route to the MONAI lung nodule detection bundle when the bundle checkpoint is configured and the study appears chest-focused.",
            },
            {
                "name": "mr-brats-segmentation",
                "modalities": {"MR", "MRI"},
                "adapter": "mr_brats",
                "reason": "Brain MRI studies route to the BraTS SegResNet bundle when the required four-sequence input set is available.",
            },

            {
                "name": "us-echo-lite-lv-function",
                "modalities": {"US"},
                "adapter": "us_echo_lite",
                "reason": "Ultrasound echocardiographic cine clips route first to the lightweight LV function estimator to derive an EF-focused screening summary without heavy model downloads.",
            },
            {
                "name": "us-hf-2d-screening",
                "modalities": {"US"},
                "adapter": "us_hf_2d",
                "reason": "Ultrasound studies can route to a US-specific Hugging Face image-to-text fallback using a representative DICOM-derived PNG when configured.",
            },
            {
                "name": "xa-stenosis-screening",
                "modalities": {"XA"},
                "adapter": "xa_stenunet",
                "reason": "XA studies route to the StenUNet research screening model when the repository and checkpoint are available.",
            },
            {
                "name": "chest-radiograph-screening",
                "modalities": {"CR", "DX", "MG"},
                "body_parts": {"CHEST", "LUNG"},
                "adapter": "xrayvision",
                "reason": "Projection chest studies route to chest X-ray screening model.",
            },
            {
                "name": "hf-2d-screening",
                "modalities": {"CR", "DX", "MG", "CT", "MR", "MRI", "US", "RF", "XA"},
                "adapter": "hf_2d",
                "reason": "Projection X-ray and other single-frame studies can route to a Hugging Face 2D model using a DICOM-derived PNG when configured.",
            },
            {
                "name": "unsupported-modalities",
                "modalities": {"XA", "RF", "US", "NM", "PT", "OCT"},
                "adapter": "unsupported",
                "reason": "Unsupported modality is routed directly to manual review.",
            },
            {
                "name": "ct-anatomy-segmentation",
                "modalities": {"CT"},
                "adapter": "totalseg_ct",
                "reason": "CT studies route to TotalSegmentator CT anatomy pipeline when no pathology model is configured.",
            },
            {
                "name": "mr-anatomy-segmentation",
                "modalities": {"MR", "MRI"},
                "adapter": "totalseg_mr",
                "reason": "MR studies route to TotalSegmentator MR anatomy pipeline when no pathology model is configured.",
            },
            {
                "name": "fallback",
                "modalities": set(),
                "adapter": "fallback",
                "reason": "No modality-specific route matched, so the deterministic fallback engine is used.",
            },
        ]

    def _select_adapters(self, context: StudyContext) -> List[tuple[str, ModalityAdapter, str]]:
        modality = context.modality.upper()
        body_part = context.body_part.upper()
        selected: List[tuple[str, ModalityAdapter, str]] = []
        seen: set[str] = set()

        for rule in self.routing_rules:
            allowed_modalities = rule.get("modalities") or set()
            allowed_bodies = rule.get("body_parts") or set()
            if allowed_modalities and modality not in allowed_modalities:
                continue
            if allowed_bodies and not any(token in body_part for token in allowed_bodies):
                continue
            adapter_name = rule["adapter"]
            if adapter_name in seen:
                continue
            adapter = self.adapters[adapter_name]
            selected.append((rule["name"], adapter, rule["reason"]))
            seen.add(adapter_name)

        if "fallback" not in seen:
            selected.append(("fallback", self.adapters["fallback"], "Fallback adapter appended as last resort."))
        return selected

    def analyze(self, context: StudyContext) -> Dict[str, Any]:
        errors: List[str] = []
        for route_name, adapter, route_reason in self._select_adapters(context):
            if not adapter.supports(context):
                continue
            try:
                result = adapter.analyze(context)
                result["adapter_name"] = adapter.__class__.__name__
                result["routing_decision"] = {
                    "route_name": route_name,
                    "reason": route_reason,
                    "modality": context.modality,
                    "body_part": context.body_part,
                }
                result["support_matrix"] = describe_support(route_name)
                if errors:
                    result.setdefault("limitations", []).append(
                        f"Earlier adapters failed before success: {'; '.join(errors)}"
                    )
                return result
            except Exception as exc:
                errors.append(f"{adapter.__class__.__name__}: {exc}")
                continue
        raise RuntimeError(f"No adapter succeeded for modality {context.modality}. Errors: {'; '.join(errors)}")



