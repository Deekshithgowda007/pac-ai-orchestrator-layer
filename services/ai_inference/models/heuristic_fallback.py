import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pydicom


class HeuristicFallbackEngine:
    def __init__(self, development_mode: bool = True):
        self.development_mode = development_mode
        self.minimum_slices = 2 if development_mode else 20

    def run(self, dicom_path: str) -> Dict:
        if not os.path.exists(dicom_path):
            return self._fallback("Path not found")

        dicom_files = self._collect_dicom_files(dicom_path)
        if not dicom_files:
            return self._fallback("No DICOM files found")

        sample_ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True, force=True)
        modality = str(getattr(sample_ds, "Modality", "") or "").upper()
        body_part = self._clean_text(
            getattr(sample_ds, "BodyPartExamined", None)
            or getattr(sample_ds, "StudyDescription", None)
            or getattr(sample_ds, "SeriesDescription", None)
            or "UNKNOWN"
        )

        if modality in {"CT", "MR", "MRI"}:
            return self._analyze_volume(dicom_files, modality, body_part)

        if modality in {"CR", "DX", "XA", "RF", "MG"}:
            return self._analyze_projection(dicom_files[0], modality, body_part)

        return self._fallback(f"Unsupported modality: {modality or 'UNKNOWN'}")

    def _collect_dicom_files(self, dicom_path: str) -> List[str]:
        if os.path.isdir(dicom_path):
            files: List[str] = []
            for root, _, filenames in os.walk(dicom_path):
                for filename in filenames:
                    if filename.lower().endswith(".dcm"):
                        files.append(os.path.join(root, filename))
            return files
        return [dicom_path]

    def _analyze_volume(self, dicom_files: List[str], modality: str, body_part: str) -> Dict:
        series_map: Dict[str, List[pydicom.Dataset]] = defaultdict(list)
        for path in dicom_files:
            try:
                ds = pydicom.dcmread(path, force=True)
                if hasattr(ds, "PixelData"):
                    series_uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
                    series_map[series_uid].append(ds)
            except Exception:
                continue

        if not series_map:
            return self._fallback("No valid image slices")

        series = max(series_map.values(), key=len)
        estimated_slice_count = sum(self._dataset_frame_count(ds) for ds in series)
        if estimated_slice_count < self.minimum_slices:
            return {
                "model_name": f"{modality.lower()}-rule-based-fallback",
                "finding": f"Only {estimated_slice_count} {modality} slice(s) available; full series required for volume analysis.",
                "abnormal": False,
                "confidence": 0.0,
                "analysis_type": "insufficient-series",
                "anatomy_involved": body_part,
                "region": "diffuse/undetermined",
                "observations": [
                    f"Received only {estimated_slice_count} image slice(s) for a {modality} study.",
                    f"{modality} volume analysis requires a full image series, not a single exported instance.",
                    "Current result is non-diagnostic and should not be treated as a radiology conclusion.",
                ],
                "impact": "Upload the complete DICOM series or full study to enable modality-specific analysis.",
                "metrics": {"slice_count": float(estimated_slice_count)},
                "limitations": [
                    "A single CT or MR image is not enough for reliable volume-based analysis.",
                    "Automated output must be reviewed by a qualified radiologist.",
                ],
            }

        volume, stats = self._build_volume(series)
        if volume is None:
            return self._fallback("Unable to reconstruct volume")

        anomaly_mask = self._detect_anomaly_mask(volume)
        anomaly_fraction = float(anomaly_mask.mean())
        abnormal = anomaly_fraction > 0.015 or stats["std"] > 1.35
        region = self._describe_region_3d(anomaly_mask)
        confidence = self._bounded_confidence(0.35 + min(0.55, anomaly_fraction * 6.0 + stats["edge_density"]))

        observations = [
            f"{modality} series reconstructed with {int(stats['slice_count'])} slices.",
            f"Image intensity spread is {'broad' if stats['std'] > 1.35 else 'moderate'} for this series.",
            f"Estimated abnormal signal burden involves about {anomaly_fraction * 100:.1f}% of sampled voxels."
            if abnormal else
            "No dominant focal signal burden is detected by the fallback rules.",
        ]

        if region != "diffuse/undetermined":
            observations.append(f"Most conspicuous signal change is centered in the {region}.")
        else:
            observations.append("Distribution of signal change is diffuse or not spatially stable.")

        conclusion = (
            f"Preliminary automated review suggests abnormal {modality} appearance in the {region}."
            if abnormal else
            f"Preliminary automated review does not detect a dominant {modality} abnormality."
        )

        impact = (
            "Escalate for radiologist review because the fallback engine flagged potentially abnormal signal."
            if abnormal else
            "No dominant abnormality was flagged, but final interpretation still requires a radiologist."
        )

        return {
            "model_name": f"{modality.lower()}-rule-based-fallback",
            "finding": conclusion,
            "abnormal": abnormal,
            "confidence": confidence,
            "analysis_type": "preliminary-rule-based",
            "anatomy_involved": body_part,
            "region": region,
            "observations": observations,
            "impact": impact,
            "metrics": stats,
            "limitations": [
                "This engine uses intensity and morphology heuristics, not a trained diagnostic model.",
                "Automated output must be reviewed by a qualified radiologist.",
            ],
        }

    def _analyze_projection(self, dicom_file: str, modality: str, body_part: str) -> Dict:
        ds = pydicom.dcmread(dicom_file, force=True)
        if not hasattr(ds, "PixelData"):
            return self._fallback("No PixelData found")

        image = self._prepare_projection_image(ds.pixel_array.astype(np.float32))
        image = self._normalize(image)

        mean_intensity = float(image.mean())
        std_intensity = float(image.std())
        edge_density = float(self._edge_density(image))
        anomaly_mask = np.abs(image - mean_intensity) > max(0.18, std_intensity * 1.1)
        anomaly_fraction = float(anomaly_mask.mean())
        abnormal = anomaly_fraction > 0.08 or edge_density > 0.22
        region = self._describe_region_2d(anomaly_mask)
        confidence = self._bounded_confidence(0.35 + min(0.55, anomaly_fraction + edge_density))

        observations = [
            f"{modality} projection image analyzed using exposure and structure heuristics.",
            f"Exposure appears {'acceptable' if 0.2 <= mean_intensity <= 0.8 else 'suboptimal'} based on normalized intensity.",
            f"Structural edge density is {edge_density:.2f}, which is {'elevated' if edge_density > 0.22 else 'within fallback limits'}.",
        ]
        if abnormal:
            observations.append(f"Most conspicuous deviation is seen in the {region}.")
        else:
            observations.append("No focal projection abnormality is strongly emphasized by the fallback rules.")

        conclusion = (
            f"Preliminary automated review suggests a focal abnormal projection pattern in the {region}."
            if abnormal else
            "Preliminary automated review does not detect a dominant focal projection abnormality."
        )

        return {
            "model_name": f"{modality.lower()}-rule-based-fallback",
            "finding": conclusion,
            "abnormal": abnormal,
            "confidence": confidence,
            "analysis_type": "preliminary-rule-based",
            "anatomy_involved": body_part,
            "region": region,
            "observations": observations,
            "impact": (
                "Prompt radiologist review is advised because the projection image contains flagged focal change."
                if abnormal else
                "No dominant abnormality was flagged, but radiologist confirmation remains required."
            ),
            "metrics": {
                "mean_intensity": round(mean_intensity, 4),
                "std_intensity": round(std_intensity, 4),
                "edge_density": round(edge_density, 4),
                "anomaly_fraction": round(anomaly_fraction, 4),
            },
            "limitations": [
                "This engine is a deterministic fallback and not a validated diagnostic classifier.",
                "Localization is approximate and based on image heuristics.",
            ],
        }

    def _build_volume(self, series: List[pydicom.Dataset]) -> Tuple[np.ndarray | None, Dict[str, float]]:
        def sort_key(ds: pydicom.Dataset) -> float:
            if hasattr(ds, "ImagePositionPatient"):
                try:
                    return float(ds.ImagePositionPatient[2])
                except Exception:
                    pass
            return float(getattr(ds, "InstanceNumber", 0) or 0)

        ordered = sorted(series, key=sort_key)
        try:
            slices: List[np.ndarray] = []
            for ds in ordered:
                pixels = np.asarray(ds.pixel_array, dtype=np.float32)
                pixels = np.squeeze(pixels)
                if pixels.ndim == 2:
                    slices.append(pixels)
                elif pixels.ndim == 3:
                    if pixels.shape[-1] <= 4 and pixels.shape[0] > 4 and pixels.shape[1] > 4:
                        for index in range(pixels.shape[-1]):
                            slices.append(pixels[..., index])
                    else:
                        for frame in pixels:
                            slices.append(np.squeeze(frame))
                else:
                    reshaped = pixels.reshape((-1, pixels.shape[-2], pixels.shape[-1]))
                    for frame in reshaped:
                        slices.append(np.squeeze(frame))
            volume = np.stack(slices).astype(np.float32)
        except Exception:
            return None, {}

        slope = float(getattr(ordered[0], "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ordered[0], "RescaleIntercept", 0.0) or 0.0)
        volume = volume * slope + intercept
        volume = self._normalize(volume)

        gradients = np.gradient(volume)
        edge_density = float(np.mean(np.sqrt(sum(g * g for g in gradients)) > 0.25))
        stats = {
            "slice_count": float(volume.shape[0]),
            "mean": round(float(volume.mean()), 4),
            "std": round(float(volume.std()), 4),
            "min": round(float(volume.min()), 4),
            "max": round(float(volume.max()), 4),
            "edge_density": round(edge_density, 4),
        }
        return volume, stats

    def _detect_anomaly_mask(self, volume: np.ndarray) -> np.ndarray:
        z = (volume - float(volume.mean())) / (float(volume.std()) + 1e-6)
        return np.abs(z) > 2.0

    def _describe_region_3d(self, mask: np.ndarray) -> str:
        if not np.any(mask):
            return "diffuse/undetermined"
        coords = np.argwhere(mask)
        center = coords.mean(axis=0) / np.array(mask.shape)
        depth = "upper" if center[0] < 0.33 else "mid" if center[0] < 0.66 else "lower"
        vertical = "anterior/superior" if center[1] < 0.33 else "central" if center[1] < 0.66 else "posterior/inferior"
        lateral = "left" if center[2] < 0.33 else "midline" if center[2] < 0.66 else "right"
        return f"{depth} {vertical} {lateral}"

    def _describe_region_2d(self, mask: np.ndarray) -> str:
        if not np.any(mask):
            return "diffuse/undetermined"
        coords = np.argwhere(mask)
        center = coords.mean(axis=0) / np.array(mask.shape)
        vertical = "upper" if center[0] < 0.33 else "mid" if center[0] < 0.66 else "lower"
        lateral = "left" if center[1] < 0.33 else "central" if center[1] < 0.66 else "right"
        return f"{vertical} {lateral}"

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        arr -= float(arr.min())
        max_val = float(arr.max())
        if max_val > 0:
            arr /= max_val
        return arr

    def _edge_density(self, image: np.ndarray) -> float:
        if image.ndim != 2:
            image = self._prepare_projection_image(image)
        gradients = np.gradient(image)
        if len(gradients) < 2:
            return 0.0
        magnitude = np.sqrt(sum(gradient * gradient for gradient in gradients[:2]))
        return float(np.mean(magnitude > 0.2))

    def _prepare_projection_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        image = np.squeeze(image)
        if image.ndim == 2:
            return image
        if image.ndim == 3:
            axis = int(np.argmin(image.shape))
            return np.take(image, indices=0, axis=axis)
        if image.ndim > 3:
            reshape = image.reshape((-1, image.shape[-2], image.shape[-1]))
            return reshape[0]
        raise ValueError(f"Unsupported projection image shape: {image.shape}")

    def _dataset_frame_count(self, ds: pydicom.Dataset) -> int:
        number_of_frames = getattr(ds, "NumberOfFrames", None)
        if number_of_frames:
            try:
                return int(number_of_frames)
            except Exception:
                pass
        try:
            pixels = np.asarray(ds.pixel_array)
            pixels = np.squeeze(pixels)
            if pixels.ndim == 2:
                return 1
            if pixels.ndim == 3:
                if pixels.shape[-1] <= 4 and pixels.shape[0] > 4 and pixels.shape[1] > 4:
                    return int(pixels.shape[-1])
                return int(pixels.shape[0])
            if pixels.ndim > 3:
                return int(np.prod(pixels.shape[:-2]))
        except Exception:
            pass
        return 1

    def _bounded_confidence(self, value: float) -> float:
        return round(float(max(0.05, min(0.95, value))), 3)

    def _clean_text(self, value: object) -> str:
        text = str(value or "").strip()
        return text if text else "UNKNOWN"

    def _fallback(self, message: str) -> Dict:
        return {
            "model_name": "system-fallback",
            "finding": message,
            "abnormal": False,
            "confidence": 0.0,
            "analysis_type": "fallback-error",
            "anatomy_involved": "UNKNOWN",
            "region": "diffuse/undetermined",
            "observations": [message],
            "impact": "No automated impact assessment available.",
            "metrics": {},
            "limitations": ["The fallback engine could not complete analysis."],
        }
