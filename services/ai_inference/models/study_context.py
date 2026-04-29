from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
from PIL import Image

try:
    from pydicom.pixels import apply_voi_lut
except Exception:  # pragma: no cover - compatibility fallback
    from pydicom.pixel_data_handlers.util import apply_voi_lut


@dataclass
class StudyContext:
    dicom_path: str
    files: List[str]
    metadata: Dict[str, str]
    modality: str
    body_part: str
    series_uid: str
    study_uid: str

    def load_series_map(self) -> Dict[str, List[pydicom.Dataset]]:
        series_map: Dict[str, List[pydicom.Dataset]] = defaultdict(list)
        for path in self.files:
            try:
                ds = pydicom.dcmread(path, force=True)
                if hasattr(ds, "PixelData"):
                    series_map[str(getattr(ds, "SeriesInstanceUID", "unknown"))].append(ds)
            except Exception:
                continue
        return series_map

    def load_primary_series(self) -> Tuple[List[pydicom.Dataset], Dict[str, float]]:
        series_map = self.load_series_map()
        if not series_map:
            return [], {}
        primary_series = max(series_map.values(), key=len)
        slice_count = float(sum(self.dataset_frame_count(ds) for ds in primary_series))
        return primary_series, {"series_count": float(len(series_map)), "slice_count": slice_count}

    def _ordered_slices(self, series: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
        def sort_key(ds: pydicom.Dataset) -> float:
            if hasattr(ds, "ImagePositionPatient"):
                try:
                    return float(ds.ImagePositionPatient[2])
                except Exception:
                    pass
            return float(getattr(ds, "InstanceNumber", 0) or 0)

        return sorted(series, key=sort_key)

    @staticmethod
    def _extract_frames(ds: pydicom.Dataset) -> List[np.ndarray]:
        pixels = np.asarray(ds.pixel_array, dtype=np.float32)
        pixels = np.squeeze(pixels)
        if pixels.ndim == 2:
            return [pixels]
        if pixels.ndim == 3:
            if pixels.shape[-1] <= 4 and pixels.shape[0] > 4 and pixels.shape[1] > 4:
                return [pixels[..., index] for index in range(pixels.shape[-1])]
            return [np.squeeze(frame) for frame in pixels]
        reshaped = pixels.reshape((-1, pixels.shape[-2], pixels.shape[-1]))
        return [np.squeeze(frame) for frame in reshaped]

    def build_hu_volume(self, series: List[pydicom.Dataset]) -> Tuple[Optional[np.ndarray], np.ndarray, Dict[str, float]]:
        if not series:
            return None, np.eye(4, dtype=np.float32), {}

        ordered = self._ordered_slices(series)
        try:
            slices: List[np.ndarray] = []
            for ds in ordered:
                slices.extend(self._extract_frames(ds))
            volume = np.stack(slices, axis=-1).astype(np.float32)
        except Exception:
            return None, np.eye(4, dtype=np.float32), {}

        first = ordered[0]
        slope = float(getattr(first, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(first, "RescaleIntercept", 0.0) or 0.0)
        volume = volume * slope + intercept

        pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
        spacing_between = getattr(first, "SpacingBetweenSlices", None) or getattr(first, "SliceThickness", None) or 1.0
        try:
            sx = float(pixel_spacing[0])
            sy = float(pixel_spacing[1])
            sz = float(spacing_between)
        except Exception:
            sx, sy, sz = 1.0, 1.0, 1.0

        affine = np.diag([sx, sy, sz, 1.0]).astype(np.float32)
        if min(volume.shape) >= 2:
            gradients = np.gradient(volume)
            edge_density = float(np.mean(np.sqrt(sum(g * g for g in gradients)) > 75.0))
        else:
            edge_density = 0.0
        stats = {
            "slice_count": float(volume.shape[-1]),
            "mean_hu": round(float(volume.mean()), 4),
            "std_hu": round(float(volume.std()), 4),
            "min_hu": round(float(volume.min()), 4),
            "max_hu": round(float(volume.max()), 4),
            "edge_density": round(edge_density, 4),
            "spacing_x_mm": round(sx, 4),
            "spacing_y_mm": round(sy, 4),
            "spacing_z_mm": round(sz, 4),
        }
        return volume, affine, stats

    def build_volume(self, series: List[pydicom.Dataset]) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        volume_hu, _, hu_stats = self.build_hu_volume(series)
        if volume_hu is None:
            return None, {}
        volume = self.normalize(volume_hu)
        return volume, {
            "slice_count": hu_stats.get("slice_count", 0.0),
            "mean": round(float(volume.mean()), 4),
            "std": round(float(volume.std()), 4),
            "min": round(float(volume.min()), 4),
            "max": round(float(volume.max()), 4),
            "edge_density": hu_stats.get("edge_density", 0.0),
        }

    def _load_first_image_dataset(self) -> Optional[pydicom.Dataset]:
        if not self.files:
            return None
        try:
            ds = pydicom.dcmread(self.files[0], force=True)
            if not hasattr(ds, "PixelData"):
                return None
            return ds
        except Exception:
            return None

    @staticmethod
    def _extract_display_frame(ds: pydicom.Dataset) -> Optional[np.ndarray]:
        try:
            image = np.asarray(ds.pixel_array, dtype=np.float32)
        except Exception:
            return None
        image = np.squeeze(image)
        if image.ndim == 3:
            axis = int(np.argmin(image.shape))
            image = np.take(image, indices=0, axis=axis)
        elif image.ndim > 3:
            image = image.reshape((-1, image.shape[-2], image.shape[-1]))[0]
        return np.asarray(image, dtype=np.float32)

    @staticmethod
    def _to_display_uint8(ds: pydicom.Dataset, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept

        try:
            arr = np.asarray(apply_voi_lut(arr, ds), dtype=np.float32)
        except Exception:
            pass

        if str(getattr(ds, "PhotometricInterpretation", "") or "").upper() == "MONOCHROME1":
            arr = arr.max() - arr

        arr -= float(arr.min())
        max_val = float(arr.max())
        if max_val > 0:
            arr /= max_val
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    def load_projection_image(self) -> Optional[np.ndarray]:
        ds = self._load_first_image_dataset()
        if ds is None:
            return None
        image = self._extract_display_frame(ds)
        if image is None:
            return None
        return self.normalize(self._to_display_uint8(ds, image).astype(np.float32))

    def export_projection_png(self, output_path: str) -> bool:
        ds = self._load_first_image_dataset()
        if ds is None:
            return False
        image = self._extract_display_frame(ds)
        if image is None:
            return False
        Image.fromarray(self._to_display_uint8(ds, image), mode="L").save(output_path, format="PNG")
        return True

    def load_primary_cine_frames(self, max_frames: int = 48) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        series, series_stats = self.load_primary_series()
        if not series:
            return None, {}

        ordered = self._ordered_slices(series)
        collected: List[np.ndarray] = []
        for ds in ordered:
            for frame in self._extract_frames(ds):
                collected.append(self._to_display_uint8(ds, frame).astype(np.float32) / 255.0)

        if not collected:
            return None, {}

        total_frames = len(collected)
        if total_frames > max_frames:
            indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int)
            collected = [collected[int(index)] for index in indices]

        cine = np.stack(collected, axis=0).astype(np.float32)
        first = ordered[0]
        frame_time_ms = 0.0
        try:
            frame_time_ms = float(getattr(first, "FrameTime", 0.0) or 0.0)
        except Exception:
            frame_time_ms = 0.0

        stats = {
            "series_count": float(series_stats.get("series_count", 1.0)),
            "slice_count": float(cine.shape[0]),
            "frame_count": float(cine.shape[0]),
            "height": float(cine.shape[1]),
            "width": float(cine.shape[2]),
            "frame_time_ms": round(frame_time_ms, 4),
        }
        return cine, stats

    @staticmethod
    def normalize(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        arr -= float(arr.min())
        max_val = float(arr.max())
        if max_val > 0:
            arr /= max_val
        return arr

    @staticmethod
    def dataset_frame_count(ds: pydicom.Dataset) -> int:
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


def collect_study_context(dicom_path: str) -> Optional[StudyContext]:
    files: List[str] = []
    if os.path.isdir(dicom_path):
        for root, _, filenames in os.walk(dicom_path):
            for filename in filenames:
                if filename.lower().endswith(".dcm"):
                    files.append(os.path.join(root, filename))
    elif os.path.exists(dicom_path):
        files = [dicom_path]

    if not files:
        return None

    sample = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
    metadata: Dict[str, str] = {}
    for elem in sample:
        if elem.keyword and elem.keyword != "PixelData":
            try:
                metadata[elem.keyword] = str(elem.value)
            except Exception:
                metadata[elem.keyword] = repr(elem.value)

    modality = str(metadata.get("Modality", "") or "").upper()
    body_part = (
        metadata.get("BodyPartExamined")
        or metadata.get("StudyDescription")
        or metadata.get("SeriesDescription")
        or "UNKNOWN"
    )
    return StudyContext(
        dicom_path=dicom_path,
        files=sorted(files),
        metadata=metadata,
        modality=modality,
        body_part=str(body_part),
        series_uid=str(metadata.get("SeriesInstanceUID", "unknown")),
        study_uid=str(metadata.get("StudyInstanceUID", "unknown")),
    )
