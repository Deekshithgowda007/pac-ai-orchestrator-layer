import os
import pydicom
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from models.ct_model import CTVolumeModel
from models.mri_model import MRIVolumeModel
from models.xray_model import XRayModel


class InferenceEngine:

    def __init__(self, development_mode: bool = True):

        self.ct_model = CTVolumeModel()
        self.mri_model = MRIVolumeModel()
        self.xray_model = XRayModel()

        self.minimum_confidence = 0.6
        self.development_mode = development_mode
        self.minimum_slices = 2 if development_mode else 20

    # ==========================================================
    def run(self, dicom_path: str) -> Dict:

        if not os.path.exists(dicom_path):
            return self._fallback("Path not found")

        if os.path.isdir(dicom_path):
            dicom_files = self._get_dicom_files(dicom_path)
            if not dicom_files:
                return self._fallback("No DICOM files found")
        else:
            dicom_files = [dicom_path]

        # Read first file
        sample_ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
        modality = getattr(sample_ds, "Modality", None)

        if modality is None:
            return self._fallback("Missing Modality")

        if modality in ["CT", "MR"]:
            return self._handle_volume_modality(dicom_files, modality)

        if modality in ["CR", "DX"]:
            image = pydicom.dcmread(dicom_files[0]).pixel_array
            result = self.xray_model.predict(image)
            return self._postprocess(result)

        return self._fallback(f"Unsupported modality: {modality}")

    # ==========================================================
    def _handle_volume_modality(self, dicom_files, modality):

        series_map = defaultdict(list)

        for file in dicom_files:
            try:
                ds = pydicom.dcmread(file)
                if not hasattr(ds, "pixel_array"):
                    continue

                series_uid = getattr(ds, "SeriesInstanceUID", None)
                if series_uid:
                    series_map[series_uid].append(ds)
            except Exception:
                continue

        if not series_map:
            return self._fallback("No valid image slices")

        largest_series = max(series_map.values(), key=len)

        # 3D path
        if len(largest_series) >= self.minimum_slices:
            volume = self._build_volume_from_series(largest_series)

            if volume is not None:
                if modality == "CT":
                    result = self.ct_model.predict(volume)
                else:
                    result = self.mri_model.predict(volume)

                return self._postprocess(result)

        # -----------------------------
        # 2D FALLBACK
        # -----------------------------
        print("⚠️ Falling back to 2D slice inference")

        mid_slice = largest_series[len(largest_series) // 2]
        image = mid_slice.pixel_array

        result = self.xray_model.predict(image)
        result["model_name"] = f"{modality}_2D_Fallback"

        return self._postprocess(result)

    # ==========================================================
    def _build_volume_from_series(self, series):

        def sort_key(ds):
            if hasattr(ds, "ImagePositionPatient"):
                return float(ds.ImagePositionPatient[2])
            return float(getattr(ds, "InstanceNumber", 0))

        series.sort(key=sort_key)

        try:
            volume = np.stack([ds.pixel_array for ds in series])
        except Exception:
            return None

        volume = volume.astype(np.float32)

        slope = float(getattr(series[0], "RescaleSlope", 1.0))
        intercept = float(getattr(series[0], "RescaleIntercept", 0.0))
        volume = volume * slope + intercept

        std = np.std(volume)
        if std < 1e-6:
            std = 1e-6

        volume = (volume - np.mean(volume)) / std
        return volume

    # ==========================================================
    def _get_dicom_files(self, folder_path):
        files = []
        for root, _, filenames in os.walk(folder_path):
            for f in filenames:
                if f.lower().endswith(".dcm"):
                    files.append(os.path.join(root, f))
        return files

    # ==========================================================
    def _postprocess(self, result):

        confidence = float(result.get("confidence", 0.0))

        if confidence < self.minimum_confidence:
            result["finding"] = "Inconclusive - Requires Radiologist Review"
            result["abnormal"] = False

        return {
            "model_name": result.get("model_name", "UnknownModel"),
            "finding": result.get("finding", "Unknown"),
            "abnormal": bool(result.get("abnormal", False)),
            "confidence": confidence
        }

    # ==========================================================
    def _fallback(self, message):

        return {
            "model_name": "SystemGuard",
            "finding": message,
            "abnormal": False,
            "confidence": 0.0
        }
