import csv
import io
import os
import shutil
import subprocess
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pydicom
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from scipy.ndimage import zoom

load_dotenv()

app = Flask(__name__)

SYNTHSEG_HOME = os.getenv("SYNTHSEG_HOME", "/opt/external/SynthSeg")
SYNTHSEG_MODEL_VERSION = os.getenv("SYNTHSEG_MODEL_VERSION", "v1").lower()
SYNTHSEG_THREADS = os.getenv("SYNTHSEG_THREADS", "1")
SYNTHSEG_MAX_INPLANE = int(os.getenv("SYNTHSEG_MAX_INPLANE", "128"))
SYNTHSEG_MAX_VOXELS = int(os.getenv("SYNTHSEG_MAX_VOXELS", "1500000"))


def extract_metadata(file_bytes: bytes) -> Dict[str, str]:
    ds = pydicom.dcmread(io.BytesIO(file_bytes), stop_before_pixels=True, force=True)
    metadata: Dict[str, str] = {}
    for elem in ds:
        if elem.keyword and elem.keyword != "PixelData":
            try:
                metadata[elem.keyword] = str(elem.value)
            except Exception:
                metadata[elem.keyword] = repr(elem.value)
    return metadata


def run_synthseg(nifti_path: str, seg_path: str, volumes_csv: str) -> None:
    command = [
        "python",
        os.path.join(SYNTHSEG_HOME, "scripts", "commands", "SynthSeg_predict.py"),
        "--i",
        nifti_path,
        "--o",
        seg_path,
        "--vol",
        volumes_csv,
        "--cpu",
        "--threads",
        SYNTHSEG_THREADS,
    ]
    if SYNTHSEG_MODEL_VERSION == "v1":
        command.append("--v1")

    env = os.environ.copy()
    env["PYTHONPATH"] = SYNTHSEG_HOME + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(command, check=True, env=env, cwd=SYNTHSEG_HOME, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        if exc.returncode == -9:
            raise RuntimeError(
                "SynthSeg process was killed with SIGKILL, likely because the container ran out of memory during inference."
            ) from exc
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        if len(detail) > 1600:
            detail = detail[:1600] + "...<truncated>"
        raise RuntimeError(f"SynthSeg execution failed: {detail}") from exc


def maybe_downsample_volume(volume: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    original_shape = tuple(int(v) for v in volume.shape)
    factors = [1.0, 1.0, 1.0]

    max_inplane = max(volume.shape[0], volume.shape[1])
    if max_inplane > SYNTHSEG_MAX_INPLANE:
        inplane_factor = float(SYNTHSEG_MAX_INPLANE) / float(max_inplane)
        factors[0] = min(factors[0], inplane_factor)
        factors[1] = min(factors[1], inplane_factor)

    projected_voxels = float(np.prod(np.array(volume.shape, dtype=np.float64) * np.array(factors, dtype=np.float64)))
    if projected_voxels > SYNTHSEG_MAX_VOXELS:
        voxel_factor = (float(SYNTHSEG_MAX_VOXELS) / projected_voxels) ** (1.0 / 3.0)
        factors = [factor * voxel_factor for factor in factors]

    if all(abs(factor - 1.0) < 1e-3 for factor in factors):
        return volume, affine, {
            "input_rows": float(original_shape[0]),
            "input_cols": float(original_shape[1]),
            "input_slices": float(original_shape[2]),
            "resampled_rows": float(original_shape[0]),
            "resampled_cols": float(original_shape[1]),
            "resampled_slices": float(original_shape[2]),
            "downsample_factor_x": 1.0,
            "downsample_factor_y": 1.0,
            "downsample_factor_z": 1.0,
        }

    resized = zoom(volume, zoom=tuple(factors), order=1)
    new_affine = affine.copy()
    for axis, factor in enumerate(factors[:3]):
        if factor > 0:
            new_affine[axis, axis] = new_affine[axis, axis] / factor

    return resized.astype(np.float32), new_affine, {
        "input_rows": float(original_shape[0]),
        "input_cols": float(original_shape[1]),
        "input_slices": float(original_shape[2]),
        "resampled_rows": float(resized.shape[0]),
        "resampled_cols": float(resized.shape[1]),
        "resampled_slices": float(resized.shape[2]),
        "downsample_factor_x": round(float(factors[0]), 4),
        "downsample_factor_y": round(float(factors[1]), 4),
        "downsample_factor_z": round(float(factors[2]), 4),
    }


def convert_dicom_series_to_nifti(dicom_dir: str, output_dir: str) -> Tuple[str, Dict[str, float]]:
    datasets: List[pydicom.Dataset] = []
    for name in sorted(os.listdir(dicom_dir)):
        path = os.path.join(dicom_dir, name)
        try:
            ds = pydicom.dcmread(path, force=True)
            if hasattr(ds, "PixelData"):
                datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise RuntimeError("No readable DICOM slices were found for SynthSeg conversion.")

    def sort_key(ds: pydicom.Dataset) -> float:
        if hasattr(ds, "ImagePositionPatient"):
            try:
                return float(ds.ImagePositionPatient[2])
            except Exception:
                pass
        return float(getattr(ds, "InstanceNumber", 0) or 0)

    ordered = sorted(datasets, key=sort_key)
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

    if not slices:
        raise RuntimeError("No pixel data could be extracted from the uploaded DICOM series.")

    volume = np.stack(slices, axis=-1).astype(np.float32)
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
    affine = np.diag([sx, sy, sz, 1.0])

    volume, affine, resize_metrics = maybe_downsample_volume(volume, affine)

    nifti_path = os.path.join(output_dir, "series.nii.gz")
    nib.save(nib.Nifti1Image(volume, affine), nifti_path)
    return nifti_path, resize_metrics


def parse_volume_csv(volumes_csv: str) -> Dict[str, float]:
    if not os.path.exists(volumes_csv):
        return {}
    with open(volumes_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        try:
            row = next(reader)
        except StopIteration:
            return {}
    volumes: Dict[str, float] = {}
    for key, value in row.items():
        if key is None:
            continue
        if key.lower() in {"subject", "scan", "id", "filename", "file"}:
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if numeric > 0:
            volumes[key] = numeric
    return volumes


def build_report(metadata: Dict[str, str], segmentation_path: str, volumes_csv: str, resize_metrics: Dict[str, float]) -> Dict[str, Any]:
    volumes = parse_volume_csv(volumes_csv)
    image = nib.load(segmentation_path)
    seg = np.asarray(image.get_fdata())
    nonzero_labels = sorted(int(v) for v in np.unique(seg) if int(v) != 0)
    top_structures = sorted(volumes.items(), key=lambda item: item[1], reverse=True)[:8]
    top_names = [name for name, _ in top_structures]

    observations = [
        f"SynthSeg segmented {len(nonzero_labels)} non-background brain label(s) from the uploaded MR study.",
        f"Top segmented structures by volume: {', '.join(top_names)}." if top_names else "Volume table was not available for named structures.",
    ]
    if resize_metrics.get("downsample_factor_x", 1.0) < 0.999 or resize_metrics.get("downsample_factor_y", 1.0) < 0.999 or resize_metrics.get("downsample_factor_z", 1.0) < 0.999:
        observations.append(
            "Input volume was downsampled before SynthSeg inference to reduce container memory usage."
        )

    return {
        "analysis_type": "synthseg-brain-segmentation",
        "model_name": "synthseg-brain-mri-v1",
        "diagnostic_support": "anatomy-only",
        "diagnostic_available": False,
        "report_type": "non-diagnostic-brain-segmentation",
        "modality": metadata.get("Modality", "MR"),
        "body_part": metadata.get("BodyPartExamined") or metadata.get("StudyDescription") or "BRAIN",
        "anatomy_involved": "BRAIN",
        "abnormality_status": "anatomy-segmentation-completed",
        "confidence": None,
        "region_of_interest": "brain-volume",
        "observations": observations,
        "abnormalities": [],
        "impact": (
            "Whole-brain structure segmentation is available for review, but this output is not a pathology diagnosis."
        ),
        "conclusion": (
            "SynthSeg whole-brain MR segmentation completed successfully. This output provides anatomical context only."
        ),
        "recommendation": (
            "Use this result as brain structure context only and continue with radiologist review for final interpretation."
        ),
        "limitations": [
            "SynthSeg is a contrast-agnostic brain MRI segmentation model, not a final diagnosis engine.",
            "This route provides anatomical segmentation support, not validated pathology classification.",
        ],
        "metrics": {
            **resize_metrics,
            "segmented_label_count": float(len(nonzero_labels)),
            "segmentation_nonzero_voxels": float(np.count_nonzero(seg)),
            "top_segmented_structures_mm3": {name: round(value, 2) for name, value in top_structures},
        },
        "routing_decision": {},
        "support_matrix": {},
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "synthseg_home": SYNTHSEG_HOME, "model_version": SYNTHSEG_MODEL_VERSION})


@app.route("/upload", methods=["POST"])
def upload():
    temp_root: Optional[str] = None
    try:
        uploaded_files = request.files.getlist("files") or request.files.getlist("file")
        if not uploaded_files:
            return jsonify({"ok": False, "error": "No DICOM files uploaded"}), 400

        temp_root = tempfile.mkdtemp(prefix="synthseg-")
        dicom_dir = os.path.join(temp_root, "dicom")
        nifti_dir = os.path.join(temp_root, "nifti")
        os.makedirs(dicom_dir, exist_ok=True)
        os.makedirs(nifti_dir, exist_ok=True)

        first_file_bytes: Optional[bytes] = None
        for index, uploaded in enumerate(uploaded_files):
            file_bytes = uploaded.read()
            if not file_bytes:
                continue
            if first_file_bytes is None:
                first_file_bytes = file_bytes
            filename = uploaded.filename or f"instance-{index:04d}.dcm"
            if not filename.lower().endswith(".dcm"):
                filename = f"{filename}.dcm"
            with open(os.path.join(dicom_dir, filename), "wb") as handle:
                handle.write(file_bytes)

        if first_file_bytes is None:
            return jsonify({"ok": False, "error": "No readable DICOM payload was supplied"}), 400

        metadata = extract_metadata(first_file_bytes)
        nifti_path, resize_metrics = convert_dicom_series_to_nifti(dicom_dir, nifti_dir)
        seg_path = os.path.join(temp_root, "synthseg_seg.nii.gz")
        volumes_csv = os.path.join(temp_root, "synthseg_volumes.csv")
        run_synthseg(nifti_path, seg_path, volumes_csv)
        report = build_report(metadata, seg_path, volumes_csv, resize_metrics)

        return jsonify(
            {
                "ok": True,
                "model_id": report["model_name"],
                "dicom_metadata": metadata,
                "report": report,
                "summary": report["conclusion"],
            }
        )
    except Exception as exc:
        app.logger.exception("SynthSeg upload failed")
        return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 500
    finally:
        if temp_root and os.path.exists(temp_root):
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, threaded=True)


