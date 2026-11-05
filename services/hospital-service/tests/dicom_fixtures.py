"""Synthetic DICOM fixtures for the upload tests. Generated pixels, no patient data."""
import hashlib
import io

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    DeflatedExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGLSLossless,
    RLELossless,
    generate_uid,
)

# name -> (transfer syntax, photometric interpretation)
FIXTURES = {
    "jpeg-ls-lossless": (JPEGLSLossless, "MONOCHROME2"),
    "jpeg-2000-lossless": (JPEG2000Lossless, "MONOCHROME2"),
    "rle-lossless": (RLELossless, "MONOCHROME2"),
    "deflated-explicit-le": (DeflatedExplicitVRLittleEndian, "MONOCHROME2"),
    "explicit-big-endian": (ExplicitVRBigEndian, "MONOCHROME2"),
    "monochrome1-explicit-le": (ExplicitVRLittleEndian, "MONOCHROME1"),
}
COMPRESSED = (JPEGLSLossless, JPEG2000Lossless, RLELossless)


def source_pixels(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:64, 0:64]
    # asymmetric gradient + noise inside 12 bits: a byte swap or an inversion changes the hash
    return ((x * 53 + y * 17) % 4032 + rng.integers(0, 64, (64, 64))).astype(np.uint16)


def pixel_sha256(ds) -> str:
    array = ds.pixel_array
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def build(name: str) -> tuple[bytes, np.ndarray]:
    """Return (file bytes, source array) for one fixture."""
    transfer_syntax, photometric = FIXTURES[name]
    pixels = source_pixels()
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientName = "SYNTHETIC^FIXTURE"
    ds.PatientID = "SYNTH-0001"
    ds.Modality = "CT"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.PixelRepresentation = 0
    save_kwargs = {"enforce_file_format": True}
    if transfer_syntax in COMPRESSED:
        ds.PixelData = pixels.tobytes()
        ds.compress(transfer_syntax)
    elif transfer_syntax == ExplicitVRBigEndian:
        ds.PixelData = pixels.astype(">u2").tobytes()
        ds.file_meta.TransferSyntaxUID = transfer_syntax
        save_kwargs.update(implicit_vr=False, little_endian=False)
    else:
        ds.PixelData = pixels.tobytes()
        ds.file_meta.TransferSyntaxUID = transfer_syntax
    buffer = io.BytesIO()
    ds.save_as(buffer, **save_kwargs)
    blob = buffer.getvalue()
    # A fixture that does not decode back to its source proves nothing downstream.
    check = pydicom.dcmread(io.BytesIO(blob))
    assert str(check.file_meta.TransferSyntaxUID) == str(transfer_syntax), name
    assert np.array_equal(check.pixel_array, pixels), f"fixture {name} is not lossless"
    return blob, pixels
