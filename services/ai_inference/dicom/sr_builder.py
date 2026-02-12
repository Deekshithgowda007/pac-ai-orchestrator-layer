from datetime import datetime
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import (
    generate_uid,
    ExplicitVRLittleEndian,
    ComprehensiveSRStorage
)


def build_ai_sr(study_uid, patient_id, patient_name, result_text):
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = ComprehensiveSRStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        None,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128
    )

    # Patient
    ds.PatientID = patient_id
    ds.PatientName = patient_name

    # Study / Series
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    ds.Modality = "SR"
    ds.SeriesDescription = "AI Analysis Result"
    ds.Manufacturer = "PAC-AI"

    # Time
    now = datetime.now()
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")

    # Minimal SR text
    ds.ValueType = "TEXT"
    ds.TextValue = result_text

    return ds
