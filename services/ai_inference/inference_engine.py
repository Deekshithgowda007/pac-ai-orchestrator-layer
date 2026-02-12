import os
import pydicom
from pydicom.errors import InvalidDicomError
from models.ct_abnormal_monai import CTAbnormalMONAI


class InferenceEngine:

    def __init__(self):

        self.models = {
            "1595cae9-83ca-4fad-9637-71a39e0e7289": CTAbnormalMONAI(),
            "00000000-0000-0000-0000-000000000001": CTAbnormalMONAI(),
            "ct-lung-model-v1": CTAbnormalMONAI(),
        }

    # ---------------------------------------------------
    def _resolve_dicom_file(self, dicom_path: str) -> str:

        if not dicom_path:
            raise Exception("dicom_path is empty")

        if not os.path.exists(dicom_path):
            raise Exception(f"Path does not exist: {dicom_path}")

        if os.path.isdir(dicom_path):
            dcm_files = [
                os.path.join(dicom_path, f)
                for f in os.listdir(dicom_path)
                if f.lower().endswith(".dcm")
            ]

            if not dcm_files:
                raise Exception(f"No DICOM files found in directory: {dicom_path}")

            selected_file = dcm_files[0]
            print(f"📂 Using DICOM file: {selected_file}")
            return selected_file

        return dicom_path

    # ---------------------------------------------------
    def run(self, model_id: str, dicom_path: str):

        if model_id not in self.models:
            raise Exception(f"Model {model_id} not registered")

        model = self.models[model_id]

        resolved_path = self._resolve_dicom_file(dicom_path)

        print(f"📄 Reading DICOM: {resolved_path}")

        try:
            ds = pydicom.dcmread(resolved_path, force=True)
        except InvalidDicomError:
            raise Exception("Invalid DICOM format")
        except Exception as e:
            raise Exception(f"Failed to read DICOM: {str(e)}")

        if "PixelData" not in ds:
            raise Exception("DICOM has no PixelData")

        pixel_array = ds.pixel_array

        print(f"🧠 Running model {model_id}")

        result = model.predict(ds, pixel_array)

        print(f"✅ Model completed: {result}")

        return result
