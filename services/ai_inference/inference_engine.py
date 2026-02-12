import os
import pydicom
from pydicom.errors import InvalidDicomError
from models.demo_ct_model import DemoCTModel


class InferenceEngine:

    def __init__(self):
        """
        Registry mapping:
        DB ai_model_id (UUID)  --->  actual model instance
        """

        self.models = {
            # UUID from routing_rules table
            "1595cae9-83ca-4fad-9637-71a39e0e7289": DemoCTModel(),

            # fallback / demo model
            "00000000-0000-0000-0000-000000000001": DemoCTModel(),

            # optional readable aliases (for testing)
            "demo-otsu": DemoCTModel(),
            "ct-lung-model-v1": DemoCTModel(),
        }

    # ---------------------------------------------------
    # Resolve actual DICOM file
    # ---------------------------------------------------
    def _resolve_dicom_file(self, dicom_path: str) -> str:
        """
        Handles both:
        - Folder path  → picks first .dcm file
        - File path    → returns directly
        """

        if not dicom_path:
            raise Exception("dicom_path is empty")

        if not os.path.exists(dicom_path):
            raise Exception(f"Path does not exist: {dicom_path}")

        # Case 1: Directory → select first .dcm file
        if os.path.isdir(dicom_path):
            dcm_files = [
                os.path.join(dicom_path, f)
                for f in os.listdir(dicom_path)
                if f.lower().endswith(".dcm")
            ]

            if not dcm_files:
                raise Exception(f"No DICOM files found in directory: {dicom_path}")

            selected_file = dcm_files[0]
            print(f"📂 Using DICOM file from folder: {selected_file}")

            return selected_file

        # Case 2: Direct file
        if os.path.isfile(dicom_path):
            print(f"📂 Using direct DICOM file: {dicom_path}")
            return dicom_path

        raise Exception(f"Invalid DICOM path: {dicom_path}")

    # ---------------------------------------------------
    # MAIN EXECUTION ENTRY
    # ---------------------------------------------------
    def run(self, model_id: str, dicom_path: str):

        if model_id not in self.models:
            raise Exception(f"Model {model_id} not registered")

        model = self.models[model_id]

        # Resolve folder → actual file
        resolved_path = self._resolve_dicom_file(dicom_path)

        print(f"📄 Reading DICOM: {resolved_path}")

        # ---------------------------------------------------
        # Load DICOM safely
        # ---------------------------------------------------
        try:
            # 🔥 IMPORTANT FIX
            ds = pydicom.dcmread(resolved_path, force=True)
        except InvalidDicomError:
            raise Exception("Invalid DICOM format (corrupted file)")
        except Exception as e:
            raise Exception(f"Failed to read DICOM: {str(e)}")

        # ---------------------------------------------------
        # Ensure pixel data exists
        # ---------------------------------------------------
        if "PixelData" not in ds:
            raise Exception("DICOM has no PixelData element")

        try:
            pixel_array = ds.pixel_array
        except Exception as e:
            raise Exception(f"Failed to decode pixel data: {str(e)}")

        print(f"🧠 Running model {model_id}")

        # ---------------------------------------------------
        # Run model prediction
        # ---------------------------------------------------
        result = model.predict(pixel_array)

        # ---------------------------------------------------
        # Structured response
        # ---------------------------------------------------
        structured_result = {
            "model_name": result.get("model_name", "DemoCTModel"),
            "finding": result.get("finding", "Unknown"),
            "abnormal": result.get("abnormal", False),
            "confidence": result.get("confidence", 0.0)
        }

        print(f"✅ Model completed: {structured_result}")

        return structured_result
