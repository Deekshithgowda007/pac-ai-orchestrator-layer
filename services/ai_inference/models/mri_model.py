import torch
import numpy as np
from monai.networks.nets import DenseNet121
from monai.transforms import Resize

from models.base_model import BaseModel


class MRIVolumeModel(BaseModel):

    def __init__(self):
        self.device = torch.device("cpu")

        self.model = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=2
        ).to(self.device)

        self.model.eval()

        self.resizer = Resize((64, 128, 128))

    def predict(self, volume: np.ndarray):

        if volume.ndim != 3:
            raise Exception("Expected 3D volume for MRI model")

        volume_tensor = torch.tensor(volume, dtype=torch.float32)
        volume_tensor = volume_tensor.unsqueeze(0)
        volume_tensor = self.resizer(volume_tensor)
        volume_tensor = volume_tensor.unsqueeze(0)
        volume_tensor = volume_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(volume_tensor)
            probabilities = torch.softmax(logits, dim=1)

        prob_abnormal = probabilities[0][1].item()

        return {
            "model_name": "mri-3d-densenet-monai",
            "finding": "Abnormal MRI features detected."
            if prob_abnormal > 0.5
            else "No significant abnormal MRI features detected.",
            "abnormal": prob_abnormal > 0.5,
            "confidence": float(prob_abnormal),
        }
