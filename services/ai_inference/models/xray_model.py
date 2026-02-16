import torch
import numpy as np
from monai.networks.nets import DenseNet121

from models.base_model import BaseModel


class XRayModel(BaseModel):
    """
    2D X-ray classification model
    CPU compatible
    """

    def __init__(self):

        self.device = torch.device("cpu")

        self.model = DenseNet121(
            spatial_dims=2,
            in_channels=1,
            out_channels=2
        ).to(self.device)

        self.model.eval()

    def predict(self, image: np.ndarray):

        if image.ndim != 2:
            raise Exception("Expected 2D image for X-ray model")

        image_tensor = torch.tensor(image, dtype=torch.float32)

        # Normalize
        image_tensor = (image_tensor - image_tensor.mean()) / (
            image_tensor.std() + 1e-5
        )

        # Add channel dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)

        prob_abnormal = probabilities[0][1].item()

        return {
            "model_name": "xray-2d-densenet-monai",
            "finding": "Abnormal X-ray features detected."
            if prob_abnormal > 0.5
            else "No significant abnormal X-ray features detected.",
            "abnormal": prob_abnormal > 0.5,
            "confidence": float(prob_abnormal),
        }
