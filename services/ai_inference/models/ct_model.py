import torch
import numpy as np
from monai.networks.nets import DenseNet121
from monai.transforms import Resize

from models.base_model import BaseModel


class CTVolumeModel(BaseModel):
    """
    3D CT Volume Classification Model
    Uses MONAI DenseNet121 (3D)
    CPU compatible
    """

    def __init__(self, weights_path: str | None = None, device: str | None = None):
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)

        # 3D DenseNet
        self.model = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=2  # binary classification (normal / abnormal)
        ).to(self.device)

        if weights_path:
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("state_dict")
                    or checkpoint.get("model_state_dict")
                    or checkpoint.get("network_weights")
                    or checkpoint
                )
            else:
                state_dict = checkpoint
            model_state = self.model.state_dict()
            cleaned = {}
            for key, value in state_dict.items():
                resolved_key = key[7:] if isinstance(key, str) and key.startswith("module.") else key
                if resolved_key in model_state and tuple(model_state[resolved_key].shape) == tuple(value.shape):
                    cleaned[resolved_key] = value

            if not cleaned or (len(cleaned) / max(len(model_state), 1)) < 0.5:
                raise RuntimeError(
                    "Checkpoint is not compatible with the temporary CT DenseNet wrapper. "
                    "This MONAI bundle requires a bundle-specific adapter."
                )

            self.model.load_state_dict(cleaned, strict=False)

        self.model.eval()

        # Resize to manageable volume (CPU safe)
        self.resizer = Resize((64, 128, 128))  # Depth, Height, Width

    # ==========================================================
    # MAIN PREDICTION
    # ==========================================================
    def predict(self, volume: np.ndarray):

        if volume.ndim != 3:
            raise Exception("Expected 3D volume for CT model")

        # Convert to torch tensor
        volume_tensor = torch.tensor(volume, dtype=torch.float32)

        # Add channel dimension → (C, D, H, W)
        volume_tensor = volume_tensor.unsqueeze(0)

        # Resize for model stability
        volume_tensor = self.resizer(volume_tensor)

        # Add batch dimension → (B, C, D, H, W)
        volume_tensor = volume_tensor.unsqueeze(0)

        volume_tensor = volume_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(volume_tensor)
            probabilities = torch.softmax(logits, dim=1)

        prob_abnormal = probabilities[0][1].item()

        return self._format_result(prob_abnormal, volume)

    # ==========================================================
    # POST ANALYSIS
    # ==========================================================
    def _format_result(self, prob_abnormal: float, volume: np.ndarray):

        mean_hu = float(np.mean(volume))
        std_hu = float(np.std(volume))

        abnormal = prob_abnormal > 0.5

        finding = (
            "Abnormal CT features detected."
            if abnormal
            else "No significant abnormal features detected."
        )

        return {
            "model_name": "ct-3d-densenet-monai",
            "finding": finding,
            "abnormal": abnormal,
            "confidence": float(prob_abnormal),
            "density_analysis": {
                "mean_hu": mean_hu,
                "std_hu": std_hu
            }
        }
