import torch
import numpy as np
import cv2
from monai.networks.nets import DenseNet121


class CTAbnormalMONAI:

    def __init__(self):

        self.model = DenseNet121(
            spatial_dims=2,
            in_channels=1,
            out_channels=2
        )

        torch.manual_seed(42)
        self.model.eval()

    # -------------------------------------
    # Convert to Hounsfield Units
    # -------------------------------------
    def to_hu(self, ds, pixel_array):
        intercept = getattr(ds, "RescaleIntercept", 0)
        slope = getattr(ds, "RescaleSlope", 1)
        return pixel_array * slope + intercept

    # -------------------------------------
    # Preprocessing
    # -------------------------------------
    def preprocess(self, image):

        image = np.clip(image, -1000, 400)
        image = (image + 1000) / 1400.0
        image = cv2.resize(image, (224, 224))

        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0)

        return image

    # -------------------------------------
    # Density analysis
    # -------------------------------------
    def analyze_density(self, hu_image):

        mean_hu = float(np.mean(hu_image))
        std_hu = float(np.std(hu_image))

        hyperdense_ratio = float(np.sum(hu_image > 100) / hu_image.size)
        hypodense_ratio = float(np.sum(hu_image < -800) / hu_image.size)

        description = []
        description.append(f"Mean attenuation: {round(mean_hu,1)} HU.")
        description.append(f"Standard deviation: {round(std_hu,1)} HU.")

        if hyperdense_ratio > 0.05:
            description.append(
                "Areas of increased density detected, which may represent calcification, hemorrhage, or contrast enhancement."
            )

        if hypodense_ratio > 0.20:
            description.append(
                "Extensive low attenuation regions noted, possibly representing air spaces or emphysematous changes."
            )

        if hyperdense_ratio <= 0.05 and hypodense_ratio <= 0.20:
            description.append(
                "No significant abnormal density distribution detected."
            )

        return " ".join(description)

    # -------------------------------------
    # Prediction
    # -------------------------------------
    def predict(self, ds, pixel_array):

        hu_image = self.to_hu(ds, pixel_array)
        input_tensor = self.preprocess(hu_image)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)

        abnormal_prob = float(probs[0][1].item())

        density_text = self.analyze_density(hu_image)

        finding_text = (
            "Imaging features suggest possible abnormality."
            if abnormal_prob > 0.5
            else "No acute abnormality suggested by neural network."
        )

        return {
            "model_name": "ct-monai-densenet-v1",
            "finding": finding_text,
            "density_analysis": density_text,
            "abnormal": abnormal_prob > 0.5,
            "confidence": round(abnormal_prob, 3)
        }
