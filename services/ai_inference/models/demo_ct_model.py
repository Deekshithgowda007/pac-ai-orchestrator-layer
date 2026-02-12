import numpy as np

class DemoCTModel:
    """
    Simple demo AI model:
    - Calculates mean pixel intensity
    - If above threshold → abnormal
    """

    def __init__(self):
        self.name = "demo-ct-model-v1"
        self.threshold = 100

    def predict(self, pixel_array: np.ndarray):
        mean_intensity = float(np.mean(pixel_array))

        if mean_intensity > self.threshold:
            finding = "Possible abnormal density detected"
            abnormal = True
            confidence = 0.82
        else:
            finding = "No obvious abnormality"
            abnormal = False
            confidence = 0.93

        return {
            "model_name": self.name,
            "finding": finding,
            "abnormal": abnormal,
            "confidence": confidence,
            "mean_intensity": mean_intensity
        }
