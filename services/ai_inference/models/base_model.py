from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class BaseModel(ABC):

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> Dict:
        pass