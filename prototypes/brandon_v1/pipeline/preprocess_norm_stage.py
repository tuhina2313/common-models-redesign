import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import Stage
from sklearn.preprocessing import StandardScaler

class PreprocessNormStage(Stage):
    def execute(self):
        scaler = StandardScaler(with_mean=True, with_std=True) # Z-norm
        self._outputData = self._inputData
        self._outputData.train_data = scaler.fit_transform(self._inputData.train_data)
        self._outputData.test_data = scaler.transform(self._inputData.test_data)
        return
