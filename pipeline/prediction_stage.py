import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase

class PredictionStage(StageBase):
    def execute(self):
        self._outputData = self._inputData
        self._outputData.predictions = self._inputData.model.predict(self._inputData.test_data)
        return
