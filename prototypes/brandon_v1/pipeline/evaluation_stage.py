import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import Stage
from sklearn.metrics import mean_squared_error

class EvaluationStage(Stage):
    def execute(self):
        self._outputData = self._inputData
        self._outputData.evaluation = mean_squared_error(self._inputData.test_y, self._inputData.predictions)
        return
