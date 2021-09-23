import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import Stage
from sklearn.linear_model import LinearRegression

class TrainingStage(Stage):
    def execute(self):
        self._outputData = self._inputData
        model = LinearRegression()
        model.fit(self._inputData.train_data, self._inputData.train_y)
        self._outputData.model = model
        return
