import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase
from sklearn.model_selection import train_test_split

class PartitionStage(StageBase):
    def execute(self):
        self._outputData = self._inputData
        self._outputData.train_data, self._outputData.test_data, self._outputData.train_y, self._outputData.test_y = train_test_split(self._inputData.train_data, self._inputData.train_y, test_size=0.2)
        return
