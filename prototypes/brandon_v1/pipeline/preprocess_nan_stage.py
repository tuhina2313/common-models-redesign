import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase


class PreprocessNanStage(StageBase):
    def execute(self):
        self._outputData = self._inputData
        nan_rows = (self._inputData.train_data.isna().sum(axis=1) > 0).values.flatten()
        nan_rows = np.logical_or(self._inputData.train_y.isna().values.flatten(), nan_rows)
        self._outputData.train_data = self._inputData.train_data.loc[~nan_rows,:]
        self._outputData.train_y = self._inputData.train_y[~nan_rows]
        return
