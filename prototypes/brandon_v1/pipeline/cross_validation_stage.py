import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import Stage
from pipeline import PipelineStage
from sklearn.model_selection import KFold

class CrossValidationStage(Stage):
    def __init__(self):
        self._pipeline = PipelineStage()
        super().__init__()
        self._pipeline.setLoggingPrefix('CrossValidationStage: ')
        self.setLoggingPrefix('CrossValidationStage: ')

    def addStage(self, stage):
        self._pipeline.addStage(stage)

    def execute(self):
        kf = KFold(n_splits=5, shuffle=True)
        results = []
        for train_idx, test_idx in kf.split(self._inputData.train_data):
            cv_data = self._inputData.copy() # TODO: Make a new one or copy?
            cv_data.train_data = self._inputData.train_data.iloc[train_idx]
            cv_data.train_y = self._inputData.train_y.iloc[train_idx]
            cv_data.test_data = self._inputData.train_data.iloc[test_idx]
            cv_data.test_y = self._inputData.train_y.iloc[test_idx]

            self._pipeline.setInput(cv_data)
            self._pipeline.execute()
            cv_output = self._pipeline.getOutput()

            results.append(cv_output)
 
        # TODO: This is where we should select optimal hyperparams
        cv_data = self._inputData.copy() # TODO: Make a new one or copy?
        cv_data.train_data = self._inputData.train_data
        cv_data.train_y = self._inputData.train_y
        cv_data.test_data = self._inputData.test_data
        cv_data.test_y = self._inputData.test_y

        self._pipeline.setInput(cv_data)
        self._pipeline.execute()
        self._outputData = self._pipeline.getOutput()
        return
