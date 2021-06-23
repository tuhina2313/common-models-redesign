import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase
from pipeline import Pipeline
from dask_ml.model_selection import KFold



class GenerateCVFolds(StageBase):
    def __init__(self, strategy, strategy_args):
        self.strategy = strategy.lower()
        self.strategy_args = strategy_args
        super().__init__()
    #    self._check_args()
        
    # def _check_args(self): # TODO
    #     # check that strategy and arguments are supported
    #     def _check_random_args(args):
    #         # TODO
    #         return True
            
    #     def _check_predefined_folds_args(args):
    #         # TODO
    #         return True
        
    #     strategy_checks = {
    #         'random': _check_random_args(self.strategy_args),
    #         'predefined_folds': _check_predefined_folds_args(self.strategy_args),
    #         #'stratified': () TODO
    #         }
        
    #     if self.strategy not in strategy_checks.keys():
    #         raise ValueError("strategy must be one of: {}".format(strategy_checks.keys()))
    #     return strategy_checks[self.strategy](self.strategy_args)
        
    def _generate_splits(self, data):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        return kf.split(data)
    
    def execute(self):
        dc = self._inputData
        X = dc.get_item('data')
        splits = self._generate_splits(X)
        dc.set_item('cv_splits', splits)
        self._outputData = dc


class CrossValidationStage(StageBase):
    def __init__(self, train_test_splits, train_val_splits, ):
        self._pipeline = Pipeline()
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
