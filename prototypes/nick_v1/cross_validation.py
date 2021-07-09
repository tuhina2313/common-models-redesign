import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase
from pipeline import Pipeline
from model_training import ModelTrainingStage
from dask_ml.model_selection import KFold

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np

import pdb

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
        splits = self._generate_splits(X.to_dask_array(lengths=True))
        dc.set_item('cv_splits', splits)
        self._outputData = dc




class CrossValidationStage_dep(StageBase):
    def __init__(self):
        # TODO: make models_to_run generator function
        self.models_to_run = None
        self._pipeline = Pipeline()
        super().__init__()
        self._pipeline.setLoggingPrefix('CrossValidationStage: ')
        self.setLoggingPrefix('CrossValidationStage: ')

    def addStage(self, stage):
        self._pipeline.addStage(stage)

    def execute(self):
        dc = self._inputData
        splits = dc.get_item("cv_splits")
        self.models_to_run = dc.get_item('models_to_run')
        for m in self.models_to_run:
            m_name = m[0]
            model = m[1]['model']
            features = m[1]['feature_col_names'] # what about nltk n-grams?
            labels_to_predict = m[1]['y_label']
            for l in labels_to_predict:
                results = []
                predictions = np.array([])
                for s in splits:
                    train_idx, test_idx = s
                    cv_data = dc.get_item('data')
                    cv_data = cv_data.copy()
                    cv_data_X = cv_data[features]
                    cv_data_y = cv_data[l]
                    # map data to CV partitions
                    cv_data_train_X = cv_data_X.map_partitions(lambda x: x[x.index.isin(train_idx.compute())])
                    cv_data_train_y = cv_data_y.map_partitions(lambda x: x[x.index.isin(train_idx.compute())])
                    cv_data_test_X = cv_data_X.map_partitions(lambda x: x[x.index.isin(test_idx.compute())])
                    cv_data_test_y = cv_data_y.map_partitions(lambda x: x[x.index.isin(test_idx.compute())])
                    # fit model
                    with joblib.parallel_backend('dask'):
                        fitted_model = model.fit(cv_data_train_X, cv_data_train_y)
                        y_preds = fitted_model.predict(cv_data_test_X)
                        predictions = np.append(predictions, y_preds)
                        results.append(accuracy_score(cv_data_test_y, y_preds))
            dc.set_item(m_name + '_accuracy', results)
            dc.set_item(m_name + '_predictions', predictions)
            # write out predictions here
        self._outputData = dc
        return



class CrossValidationStage(StageBase):
    def __init__(self):
        # TODO: make models_to_run generator function
        self.models_to_run = None
        self._pipeline = Pipeline()
        super().__init__()
        self._pipeline.setLoggingPrefix('CrossValidationStage: ')
        self.setLoggingPrefix('CrossValidationStage: ')

    def addStage(self, stage):
        self._pipeline.addStage(stage)
        
    def execute(self):
        dc = self._inputData
        splits = dc.get_item("cv_splits")
        self.models_to_run = dc.get_item('models_to_run')
        for m in self.models_to_run:
            model = m[1]['model']
            features = m[1]['feature_col_names'] # what about nltk n-grams?
            labels_to_predict = m[1]['y_label']
            for l in labels_to_predict:
                m_name = m[0] + '_' + l
                for s in splits:
                    train_idx, test_idx = s
                    cv_data = dc.get_item('data')
                    cv_data = cv_data.copy()
                    cv_data_X = cv_data[features]
                    cv_data_y = cv_data[l]
                    # map data to CV partitions
                    cv_data_train_X = cv_data_X.map_partitions(lambda x: x[x.index.isin(train_idx.compute())])
                    cv_data_train_y = cv_data_y.map_partitions(lambda x: x[x.index.isin(train_idx.compute())])
                    cv_data_test_X = cv_data_X.map_partitions(lambda x: x[x.index.isin(test_idx.compute())])
                    cv_data_test_y = cv_data_y.map_partitions(lambda x: x[x.index.isin(test_idx.compute())])
                    mt_stage = ModelTrainingStage(m_name, model, s, cv_data_train_X, cv_data_train_y, cv_data_test_X, cv_data_test_y)
                    self.addStage(mt_stage)
                    # pass in s (cv splits) to model training stage for 
                    #  bookkeeping
                # self.addStage(ModelEvaluationStage(m_name))
        self._pipeline.setInput(dc)
        self._pipeline.execute()
        dc = self._pipeline.getOutput()
        self._outputData = dc
        return
