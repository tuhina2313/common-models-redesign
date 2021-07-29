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


class GenerateCVFolds(StageBase):
    def __init__(self, k_folds, strategy, strategy_args):
        self.k_folds = k_folds
        self.strategy = strategy.lower()
        self.strategy_args = strategy_args
        super().__init__()
    #    self._check_args()
        
    # def _check_args(self): # TODO
    #     # check that strategy and arguments are supported
    #     def _check_random_args(args):
    #         # TODO
    #         return True
        
    #     strategy_checks = {
    #         'random': _check_random_args(self.strategy_args),
    #         #'stratified': () TODO
    #         }
        
    #     if self.strategy not in strategy_checks.keys():
    #         raise ValueError("strategy must be one of: {}".format(strategy_checks.keys()))
    #     return strategy_checks[self.strategy](self.strategy_args)
        
    def _generate_splits(self, data):
        kf = KFold(n_splits=self.k_folds, shuffle=False, random_state=42)
        return kf.split(data)
    
    def execute(self, dc):
        self.logInfo("Generating CV Folds")
        X = dc.get_item('data')
        splits = self._generate_splits(X.to_dask_array(lengths=True))
        dc.set_item('cv_splits', splits)
        return dc




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

    def execute(self, dc):
        self.logInfo("Starting Cross-Validation Stage")
        splits = dc.get_item("cv_splits")
        self.models_to_run = dc.get_item('models_to_run')
        data = dc.get_item('data')
        for m in self.models_to_run:
            m_name = [k for k in m.keys()][0] 
            model_params = m[m_name]['params']
            model = model_params['model']   
            features = model_params['feature_col_names'] # what about nltk n-grams?
            labels_to_predict = model_params['y_labels'] 
            backend = model_params['backend']
            for l in labels_to_predict:
                self.logInfo("Starting CV for {} for label {}".format(m_name, l))
                l_name = l
                predictions = np.zeros((len(data.index),1)) # TODO - handle non numeric types and use Dask
                cv_counter = 0
                for s in splits:
                    cv_counter += 1
                    self.logInfo("Running CV for fold {}".format(cv_counter))
                    train_idx, test_idx = s
                    #data = data.copy()
                    data_X = data[features]
                    data_y = data[l]
                    # map data to CV partitions
                    data_train_X = data_X.map_partitions(lambda x: x[x.index.isin(train_idx.compute())])
                    data_train_y = data_y.map_partitions(lambda x: x[x.index.isin(train_idx.compute())])
                    data_test_X = data_X.map_partitions(lambda x: x[x.index.isin(test_idx.compute())])
                    data_test_y = data_y.map_partitions(lambda x: x[x.index.isin(test_idx.compute())])
                    # fit model
                    with joblib.parallel_backend('dask'):
                        if backend == 'sklearn':
                            fitted_model = model.fit(data_train_X, data_train_y)
                            y_preds = fitted_model.predict(data_test_X)
                            predictions[test_idx.compute(),0] = y_preds
                        elif backend == 'tensorflow':
                            history = self.model.fit(self.train_X, self.train_y, epochs=self.epochs)
                            y_pred = self.model.predict_classes(self.test_X)
                # write out predictions here
                dc.set_item(m_name + '_' + l_name + '_predictions', predictions)
        return dc



# TODO: This does not tune hyperparameters yet and shouldn't be used until it's implemented 
class NestedCrossValidationStage(StageBase):
    def __init__(self):
        # TODO: make models_to_run generator function
        self.models_to_run = None
        self._pipeline = Pipeline()
        super().__init__()
        self._pipeline.setLoggingPrefix('CrossValidationStage: ')
        self.setLoggingPrefix('CrossValidationStage: ')

    def addStage(self, stage):
        self._pipeline.addStage(stage)
        
    def execute(self, dc):
        splits = dc.get_item("cv_splits")
        self.models_to_run = dc.get_item('models_to_run')
        for m in self.models_to_run:
            model = m[1]['model']
            features = m[1]['feature_col_names'] # what about nltk n-grams?
            labels_to_predict = m[1]['y_labels']
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
                #dc.set_item(m_name + '_accuracy', results)
                dc.set_item(m_name + '_predictions', predictions)
                # self.addStage(ModelEvaluationStage(m_name))
        return dc
