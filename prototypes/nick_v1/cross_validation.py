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
        self._k_folds = k_folds
        self._strategy = strategy.lower()
        self._strategy_args = strategy_args
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
        kf = KFold(n_splits=self._k_folds, shuffle=False, random_state=self._strategy_args['seed'])
        return kf.split(data)
    
    def execute(self, dc):
        self.logInfo("Generating CV Folds")
        X = dc.get_item('data')
        if self._strategy == 'random':
            splits = self._generate_splits(X.to_dask_array(lengths=True))
        elif self._strategy == 'stratified':
            # TODO: Stratified CV Folds
            raise RuntimeError("Stratified CV Folds not yet implemented")
        else:
            raise ValueError("{} is not a supported strategy for GenerateCVFolds".format(self._strategy))
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
# Note: default to avg across parameter grid
class NestedCrossValidationTrainingStage(StageBase):
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




# TODO: This does not tune hyperparameters yet and shouldn't be used until it's implemented 
# Note: default to avg across parameter grid
class NestedCrossValidationTrainingStage(StageBase):
    def __init__(self, training_context):
        # TODO: make models_to_run generator function
        self.training_context = training_context
        self._pipeline = Pipeline()
        self._preprocessing_pipeline = Pipeline()
        super().__init__()

    def addStage(self, stage):
        self._pipeline.addStage(stage)

    def addPreprocessingStage(self, stage):
        # TODO: check that stage is valid preprocessing stage
        self._preprocessing_pipeline.addStage(stage)
        
    def execute(self, dc):
        features = self.training_context.feature_cols
        y_label = self.training_context.ylabel

        cv_splits = dc.get_item("cv_splits")
        for split in cv_splits:
            train_idx, test_idx = split
            data = dc.get_item('data')
            data_X = data[features]
            data_y = data[y_label]
            # map data to CV partitions
            data_train_X = data_X.map_partitions(lambda x: x[x.index.isin(train_idx.compute())])
            data_train_y = data_y.map_partitions(lambda x: x[x.index.isin(train_idx.compute())])
            data_test_X = data_X.map_partitions(lambda x: x[x.index.isin(test_idx.compute())])
            data_test_y = data_y.map_partitions(lambda x: x[x.index.isin(test_idx.compute())])
            # PSEUDO
            # run preprocessing pipeline on data_train_X
            # run preprocessing pipeline on data_test_X

            # maybe this is where we use the other pipeline
            # load data into pipeline DC
            # split data_train_X, data_train_Y into validation folds
            # for each fold assignment in val folds:
            #   for each hyperparameter combination (training_context.param_grid):
            #       train a model using hyperparameters (training_context.model)
            #       write down model results for those hyperparameters using (training_context.param_eval_func)
            # now use averaging or ranking to choose best hyperparameter combo (training_context.param_eval_goal)
            # exit pipeline and return best hyperparameters (?)

            # write down best hyperparameters
            # train final model on all training data using best hyperparameters
            # make predictions on data_test_X and write them down
            # do we want to check model performance per fold?
        return dc



# Validation folds? Where do we tell it how many? 
# Repeating preprocessing steps between each round of CV - best way to do this?
# Spelling conventions for methods: camelCase or under_scores ?