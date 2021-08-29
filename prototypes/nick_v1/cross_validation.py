import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase
from pipeline import Pipeline
from model_training import ModelTrainingStage
from preprocessing import PreprocessingStageBase
from load_data import LoadDataFromMemory
from training_context import SupervisedTrainParamGridContext
from evaluation_stage import EvaluationStage

from dask_ml.model_selection import KFold
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import itertools
from collections.abc import Iterable

#TODO: Add commented descriptions of each class above Class name

#class LoadTrainTestFoldStage(StageBase):
#    def __init__(self, train_idx, test_idx):
#        super().__init__()
#        self._train_idx = train_idx
#        self._test_idx = test_idx
#
#    def execute(self, dc):
#        self.logInfo("Loading training and test folds")
#        splits = [(self._train_idx, self._test_idx)]
#        dc.set_item('cv_splits', splits)
#        return dc

class GenerateCVFoldsStage(StageBase):
    def __init__(self, k_folds, strategy, strategy_args):
        super().__init__()
        self._k_folds = k_folds
        self._strategy = strategy.lower()
        self._strategy_args = strategy_args
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
        return [x for x in kf.split(data)]
    
    def execute(self, dc):
        self.logInfo("Generating CV Folds")
        X = dc.get_item('data')
        if self._strategy == 'random':
            splits = self._generate_splits(X.to_dask_array(lengths=True))
        elif self._strategy == 'stratified':
            # TODO: Stratified CV Folds
            raise RuntimeError("Stratified CV Folds not yet implemented")
        else:
            raise ValueError("{} is not a supported strategy for GenerateCVFoldsStage".format(self._strategy))
        dc.set_item('cv_splits', splits)
        return dc


# CLASS DESCRIPTION:
# Treats each subset in the data partition as a test set and remaining data as training set. 
# Uses cross validation to make predictions on each test set. 
# Does not tune hyperparameters.
class CrossValidationStage(StageBase):
    def __init__(self):
        super().__init__()
        self._preprocessing_stages = []
        self._training_context = None
        self.setLoggingPrefix('CrossValidationStage: ')

    def _validate(self):
        for stage in self._preprocessing_stages:
            if not issubclass(type(stage), PreprocessingStageBase):
                raise ValueError("addPreprocessingStage only accepts PreprocessingStageBase arg type")
        if not issubclass(type(self._training_context), SupervisedTrainParamGridContext):
            # TODO: check that param grid has a single point
            raise ValueError("setTrainingContext requires SupervisedTrainParamGridContext arg type")
        return
        
    def addPreprocessingStage(self, stage):
        self._preprocessing_stages.append(stage)

    def setTrainingContext(self, training_context):
        self._training_context = training_context

    def execute(self, dc):
        self._validate()
        self.logInfo("Starting Cross-Validation Stage")

        cols = [x for x in self._training_context.feature_cols]
        if isinstance(self._training_context.y_label, Iterable):
            cols = cols + [x for x in self._training_context.y_label]
        else:
            cols.append(self._training_context.y_label)

        cv_splits = dc.get_item("cv_splits")
        cv_results = {
            'predictions': (len(cv_splits[0][0] + len(cv_splits[0][1])))*[np.nan] 
            # TODO initialize prediction vector wrt to classification (int)/regression (float) arg provided by user
            }
        
        for i in range(len(cv_splits)):
            self.logInfo("Running CV for fold {}".format(i))
            split = cv_splits[i]
            train_idx, test_idx = split
            train_idx = train_idx.compute()
            test_idx = test_idx.compute()
            data = dc.get_item('data')
            data = data[cols]  # x and y cols

            data_train = data.map_partitions(lambda x: x[x.index.isin(train_idx)]) # TODO: can we move compute elsewhere to improve speed?
            data_test = data.map_partitions(lambda x: x[x.index.isin(test_idx)])

            # unpack training_context
            param_grid = self._training_context.param_grid
            for k in param_grid.keys():
                if isinstance(param_grid[k], Iterable):
                    item = [x for x in param_grid[k]][0]
                    param_grid[k] = item
            
            p = Pipeline()
            #self._pipeline.setLoggingPrefix('CrossValidationStage: ')
            p.addStage(LoadDataFromMemory(data))
            for stage in self._preprocessing_stages:
                stage.set_fit_transform_data_idx(train_idx)
                stage.set_transform_data_idx(test_idx)
                p.addStage(stage)
            #p.addStage(LoadTrainTestFoldStage(train_idx, test_idx))
            cv_stage_context = SupervisedTrainParamGridContext()
            cv_stage_context.param_grid = param_grid
            training_stage = ModelTrainingStage(train_idx)
            training_stage.setTrainingContext(cv_stage_context)
            p.addStage(training_stage)
            prediction_stage = ModelPredictionStage(test_idx)
            prediction_stage.setTrainingContext(cv_stage_context)
            p.addStage(prediction_stage)
            p.run()
            dc = p.getDC()
            preds = dc.get_item('predictions')
            cv_results['predictions'][test_idx] = preds

        dc['training_results'] = cv_results

        return dc


# TODO: This does not tune hyperparameters yet and shouldn't be used until it's implemented 
# Note: default to avg across parameter grid
class NestedCrossValidationTrainingStage(StageBase):
    def __init__(self):
        # TODO: make models_to_run generator function
        self._training_context = None
        self._validation_cv_stage = None
        self._preprocessing_stages = []
        super().__init__()
        self.setLoggingPrefix('NestedCrossValidationTrainingStage: ')

    def _validate(self):
        for stage in self._preprocessing_stages:
            if not issubclass(type(stage), PreprocessingStageBase):
                raise ValueError("addPreprocessingStage only accepts PreprocessingStageBase arg type")
        if not isinstance(self._validation_cv_stage, GenerateCVFoldsStage):
            raise ValueError("setValidationCVFoldsStage requires GenerateCVFoldsStage arg type")
        if not issubclass(type(self._training_context), SupervisedTrainParamGridContext):
            raise ValueError("setTrainingContext requires SupervisedTrainParamGridContext arg type")
        return

    def _createIterableParameterGrid(self, param_grid):
        param_vals = list(itertools.product(*[v for v in param_grid.values()]))
        param_keys = [k for k in param_grid.keys()]
        param_grid_list = [dict(zip(param_keys, param_vals[i])) for i in range(len(param_vals))]
        return param_grid_list


    def addPreprocessingStage(self, stage):
        self._preprocessing_stages.append(stage)

    def setValidationCVFoldsStage(self, stage):
        self._validation_cv_stage = stage

    def setTrainingContext(self, training_context):
        self._training_context = training_context
        
    def execute(self, dc):
        self._validate()
        
        cols = [x for x in self._training_context.feature_cols]
        if isinstance(self._training_context.y_label, Iterable):
            cols = cols + [x for x in self._training_context.y_label]
        else:
            cols.append(self._training_context.y_label)

        cv_splits = dc.get_item("cv_splits")
        nested_cv_results = {
            'results_per_fold':{},
            'results': {},
            'predictions': (len(cv_splits[0][0] + len(cv_splits[0][1])))*[np.nan]
            # TODO initialize prediction vector wrt to classification (int)/regression (float) arg provided by user
            }
        
        
        for i in range(len(cv_splits)):
            split = cv_splits[i]
            train_idx, test_idx = split
            data = dc.get_item('data')
            data = data[cols]  # x and y cols
            # map data to CV partitions
            data_train = data.map_partitions(lambda x: x[x.index.isin(train_idx.compute())]) # TODO: can we move compute elsewhere to improve speed?
            data_test = data.map_partitions(lambda x: x[x.index.isin(test_idx.compute())])

            # upack training_context
            param_grid = self._createIterableParameterGrid(self._training_context.param_grid)
            eval_func = self._training_context.param_eval_func
            eval_goal = self._training_context.param_eval_goal
            
            param_grid_eval_results = []
            for point in param_grid:
                p = Pipeline()
                p.addStage(LoadDataFromMemory(data_train))
                for stage in self._preprocessing_stages:
                    stage.set_fit_transform_data_idx(train_idx)
                    stage.set_transform_data_idx(test_idx)
                    p.addStage(stage)
                p.addStage(self._validation_cv_stage)
                cv_stage_context = SupervisedTrainParamGridContext()
                cv_stage_context.param_grid = point
                cv_stage = CrossValidationStage()
                cv_stage.setTrainingContext(cv_stage_context)
                p.addStage(cv_stage)
                p.addStage(EvaluationStage(eval_func))
                p.run()
                dc = p.getDC()
                eval_result = dc.get_item('model_evaluation_results')[eval_func]
                param_grid_eval_results.append((point, eval_result))  #TODO: update "point" as dictionary

            param_grid_eval_results.sort(key=lambda x: x[1], reverse=(eval_goal=='max'))
            best_point = param_grid_eval_results[0]
            # store results in dict - described below
            nested_cv_results['results_per_fold'][i]['param_grid_results'] = param_grid_eval_results
            nested_cv_results['results_per_fold'][i]['best_params'] = best_point
        
        # results_all_folds acts a temp object for computing averages
        results_all_folds = []
        for key, value in nested_cv_results['results_per_fold']:
            for p, p_val in value['param_grid_results']:
                all_ps = [x[0] for x in results_all_folds] # Get out all points that have been seen so far
                if p in all_ps:
                    p_index = all_ps.index(p)
                    results_all_folds[p_index][1].append(p_val) # save p_val to list of vals held by p
                else:
                    results_all_folds.append([p,[p_val]])
                    
        avg_results = []
        for p_results in results_all_folds:
            avg = np.mean(p_results[1])
            avg_results.append((p_results[0], avg))
        avg_results(key = lambda x: x[1], reverse=(eval_goal=='max'))
        nested_cv_results['results']['average_param_grid_results'] = avg_results
        nested_cv_results['results']['best_avg_params'] = avg_results[0]
        
        dc['training_results'] = nested_cv_results
        
        return dc

# dict
#   'results_per_fold'
#     '1'
#       'param_grid_results': [(param_point1, value), (param_point2, value)]
#       'best_params': (param_point, value)
#     '2'
#       ...
#     ...
#   'results'
#      'avg_param_grid_results': [(param_point1, avg_value), (param_point2, avg_value)]
#      'best_avg_params': (param_point, value)

#   'predictions': y_pred
