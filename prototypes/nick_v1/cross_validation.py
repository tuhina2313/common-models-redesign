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




class CrossValidationStage(StageBase): # TODO update this
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
    def __init__(self):
        # TODO: make models_to_run generator function
        self._training_context = None
        self._validation_cv_stage = None
        self._preprocessing_stages = []
        super().__init__()
        self.setLoggingPrefix('NestedCrossValidationTrainingStage: ')

    def _validate(self):
        for stage in self._preprocessing_stages:
            if not issubclass(stage, PreprocessingStageBase):
                raise ValueError("addPreprocessingStage only accepts PreprocessingStageBase arg type")
        if not isinstance(self._validation_cv_stage, GenerateCVFolds):
            raise ValueError("setValidationCVFoldsStage requires GenerateCVFolds arg type")
        if not issubclass(self._training_context, SupervisedTrainParamGridContext):
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
        
        cv_splits = dc.get_item("cv_splits")
        nested_cv_results = {'results_per_fold':{},
                             'results': {},
                             'predictions': (len(cv_splits[0][0] + len(cv_splits[0][1])))*[np.nan]
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
            eval_goal = self._training_context.eval_goal
            
            param_grid_eval_results = []
            for point in param_grid:
                p = Pipeline()
                p.addStage(LoadDataFromMemory(data_train))
                for stage in self._preprocessing_stages:
                    p.addStage(stage)
                p.addStage(self._validation_cv_stage)
                cv_stage_context = SupervisedTrainParamGridContext()
                cv_stage_context.param_grid = point
                cv_stage = CrossValidationStage(cv_stage_context)
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
        
        results_all_folds = [] # essentially mimicking a dictionary without being one
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
        
        dc['nested_cv_results'] = nested_cv_results
        
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