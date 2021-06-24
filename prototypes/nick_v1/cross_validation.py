import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase
from pipeline import Pipeline
from dask_ml.model_selection import KFold
import dask.dataframe 

# testing
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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




class CrossValidationStage(StageBase):
    def __init__(self, models_to_run, labels_to_predict):
        # TODO: make models_to_run generator function
        self.models_to_run = [(RandomForestClassifier(), ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])]
        self.labels_to_predict = ['species']
        self._pipeline = Pipeline()
        super().__init__()
        self._pipeline.setLoggingPrefix('CrossValidationStage: ')
        self.setLoggingPrefix('CrossValidationStage: ')

    def addStage(self, stage):
        self._pipeline.addStage(stage)

    def execute(self):
        dc = self._inputData
        splits = dc.get_item("cv_splits")
        for m in self.models_to_run:
            model = m[0]
            features = m[1] # what about nltk n-grams?
            for l in self.labels_to_predict:
                results = []
                predictions = []
                for s in splits:
                    train_idx, test_idx = s
                    cv_data = dc.get_item('data')
                    cv_data = cv_data.copy()
                    cv_data_X = cv_data[features]
                    cv_data_y = cv_data[l]
                    # TODO: FIX THIS
                    cv_data_train_X = cv_data_X.iloc[train_idx]
                    cv_data_train_y = cv_data_y.iloc[train_idx]
                    cv_data_test_X = cv_data_X.iloc[test_idx] 
                    cv_data_test_y = cv_data_y.iloc[test_idx] 
                    print('3')
                    # fit model
                    with joblib.parallel_backend('dask'):
                        fitted_model = model.fit(cv_data_train_X, cv_data_train_y)
                        y_preds = fitted_model.predict(cv_data_test_X)
                        y_preds.compute()
                        predictions.append(y_preds)
                        results.append(roc_auc_score(cv_data_test_y, y_preds))
            dc.set_item('model_' + str(self.models_to_run.index(m) + 1) + '_auroc', results)
            # write out predictions here
        self._outputData = dc
        return
