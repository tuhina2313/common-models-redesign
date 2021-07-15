# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:55:30 2021

@author: nickh

pseudo-code for models-to-run to be used in CV & hyperparameter tuning
"""
from stage_base import StageBase

from sklearn.ensemble import RandomForestClassifier


class ModelInitializer(StageBase):
    def __init__(self):
        self.models_to_run = []
        super().__init__()
        
    def add_model(self, model, model_params, feature_col_names, y_label, scoring_func):
        m = {
            'model': model,
            'model_params': model_params,
            'feature_col_names': feature_col_names,
            'y_label': y_label,
            'scoring_func': scoring_func
            }
        i = len(self.models_to_run) + 1
        m_name = "m_{}".format(i)
        self.models_to_run.append((m_name, m))
        self.logInfo("appending model {} - {} to model training list".format(m_name, m))
    
    def get_models(self):
        return self.models_to_run
    
    def execute(self):
        dc = self._inputData
        models = self.get_models()
        dc.set_item('models_to_run', models)
        self._outputData = dc
        return


class NNModelInitializer(ModelInitializer):
    def __init__(self):
        super().__init__()
    
    def add_model(self, model_build_function):
        model = model_build_function()
        self.models_to_run.append(model)



# testing
# if __name__=='__main__':
#     models = ModelInitializer()
#     models.add_model(
#         model=RandomForestClassifier(),
#         model_params={'n_estimators': [100, 200],},
#         feature_col_names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
#         y_label='species',
#         scoring_func='roc_auc'
#         )