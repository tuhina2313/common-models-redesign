# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:55:30 2021

@author: nickh

pseudo-code for models-to-run to be used in CV & hyperparameter tuning
"""
from stage_base import StageBase

from sklearn.ensemble import RandomForestClassifier

from scikeras.wrappers import KerasClassifier

class ModelInitializer(StageBase):
    def __init__(self):
        self.models_to_run = []
        super().__init__()
        
    # def add_model(self, model = None, model_params = None, feature_col_names = None, y_label = None, scoring_func = None, model_build_function = None):
    def add_model(self, params):
        backend = params['backend']
        if backend == "sklearn":
            #todo - check argument logic
            self.logInfo("checking sklearn model arguments")
            

        elif backend == 'tensorflow':
            # todo - also check arguments
            model_build_function = params['model_build_function']
            model = KerasClassifier(build_fn=model_build_function) # TESTING
            params['model'] = model
            self.logInfo("checking tensorflow arguments")

        else:
            raise ValueError('This backend is not supported. Choose sklearn or tensorflow.')
        
        m = params
        i = len(self.models_to_run) + 1
        m_name = "model_{}".format(i)
        diception_mobject = {
            m_name: {
                "params": m,
                "output": {},
                }
            }
        self.models_to_run.append(diception_mobject)
        self.logInfo("appending model {} - {} to model training list".format(m_name, m))
        
    
    def get_models(self):
        return self.models_to_run
    
    def execute(self, dc):
        models = self.get_models()
        dc.set_item('models_to_run', models)
        return dc



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
