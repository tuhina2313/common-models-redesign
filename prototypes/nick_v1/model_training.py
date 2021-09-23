 # -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:13:11 2021

@author: nickh
"""
import joblib
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da

from stage_base import Stage

from scikeras.wrappers import KerasClassifier

#TODO: make compatible with model.py
       #data container stores train and test data

#CLASS DESCRIPTION:
# trains on training data(in dc)
class ModelTrainingStage(Stage):
    def __init__(self, m_name, model, train_X, train_y, test_X, test_y):
        self.m_name = m_name
        self.model = model
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        super().__init__()
    
    def execute(self, dc):
        with joblib.parallel_backend('dask'):
            self.logInfo("fitting model {}".format(self.m_name))
            fitted_model = self.model.fit(self.train_X, self.train_y)
            y_preds = fitted_model.predict(self.test_X)

            preds_key = self.m_name + '_predictions'
            dc_keys = dc.get_keys()
            if preds_key not in dc_keys:
                dc.set_item(preds_key, np.array([]))
                
            past_preds = dc.get_item(preds_key)
            model_preds = np.append(past_preds, y_preds)
            dc.set_item(preds_key, model_preds)
        return dc



class NNModelTrainingStage(Stage):
    def __init__(self):
        self.m_name = 'NN_mnist_fashion'
        self.model = None
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.validation_data = None
        self.epochs = 30
        super().__init__()
        
    def execute(self):
        dc = self._inputData
        self.model = dc.get_item("models_to_run")
        self.model = self.model[0]
        data = dc.get_item("fashion_mnist_data")
        self.train_X = data["X_train"]
        self.train_y = data["y_train"]
        self.test_X = data["X_test"]
        self.test_y = data["y_test"]
        self.validation_data = (data["X_valid"], data["y_valid"])
        with joblib.parallel_backend('dask'):
            history = self.model.fit(self.train_X, self.train_y, epochs=self.epochs, validation_data=self.validation_data)
            y_pred = self.model.predict_classes(self.test_X)
        dc.set_item('y_preds', y_pred)
        self._outputData = dc