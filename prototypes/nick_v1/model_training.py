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

from stage_base import StageBase
class ModelPredictionStage(StageBase):
    def __init__(self, test_idx):
        super().__init__()
        self._test_idx = test_idx
        self._training_context = None

    def _validate(self, dc):
        if not issubclass(self._training_context, SupervisedTrainParamGridContext):
            raise ValueError("{} requires a context of type {}".format(type(self).__name__, type(SupervisedTrainingParamGridContext)))
        if min(self._test_idx) > 0 or max(self._test_idx) >= len(dc.get_item("data").index):
            raise ValueError("Test indices exceed bounds of the data size in {}".format(type(self).__name__))

    def setTrainingContext(self, training_context):
        self._training_context = training_context

    def execute(self, dc):
        self._validate(dc)
        trained_model = dc.get_item("trained_model")
        self.logInfo("Making predictions with model {}".format(type(trained_model).__name__))

        data = dc.get_item('data')
        data_test = data[self._training_context.feature_cols]
        data_test = data_test.map_partitions(lambda x: x[x.index.isin(self._test_idx)])

        with joblib.parallel_backend('dask'):
            predictions = trained_model.predict(data_test)

        dc.set_item('predictions', predictions)
        return dc

# trains on the input indices (rows) of the data stored in the dc
class ModelTrainingStage(StageBase):
    def __init__(self, train_idx):
        super().__init__()
        self._train_idx = train_idx
        self._training_context = None

    def _validate(self, dc):
        if not issubclass(self._training_context, SupervisedTrainParamGridContext):
            raise ValueError("{} requires a context of type {}".format(type(self).__name__, type(SupervisedTrainingParamGridContext).__name__))
        if min(self._train_idx) < 0 or max(self._train_idx) >= len(dc.get_item("data").index):
            raise ValueError("Training indices exceed bounds of the data size in {}".format(type(self).__name__))

    def setTrainingContext(self, training_context):
        self._training_context = training_context
    
    def execute(self, dc):
        self._validate(dc)
        self.logInfo("Starting model training stage with model {}".format(type(self._training_context.model).__name__))

        data = dc.get_item('data')
        data_train = data[self._training_context.feature_cols]
        label_train = data[self._training_context.ylabel]
        data_train = data_train.map_partitions(lambda x: x[x.index.isin(self._train_idx)])
        label_train = label_train.map_partitions(lambda y: y[y.index.isin(self._train_idx)])

        with joblib.parallel_backend('dask'):
            fitted_model = self._training_context.model.fit(data_train, label_train)

        dc.set_item('trained_model', fitted_model)
        return dc



class NNModelTrainingStage(StageBase):
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
