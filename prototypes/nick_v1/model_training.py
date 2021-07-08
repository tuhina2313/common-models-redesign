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

import pdb

class ModelTrainingStage(StageBase):
    def __init__(self, m_name, model, primary_key_col, train_X, train_y, test_X, test_y):
        self.m_name = m_name
        self.model = model
        self.primary_key_col = primary_key_col
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        super().__init__()
    
    def execute(self):
        dc = self._inputData
        primary_key = dc.get_item('primary_key')
        with joblib.parallel_backend('dask'):
            self.logInfo("fitting model {}".format(self.m_name))
            fitted_model = self.model.fit(self.train_X, self.train_y)
            y_preds = fitted_model.predict(self.test_X)
            y_preds = da.from_array(y_preds, chunks=1000)
            pk_col = self.primary_key_col.to_dask_array()
            #pdb.set_trace()
            
            # generate dask dataframe
            df = dd.concat([dd.from_dask_array(c) for c in [pk_col,y_preds]])#, axis = 1) 
            # name columns
            df.columns = [primary_key, 'y_preds']

            preds_key = self.m_name + '_predictions'
            dc_keys = dc.get_keys()
            if preds_key not in dc_keys:
                dc.set_item(preds_key, df)
                
            past_preds = dc.get_item(preds_key)
            model_preds = past_preds.concat(y_preds)
            dc.set_item(preds_key, model_preds)
        self._outputData = dc
        return
