import os
import sys
from stage_base import Stage
from sklearn.model_selection import train_test_split
from dask_ml.model_selection import KFold

#in development

class DataPartitionStage(Stage):
    def __init__(self, data, folds):
        self.data = data
        self.folds = folds
        super().__init__()
        
    def _getfolds(self):
        
        