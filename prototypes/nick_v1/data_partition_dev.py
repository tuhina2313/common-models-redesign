import os
import sys
from stage_base import StageBase
from sklearn.model_selection import train_test_split

#in development

class DataPartitionStage(StageBase):
    def __init__(self, data, folds):
        self.data = data
        self.folds = folds
        super().__init__()
        
    def _getfolds(self):
        
        
