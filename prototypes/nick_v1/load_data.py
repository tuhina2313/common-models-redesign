import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.')) # what's this doing?
from stage_base import StageBase

import dask.dataframe as dd

class CSVReader(StageBase):
    def __init__(self, data_dir, file_name):
        # TODO: Way to handle directory with multiple files
        self.data_path = "{}/{}".format(data_dir.rstrip('/'), file_name)
        super().__init__()

    def execute(self):
        self.logInfo("Loading data from: {}".format(self.data_path))
        if not os.path.exists(self.data_path):
            self.logError("File path does not exist: {}".format(self.data_path))
        dc = DataContainer()
        dc.set_item('data filepath', self.data_path)
        tmp = dd.read_csv(self.data_path)
        dc.set_item('data', tmp)
        self._outputData = dc
        return



class DataContainer():
    def __init__(self):
        self._keystore = {}
    
    def get_item(self, key):
        if key in self.get_keys():
            return self._keystore[key]
        else:
            raise KeyError("provided key '{}' not present in {}".format(key, type(self).__name__))
    
    def set_item(self, key, obj):
        if type(key) != str:
            raise ValueError("provided key must be string type")
        self._keystore[key] = obj
    
    def get_keys(self):
        return self._keystore.keys()