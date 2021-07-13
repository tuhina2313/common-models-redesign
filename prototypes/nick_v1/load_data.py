import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase

import dask.dataframe as dd

class CSVReader(StageBase):
    def __init__(self, data_dir, file_name):
        # TODO: Way to handle directory with multiple files
        self.data_path = os.path.join(data_dir, file_name)
        super().__init__()

    def execute(self, dc):
        self.logInfo("Loading data from: {}".format(self.data_path))
        if not os.path.exists(self.data_path):
            self.logError("File path does not exist: {}".format(self.data_path))
        dc.set_item('data filepath', self.data_path)
        df = dd.read_csv(self.data_path)
        dc.set_item('data', df)
        return dc
