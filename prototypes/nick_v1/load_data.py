import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import Stage

from tensorflow import keras

import dask.dataframe as dd

#TODO: add LoadTrainData and LoadTestData Classes, push to c as "train_data" and "test_data"
    
class DataLoaderStage(Stage):
    def __init__(self):
        super().__init__()
        self.setLoggingPrefix('DataLoaderStage: ')


class CSVReaderDataLoaderStage(DataLoaderStage):
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


class LoadDataFromMemoryDataLoaderStage(DataLoaderStage):
    def __init__(self, data):
        self.data = data
        super().__init__()
    
    def execute(self, dc):
        dc.set_item('data', data)
        return dc


class LoadFashionMNISTDataLoaderStage(DataLoaderStage):
    def __init__(self):
        self.input_dir = "../sample_data/mnist/"
        super().__init__()
        
    def execute(self, dc):
        self.logInfo("loading fashion_mnist dataset")
        fashion_mnist = keras.datasets.fashion_mnist
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
        # make validation set, scale vals to range from 0-1
        X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
        y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
        X_test = X_test / 255.0
        data = {
            'X_valid': X_valid,
            'X_train': X_train,
            'y_valid': y_valid,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
            }
        dc.set_item('fashion_mnist_data', data)
        return dc