import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase

from dask_ml.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from dask_ml.impute import SimpleImputer
import pdb

class ImputeMissingVals(StageBase):
    def __init__(self, cols, strategy, fill_value=None):
        self.cols = cols
        self.strategy = strategy.lower()
        self.fill_value = fill_value
        super().__init__()

    def _get_imputer(self, strategy, fill_value):
        imputers = {
            'constant': SimpleImputer(strategy=strategy, fill_value=fill_value),
            'most_frequent': SimpleImputer(strategy=strategy),
            'mean': SimpleImputer(strategy=strategy),
            'median': SimpleImputer(strategy=strategy),
            }
        if strategy not in imputers.keys():
            self.logError("strategy arg must be one of {}".format(imputers.keys()))
        self.logInfo("Imputer strategy selected as {}".format(strategy))
        return imputers[strategy]

    def execute(self):
        dc = self._inputData
        imputer = self._get_imputer(self.strategy, self.fill_value)
        self.logInfo("Imputing missing values for columns: {}".format(self.cols))
        X = dc.get_item('data')
        cols_to_impute = X[self.cols]
        imputer = imputer.fit(cols_to_impute)
        imputed_cols = imputer.transform(cols_to_impute)
        imputed_cols.compute()
        X[self.cols] = imputed_cols
        dc.set_item('data', X)
        self._outputData = dc
        return


class FeatureScaler(StageBase):
    def __init__(self, cols, strategy, feature_range=(0,1)):
        self.cols = cols
        self.strategy = strategy.lower()
        self.feature_range = feature_range
        super().__init__()

    def _get_scaler(self, strategy, feature_range):
        scalers = {
            'min-max': MinMaxScaler(feature_range=feature_range),
            'standardize': StandardScaler()
            }
        if strategy not in scalers.keys():
            self.logError("strategy arg must be one of {}".format(scalers.keys()))
        self.logInfo("Scaler strategy selected as {}".format(strategy))
        return scalers[strategy]

    def execute(self):
        dc = self._inputData
        scaler = self._get_scaler(self.strategy, self.feature_range)
        self.logInfo("Scaling values for columns: {}".format(self.cols))
        X = dc.get_item('data')
        cols_to_scale = X[self.cols]
        scaler = scaler.fit(cols_to_scale)
        scaled_cols = scaler.transform(cols_to_scale)
        scaled_cols.compute()
        X[self.cols] = scaled_cols
        dc.set_item('data', X)
        self._outputData = dc
        return
    
class EncodeLabels(StageBase):
    def __init__(self, cols, strategy):
        self.cols = cols
        self.strategy = strategy.lower()
        super().__init__()        
        
    def _get_encoder(self, strategy):
        encoders = {
            'onehot': OneHotEncoder(),
            'labelencoder': LabelEncoder()
            }
        if strategy not in encoders.keys():
            self.logError("encoder arg must be one of {}".format(encoders.keys()))
        self.logInfo("Encoder strategy selected as {}".format(strategy))
        return encoders[strategy]
            
    def execute(self):
        dc = self._inputData
        self.logInfo("Encoding labels for columns: {}".format(self.cols))
        X = dc.get_item('data')
        cols_to_encode = X[self.cols].astype('category').categorize()
        #print (cols_to_encode)
        encoder = self._get_encoder(self.strategy)
        #pdb.set_trace()
        encoded_cols = encoder.fit_transform(cols_to_encode)
        pdb.set_trace()
        encoded_cols.compute()
        for col in self.cols:
            X[col + '_encoded'] = encoded_cols[col]
        X[self.cols] = encoded_cols
        dc.set_item('data', X)
        self._outputData = dc
        return