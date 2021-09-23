import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import Stage

from dask_ml.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from dask_ml.impute import SimpleImputer



class PreprocessingStage(Stage):
    def __init__(self):
        super().__init__()
        self.setLoggingPrefix('PreprocessingStage: ')


class ImputeMissingValsPreprocessingStage(PreprocessingStage):
    def __init__(self, cols, strategy, fill_value=None): 
        self.cols = cols
        self.strategy = strategy.lower()
        self.fill_value = fill_value
        self._fit_transform_data_idx = None
        self._transform_data_idx = None
        super().__init__()

    def _validate(self):
        # TODO: add validate function for basic type checking
        if self._fit_transform_data_idx is None:
            raise ValueError("Must provide _fit_transform_data_idx to fit imputer on in ImputeMissingValsPreprocessingStage stage")
        if self._transform_data_idx is None:
            raise ValueError("Must provide _transform_data_idx to impute in ImputeMissingValsPreprocessingStage stage")

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

    def set_fit_transform_data_idx(self, fit_transform_data_idx):
        self._fit_transform_data_idx = fit_transform_data_idx

    def set_transform_data_idx(self, transform_data_idx):
        self._transform_data_idx = transform_data_idx

    def execute(self, dc):
        imputer = self._get_imputer(self.strategy, self.fill_value)
        self.logInfo("Imputing missing values for columns: {}".format(self.cols))
        X = dc.get_item('data')
        data_to_impute = X[self.cols]
        fit_transform_data = data_to_impute.map_partitions(lambda x: x[x.index.isin(self._fit_transform_data_idx.compute())])
        imputer = imputer.fit(fit_transform_data)
        fit_transform_data = imputer.transform(fit_transform_data)
        transform_data = data_to_impute.map_partitions(lambda x: x[x.index.isin(self._transform_data_idx.compute())])
        transform_data = imputer.transform(transform_data)
        fit_transform_data.compute()
        transform_data.compute()
        X.loc[self._fit_transform_data_idx, self.cols] = fit_transform_data   # TODO: make this dask compatible
        X.loc[self._transform_data_idx, self.cols] = transform_data
        dc.set_item('data', X)
        return dc


class FeatureScalerPreprocessingStage(PreprocessingStage):
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

    def execute(self, dc):
        scaler = self._get_scaler(self.strategy, self.feature_range)
        self.logInfo("Scaling values for columns: {}".format(self.cols))
        X = dc.get_item('data')
        cols_to_scale = X[self.cols]
        scaler = scaler.fit(cols_to_scale)
        scaled_cols = scaler.transform(cols_to_scale)
        scaled_cols.compute()
        X[self.cols] = scaled_cols
        dc.set_item('data', X)
        return dc
    
    
    
class EncodeLabelsPreprocessingStage(PreprocessingStage):
    def __init__(self, cols, strategy):
        self.cols = cols
        self.strategy = strategy.lower()
        super().__init__()        
        
    def _get_encoder(self, strategy):
        #
        def _one_hot_encode(df, cols):
            for col in cols:
                encoder = OneHotEncoder()
                col_to_encode = df[[col]].astype('category').categorize()
                encoded_cols = encoder.fit_transform(col_to_encode)
                for c in encoded_cols:
                    df[c] = encoded_cols[c]
                df = df.drop(col, axis=1)
            return df
        
        def _label_encode(df, cols):
            for col in cols:
                encoder = LabelEncoder()
                encoded_col = encoder.fit_transform(df[col])
                df[col] = encoded_col
            return df
            
        encoders = {
            'onehotencoder': _one_hot_encode,
            'labelencoder': _label_encode
            }
        if strategy not in encoders.keys():
            self.logError("Encoder arg must be one of {}".format(encoders.keys()))
        self.logInfo("Encoder strategy selected as {}".format(strategy))
        return encoders[strategy]
            
    def execute(self, dc):
        self.logInfo("Encoding labels for columns: {}".format(self.cols))
        X = dc.get_item('data')
        encoder = self._get_encoder(self.strategy)
        X = encoder(X, self.cols)
        dc.set_item('data', X)
        return dc
