import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer



class PreprocessingStageBase(StageBase):
    def __init__(self):
        super().__init__()
        self.setLoggingPrefix('PreprocessingStage: ')
        self._fit_transform_data_idx = None
        self._transform_data_idx = None

    def set_fit_transform_data_idx(self, fit_transform_data_idx):
        self._fit_transform_data_idx = fit_transform_data_idx

    def set_transform_data_idx(self, transform_data_idx):
        self._transform_data_idx = transform_data_idx

    def _validate(self):
        if self._fit_transform_data_idx is None:
            raise ValueError("set_fit_transform_data_idx must be called for preprocessing stages")
        if self._transform_data_idx is None:
            raise ValueError("set_transform_data_idx must be called for preprocessing stages")


class ImputerPreprocessingStage(PreprocessingStageBase):
    def __init__(self, cols, strategy, fill_value=None): 
        super().__init__()
        self._cols = cols
        self._strategy = strategy.lower()
        self._fill_value = fill_value

    def _validate(self):
        super()._validate()
        if self._get_imputer() is None:
            raise ValueError("Unknown strategy passed to {}".format(type(self).__name__))

    def _get_imputer(self):
        if self._strategy not in self._valid_imputers:
            return None
        else:
            return SimpleImputer(strategy=self._strategy, fill_value=self._fill_value)

    def execute(self, dc):
        self._validate()
        imputer = self._get_imputer()
        self.logInfo("Imputing missing values for columns: {}".format(self._cols))
        X = dc.get_item('data')
        data_to_impute = X[self._cols]
        fit_transform_data = data_to_impute.iloc[self._fit_transform_data_idx, :]
        imputer = imputer.fit(fit_transform_data)
        fit_transform_data = imputer.transform(fit_transform_data)
        transform_data = data_to_impute.iloc[self._transform_data_idx, :]
        transform_data = imputer.transform(transform_data)
        X.loc[self._fit_transform_data_idx, self._cols] = fit_transform_data
        X.loc[self._transform_data_idx, self._cols] = transform_data
        dc.set_item('data', X)
        return dc

ImputerPreprocessingStage._valid_imputers = ['mean', 'median', 'most_frequent', 'constant']


class FeatureScalerPreprocessingStage(PreprocessingStageBase):
    _scalers = {
        'min-max': lambda feat_range: MinMaxScaler(feature_range=feat_range),
        'standardize': lambda dummy: StandardScaler()
        }

    def __init__(self, cols, strategy, feature_range=(0,1)):
        super().__init__()
        self._cols = cols
        self._strategy = strategy.lower()
        self._feature_range = feature_range

    def _validate(self):
        super()._validate()
        if self._strategy not in self._scalers.keys():
            raise ValueError("Unknown strategy passed to {}; must be one of {}".format(type(self).__name__, self._scalers.keys()))

    def _get_scaler(self):
        self.logInfo("Scaler strategy selected as {}".format(self._strategy))
        return self._scalers[self._strategy](self._feature_range)

    def execute(self, dc):
        self._validate()
        scaler = self._get_scaler()
        self.logInfo("Scaling values for columns: {}".format(self._cols))
        X = dc.get_item('data')
        data_to_scale = X[self._cols]
        fit_transform_data = data_to_scale.iloc[self._fit_transform_data_idx, :]
        scaler = scaler.fit(fit_transform_data)
        fit_transform_data = scaler.transform(fit_transform_data)
        transform_data = data_to_scale.iloc[self._transform_data_idx, :]
        transform_data = scaler.transform(transform_data)
        X.loc[self._fit_transform_data_idx, self._cols] = fit_transform_data
        X.loc[self._transform_data_idx, self._cols] = transform_data
        dc.set_item('data', X)
        return dc
    
    
class EncodeLabels(PreprocessingStageBase):
    def __init__(self, cols, strategy):
        super().__init__()        
        self._cols = cols
        self._strategy = strategy.lower()

    def _validate(self):
        # Intentionally commented out.  No fit or transform indices are required
        #super()._validate()
        if self._strategy not in self._encoders.keys():
            raise ValueError("Unknown strategy passed to {}; must be one of {}".format(type(self).__name__, self._encoders.keys()))

    @classmethod
    def _one_hot_encode(cls, df, cols):
        for col in cols:
            encoder = OneHotEncoder()
            col_to_encode = df[[col]].astype('category').categorize()
            encoded_cols = encoder.fit_transform(col_to_encode)
            for c in encoded_cols:
                df[c] = encoded_cols[c]
            df = df.drop(col, axis=1)
        return df

    @classmethod
    def _label_encode(cls, df, cols):
        for col in cols:
            encoder = LabelEncoder()
            encoded_col = encoder.fit_transform(df[col])
            df[col] = encoded_col
        return df
        
    def _get_encoder(self):
        self.logInfo("Label encoder strategy selected as {}".format(self._strategy))
        return self._encoders[self._strategy]
            
    def execute(self, dc):
        self._validate()
        encoder = self._get_encoder()
        self.logInfo("Encoding labels for columns: {}".format(self._cols))
        X = dc.get_item('data')
        X = encoder(X, self._cols)
        dc.set_item('data', X)
        return dc

EncodeLabels._encoders = {
    'onehotencoder': EncodeLabels._one_hot_encode,
    'labelencoder': EncodeLabels._label_encode
}
