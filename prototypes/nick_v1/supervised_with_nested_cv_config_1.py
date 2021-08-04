from load_data import CSVReader
from preprocessing import ImputeMissingVals, FeatureScaler, EncodeLabels
from pipeline import Pipeline
from cross_validation import GenerateCVFolds, NestedCrossValidationTrainingStage
from model_training import ModelTrainingStage
from evaluation_stage import EvaluationStage

from model import SklearnModel, TensorFlowModel
from training_context import SklearnSupervisedTrainParamGridContext, TensorFlowSupervisedTrainParamGridContext

from dask.distributed import Client
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras

# Use case 3: hyperparameter tuning and training



def my_sklearn_model_func(params):
    model = RandomForestClassifier()
    model.set_params(**params)
    return model

def my_tf_model_func(params):
    n = params['hidden_layer_size']
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(4,)))
    model.add(keras.layers.Dense(n, activation="relu"))
    model.add(keras.layers.Dense(3, activation="softmax"))
    return model


skm = SklearnModel()
skm.set_model_create_func(my_sklearn_model_func)

tfm = TensorFlowModel()
tfm.set_model_create_func(my_tf_model_func)


data_dir = '../../sample_data'
filename = 'iris_data_w_nans.csv'
s0 = CSVReader(data_dir, filename)

cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
categorical_cols = ['species']

s1 = EncodeLabels(categorical_cols, 'labelencoder')

s2 = GenerateCVFolds(k_folds=5, strategy='random', strategy_args={'seed':42}) 


# init training context for sklearn supervised model with nested CV params
train_context_skm = SklearnSupervisedTrainParamGridContext()
train_context_skm.model = skm
train_context_skm.feature_cols = cols
train_context_skm.ylabels = categorical_cols
train_context_skm.param_grid = {'n_estimators': [10,100,1000], 'max_depth': [1,10,None]}
train_context_skm.scoring_func = 'accuracy'

# init training context for tensorflow supervised model with nested CV params
train_context_tfm = TensorFlowSupervisedTrainParamGridContext()
train_context_tfm.model = tfm
train_context_tfm.feature_cols = cols
train_context_tfm.ylabels = categorical_cols
train_context_tfm.param_grid = {'hidden_layer_size': [3,4,5,6]}
train_context_tfm.scoring_func = 'accuracy'
train_context_tfm.optimizer = 'sgd'


s3 = NestedCrossValidationTrainingStage(train_context_skm)
#s3 = NestedCrossValidationTrainingStage(train_context_tfm)  # train one context per pipeline execution
s3.add_stage(ImputeMissingVals(cols, 'constant', fill_value=0))
s3.add_stage(FeatureScaler(cols, 'min-max'))
# s3.add_stage(TransformFeaturesStage(NGramTransformer(col='transcript'))) # example n-gram feature transformer stage

s4 = EvaluationStage(['auroc', 'accuracy', 'auprc'])


p = Pipeline()
p.add_stage(s1)
p.add_stage(s2)
p.add_stage(s3)
p.add_stage(s4)
p.run()







################ NOTES ##################
# class PredictionContext():
#   def __init__(self):
#    self._feature_cols = ...
#    self._models = ...
#   def setFeatureCols(feat_cols)
#   def validate():
#     return True/False
#
# class SupervisedPredictionContext():
#    self._labels = ...
#
# class TrainingContext(PredictionContext):
#   # adds cv_folds, cv_eval_func, 
#   def validate(isSupervised):