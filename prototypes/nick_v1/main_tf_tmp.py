from load_data import LoadFashionMNIST
from preprocessing import ImputeMissingVals, FeatureScaler, EncodeLabels
from pipeline import Pipeline
from cross_validation import GenerateCVFolds, CrossValidationStage
from model_init import ModelInitializer, NNModelInitializer
from model_training import ModelTrainingStage, NNModelTrainingStage

from sklearn.ensemble import RandomForestClassifier

from dask.distributed import Client


import tensorflow as tf
from tensorflow import keras


def build_model_1():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28,28]))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
    return model


if __name__=='__main__':
    client = Client()
    
    try:
        # get fashion MNIST dataset
        s0 = LoadFashionMNIST()
        
        s1 = NNModelInitializer()
        s1.add_model(build_model_1)
        
        s2 = NNModelTrainingStage()
        
        p = Pipeline()
        p.addStage(s0)
        p.addStage(s1)
        p.addStage(s2)
        p.execute()
        
        # TESTING
        dc = p.getOutput()
        data = dc.get_item("fashion_mnist_data")
    except Exception as e:
        print(e)
        client.close()
    
    client.close()
