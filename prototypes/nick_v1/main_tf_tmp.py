from load_data import CSVReader
from preprocessing import ImputeMissingVals, FeatureScaler, EncodeLabels
from pipeline import Pipeline
from cross_validation import GenerateCVFolds, CrossValidationStage
from model_init import ModelInitializer 
from model_training import ModelTrainingStage
from evaluation_stage import EvaluationStage

from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow import keras


def build_model_fashion_mnist():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28,28]))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
    return model


def build_model_iris():
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(4,)))
    model.add(keras.layers.Dense(4, activation="relu"))
    model.add(keras.layers.Dense(3, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
    return model


if __name__=='__main__':
    client = Client()
    
    try:
        # init dataloader stage
        data_dir = '../../sample_data'
        filename = 'iris_data_w_nans.csv'
        s0 = CSVReader(data_dir, filename)
        
        # model initializer
        s1 = ModelInitializer()
        s1.add_model(
            {
            'model_build_function': build_model_iris,
            'feature_col_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
            'y_labels': ['species'],
            'backend' : 'tensorflow'
            }
        )

        # Column declaration stage? i.e. list of features, labels to predict

        # init preprocessing stages
        cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        s2 = ImputeMissingVals(cols, 'constant', fill_value=0)
        s3 = FeatureScaler(cols, 'min-max')
        categorical_cols = ['species']
        s4 = EncodeLabels(categorical_cols, 'labelencoder')

        # cross-validation
        # TESTING
        #cv_strategy = CrossValidationStrategy(strategy='random', strategy_args=(train_test_splits=8, train_val_splits=3, groupby=['TeacherID']))
        #cv_strategy = CrossValidationStrategy(strategy='predefined_folds', strategy_args=(folds_path='../file.csv'))
        #cv_strategy = CrossValidationStrategy(strategy='stratified', strategy_args=(train_test_splits=8, train_val_splits=3, cols=['gender','income']))

        # Generate folds
        s5 = GenerateCVFolds(strategy='random', strategy_args=[1,2,3])

        # Cross validation
        s6 = CrossValidationStage()

        s7 = EvaluationStage(method='accuracy')

        p = Pipeline()
        p.addStage(s0)
        p.addStage(s1)
        p.addStage(s2)
        p.addStage(s3)
        p.addStage(s4)
        p.addStage(s5)
        p.addStage(s6)
        p.addStage(s7)
        p.run()

        # TESTING
        dc = p.getDC()
        data = dc.get_item('data')
        data = data.compute()
        preds = dc.get_item('model_1_species_predictions')
        results = dc.get_item('model_1_species_evaluation')
        print("Results: " + str(results))
        
        #return dc
        
        #s1 = ModelInitializer()
        #s1.add_model(
        #    {
        #        "model_build_function": build_model_1,
        #        "backend": "tensorflow"
        #        }
        #    )
        
        #s2 = ModelTrainingStage()
        
        #p = Pipeline()
        #p.addStage(s0)
        #p.addStage(s1)
        #p.addStage(s2)
        #p.execute()
        
        # TESTING
        #dc = p.getOutput()
        #data = dc.get_item("fashion_mnist_data")
    except Exception as e:
        print(e)
        client.close()
    
    client.close()
