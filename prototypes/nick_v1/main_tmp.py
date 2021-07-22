import os
import sys

from sklearn.ensemble import RandomForestClassifier

from dask.distributed import Client

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from load_data import CSVReader
from preprocessing import ImputeMissingVals, FeatureScaler, EncodeLabels
from pipeline import Pipeline
from cross_validation import GenerateCVFolds, CrossValidationStage
from model_init import ModelInitializer
from evaluation_stage import EvaluationStage

def RunTmp():
    try:
        # init dataloader stage
        data_dir = '../../sample_data'
        filename = 'iris_data_w_nans.csv'
        s0 = CSVReader(data_dir, filename)

        # model initializer
        s1 = ModelInitializer()
        s1.add_model(
            {
            'model': RandomForestClassifier(),
            'model_params': {'n_estimators': [100],},
            'feature_col_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
            'y_labels': ['species'],
            'scoring_func': 'roc_auc',
            'backend' : 'sklearn'
            }
        )
            'y_label': ['species'],
            'scoring_func': 'roc_auc',
            'backend' : 'sklearn'
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
        return dc
    except Exception as e:
        print(e)
    
    return 
        preds = dc.get_item('m_1_species_predictions')
        results = dc.get_item('m_1_species_evaluation')
        print("Results: " + str(results))
    except Exception as e:
        print(e)
    
    return

if __name__=='__main__':
    __spec__ = "None" # BB - for compatibility when running with pdb debugger
    client = Client(n_workers=4)
    dc = RunTmp()
    RunTmp()
    client.close()
