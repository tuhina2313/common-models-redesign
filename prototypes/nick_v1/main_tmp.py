from load_data import CSVReader
from preprocessing import ImputeMissingVals, FeatureScaler
from pipeline import Pipeline
from cross_validation import GenerateCVFolds

from dask.distributed import Client


if __name__=='__main__':
    client = Client(n_workers=4)
    
    try:
        # init dataloader stage
        data_dir = '../../sample_data'
        filename = 'iris_data_w_nans.csv'
        s1 = CSVReader(data_dir, filename)
        
        # init preprocessing stages
        cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        s2 = ImputeMissingVals(cols, 'constant', fill_value=0)
        s3 = FeatureScaler(cols, 'min-max')
        
        # cross-validation
        # TESTING
        #cv_strategy = CrossValidationStrategy(strategy='random', strategy_args=(train_test_splits=8, train_val_splits=3, groupby=['TeacherID']))
        #cv_strategy = CrossValidationStrategy(strategy='predefined_folds', strategy_args=(folds_path='../file.csv'))
        #cv_strategy = CrossValidationStrategy(strategy='stratified', strategy_args=(train_test_splits=8, train_val_splits=3, cols=['gender','income']))
        
        # Generate folds
        s4 = GenerateCVFolds(strategy='random', strategy_args=[1,2,3])
        
        
        #s4 = CrossValidationStage(train_test_splits=8, train_val_splits=3)
        
        
        p = Pipeline()
        p.addStage(s1)
        p.addStage(s2)
        p.addStage(s3)
        p.addStage(s4)
        p.execute()
    except:
        client.close()
    
    client.close()
