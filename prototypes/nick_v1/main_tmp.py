from load_data import DataLoader
from preprocessing import ImputeMissingVals, FeatureScaler
from pipeline import Pipeline

from dask.distributed import Client


if __name__=='__main__':
    client = Client(n_workers=4)
    
    # init dataloader stage
    data_dir = '../../sample_data'
    filename = 'iris_data_w_nans.csv'
    s1 = DataLoader(data_dir, filename)
    
    # init preprocessing stages
    cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    s2 = ImputeMissingVals(cols, 'constant', fill_value=0)
    s3 = FeatureScaler(cols, 'min-max')
    
    p = Pipeline()
    p.addStage(s1)
    p.addStage(s2)
    p.addStage(s3)
    p.execute()
    
    client.close()
