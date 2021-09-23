import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))
from preprocess_nan_stage import PreprocessNanStage
from preprocess_norm_stage import PreprocessNormStage
from training_stage import TrainingStage
from partition_stage import PartitionStage
from prediction_stage import PredictionStage
from evaluation_stage import EvaluationStage
from cross_validation_stage import CrossValidationStage
from pipeline import PipelineStage
sys.path.append(os.path.join(os.path.dirname(__file__), 'logging'))
from logger import Logger
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from supervised_data_frame import SupervisedDataFrame
import pandas as pd
import numpy as np

# TODO: Consider using these tools:
# joblib, for parallelism and speed
# Dask, for scaling ML jobs to multiple machines

def DoRunPipelineTest():
    p = PipelineStage()

    p.addStage(PreprocessNanStage())

    p.addStage(PartitionStage())

    cv = CrossValidationStage()
    cv.addStage(PreprocessNormStage())
    cv.addStage(PartitionStage())
    cv.addStage(TrainingStage())
    cv.addStage(PredictionStage())
    cv.addStage(EvaluationStage())
    p.addStage(cv)

    p.addStage(PredictionStage())

    p.addStage(EvaluationStage())

    # Dummy input
    sdf = SupervisedDataFrame()
    sdf.train_data = pd.DataFrame(data={'id': list(range(10))+[np.nan], 'group': 5*[1]+5*[2]+[np.nan]})
    sdf.train_y = pd.DataFrame(data={'value': np.array(range(10,21))+np.random.uniform(-1,1,11)})

    p.setInput(sdf)
    p.execute()
    out_df = p.getOutput()

    Logger.getInst().info("===Pipeline has finished processing===")
    Logger.getInst().info('Evaluation metric (MSE): '+str(out_df.evaluation))
    return

if __name__ == '__main__':
    DoRunPipelineTest()
