from sklearn.metrics import get_scorer, SCORERS
from tensorflow.keras.losses import get as get_loss

# TODO: Utility function to check and return proper scoring function

def get_sklearn_scoring_func(scoring_str):
    if scoring_str in SCORERS.keys():   
        scorer = get_scorer(scoring_str)
        # BB - Accessing the private _score_func can lead to weird side-effects! Please be aware
        # this approach may not work for all scoring functions:
        # https://stackoverflow.com/questions/63943410/getting-a-scoring-function-by-name-in-scikit-learn
        scoring_ref = scorer._score_func
        if scoring_str.startswith('neg_'):
            scoring_ref = lambda x,y: -scoring_ref(x,y)
        return scoring_ref
    else:
        raise ValueError("Scoring func {} not supported by ScikitLearn".format(scoring_str))


def get_tensorflow_loss_func(loss_str):
    tensorflow_loss_strs = [
        'binary_crossentropy',
        'categorical_crossentropy',
        'categorical_hinge',
        'cosine_similarity',
        'deserialize',
        'hinge',
        'huber',
        'kl_divergence',
        'kld',
        'kullback_leibler_divergence',
        'log_cosh',
        'logcosh',
        'mae',
        'mape',
        'mean_absolute_error',
        'mean_absolute_percentage_error',
        'mean_squared_error',
        'mean_squared_logarithmic_error',
        'mse',
        'msle',
        'poisson',
        'sparse_categorical_crossentropy',
        'squared_hinge'
         ]

    if loss_str in tensorflow_loss_strs:
        loss_fn = get_loss(loss_str)
        return loss_fn
    else:
        raise ValueError('Loss function "{}" not supported by Tensorflow'.format(loss_str))
        
    
def get_tensorflow_metric_func(metric_str):
    tensorflow_metric_strs = [
        'binary_accuracy',
        'binary_crossentropy',
        'categorical_accuracy',
        'categorical_crossentropy',
        'hinge',
        'kl_divergence',
        'kld',
        'kullback_leibler_divergence',
        'log_cosh',
        'logcosh',
        'mae',
        'mape',
        'mean_absolute_error',
        'mean_absolute_percentage_error',
        'mean_squared_error',
        'mean_squared_logarithmic_error',
        'mse',
        'msle',
        'poisson',
        'sparse_categorical_accuracy',
        'sparse_categorical_crossentropy',
        'sparse_top_k_categorical_accuracy',
        'squared_hinge',
        'top_k_categorical_accuracy'
         ]

    if metric_str in tensorflow_metric_strs:
        metric_fn = get_loss(metric_str)
        return metric_fn
    else:
        raise ValueError('Metric function {} not supported by Tensorflow'.format(metric_str))
