from collections.abc import Iterable, Mapping

from dask.dataframe import dataframe as dd
from sklearn.metrics import get_scorer
from tensorflow.keras.losses import get as get_loss
from tensorflow.keras.optimizer import get as get_optimizer

from model import ModelBase, SklearnModel, TensorFlowModel


class TrainPredictContext():
	def __init__(self):
		self._feature_cols = None
		self._model = None

	def get_feature_cols(self):
		return self._feature_cols

	def set_feature_cols(self, cols):
		self._feature_cols = cols

	def get_model(self):
		return self._model

	def set_model(self, model:ModelBase):
		self._model = model

	def validate(self):
		# Column validity checks happen inside the training stage of the pipeline
		if not isinstance(self.feature_cols, Iterable):
			raise TypeError("feature_cols must be initialized to Iterable type")
		if self.model is None:
			raise RuntimeError("model must be initialized")
		return True

	feature_cols = property(get_feature_cols, set_feature_cols)
	model = property(get_model, set_model)


class SupervisedTrainPredictContext(TrainPredictContext):
	def __init__(self):
		super().__init__()
		self._ylabel = None

	def get_ylabel(self):
		return self._ylabel

	def set_ylabel(self, labels):
		if isinstance(labels, str) or isinstance(labels, Iterable):
			self._ylabel = labels
		else:
			raise ValueError('labels argument must be Iterable or string type')

	def validate(self):
		super().validate()
		if not isinstance(self.ylabel, str) or not isinstance(self.ylabel, Iterable):
			raise TypeError("ylabel must be initialized to Iterable or string type")
		return True

	ylabel = property(get_ylabel, set_ylabel)


class SupervisedTrainParamGridContext(SupervisedTrainPredictContext):
	def __init__(self):
		super().__init__()
		self._param_grid = None
		self._scoring_func = None

	def get_param_grid(self):
		return self._param_grid

	def set_param_grid(self, param_grid):
		if isinstance(param_grid, dict):
			self._param_grid = param_grid
		else:
			raise ValueError('param_grid argument must be dict type')

	def get_scoring_func(self):
		return self._scoring_func

	def set_scoring_func(self, scoring_func):
		if isinstance(scoring_func, str):
			# TODO: Utility function to check and return proper scoring function
			scoring_func = get_scorer(scoring_func)
			self._scoring_func = scoring_func
		elif callable(scoring_func):
			self._scoring_func = scoring_func
		else:
			raise ValueError('scoring_func argument must be dict type')

	def validate(self):
		super().validate()
		if not isinstance(self.param_grid, Mapping):
			raise TypeError('param_grid must be initialized to Mapping (dict) type')
		if not callable(self.scoring_func):
			raise RuntimeError('scoring_func must be initialized to a callable type')
		return True

	param_grid = property(get_param_grid, set_param_grid)
	scoring_func = property(get_scoring_func, set_scoring_func)


class SklearnSupervisedTrainParamGridContext(SupervisedTrainParamGridContext):
	def __init__(self):
		super().__init__()

	def validate(self):
		super().validate()
		if not issubclass(type(self.model), SklearnModel):
			raise TypeError("SklearnSupervisedTrainParamGridContext must take SklearnModel type as model arg")
		return True

class TensorFlowSupervisedTrainParamGridContext(SupervisedTrainPredictContext):
	def __init__(self):
		super().__init__()
		self._optimizer = None

	def get_optimizer(self):
		return self._optimizer

	def set_optimizer(self, optimizer):
		tf_optimizers = [
			'adadelta',
			'adagrad',
			'adam',
			'adamax',
			'nadam',
			'rmsprop',
			'sgd',
			'ftrl',
			'lossscaleoptimizer',
			'lossscaleoptimizerv1'
			]
		optimizer = optimizer.lower()
		if optimizer in tf_optimizers:
			self._optimizer = optimizer
		else:
			raise ValueError("optimizer: {} not supported by TensorFlow".format(optimizer))

	def validate(self):
		super().validate()
		if not issubclass(type(self.model), TensorFlowModel):
			raise TypeError("TensorFlowSupervisedTrainParamGridContext must take TensorFlowModel type as model arg")
		if not isinstance(self.optimizer, str):
			raise TypeError("TensorFlowSupervisedTrainParamGridContext optimizer must be string type arg")
		return True

	optimizer = property(get_optimizer, set_optimizer)



