from collections.abc import Iterable, Mapping

from dask import dataframe as dd
from sklearn.metrics import get_scorer
from tensorflow.keras.losses import get as get_loss
#from tensorflow.keras.optimizer import get as get_optimizer

from model import ModelBase, SklearnModel, TensorFlowModel
from utils import get_sklearn_scoring_func, get_tensorflow_loss_func, get_tensorflow_metric_func

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
		self._y_label = None

    #move getter and setter for y labels to main class (but then what is unique about this class?)
	def get_y_label(self):
		return self._y_label

	def set_y_label(self, labels):
		if isinstance(labels, str) or isinstance(labels, Iterable):
			self._y_label = labels
		else:
			raise ValueError('labels argument must be Iterable or string type')

	def validate(self):
		super().validate()
		if not isinstance(self.y_label, str) or not isinstance(self.y_label, Iterable):
			raise TypeError("y_label must be initialized to Iterable or string type")
		return True

	y_label = property(get_y_label, set_y_label)


class SupervisedTrainParamGridContext(SupervisedTrainPredictContext):
    #split into two contexts (one has eval_funct and eval_goal)?
    #OR keep class and validate checks eval_func (do this)
	def __init__(self):
		super().__init__()
		self._param_grid = None
		self._param_eval_func = None
		self._param_eval_goal = None
		

	def get_param_grid(self):
		return self._param_grid

	def set_param_grid(self, param_grid):
		if isinstance(param_grid, dict):
			self._param_grid = param_grid
		else:
			raise ValueError('param_grid argument must be dict type')

	def get_param_eval_func(self):
		return self._param_eval_func

	def set_param_eval_func(self, param_eval_func):
		if callable(param_eval_func):
			self._param_eval_func = param_eval_func
		else:
			raise ValueError('param_eval_func argument must be callable type')

	def get_param_eval_goal(self):
		return self._param_eval_goal

	def set_param_eval_goal(self, param_eval_goal):
		param_eval_goal = param_eval_goal.lower()
		if param_eval_goal not in ['min', 'max']: # TODO: make enum object with min max types
			raise ValueError('param_eval_goal must be either min or max')
		self._param_eval_goal = param_eval_goal


	def validate(self):
		super().validate()
		if not isinstance(self.param_grid, Mapping):
			raise TypeError('param_grid must be initialized to Mapping (dict) type')
		if not callable(self.param_eval_func):
			raise RuntimeError('param_eval_func must be initialized to a callable type')
		if self.param_eval_goal is None:
			raise RuntimeError('param_eval_goal must be initialized to min or max') # TODO: Update after enum type
		return True

	param_grid = property(get_param_grid, set_param_grid)
	param_eval_func = property(get_param_eval_func, set_param_eval_func)
	param_eval_goal = property(get_param_eval_goal, set_param_eval_goal)


class SklearnSupervisedTrainParamGridContext(SupervisedTrainParamGridContext):
	def __init__(self):
		super().__init__()

	def get_param_eval_func(self):
		return self._param_eval_func

	def set_param_eval_func(self, param_eval_func):
		if isinstance(param_eval_func, str):
			self.param_eval_func = get_sklearn_scoring_func(param_eval_func)
		else:
			super().set_param_eval_func(param_eval_func)

	def validate(self):
		super().validate()
		if not issubclass(type(self.model), SklearnModel):
			raise TypeError("SklearnSupervisedTrainParamGridContext must take SklearnModel type as model arg")
		return True

	param_eval_func = property(get_param_eval_func, set_param_eval_func)


class TensorFlowSupervisedTrainParamGridContext(SupervisedTrainParamGridContext):
	def __init__(self):
		super().__init__()
		self._optimizer = None

	def get_param_eval_func(self):
		return self._param_eval_func

	def set_param_eval_func(self, param_eval_func):
		if isinstance(param_eval_func, str):
			self.param_eval_func = get_tensorflow_loss_func(param_eval_func)
		else:
			super().set_param_eval_func(param_eval_func)

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
	param_eval_func = property(get_param_eval_func, set_param_eval_func)


