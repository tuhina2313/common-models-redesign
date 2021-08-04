from model import SklearnModel, TensorFlowModel


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

	def set_model(self, model):
		# TODO: check that model is SklearnModel or TensorFlowModel type
		self._model = model

	def validate(self):
		# TODO: check that cols exist in data
		# TODO: map column names -- or maybe somewhere else
		# if checks fail, return False
		return True

	feature_cols = property(get_feature_cols, set_feature_cols)
	model = property(get_model, set_model)


class SupervisedTrainPredictContext(TrainPredictContext):
	def __init__(self):
		super().__init__()
		self._ylabels = []

	def get_ylabels(self):
		return self._ylabels

	def set_ylabels(self, labels):
		if isinstance(labels, list):
			self._ylabels = labels
		else:
			raise ValueError('labels argument must be list type')

	def validate(self):
		super().validate()
		# TODO: check that ylabels exist in data
		# if checks fail, return False
		return True

	ylabels = property(get_ylabels, set_ylabels)


class SupervisedTrainParamGridContext(SupervisedTrainPredictContext):
	def __init__(self):
		super().__init__()
		self._param_grid = {}
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
			# TODO get scorer function w. str arg
			self._scoring_func = scoring_func
		elif callable(scoring_func):
			self._scoring_func = scoring_func
		else:
			raise ValueError('scoring_func argument must be dict type')

	def validate(self):
		super().validate()
		# TODO: check that parameters are valid
		# TODO: check that scoring func is valid
		# if checks fail, return False
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
		# TODO get optimizer w. str arg
		self._optimizer = optimizer

	def validate(self):
		super().validate()
		if not issubclass(type(self.model), TensorFlowModel):
			raise TypeError("TensorFlowSupervisedTrainParamGridContext must take TensorFlowModel type as model arg")
		if not isinstance(self.optimizer, str):
			raise TypeError("TensorFlowSupervisedTrainParamGridContext optimizer must be string type arg")
		return True

	optimizer = property(get_optimizer, set_optimizer)



