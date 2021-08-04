import os
import sys
from abc import ABCMeta, abstractmethod
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'logging'))
from logger import Logger

from inspect import signature

from sklearn.base import BaseEstimator
from tensorflow.keras.models import Model


class ModelBase(metaclass=ABCMeta):
	def __init__(self):
		self._model = None
		self._model_create_fn = None
		self._params = None
		self._finalized = False

	def get_model(self):
		if self._finalized:
			return self._model
		else:
			raise RuntimeError("Models must be finalized prior to getting")

	def set_model_create_func(self, func):
		sig = signature(func)
		if len(sig.parameters) != 1:
			raise ValueError("model_create_fn must accept a single argument")
		param = [v for v in sig.parameters.values()][0]
		if not (param.kind == param.POSITIONAL_ONLY or param.kind == param.POSITIONAL_OR_KEYWORD):
			raise ValueError("model_create_fn must have similar prototype to `def func(params):`")
		if not (param.default is param.empty):
			raise ValueError("model_create_fn argument cannot have default value")
		self._model_create_fn = func

	@abstractmethod
	def get_params(self, deep=False):
		pass

	def set_params(self, params):
		self._params = params

	def finalize(self, params_dict):
		self._model = self._model_create_fn(self._params)
		self._finalized = True

	def fit(self, X, y):
		if not self._finalized:
			raise RuntimeError("Models must be finalized prior to fitting")

	def predict(self, X):
		if not self._finalized:
			raise RuntimeError("Models must be finalized prior to prediction")




class SklearnModel(ModelBase):
	def __init__(self):
		super().__init__()

	def get_params(self, deep=False):
		if self._finalized:
			return self._model.get_params(deep=deep)
		else:
			raise RuntimeError("Models must be finalized prior to getting") 

	def fit(self, X, y):
		super().fit(X,y)
		self._model.fit(X, y)

	def predict(self, X):
		super().predict(X)
		preds = self._model.predict(X)
		return preds





class TensorFlowModel(ModelBase):
	def __init__(self):
		super().__init__()

	def get_params(self, deep=False):
		pass

	def set_params(self, params):
		pass

	def finalize(self, params_dict):
		super().finalize()
		keys = params_dict.keys()
		if not 'loss' in keys:
			raise ValueError('TensorFlowModel requires `loss` parameter (e.g. "loss":"mse")')
		if not 'optimizer' in keys:
			raise ValueError('TensorFlowModel requires `optimizer` parameter (e.g. "optimizer":"sgd")')
		if not 'metrics' in keys:
			raise ValueError('TensorFlowModel requires `metrics` parameter (e.g. "metrics":["accuracy"])')
		try:
			# catch unexpected errors (e.g. typos)
			self._model.compile(**params_dict) 
		except Exception as e:
			raise RuntimeError("Failed to compile TensorFlowModel with exception: {}".format(str(e)))

	def fit(self, X, y):
		super().fit(X,y)

	def predict(self, X):
		super().predict(X)