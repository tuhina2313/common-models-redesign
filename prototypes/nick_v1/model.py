import os
import sys
from abc import ABCMeta, abstractmethod
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'logging'))
from logger import Logger

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

	def set_model_create_fn(self, func):
		# check that func accepts a single arg for a param dict, else throw an error
		# if check passes, set local var to the callback function provided (func)
		pass

	@abstractmethod
	def get_params(self, deep=False):
		pass

	def set_params(self, params):
		self._params = params

	def finalize(self, **kwargs):
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

	def finalize(self, **kwargs):
		super().finalize()
		try:
			self._model.compile(kwargs) # check that required parameters exist
		except Exception as e:
			# log e
			raise RuntimeError("print something meaningful")

	def fit(self, X, y):
		super().fit(X,y)

	def predict(self, X):
		super().predict(X)