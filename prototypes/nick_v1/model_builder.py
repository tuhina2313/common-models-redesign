import os
import sys
from abc import ABCMeta
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'logging'))
from logger import Logger


class ModelBuilderInterface(metaclass=ABCMeta):
	def __init__(self):
		pass


class SklearnBuilder(ModelBuilderInterface):
	def __init__(self):
		pass


class TensorFlowBuilder(ModelBuilderInterface):
	def __init__(self):
		pass