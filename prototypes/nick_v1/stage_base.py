import os
import sys
from abc import ABC, abstractmethod
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'logging'))
from logger import Logger

class Stage(ABC):
    def __init__(self):
        self._loggingPrefix = type(self).__name__+": "
        self._enableCaching = False # TODO
        self._inputData = None
        self._outputData = None

    def setLoggingPrefix(self, prefix):
        self._loggingPrefix = prefix+type(self).__name__+": "
        return

    def logInfo(self, message):
        Logger.getInst().info(self._loggingPrefix+message)
        return

    def logError(self, message):
        Logger.getInst().error(self._loggingPrefix+message)
        return

    def setCached(self, enableCaching):
        self._enableCaching = False #enableCaching

    # This function receives an object (dc) of type DataContainer which should be used
    # for input and output
    @abstractmethod
    def execute(self, dc):
        pass
