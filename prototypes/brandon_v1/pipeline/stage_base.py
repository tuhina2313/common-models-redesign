import os
import sys
from abc import ABC, abstractmethod
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'logging'))
from logger import Logger

class Stage(ABC):
    def __init__(self):
        self._loggingPrefix = type(self).__name__+": "
        self._enableCaching = False # TODO
        self._numJobs = 1
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

    def setCached(enableCaching):
        self._enableCaching = enableCaching

    def setParallelizable(numJobs=1):
        self._numJobs = numJobs

    def setInput(self, inputData):
        Logger.getInst().info(self._loggingPrefix+type(self).__name__+": Receiving input of type "+type(inputData).__name__)
        self._inputData = inputData
        return

    def getOutput(self):
        if not self._outputData:
            Logger.getInst().error(self._loggingPrefix+type(self).__name__+": getOutput() called but self._output was not set at the end of execute()")
        Logger.getInst().info(self._loggingPrefix+type(self).__name__+": Outputting data")
        #if self._enableCaching:
        #    mem.cache(self._output)
        return self._outputData

    # This function should use self._inputData and save the output to self._output
    @abstractmethod
    def execute(self):
        pass
