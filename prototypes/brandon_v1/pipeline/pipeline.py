import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase

class Pipeline(StageBase):
    def __init__(self):
        self._stages = []
        super().__init__()

    def addStage(self, stage):
        if isinstance(stage, StageBase):
            self._stages.append(stage)
        else:
            self.logError("addStage() called with an object which is not derived from "+type(StageBase))
        return

    def execute(self):
        data = self._inputData
        for stage in self._stages:
            stage.setInput(data)
            stage.execute()
            data = stage.getOutput()
        self._outputData = data
        return
