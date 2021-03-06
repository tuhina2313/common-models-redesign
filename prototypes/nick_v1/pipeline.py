import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase
from data_container import DataContainer

class Pipeline(StageBase):
    def __init__(self):
        self._stages = []
        self._dc = DataContainer()
        super().__init__()

    def addStage(self, stage):
        if isinstance(stage, StageBase):
            self._stages.append(stage)
        else:
            self.logError("addStage() called with an object which is not derived from "+ type(StageBase))
        return

    def getDC(self):
        return self._dc

    def run(self):
        self._dc = self.execute(self._dc)
        return

    def execute(self, dc):
        for stage in self._stages:
            dc = stage.execute(dc)
        return dc

