import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase

from utils import get_sklearn_scoring_func as get_eval_func

class EvaluationStage(StageBase):
    def __init__(self, methods):
        self._methods = methods
        self._eval_funcs = []
        try:
            for method in methods:
                eval_func = get_eval_func(method)
                self._eval_funcs.append(eval_func)
        except:
            self.logError("Unable to map input method '%s' to an evaluation function"%(method))
            self._eval_funcs = None
        super().__init__()

    def execute(self, dc):
        self.logInfo("Running Model Evaluation Stage")
        data = dc.get_item('data')
        preds = dc.get_item('model_predictions')
        training_context = dc.get_item('training_context')
        eval_results = {}

        for i in range(len(self._methods)):
            method = self._methods[i]
            eval_func = self._eval_funcs[i]
            eval_value = eval_func(data[training_context.y_label], preds)
            eval_results[method] = eval_value
        
        dc.set_item('model_evaluation_results', eval_results)
        return dc
