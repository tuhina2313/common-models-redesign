import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase

class EvaluationStage(StageBase):
    def __init__(self):
        super().__init__()
        self._training_context = None

    def setTrainingContext(self, trainingContext):
        self._training_context = trainingContext

    def _validate(self, dc):
        if not issubclass(self._training_context, SupervisedTrainPredEvalContext):
            raise ValueError("Training context must be a subclass of {}".format(type(SupervisedTrainPredEvalContext).__name__))
        if not isinstance(self._training_context.ylabel, str) and len(self._training_context.y_label) > 1:
            raise ValueError("Multi-target evaluation is not supported yet.  Ensure only one label is provided to {}".format(type(self).__name__))

    def execute(self, dc):
        self.logInfo("Running Model Evaluation Stage")
        data = dc.get_item('data')
        preds = dc.get_item('predictions')
        eval_func_names = self._training_context.get_eval_func_names()
        eval_results = {}

        y_label = self._training_context.y_label # TODO: support multi-target labels

        for i in range(len(self._training_context.eval_funcs)):
            eval_func = self._training_context.eval_funcs[i]
            eval_labels = data[y_label]
            eval_value = eval_func(eval_labels, preds)
            eval_results[eval_func_names[i]] = eval_value
        
        dc.set_item('evaluation_results', eval_results)
        return dc
