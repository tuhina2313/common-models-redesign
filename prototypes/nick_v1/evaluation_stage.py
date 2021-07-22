import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from stage_base import StageBase
from sklearn.metrics import get_scorer

class EvaluationStage(StageBase):
    def __init__(self, method, pred_suffix = '_predictions', eval_suffix = '_evaluation'):
        self._pred_suffix = pred_suffix
        self._eval_suffix = eval_suffix
        try:
            self._scorer = get_scorer(method)
        except:
            self.logError("Unable to map input method '%s' to an evaluation function"%(method))
            self._scorer = None
        super().__init__()

    def execute(self, dc):
        self.logInfo("Running Model Evaluation Stage")
        #model_and_labels = [(m[1]['model'], label_name) for m in dc.get_item('models_to_run') for label_name in m[1]['y_label']]
        model_and_labels = []
        models_to_run = dc.get_item('models_to_run')
        #model_names = [[k for k in x.keys()][0] for x in models_to_run]
        #print(model_names)
        for m in models_to_run:
            name = [k for k in m.keys()][0]
            params = m[name]['params']
            model = params['model']
            y_labels = params['y_labels']
            tmp = [(model,y) for y in y_labels]
            model_and_labels = model_and_labels + tmp
        model_and_labels = [(m[1]['model'], label_name) for m in dc.get_item('models_to_run') for label_name in m[1]['y_label']]
        data = dc.get_item('data')
        for model, label_name in model_and_labels:
            label = data[label_name]
            # Find all predictions by suffix which contain the label_name
            # TODO - find a better way to store the predictions so we don't have to do string matching
            pred_keys = [key for key in dc.get_keys() if label_name in key and key.endswith(self._pred_suffix)]
            for pred_key in pred_keys:
                # BB - Accessing the private _score_func can lead to weird side-effects! Please be aware
                # this approach may not work for all scoring functions:
                # https://stackoverflow.com/questions/63943410/getting-a-scoring-function-by-name-in-scikit-learn
                eval_value = self._scorer._score_func(dc.get_item(pred_key), label.compute())
                dc.set_item(pred_key[:-len(self._pred_suffix)]+self._eval_suffix, eval_value)
        return dc
