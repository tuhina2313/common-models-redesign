import copy

class SupervisedDataFrame(object):
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.train_y = None
        self.test_y = None
        self.model = None
        self.predictions = None
        self.evaluation = None

    def copy(self):
        return copy.copy(self)
