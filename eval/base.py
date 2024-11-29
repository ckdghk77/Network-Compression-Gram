
class Evaluator() :
    def __init__(self,  gpu):
        self.gpu = gpu;

    def eval_acc(self, dloader, model):
        raise NotImplementedError

    def eval_feature(self, dloader, model):
        raise NotImplementedError

