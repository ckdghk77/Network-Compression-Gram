import torch
from eval.base import Evaluator
from tqdm import tqdm

class TorchEvaluator(Evaluator) :
    def __init__(self, gpu=False):
        super().__init__(gpu);

    @torch.no_grad()
    def eval_acc(self, dloader, model):
        correct = 0;
        nums = 0;

        with tqdm(desc="Evaluate Acc.", total= len(dloader)) as pbar :
            for b_idx, (data, label) in enumerate(dloader) :
                if self.gpu :
                    data = data.cuda();

                yhat = model(data)
                correct += (yhat.argmax(1).cpu() == label).sum()
                nums += len(label)
                pbar.update(1)

        return (correct.item()/nums) * 100.0;

    def eval_feature(self, dloader, model):
        raise NotImplementedError
