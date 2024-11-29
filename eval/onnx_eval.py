from eval.base import Evaluator
import onnxruntime
from tqdm import tqdm

def to_numpy(tensor):
    return tensor.numpy()
class OnnxEvaluator(Evaluator) :
    def __init__(self, gpu=False):
        super().__init__(gpu);

    def eval_acc(self, dloader, model):
        correct = 0;
        nums = 0;
        provider = "CUDAExecutionProvider" if self.gpu else "CPUExecutionProvider";
        ort_session = onnxruntime.InferenceSession(model,
                                                   providers=[provider])
        with tqdm(desc="Evaluate Acc.", total= len(dloader)) as pbar :
            for b_idx, (data, label) in enumerate(dloader) :
                if self.gpu :
                    data = data.cuda();

                ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
                ort_outs = ort_session.run(None, ort_inputs)
                yhat = ort_outs[0]
                correct += (yhat.argmax(1) == to_numpy(label)).sum()
                nums += len(label)
                pbar.update(1)

        return (correct/nums) * 100.0;

    def eval_feature(self, dloader, model):
        raise NotImplementedError
