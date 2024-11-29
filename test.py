import onnx
from onnx import numpy_helper
from eval.onnx_eval import OnnxEvaluator
import argparse
from dl.cifar100 import cifar100
import timeit


def Count_Params(args):
    onnx_model = onnx.load(args.onnx_file)
    def count_parameters(model):
        total_parameters = 0
        for initializer in model.graph.initializer:
            total_parameters += numpy_helper.to_array(initializer).size
        return total_parameters

    num_params = count_parameters(onnx_model)
    return num_params

def Eval_Latency(args):
    nruns=100
    ort_session = '''
import onnxruntime
import numpy as np
ort_session=onnxruntime.InferenceSession(\"{}\", providers=['CPUExecutionProvider'])
np_x= np.random.randn(1,3,224,224).astype(np.float32)   
'''.format(args.onnx_file)
    elapsed_time = timeit.timeit(stmt='''ort_session.run(None,{"input": np_x},)''',
                  setup=ort_session,number=nruns)
    return elapsed_time/nruns

def Eval_Acc(args, test_loader):
    oeval = OnnxEvaluator();
    acc = oeval.eval_acc(test_loader, args.onnx_file)

    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="./data")
    parser.add_argument('--onnx-file', type=str, default="./omodels/P(0.4_L1)_KD(0)_SEED(0).onnx")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)

    args = parser.parse_args()

    c100 = cifar100(data_dir=args.data_dir);
    test_loader = c100.load_cifar100(is_train=False,
                                     batch_size=args.batch_size, num_workers=args.num_workers);

    print("Testing {} on CIFAR100 Test dataset".format(args.onnx_file))

    ########
    # 1. Number of Parameters
    ########
    num_params = Count_Params(args)
    print("Params: {:.2f} M".format(num_params / 1e6))

    ########
    # 2. Latency
    ########
    latency = Eval_Latency(args)
    print("Latency: {:.2f} ms".format(1000 * latency))

    ########
    # 3. Accuracy
    ########
    acc = Eval_Acc(args, test_loader)
    print("Accuracy: {:.2f} %".format(acc))
