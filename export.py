import os, torch
import argparse


def export(model, fout):

    torch_model = torch.load(model, map_location='cpu');
    export_path = fout

    torch_model.eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = torch_model(x);

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  export_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=19,          # the ONNX version to export the model to
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch-file', type=str, default="./pmodels/P(0.3_GN)_KD(3)_SEED(0) Step 2.pt")
    parser.add_argument('--out-file', type=str, default="model.onnx")
    args = parser.parse_args()

    export(args.torch_file, args.out_file);

