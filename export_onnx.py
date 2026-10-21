"""Export the 128-d MobileFaceNet backbone to ONNX for ESP-PPQ / ESP-DL.

Usage:
    python export_onnx.py                          # from BEST_checkpoint.tar
    python export_onnx.py --weights mobilefacenet.pt
    python export_onnx.py --weights mobilefacenet.pt --opset 13

Input is fixed at [1, 3, 112, 112]; output is the [1, 128] embedding
(the backbone's final BatchNorm2d output flattened). Keep the opset at
11 (default) or 13: ESP-PPQ's ONNX parser is happiest there, and this
graph only needs Conv / BatchNorm / Relu / Relu6(Clip) / Add / Reshape.
Remember the model expects the normalization chosen in config.py
(NORM_PRESET) -- your ESP-PPQ calibration dataloader must apply the
same preprocessing.
"""
import argparse

import torch

from facenet_utils import load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Export MobileFaceNet to ONNX')
    parser.add_argument('--weights', default='BEST_checkpoint.tar',
                        help='checkpoint (.tar) or bare state_dict (.pt)')
    parser.add_argument('--out', default='mobilefacenet.onnx')
    parser.add_argument('--opset', type=int, default=11)
    args = parser.parse_args()

    model = load_model_from_checkpoint(args.weights, torch.device('cpu'))
    dummy = torch.randn(1, 3, 112, 112)
    kwargs = dict(input_names=['input'],
                  output_names=['embedding'],
                  opset_version=args.opset,
                  dynamic_axes=None)  # fixed shape on purpose (ESP-DL)
    with torch.no_grad():
        try:
            # torch >= 2.5: default exporter is dynamo-based (needs
            # onnxscript and produces a different graph); ESP-PPQ wants
            # the classic TorchScript-based export.
            torch.onnx.export(model, dummy, args.out, dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, args.out, **kwargs)
    print('exported {} (opset {})'.format(args.out, args.opset))


if __name__ == '__main__':
    main()
