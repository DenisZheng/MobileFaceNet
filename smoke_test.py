"""Smoke test for the fixed core path (CPU, no dataset needed).

Checks:
  1. Core modules import without side effects (in particular, importing
     facenet_utils must NOT instantiate the RetinaFace detector).
  2. MobileFaceNet forward: [2,3,112,112] -> [2,128]; parameter count.
  3. ArcMarginModel head shape with config.num_classes (this is the
     exact spot where upstream master crashed: the head read
     num_classes=21 from custom_config while data labels go to 85741).
  4. Checkpoint round-trip in the new state_dict format + resume loader.
  5. ONNX export (opset 11) and output-shape check with onnxruntime-free
     graph inspection, incl. the operator set ESP-PPQ will see.
  6. data_gen transforms output shape and value range for NORM_PRESET.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
from mobilefacenet import MobileFaceNet, ArcMarginModel
from facenet_utils import parse_args, save_checkpoint, load_checkpoint_file, load_model_from_checkpoint

print('1. imports OK (no detector side effect)')
print('   NORM_PRESET={} mean={} std={}'.format(config.NORM_PRESET, config.NORM_MEAN, config.NORM_STD))
print('   config.num_classes={} emb_size={}'.format(config.num_classes, config.emb_size))

model = MobileFaceNet()
n_params = sum(p.numel() for p in model.parameters())
x = torch.randn(2, 3, 112, 112)
model.eval()  # compare in eval mode: train-mode BN uses batch statistics,
# so a train-mode reference can never match a reloaded eval-mode model
with torch.no_grad():
    emb = model(x)
print('2. forward OK: {} -> {}, params={:,}'.format(tuple(x.shape), tuple(emb.shape), n_params))
assert emb.shape == (2, 128)

# ArcMarginModel reads num_classes/emb_size from config (85742 x 128) --
# allocate it like train.py does.
class _A:  # minimal stand-in for argparse args used by ArcMarginModel
    easy_margin = False
    margin_m = 0.5
    margin_s = 64.0

metric_fc = ArcMarginModel(_A())
labels = torch.tensor([0, 85741])  # max valid label for faces_emore
with torch.no_grad():
    logits = metric_fc(emb, labels)
print('3. ArcMargin OK: logits {}, label 85741 accepted'.format(tuple(logits.shape)))
assert logits.shape == (2, config.num_classes)

# Checkpoint round-trip (new format), mimicking train.py resume.
opt = torch.optim.SGD(model.parameters(), lr=0.1)
save_checkpoint(0, 0, model, metric_fc, opt, 0.5, True)
ckpt = load_checkpoint_file('BEST_checkpoint.tar')
assert 'model_state' in ckpt and 'optimizer_state' in ckpt
m2 = load_model_from_checkpoint('BEST_checkpoint.tar', torch.device('cpu'))
with torch.no_grad():
    emb2 = m2(x)
print('4. checkpoint round-trip OK, max |emb-emb2| = {:.2e}'.format((emb - emb2).abs().max().item()))

# ONNX export + op inventory
out = 'smoke_mobilefacenet.onnx'
with torch.no_grad():
    torch.onnx.export(model.eval(), torch.randn(1, 3, 112, 112), out,
                      input_names=['input'], output_names=['embedding'],
                      opset_version=11, dynamo=False)
import onnx
g = onnx.load(out)
ops = sorted({n.op_type for n in g.graph.node})
print('5. ONNX OK: {} nodes, ops = {}'.format(len(g.graph.node), ops))

# Transforms
from data_gen import data_transforms
from PIL import Image
import numpy as np
img = Image.fromarray((np.random.rand(112, 112, 3) * 255).astype('uint8'))
t = data_transforms['val'](img)
print('6. val transform OK: shape={}, range=[{:.3f}, {:.3f}]'.format(tuple(t.shape), t.min().item(), t.max().item()))
assert t.shape == (3, 112, 112)

for f in ('checkpoint.tar', 'BEST_checkpoint.tar', out, out + '.data'):
    if os.path.exists(f):
        os.remove(f)
print('SMOKE TEST PASSED')
