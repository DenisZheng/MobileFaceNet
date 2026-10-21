# MobileFaceNets (fixed)

**English** | [简体中文](README.zh-CN.md)

> This repository is a repaired fork of [foamliu/MobileFaceNet](https://github.com/foamliu/MobileFaceNet) (Apache-2.0), based on upstream master `a6cc903`. After 2020, upstream master absorbed several third-party PRs and its training pipeline no longer runs as-is (inconsistent configuration sources, polluted class counts, a checkpoint format incompatible with modern PyTorch, and more). This fork fixes 21 engineering issues **without changing the 128-d MobileFaceNet model architecture**, so that it trains, evaluates, and exports ONNX on PyTorch 2.x, targeting ESP-PPQ / ESP-DL quantization.
>
> - Fix-by-fix notes: [PATCH_NOTES.md](PATCH_NOTES.md)
> - Full training guide (MS1M-ArcFace + LFW, RTX 4080; venv / Docker environment): [MobileFaceNet-foamliu修复版-训练流程.md](MobileFaceNet-foamliu修复版-训练流程.md)
> - Conda edition of the training guide (standalone document, Conda environment layer, with `environment.yml`): [MobileFaceNet-foamliu修复版-训练流程-Conda.md](MobileFaceNet-foamliu修复版-训练流程-Conda.md)
> - Environment self-check: `python smoke_test.py`
> - Note: third-party weight files under `weights/` and `pretrained_model/` are not distributed with this fork (unknown provenance or oversized). The RetinaFace weights needed for LFW alignment are in `retinaface/weights/`. Datasets (MS1M-ArcFace, LFW) are not distributed either; download them via the links in the training guides.
> - For the original implementation and citation, see the upstream repository and the paper link below.

# MobileFaceNets

![apm](https://img.shields.io/apm/l/vim-mode.svg)

PyTorch implementation of MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices.
[paper](https://arxiv.org/abs/1804.07573).

## Performance

|Accuracy|LFW|MegaFace|Download|
|---|---|---|---|
|paper|99.55%|92.59%||
|ours|99.48%|82.55%|[Link](https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet_scripted.pt)|

## Dataset
### Introduction

Refined MS-Celeb-1M dataset for training, 3,804,846 faces over 85,164 identities.
LFW and Megaface datasets for testing.

*Note: this fixed version's training guides use MS1M-ArcFace (faces_emore) instead: 5,822,653 faces over 85,742 identities.*

## Dependencies
- Python 3.10+
- PyTorch 2.x

See `requirements.txt` (venv / Docker route) or `environment.yml` (Conda route).

## Usage

### Data preprocess
Extract images:
```bash
$ python extract.py
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

To visualize the training process:
```bash
$ tensorboard --logdir=runs
```
