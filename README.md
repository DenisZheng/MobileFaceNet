# MobileFaceNets（修复版 / fixed）

> 本仓库基于 [foamliu/MobileFaceNet](https://github.com/foamliu/MobileFaceNet)（Apache-2.0）master `a6cc903` 修复而来。上游 master 在 2020 年后被多个第三方 PR 改动，核心训练链已无法直接运行（配置来源不一致、类别数被污染、checkpoint 格式与现代 PyTorch 不兼容等）。本版在**不改变 128 维 MobileFaceNet 模型结构**的前提下修复了 21 项工程问题，使其在 PyTorch 2.x 上可训练、可评估、可导出 ONNX（面向 ESP-PPQ / ESP-DL 量化部署）。
>
> - 逐项修复说明：[PATCH_NOTES.md](PATCH_NOTES.md)
> - 完整训练流程（MS1M-ArcFace + LFW，RTX 4080）：[MobileFaceNet-foamliu修复版-训练流程.md](MobileFaceNet-foamliu修复版-训练流程.md)（venv / Docker 环境）
> - Conda 版训练流程（独立文档，环境层为 Conda，配套 `environment.yml`）：[MobileFaceNet-foamliu修复版-训练流程-Conda.md](MobileFaceNet-foamliu修复版-训练流程-Conda.md)
> - 环境自检：`python smoke_test.py`
> - 说明：`weights/` 与 `pretrained_model/` 中的第三方权重文件未随本仓库分发（来源不明或体积过大）；LFW 对齐所需的 RetinaFace 权重在 `retinaface/weights/` 内。数据集（MS1M-ArcFace、LFW）请按训练流程文档中的链接自行下载，本仓库不分发数据集。
> - 原始实现与引用请见上游仓库及下方论文链接。

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

## Dependencies
- Python 3.6.8
- PyTorch 1.3.0

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

To visualize the training process：
```bash
$ tensorboard --logdir=runs
```



