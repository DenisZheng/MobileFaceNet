# MobileFaceNets（修复版）

[English](README.md) | **简体中文**

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

MobileFaceNets 的 PyTorch 实现：面向移动设备的高效实时人脸验证 CNN。
[论文](https://arxiv.org/abs/1804.07573)。

## 性能

|精度|LFW|MegaFace|下载|
|---|---|---|---|
|论文|99.55%|92.59%||
|本项目（上游）|99.48%|82.55%|[链接](https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet_scripted.pt)|

## 数据集
### 介绍

训练使用精炼版 MS-Celeb-1M 数据集，85,164 人共 3,804,846 张人脸。
测试使用 LFW 与 MegaFace 数据集。

*注：本修复版的训练流程改用 MS1M-ArcFace（faces_emore）：85,742 人共 5,822,653 张。*

## 依赖
- Python 3.10+
- PyTorch 2.x

见 `requirements.txt`（venv / Docker 路线）或 `environment.yml`（Conda 路线）。

## 使用

### 数据预处理
展开图片：
```bash
$ python extract.py
$ python pre_process.py
```

### 训练
```bash
$ python train.py
```

可视化训练过程：
```bash
$ tensorboard --logdir=runs
```
