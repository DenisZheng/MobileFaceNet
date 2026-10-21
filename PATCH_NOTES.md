# foamliu/MobileFaceNet master 修复说明

基线：upstream master `a6cc903`（2025-12-17）。模型结构（128d MobileFaceNet）
**一字未改**——`mobilefacenet.py` 只改了一行 import。修的全是让它跑不起来、
或会训出废模型的工程问题。

## 修了什么

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `mobilefacenet.py` | 从 `custom_config` 读 `num_classes=21`（贡献者的小数据集配置），与训练数据 85,742 类对不上，ArcFace 头第一步就崩 | 改回 `from config import ...`，全仓单一配置来源 |
| 2 | `config.py` | `num_classes=2964 / num_samples=200000`（有人用 emore 前 20 万张子集练手留下的值，与任何公开数据集都不符） | 恢复 faces_emore 全量值 85742 / 5822653（原作者最终版口径） |
| 3 | `config.py` + `data_gen.py` | 归一化写死 ImageNet mean/std，和 ESP 部署的 127.5 口径没有统一入口 | 新增 `NORM_PRESET`：`'esp'`（默认，(x−127.5)/127.5，自训练→ESPDL 用）/ `'imagenet'`（评估作者发布的预训练权重时必须切这个）。训练、LFW 评估共用同一常量 |
| 4 | `facenet_utils.py` | 顶层 `from retinaface.detector import detector`，导入瞬间就在 CUDA 上实例化检测器——连 train.py 都 import 不了 | 改懒加载 `get_detector()`，首次做人脸对齐时才建，自动选 cuda/cpu |
| 5 | `facenet_utils.py` `save_checkpoint` | 存的是整个模型对象（pickle），PyTorch ≥2.6 默认 `weights_only=True` 直接拒绝加载；还把 DataParallel 和作者的类路径焊死在文件里 | 改存 state_dict（model/metric_fc/optimizer）。新增 `load_checkpoint_file` / `load_model_from_checkpoint`：新格式直接读，旧格式（原版 BEST_checkpoint.tar）自动降级兼容 |
| 6 | `facenet_utils.py` `parse_args` | 默认值被子集实验污染：lr 0.0001（ArcFace 不收敛）、batch 256、end-epoch 30；且 `type=bool` 使 `--focal-loss False` 反而开启 | 恢复 lr 0.1 / batch 512；end-epoch 设 50 与里程碑配套（原版 1000 实为"手动停"）；bool 参数改 `str2bool` 显式解析 |
| 7 | `train.py` | resume 靠 unpickle 整模型；单卡也无条件套 DataParallel；调度里程碑被改成 [5,10,15,20] | 统一先建模型再 load state_dict resume（旧格式 checkpoint 请先用 export.py 转一次）；仅多卡时套 DataParallel；里程碑恢复 [10,20,30,40]，`scheduler.step()` 现代写法。注意：resume 后调度器计数重置，LR 曲线不是位级续接 |
| 8 | `megaface.py` | `from utils import ...`，而 utils 已被 vendored 的 YOLOv5 包占用（空 `__init__`），ImportError | 改从 `facenet_utils` 导入，并用统一的 checkpoint loader |
| 9 | `lfw_eval.py` / `export.py` | `__main__` 里 `checkpoint['model'].module` 的旧式加载，现代 torch 加载即崩 | 统一走 `load_model_from_checkpoint`（新旧格式都吃） |
| 10 | `pre_process.py` | 循环写死 `range(200000)`（同 #2 的子集污染）；强依赖已停止维护的 mxnet（新 Python 装不上） | 改用 `config.num_samples` 全量；新增内置 RecordIO 读取器作 fallback（mxnet 在时仍优先用 mxnet），解码后会校验类别数并警告与 config 不符 |
| 11 | `requirements.txt` | 缺 `tensorboard`（train.py 一 import 就缺包）；列了没人用的 torchsummary | 补 tensorboard，去掉 torchsummary，注明 mxnet 可选 |
| 12 | `export_onnx.py`（新增） | 原仓没有可用的 ONNX 导出（weights/ 里那个是贡献者的，opset 18 且绑死他的权重） | 新增：固定输入 [1,3,112,112]、opset 11、`dynamo=False`（torch≥2.5 必须显式关，否则走 dynamo 导出、图不一样），输出 128d embedding |
| 13 | `lfw_eval.py` `lfw_test()` | 对齐缓存的判断被注释掉，每个 epoch 都对 13,233 张 LFW 图重跑一遍 RetinaFace 检测+对齐 | 恢复缓存判断：`lfw_funneled.pkl` 存在时直接复用，只在首次评估时处理 |
| 14 | `make_lfw_pairs.py`（新增） | `evaluate()` 需要的 `data/lfw_test_pair.txt`（6000 对）上游从未提供生成器；贡献者的 `create_lfw_test_pair.py` 要求另一种格式的预分割文件，同样无处可下 | 新增：把官方 `pairs.txt`（与 lfw-funneled.tgz 同源下载）转换成 foamliu 格式 `路径 路径 0/1` |

## 没动的部分（知情即可）

- `custom_*.py` + `custom_config.py`：贡献者的 21 类小数据集实验子系统，自成一套，
  核心训练链已不再依赖它。留着不碍事，别拿它当主线。
- `weights/` 里的 `.pt/.onnx`：贡献者自己的权重，来源不明，不要当精度基线。
- Release v1.0 的权重是 512d 的另一套结构，与本代码（128d）不兼容，不要混用。

## 验证结果（CPU, torch 2.14.1+cpu, `smoke_test.py`）

- 核心模块全部可 import，且 import 不再触发检测器实例化
- 前向：[2,3,112,112] → [2,128]；参数量 **999,936**（与论文 0.99M 一致）
- ArcMargin 头：logits [2, 85742]，最大标签 85741 可正常索引（即原 #1 故障点）
- checkpoint 新格式存→读往返，重载模型 embedding 与原模型**逐位一致**（eval 模式）
- 旧格式（整对象 pickle）checkpoint 可经新 loader 读出，embedding 逐位一致
- ONNX 导出（opset 11）：210 节点，算子集仅
  {Conv, Add, Clip(=ReLU6), Relu, Reshape, Constant, Identity}——正是 ESP-PPQ 最稳的一类图
- 无 mxnet 的 RecordIO fallback：自造 3 条记录的 .rec/.idx，标签全部正确读回
- `export.py` / `export_onnx.py` 端到端跑通（mobilefacenet.pt 4.0MB）

未验证：在真实 GPU + 完整 faces_emore 上跑通一个 epoch 并复现 LFW 99.48%——
需要在你的机器上做（沙盒无 GPU、无数据集）。建议第一步先用作者发布的预训练
权重 + `NORM_PRESET='imagenet'` 跑 `lfw_eval.py` 验基线，再切回 `'esp'` 从头训。

## 使用

```bash
pip install -r requirements.txt
python extract.py        # 解压 data/faces_emore.zip（需自行下载数据集）
python pre_process.py    # 生成 data/images + pickle（无 mxnet 自动走内置读取器）
python train.py          # 默认 lr 0.1 / batch 512 / 50 epoch，每 epoch 跑 LFW
python export.py         # BEST_checkpoint.tar -> mobilefacenet.pt + scripted
python export_onnx.py    # -> mobilefacenet.onnx（给 ESP-PPQ 用）
python smoke_test.py     # 随时可跑的环境自检
```
