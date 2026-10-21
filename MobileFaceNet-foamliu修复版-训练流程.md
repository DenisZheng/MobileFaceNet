# MobileFaceNet（foamliu 修复版）构建、验证与训练流程（RTX 4080 / i5-13600K / 16GB RAM）

本流程与附件《MobileFaceNet-ArcFace 构建、验证与训练流程》的区别先说清：

```text
代码基线：foamliu/MobileFaceNet master a6cc903 + 本仓库修复版（见 PATCH_NOTES.md，共 14 项修复）
不用：InsightFace arcface_torch / train_v2.py / MS1M-RetinaFace(MS1MV3) / 512 维 mbf
训练集：MS1M-ArcFace（faces_emore），85,742 人 / 5,822,653 张
测试集：LFW funneled 版（6,000 对）
模型：MobileFaceNet 128 维，999,936 参数
最终目标：ONNX（opset 11）→ ESP-PPQ 量化 → .espdl
```

## 1. 当前硬件

```text
CPU:
Intel Core i5-13600K

RAM:
16GB

GPU:
NVIDIA GeForce RTX 4080

VRAM:
16376 MiB

NVIDIA Driver:
595.84

nvidia-smi CUDA Version:
13.2
```

训练环境（本流程主线为原生 venv，Docker 为可选项，见第三节）：

```text
Python 3.10 ~ 3.12
PyTorch 2.x（CUDA 12.x 构建）
其余依赖见仓库 requirements.txt
```

修复版已在 PyTorch 2.14.1+cpu 上通过仓库自带 `smoke_test.py` 的全部检查（前向、ArcFace 头、checkpoint 往返、ONNX 导出、无 mxnet 的 RecordIO 读取）。GPU 训练环境以第二节的分层验证为准。

---

# 一、正确目录结构

foamliu 的代码所有路径都是相对路径（`data/...`、`checkpoint.tar`、`runs/` 都落在当前工作目录），因此**仓库根目录就是运行目录**，数据集必须放在仓库内的 `data/` 下，不能像 InsightFace 版那样把数据挂到别处再改 config 指过去。磁盘放不下时用软链接把 `data` 指到大盘，不要改代码里的路径。

```text
~/mobilefacenet-lab/
├── mobilefacenet-fixed/        # 修复版仓库（运行目录）
│   ├── config.py
│   ├── train.py
│   ├── data/                   # 数据集（见第四节）
│   │   ├── faces_emore.zip
│   │   ├── faces_emore/        # train.rec / train.idx / property
│   │   ├── images/             # pre_process.py 展开的单张 jpg
│   │   ├── faces_ms1m_112x112.pickle
│   │   ├── lfw-funneled.tgz
│   │   ├── pairs.txt
│   │   └── lfw_test_pair.txt
│   ├── runs/                   # TensorBoard 日志（自动生成）
│   ├── checkpoint.tar          # 最新 checkpoint（自动生成）
│   └── BEST_checkpoint.tar     # LFW 最优 checkpoint（自动生成）
└── logs/                       # 宿主机侧保存的训练日志与环境记录
```

---

# 二、宿主机准备

创建工作目录并放入修复版代码：

```bash
mkdir -p "$HOME/mobilefacenet-lab/logs"
cd "$HOME/mobilefacenet-lab"
# 把交付的 mobilefacenet-fixed.zip 解压到这里
unzip mobilefacenet-fixed.zip
cd mobilefacenet-fixed
```

记录基线版本（两个都要记，缺一不可复现）：

```bash
# 上游基线：foamliu/MobileFaceNet master a6cc9032a659b615f477833e1a70b5e7931bcccc
cat PATCH_NOTES.md | head -5
sha256sum config.py train.py mobilefacenet.py facenet_utils.py \
  > "$HOME/mobilefacenet-lab/logs/source-sha256.txt"
```

创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

注意：

```text
不要装 mxnet。pre_process.py 在无 mxnet 时自动走内置 RecordIO 读取器（已实测）。
torch 必须 ≥ 2.0：修复版的 checkpoint 加载用了 weights_only 参数，1.x 会报错。
```

可选 Docker 路线：用官方 `pytorch/pytorch:2.x-cuda12.x-cudnn9-devel` 镜像，把仓库目录整体挂到容器工作目录后在容器内执行本文全部命令；挂载点必须是仓库根目录本身，不要只挂 data。

---

# 三、环境分层验证

逐层验证，上一层不过不要往下走。

## 第一层：GPU / NVIDIA Runtime

```bash
nvidia-smi
```

应看到：

```text
NVIDIA GeForce RTX 4080
Driver 595.84
```

`nvidia-smi` 显示的 CUDA Version 是驱动能力上限，不代表 PyTorch 使用的 CUDA 版本。

## 第二层：PyTorch / CUDA / cuDNN

```bash
python - <<'PY'
import sys
import torch
import torchvision

print("Python      :", sys.version)
print("PyTorch     :", torch.__version__)
print("TorchVision :", torchvision.__version__)
print("CUDA build  :", torch.version.cuda)
print("cuDNN       :", torch.backends.cudnn.version())
print("CUDA avail  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
    print("Capability  :", torch.cuda.get_device_capability(0))
PY
```

目标：

```text
CUDA avail  : True
GPU         : NVIDIA GeForce RTX 4080
```

## 第三层：CUDA 实算

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available()
device = torch.device("cuda:0")
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)
c = a @ b
torch.cuda.synchronize()
print("GPU:", torch.cuda.get_device_name(0))
print("result:", c.shape)
print("allocated MB:", torch.cuda.memory_allocated() / 1024 / 1024)
print("CUDA calculation: OK")
PY
```

必须看到 `CUDA calculation: OK`。

## 第四层：仓库自检（本流程特有，不要跳过）

```bash
python smoke_test.py
```

必须看到结尾：

```text
SMOKE TEST PASSED
```

它实际检查了：

```text
核心模块可 import，且 import 不再触发 RetinaFace 检测器实例化
MobileFaceNet 前向 [2,3,112,112] -> [2,128]，参数量 999,936
ArcFace 头类别数 = config 的 85,742，最大标签 85741 可正常索引
checkpoint 新格式存/读往返，重载模型 embedding 逐位一致
ONNX（opset 11）可导出，算子集只有 Conv/Add/Clip/Relu/Reshape 等
data_gen 的归一化与 config.NORM_PRESET 一致
```

## 第五层：依赖与环境记录

```bash
python -m pip check
python -m pip freeze | sort > "$HOME/mobilefacenet-lab/logs/pip-freeze.txt"
nvidia-smi > "$HOME/mobilefacenet-lab/logs/nvidia-smi.txt"
```

---

# 四、数据集下载与准备

## 1. 训练集：MS1M-ArcFace（faces_emore）

来源：InsightFace 官方数据仓库页的 **MS1M-ArcFace**（85K ids / 5.8M images）。foamliu 的 `faces_emore` 就是这个包。

```text
Google Drive: https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view?usp=sharing
百度网盘:     https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ
```

下载（宿主机装 gdown 后）：

```bash
python3 -m pip install --user gdown
cd "$HOME/mobilefacenet-lab/mobilefacenet-fixed/data" 2>/dev/null || mkdir -p "$HOME/mobilefacenet-lab/mobilefacenet-fixed/data"
python3 -m gdown 1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR -O faces_emore.zip
```

`extract.py` 写死了 `data/faces_emore.zip` 这个路径，下载文件名对不上时改名为 `faces_emore.zip` 放进仓库 `data/`。Google Drive 失败时用上面的百度盘链接手工下载，放到同一位置。

解压：

```bash
cd "$HOME/mobilefacenet-lab/mobilefacenet-fixed"
python extract.py
```

检查核心文件：

```bash
test -f data/faces_emore/train.rec && echo "train.rec: OK"
test -f data/faces_emore/train.idx && echo "train.idx: OK"
ls -lh data/faces_emore/
```

官方包通常还带 `property` 和 `lfw.bin / cfp_fp.bin / agedb_30.bin`。注意：**这三个 .bin 本代码不用**（foamliu 的评估走 LFW 图片版，见下），留着不碍事，不要为了它们改流程。

## 2. 展开训练图片：pre_process.py

```bash
python pre_process.py
```

它做的事：读 `train.rec/train.idx` → 每张图写成 `data/images/<序号>.jpg` → 生成 `data/faces_ms1m_112x112.pickle`（样本清单）。

开始前必须知道：

```text
耗时：580 万张图的解码+写盘，按小时计，先在 tmux 里跑（见第六节的 tmux 用法）
磁盘：展开后是 580 万个小文件，预留数十 GB SSD 空间并确认文件系统 inode 充足
内存：样本清单在内存中拼装，16GB RAM 机器上跑它时关掉其他大内存程序
```

跑完必须核对它打印的汇总（修复版加了校验）：

```text
num_samples: 5822653
num_classes: 85742
```

出现 `WARNING: expected 85742 classes` 说明包下错了或解压不完整，**不要带着警告继续训练**——`config.num_classes` 与真实类别数不一致时 ArcFace 头和标签会对不上。

## 3. 测试集：LFW（funneled 版）

`lfw_eval.py` 期望两个文件都在 `data/` 下：

```bash
cd "$HOME/mobilefacenet-lab/mobilefacenet-fixed/data"
wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
wget http://vis-www.cs.umass.edu/lfw/pairs.txt
md5sum lfw-funneled.tgz   # 应为 1b42dfed7d15c9b2dd63d5e5840c86ad
```

必须是 **funneled** 版：`lfw_test()` 解压后会断言 5,749 个 subjects、13,233 张图，下成原版或 deepfunneled 会直接断言失败。

生成 foamliu 格式的测试对文件（上游缺这一步，修复版新增脚本）：

```bash
cd "$HOME/mobilefacenet-lab/mobilefacenet-fixed"
python make_lfw_pairs.py
# 应输出：wrote 6000 pairs to data/lfw_test_pair.txt
head -2 data/lfw_test_pair.txt
```

应类似：

```text
Abel_Pacheco/Abel_Pacheco_0001.jpg Abel_Pacheco/Abel_Pacheco_0004.jpg 1
Aaron_Eckhart/Aaron_Eckhart_0001.jpg Aaron_Guiel/Aaron_Guiel_0001.jpg 0
```

注意：LFW 数据必须在**开始训练前**就位。`train.py` 每个 epoch 结束都会调用 `lfw_test()`，缺文件会在第一个 epoch 末尾才报错，白跑一轮。首次评估时它会对全部 13,233 张图做一次 RetinaFace 检测+对齐并缓存为 `data/lfw_funneled.pkl`（修复版已恢复缓存判断，之后每轮直接复用），首次明显慢是正常现象。

## 4. 数据许可

```text
MS1M-ArcFace 源自 MS-Celeb-1M，微软已撤回原始分发；
InsightFace 镜像的条款是：仅限非商业研究。
```

本流程定位是复现与技术验证。模型若要进入商用产品，训练数据必须换成有明确商业授权的数据集，并重新评估，那是另一个阶段的事，不要默认 emore 训出的权重可以商用。

---

# 五、训练配置检查

正式训练前把下面这张表逐项对着代码核一遍并记录（值来自修复版当前代码，不要凭记忆）：

```text
backbone          MobileFaceNet（mobilefacenet.py，结构未改）
embedding_size    128（config.emb_size；注意 --emb-size 参数不决定输出维度）
input             112x112 RGB，已对齐人脸
num_classes       85742（config.py，必须与 pre_process 汇总一致）
num_samples       5822653
batch_size        512（--batch-size）
steps/epoch       约 11,372
optimizer         SGD, momentum 0.9, nesterov=True
learning rate     0.1（--lr），MultiStepLR 里程碑 [10, 20, 30, 40]，gamma 0.1
weight decay      全局 4e-5；conv3 单独 4e-4（train.py 参数组）
loss              CrossEntropy + ArcMargin（margin m=0.5, s=64, easy_margin=False）
                  focal loss 默认关（--focal-loss False；修复版已支持正确写 False）
grad clip         5.0
num_workers       4（train.py 中写死，改并发改这一行）
end_epoch         50（--end-epoch）
归一化            NORM_PRESET='esp'，即 (x-127.5)/127.5（config.py）
                  警告：评估作者发布的预训练权重时必须切 'imagenet'，自训练全程不要动它
随机种子          7（train.py 内固定）
```

特别复查两项（都是上游出过事的地方）：

```bash
grep -n "NORM_PRESET\|num_classes\|num_samples" config.py
grep -n "default=" facenet_utils.py | head -15
```

---

# 六、RTX 4080 与 16GB RAM 的配置原则

显存 16GB 对这个模型很宽裕：backbone 只有约 1M 参数，显存大头是 ArcFace 头的 logits——batch 512 × 85,742 类的 fp32 logits 一份约 176MB，前向+反向数份叠加仍在可控范围。因此 batch 512 是合理起点，但仍按第七节阶梯实测，不要直接满负荷长跑。

系统 RAM 16GB 才是紧的一头：

```text
ArcFaceDataset 初始化时把 580 万条样本清单整个 pickle 读进内存（数 GB 量级）
DataLoader worker 靠 fork 的写时复制共享这份清单，不要改成 spawn
num_workers 先用 4；RAM 吃紧先降 worker，不要先动 batch
不要学大内存服务器把数据放 tmpfs，16GB RAM 放不下
监控 Swap：Swap 持续增长 = 内存不够，先降 worker / 关其他程序
```

tmux（宿主机安装，长任务都放 tmux 里跑）：

```bash
sudo apt install -y tmux
tmux new -s mobilefacenet
# 断开：Ctrl+B 然后 D；重连：tmux attach -t mobilefacenet
```

`pre_process.py` 和正式训练都在 tmux 会话中执行，SSH 断开不影响。

---

# 七、batch size 阶梯测试与短跑验证

## 1. batch 阶梯

每次只改 batch，其余参数不动，各跑约 100 个 batch，观察显存、GPU 利用率、loss 和速度：

```text
128 → 256 → 512
```

```bash
python train.py --batch-size 128 --end-epoch 1   # 看日志后 Ctrl+C 停掉即可
```

不要把正式配置设到刚好不 OOM 的极限；正式值取阶梯中稳定值，默认回到 512。

## 2. num_workers

默认 4。若 GPU 经常等数据（`nvidia-smi dmon` 里 SM 利用率频繁掉底）且 RAM、Swap 都稳，可改 `train.py` 中 DataLoader 的 `num_workers` 试 6 / 8；RAM 报警则退回。正式值以实测为准并记入日志。

## 3. 短跑验证（正式训练的门禁）

用正式参数启动一次，盯到下面全部满足再放长跑：

```text
loss 从高位持续下降，无 NaN / Inf
每 100 batch 有日志输出（print_freq=100）
第一个 epoch 结束时 lfw_test 完整跑完（首次含对齐缓存构建，明显更慢属正常）
checkpoint.tar 与 BEST_checkpoint.tar 出现在仓库根目录
TensorBoard 能打开并看到 model/train_loss、model/lfw_acc 曲线
RAM / Swap / 显存全程稳定
```

TensorBoard：

```bash
tensorboard --logdir=runs --port 6006
```

---

# 八、正式训练

在 tmux 会话中执行，完整记录日志和退出码：

```bash
cd "$HOME/mobilefacenet-lab/mobilefacenet-fixed"
source .venv/bin/activate

timestamp=$(date +%Y%m%d_%H%M%S)
LOG="$HOME/mobilefacenet-lab/logs/mobilefacenet-train-${timestamp}.log"

set -o pipefail

PYTHONUNBUFFERED=1 \
python train.py 2>&1 | tee "$LOG"

status=$?
echo "Training exit code: $status" | tee -a "$LOG"
echo "Log: $LOG"
exit "$status"
```

改参数就用命令行覆盖并把完整命令记进日志文件头，例如显存或时间不够时：

```bash
python train.py --batch-size 256 --end-epoch 50
```

断点续训：

```bash
python train.py --checkpoint checkpoint.tar
```

须知（修复版已处理，但语义要清楚）：

```text
checkpoint.tar / BEST_checkpoint.tar 均为 state_dict 格式，含 backbone、
ArcFace 头和 optimizer 状态；续训从 epoch+1 继续。
续训时调度器以 last_epoch=start_epoch-1 重建，LR 里程碑仍落在原来的 epoch
（10/20/30/40），中断多少次都不会让 LR 曲线整体后移。
续训前可在目标机器上跑一次 python test_resume_semantics.py 确认这一点。
注意：每次续训会重新播种随机数、重建 DataLoader 的 shuffle，所以数据顺序不是
位级续接，但 LR 与优化器状态是连续的。
训练期间不要在仓库根目录运行任何会写 checkpoint.tar 的自检脚本。
旧格式（整对象 pickle）的上游 checkpoint 不能直接续训，先用 export.py 转出权重。
```

时长预期：每 epoch 约 11,372 步（batch 512）。先用短跑实测的 steps/s 推算单 epoch 时长，再决定是跑满 50 epoch 还是提前停。经验上 LFW 先到平台期：`train.py` 每轮按 LFW 保存 BEST，曲线走平后继续跑的收益很小，可以停——BEST_checkpoint.tar 里就是最好的那一轮，不会丢。

---

# 九、实时监控

GPU：

```bash
watch -n 1 nvidia-smi
# 或
nvidia-smi dmon
```

CPU / RAM：

```bash
watch -n 2 free -h
btop   # 如已安装
```

日志：

```bash
tail -f "$HOME/mobilefacenet-lab/logs"/mobilefacenet-train-*.log
```

每个 epoch 重点看三行：`Learning rate=`、LFW 的 `Accuracy: ...%, threshold: ...`、以及是否有异常栈。

---

# 十、异常排查

| 现象 | 先查什么 |
|---|---|
| CUDA OOM | 降 `--batch-size`（512→256）；确认没有别的进程占显存（`nvidia-smi`） |
| 系统 RAM 爆 / 进程被 kill | `free -h` 看 Swap；`sudo journalctl -k --since "-10 min" \| grep -Ei 'oom\|killed process'`；降 num_workers、关其他程序 |
| loss 不降或 NaN | 先核对第五节配置表：lr 是否 0.1、`NORM_PRESET` 是否被误改、num_classes 是否与 pre_process 汇总一致；再查数据是否展开完整 |
| LFW 断言 subjects/files 不符 | 下错了 LFW 版本，必须是 funneled 版（5,749 / 13,233） |
| 第一个 epoch 末尾才报错缺文件 | LFW 三件套（tgz、pairs、`lfw_test_pair.txt`）没在训练前备齐，回到第四节第 3 步 |
| pre_process 报类别警告 | 数据包不完整或下错包，不要忽略警告继续训 |
| NVIDIA 驱动异常 | `sudo journalctl -k --since "-10 min" \| grep -Ei 'NVRM\|Xid'` |
| 续训加载失败 | checkpoint 是否为修复版 state_dict 格式；旧格式先用 `export.py` 转权重，不要改加载代码绕过 |

排查时保留：完整 traceback、最后 100~300 行训练日志、当时的 config 快照、`pip freeze`、`nvidia-smi` 输出。

---

# 十一、训练期验证（LFW）

`train.py` 每个 epoch 结束自动在 LFW 上评估并打印 `Accuracy / threshold`，TensorBoard 同步记录 `model/lfw_acc`。

先把口径说清，避免和 InsightFace 版文档的数字混比：

```text
foamliu 口径：在全部 6,000 对上做单一角度阈值寻优得到的最高 accuracy；
             每张图用 原图+水平翻转 两份特征相加再归一化。
不是：10-fold 交叉验证的 mean ± std（那是 InsightFace arcface_torch 的口径）。
两者数值接近但定义不同，报告里必须注明用的是哪一种。
```

作者公布的参照值：LFW **99.48%**（MS1M-ArcFace 训练）。注意作者当年用的是 ImageNet 归一化口径，本流程是 `'esp'` 归一化从头训，数值可比但不保证逐位复现这个数；以实测为准，差零点几个点先查配置和数据完整性，不要先怀疑模型。

每个 epoch 记录一行（BEST 更新时立刻算哈希，事后补不回来）：

```bash
sha256sum BEST_checkpoint.tar
```

| epoch | LFW accuracy | threshold（角度，度） | BEST_checkpoint.tar SHA-256 | 备注 |
|---|---|---|---|---|
| 待补 | 待补 | 待补 | 待补 |  |

阈值语义：这里的 threshold 是两张脸特征向量的**夹角（度）**。部署侧若用余弦相似度，换算为 `cos(threshold°)`，并在自己的业务数据上重新标定，不要直接把 LFW 阈值搬进产品。

---

# 十二、候选模型与最终模型选择

本代码与 InsightFace 版有一个关键不同，省了不少事：

```text
arcface_torch：只按 epoch 覆盖保存，验证最高分不自动留存，要自己做候选管理。
foamliu 修复版：save_checkpoint 按 LFW accuracy 自动维护 BEST_checkpoint.tar，
               LFW 刷新纪录才覆盖。最终模型默认就是它。
```

正式留存规则：

1. 训练结束（或手动停）后，把 `BEST_checkpoint.tar` 复制为不可覆盖的归档名：`best-epoch<E>-lfw<acc>.tar`，`<E>` 和 `<acc>` 从日志里对应 epoch 抄。
2. 同时归档该 run 的完整日志、`config.py` 快照、`pip freeze`。
3. 每个归档文件算 SHA-256 记入交付清单。
4. 部署只用 backbone 权重：下一步 `export.py` 从 BEST 导出 `mobilefacenet.pt`（纯 state_dict）。ArcFace 头（85,742 类分类权重）是训练专用，**不要带进部署包**。

---

# 十三、训练后独立评估

脱离训练进程，用最终权重单独复评一次 LFW，确认 BEST 的分数可复现：

```bash
cp <归档的 best>.tar BEST_checkpoint.tar
python lfw_eval.py
```

它会依次打印评估进度、阈值和 `Accuracy: ...%`（`lfw_funneled.pkl` 缓存存在时跳过对齐重建）。把这次独立复评的 accuracy、threshold、权重 SHA-256 记入交付报告；与训练期对应 epoch 的数字应一致，对不上先查归一化开关和权重文件是否拿错。

MegaFace 本流程不做：官方数据集 2020 年已停止分发，为一个附加指标去找来路不明的镜像不值得。需要更多验证维度时，用 emore 包里自带的 `cfp_fp.bin / agedb_30.bin` 另配 InsightFace 的 verification 工具评，那属于扩展项，不在本流程门禁内。

有实际业务场景（门禁、考勤、设备解锁等）时，公开集分数只是入场券：另备业务 dev 集标定阈值、业务 test 集在阈值冻结后做最终验收，报告至少含 ROC、TAR@FAR、最终阈值和按距离/姿态/光照/遮挡切片的失败样本。目标 FAR 先于看 test 结果定死。

---

# 十四、ONNX 与 ESPDL 部署验证

## 1. 导出 ONNX

```bash
python export.py          # BEST_checkpoint.tar -> mobilefacenet.pt（+ scripted 版）
python export_onnx.py     # -> mobilefacenet.onnx
```

固定契约：

```text
opset 11，固定输入 [1, 3, 112, 112]，不带动态轴
输入名 input，输出名 embedding
输出语义：128 维 embedding（未做 L2 归一化，归一化在部署侧做）
预处理：RGB、5 点对齐到 112x112、(x-127.5)/127.5（与训练 NORM_PRESET='esp' 一致）
```

## 2. PyTorch / ONNX 数值对照

用同一批对齐好的人脸图（从 LFW 缓存里取几十张即可）比对：

```text
同一输入分别过 mobilefacenet.pt 与 mobilefacenet.onnx（ONNX Runtime）
逐图比对 embedding：最大/平均绝对误差、cosine similarity
判定线在跑之前定好（fp32 导出的 cosine 通常应 >0.9999，达不到先查预处理
是否两边一致，尤其是 RGB/BGR 和归一化，不要先怀疑导出本身）
```

## 3. ESP-PPQ 量化（本流程的终点）

```text
工具：esp-ppq 的 espdl_quantize_onnx
target：按实际芯片选 esp32s3 / esp32p4
quant_type：先 w8a8 全量化；精度掉得多时对敏感层用 dispatching_table 升 w8a16
calib_dataloader：必须是真实对齐人脸，预处理与训练完全一致
                 （112x112、RGB、(x-127.5)/127.5），calib_steps 先 32
input_shape：[1, 3, 112, 112]
```

量化后必做：拿 LFW 的一个子集（或业务 dev 集）用 .espdl 复评 accuracy，与 fp32 结果并排记录掉点幅度；再记录 .espdl 文件大小（0.99M 参数 int8 后约 1MB 量级）和目标芯片上的实测时延。能导出不等于能上线，掉点和时延都过线才算完。

---

# 十五、最终交付

```text
delivery/
├── MANIFEST.md               # 基线 commit、修复说明索引、每个文件的用途与哈希
├── checksums.sha256
├── source/                   # 修复版代码快照 + PATCH_NOTES.md
├── environment/              # pip-freeze.txt、nvidia-smi.txt、硬件记录
├── training/
│   ├── config.py             # 训练时实际配置快照
│   ├── train-<timestamp>.log # 完整训练日志（含退出码）
│   └── best-epoch<E>-lfw<acc>.tar
├── model/
│   ├── mobilefacenet.pt      # backbone state_dict
│   ├── mobilefacenet.onnx
│   └── mobilefacenet.espdl
├── evaluation/
│   ├── lfw-report.md         # 独立复评 accuracy / threshold / 口径说明 / 权重哈希
│   └── quant-report.md       # fp32 vs int8 掉点、时延、模型大小
└── contract/
    ├── preprocessing.md      # 检测/对齐/颜色/归一化契约
    └── embedding-and-similarity.md  # 128 维、L2 归一化、余弦与角度阈值换算
```

核心交付必须包含：最终权重及其 SHA-256、选择依据（哪一轮 LFW 最高）、完整训练配置与日志、LFW 独立复评报告、ONNX 与 .espdl 的数值/精度对照。只给一个 .pt 或 .onnx 文件不构成交付。

---

# 十六、推荐完整顺序

```text
 1. 放入修复版代码，记录基线（上游 a6cc903 + PATCH_NOTES）与关键文件哈希
 2. 建 venv 装依赖，过三节分层验证，smoke_test.py 必须 PASSED
 3. 下载 MS1M-ArcFace，extract + pre_process，核对 5822653 / 85742 无警告
 4. 下载 LFW funneled + pairs.txt，生成 lfw_test_pair.txt（6000 对）
 5. 按第五节冻结配置表，复查 NORM_PRESET 与 num_classes
 6. batch 阶梯 + num_workers 实测，定正式参数
 7. 短跑验证过门禁（loss、首轮 LFW、checkpoint、TensorBoard、资源稳定）
 8. tmux 里正式训练，全程记日志与退出码；每轮记录 LFW 与 BEST 哈希
 9. 归档 BEST（改名+哈希+日志+config 快照），export.py 导出 backbone
10. lfw_eval.py 独立复评，数字与训练期对齐
11. export_onnx.py 导出，PyTorch/ONNX 数值对照过线
12. ESP-PPQ 量化，LFW 子集复评掉点，记录 .espdl 大小与板端时延
13. （有业务时）dev 集标定阈值并冻结，test 集最终验收
14. 生成 delivery/、MANIFEST 与 checksums
```

---

# 十七、依据与版本记录

```text
上游代码：foamliu/MobileFaceNet master
  a6cc9032a659b615f477833e1a70b5e7931bcccc（2025-12-17）
修复版：本仓库 PATCH_NOTES.md（14 项，含 LFW 对齐缓存与测试对生成器两项补充）
数据集页面：InsightFace recognition/_datasets_（MS1M-ArcFace / LFW 条目）
LFW 官方站：http://vis-www.cs.umass.edu/lfw/
作者公布参照值：LFW 99.48%、MegaFace 82.55%（本流程不做 MegaFace）
```

上游 master 若再变化，不要直接覆盖本修复版；先逐项核对 PATCH_NOTES 的 14 项问题在新版里是否还存在，再决定迁移。
