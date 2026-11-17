# even-ai

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%2Bcpu-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

用 PyTorch 训练一个小型卷积神经网络，在 MNIST 手写数字（0–9）上做分类。这是一份个人学习记录：仓库里目前只有这一份训练脚本，以及已经放好的 MNIST 原始数据。上面的徽章是本文验证时使用的版本，仓库没有锁定依赖。

## 仓库里有什么

- `ai-train-proj/train.py`：加载数据、定义网络、训练 5 个 epoch、在测试集上计算准确率，并用 Matplotlib 显示训练损失曲线。
- `ai-train-proj/data/MNIST/raw/`：MNIST 原始文件。训练集 60,000 张、测试集 10,000 张，均为 28×28 灰度图。`.gz` 压缩包和解压后的 `*-ubyte` 文件都在仓库里。

脚本不会把权重写到磁盘。`torch.save(...)` 那两行处于注释状态，仓库里也没有 `.pth` 权重文件。训练结束后只在终端打印准确率，并调用 `plt.show()` 显示损失曲线；曲线不会保存成图片。

## 脚本在做什么

`SimpleCNN` 的数据流：

| 步骤 | 输出形状 |
| --- | --- |
| 输入 | 1 × 28 × 28 |
| `Conv2d(1→16, kernel 5, padding 2)` + ReLU + `MaxPool2d(2)` | 16 × 14 × 14 |
| `Conv2d(16→32, kernel 5, padding 2)` + ReLU + `MaxPool2d(2)` | 32 × 7 × 7 |
| 展平后 `Linear(1568→10)` | 10 类（数字 0–9） |

按上述定义，可训练参数约为 2.9 万（卷积层含偏置 416 + 12,832，全连接层 15,690）。

写在脚本顶部的超参数：

| 名称 | 值 |
| --- | --- |
| `EPOCHS` | 5 |
| `BATCH_SIZE` | 64 |
| `LEARNING_RATE` | 0.001 |
| 损失函数 | `CrossEntropyLoss` |
| 优化器 | Adam |
| 归一化 | 均值 `0.1307`，标准差 `0.3081` |

有 CUDA 时用 GPU，否则用 CPU。脚本没有固定随机种子，重复运行时损失和准确率会有小幅波动。

torchvision 0.29 直接读取 `raw/` 里未压缩的 `*-ubyte` 文件。这些文件已在仓库中并通过完整性检查时，`download=True` 不会再次下载，也不会生成旧版的 `processed/*.pt` 缓存。请在 `ai-train-proj` 目录下运行，数据路径是相对路径 `./data`。

## 技术栈

依赖全部来自 `train.py` 的 import，仓库没有 `requirements.txt` 或锁文件。

| 依赖 | 在脚本中的用途 | 本文验证版本 |
| --- | --- | --- |
| Python | 运行 `train.py` | 3.12.3 |
| `torch` | 网络、损失、优化器、设备选择 | 2.15.0+cpu |
| `torchvision` | `datasets.MNIST`、`ToTensor`、`Normalize` | 0.29.2+cpu |
| `matplotlib` | 绘制每个 epoch 的平均训练损失 | 3.11.2 |

## 目录结构

```text
.
├── README.md
└── ai-train-proj/
    ├── train.py
    └── data/
        └── MNIST/
            └── raw/
                ├── train-images-idx3-ubyte
                ├── train-images-idx3-ubyte.gz
                ├── train-labels-idx1-ubyte
                ├── train-labels-idx1-ubyte.gz
                ├── t10k-images-idx3-ubyte
                ├── t10k-images-idx3-ubyte.gz
                ├── t10k-labels-idx1-ubyte
                └── t10k-labels-idx1-ubyte.gz
```

## 环境要求

- Python 3。本文在 Linux、Python 3.12.3 上验证。
- 可安装 PyTorch 与 torchvision。没有 GPU 也能跑，脚本会自动使用 CPU。
- 希望 `plt.show()` 弹出窗口时，需要可用的图形界面。无界面环境设置 `MPLBACKEND=Agg` 即可跑完训练并打印准确率；窗口不会出现，Matplotlib 会提示 `FigureCanvasAgg is non-interactive, and thus cannot be shown`。

## 安装与运行

在仓库根目录执行。PyTorch CPU 索引里没有 Matplotlib，所以分成两条 `pip install`。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib

cd ai-train-proj
MPLBACKEND=Agg python train.py
```

Debian/Ubuntu 上若 `python3 -m venv` 提示 `ensurepip is not available`，先安装 venv 支持再创建环境：

```bash
sudo apt install python3-venv
```

有 NVIDIA GPU、希望走 CUDA 时，按 [PyTorch 本机安装说明](https://pytorch.org/get-started/locally/) 安装对应的 `torch` 和 `torchvision`，然后在 `ai-train-proj` 目录执行 `python train.py`。本文的验证机器没有 CUDA，只跑通了 CPU 路径。

本地 `raw/` 里的未压缩文件齐全时不会重新下载。缺少这些文件时，`download=True` 才会让 torchvision 从网络拉取。

### 已在本机验证

环境：Linux，Python 3.12.3，`torch 2.15.0+cpu`，`torchvision 0.29.2+cpu`，`matplotlib 3.11.2`，`torch.cuda.is_available()` 为 false。

命令：在 `ai-train-proj` 下执行 `MPLBACKEND=Agg python train.py`。

结果：进程退出码 0，约 40 秒跑完 5 个 epoch。终端先打印 `Using device: cpu`，每个 epoch 结束打印平均训练损失，最后为：

```text
Finished Training
Accuracy of the model on the 10000 test images: 99.01 %
```

99.01% 是这一次运行的结果。脚本未固定随机种子，再次运行数值会略有不同。

## 备注

- 若要保存权重，取消 `train.py` 里 `torch.save(model.state_dict(), 'mnist_cnn_model.pth')` 附近两行的注释。按当前代码，运行后不会生成该文件。
- 损失曲线只通过 `plt.show()` 显示，仓库中没有截图或导出的图片。
- 数据集来自 [MNIST](http://yann.lecun.com/exdb/mnist/)（Yann LeCun 等）。本仓库附带的是其原始 idx 文件，训练代码是在此之上的学习脚本。
