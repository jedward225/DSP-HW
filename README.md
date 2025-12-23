# DSP 课程大作业：ESC-50 声音分类与检索

**团队成员：** 刘嘉骏、孙浩翔、田原、叶栩言、林梓杰

---

## 项目简介

本项目基于 ESC-50 数据集实现声音分类与检索系统，主要内容包括：

1. **自实现 DSP 算法**：FFT（Cooley-Tukey）、STFT、MFCC，使用 Numba JIT 加速
2. **多种分类模型**：ResNet18、BEATs、CNN14、CLAP
3. **完整实验分析**：超参数对比、消融实验、模型对比

### 核心结果

| 任务 | 方法 | 性能 |
|------|------|------|
| 分类 | BEATs + SpecAugment + Mixup | **96.50%** |
| 分类 | CLAP Ultimate Optimization | **98%+** |
| 分类 | CNN14 + SpecAugment + Mixup | **92.75%** |
| 检索 | CLAP | **99.50% Hit@10** |
| 检索 | DTW（非ML基线） | **70.45% Hit@10** |

---

## 仓库结构

```
DSP-HW/
├── ESC-50/                     # 数据集（不上传git）
├── src/
│   ├── dsp_core/               # 自实现DSP算法
│   │   ├── fft.py              # Cooley-Tukey FFT
│   │   ├── stft.py             # 短时傅里叶变换
│   │   └── mfcc.py             # 梅尔频率倒谱系数
│   ├── classification/
│   │   ├── models/             # 分类模型
│   │   │   ├── resnet.py       # ResNet18/34/50
│   │   │   ├── beats.py        # BEATs + Adapter
│   │   │   └── clap.py         # CLAP 零样本/微调
│   │   ├── CLAP/               # CLAP 源码
│   │   ├── train.py            # 训练脚本
│   │   ├── features.py         # 特征提取
│   │   └── augment.py          # 数据增强
│   └── utils/
│       └── dataset.py          # ESC-50 数据加载器
├── external/
│   └── BEATs.py                # BEATs 模型（Microsoft）
├── scripts/
│   ├── run_frame_experiments.py # 帧长/帧移实验
│   ├── grid_search.py          # 超参数搜索
│   └── test_clap_zeroshot.py   # CLAP 零样本测试
├── experiments/
│   └── 01_dsp_verification.ipynb # DSP 算法验证
├── checkpoints/                # 模型权重
├── report/
│   └── main.tex                # 最终报告（LaTeX）
└── ref/                        # 参考资料
```

---

## 快速开始

### 1. 环境配置

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install torch torchvision torchaudio
pip install numpy scipy librosa numba tqdm
pip install laion-clap transformers
```

### 2. 下载 ESC-50 数据集

```bash
# 数据集放置在 ESC-50/ 文件夹
# 下载地址：https://github.com/karolpiczak/ESC-50
```

### 3. 下载模型权重

```bash
# BEATs 权重
# 从 https://github.com/microsoft/unilm/tree/master/beats 下载
# 放置于：checkpoints/BEATs_iter3_plus_AS2M.pt
```

---

## 运行实验

### ResNet18 分类

```bash
# 标准训练（Mel 频谱图）
python -m src.classification.train --model resnet18 --feature mel --epochs 50

# 不同帧长配置
python -m src.classification.train --model resnet18 --n_fft 2048 --hop 512
python -m src.classification.train --model resnet18 --n_fft 4096 --hop 1024
```

### BEATs 分类

```bash
# BEATs + Adapter（冻结编码器）
python -m src.classification.train --model beats --mode adapter --epochs 50

# BEATs + SpecAugment + Mixup + 解冻
python -m src.classification.train --model beats --mode adapter \
    --spec_augment medium --mixup_lam 0.2 --unfreeze_epoch 10
```

### CLAP 分类

```bash
# 零样本（无需训练）
python -m src.classification.train --model clap --mode zeroshot

# 微调（冻结编码器）
python -m src.classification.train --model clap --mode finetune_frozen
```

### 帧长/帧移超参数实验

```bash
# 完整实验（ResNet18 不同 n_fft, hop_length）
python scripts/run_frame_experiments.py --epochs 30 --gpu 0

# 快速测试
python scripts/run_frame_experiments.py --quick
```

---

## 作业要求完成情况

### 任务一：声音检索
- [x] 自实现 FFT、STFT、MFCC
- [x] Top-10、Top-20 检索精度
- [x] 帧长/帧移超参数对比
- [x] ML vs 非ML 对比（CLAP vs DTW）

### 任务二：声音分类
- [x] 自实现 FFT、STFT、MFCC
- [x] 多种神经网络模型
- [x] 帧长/帧移超参数对比
- [x] 与大模型对比（CLAP、BEATs）
- [x] 将分类模型用于检索任务

---

## 报告

最终报告位于 `report/main.tex`，包含：

- DSP 算法实现细节
- 完整实验结果
- 消融实验与分析
- 速度优化技术

---

## 参考资料

- ESC-50: https://github.com/karolpiczak/ESC-50
- BEATs: https://github.com/microsoft/unilm/tree/master/beats
- CLAP: https://github.com/microsoft/CLAP
- PANNs (CNN14): https://github.com/qiuqiangkong/audioset_tagging_cnn
