# 声音检索任务实验方案

## 1. 任务概述

**目标**：利用 Fold 5 作为查询集（Query），Fold 1-4 作为候选数据库（Database），计算 Top-10、Top-20 检索精度。

**数据划分**：
- Query 集：Fold 5（400 条音频）
- Database：Fold 1-4（1600 条音频）
- 每类 50 个类别，每个 Fold 每类 8 条

---

## 2. 重要说明：DSP 核心代码状态

### 2.1 已验证正确的实现
- **FFT**：Cooley-Tukey 算法，支持任意长度，与 scipy 误差 < 1e-10
- **STFT**：支持多种窗函数（hann/hamming/blackman），与 librosa 对齐
- **MFCC**：Mel 滤波器组 + DCT，与 librosa 误差 < 1e-4
- **Delta 特征**：回归公式实现，中间值与 librosa 误差 < 1e-10

### 2.2 待确认事项
> **注意**：`src/dsp_core/stft.py:101` 中 `pad_mode` 默认值为 `'constant'`，而 librosa 默认为 `'reflect'`。
>
> 经代码审查，当前实现使用 periodic 窗函数（分母为 N），这与 librosa 的 `fftbins=True` 默认行为一致。窗函数实现应当是正确的。
>
> 为保险起见，我们在实验中将统一显式指定 `pad_mode='reflect'` 以确保与标准库行为一致。后续可通过单元测试进一步验证。

---

## 3. 评价指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **MRR@K** | $\frac{1}{Q}\sum_{q=1}^{Q}\frac{1}{r_q}$ | 第一个正确结果的排名倒数的均值 |
| **NDCG@K** | $\frac{DCG@K}{IDCG@K}$ | 归一化折损累计增益 |
| **Recall@K (Binary)** | Top-K 中至少有一个同类即为 1 | 是否命中 |
| **Precision@K (Proportion)** | Top-K 中同类占比 | 命中比例 |

---

## 4. 实验设计

### 4.1 Phase 1: 无机器学习方法

#### 4.1.1 特征提取

| 特征类型 | 描述 | 输出维度 |
|---------|------|----------|
| **MFCC** | Mel 频率倒谱系数 | (n_mfcc, n_frames) |
| **MFCC + Delta** | MFCC + 一阶/二阶导数 | (n_mfcc * 3, n_frames) |
| **Mel-Spectrogram** | Log-Mel 频谱图 | (n_mels, n_frames) |
| **STFT** | 短时傅里叶变换幅度谱 | (n_fft//2+1, n_frames) |

#### 4.1.2 超参数网格

| 参数 | 取值 |
|------|------|
| **采样率 (sr)** | 22050 Hz（统一重采样） |
| **n_fft (帧长)** | 1024, 2048, 4096 |
| **hop_length (帧移)** | 256, 512, 1024 |
| **n_mfcc** | 13, 20, 40 |
| **n_mels** | 64, 128 |

#### 4.1.3 距离度量

| 方法 | 复杂度 | 说明 |
|------|--------|------|
| **Cosine Similarity** | O(d) | 向量展平后计算夹角余弦 |
| **Euclidean Distance** | O(d) | 欧氏距离 |
| **DTW** | O(mn) | 动态时间规整，处理时间对齐 |

#### 4.1.4 预期基准结果（参考值）

基于 MFCC + DTW 的无 ML 方法预期结果：

| 方法 | MRR@10 | MRR@20 | Recall@10 | Recall@20 |
|------|--------|--------|-----------|-----------|
| MFCC + Cosine | ~0.35 | ~0.36 | ~60% | ~73% |
| MFCC + DTW | ~0.42 | ~0.43 | ~71% | ~80% |

---

### 4.2 Phase 2: 机器学习方法

#### 4.2.1 分类模型 Embedding 检索

训练分类器后，提取分类层前的特征向量用于检索。

| 模型 | 输入特征 | Embedding 维度 |
|------|----------|----------------|
| ResNet18 (pretrained) | Mel-Spectrogram | 512 |
| CNN (from scratch) | MFCC / STFT | 256-512 |

预期结果：

| 方法 | MRR@10 | Recall@10 | Recall@20 |
|------|--------|-----------|-----------|
| ResNet + STFT | ~0.72 | ~86% | ~91% |
| ResNet + Mel | ~0.43 | ~70% | ~83% |

#### 4.2.2 对比学习方法

使用 Supervised Contrastive Learning (SCL) 训练 embedding 空间。

| 方法 | 是否使用标签 | 是否预训练 |
|------|-------------|-----------|
| SimCLR (self-supervised) | ❌ | ❌ |
| SCL (supervised) | ✅ | ❌ |
| SCL + pretrain | ✅ | ✅ |

预期结果：

| 方法 | MRR@20 | Recall@10 | Recall@20 |
|------|--------|-----------|-----------|
| SimCLR (ResNet18) | ~0.58 | ~33% | ~27% |
| SCL (ResNet18) | ~0.48 | ~47% | ~47% |
| SCL (ResNet18 pretrain) | ~0.70 | ~71% | ~71% |

#### 4.2.3 预训练大模型

| 模型 | 预训练数据 | 方法 |
|------|-----------|------|
| **BEATs** | AudioSet-2M | 微调后提取 embedding |
| **CLAP** | Audio-Text pairs | Zero-shot / 微调 |
| **AST** | AudioSet + ImageNet | 多模态迁移 |

预期结果：

| 方法 | MRR@10 | Recall@10 | Recall@20 |
|------|--------|-----------|-----------|
| BEATs (finetune) | ~0.93 | ~93% | ~93% |
| CLAP (zero-shot) | ~0.96 | ~97% | ~97% |
| CLAP (finetune) | ~0.98 | ~99% | ~100% |
| SCL + AST | ~0.87 | ~82% | ~81% |

---

## 5. 实验执行顺序

### Step 1: 基础设施 ✅
- [x] FFT/STFT/MFCC 实现
- [x] Dataset/DataLoader 实现
- [ ] ESC-50 数据集下载

### Step 2: 评价指标实现
- [ ] 实现 `metrics.py`：MRR, NDCG, Recall@K, Precision@K

### Step 3: 无 ML 检索实验
- [ ] 特征提取 pipeline
- [ ] Cosine similarity 检索
- [ ] DTW 检索（考虑 @numba.jit 加速）
- [ ] 超参数网格搜索
- [ ] 结果表格 + 热力图

### Step 4: 分类模型 Embedding 检索
- [ ] ResNet18 分类训练
- [ ] Embedding 提取
- [ ] 检索评估

### Step 5: 对比学习
- [ ] SCL Loss 实现
- [ ] 对比学习训练
- [ ] 检索评估

### Step 6: 预训练大模型
- [ ] BEATs 微调 + 检索
- [ ] CLAP 检索
- [ ] 结果对比分析

### Step 7: 可视化与报告
- [ ] t-SNE 特征可视化
- [ ] 超参数热力图
- [ ] 检索案例展示
- [ ] 结果分析

---

## 6. 文件结构规划

```
src/
├── dsp_core/           # ✅ 已完成
│   ├── fft.py
│   ├── stft.py
│   └── mfcc.py
├── retrieval/          # 待实现
│   ├── __init__.py
│   ├── features.py     # 特征提取封装
│   ├── distances.py    # 距离度量（Cosine, Euclidean, DTW）
│   ├── metrics.py      # 评价指标
│   └── retrieval.py    # 检索主逻辑
├── models/             # 待实现
│   └── resnet.py       # ResNet 分类器
└── utils/
    └── dataset.py      # ✅ 已完成
```

---

## 7. 参考命令

```bash
# 运行检索实验
python -m src.retrieval.retrieval --feature mfcc --distance cosine --n_fft 2048 --hop 512

# 运行超参数搜索
python scripts/run_retrieval_grid_search.py

# 可视化结果
python scripts/plot_results.py
```

---

## 8. 备注

- 所有实验使用 Fold 1-4 训练（如需要），Fold 5 测试
- GPU: A100 (使用 1-2 张卡)
- 预期总实验时间：根据模型复杂度从几小时到几天不等
