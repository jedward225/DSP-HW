这是一个非常经典的DSP结合机器学习的课程大作业。要拿到高分并“做充分做足”，核心在于**基础算法（DSP底层）的严谨性**与**上层应用（深度学习/大模型）的前沿性**相结合。

根据文档内容 ，我为你制定了一份详细的**Repo统筹规划与实验执行方案**。



------



### 一、 核心策略：稳扎稳打 + 降维打击



1. **基础分（稳）：** 手写 FFT/STFT/MFCC 必须代码清晰、数学原理正确，且必须通过单元测试证明其正确性（与 `librosa` 等库对比误差）。
2. **核心分（准）：** 检索和分类任务的 Pipeline 要完整，遵循 5-Fold 交叉验证的标准流程。
3. **高分点（狠）：**
   - **多维度对比：** 不仅对比超参数，还要对比特征（Spectrogram vs MFCC）、对比距离度量（欧氏距离 vs 余弦相似度 vs DTW）。
   - **模型深度：** CNN (ResNet) vs Transformer (ViT/AST) vs 大模型 (Wav2Vec2.0/HuBERT)。
   - **可视化：** 极其详尽的图表（t-SNE聚类图、混淆矩阵、超参数热力图）。

------



### 二、 Repository 结构规划



建议采用标准的机器学习工程结构，体现专业性：

```
DSP_Final_Project/
├── README.md               # 项目介绍、运行指南、成员分工
├── requirements.txt        # 依赖库
├── data/                   # 数据存放 (不要上传到git，只放占位符)
│   ├── raw/                # 原始WAV文件
│   └── processed/          # 提取好的特征 (npy文件)
├── src/                    # 源代码
│   ├── dsp_core/           # 【核心】手写DSP算法模块
│   │   ├── __init__.py
│   │   ├── fft.py          # 手写FFT实现
│   │   ├── stft.py         # 手写STFT实现
│   │   └── mfcc.py         # 手写MFCC实现
│   ├── models/             # 神经网络模型
│   │   ├── resnet.py
│   │   └── transformer.py
│   ├── utils/              # 工具函数
│   │   ├── dataset.py      # PyTorch Dataset/Dataloader
│   │   ├── metrics.py      # 计算 Top-K Accuracy, Precision 等
│   │   └── plot.py         # 绘图脚本
│   └── train.py            # 训练脚本
├── experiments/            # 实验记录 (Jupyter Notebooks)
│   ├── 01_dsp_verification.ipynb  # 验证手写算法与标准库的误差
│   ├── 02_task1_retrieval.ipynb   # 任务1：检索实验与超参数分析
│   ├── 03_task2_classification.ipynb # 任务2：分类训练可视化
│   └── 04_large_model_compare.ipynb  # 大模型对比实验
├── scripts/                # 一键运行脚本 (run_task1.sh, run_task2.sh)
└── report/                 # 最终报告与图表资源
```

------



### 三、 实验规划：做什么？怎么做？





#### 阶段 1：底层算法实现与验证（必须最先完成）





**要求：** 需要自己实现 FFT, STFT, MFCC 。



- **做法：**
  - 仅使用 `numpy` 实现上述算法。
  - **关键动作（加分项）：** 编写 `test_dsp.py`。输入一段正弦波或真实音频，计算你实现的 `my_stft` 和 `scipy.signal.stft` 或 `librosa.stft` 的结果，计算 MSE（均方误差）。
  - **报告体现：** 展示两者的差值几乎为0（浮点误差级别），证明你的底层实现是正确的。



#### 阶段 2：任务1 - 声音检索 (Retrieval)





**要求：** 1个fold查询，4个fold做库；Top10/20 精度；比较帧长、帧移 。



- **实验设计：**
  1. **基准线 (Baseline)：** 使用 MFCC 特征 + 余弦相似度 (Cosine Similarity)。
  2. **超参数网格搜索：**
     - Frame Length: [20ms, 25ms, 50ms]
     - Frame Shift (Hop Length): [10ms, 12.5ms, 25ms]
     - 绘制热力图，横轴纵轴为参数，颜色深浅表示 Top-10 Accuracy。
  3. **进阶对比（做足）：**
     - **距离度量对比：** 对比 欧式距离 vs 余弦距离 vs **DTW (动态时间规整)**。DTW 对于处理长度不一致的音频序列效果更好，这是一个很好的技术亮点。



#### 阶段 3：任务2 - 声音分类 (Classification)





**要求：** 自由选择神经网络；比较超参数；对比检索任务中的有无ML效果；与大模型对比 。



- **模型选择策略：**
  - **CNN代表：** ResNet-18（修改第一层输入通道为1，接受单通道Spectrogram）。
  - **Transformer代表：** 简单的 Vision Transformer (ViT) 或者专门的 AST (Audio Spectrogram Transformer)。
- **实验内容：**
  1. **训练可视化：** 记录 Loss 和 Accuracy 曲线 (Train/Val)。
  2. **特征有效性分析：** 比较直接输入 Spectrogram 图谱 vs 输入 MFCC 特征进神经网络的效果。
  3. **“ML辅助检索”对比（关键点）：**
     - 提取训练好的 ResNet 倒数第二层（FC层之前）的输出向量 (Embedding)。
     - 用这个 Embedding 重新做任务1的检索。
     - **预期结论：** 神经网络提取的语义特征检索精度应显著高于任务1中基于生肉特征（Raw Features）的检索。



#### 阶段 4：大模型降维打击 (Bonus)





**要求：** 需要与大模型直接分类做对比 。



- **做法：**
  - 不要自己训练大模型（资源不够）。
  - 使用 HuggingFace 的预训练模型，例如 `Wav2Vec 2.0` 或 `Hubert` 或 `CLAP`。
  - **Zero-shot 或 Linear Probing：** 冻结大模型参数，只训练最后一层分类器，或者直接提取大模型特征做分类/检索。
  - **展示：** 大模型在小样本下通常具有碾压优势，展示这种优势对比。

------

四、 团队分工建议 (3-5人) 



假设团队为4人，建议如下分工：

- **成员 A (DSP专家):**
  - **核心职责：** 负责 `src/dsp_core` 的手写实现（FFT/STFT/MFCC）。
  - **产出：** 算法代码、正确性验证报告（误差分析图）。
- **成员 B (任务1负责人):**
  - **核心职责：** 完成声音检索流程，实现 DTW/余弦距离计算。
  - **产出：** 任务1的超参数对比热力图，Top-K 精度表格。
- **成员 C (深度学习负责人):**
  - **核心职责：** 搭建 PyTorch 模型 (ResNet/Transformer)，编写训练 Pipeline。
  - **产出：** 训练曲线图，混淆矩阵，Neural Retrieval 的实验结果。
- **成员 D (大模型与统筹):**
  - **核心职责：** 调用 HuggingFace 大模型进行对比实验；负责最终报告的整合与润色；PPT制作。
  - **产出：** 大模型对比数据，t-SNE 特征可视化图（对比手工特征、CNN特征、大模型特征的聚类效果）。