"""
ESC-50 数据集加载器

使用前请先运行 preprocess.py

功能描述:
    该模块实现了一个 PyTorch Dataset 类，用于加载经 preprocess.py 预处理后的 ESC-50 音频特征数据 (.npz)。

核心依赖:
    - torch: 用于生成 Tensor
    - numpy: 加载 .npy 文件及矩阵运算
    - pandas: 读取元数据 CSV
    - src.dsp_core.mfcc: 调用其中的 delta 函数进行实时计算

使用示例:
    详见单元测试
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from src.dsp_core.mfcc import delta

class ESC50Dataset(Dataset):
    def __init__(self, meta_path, processed_dir, folds, delta_width=9, delta_order=0):
        """
        Args:
            meta_path (str): esc50.csv 的路径
            processed_dir (str): 存放 .npy 文件的文件夹路径
            folds (list): 需要加载的折
            delta_width (int): 计算差分时的窗口宽度
            delta_order (int): 差分阶数 (0=仅静态, 1=静态+一阶, 2=静态+一阶+二阶)
        """
        self.processed_dir = processed_dir
        self.delta_width = delta_width
        self.delta_order = delta_order
        
        # 1. 读取 Meta CSV
        self.df = pd.read_csv(meta_path)
        
        # 2. 筛选 Fold
        self.df = self.df[self.df['fold'].isin(folds)]
        
        # 3. 存储数字标签到自然语言标签的映射 (调试时可用)
        self.idx2cat = dict(zip(self.df['target'], self.df['category']))
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 1. 获取索引对应的行并构建文件路径
        row = self.df.iloc[idx]
        filename = row['filename']
        npy_name = filename.replace('.wav', '.npy')
        npy_path = os.path.join(self.processed_dir, npy_name)
        
        # 2. 加载 MFCC 矩阵
        # 形状: (n_mfcc, time_steps)
        data = np.load(npy_path)
        
        # 3. 计算并拼接 Delta 特征
        # 形状: (delta_order * n_mfcc, time_steps)
        channels = [data]
        current_features = data
        for _ in range(self.delta_order):
            current_features = delta(current_features, width=self.delta_width, order=1)
            channels.append(current_features)
        data = np.concatenate(channels, axis=0)
        
        # 4. 转换为 Tensor 并增加 Channel 维度
        # 形状: (1, delta_order * n_mfcc, time_steps)
        data_tensor = torch.from_numpy(data).float().unsqueeze(0)
        
        # 5. 获取标签
        label = int(row['target'])
        label_tensor = torch.tensor(label).long()
        
        return data_tensor, label_tensor
    
if __name__ == "__main__":
    # --- 简单的单元测试 ---
    from torch.utils.data import DataLoader

    print("="*60)
    print("Dataset.py 单元测试")
    print("="*60)

    # 1. 设置测试路径
    test_config_name = "baseline" 
    test_meta_path = PROJECT_ROOT / "data" / "raw" / "meta" / "esc50.csv"
    test_processed_dir = PROJECT_ROOT / "data" / "processed" / test_config_name
    
    if not test_meta_path.exists() or not test_processed_dir.exists():
        print("✗ 错误: 未找到以下某个文件")
        print(f"Meta Path: {test_meta_path}")
        print(f"Processed_dir: {test_processed_dir}")
        print("请先运行 src/utils/preprocess.py")
        sys.exit(1)

    # 2. 测试不同的 Delta 阶数
    expected_len = 1 * 400 # 此处根据后续测试的折数修改
    expected_n_mfcc = 13 # 此处根据实际 mfcc 特征维度修改
    expected_time_steps = 216 # 此处根据实际帧数修改
    test_orders = [0, 1, 2]

    for order in test_orders:
        print(f"\n形状测试 Delta Order = {order}")
        
        try:
            ds = ESC50Dataset(
                meta_path=test_meta_path,
                processed_dir=test_processed_dir,
                folds=[1],
                delta_order=order,
                delta_width=9
            )
            
            print(f"\t数据集样本数: {len(ds)}") # 预期为 (折的个数 * 400)
            data, label = ds[0]
            print(f"\t输出 Tensor 形状: {tuple(data.shape)}")
            print(f"\t对应标签 ID: {label}")
            
            expected_height = expected_n_mfcc * (1 + order)

            assert len(ds) == expected_len, f"数据集样本数预期为 {expected_len}, 实际为 {len(ds)}"
            assert data.ndim == 3, "输出必须是 3D Tensor (Channel, Freq, Time)"
            assert data.shape[0] == 1, f"Channel 维度预期为 1, 实际为 {data.shape[0]}"
            assert data.shape[1] == expected_height, \
                f"特征维度预期为 {expected_height}, 实际为 {data.shape[1]}"
            assert data.shape[2] == expected_time_steps, f"帧数预期为 {expected_time_steps}, 实际为 {data.shape[2]}"
            
            print("✓ 形状检查通过")

        except Exception as e:
            print(f"✗ 形状测试失败: {e}")
            # 打印详细报错方便排查
            import traceback
            traceback.print_exc()

    # 3. 测试 DataLoader 集成 (模拟 Batch 读取)
    batch_size = 32
    print("\nDataLoader 批量读取测试")
    try:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        batch_data, batch_labels = next(iter(loader))
        
        print(f"\tBatch Data Shape: {tuple(batch_data.shape)}")
        print(f"\tBatch Labels: {batch_labels}")

        assert batch_data.ndim == 4, "输出必须是 4D Tensor (Batch_size, Channel, Freq, Time)"
        assert batch_data.shape[0] == batch_size, f"批次大小预期为 {batch_size}, 实际为 {batch_data.shape[0]}"
        assert batch_data.shape[1] == 1, f"Channel 维度预期为 1, 实际为 {batch_data.shape[1]}"
        assert batch_data.shape[2] == expected_height, \
            f"特征维度预期为 {expected_height}, 实际为 {batch_data.shape[2]}"
        assert batch_data.shape[3] == expected_time_steps, f"帧数预期为 {expected_time_steps}, 实际为 {batch_data.shape[3]}"

        print("✓ DataLoader 测试通过")
        
    except Exception as e:
        print(f"✗ DataLoader 测试失败: {e}")

    print("\n" + "="*60)
    print("所有测试结束")