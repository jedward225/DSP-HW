import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

class ESC50Dataset(Dataset):
    def __init__(self, meta_path, processed_dir, folds, log_mel=False, delta_order=0):
        """
        Args:
            meta_path (str): esc50.csv 的路径
            processed_dir (str): 存放 .npy 文件的文件夹路径
            folds (list): 需要加载的折
            log_mel (bool): 是否使用对数梅尔谱
            delta_order (int): 差分阶数
        """
        processed_dir = Path(processed_dir)
        self.data_dir = [processed_dir / ("log_mel_spec" if log_mel else "mfcc")]
        for i in range(0 if log_mel else delta_order):
            self.data_dir.append(processed_dir / f"delta_{i+1}")
        
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

        # 2. 加载 MFCC 矩阵
        channels = []
        for dir in self.data_dir:
            data_path = os.path.join(dir, npy_name)
            data = np.load(data_path)
            channels.append(data)            
        
        # 3. 转换为 Tensor
        data = np.stack(channels, axis=0)
        data_tensor = torch.from_numpy(data).float()
        
        # 4. 获取标签
        label = int(row['target'])
        label_tensor = torch.tensor(label).long()
        
        return data_tensor, label_tensor