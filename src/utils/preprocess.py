"""
数据预处理脚本

使用前请先在项目根目录下建立 data 文件夹，并在该文件夹下建立 raw 文件夹
将 ESC-50 数据集的 audio 和 meta 文件夹复制到 data/raw 下

功能描述:
    该脚本用于批量处理 ESC-50 数据集的原始 WAV 音频文件。
    它会读取元数据 CSV，加载音频，利用手写的 DSP Core 提取 MFCC 特征，
    并将生成的特征矩阵保存为 .npy 文件，以便后续训练加速。

核心依赖:
    - numpy: 矩阵运算
    - pandas: 读取 CSV 元数据
    - librosa: 读取 WAV 文件和重采样
    - src.dsp_core: 手写 DSP 算法库

使用示例:
    # 使用默认参数
    python src/utils/preprocess.py 
    # 自定义参数
    python src/utils/preprocess.py --config_name name --n_fft 4096 --hop_length 1024
"""

import sys
import argparse
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

# 将 src 加入到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.dsp_core.mfcc import mfcc

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ESC-50 音频特征提取预处理工具")
    
    # 参数配置
    parser.add_argument('--config_name', type=str, default='baseline', 
                        help='配置名称')
    parser.add_argument('--raw_dir', type=str, default='data/raw', 
                        help='原始 ESC-50 数据集根目录 (包含 meta/ 和 audio/)')
    parser.add_argument('--sr', type=int, default=22050, 
                        help='目标采样率 (Hz)')
    parser.add_argument('--duration', type=float, default=5.0, 
                        help='音频目标时长 (秒)')
    parser.add_argument('--n_mfcc', type=int, default=13, 
                        help='MFCC 特征维度')
    parser.add_argument('--n_fft', type=int, default=2048, 
                        help='FFT 窗口大小')
    parser.add_argument('--hop_length', type=int, default=512, 
                        help='STFT 步长')
    
    return parser.parse_args()

def process_audio(file_path, args):
    """
    读取单个音频并提取特征
    """
    # 1. 加载音频 (使用 librosa 读取和重采样)
    y, _ = librosa.load(file_path, sr=args.sr, duration=args.duration)
    
    # 2. 长度统一
    current_samples = len(y)
    target_samples = int(args.sr * args.duration)
    if current_samples < target_samples:
        # 补零 (Padding)
        padding = target_samples - current_samples
        y = np.pad(y, (0, padding), mode='constant')
    elif current_samples > target_samples:
        # 裁剪 (Cropping)
        y = y[:target_samples]
        
    # 3. 调用手写 DSP Core 提取特征
    features = mfcc(
        y=y,
        sr=args.sr,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )
    
    return features.astype(np.float32)

def main():
    args = parse_args()
    
    # 路径检查
    raw_path = PROJECT_ROOT / args.raw_dir
    audio_dir = raw_path / "audio"
    meta_csv = raw_path / "meta" / "esc50.csv"
    
    if not audio_dir.exists() or not meta_csv.exists():
        print(f"✗ 错误: 在 {args.raw_dir} 下未找到 audio/ 目录或 meta/esc50.csv 文件。")
        sys.exit(1)
        
    # 输出路径
    out_dir = PROJECT_ROOT / "data" / "processed" / args.config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"开始预处理: {args.config_name}")
    print(f"参数配置: SR={args.sr}, N_MFCC={args.n_mfcc}, N_FFT={args.n_fft}, HOP={args.hop_length}")
    print(f"输入目录: {audio_dir}")
    print(f"输出目录: {out_dir}")
    print("="*60)
    
    # 读取元数据
    df = pd.read_csv(meta_csv)
    total_files = len(df)
    
    success_count = 0
    fail_count = 0
    
    for _, row in tqdm(df.iterrows(), total=total_files, desc="Processing"):
        filename = row['filename']
        input_path = audio_dir / filename
        output_name = filename.replace('.wav', '.npy')
        output_path = out_dir / output_name
        
        # 跳过已处理的音频文件
        if output_path.exists(): continue

        try:
            mfcc_matrix = process_audio(input_path, args)
            np.save(output_path, mfcc_matrix)
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            print(f"\n! ! ! 处理失败: {filename} - {str(e)}")
            
    print("\n" + "="*60)
    print(f"预处理完成")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"数据已保存至: {out_dir}")
    print("="*60)

if __name__ == "__main__":
    main()