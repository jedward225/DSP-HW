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
from src.dsp_core.mfcc import mfcc, delta

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ESC-50 音频特征提取预处理工具")
    
    # 参数配置
    parser.add_argument('--config_name', type=str, default='baseline', 
                        help='配置名称')
    parser.add_argument('--raw_dir', type=str, default='data/raw', 
                        help='原始 ESC-50 数据集根目录 (包含 meta/ 和 audio/)')
    parser.add_argument('--duration', type=float, default=5.0, 
                        help='音频目标时长 (秒)')
    parser.add_argument('--log_mel', action='store_true',
                        help='开启计算 Log Mel Spectrogram')
    parser.add_argument('--mfcc', action='store_true',
                        help='开启计算 MFCC (如果开启，会根据 delta_order 计算 Delta)')
    parser.add_argument('--delta_order', type=int, default=2,
                        help='delta 的阶数')
    parser.add_argument('--sr', type=int, default=22050, 
                        help='目标采样率 (Hz)')
    parser.add_argument('--preemphasis', type=float, default=0.97, 
                        help='预加重系数')
    parser.add_argument('--n_mels', type=int, default=64,
                        help='LOG_MEL_SPEC 特征维度')
    parser.add_argument('--n_mfcc', type=int, default=13, 
                        help='MFCC 特征维度')
    parser.add_argument('--n_fft', type=int, default=2048, 
                        help='FFT 窗口大小')
    parser.add_argument('--win_length', type=int, default=2048, 
                        help='FFT 窗口大小')
    parser.add_argument('--hop_length', type=int, default=512, 
                        help='STFT 步长')
    parser.add_argument('--window', type=str, default='hann', 
                        choices=['hann', 'hamming', 'blackman', 'bartlett', 'tukey'],
                        help='STFT 窗函数类型')
    parser.add_argument('--fmin', type=float, default=0.0)
    parser.add_argument('--fmax', type=float, default=None)
    
    return parser.parse_args()

def process_audio(file_path, args, log_mel):
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
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        window=args.window,
        win_length=args.win_length,
        preemphasis=args.preemphasis,
        fmin=args.fmin,
        fmax=args.fmax,
        log_mel=log_mel
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
    
    # 预先创建必要的文件夹
    if args.log_mel:
        (out_dir / "log_mel_spec").mkdir(parents=True, exist_ok=True)
    if args.mfcc:
        (out_dir / "mfcc").mkdir(parents=True, exist_ok=True)
        for i in range(args.delta_order):
            (out_dir / f"delta_{i+1}").mkdir(parents=True, exist_ok=True)

    print("="*90)
    print(f"开始预处理: {args.config_name}")
    print(f"任务模式: LogMel={args.log_mel}, MFCC={args.mfcc} (Delta Order={args.delta_order if args.mfcc else 0})")
    print(f"参数配置: SR={args.sr}, PREEMPHASIS={args.preemphasis}, WINDOW={args.window}, WIN_LENGTH={args.win_length}")
    print(f"\tN_MELS = {args.n_mels}, N_MFCC={args.n_mfcc}, N_FFT={args.n_fft}, HOP_LENGTH={args.hop_length}")
    print(f"输入目录: {audio_dir}")
    print(f"输出目录: {out_dir}")
    print("="*90)
    
    if not args.log_mel and not args.mfcc:
        print("! 警告: 未指定 --log_mel 或 --mfcc，本次运行将不执行任何计算。")
        return

    # 读取元数据
    df = pd.read_csv(meta_csv)
    total_files = len(df)
    
    success_count = 0
    fail_count = 0
    
    for _, row in tqdm(df.iterrows(), total=total_files, desc="Processing"):
        filename = row['filename']
        input_path = audio_dir / filename
        output_name = filename.replace('.wav', '.npy')

        try:
            # === 1. Log Mel Spectrogram 部分 ===
            if args.log_mel:
                log_mel_dir = out_dir / "log_mel_spec"
                output_path_logmel = log_mel_dir / output_name
                
                if not output_path_logmel.exists(): 
                    log_mel_spec = process_audio(input_path, args, log_mel=True)
                    np.save(output_path_logmel, log_mel_spec)

            # === 2. MFCC & Delta 部分 ===
            if args.mfcc:
                mfcc_dir = out_dir / "mfcc"
                output_path_mfcc = mfcc_dir / output_name
                
                if output_path_mfcc.exists():
                    mfcc_matrix = np.load(output_path_mfcc)
                else:
                    mfcc_matrix = process_audio(input_path, args, log_mel=False)
                    np.save(output_path_mfcc, mfcc_matrix)
                
                # 计算 Delta
                current = mfcc_matrix
                for i in range(args.delta_order):
                    delta_dir = out_dir / f"delta_{i+1}"
                    output_path_delta = delta_dir / output_name
                    
                    if output_path_delta.exists():
                        delta_matrix = np.load(output_path_delta)
                    else:
                        delta_matrix = delta(current)
                        np.save(output_path_delta, delta_matrix)
                    current = delta_matrix

            success_count += 1
            
        except Exception as e:
            fail_count += 1
            print(f"\n处理失败: {filename} - {str(e)}")
            
    print("\n" + "="*90)
    print(f"预处理完成")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"数据已保存至: {out_dir}")
    print("="*90)

if __name__ == "__main__":
    main()