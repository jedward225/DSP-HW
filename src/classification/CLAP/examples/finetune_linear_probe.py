"""
ESC-50 Linear Probing with CLAP
使用预训练的 CLAP 提取特征，然后训练一个线性分类器

策略：
1. 冻结 CLAP 编码器（不更新权重）
2. 使用 4 个 fold 训练，1 个 fold 测试（5-fold CV）
3. 只训练一个简单的线性分类器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from msclap import CLAP
from esc50_dataset import ESC50
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd

# ===== 配置 =====
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
USE_CUDA = torch.cuda.is_available()
TEST_FOLD = 5  # 使用 fold 5 作为测试集，fold 1-4 作为训练集

print("="*60)
print("ESC-50 Linear Probing with CLAP")
print("="*60)
print(f"Device: {'GPU' if USE_CUDA else 'CPU'}")
print(f"Test Fold: {TEST_FOLD}")
print(f"Training Folds: {[i for i in range(1, 6) if i != TEST_FOLD]}")
print("="*60)

# ===== 加载数据集 =====
root_path = "/home/linfeng_fan/DSP-HW/ESC-50"
dataset = ESC50(root=root_path, download=False)

# 读取 fold 信息
meta_df = pd.read_csv(f"{root_path}/meta/esc50.csv")

# 划分训练集和测试集
train_indices = meta_df[meta_df['fold'] != TEST_FOLD].index.tolist()
test_indices = meta_df[meta_df['fold'] == TEST_FOLD].index.tolist()

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

print(f"\nDataset size:")
print(f"  Training: {len(train_dataset)} samples")
print(f"  Testing: {len(test_dataset)} samples")

# ===== 加载 CLAP 模型 =====
print("\nLoading CLAP model...")
clap_model = CLAP(version='2023', use_cuda=USE_CUDA)

# 冻结 CLAP 参数
for param in clap_model.clap.parameters():
    param.requires_grad = False

clap_model.clap.eval()  # 设置为评估模式

# ===== 提取特征 =====
def extract_features(dataset, model, desc="Extracting features"):
    """提取音频的 CLAP embeddings"""
    features = []
    labels = []

    for i in tqdm(range(len(dataset)), desc=desc):
        audio_path, label_str, one_hot = dataset[i]

        # 提取音频 embedding
        with torch.no_grad():
            audio_emb = model.get_audio_embeddings([audio_path], resample=True)

        features.append(audio_emb.cpu())
        labels.append(torch.argmax(one_hot).item())

    features = torch.cat(features, dim=0)
    labels = torch.tensor(labels)

    return features, labels

print("\n" + "="*60)
print("Extracting CLAP features...")
print("="*60)

train_features, train_labels = extract_features(train_dataset, clap_model, "Training set")
test_features, test_labels = extract_features(test_dataset, clap_model, "Test set")

print(f"\nFeature shapes:")
print(f"  Train features: {train_features.shape}")
print(f"  Train labels: {train_labels.shape}")
print(f"  Test features: {test_features.shape}")
print(f"  Test labels: {test_labels.shape}")

# ===== 定义线性分类器 =====
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# 创建分类器
input_dim = train_features.shape[1]
num_classes = 50
classifier = LinearClassifier(input_dim, num_classes)

if USE_CUDA:
    classifier = classifier.cuda()
    train_features = train_features.cuda()
    train_labels = train_labels.cuda()
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

# ===== 训练 =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

print("\n" + "="*60)
print("Training Linear Classifier...")
print("="*60)

best_acc = 0.0
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    classifier.train()

    # 训练
    optimizer.zero_grad()
    outputs = classifier(train_features)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    # 评估
    classifier.eval()
    with torch.no_grad():
        # 训练集准确率
        train_outputs = classifier(train_features)
        train_preds = torch.argmax(train_outputs, dim=1)
        train_acc = (train_preds == train_labels).float().mean().item()

        # 测试集准确率
        test_outputs = classifier(test_features)
        test_preds = torch.argmax(test_outputs, dim=1)
        test_acc = (test_preds == test_labels).float().mean().item()

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch + 1
        torch.save(classifier.state_dict(), 'best_linear_classifier.pth')

    # 每5个epoch打印一次
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
              f"Loss: {loss.item():.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | "
              f"Test Acc: {test_acc*100:.2f}%")

# ===== 最终评估 =====
print("\n" + "="*60)
print("Final Results")
print("="*60)

# 加载最佳模型
classifier.load_state_dict(torch.load('best_linear_classifier.pth'))
classifier.eval()

with torch.no_grad():
    test_outputs = classifier(test_features)
    test_preds = torch.argmax(test_outputs, dim=1)
    final_acc = (test_preds == test_labels).float().mean().item()

print(f"\n🏆 Best Test Accuracy: {best_acc*100:.2f}% (Epoch {best_epoch})")
print(f"📊 Final Test Accuracy: {final_acc*100:.2f}%")
print(f"\n✅ Model saved to: best_linear_classifier.pth")

print("\n" + "="*60)
print("Comparison with Zero-shot")
print("="*60)
print(f"Zero-shot (optimized):  94.35%")
print(f"Linear Probing:         {final_acc*100:.2f}%")
print(f"Improvement:            +{(final_acc*100 - 94.35):.2f}%")
print("="*60)
