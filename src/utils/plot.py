import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(confusion, save_dir):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(20, 16))
    sns.heatmap(confusion, annot=False, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_path = os.path.join(save_dir, f'confusion_matrix.png')
    plt.savefig(save_path, dpi=300)
    print(f"混淆矩阵已保存至 {save_path}")
    plt.close()

def plot_loss_curve(train_losses, val_losses, save_dir):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, f'loss_curve.png')
    plt.savefig(save_path, dpi=300)
    print(f"损失曲线已保存至 {save_path}")
    plt.close()

def plot_acc_curve(accuracy, save_dir):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy, label='Accuracy')
    plt.title(f'Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, f'acc_curve.png')
    plt.savefig(save_path, dpi=300)
    print(f"准确率曲线已保存至 {save_path}")
    plt.close()