import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.models.ResNet import AudioResNet
from src.models.CNN import AudioCNN14
from src.models.AST import AudioAST
from src.utils.dataset import ESC50Dataset
from src.utils.plot import plot_confusion_matrix, plot_loss_curve, plot_acc_curve

def load_pretrained_weights(model, checkpoint_path):
    """
    加载 PANNs 预训练权重，自动跳过不匹配的层
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_dict = checkpoint['model']

    model_dict = model.state_dict()
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        # 跳过计算 log_mel_spectrum 的层
        if 'spectrogram_extractor' in k or 'logmel_extractor' in k:
            continue

        # 如果该层在模型中不存在，跳过
        if k not in model_dict:
            continue
            
        # 检查形状是否匹配
        if v.shape != model_dict[k].shape:
            continue

        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model

def train_one_epoch(model, loader, optimizer, CONFIG):
    model.train()
    epoch_loss = 0.0

    for inputs, labels in loader:
        inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])

        # 生成混合系数 lambda (Beta 分布)
        lam = np.random.beta(1.0, 1.0) 
        
        # 构建 mixup_lambda 张量
        batch_size = inputs.size(0)
        mixup_lambda = torch.zeros(batch_size).to(CONFIG['device'])
        mixup_lambda[0::2] = lam      # 偶数索引: lam
        mixup_lambda[1::2] = 1 - lam  # 奇数索引: 1-lam
        
        # 3. 前向传播, outputs 的 Batch Size 会变成原来的一半
        outputs = model(inputs, mixup_lambda=mixup_lambda) 
        
        # 4. 计算 Loss
        targets_a = labels[0::2] # 偶数位标签
        targets_b = labels[1::2] # 奇数位标签
        loss = lam * CONFIG['criterion'](outputs, targets_a) + (1 - lam) * CONFIG['criterion'](outputs, targets_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)

    epoch_loss = epoch_loss / len(loader.dataset)
    return epoch_loss

def validate(model, loader, CONFIG):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(inputs)
            loss = CONFIG['criterion'](outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / len(loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    return val_loss, val_acc

def train(train_loader, val_loader, CONFIG):
    # 获取模型并冻结分类层以外的层, 先训练 50 个 epoch
    if CONFIG['model_type'] in ["ResNet18", "ResNet34"]:
        model = AudioResNet(num_classes=50, model_type=CONFIG['model_type'], 
                            chennels=(1 if CONFIG['data_type'] == 'log_mel_spec' else 3), pretrained=True)
        model = model.to(CONFIG["device"])
        for name, param in model.named_parameters():
            if "fc" in name: continue
            param.requires_grad = False
    
    elif CONFIG['model_type'] == "CNN14":
        model = AudioCNN14(classes_num=50)
        checkpoint_path = PROJECT_ROOT / "audioset_tagging_cnn" / "Cnn14_mAP=0.431.pth"
        model = load_pretrained_weights(model, checkpoint_path)
        model = model.to(CONFIG['device'])
        for name, param in model.named_parameters():
            if "fc_audioset" in name: continue
            param.requires_grad = False
    
    elif CONFIG['model_type'] == "AST":
        model = AudioAST(num_classes=50)
        model = model.to(CONFIG['device'])
        for name, param in model.named_parameters():
            if "classifier" in name: continue
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=CONFIG['lr'], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    best_model_state = None
    best_val_acc = 0.0
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(CONFIG["epochs"]):
        if epoch == 50:
            # 解冻所有层
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"]-50, eta_min=1e-6)

        train_loss = train_one_epoch(model, train_loader, optimizer, CONFIG)
        val_loss, val_acc = validate(model, val_loader, CONFIG)
        scheduler.step()
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch: {epoch+1}, train_loss: {train_loss}\n\tval_loss: {val_loss}, val_acc: {val_acc}")
        
        if val_acc > best_val_acc:
          best_model_state = deepcopy(model.state_dict())
          best_val_acc = val_acc
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_acc, train_loss_history, val_loss_history, val_acc_history

def test(model, loader, CONFIG):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    confusion = confusion_matrix(all_labels, all_preds)
    return accuracy, precision, recall, f1, confusion

if __name__ == "__main__":
    np.random.seed(3407)
    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(3407)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

    processed_dir = PROJECT_ROOT / "data" / "processed"
    meta_path = PROJECT_ROOT / "data" / "raw" / "meta" / "esc50.csv"

    CONFIG = {
        "config_name": "deep_feature",
        "model_type": "ResNet34",
        "data_type": "mfcc",
        "batch_size": 32,
        "epochs": 200,
        "lr": 1e-4,
        "criterion": nn.CrossEntropyLoss(label_smoothing=0.1),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    print("="*90)
    print(f"开始训练 {CONFIG['config_name']}, 模型: {CONFIG['model_type']}, 使用数据: {CONFIG['data_type']}")
    print(f"总轮数: {CONFIG['epochs']}, 初始学习率: {CONFIG['lr']}, 批次大小: {CONFIG['batch_size']}, 损失函数: 交叉熵损失")
    print("="*90)

    data_dir = processed_dir / CONFIG['config_name']
    result_dir = PROJECT_ROOT / "results" / "classification" / CONFIG["config_name"] / CONFIG['model_type'] / CONFIG['data_type']
    os.makedirs(result_dir, exist_ok=True)

    train_dataset = ESC50Dataset(meta_path, data_dir, [1, 2, 3], log_mel=(CONFIG["data_type"]=="log_mel_spec"), delta_order=2)
    val_dataset = ESC50Dataset(meta_path, data_dir, [4], log_mel=(CONFIG["data_type"]=="log_mel_spec"), delta_order=2)
    test_dataset = ESC50Dataset(meta_path, data_dir, [5], log_mel=(CONFIG["data_type"]=="log_mel_spec"), delta_order=2)
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    model, val_acc, train_loss_history, val_loss_history, val_acc_history = train(train_dataloader, val_dataloader, CONFIG)

    print('\n' + "="*90)
    print(f"{CONFIG['config_name']} 训练完成, 验证集准确率: {val_acc}")
    torch.save(model.state_dict(), os.path.join(result_dir, "model.pth"))
    print(f"模型已保存至 {os.path.join(result_dir, 'model.pth')}")
    plot_loss_curve(train_loss_history, val_loss_history, result_dir)
    plot_acc_curve(val_acc_history, result_dir)
    print("="*90)

    print('\n' + "="*90)
    print(f"测试 {CONFIG['config_name']}, 模型: {CONFIG['model_type']}")
    accuracy, precision, recall, f1, confusion = test(model, test_dataloader, CONFIG)
    print(f"准确率: {accuracy}, 精确率: {precision}, 召回率: {recall}, F1: {f1}")
    with open(os.path.join(result_dir, "metric.json"), 'w', encoding='utf-8') as f:
        json.dump({"accuracy": accuracy, "precision": precision, "recall": recall, "F1": f1}, f, indent=4, ensure_ascii=False)
    plot_confusion_matrix(confusion, result_dir)
    print("="*90)