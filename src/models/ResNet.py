import sys
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from pathlib import Path
from torchlibrosa import SpecAugmentation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from audioset_tagging_cnn.pytorch_utils import do_mixup

def normalize(x):
    """
    Instance Normalization: 对每个样本单独进行标准化
    x: shape: (Batch, Channels, Freq, Time)
    """
    # 在 (Freq, Time) 维度上求均值和标准差
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + 1e-8)

class AudioResNet(nn.Module):
    def __init__(self, num_classes=50, model_type="resnet18", chennels=3, pretrained=True):
        """
        适用于音频 MFCC 分类的 ResNet 模型
        
        Args:
            model_type (str): 'resnet18' 或 'resnet34'
            num_classes (int): 分类数量
            pretrained (bool): 是否使用 ImageNet 预训练权重
        """
        super(AudioResNet, self).__init__()
        
        if model_type == "ResNet18":
            base_model = resnet18
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        elif model_type == "ResNet34":
            base_model = resnet34
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = base_model(weights=weights)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        # 适配输入
        if chennels != 3:
            original_conv1 = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                in_channels=chennels,
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False
            )
            if pretrained:
                with torch.no_grad():
                    self.model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

        # 修改输出层
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, mixup_lambda=None):
        x = x.transpose(2, 3)
        x = normalize(x)

        if self.training:
            x = self.spec_augmenter(x)
        
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        return self.model(x)