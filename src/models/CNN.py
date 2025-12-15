import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from audioset_tagging_cnn.models import Cnn14
from audioset_tagging_cnn.pytorch_utils import do_mixup

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class AudioCNN14(Cnn14):
    def __init__(self, classes_num=50, mel_bins=64):
        super(AudioCNN14, self).__init__(sample_rate=32000, 
                                           window_size=1024, 
                                           hop_size=320, 
                                           mel_bins=mel_bins, 
                                           fmin=50.0,
                                           fmax=14000.0, 
                                           classes_num=classes_num)
        
        # 替换为空壳，节省内存
        self.spectrogram_extractor = Identity()
        self.logmel_extractor = Identity()
        
        # 确保 BN0 匹配 Mel Bins
        if mel_bins != 64:
            self.bn0 = nn.BatchNorm2d(mel_bins)

    def forward(self, x, mixup_lambda=None):
        # 输入形状: (Batch, 1, N_mels, Time)
        # 更改为期望形状: (Batch, 1, Time, N_mels)
        x = x.transpose(2, 3)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_audioset(x) # 去除 Sigmoid

        return clipwise_output