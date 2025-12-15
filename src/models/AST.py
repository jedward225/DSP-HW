import torch.nn as nn
from transformers import ASTModel
from torchlibrosa import SpecAugmentation
from audioset_tagging_cnn.pytorch_utils import do_mixup

class AudioAST(nn.Module):
    def __init__(self, num_classes=50):
        super(AudioAST, self).__init__()
        
        self.backbone = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        target_seq_len = 590 
        
        # 获取当前的位置编码参数
        current_pos_embed = self.backbone.embeddings.position_embeddings
        
        # 如果当前的长度大于我们需要的目标长度，进行裁剪
        if current_pos_embed.shape[1] > target_seq_len:
            # 这是一个 Parameter，需要重新封装
            # 我们直接取前 590 个位置 (包含前面的 Token 和前半段时间的 Patch)
            new_pos_embed = current_pos_embed[:, :target_seq_len, :]
            
            self.backbone.embeddings.position_embeddings = nn.Parameter(new_pos_embed)
            
            # 同时更新 config 中的 max_length，防止潜在的报错
            self.backbone.config.max_length = target_seq_len

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=16, freq_stripes_num=2
        )

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )

    def normalize(self, x):
        # AST 官方归一化参数
        AST_MEAN = -4.2677393
        AST_STD = 4.5689974
        return (x - AST_MEAN) / (AST_STD * 2)

    def forward(self, x, mixup_lambda=None):
        x = x.transpose(2, 3)
        x = self.normalize(x)

        if self.training:
            x = self.spec_augmenter(x)
        
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        # 转换为 AST 期望输入形状 (Batch, Time, Freq)
        x = x.squeeze(1) 

        outputs = self.backbone(input_values=x)
        embeddings = outputs.pooler_output 
        logits = self.classifier(embeddings)
        
        return logits