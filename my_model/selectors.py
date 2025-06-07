import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from efficientnet_pytorch import EfficientNet

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 50):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class WinnerSelector(nn.Module):
    def __init__(
        self,
        context_size=5,
        encoder_rgb_name='efficientnet-b0',
        encoding_size=512,
        ctx_mha_heads=8,
        ctx_mha_layers=2,
        ctx_ff_dim_factor=4,
        mask_seq_mha_heads=8,
        mask_seq_mha_layers=2,
        mask_seq_ff_dim_factor=4,
        fusion_hidden_dim_factor=1,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.N = context_size + 1
        self.D = encoding_size

        # 定义RGB和Mask特征图的扁平化长度
        # EfficientNet-B0对于128x160输入，特征图大小是4x5
        self.spatial_flatten_len = 4 * 5 # 统一为 20

        # PositionalEncoding 的最大序列长度
        self.max_total_seq_len = self.N * self.spatial_flatten_len # N * 20


        if self.D % ctx_mha_heads != 0:
            raise ValueError("encoding_size must be divisible by ctx_mha_heads")
        if self.D % mask_seq_mha_heads != 0:
            raise ValueError("encoding_size must be divisible by mask_seq_mha_heads")

        # RGB encoder
        self.rgb_cnn_encoder = EfficientNet.from_name(encoder_rgb_name, include_top=False)
        in_feats_rgb = 1280
        self.compress_rgb = nn.Conv2d(in_feats_rgb, self.D, kernel_size=1)

        # Mask encoder - 调整第三个 Conv2d 的 stride，使其输出 4x5
        self.mask_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True), # 128x160 -> 64x80
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True), # 64x80 -> 32x40
            nn.Conv2d(32, self.D, kernel_size=3, stride=8, padding=1), nn.ReLU(inplace=True), # 32x40 -> 4x5
        )
        # 注意：这里假设输入mask的 H,W 是 128, 160。如果不是，需要重新计算 stride/padding 来达到 4x5。

        # 位置编码的最大序列长度
        self.pos_encoder_n_frames = PositionalEncoding(self.D, max_seq_len=self.max_total_seq_len)

        context_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D, nhead=ctx_mha_heads, dim_feedforward=self.D * ctx_ff_dim_factor,
            dropout=dropout_rate, batch_first=True, norm_first=True
        )
        self.rgb_context_transformer = nn.TransformerEncoder(context_encoder_layer, num_layers=ctx_mha_layers)

        mask_seq_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D, nhead=mask_seq_mha_heads, dim_feedforward=self.D * mask_seq_ff_dim_factor,
            dropout=dropout_rate, batch_first=True, norm_first=True
        )
        self.mask_sequence_transformer = nn.TransformerEncoder(mask_seq_encoder_layer, num_layers=mask_seq_mha_layers)

        self.fusion_predictor = nn.Sequential(
            nn.LayerNorm(self.D * 2),
            nn.Linear(self.D * 2, self.D * fusion_hidden_dim_factor), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.D * fusion_hidden_dim_factor, 1)
        )

    def forward(self, rgb_imgs, mask_imgs, invalid):
        B, _, H_rgb, W_rgb = rgb_imgs.shape
        P = mask_imgs.shape[1]

        # RGB 路径
        flat_rgb = rgb_imgs.view(B * self.N, 3, H_rgb, W_rgb)
        rgb_feats = self.rgb_cnn_encoder.extract_features(flat_rgb)  # (B*N, C, 4, 5)
        rgb_feats = self.compress_rgb(rgb_feats)  # (B*N, D, 4, 5)

        # 展平空间维度并转置，然后重塑
        rgb_feats = rgb_feats.flatten(2).transpose(1, 2)  # (B*N, spatial_flatten_len, D) -> (B*N, 20, D)
        rgb_feats = rgb_feats.reshape(B, self.N * self.spatial_flatten_len, self.D)  # (B, N*20, D)
        
        rgb_feats_pe = self.pos_encoder_n_frames(rgb_feats)
        rgb_out = self.rgb_context_transformer(rgb_feats_pe)
        rgb_context_summary = rgb_out.mean(dim=1)

        # Mask 路径
        flat_mask = mask_imgs.view(B * P * self.N, 1, mask_imgs.shape[3], mask_imgs.shape[4]).float()
        # mask_feats 现在会输出 (B*P*N, D, 4, 5)
        mask_feats = self.mask_cnn_encoder(flat_mask) 

        # 展平空间维度并转置
        mask_feats = mask_feats.flatten(2).transpose(1, 2)  # (B*P*N, spatial_flatten_len, D) -> (B*P*N, 20, D)
        
        # 重塑，现在可以直接用统一的 spatial_flatten_len
        mask_feats = mask_feats.reshape(B * P, self.N * self.spatial_flatten_len, self.D)  # (B*P, N*20, D)
        
        mask_feats_pe = self.pos_encoder_n_frames(mask_feats)
        mask_out = self.mask_sequence_transformer(mask_feats_pe)
        person_mask_summary = mask_out.mean(dim=1).view(B, P, self.D)

        # 融合与预测
        rgb_context_expanded = rgb_context_summary.unsqueeze(1).expand(-1, P, -1)
        fused = torch.cat([rgb_context_expanded, person_mask_summary], dim=-1)
        logits = self.fusion_predictor(fused.reshape(B * P, -1)).view(B, P)
        logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)

        return logits