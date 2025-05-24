import torch
import torch.nn as nn
import math
from typing import Optional, List
from efficientnet_pytorch import EfficientNet


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()

        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer('pos_enc', pos_enc.unsqueeze(0))

    def forward(self, x):
        return x + self.pos_enc[:, :x.size(1), :]


class GazeSelector(nn.Module):
    def __init__(
            self,
            context_size: int = 5,
            encoder: Optional[str] = 'efficientnet-b0',
            encoding_size: Optional[int] = 512,
            mha_num_attention_heads: Optional[int] = 4,
            mha_num_attention_layers: Optional[int] = 2,
            mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:

        super(GazeSelector, self).__init__()
        self.N = context_size + 1
        D = encoding_size

        # RGB Encoder
        self.rgb_encoder = EfficientNet.from_name(encoder, in_channels=3)
        in_feats = self.rgb_encoder._fc.in_features
        self.compress_rgb = nn.Linear(in_feats, D) if in_feats != D else nn.Identity()

        # Mask Encoder
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, D)
        )

        # Postional & Type Embedding
        max_seq = 2 * self.N
        self.pos_enc = PositionalEncoding(D, max_seq)
        self.type_emb = nn.Embedding(2, D)  # 0 = RGB, 1 = MASK

        # Transformer Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=mha_num_attention_heads,
            dim_feedforward=D * mha_ff_dim_factor,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=mha_num_attention_layers)

        # Prediction Head
        self.predictor = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, 1)
        )

    def forward(
        self,
        rgb_imgs: torch.Tensor,
        mask_imgs: torch.Tensor,
        invalid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rgb_imgs: (B, 3*N, H, W)
            mask_imgs: (B, P, N, H, W)
            exist: (B, P) bool, True stands for padding
        Returns:
            logits: (B, P)
        """
        B, _, H, W = rgb_imgs.shape
        P = mask_imgs.shape(1)
        D = self.head[1].in_features  # encoding size
        
        # Encode RGB: (B, 3*N ,H, W) -> (B, N, D)
        flat_rgb = rgb_imgs.view(B*self.N, 3, H, W)
        feat_map = self.rgb_encoder.extract_features(flat_rgb) 
        pooled = self.rgb_encoder._avg_pooling(feat_map).flatten(1)
        rgb_feat = self.compress_rgb(pooled).view(B, self.N, D) 

        # Encode Masks: (B*P*N, 1, H, W)
        flat_mask = mask_imgs.view(B*P*self.N, H, W).unsqueeze(1).float()
        mfeat = self.mask_encoder(flat_mask)  # (B*P*N, D)
        mask_feat = mfeat.view(B, P, self.N, D)
        
        # Broadcast RGB to per-person: (B,1,N,D)->(B,P,N,D)
        rgb_feat = rgb_feat.unsqueeze(1).expand(-1, P, -1, -1)

        # Concatenate along time: (B,P,2N,D)
        tokens = torch.stack([rgb_feat, mask_feat], dim=3)  # (B, P, N, 2, D)
        tokens = tokens.reshape(B, P, 2 * self.N, D)  # (B, P, 2N, D)
        tokens = tokens.view(B * P, 2 * self.N, D)

        # Position & Type Embedding
        # postion embedding
        pos_ids = torch.arange(2 * self.N, device=tokens.device)
        tokens = tokens + self.pos_enc.pos_enc[:, : 2 * self.N, :]
        # type embedding
        type_ids = (pos_ids % 2).long()
        tokens = tokens + self.type_emb(type_ids)

        out = self.decoder(tokens)

        # 8) Pooling & Predict -> (B*P,1)
        pooled = out.mean(dim=1)
        logits = self.head(pooled).view(B, P)  # (B,P)

        # 9) Set padding items -inf
        logits = logits.masked_fill(invalid, float('-inf'))
        return logits