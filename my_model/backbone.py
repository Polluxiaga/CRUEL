import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from efficientnet_pytorch import EfficientNet


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()

        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input based on the seq_len
        return x + self.pos_enc[:, :x.size(1), :]
    

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """ 自定义 TransformerEncoderLayer，返回 attn_weights """
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask, 
            need_weights=True
        )  # 强制输出注意力权重
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights  # 直接返回注意力权重


class base_MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(base_MultiLayerDecoder, self).__init__()
        
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        
        self.sa_layer = CustomTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim_factor * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.sa_decoder = nn.ModuleList([self.sa_layer for _ in range(num_layers)])  # 手动堆叠多层

        # 全连接层
        self.output_layer = nn.ModuleList([
            nn.Linear(seq_len * embed_dim, output_layers[0])] + [
            nn.Linear(output_layers[i], output_layers[i + 1]) for i in range(len(output_layers) - 1)])

    def forward(self, x):
        x = self.positional_encoding(x)
        attn_scores_list = []
        for layer in self.sa_decoder:
            x, attn_weights = layer(x)
            attn_scores_list.append(attn_weights)

        # 计算平均注意力权重
        avg_attention_scores = torch.mean(torch.stack(attn_scores_list, dim=0), dim=0)  # [batch_size, seq_len, seq_len]

        x = x.reshape(x.shape[0], -1)
        for layer in self.output_layer:
            x = F.relu(layer(x))

        return x, avg_attention_scores


class base_model(nn.Module):
    def __init__(
        self,
        method: str = "base",
        context_size: int = 5,
        len_traj_pred: int = 3,
        encoder: Optional[str] = "efficientnet-b0",
        encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        
        super(base_model, self).__init__()
        self.method = method
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.encoding_size = encoding_size
        self.num_action_params = 2
        self.mha_num_attention_heads = mha_num_attention_heads
        self.mha_num_attention_layers = mha_num_attention_layers
        self.mha_ff_dim_factor = mha_ff_dim_factor

        """ # 初始化保存梯度的占位变量
        self._grad_obs_features = None """

        if encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(encoder, in_channels=3)  # context
            self.num_obs_features = self.obs_encoder._fc.in_features

        if self.num_obs_features != self.encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        self.gaze_conv = nn.Conv2d(self.num_obs_features, 1, kernel_size=1)

        self.decoder = None
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_traj_pred * self.num_action_params),
        )

    def forward(self, obs_img: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #  第一次forward时初始化解码器
        if self.decoder is None:
            # 获取一个样本的特征图大小
            with torch.no_grad():
                sample_features = self.obs_encoder.extract_features(obs_img[0:1, 0:3])
                H_feature = sample_features.shape[2]  # H/32
                W_feature = sample_features.shape[3]  # W/32
            
            self.decoder = base_MultiLayerDecoder(
                embed_dim=self.encoding_size,
                seq_len=(self.context_size+1) * H_feature * W_feature,
                output_layers=[256, 128, 64, 32],
                nhead=self.mha_num_attention_heads,
                num_layers=self.mha_num_attention_layers,
                ff_dim_factor=self.mha_ff_dim_factor,
            ).to(obs_img.device)  # 确保在同一设备上

        """ # 每次forward时先清空之前的梯度（重要）
        self._grad_obs_features = None """

        # 将输入拆分为 (context_size+1) 个图像
        obs_img = torch.split(obs_img, 3, dim=1)  # [batch_size, 3, H, W]*(context_size+1)
        obs_img = torch.concat(obs_img, dim=0)     # [batch_size*(context_size+1), 3, H, W]

        """ # 1. 获取所需层特征并注册hook（用于辅助任务）
        def get_requested_features(module, input, output):
            raw_obs_features = output  # [N, channel_num, H/32, W/32]
            raw_obs_features.retain_grad()
            raw_obs_features.register_hook(self._capture_obs_features_grad)
            
            # 重塑为所需维度
            N = raw_obs_features.shape[0]
            batch_size = N // (self.context_size + 1)
            self._raw_obs_features = raw_obs_features.view(
                self.context_size + 1,
                batch_size,
                raw_obs_features.shape[1],  # channel_num
                raw_obs_features.shape[2],  # H/32
                raw_obs_features.shape[3]   # W/32
            )  # [context_size+1, batch_size, channel_num, H/32, W/32]

        # 注册任意层的hook
        requested_layer = list(self.obs_encoder._blocks)[-1]  # 获取H/8分辨率的中间层
        handle = requested_layer.register_forward_hook(get_requested_features) """

        # 2. 正常前向传播，获取最终的全局特征
        obs_features = self.obs_encoder.extract_features(obs_img)  # [N, 1280, H/32, W/32]
        N, C, H, W = obs_features.shape  # N = batch_size * (context_size+1)
        
        if self.method == "cnnaux":
            # 生成gaze_use_map
            gaze_use_map = self.gaze_conv(obs_features)
            gaze_use_map = gaze_use_map.view(self.context_size + 1, -1, 1, H * W).squeeze(2)  # [context_size+1, batch_size, H/32 * W/32]
            gaze_use_map = gaze_use_map.permute(1, 0, 2) # [batch_size, context_size+1, H/32 * W/32]
            gaze_use_map = gaze_use_map.reshape(-1, (self.context_size+1)*H*W) # [batch_size, (context_size+1)*H/32*W/32]

        # 继续原有的处理流程
        obs_encoding = obs_features.permute(0, 2, 3, 1)  # [N, H/32, W/32, 1280]
        obs_encoding = obs_encoding.reshape(N, H*W, C)  # [N, H/32*W/32, 1280]

        if self.obs_encoder._global_params.include_top:
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        
        # 压缩到指定维度
        obs_encoding = self.compress_obs_enc(obs_encoding)  # [N, H/32*W/32, encoding_size]
        
        # 重塑为序列形式
        obs_encoding = obs_encoding.reshape(
            (self.context_size+1, -1, H*W, self.encoding_size)
        )  # [context_size+1, batch_size, H/32*W/32, encoding_size]
        
        # 转置为transformer期望的输入格式
        tokens = obs_encoding.permute(1, 0, 2, 3)  # [batch_size, context_size+1, H/32*W/32, encoding_size]
        batch_size = tokens.shape[0]
        tokens = tokens.reshape(batch_size, (self.context_size+1)*H*W, self.encoding_size)  # [batch_size, (context_size+1)*H/32*W/32, encoding_size]
        
        # 3. Transformer解码器处理
        final_repr, attention_scores = self.decoder(tokens)  # [batch_size, 32]
        action_pred = self.action_predictor(final_repr)
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_traj_pred, self.num_action_params)
        )
        action_pred = torch.cumsum(action_pred, dim=1)  # 将位置增量累计为 waypoints

        """ # 清理hook
        handle.remove() """

        # 返回预测结果和中间特征
        if self.method == "cnnaux":
            return action_pred, attention_scores, gaze_use_map
        else:
            # 如果不是cnnaux方法，则不需要gaze_use_map, 仅返回动作预测和原始观察特征
            return action_pred, attention_scores

    """ @torch.utils.hooks.unserializable_hook
    def _capture_obs_features_grad(self, grad):
        # 保存obs_features的梯度 (在backward时触发)
        self._grad_obs_features = grad """
    

class channel_model(nn.Module):
    def __init__(
        self,
        method: str = "base",
        context_size: int = 5,
        len_traj_pred: int = 3,
        encoder: Optional[str] = "efficientnet-b0",
        encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        
        super(channel_model, self).__init__()
        self.method = method
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.encoding_size = encoding_size
        self.num_action_params = 2
        self.mha_num_attention_heads = mha_num_attention_heads
        self.mha_num_attention_layers = mha_num_attention_layers
        self.mha_ff_dim_factor = mha_ff_dim_factor

        if encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(encoder, in_channels=4)  # context
            self.num_obs_features = self.obs_encoder._fc.in_features

        if self.num_obs_features != self.encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        self.decoder = None
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_traj_pred * self.num_action_params),
        )

    def forward(self, obs_img: torch.tensor, attention) -> Tuple[torch.Tensor, torch.Tensor]:
        #  第一次forward时初始化解码器
        if self.decoder is None:
            # 获取一个样本的特征图大小
            with torch.no_grad():
                sample_rgb = obs_img[0:1, 0:3]
                sample_attn = attention[0:1, 0:1]
                sample_input = torch.cat([sample_rgb, sample_attn], dim=1)
                sample_features = self.obs_encoder.extract_features(sample_input)
                H_feature = sample_features.shape[2]  # H/32
                W_feature = sample_features.shape[3]  # W/32
            
            self.decoder = base_MultiLayerDecoder(
                embed_dim=self.encoding_size,
                seq_len=(self.context_size+1) * H_feature * W_feature,
                output_layers=[256, 128, 64, 32],
                nhead=self.mha_num_attention_heads,
                num_layers=self.mha_num_attention_layers,
                ff_dim_factor=self.mha_ff_dim_factor,
            ).to(obs_img.device)  # 确保在同一设备上

        # 将输入拆分为 (context_size+1) 个图像
        obs_img = torch.split(obs_img, 3, dim=1)  # [batch_size, 3, H, W]*(context_size+1)
        attention = torch.split(attention, 1, dim =1)  # [batch_size, H, W]*(context_size+1)

        #  合并RGB和attention通道
        combined_input=[]
        for rgb, attn in zip(obs_img, attention):
            combined = torch.cat([rgb, attn], dim=1)  # [batch_size, 4, H, W]
            combined_input.append(combined)

        combined_input = torch.concat(combined_input, dim=0)  # [batch_size*(context_size+1), 4, H, W]

        # 使用4通道特征提取
        obs_features = self.obs_encoder.extract_features(combined_input)  # [N, 1280, H/32, W/32]
        N, C, H, W = obs_features.shape  # N = batch_size * (context_size+1)

        # 继续原有的处理流程
        obs_encoding = obs_features.permute(0, 2, 3, 1)  # [N, H/32, W/32, 1280]
        obs_encoding = obs_encoding.reshape(N, H*W, C)  # [N, H/32*W/32, 1280]

        if self.obs_encoder._global_params.include_top:
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        
        # 压缩到指定维度
        obs_encoding = self.compress_obs_enc(obs_encoding)  # [N, H/32*W/32, encoding_size]
        
        # 重塑为序列形式
        obs_encoding = obs_encoding.reshape(
            (self.context_size+1, -1, H*W, self.encoding_size)
        )  # [context_size+1, batch_size, H/32*W/32, encoding_size]
        
        # 转置为transformer期望的输入格式
        tokens = obs_encoding.permute(1, 0, 2, 3)  # [batch_size, context_size+1, H/32*W/32, encoding_size]
        batch_size = tokens.shape[0]
        tokens = tokens.reshape(batch_size, (self.context_size+1)*H*W, self.encoding_size)  # [batch_size, (context_size+1)*H/32*W/32, encoding_size]
        
        # 3. Transformer解码器处理
        final_repr, attention_scores = self.decoder(tokens)  # [batch_size, 32]
        action_pred = self.action_predictor(final_repr)
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_traj_pred, self.num_action_params)
        )
        action_pred = torch.cumsum(action_pred, dim=1)  # 将位置增量累计为 waypoints
        
        return action_pred, attention_scores
    

class catoken_model(nn.Module):
    def __init__(
        self,
        method: str = "base",
        context_size: int = 5,
        len_traj_pred: int = 3,
        encoder: Optional[str] = "efficientnet-b0",
        encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        
        super(catoken_model, self).__init__()
        self.method = method
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.encoding_size = encoding_size
        self.num_action_params = 2
        self.mha_num_attention_heads = mha_num_attention_heads
        self.mha_num_attention_layers = mha_num_attention_layers
        self.mha_ff_dim_factor = mha_ff_dim_factor

        # Separate encoders for RGB and attention
        if encoder.split("-")[0] == "efficientnet":
            self.rgb_encoder = EfficientNet.from_name(encoder, in_channels=3)  # RGB encoder
            self.attn_encoder = EfficientNet.from_name(encoder, in_channels=1)  # Attention encoder
            self.num_obs_features = self.rgb_encoder._fc.in_features

        if self.num_obs_features != self.encoding_size:
            self.compress_rgb_enc = nn.Linear(self.num_obs_features, self.encoding_size)
            self.compress_attn_enc = nn.Linear(self.num_obs_features, self.encoding_size)
        else:
            self.compress_rgb_enc = nn.Identity()
            self.compress_attn_enc = nn.Identity()

        self.rgb_modality_embedding = nn.Parameter(torch.randn(1, 1, encoding_size)* 0.02)
        self.attn_modality_embedding = nn.Parameter(torch.randn(1, 1, encoding_size)* 0.02)
        
        self.decoder = None
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_traj_pred * self.num_action_params),
        )

    def forward(self, obs_img: torch.tensor, attention) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.decoder is None:
            with torch.no_grad():
                sample_rgb = obs_img[0:1, 0:3]
                sample_features = self.rgb_encoder.extract_features(sample_rgb)
                H_feature = sample_features.shape[2]  # H/32
                W_feature = sample_features.shape[3]  # W/32
            
            # Adjust sequence length to account for RGB spatial tokens + attention tokens
            total_seq_len = (self.context_size + 1) * H_feature * W_feature *2  # +1 for each attention token
            
            self.decoder = base_MultiLayerDecoder(
                embed_dim=self.encoding_size,
                seq_len=total_seq_len,
                output_layers=[256, 128, 64, 32],
                nhead=self.mha_num_attention_heads,
                num_layers=self.mha_num_attention_layers,
                ff_dim_factor=self.mha_ff_dim_factor,
            ).to(obs_img.device)

        # Split inputs
        obs_img = torch.split(obs_img, 3, dim=1)  # [batch_size, 3, H, W] * (context_size+1)
        attention = torch.split(attention, 1, dim=1)  # [batch_size, 1, H, W] * (context_size+1)

        # Process RGB images
        rgb_features_list = []
        attn_features_list = []
        
        for rgb, attn in zip(obs_img, attention):
            # Process RGB
            rgb_feat = self.rgb_encoder.extract_features(rgb)  # [batch_size, C, H/32, W/32]
            N, C, H, W = rgb_feat.shape
            rgb_feat = rgb_feat.permute(0, 2, 3, 1).reshape(N, H * W, C)  # [batch_size, H/32*W/32, C]
            rgb_feat = self.compress_rgb_enc(rgb_feat)
            rgb_feat = rgb_feat + self.rgb_modality_embedding
            rgb_features_list.append(rgb_feat)

            # Process attention
            attn_feat = self.attn_encoder.extract_features(attn)  # [batch_size, C, H/32, W/32]
            attn_feat = attn_feat.permute(0 ,2 , 3, 1).reshape(N, H*W, C)  # [batch_size, C]
            attn_feat = self.compress_attn_enc(attn_feat)
            attn_feat = attn_feat + self.attn_modality_embedding
            attn_features_list.append(attn_feat)

        # Combine features
        rgb_features = torch.stack(rgb_features_list, dim=1)  # [batch_size, context_size+1, H/32*W/32, encoding_size]
        attn_features = torch.stack(attn_features_list, dim=1)  # [batch_size, context_size+1, encoding_size]

        # Flatten spatial tokens and append attention tokens
        batch_size = rgb_features.shape[0]
        seq_len = rgb_features.shape[1] * rgb_features.shape[2] + attn_features.shape[1]*attn_features.shape[2]  # (context_size+1) * [(H/32*W/32)+1]
        device = obs_img[0].device
        tokens = torch.zeros((batch_size, seq_len, self.encoding_size), device=device)
        tokens[:, :rgb_features.shape[1] * rgb_features.shape[2], :] = rgb_features.reshape(batch_size, -1, self.encoding_size)
        tokens[:, rgb_features.shape[1] * rgb_features.shape[2]:, :] = attn_features.reshape(batch_size, -1, self.encoding_size)

        # Transformer decoder
        final_repr, attention_scores = self.decoder(tokens)  # [batch_size, 32]
        action_pred = self.action_predictor(final_repr)
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_traj_pred, self.num_action_params)
        )
        action_pred = torch.cumsum(action_pred, dim=1)  # 将位置增量累计为 waypoints
        
        return action_pred, attention_scores