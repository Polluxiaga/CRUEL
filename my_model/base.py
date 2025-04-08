import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from efficientnet_pytorch import EfficientNet


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()

        # Compute the positional encoding once
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
        x = x + self.pos_enc[:, :x.size(1), :]
        return x
    

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """ 自定义 TransformerEncoderLayer，强制返回 attn_weights """
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask, 
                                            key_padding_mask=src_key_padding_mask, 
                                            need_weights=True)  # 强制输出注意力权重
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights  # 直接返回注意力权重


class bc_MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(bc_MultiLayerDecoder, self).__init__()
        
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
        attention_scores = torch.stack(attn_scores_list, dim=0)  # [num_layers, batch_size, seq_len, seq_len]
        avg_attention_scores = torch.mean(attention_scores, dim=0)  # [batch_size, seq_len, seq_len]

        x = x.reshape(x.shape[0], -1)
        for layer in self.output_layer:
            x = layer(x)
            x = F.relu(x)

        return x, avg_attention_scores


class base_model(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: int = 3,
        encoder: Optional[str] = "efficientnet-b0",
        encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        bc: 基于Transformer的架构，用于编码视觉观察，并预测动作。
        """
        super(base_model, self).__init__()
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.encoding_size = encoding_size
        self.num_action_params = 2
        self.mha_num_attention_heads = mha_num_attention_heads
        self.mha_num_attention_layers = mha_num_attention_layers
        self.mha_ff_dim_factor = mha_ff_dim_factor

        # 初始化保存梯度的占位变量
        self._grad_obs_features = None

        if encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(encoder, in_channels=3)  # context
            self.num_obs_features = self.obs_encoder._fc.in_features
        else:
            raise NotImplementedError

        if self.num_obs_features != self.encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        self.decoder = None
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_traj_pred * self.num_action_params),
        )

    def forward(self, obs_img: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        #  第一次forward时初始化解码器
        if self.decoder is None:
            # 获取一个样本的特征图大小
            with torch.no_grad():
                sample_features = self.obs_encoder.extract_features(obs_img[0:1, 0:3])
                H_feature = sample_features.shape[2]  # H/32
                W_feature = sample_features.shape[3]  # W/32
            
            self.decoder = bc_MultiLayerDecoder(
                embed_dim=self.encoding_size,
                seq_len=(self.context_size+1) * H_feature * W_feature,
                output_layers=[256, 128, 64, 32],
                nhead=self.mha_num_attention_heads,
                num_layers=self.mha_num_attention_layers,
                ff_dim_factor=self.mha_ff_dim_factor,
            ).to(obs_img.device)  # 确保在同一设备上

        # 每次forward时先清空之前的梯度（重要）
        self._grad_obs_features = None

        # 将输入拆分为 (context_size+1) 个图像
        obs_img = torch.split(obs_img, 3, dim=1)  # [batch_size, 3, H, W]*(context_size+1)
        obs_img = torch.concat(obs_img, dim=0)     # [batch_size*(context_size+1), 3, H, W]

        # 1. 获取所需层特征并注册hook（用于辅助任务）
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
        handle = requested_layer.register_forward_hook(get_requested_features)

        # 2. 正常前向传播，获取最终的全局特征
        obs_features = self.obs_encoder.extract_features(obs_img)  # [N, 1280, H/32, W/32]
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

        # 清理hook
        handle.remove()

        # 返回预测结果和中间特征
        return action_pred, self._raw_obs_features, attention_scores

    @torch.utils.hooks.unserializable_hook
    def _capture_obs_features_grad(self, grad):
        """
        保存obs_features的梯度 (在backward时触发)
        """
        self._grad_obs_features = grad

 
class cnnaux_model(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: int = 3,
        encoder: Optional[str] = "efficientnet-b0",
        encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        bc: 基于Transformer的架构，用于编码视觉观察，并预测动作。
        """
        super(cnnaux_model, self).__init__()
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.encoding_size = encoding_size
        self.num_action_params = 2
        self.mha_num_attention_heads = mha_num_attention_heads
        self.mha_num_attention_layers = mha_num_attention_layers
        self.mha_ff_dim_factor = mha_ff_dim_factor

        # 初始化保存梯度的占位变量
        self._grad_obs_features = None

        if encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(encoder, in_channels=3)  # context
            self.num_obs_features = self.obs_encoder._fc.in_features
        else:
            raise NotImplementedError

        if self.num_obs_features != self.encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        # 添加1x1卷积层用于生成注意力图
        self.gaze_conv = nn.Conv2d(self.num_obs_features, 1, kernel_size=1)

        self.decoder = None
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_traj_pred * self.num_action_params),
        )

    def forward(self, obs_img: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        #  第一次forward时初始化解码器
        if self.decoder is None:
            # 获取一个样本的特征图大小
            with torch.no_grad():
                sample_features = self.obs_encoder.extract_features(obs_img[0:1, 0:3])
                H_feature = sample_features.shape[2]  # H/32
                W_feature = sample_features.shape[3]  # W/32
            
            self.decoder = bc_MultiLayerDecoder(
                embed_dim=self.encoding_size,
                seq_len=(self.context_size+1) * H_feature * W_feature,
                output_layers=[256, 128, 64, 32],
                nhead=self.mha_num_attention_heads,
                num_layers=self.mha_num_attention_layers,
                ff_dim_factor=self.mha_ff_dim_factor,
            ).to(obs_img.device)  # 确保在同一设备上

        # 每次forward时先清空之前的梯度（重要）
        self._grad_obs_features = None

        # 将输入拆分为 (context_size+1) 个图像
        obs_img = torch.split(obs_img, 3, dim=1)  # [batch_size, 3, H, W]*(context_size+1)
        obs_img = torch.concat(obs_img, dim=0)     # [batch_size*(context_size+1), 3, H, W]

        # 1. 获取所需层特征并注册hook（用于辅助任务）
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
        handle = requested_layer.register_forward_hook(get_requested_features)

        # 2. 正常前向传播，获取最终的全局特征
        obs_features = self.obs_encoder.extract_features(obs_img)  # [N, 1280, H/32, W/32]
        N, C, H, W = obs_features.shape  # N = batch_size * (context_size+1)
        
        # 生成gaze_use_map
        gaze_use_map = self.gaze_conv(obs_features)  # [N, 1, H/32, W/32]
        
        # 重塑为所需维度
        gaze_use_map = gaze_use_map.view(
            self.context_size + 1,
            -1,  # batch_size
            1,   # channel
            H * W,   # H/32 * W/32
        ).squeeze(2)  # [context_size+1, batch_size, H/32 * W/32]
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

        # 清理hook
        handle.remove()

        # 返回预测结果和中间特征
        return action_pred, self._raw_obs_features, attention_scores, gaze_use_map

    @torch.utils.hooks.unserializable_hook
    def _capture_obs_features_grad(self, grad):
        """
        保存obs_features的梯度 (在backward时触发)
        """
        self._grad_obs_features = grad