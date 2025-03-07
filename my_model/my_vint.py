import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from efficientnet_pytorch import EfficientNet


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input based on the seq_len
        x = x + self.pos_enc[:, :x.size(1), :]
        return x
    

class MultiLayerDecoder_BC(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder_BC, self).__init__()
        
        # Define two separate positional encodings
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim_factor * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.output_layers = nn.ModuleList([nn.Linear(seq_len * embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers) - 1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x
    

class MultiLayerDecoder_GOAL(nn.Module):
    def __init__(self, embed_dim=512, seq_len_1=6, seq_len_2=3, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder_GOAL, self).__init__()
        
        # Define two separate positional encodings
        self.positional_encoding_1 = PositionalEncoding(embed_dim, max_seq_len=seq_len_1)
        self.positional_encoding_2 = PositionalEncoding(embed_dim, max_seq_len=seq_len_2)
        
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim_factor * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.output_layers = nn.ModuleList([nn.Linear((seq_len_1 + seq_len_2) * embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers) - 1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        # Split the input into two parts
        x1, x2 = x[:, :6, :], x[:, 6:, :]  # assuming first 6 tokens and next 3 tokens
        
        # Apply positional encoding for each part
        x1 = self.positional_encoding_1(x1)
        x2 = self.positional_encoding_2(x2)

        # Concatenate both parts back together
        x = torch.cat([x1, x2], dim=1)

        # Apply the transformer decoder
        x = self.sa_decoder(x)
        
        # Flatten the sequence and pass through output layers
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)

        return x
    
    
class ViNT_BC(nn.Module):
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
        ViNT_BC: 基于Transformer的架构，用于编码视觉观察，并预测动作。
        """
        super(ViNT_BC, self).__init__()
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.encoding_size = encoding_size
        self.num_action_params = 2

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

        self.decoder = MultiLayerDecoder_BC(
            embed_dim=self.encoding_size,
            seq_len=self.context_size+1,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_traj_pred * self.num_action_params),
        )

    def forward(self, obs_img: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 每次forward时先清空之前的梯度（重要）
        self._grad_obs_features = None

        # 将输入拆分为 (context_size+1) 个图像
        obs_img = torch.split(obs_img, 3, dim=1)  # [batch_size, 3, H, W]*(context_size+1)
        obs_img = torch.concat(obs_img, dim=0)     # [batch_size*(context_size+1), 3, H, W]

        # 1. 获取所需层特征并注册hook（用于辅助任务）
        def get_requested_features(module, input, output):
            raw_obs_features = output  # [N, 40, H/8, W/8]
            raw_obs_features.retain_grad()
            raw_obs_features.register_hook(self._capture_obs_features_grad)
            
            # 重塑为所需维度
            N = raw_obs_features.shape[0]
            batch_size = N // (self.context_size + 1)
            self._raw_obs_features = raw_obs_features.view(
                self.context_size + 1,
                batch_size,
                raw_obs_features.shape[1],  # 40 channels
                raw_obs_features.shape[2],  # H/8
                raw_obs_features.shape[3]   # W/8
            )  # [context_size+1, batch_size, 40, H/8, W/8]

        # 注册任意层的hook
        requested_layer = list(self.obs_encoder._blocks)[4]  # 获取H/8分辨率的中间层
        handle = requested_layer.register_forward_hook(get_requested_features)

        # 2. 正常前向传播，获取最终的全局特征
        obs_features = self.obs_encoder.extract_features(obs_img)  # [N, 1280, H/32, W/32]
        obs_encoding = self.obs_encoder._avg_pooling(obs_features)  # [N, 1280, 1, 1]
        
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)  # [N, 1280]
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        
        # 压缩到指定维度
        obs_encoding = self.compress_obs_enc(obs_encoding)  # [N, encoding_size]
        
        # 重塑为序列形式
        obs_encoding = obs_encoding.reshape(
            (self.context_size+1, -1, self.encoding_size)
        )  # [context_size+1, batch_size, encoding_size]
        
        # 转置为transformer期望的输入格式
        tokens = torch.transpose(obs_encoding, 0, 1)  # [batch_size, context_size+1, encoding_size]

        # 3. Transformer解码器处理
        final_repr = self.decoder(tokens)  # [batch_size, 32]
        action_pred = self.action_predictor(final_repr)
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_traj_pred, self.num_action_params)
        )
        action_pred = torch.cumsum(action_pred, dim=1)  # 将位置增量累计为 waypoints

        # 清理hook
        handle.remove()

        # 返回预测结果和中间特征
        return action_pred, self._raw_obs_features

    @torch.utils.hooks.unserializable_hook
    def _capture_obs_features_grad(self, grad):
        """
        保存obs_features的梯度 (在backward时触发)
        """
        self._grad_obs_features = grad
    

class ViNT_GOAL(nn.Module):
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
        ViNT_GOAL: 基于Transformer的架构，用于编码视觉观察和目标，并预测动作。
        """
        super(ViNT_GOAL, self).__init__()
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.encoding_size = encoding_size
        self.num_action_params = 2

        # 初始化保存梯度的占位变量
        self._grad_obs_features = None

        if encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(encoder, in_channels=3)  # context图像编码器
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)  # obs+goal 编码器
            self.num_goal_features = self.goal_encoder._fc.in_features
        else:
            raise NotImplementedError
        
        if self.num_obs_features != self.encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        self.decoder = MultiLayerDecoder_GOAL(
            embed_dim=self.encoding_size,
            seq_len_1=self.context_size+1,
            seq_len_2=self.len_traj_pred,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_traj_pred * self.num_action_params),
        )

    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 每次forward时先清空之前的梯度（重要）
        self._grad_obs_features = None
        
        # 将输入拆分为 (context_size+1) 个图像
        obs_img = torch.split(obs_img, 3, dim=1)  # [batch_size, 3, H, W]*(context_size+1)
        obs_img = torch.concat(obs_img, dim=0)     # [batch_size*(context_size+1), 3, H, W]
        
        # 1. 获取所需层特征并注册hook（用于辅助任务）
        def get_requested_features(module, input, output):
            raw_obs_features = output  # [N, 40, H/8, W/8]
            raw_obs_features.retain_grad()
            raw_obs_features.register_hook(self._capture_obs_features_grad)

            # 重塑为所需维度
            N = raw_obs_features.shape[0]
            batch_size = N // (self.context_size + 1)
            self._raw_obs_features = raw_obs_features.view(
                self.context_size + 1,
                batch_size,
                raw_obs_features.shape[1],  # 40 channels
                raw_obs_features.shape[2],  # H/8
                raw_obs_features.shape[3]   # W/8
            )  # [context_size+1, batch_size, 40, H/8, W/8]
        
        # 注册任意层的hook
        requested_layer = list(self.obs_encoder._blocks)[4]  # 获取H/8分辨率的中间层
        handle = requested_layer.register_forward_hook(get_requested_features)

        # 对目标编码
        curr_img = obs_img[-(obs_img.shape[0]//(self.context_size+1)):, ...]  # 当前图像部分：取最后一段
        curr_img = curr_img.repeat(self.len_traj_pred, 1, 1, 1)  # [batch_size * len_traj_pred, 3, H, W]
        goal_img = torch.split(goal_img, 3, dim=1)  # 切分目标图像为多个 [batch_size, 3, H, W]
        goal_img = torch.concat(goal_img, dim=0)      # [batch_size * len_traj_pred, 3, H, W]
        obsgoal_img = torch.cat([curr_img, goal_img], dim=1)  # [batch_size * len_traj_pred, 6, H, W]
        
        # 2. 正常前向传播，获取最终的全局特征
        # 目标编码部分
        goal_features = self.goal_encoder.extract_features(obsgoal_img)  # [batch_size * len_traj_pred, num_goal_features, H/32, W/32]
        goal_encoding = self.goal_encoder._avg_pooling(goal_features)
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self.goal_encoder._dropout(goal_encoding)
        goal_encoding = self.compress_goal_enc(goal_encoding)  # [batch_size * len_traj_pred, encoding_size]
        goal_encoding = goal_encoding.reshape((self.len_traj_pred, -1, self.encoding_size))  # [len_traj_pred, batch_size, encoding_size]
        goal_encoding = torch.transpose(goal_encoding, 0, 1)  # [batch_size, len_traj_pred, encoding_size]
        
        # 观察编码部分
        obs_features = self.obs_encoder.extract_features(obs_img)  # [N, 1280, H/32, W/32]
        obs_encoding = self.obs_encoder._avg_pooling(obs_features)  # [N, 1280, 1, 1]

        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        
        # 压缩到指定维度
        obs_encoding = self.compress_obs_enc(obs_encoding)

        # 重塑为序列形式
        obs_encoding = obs_encoding.reshape(
            (self.context_size+1, -1, self.encoding_size)
        )  # [context_size+1, batch_size, encoding_size]

        # 转置为transformer期望的输入格式
        obs_encoding = torch.transpose(obs_encoding, 0, 1)  # [batch_size, context_size+1, encoding_size]
        
        # 拼接观察编码与目标编码
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)

        # 3. Transformer解码器处理
        final_repr = self.decoder(tokens)
        action_pred = self.action_predictor(final_repr)
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_traj_pred, self.num_action_params)
        )
        action_pred = torch.cumsum(action_pred, dim=1)  # 将位置增量累计为 waypoints

        # 清理hook
        handle.remove()

        # 返回预测结果和中间特征
        return action_pred, obs_features, goal_features
    
    @torch.utils.hooks.unserializable_hook
    def _capture_obs_features_grad(self, grad):
        """
        保存obs_features的梯度 (在backward时触发)
        """
        self._grad_obs_features = grad