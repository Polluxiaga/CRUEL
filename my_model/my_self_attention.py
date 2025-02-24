import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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