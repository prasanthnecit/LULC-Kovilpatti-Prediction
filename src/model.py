"""
Spatiotemporal Transformer Model for LULC Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialEncoder(nn.Module):
    """CNN-based spatial feature extractor"""
    def __init__(self, input_channels, d_model=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, d_model, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(d_model)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # x: (batch, channels, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attn


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights


class SpatiotemporalTransformer(nn.Module):
    """Complete spatiotemporal transformer for LULC prediction"""
    def __init__(self, num_classes=7, d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        
        # Spatial encoder (for each timestep)
        self.spatial_encoder = SpatialEncoder(num_classes, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model*4, dropout)
            for _ in range(n_layers)
        ])
        
        # Decoder (upsample back to original resolution)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, H, W) - class indices
        batch_size, seq_len, H, W = x.shape
        
        # Convert to one-hot encoding
        x_onehot = F.one_hot(x.long(), num_classes=self.num_classes).permute(0, 1, 4, 2, 3).float()
        # (batch, seq_len, num_classes, H, W)
        
        # Encode each timestep
        spatial_features = []
        for t in range(seq_len):
            feat = self.spatial_encoder(x_onehot[:, t])  # (batch, d_model, H/4, W/4)
            spatial_features.append(feat)
        
        # Stack temporal dimension
        spatial_features = torch.stack(spatial_features, dim=1)  # (batch, seq_len, d_model, H/4, W/4)
        
        # Reshape for transformer: (batch, seq_len, d_model * H/4 * W/4)
        b, t, c, h, w = spatial_features.shape
        spatial_features = spatial_features.view(b, t, c * h * w)
        
        # Positional encoding
        x_encoded = self.pos_encoding(spatial_features)
        
        # Transformer blocks
        attn_weights_list = []
        for block in self.transformer_blocks:
            x_encoded, attn = block(x_encoded)
            attn_weights_list.append(attn)
        
        # Take last timestep
        x_final = x_encoded[:, -1, :]  # (batch, d_model * h * w)
        
        # Reshape back to spatial
        x_final = x_final.view(b, c, h, w)
        
        # Decode to original resolution
        output = self.decoder(x_final)  # (batch, num_classes, H, W)
        
        return output, attn_weights_list
