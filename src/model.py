"""
Spatiotemporal Transformer for LULC Prediction
Simplified architecture that works with real data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialEncoder(nn.Module):
    """Extract spatial features from LULC maps"""
    def __init__(self, num_classes=7, d_model=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: (batch, num_classes, H, W)
        return self.encoder(x)  # (batch, d_model, H, W)

class TemporalAttention(nn.Module):
    """Temporal attention across timesteps"""
    def __init__(self, d_model=128):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model, H, W)
        batch, seq_len, d_model, H, W = x.shape
        
        # Reshape to (batch, H, W, seq_len, d_model) for attention
        x = x.permute(0, 3, 4, 1, 2)  # (batch, H, W, seq_len, d_model)
        original_shape = x.shape
        
        # Flatten spatial for processing
        x = x.reshape(batch * H * W, seq_len, d_model)
        
        # Attention
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        
        # Reshape back
        out = out.reshape(batch, H, W, seq_len, d_model)
        out = out.permute(0, 3, 4, 1, 2)  # (batch, seq_len, d_model, H, W)
        
        return out

class SpatiotemporalTransformer(nn.Module):
    """
    Complete spatiotemporal model for LULC prediction
    Simplified architecture that works correctly with real data
    
    Supports arbitrary sequence lengths, though tested primarily with seq_len=2.
    
    Args:
        num_classes: Number of LULC classes (default: 7)
        d_model: Hidden dimension size (default: 128)
        n_layers: Number of temporal attention layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        seq_len: Number of input timesteps (default: 2)
    """
    def __init__(self, num_classes=7, d_model=128, n_layers=2, dropout=0.1, seq_len=2):
        super().__init__()
        assert n_layers > 0, "n_layers must be a positive integer"
        assert seq_len > 0, "seq_len must be a positive integer"
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(num_classes, d_model)
        
        # Temporal attention layers
        self.temporal_attention = nn.ModuleList([
            TemporalAttention(d_model) for _ in range(n_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm([d_model]) for _ in range(n_layers)
        ])
        
        # Temporal fusion (handles seq_len timesteps)
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(d_model * seq_len, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, H, W) - class indices
        batch_size, seq_len, H, W = x.shape
        
        # Convert to one-hot encoding
        x_onehot = F.one_hot(x.long(), num_classes=self.num_classes)
        # (batch, seq_len, H, W, num_classes)
        x_onehot = x_onehot.permute(0, 1, 4, 2, 3).float()
        # (batch, seq_len, num_classes, H, W)
        
        # Encode each timestep spatially
        spatial_features = []
        for t in range(seq_len):
            feat = self.spatial_encoder(x_onehot[:, t])  # (batch, d_model, H, W)
            spatial_features.append(feat)
        
        spatial_features = torch.stack(spatial_features, dim=1)
        # (batch, seq_len, d_model, H, W)
        
        # Temporal attention
        temporal_features = spatial_features
        for i, (attn_layer, norm_layer) in enumerate(zip(self.temporal_attention, self.layer_norms)):
            # Apply attention
            attn_out = attn_layer(temporal_features)
            
            # Residual connection + norm (apply per-pixel)
            # Reshape for LayerNorm
            b, t, c, h, w = temporal_features.shape
            temp_reshaped = temporal_features.permute(0, 1, 3, 4, 2).reshape(b * t * h * w, c)
            attn_reshaped = attn_out.permute(0, 1, 3, 4, 2).reshape(b * t * h * w, c)
            
            normed = norm_layer(temp_reshaped + attn_reshaped)
            temporal_features = normed.reshape(b, t, h, w, c).permute(0, 1, 4, 2, 3)
        
        # Fuse temporal information
        # Concatenate all timesteps
        fused = temporal_features.reshape(batch_size, seq_len * self.d_model, H, W)
        fused = self.temporal_fusion(fused)  # (batch, d_model, H, W)
        
        # Decode to prediction
        output = self.decoder(fused)  # (batch, num_classes, H, W)
        
        return output, None  # Return None for attention weights (compatibility)


# Simpler alternative model (fallback)
class SimpleLULCModel(nn.Module):
    """
    Simplified LULC prediction model
    Uses CNN-based spatiotemporal feature extraction
    
    Supports arbitrary sequence lengths, though tested primarily with seq_len=2.
    
    Args:
        num_classes: Number of LULC classes (default: 7)
        d_model: Hidden dimension size for encoder output (default: 128)
        seq_len: Number of input timesteps (default: 2)
    """
    def __init__(self, num_classes=7, d_model=128, seq_len=2):
        super().__init__()
        assert seq_len > 0, "seq_len must be a positive integer"
        
        self.num_classes = num_classes
        self.seq_len = seq_len
        
        # Encoder for concatenated timesteps
        self.encoder = nn.Sequential(
            nn.Conv2d(num_classes * seq_len, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, H, W)
        batch_size, seq_len, H, W = x.shape
        
        # Convert to one-hot
        x_onehot = F.one_hot(x.long(), num_classes=self.num_classes).float()
        # (batch, seq_len, H, W, num_classes)
        x_onehot = x_onehot.permute(0, 1, 4, 2, 3)
        # (batch, seq_len, num_classes, H, W)
        
        # Concatenate timesteps
        x_concat = x_onehot.reshape(batch_size, seq_len * self.num_classes, H, W)
        # (batch, seq_len * num_classes, H, W)
        
        # Encode
        features = self.encoder(x_concat)
        
        # Decode
        output = self.decoder(features)
        
        return output, None
