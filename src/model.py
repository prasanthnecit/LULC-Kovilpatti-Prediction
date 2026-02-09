"""
Causal Spatiotemporal Transformer for LULC Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class SpatialEncoder(nn.Module):
    """CNN-based spatial feature encoder."""
    
    def __init__(self, in_channels: int = 7, d_model: int = 256):
        """
        Initialize spatial encoder.
        
        Args:
            in_channels: Number of input channels (num_classes for one-hot)
            d_model: Dimension of model features
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, d_model, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(d_model)
        self.pool3 = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input of shape (B, C, H, W)
            
        Returns:
            Features of shape (B, d_model, H//8, W//8)
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x


class CausalAttention(nn.Module):
    """Multi-head causal attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize causal attention.
        
        Args:
            d_model: Dimension of model
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query tensor of shape (B, T, d_model)
            key: Key tensor of shape (B, T, d_model)
            value: Value tensor of shape (B, T, d_model)
            mask: Optional causal mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and apply output linear
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)
        
        return output, attn_weights


class SpatiotemporalTransformerBlock(nn.Module):
    """Spatiotemporal transformer block with temporal and spatial attention."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            d_model: Dimension of model
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Temporal attention
        self.temporal_attn = CausalAttention(d_model, n_heads, dropout)
        self.temporal_norm1 = nn.LayerNorm(d_model)
        
        # Spatial attention
        self.spatial_attn = CausalAttention(d_model, n_heads, dropout)
        self.spatial_norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        temporal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input of shape (B, T, N, d_model) where N is spatial dimension
            temporal_mask: Optional temporal causal mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, T, N, D = x.shape
        
        # Temporal attention (across time for each spatial location)
        x_temp = x.permute(0, 2, 1, 3).reshape(B * N, T, D)  # (B*N, T, D)
        
        if temporal_mask is not None:
            temporal_mask = temporal_mask.unsqueeze(0).expand(B * N, -1, -1, -1)
        
        attn_out, attn_weights = self.temporal_attn(x_temp, x_temp, x_temp, temporal_mask)
        x_temp = x_temp + self.dropout(attn_out)
        x_temp = self.temporal_norm1(x_temp)
        
        # Reshape back
        x = x_temp.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)
        
        # Spatial attention (across space for each time step)
        x_spat = x.reshape(B * T, N, D)  # (B*T, N, D)
        
        attn_out, _ = self.spatial_attn(x_spat, x_spat, x_spat, None)
        x_spat = x_spat + self.dropout(attn_out)
        x_spat = self.spatial_norm1(x_spat)
        
        # Reshape back
        x = x_spat.reshape(B, T, N, D)  # (B, T, N, D)
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.ffn_norm(x)
        
        return x, attn_weights


class PhysicsInformedLoss(nn.Module):
    """Physics-informed constraints for LULC prediction."""
    
    def __init__(self):
        """Initialize physics loss."""
        super().__init__()
    
    def transition_constraint(
        self, 
        pred: torch.Tensor, 
        prev_lulc: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize invalid LULC transitions.
        
        Args:
            pred: Predicted logits of shape (B, num_classes, H, W)
            prev_lulc: Previous LULC map of shape (B, H, W) with class indices
            
        Returns:
            Transition loss (scalar)
        """
        # Define invalid transitions (hard constraints)
        # For example: Water (3) should not directly become Urban (0)
        invalid_pairs = [
            (3, 0),  # Water -> Urban (unrealistic)
            (1, 0),  # Forest -> Urban (discouraged, should go through agriculture)
        ]
        
        # Get predicted classes
        pred_classes = torch.argmax(pred, dim=1)  # (B, H, W)
        
        # Calculate penalty for invalid transitions
        penalty = 0.0
        for prev_class, next_class in invalid_pairs:
            invalid_mask = (prev_lulc == prev_class) & (pred_classes == next_class)
            penalty += invalid_mask.float().sum()
        
        # Normalize by batch size and spatial dimensions
        B, H, W = prev_lulc.shape
        penalty = penalty / (B * H * W)
        
        return penalty
    
    def spatial_continuity(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Encourage spatial smoothness in predictions.
        
        Args:
            pred: Predicted logits of shape (B, num_classes, H, W)
            
        Returns:
            Continuity loss (scalar)
        """
        # Compute spatial gradients
        dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        # L2 norm of gradients (total variation)
        grad_loss = (dx ** 2).mean() + (dy ** 2).mean()
        
        return grad_loss


class CausalSpatiotemporalTransformer(nn.Module):
    """Main Causal Spatiotemporal Transformer model for LULC prediction."""
    
    def __init__(
        self,
        num_classes: int = 7,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        img_size: int = 256
    ):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of LULC classes
            d_model: Dimension of model
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            dropout: Dropout rate
            img_size: Size of input images
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.img_size = img_size
        
        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(num_classes, d_model)
        
        # Calculate spatial dimension after encoding
        self.spatial_dim = (img_size // 8) ** 2  # After 3 maxpool layers
        
        # Positional encoding
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, 10, 1, d_model) * 0.02  # Support up to 10 timesteps
        )
        self.spatial_pos_encoding = nn.Parameter(
            torch.randn(1, 1, self.spatial_dim, d_model) * 0.02
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            SpatiotemporalTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Decoder (transposed convolutions to upsample)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        )
    
    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for temporal attention.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask of shape (1, 1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input sequence of shape (B, T, num_classes, H, W)
            
        Returns:
            Tuple of (predictions, attention_weights):
                - predictions: Predicted LULC of shape (B, num_classes, H, W)
                - attention_weights: Attention from last layer
        """
        B, T, C, H, W = x.shape
        
        # Encode each timestep spatially
        x = x.reshape(B * T, C, H, W)
        encoded = self.spatial_encoder(x)  # (B*T, d_model, H//8, W//8)
        
        _, D, H_enc, W_enc = encoded.shape
        N = H_enc * W_enc
        
        # Reshape to sequence format
        encoded = encoded.reshape(B, T, D, N).permute(0, 1, 3, 2)  # (B, T, N, D)
        
        # Add positional encodings
        encoded = encoded + self.temporal_pos_encoding[:, :T, :, :]
        encoded = encoded + self.spatial_pos_encoding
        
        # Create causal mask
        causal_mask = self.create_causal_mask(T, x.device)
        
        # Apply transformer blocks
        attn_weights = None
        for block in self.transformer_blocks:
            encoded, attn_weights = block(encoded, causal_mask)
        
        # Take last timestep for prediction
        last_timestep = encoded[:, -1, :, :]  # (B, N, D)
        
        # Reshape back to spatial
        last_timestep = last_timestep.permute(0, 2, 1).reshape(B, D, H_enc, W_enc)
        
        # Decode to prediction
        prediction = self.decoder(last_timestep)  # (B, num_classes, H, W)
        
        return prediction, attn_weights
