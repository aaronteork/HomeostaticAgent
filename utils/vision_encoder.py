import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class VisionEncoder(nn.Module):
    """Vision encoder
    """

    def __init__(self, input_channels=12, depth=32, latent_dim=200):
        super().__init__()
        
        # 1. Feature Extraction: 4-stage hierarchy
        self.convnet = nn.Sequential(
            nn.Conv2d(input_channels, depth, kernel_size=4, stride=2, padding=1, bias=False),
            nn.SiLU(),
            LayerNormChannelLast(depth),
            
            nn.Conv2d(depth, depth * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.SiLU(),
            LayerNormChannelLast(depth * 2),
            
            nn.Conv2d(depth * 2, depth * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.SiLU(),
            LayerNormChannelLast(depth * 4),
            
            nn.Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.SiLU(),
            LayerNormChannelLast(depth * 8),
            nn.Flatten()
        )
        
        # 2. Compute flatten dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 64, 64)
            out = self.convnet(dummy)
            self._conv_out_dim = out.shape[-1]
            
        # 3. Projection Head: Maps high-dim CNN output to task-specific latent
        self.proj = nn.Sequential(
            nn.Linear(self._conv_out_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.convnet(x)
        return self.proj(features)


class LayerNormChannelLast(nn.LayerNorm):
    """Obtained from SheepRL"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Input tensor must be 4D (NCHW), received {len(x.shape)}D instead: {x.shape}")
        input_dtype = x.dtype
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.to(input_dtype)
