import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class VisionEncoder(nn.Module):
    """Vision encoder shared between PPO and Dreamer
    depth is given by 256 / 16 = 16, which is the default value for depth
    channel multiplication is (2, 3, 4, 4) which follows their official implementation
    kernel size of 4 differs from their implementation, but follows SheepRL
    """

    def __init__(self, input_channels=12, depth=16):
        super().__init__()
        
        # 1. Feature Extraction: 4-stage hierarchy
        self.convnet = nn.Sequential(
            nn.Conv2d(input_channels, depth * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.SiLU(),
            LayerNormChannelLast(depth * 2),
            
            nn.Conv2d(depth * 2, depth * 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.SiLU(),
            LayerNormChannelLast(depth * 3),
            
            nn.Conv2d(depth * 3, depth * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.SiLU(),
            LayerNormChannelLast(depth * 4),
            
            nn.Conv2d(depth * 4, depth * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.SiLU(),
            LayerNormChannelLast(depth * 4),
            nn.Flatten()
        )

    def forward(self, x):
        features = self.convnet(x)
        return features


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
