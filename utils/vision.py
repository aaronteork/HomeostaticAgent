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
            LayerNormChannelLast(depth * 2),
            nn.SiLU(),
            
            nn.Conv2d(depth * 2, depth * 3, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(depth * 3),
            nn.SiLU(),
            
            nn.Conv2d(depth * 3, depth * 4, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(depth * 4),
            nn.SiLU(),
            
            nn.Conv2d(depth * 4, depth * 4, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(depth * 4),
            nn.SiLU(),
            nn.Flatten()
        )

    def forward(self, x):
        features = self.convnet(x)
        return features


class VisionDecoder(nn.Module):
    """
    Mirror image of the VisionEncoder.
    Upsamples flattened features back to image space.
    """
    def __init__(self, feature_dim, output_channels=4, depth=16, output_shape=(64, 64)):
        super().__init__()
        self.output_shape = output_shape
        self.output_channels = output_channels

        # 1. Project flattened features back to spatial dimensions
        # Assuming the last conv layer of encoder was (depth*4) channels at 1/16 spatial scale
        self.feature_map_size = (depth * 4, output_shape[0] // 16, output_shape[1] // 16)
        self.fc = nn.Linear(feature_dim, self.feature_map_size[0] * self.feature_map_size[1] * self.feature_map_size[2])
        
        # 2. Upsampling hierarchy
        # Needs to reverse compared to the encoder
        self.convnet = nn.Sequential(
            # Stage 4 -> 3
            nn.ConvTranspose2d(depth * 4, depth * 4, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(depth * 4),
            nn.SiLU(),            
            
            # Stage 3 -> 2
            nn.ConvTranspose2d(depth * 4, depth * 3, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(depth * 3),
            nn.SiLU(),
            
            # Stage 2 -> 1
            nn.ConvTranspose2d(depth * 3, depth * 2, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(depth * 2),
            nn.SiLU(),
            
            # Stage 1 -> Output
            nn.ConvTranspose2d(depth * 2, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(output_channels),
            nn.SiLU(),
            
            # Add in a sigmoid activation to ensure output is in [0, 1] range for image reconstruction
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Project and reshape
        x = self.fc(x)
        x = x.view(x.size(0), *self.feature_map_size)
        
        # Upsample
        return self.convnet(x)


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
