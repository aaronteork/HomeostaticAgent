import numpy as np

import torch
import torch.nn as nn
from typing import Tuple


class VisionEncoder(nn.Module):
    """Original Vision encoder used for PPO that follows the paper's implementation
    

    - Input: (B, C, H, W) - C is the number of channels, 4 for RGBD
    - Two conv layers, 32 filters, 3x3 kernels, no bias, ELU
    - First stride=2, second stride=1
    - Flatten -> Linear(->200) -> LayerNorm -> tanh
    """

    def __init__(self, channels: int=12) -> None:
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, bias=False),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
            nn.ELU(),
            nn.Flatten()
        )

        # compute flattened conv output dim
        with torch.no_grad():
            dummy = torch.zeros(1, channels, 64, 64)
            x = self.convnet(dummy)
            self._conv_out_dim = int(x.numel() // x.shape[0])

        self.proj = nn.Sequential(
            nn.Linear(self._conv_out_dim, 200),
            nn.LayerNorm(200),
            nn.Tanh(),
        )

    def forward(self, vision: torch.Tensor) -> torch.Tensor:
        # vision: (B, H, W, C) -> (B, C, H, W)
        x = self.convnet(vision)
        x = self.proj(x)
        return x
