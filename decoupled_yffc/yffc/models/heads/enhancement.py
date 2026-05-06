"""High-frequency enhancement and output refinement heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighFrequencyEnhancement(nn.Module):
    """
    高频特征增强模块。

    显式提取编码特征中的高频分量（边缘/跃变），为高反光检测提供清晰的边界信号。
    使用固定 Laplacian 核 + 可学习多尺度分支 + SE 通道注意力。

    在整个模型中的位置：
        Encoder 的 FFC bottleneck 之后，对编码特征 x_c 做高频增强。
    """

    def __init__(self, channels):
        super().__init__()

        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer('lap_kernel', lap.unsqueeze(0).unsqueeze(0))

        half = channels // 2
        self.hf_branch = nn.Sequential(
            nn.Conv2d(channels, half, 3, 1, 1, bias=False),
            nn.BatchNorm2d(half),
            nn.GELU(),
            nn.Conv2d(half, half, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(half),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels + channels + half, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        se_mid = max(channels // 8, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, se_mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        lap_k = self.lap_kernel.expand(C, 1, 3, 3).to(x.dtype)
        lap_hf = F.conv2d(x, lap_k, padding=1, groups=C)
        learn_hf = self.hf_branch(x)
        fused = self.fusion(torch.cat([x, lap_hf, learn_hf], dim=1))
        fused = fused * self.se(fused)
        return x + fused


class ResidualRefinementHead(nn.Module):
    """[P1] 残差精修头：硬减法后做像素级修正，Tanh 限制 ±0.05。"""
    def __init__(self, scale=0.05):
        super().__init__()
        self.scale = scale
        self.refine = nn.Sequential(
            nn.Conv2d(2, 16, 3, 1, 1, bias=False), nn.GELU(),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False), nn.GELU(),
            nn.Conv2d(16, 1, 3, 1, 1, bias=True), nn.Tanh(),
        )
    def forward(self, coarse, original):
        delta = self.refine(torch.cat([coarse, original], dim=1))
        return torch.clamp(coarse + self.scale * delta, 0.0, 1.0)
