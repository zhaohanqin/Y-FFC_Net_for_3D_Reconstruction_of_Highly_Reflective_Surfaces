"""High-frequency enhancement and output refinement heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import _safe_tensor


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
        # HF branch is useful but should start as a small residual correction.
        self.gamma = nn.Parameter(torch.tensor(0.10))

    def forward(self, x):
        B, C, H, W = x.shape
        lap_k = self.lap_kernel.expand(C, 1, 3, 3).to(x.dtype)
        lap_hf = F.conv2d(x, lap_k, padding=1, groups=C)
        learn_hf = self.hf_branch(x)
        fused = self.fusion(torch.cat([x, lap_hf, learn_hf], dim=1))
        fused = fused * self.se(fused)
        out = x + self.gamma.clamp(0.0, 0.3) * fused
        return _safe_tensor(out, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=(-1.0e4, 1.0e4))

class ResidualRefinementHead(nn.Module):
    """[P1] 残差精修头：硬减法后做像素级修正，Tanh 限制小幅残差。

    ★[D2] V3.2 升级：原版 scale=0.08，2→16→16→1 共 3 层，输入只有 (coarse, original)。
    问题：实际推理中 est_b 误差常达到 0.15~0.30，±0.08 的修正幅度根本不够补救。
    升级方案：
      - scale 0.08 → 0.15（修正能力提高近一倍，但仍受 Tanh 限制）
      - 网络加深: 3 层 → 5 层（含残差），表达能力更强
      - 输入升级: 2 通道(coarse, original) → 3 通道(coarse, original, r_direct)
        其中 r_direct 是 MainDecoder 直接预测的漫反射，
        为精修头提供"应当指向的目标"，类似 teacher hint。
    """
    def __init__(self, scale=0.15):
        super().__init__()
        self.scale = scale
        # ★[D2] 输入由 2 通道 → 3 通道（加入 r_direct 作为参考）
        self.refine_in = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1, bias=False), nn.GELU(),
        )
        # 中间一个轻量残差块，提升表达能力
        self.refine_mid = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 1, bias=False), nn.GELU(),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
        )
        self.refine_out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(24, 16, 3, 1, 1, bias=False), nn.GELU(),
            nn.Conv2d(16, 1, 3, 1, 1, bias=True), nn.Tanh(),
        )

    def forward(self, coarse, original, r_direct=None):
        # 若不传 r_direct（向后兼容旧调用），用 coarse 自身填充
        if r_direct is None:
            r_direct = coarse
        feat = self.refine_in(torch.cat([coarse, original, r_direct], dim=1))
        feat = feat + self.refine_mid(feat)  # 残差
        delta = self.refine_out(feat)
        return torch.clamp(coarse + self.scale * delta, 0.0, 1.0)
