"""Spectral transform module built on top of FourierUnit."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier_unit import FourierUnit


class SpectralTransform(nn.Module):
    """
    谱变换模块。

    流程：输入 → 1×1 Conv 降维 → 全局 FU + [F3] 独立 lfu_top/lfu_bot →
          [F4] 门控融合 → 1×1 Conv 升维 → 输出

    在整个模型中的位置：
        被 FFC_BN_ACT 的全局分支（convg2l / convg2g）调用。
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=1,
                 enable_lfu=True, **kwargs):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu

        if groups == 1:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=1,
                          groups=1, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True),
            )
            fu_ch = out_channels // 2
            self.fu = FourierUnit(fu_ch, fu_ch, groups, **kwargs)
            if self.enable_lfu:
                # [F3] 拆分为两个独立 FourierUnit
                self.lfu_top = FourierUnit(fu_ch, fu_ch, groups)
                self.lfu_bot = FourierUnit(fu_ch, fu_ch, groups)
            self.conv2 = nn.Conv2d(fu_ch, out_channels, kernel_size=1,
                                   groups=1, bias=False)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            fu_ch = out_channels
            self.fu = FourierUnit(fu_ch, fu_ch, groups, **kwargs)
            if self.enable_lfu:
                self.lfu_top = FourierUnit(fu_ch, fu_ch, groups)
                self.lfu_bot = FourierUnit(fu_ch, fu_ch, groups)
            self.conv2 = nn.Conv2d(fu_ch, out_channels, kernel_size=1,
                                   groups=groups, bias=False)

        # [F4] FU / LFU 门控融合
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(fu_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        fu_out = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_s = h // 2
            # [F3] 独立 lfu_top / lfu_bot
            top_out = self.lfu_top(x[:, :, :split_s, :])
            bot_out = self.lfu_bot(x[:, :, split_s:, :])
            lfu_out = torch.cat([top_out, bot_out], dim=2)
            # [F4] 门控融合
            gate = self.fusion_gate(fu_out)
            combined = gate * fu_out + (1.0 - gate) * lfu_out
        else:
            gate = self.fusion_gate(fu_out)
            combined = gate * fu_out + (1.0 - gate) * x

        x = self.conv2(combined)
        return x
