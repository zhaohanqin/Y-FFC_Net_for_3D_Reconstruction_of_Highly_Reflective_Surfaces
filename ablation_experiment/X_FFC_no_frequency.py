# -*- coding: utf-8 -*-
"""
X_FFC_no_frequency.py — 消融实验 A：去除频域模块（FFC → 纯空间卷积）
====================================================================

【消融对象】
    SpectralTransform（FFC 全局分支的频域核心，内含 FourierUnit 全局频域
    混合 + LFU 局部频域混合 + 门控融合）。

【实现方式】
    定义构造签名、输入输出通道与原 SpectralTransform 完全一致的
    PlainSpectralTransform（纯空间卷积：1x1 降维 → 3x3 → 膨胀 3x3 残差
    → 1x1 升维），然后对 X_FFC_unified 模块做 monkey-patch：

        base.SpectralTransform = PlainSpectralTransform

    由于 FFC_BN_ACT.__init__ 在实例化时按"模块级名称"解析 SpectralTransform，
    一处替换即可让全网所有 FFC 位置同时失去 FFT：
      - Encoder.ffc_bottleneck         (ratio 0.5)
      - GradientDecoder.ffc_grad_h16/h8/h4 (ratio 0.25)
      - MainDecoder.FFC1/FFC2/ffc_h8/ffc_h4
      - EnhancementLayers 各级 _SpectralDecompositionUnit.ffc_specular
    FFC 的 L/G 双分支拓扑、四路交互、跨域门控、通道划分全部保持不变，
    被替换的仅是"全局频域混合"这一种运算 → 单变量消融。

【参数量说明】
    PlainSpectralTransform 的参数量与原频域路径（FourierUnit + 2×LFU）
    同量级且略高。若该消融模型性能仍然下降，则不能归因于容量减少，
    只能归因于失去了频域全局感受野 —— 这使结论更强。

【保持不变】
    双分支编码器 / 梯度辅助分支 / 双头输出 / 残差精修头 /
    全部损失函数（包括频域损失 l_freq —— 它是监督信号，不是结构）/
    权重调度 / 优化器分组（参数名仍含 "ffc"，落在同一 0.5x lr 组）。

【重要警告】
    本文件导入时会修改 X_FFC_unified 模块内部的 SpectralTransform 引用。
    请勿在同一个 Python 进程中同时构建 baseline XFFC 与本消融模型，
    否则 baseline 也会被替换为纯卷积版本。每个实验单独起一个进程即可。
"""

import torch
import torch.nn as nn

import X_FFC_unified as base
from X_FFC_unified import (  # noqa: F401  （供训练脚本统一导入）
    build_scheduler,
    EMA,
    replace_bn_with_gn,
    EnhancedCompositeLoss,
    EnhancedPairedTransform,
    HighlightAugmentation,
    compute_input_gradient_target,
)


class PlainSpectralTransform(nn.Module):
    """纯空间卷积版 SpectralTransform（无 FFT，仅局部感受野）。

    与原 SpectralTransform 保持完全一致的构造签名与输入输出通道：
        __init__(in_channels, out_channels, stride=1, groups=1,
                 enable_lfu=True, **kwargs)
    内部结构：
        1x1 Conv 降维 → [3x3 Conv + 膨胀 3x3 Conv] 残差体 → 1x1 Conv 升维
    用"膨胀卷积扩大一点局部感受野"模拟频域分支的占位，但感受野上限
    仅为 9x9（dilation=2 的 3x3 叠加），与 FFT 的全图感受野有本质区别。
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=1,
                 enable_lfu=True, **kwargs):
        super().__init__()
        # 与原实现一致：groups==1 时中间通道为 out_channels//2
        mid = out_channels // 2 if groups == 1 else out_channels
        mid = max(mid, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=2,
                      dilation=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )
        self.conv2 = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.body(x) + x          # 残差，对应原版 FU/LFU 门控融合的占位
        return self.conv2(x)


# ════════════════════════════════════════════════════════════════════
# ★ 核心 monkey-patch：必须发生在任何模型实例化之前（导入本文件即生效）
# ════════════════════════════════════════════════════════════════════
base.SpectralTransform = PlainSpectralTransform


class YFFC_NoFrequency(base.XFFC):
    """结构与 XFFC 完全一致，但所有 FFC 全局分支已替换为纯空间卷积。"""

    def __init__(self, swin_pretrained=True, debug_shapes=False):
        assert base.SpectralTransform is PlainSpectralTransform, \
            "SpectralTransform 替换未生效，请先导入 X_FFC_no_frequency 再构建模型"
        super().__init__(swin_pretrained=swin_pretrained,
                         debug_shapes=debug_shapes)


def build_model(swin_pretrained=True):
    return YFFC_NoFrequency(swin_pretrained=swin_pretrained)


if __name__ == "__main__":
    # 快速前向自测（CPU，小尺寸，避免下载预训练权重）
    print("=" * 60)
    print("X_FFC_no_frequency 前向自测")
    print("=" * 60)
    model = YFFC_NoFrequency(swin_pretrained=False)

    n_fft = sum(1 for m in model.modules()
                if m.__class__.__name__ == "FourierUnit")
    n_plain = sum(1 for m in model.modules()
                  if isinstance(m, PlainSpectralTransform))
    print(f"FourierUnit 残留数量: {n_fft}（应为 0）")
    print(f"PlainSpectralTransform 数量: {n_plain}（应 > 0）")
    assert n_fft == 0, "仍存在 FourierUnit，频域消融未完成！"

    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    print(f"前向成功: r={tuple(out['r'].shape)}, b={tuple(out['b'].shape)}, "
          f"g={tuple(out['g'].shape)}, s={tuple(out['s'].shape)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total:,}")
