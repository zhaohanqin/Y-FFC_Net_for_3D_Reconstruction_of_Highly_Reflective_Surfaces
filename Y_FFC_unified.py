"""
Y_FFC_unified.py — Y-FFC 去高反光网络
=====================================================

本文件包含 Y-FFC 网络的完整实现，包括以下核心组件：

  • FFC 核心模块（FourierUnit / SpectralTransform / FFC_BN_ACT / FFCResnetBlock）
  • 网络基础构件（Conv2D_BN_activa / Attention_block / MultiscaleConvBlock 等）
  • 增强版 ViT 编码器（EnhancedViTEncoderBranch / ConvPosEncoding / CrossModalFusion）
  • 高反光专用模块（SpecularAwareAttention / HighFrequencyEnhancement）
  • 损失函数与评价指标（SSIM / PerceptualLoss / CompositeLoss / Specular 损失等）
  • 数据集与数据增强（FringeDataset / PairedTransform / EnhancedPairedTransform 等）
  • 模型结构（Encoder / GradientDecoder / MainDecoder / EnhancementLayers / YFFC）
  • 训练工具（EMA / build_optimizer / build_scheduler / train_one_epoch 等）

V4 ViT 编码器增强改进：
─────────────────────────────────────────────────
  V1  多层特征提取：从 ViT 第 3/6/9/12 层提取中间特征，获得纹理→语义完整梯度
  V2  卷积位置编码（CPE）：注入局部空间先验，弥补 ViT 局部感知不足
  V3  自适应层注意力融合：通道注意力加权不同 Transformer 层的贡献
  V4  残差特征精炼：融合后的特征经任务相关卷积精炼
  V5  CNN-ViT 交叉注意力融合：替代简单拼接，实现双向信息交互
─────────────────────────────────────────────────

文件组织结构（自上而下）：
─────────────────────────────────────────────────
  第 0 层  外部依赖导入 + 通用工具函数
  第 1 层  FFC 核心模块（FourierUnit / SpectralTransform / FFC_BN_ACT / FFCResnetBlock）
  第 2 层  网络基础构件（Conv2D_BN_activa / Attention_block / EnhancedViTEncoderBranch / ...）
  第 3 层  损失函数与评价指标（SSIM / PerceptualLoss / CompositeLoss / Specular 损失 ...）
  第 4 层  高反光专用模块（SpecularAwareAttention / HighFrequencyEnhancement）
  第 5 层  数据集与数据增强（FringeDataset / PairedTransform / EnhancedPairedTransform / ...）
  第 6 层  模型结构（Encoder / GradientDecoder / MainDecoder / EnhancementLayers / YFFC）
  第 7 层  训练工具（EMA / build_optimizer / build_scheduler / train_one_epoch / ...）
  第 8 层  主入口（main）
─────────────────────────────────────────────────
"""

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      第 0 层：外部依赖导入 + 通用工具函数                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import os
from PIL import Image
import numpy as np
import argparse
import sys
import random
import math
import time


def truncated_normal_(tensor, mean=0.0, std=1.0):
    """
    截断正态分布初始化函数。
    生成截断在 [-2σ, 2σ] 范围内的正态分布随机数，避免极端值。
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def gaussian_window(size, sigma):
    """用于 SSIM 计算的高斯窗口生成函数。"""
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)


def _tukey_window(size, alpha=0.25, device=None, dtype=None):
    """[P11] Tukey 窗：中间 75% 不衰减，两端 12.5% 余弦过渡。"""
    if size <= 1:
        return torch.ones(size, device=device, dtype=dtype or torch.float32)
    n = torch.arange(size, device=device, dtype=dtype or torch.float32)
    if alpha <= 0:
        return torch.ones(size, device=device, dtype=dtype or torch.float32)
    if alpha >= 1:
        return 0.5 * (1.0 - torch.cos(2.0 * math.pi * n / (size - 1)))
    width = int(alpha * (size - 1) / 2.0)
    w = torch.ones(size, device=device, dtype=dtype or torch.float32)
    left = n[:width + 1]
    w[:width + 1] = 0.5 * (1.0 + torch.cos(math.pi * (2.0 * left / (alpha * (size - 1)) - 1.0)))
    right = n[size - width - 1:]
    w[size - width - 1:] = 0.5 * (1.0 + torch.cos(math.pi * (2.0 * right / (alpha * (size - 1)) - 2.0 / alpha + 1.0)))
    return w


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║          第 1 层：FFC 核心模块                                              ║
# ║                                                                          ║
# ║  FFC（Fast Fourier Convolution）是整个网络的频域处理基础。                   ║
# ║  核心思路：将空间特征经 FFT 转入频域，在频域做 1×1 卷积获得全局感受野，       ║
# ║  再经 IFFT 回到空间域。比纯空间大卷积核高效且能捕捉全局依赖。               ║
# ║                                                                          ║
# ║  包含的优化（相比原始 FFC 论文）：                                          ║
# ║    F1  ReLU → GELU（保护频域相位信息）                                     ║
# ║    F2  频率 SE 通道注意力（自适应频段选择）                                  ║
# ║    F3  LFU 拆分为独立 lfu_top / lfu_bot（避免参数竞争）                     ║
# ║    F4  FU / LFU 门控融合（替代硬相加）                                     ║
# ║    F5  跨域分支门控融合（防止频域噪声污染空间分支）                           ║
# ║    F6  FFT 前 Hann 窗（减少频谱泄漏）                                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝


class FourierUnit(nn.Module):
    """
    傅里叶单元 — FFC 的核心组件。

    工作流程：
        空间特征 → [F6] Hann 窗 → FFT → 频域 [real, imag] 拼接 →
        1×1 Conv → [F1] BN + GELU → [F2] SE 注意力 → IFFT → 空间域

    在整个模型中的位置：
        被 SpectralTransform 调用，SpectralTransform 被 FFC_BN_ACT 调用，
        FFC_BN_ACT 被 Encoder/Decoder/EnhancementLayers 中大量使用。
    """

    def __init__(self, in_channels, out_channels, groups=1,
                 spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None,
                 ffc3d=False, fft_norm='ortho',
                 use_tukey_window=True, tukey_alpha=0.25):
        super(FourierUnit, self).__init__()
        self.groups = groups

        # 频域卷积层：输入通道×2（实部+虚部），可选+2（位置编码）
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
            out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)

        # [F1] GELU 替代 ReLU：不 hard-zero 负的虚部，保护相位信息
        self.act = nn.GELU()

        # [F2] SE 通道注意力：在频谱上自适应选择有用频段
        _se_mid = max(out_channels // 4, 1)
        self.freq_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, _se_mid, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(_se_mid, out_channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm
        self.use_tukey_window = use_tukey_window
        self.tukey_alpha = tukey_alpha

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor,
                              mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)

        # AMP 兼容
        orig_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()

        # [F6][P11] Tukey 窗可做消融；默认保留原行为。
        if self.use_tukey_window:
            h, w = x.shape[-2], x.shape[-1]
            win_h = _tukey_window(h, alpha=self.tukey_alpha, device=x.device, dtype=x.dtype)
            win_w = _tukey_window(w, alpha=self.tukey_alpha, device=x.device, dtype=x.dtype)
            tukey_2d = torch.outer(win_h, win_w).unsqueeze(0).unsqueeze(0)
            x_win = x * tukey_2d
        else:
            x_win = x

        # 空间域 → 频域
        ffted = torch.fft.rfftn(x_win, dim=fft_dim, norm=self.fft_norm)

        # 复数分离 → 通道拼接
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # 可选频谱位置编码
        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(
                batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(
                batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        # 频域卷积 + [F1] BN + GELU
        ffted = self.conv_layer(ffted)
        ffted = self.act(self.bn(ffted))

        # [F2] 频率 SE 通道注意力
        ffted = ffted * self.freq_se(ffted)

        # 复数合并
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        if ffted.dtype == torch.float16:
            ffted = ffted.float()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        # 频域 → 空间域
        if self.ffc3d:
            output = torch.fft.irfftn(ffted, s=r_size[-3:], dim=fft_dim, norm=self.fft_norm)
        else:
            output = torch.fft.irfftn(ffted, s=r_size[-2:], dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size,
                                   mode=self.spatial_scale_mode, align_corners=False)

        if orig_dtype == torch.float16:
            output = output.half()

        return output


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


class FFC_BN_ACT(nn.Module):
    """
    FFC 模块 — 快速傅里叶卷积（带 BN 和激活）。

    核心思想：将输入特征分为局部(L)/全局(G)两个分支，四路交互：
        L2L (空间 Conv) + G2L (SpectralTransform) → 局部输出
        L2G (空间 Conv) + G2G (SpectralTransform) → 全局输出
    [F5] 跨域分支门控融合防止频域噪声污染空间分支。

    在整个模型中的位置：
        Encoder 的 ffc_bottleneck、Decoder 各级的 ffc_grad/ffc_h 均使用此模块。
        是 FFC 在网络中的统一入口。
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 activation_type='relu', **kwargs):
        super(FFC_BN_ACT, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        # 四个分支
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, **kwargs)

        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, **kwargs)

        module = nn.Identity if in_cg == 0 or out_cl == 0 else SpectralTransform
        self.convg2l = module(in_cg, out_cl, stride,
                              1 if groups == 1 else groups // 2,
                              enable_lfu, **kwargs)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride,
                              1 if groups == 1 else groups // 2,
                              enable_lfu, **kwargs)

        self.bnl = nn.BatchNorm2d(out_cl) if out_cl != 0 else nn.Identity()
        self.bng = nn.BatchNorm2d(out_cg) if out_cg != 0 else nn.Identity()

        if activation_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation_type == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Identity()

        # [F5] 跨域门控
        self._has_gate_l = (out_cl > 0 and in_cg > 0
                            and not isinstance(self.convg2l, nn.Identity))
        self._has_gate_g = (out_cg > 0 and in_cg > 0
                            and not isinstance(self.convg2g, nn.Identity))

        if self._has_gate_l:
            self.gate_l = nn.Sequential(
                nn.Conv2d(out_cl, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        if self._has_gate_g:
            self.gate_g = nn.Sequential(
                nn.Conv2d(out_cg, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, None)
        out_xl, out_xg = 0, 0

        if self.global_in_num > 0 and x_g is not None:
            # 局部输出
            out_l2l = self.convl2l(x_l) if not isinstance(self.convl2l, nn.Identity) else x_l
            out_g2l = self.convg2l(x_g) if not isinstance(self.convg2l, nn.Identity) else x_g

            if (isinstance(out_l2l, torch.Tensor) and isinstance(out_g2l, torch.Tensor)
                    and out_l2l.shape[-2:] != out_g2l.shape[-2:]):
                out_g2l = F.interpolate(out_g2l, size=out_l2l.shape[-2:],
                                        mode='bilinear', align_corners=False)

            if isinstance(out_l2l, torch.Tensor) and isinstance(out_g2l, torch.Tensor):
                if self._has_gate_l:
                    gate_l = self.gate_l(out_g2l)
                    out_xl = out_l2l + gate_l * out_g2l
                else:
                    out_xl = out_l2l + out_g2l
            else:
                out_xl = out_l2l

            # 全局输出
            out_l2g = self.convl2g(x_l) if not isinstance(self.convl2g, nn.Identity) else x_l
            out_g2g = self.convg2g(x_g) if not isinstance(self.convg2g, nn.Identity) else x_g

            if (isinstance(out_l2g, torch.Tensor) and isinstance(out_g2g, torch.Tensor)
                    and out_l2g.shape[-2:] != out_g2g.shape[-2:]):
                out_g2g = F.interpolate(out_g2g, size=out_l2g.shape[-2:],
                                        mode='bilinear', align_corners=False)

            if isinstance(out_l2g, torch.Tensor) and isinstance(out_g2g, torch.Tensor):
                if self._has_gate_g:
                    gate_g = self.gate_g(out_g2g)
                    out_xg = out_l2g + gate_g * out_g2g
                else:
                    out_xg = out_l2g + out_g2g
            else:
                out_xg = out_l2g
        else:
            out_xl = self.convl2l(x_l) if not isinstance(self.convl2l, nn.Identity) else x_l
            out_xg = self.convl2g(x_l) if not isinstance(self.convl2g, nn.Identity) else x_l

        out_xl = self.act(self.bnl(out_xl))
        out_xg = self.act(self.bng(out_xg))

        return out_xl, out_xg


class FFCResnetBlock(nn.Module):
    """
    FFC 残差块 — 用于堆叠多层 FFC。
    结构：输入 → FFC1 → FFC2 → 残差连接 → 输出
    """

    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU,
                 dilation=1, spatial_transform_kwargs=None, inline=False,
                 **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3,
                                 padding=dilation, dilation=dilation,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3,
                                 padding=dilation, dilation=dilation,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **conv_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l = x[:, :-self.conv1.global_in_num]
            x_g = x[:, -self.conv1.global_in_num:]
        else:
            x_l, x_g = x if isinstance(x, tuple) else (x, 0)

        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g

        if self.inline:
            out = torch.cat(out, dim=1)
        return out


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║          第 2 层：网络基础构件                                              ║
# ║                                                                          ║
# ║  这些是构成 Encoder / Decoder 的基础 building block，                      ║
# ║  包括卷积块、注意力块、多尺度卷积块、转置卷积块、                            ║
# ║  增强版 ViT 编码器分支、交叉注意力融合模块等。                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝


class Conv2D_BN_activa(nn.Module):
    """通用卷积/转置卷积 + BN + 激活 组合块。被网络各处大量使用。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=0, dilation=1, if_bn=True, activation="relu",
                 bias=None, initializer=None, transpose=False):
        super(Conv2D_BN_activa, self).__init__()
        if transpose:
            self.conv2d = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation,
                bias=(not if_bn) if bias is None else bias)
        else:
            self.conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation,
                bias=(not if_bn) if bias is None else bias)

        self.if_bn = if_bn
        if self.if_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

        self.initializer = initializer
        if self.initializer is not None and self.initializer == "truncated_norm":
            nn.init.normal_(self.conv2d.weight, std=0.02)
            self.conv2d.weight = truncated_normal_(self.conv2d.weight, std=0.02)

    def forward(self, x):
        x = self.conv2d(x)
        if self.activation:
            if self.if_bn:
                x = self.bn(x)
            x = self.activation(x)
        return x


class Attention_block(nn.Module):
    """
    注意力门控块。
    用于 Decoder skip connection 中对编码器特征进行注意力加权。

    特点：
      - psi 输出 F_l 通道（每通道独立注意力权重），而非 1 通道。
      - F_int 缩小以降低参数量并减轻过拟合。
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        gate = 1.0 + 0.5 * (2.0 * psi - 1.0)
        return x * gate


class MultiscaleConvBlock(nn.Module):
    """多尺度卷积块（三路并行：7×7 / 3×3 / 3×3+3×3），用于 MainDecoder 特征提取。"""

    def __init__(self, in_channels, out_channels_branch=192):
        super(MultiscaleConvBlock, self).__init__()
        self.path1 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 96, 1, 1, 0),
            Conv2D_BN_activa(96, out_channels_branch, 7, 1, 3),
        )
        self.path2 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.path3 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 256, 1, 1, 0),
            Conv2D_BN_activa(256, 256, 3, 1, 1),
            Conv2D_BN_activa(256, out_channels_branch, 3, 1, 1),
        )

    def forward(self, x):
        return torch.cat((self.path1(x), self.path2(x), self.path3(x)), 1)


class FeatureExtrationLayersB(nn.Module):
    """四路特征提取块，用于 MainDecoder 中间层。"""

    def __init__(self, in_channels):
        super(FeatureExtrationLayersB, self).__init__()
        self.path1 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 192, 7, 1, 3),
        )
        self.path2 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 192, 3, 1, 1),
        )
        self.path3 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 128, 3, 1, 1),
        )
        self.path4 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 128, 3, 1, 1),
            Conv2D_BN_activa(128, 128, 3, 1, 1),
        )

    def forward(self, x):
        return torch.cat((self.path1(x), self.path2(x),
                          self.path3(x), self.path4(x)), 1)


class TransposedConvBlock(nn.Module):
    """三路差异化 kernel 转置卷积（上采样）块。"""

    def __init__(self, in_channels, out_channels_branch=256):
        super(TransposedConvBlock, self).__init__()
        self.branch_1 = Conv2D_BN_activa(in_channels, out_channels_branch, 2, 2, 0, transpose=True)
        self.branch_2 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)
        self.branch_3 = Conv2D_BN_activa(in_channels, out_channels_branch, 6, 2, 2, transpose=True)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        if x3.shape[-2:] != x2.shape[-2:]:
            x3 = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x1, x2, x3], dim=1)


class TransposedConvBlock_same(nn.Module):
    """三路同尺寸卷积块（不改变空间分辨率）。"""

    def __init__(self, in_channels, out_channels_branch=256):
        super(TransposedConvBlock_same, self).__init__()
        self.branch_1 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.branch_2 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.branch_3 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)

    def forward(self, x):
        return torch.cat([self.branch_1(x), self.branch_2(x), self.branch_3(x)], dim=1)


class ConvPosEncoding(nn.Module):
    """
    [V2] 卷积位置编码（Convolutional Position Encoding）。

    用深度可分离卷积为 Transformer token 序列注入局部空间结构信息。
    相比标准 ViT 的固定/可学习位置编码：
      - 能感知局部邻域关系（3×3 卷积窗口）
      - 对不同分辨率输入具有天然的泛化能力
      - 保留了平移等变性
    参考：CPVT（Conditional Position encoding for Vision Transformers, ICLR 2023）
    """

    def __init__(self, dim, k=3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=True)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, N, C) — token 序列
            H, W: patch grid 的空间尺寸
        Returns:
            (B, N, C) — 加入局部位置信息后的 token 序列
        """
        B, N, C = x.shape
        feat = x.transpose(1, 2).reshape(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        return x + feat


class EnhancedViTEncoderBranch(nn.Module):
    """
    增强版 ViT-B/16 编码器分支（多层特征提取 + CPE + 自适应层融合）。

    相比原始 ViTEncoderBranch 的改进：
    ─────────────────────────────────────────────
      V1  多层特征提取：手动遍历 Transformer 的各层，从第 3/6/9/12 层
          提取中间特征。浅层保留纹理/边缘信息（对高光边界至关重要），
          深层提供全局语义（对大面积高光区域识别至关重要）。

      V2  卷积位置编码（CPE）：在 patch embedding 之后、进入 Transformer
          之前注入局部空间先验。弥补标准 ViT 对局部结构感知不足的缺陷，
          使模型能更精确地定位高光边界。

      V3  自适应层注意力融合：使用通道注意力机制（SE-style）自适应加权
          不同 Transformer 层的特征贡献。训练过程中模型会自动学习：
          浅层特征在强高光边界区域的权重更高，深层特征在大面积高光
          区域的权重更高。

      V4  残差特征精炼：融合后通过两层 3×3 Conv + BN + GELU 进行
          任务相关的特征精炼，加残差连接避免信息丢失。
    ─────────────────────────────────────────────

    在整个模型中的位置：
        被 Encoder 调用，与 ResNet50 并行提取全局语义特征。
        输出通过 CrossModalFusion 与 CNN 特征进行交叉注意力融合。
    """

    def __init__(self, vit_image_size=224, extract_layers=(2, 5, 8, 11)):
        super(EnhancedViTEncoderBranch, self).__init__()
        self.vit_image_size = vit_image_size
        self.extract_layers = list(extract_layers)
        self.pretrained_loaded = False
        self.pretrained_status = "vit_b_16: not initialized"

        # ---------- 加载 ViT-B/16 ----------
        vit_weights = None
        try:
            vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        except Exception:
            vit_weights = None

        if not hasattr(models, "vit_b_16"):
            raise RuntimeError("当前 torchvision 版本不包含 vit_b_16")

        try:
            self.vit = models.vit_b_16(weights=vit_weights)
            self.pretrained_loaded = vit_weights is not None
            self.pretrained_status = (
                "vit_b_16: ImageNet pretrained"
                if self.pretrained_loaded else
                "vit_b_16: random init (weights unavailable)"
            )
        except Exception as exc:
            self.vit = models.vit_b_16(weights=None)
            self.pretrained_loaded = False
            self.pretrained_status = f"vit_b_16: random init ({type(exc).__name__})"

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)

        if hasattr(self.vit, "heads"):
            self.vit.heads = nn.Identity()
        elif hasattr(self.vit, "head"):
            self.vit.head = nn.Identity()

        self.embed_dim = getattr(self.vit, "hidden_dim", self.vit.conv_proj.out_channels)

        # [V2] 卷积位置编码
        self.cpe = ConvPosEncoding(self.embed_dim, k=3)

        # [V3] 多层特征的层归一化 + 自适应注意力融合
        n_layers = len(self.extract_layers)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.embed_dim) for _ in range(n_layers)
        ])

        fused_dim = self.embed_dim * n_layers
        se_mid = max(n_layers * 2, 4)
        self.layer_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fused_dim, se_mid),
            nn.GELU(),
            nn.Linear(se_mid, n_layers),
            nn.Softmax(dim=-1),
        )

        # [V4] 残差特征精炼
        self.refine = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
        )

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        mean = self.imagenet_mean.to(dtype=x.dtype, device=x.device)
        std = self.imagenet_std.to(dtype=x.dtype, device=x.device)
        x = (x - mean) / std

        x = F.interpolate(x, size=(self.vit_image_size, self.vit_image_size),
                          mode="bilinear", align_corners=False)
        B = x.size(0)

        # Patch embedding
        x = self.vit.conv_proj(x)
        grid_h, grid_w = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # [V2] 注入局部空间位置先验
        x = self.cpe(x, grid_h, grid_w)

        # 添加 class token
        has_cls = hasattr(self.vit, "class_token")
        if has_cls:
            cls_token = self.vit.class_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # 添加 ViT 原始位置编码 + dropout
        if hasattr(self.vit.encoder, "pos_embedding"):
            x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)

        # [V1] 手动遍历 Transformer 层，收集中间特征
        multi_feats = []
        for i, layer in enumerate(self.vit.encoder.layers):
            x = layer(x)
            if i in self.extract_layers:
                multi_feats.append(x)

        # 确保收集到足够的特征（如果 ViT 层数不够则用最后的输出补齐）
        while len(multi_feats) < len(self.extract_layers):
            multi_feats.append(x)

        # [V3] 多层特征处理 + 自适应加权融合
        layer_feats_2d = []
        for i, feat in enumerate(multi_feats):
            feat = self.layer_norms[i](feat)
            # 移除 class token
            if has_cls and feat.shape[1] == grid_h * grid_w + 1:
                feat = feat[:, 1:, :]
            elif feat.shape[1] > grid_h * grid_w:
                feat = feat[:, 1:, :]
            feat_2d = feat.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)
            layer_feats_2d.append(feat_2d)

        # 通道注意力加权：自适应选择各层贡献
        concat_feats = torch.cat(layer_feats_2d, dim=1)  # (B, embed_dim*n, H, W)
        attn_weights = self.layer_attention(concat_feats)  # (B, n_layers)

        fused = torch.zeros_like(layer_feats_2d[0])
        for i, feat in enumerate(layer_feats_2d):
            w = attn_weights[:, i].view(B, 1, 1, 1)
            fused = fused + w * feat

        # [V4] 残差精炼
        fused = fused + self.refine(fused)

        return fused


class CrossModalFusion(nn.Module):
    """
    [V5] 轻量级 CNN-ViT 交叉注意力融合模块。

    替代原始的简单 concat + 1×1 Conv 融合方式。
    核心思想：
      - CNN → ViT 方向：CNN 特征作为 Query 去查询 ViT 的全局语义 Key/Value，
        为局部细节注入全局上下文（例如：判断某个高亮区域是局部反射还是全局照明）
      - ViT → CNN 方向：ViT 特征作为 Query 去查询 CNN 的局部细节 Key/Value，
        为全局语义补充精确的空间结构信息（例如：高光边界的精确位置）
      - 双向交叉后通过门控机制融合，输出维度为 out_dim

    使用降采样键值对减少计算量（sr_ratio），适合高分辨率特征图。

    在整个模型中的位置：
        Encoder 中，在 ResNet50 backbone 输出 x_bb_4 和 ViT 输出 vit_feat
        之后进行融合，替代原来的 cat + 1×1 Conv。
    """

    def __init__(self, cnn_dim, vit_dim, out_dim, num_heads=4, sr_ratio=2):
        super(CrossModalFusion, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.out_dim = out_dim

        # CNN 关注 ViT（获取全局上下文）
        self.q_c2v = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.k_c2v = nn.Conv2d(vit_dim, out_dim, 1, bias=False)
        self.v_c2v = nn.Conv2d(vit_dim, out_dim, 1, bias=False)
        self.proj_c2v = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        # ViT 关注 CNN（获取局部细节）
        self.q_v2c = nn.Conv2d(vit_dim, out_dim, 1, bias=False)
        self.k_v2c = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.v_v2c = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.proj_v2c = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        # 键值对空间降采样（减少计算量）
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_pool = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)

        # 门控融合：自适应平衡两个方向的贡献
        self.gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid(),
        )

        # 最终投影
        self.final_proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def _cross_attn(self, q, k, v, proj):
        """执行单方向的交叉注意力。"""
        B, C, H, W = q.shape
        Hk, Wk = k.shape[-2:]
        nh = self.num_heads
        hd = self.head_dim

        q = q.reshape(B, nh, hd, H * W).permute(0, 1, 3, 2)   # (B, nh, HW, hd)
        k = k.reshape(B, nh, hd, Hk * Wk).permute(0, 1, 3, 2) # (B, nh, HkWk, hd)
        v = v.reshape(B, nh, hd, Hk * Wk).permute(0, 1, 3, 2) # (B, nh, HkWk, hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale           # (B, nh, HW, HkWk)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return proj(out)

    def forward(self, cnn_feat, vit_feat):
        """
        Args:
            cnn_feat: (B, cnn_dim, H, W) — ResNet50 输出
            vit_feat: (B, vit_dim, H, W) — ViT 输出（已对齐到相同空间尺寸）
        Returns:
            (B, out_dim, H, W) — 融合后的特征
        """
        # 降采样键值对以减少计算量
        if self.sr_ratio > 1:
            vit_kv = self.sr_pool(vit_feat)
            cnn_kv = self.sr_pool(cnn_feat)
        else:
            vit_kv = vit_feat
            cnn_kv = cnn_feat

        # CNN → ViT 方向：CNN 查询 ViT 的全局语义
        q_c = self.q_c2v(cnn_feat)
        k_v = self.k_c2v(vit_kv)
        v_v = self.v_c2v(vit_kv)
        c2v_out = self._cross_attn(q_c, k_v, v_v, self.proj_c2v)

        # ViT → CNN 方向：ViT 查询 CNN 的局部细节
        q_v = self.q_v2c(vit_feat)
        k_c = self.k_v2c(cnn_kv)
        v_c = self.v_v2c(cnn_kv)
        v2c_out = self._cross_attn(q_v, k_c, v_c, self.proj_v2c)

        # 门控融合
        gate = self.gate(torch.cat([c2v_out, v2c_out], dim=1))
        fused = gate * c2v_out + (1.0 - gate) * v2c_out

        return self.final_proj(fused)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║          第 3 层：损失函数与评价指标                                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝


class SSIM(nn.Module):
    """SSIM 结构相似性指标。[P12] as_loss=True 时不 clamp，保留梯度流。"""

    def __init__(self, window_size=11, sigma=1.5, as_loss=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.as_loss = as_loss
        base = gaussian_window(window_size, sigma).unsqueeze(0).unsqueeze(0)
        self.register_buffer("base_window", base)

    def forward(self, img1, img2):
        c = img1.size(1)
        window = self.base_window.to(dtype=img1.dtype, device=img1.device).repeat(c, 1, 1, 1)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=c)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=c)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 ** 2, window, padding=self.window_size // 2, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window, padding=self.window_size // 2, groups=c) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=c) - mu12

        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        eps = 1e-8

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)
        # [P12-FIX] 训练时用 soft clamp 保护：允许梯度流但限制在合理范围
        # 原版 as_loss=True 完全不 clamp，可能出现 >1 的值使 1-ssim 变负
        if not self.as_loss:
            ssim_map = torch.clamp(ssim_map, 0.0, 1.0)
        else:
            # soft clamp：限制在 [-0.5, 1.0] 范围，保护梯度但阻止极端值
            ssim_map = torch.clamp(ssim_map, -0.5, 1.0)
        return ssim_map.mean()


class MSSSIM(nn.Module):
    """[P3] 多尺度 SSIM 损失：3 级降采样，加权融合对比度和结构分量。"""
    def __init__(self, window_size=11, sigma=1.5, n_scales=3, weights=None):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.n_scales = n_scales
        if weights is None:
            dw = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
            weights = dw[:n_scales]
            s = sum(weights)
            weights = [w_/s for w_ in weights]
        self.register_buffer("scale_weights", torch.tensor(weights, dtype=torch.float32))
        base = gaussian_window(window_size, sigma).unsqueeze(0).unsqueeze(0)
        self.register_buffer("base_window", base)

    def _ssim_components(self, img1, img2):
        c = img1.size(1)
        window = self.base_window.to(dtype=img1.dtype, device=img1.device).repeat(c,1,1,1)
        pad = self.window_size // 2
        mu1 = F.conv2d(img1, window, padding=pad, groups=c)
        mu2 = F.conv2d(img2, window, padding=pad, groups=c)
        mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1*mu2
        s1_sq = F.conv2d(img1**2, window, padding=pad, groups=c) - mu1_sq
        s2_sq = F.conv2d(img2**2, window, padding=pad, groups=c) - mu2_sq
        s12 = F.conv2d(img1*img2, window, padding=pad, groups=c) - mu12
        s1_sq = torch.clamp(s1_sq, min=0)
        s2_sq = torch.clamp(s2_sq, min=0)
        C1, C2 = 0.01**2, 0.03**2
        lum = (2*mu12 + C1) / (mu1_sq + mu2_sq + C1)
        cs = (2*s12 + C2) / (s1_sq + s2_sq + C2)
        return lum.mean(), cs.mean()

    def forward(self, img1, img2):
        lum_list = []
        cs_list = []
        for i in range(self.n_scales):
            if i > 0:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
                if img1.shape[-1] < self.window_size or img1.shape[-2] < self.window_size:
                    break
            lum, cs = self._ssim_components(img1, img2)
            lum_list.append(lum)
            cs_list.append(cs)
        n_actual = len(cs_list)
        w = self.scale_weights[:n_actual].to(img1.device)
        w = w / w.sum()
        msssim = 1.0
        for i in range(max(n_actual - 1, 0)):
            msssim = msssim * (torch.clamp(cs_list[i], min=1e-8, max=1.0) ** w[i])
        if n_actual > 0:
            msssim = msssim * (torch.clamp(lum_list[-1], min=1e-8, max=1.0) ** w[n_actual - 1])
        return msssim


def charbonnier_loss(pred, gt, eps=1e-6):
    """[P13] Charbonnier 损失：误差 0 处光滑可导，后期梯度更稳定。"""
    return torch.sqrt((pred - gt)**2 + eps**2).mean()


def statistical_loss(pred, gt):
    """统计损失：约束预测与 GT 的均值/标准差一致（逐样本计算）。"""
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    gt_flat = gt.view(B, -1)
    mean_loss = torch.abs(pred_flat.mean(dim=1) - gt_flat.mean(dim=1)).mean()
    pred_std = pred_flat.std(dim=1, unbiased=False)
    gt_std = gt_flat.std(dim=1, unbiased=False)
    std_loss = torch.abs(pred_std - gt_std).mean()
    return mean_loss + std_loss


class PerceptualLoss(nn.Module):
    """
    感知损失（VGG16 特征匹配）。
    使用预训练 VGG16 的浅层/中层特征计算感知距离，参数冻结不参与训练。
    """

    def __init__(self, layers=('relu1_2', 'relu2_2', 'relu3_3'),
                 weights=(1.0, 0.5, 0.25)):
        super(PerceptualLoss, self).__init__()
        self.pretrained_loaded = False
        self.pretrained_status = "vgg16: not initialized"
        vgg_weights = None
        try:
            vgg_weights = models.VGG16_Weights.IMAGENET1K_V1
        except Exception:
            vgg_weights = None
        try:
            vgg = models.vgg16(weights=vgg_weights).features
            self.pretrained_loaded = vgg_weights is not None
            self.pretrained_status = (
                "vgg16: ImageNet pretrained"
                if self.pretrained_loaded else
                "vgg16: random init (weights unavailable)"
            )
        except Exception as exc:
            vgg = models.vgg16(weights=None).features
            self.pretrained_loaded = False
            self.pretrained_status = f"vgg16: random init ({type(exc).__name__})"

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)

        layer_map = {'relu1_2': 4, 'relu2_2': 9, 'relu3_3': 16}
        self.slices = nn.ModuleList()
        prev = 0
        for layer_name in layers:
            end = layer_map[layer_name]
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:end]))
            prev = end

        for param in self.parameters():
            param.requires_grad = False

        self.weights = weights

    def forward(self, pred, gt):
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
            gt = gt.repeat(1, 3, 1, 1)
        mean = self.imagenet_mean.to(dtype=pred.dtype, device=pred.device)
        std = self.imagenet_std.to(dtype=pred.dtype, device=pred.device)
        pred = (pred - mean) / std
        gt = (gt - mean) / std
        loss = 0.0
        x, y = pred, gt
        for i, slice_module in enumerate(self.slices):
            x = slice_module(x)
            y = slice_module(y)
            loss += self.weights[i] * F.l1_loss(x, y)
        return loss


def compute_edge_weight_map(gt_specular, sigma=3.0, base_weight=1.0, edge_weight=3.0):
    """根据镜面反射分量的梯度生成边缘权重图，用于边缘感知加权 L1 损失。"""
    gx = gt_specular[:, :, :, 1:] - gt_specular[:, :, :, :-1]
    gy = gt_specular[:, :, 1:, :] - gt_specular[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0), mode='replicate')
    gy = F.pad(gy, (0, 0, 0, 1), mode='replicate')
    edge_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    k = max(3, int(6 * sigma + 1) | 1)
    pad_size = k // 2
    edge_mag = F.avg_pool2d(
        F.pad(edge_mag, [pad_size] * 4, mode='replicate'),
        kernel_size=k, stride=1, padding=0)

    batch_max = edge_mag.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    edge_mag = edge_mag / (batch_max + 1e-8)
    weight_map = base_weight + (edge_weight - base_weight) * edge_mag
    return weight_map


def compute_gradient_target(target_img, threshold=0.025, steepness=80.0):
    """
    生成稳定的边缘概率监督，避免每张图按 p995 独立归一化导致梯度强度语义漂移。
    输入支持 (1,H,W) 或 (B,1,H,W)，输出形状与输入一致。
    """
    gx = target_img[..., :, 1:] - target_img[..., :, :-1]
    gy = target_img[..., 1:, :] - target_img[..., :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0), mode='replicate')
    gy = F.pad(gy, (0, 0, 0, 1), mode='replicate')
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    return torch.sigmoid((mag - threshold) * steepness).clamp(0.0, 1.0)


# --- Specular 专用损失 ---

class SpecularFocalLoss(nn.Module):
    """高光像素 Focal 加权损失：对难预测的高亮像素给予更大权重。"""

    def __init__(self, threshold=0.05, focal_weight=5.0, gamma=2.0):
        super().__init__()
        self.threshold = threshold
        self.focal_weight = focal_weight
        self.gamma = gamma

    def forward(self, pred, gt):
        err = torch.abs(pred - gt)
        mask = torch.sigmoid(30.0 * (gt - self.threshold))
        w = 1.0 + (self.focal_weight - 1.0) * mask
        focal = (1.0 + err.detach()) ** self.gamma
        return (err * w * focal).mean()


class SpecularDiceLoss(nn.Module):
    """高光区域 Dice 损失：从分割视角约束高光图的前景重叠度。"""

    def __init__(self, threshold=0.05, smooth=1.0):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, pred, gt):
        pred_p = torch.sigmoid(20.0 * (pred - self.threshold))
        gt_m = (gt > self.threshold).float()
        inter = (pred_p * gt_m).sum(dim=(1, 2, 3))
        union = pred_p.sum(dim=(1, 2, 3)) + gt_m.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


class GradientConsistencyLoss(nn.Module):
    """Specular 边界梯度一致性损失：约束预测与 GT 的空间梯度幅值和方向一致。"""

    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sx.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sx.t().unsqueeze(0).unsqueeze(0))

    def forward(self, pred, gt):
        sx = self.sobel_x.to(pred.dtype)
        sy = self.sobel_y.to(pred.dtype)

        pgx = F.conv2d(pred, sx, padding=1)
        pgy = F.conv2d(pred, sy, padding=1)
        ggx = F.conv2d(gt, sx, padding=1)
        ggy = F.conv2d(gt, sy, padding=1)

        pm = torch.sqrt(pgx ** 2 + pgy ** 2 + 1e-8)
        gm = torch.sqrt(ggx ** 2 + ggy ** 2 + 1e-8)

        mag_loss = F.l1_loss(pm, gm)

        mask = (gm > gm.mean() * 0.5).float()
        cos = (pgx * ggx + pgy * ggy) / (pm * gm + 1e-8)
        dir_loss = ((1.0 - cos) * mask).sum() / (mask.sum() + 1e-8)

        return mag_loss + 0.3 * dir_loss


def specular_ohem_loss(pred, gt, base_fn, ratio=0.3):
    """在线难样本挖掘（OHEM）：只对误差最大的 top-k 像素额外加权。"""
    with torch.no_grad():
        err = torch.abs(pred - gt)
        B = err.shape[0]
        flat = err.view(B, -1)
        k = max(1, int(flat.shape[1] * ratio))
        _, idx = torch.topk(flat, k, dim=1)
        mask = torch.zeros_like(flat)
        mask.scatter_(1, idx, 1.0)
        mask = mask.view_as(err)
    base = base_fn(pred, gt)
    hard = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + 1e-8)
    return 0.5 * base + 0.5 * hard


# --- 复合损失函数 ---

class CompositeLoss(nn.Module):
    """
    多目标复合损失。
    整合 SSIM / L1 / 频域 / 感知 / 镜面直接监督 / 统计 / 梯度 / 边缘加权。
    """

    def __init__(self, lambda_ssim=1.0, lambda_l1=1.0, lambda_freq=0.1,
                 lambda_perceptual=0.05, lambda_spec=0.5, lambda_stat=0.1,
                 lambda_grad=0.5, use_edge_weight=True, use_perceptual=True):
        super(CompositeLoss, self).__init__()
        self.ssim = SSIM()
        self.lambda_ssim = lambda_ssim
        self.lambda_l1 = lambda_l1
        self.lambda_freq = lambda_freq
        self.lambda_perceptual = lambda_perceptual
        self.lambda_spec = lambda_spec
        self.lambda_stat = lambda_stat
        self.lambda_grad = lambda_grad
        self.use_edge_weight = use_edge_weight
        self.use_perceptual = use_perceptual
        if self.use_perceptual:
            self.perceptual = PerceptualLoss()

    def frequency_loss(self, pred, gt):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        gt_fft = torch.fft.rfft2(gt, norm='ortho')
        return F.l1_loss(torch.abs(pred_fft), torch.abs(gt_fft))

    def forward(self, pred_diffuse, target, est_b, gt_specular,
                pred_grad, gt_grad):
        l_ssim = 1.0 - self.ssim(pred_diffuse.float(), target.float())

        if self.use_edge_weight:
            weight_map = compute_edge_weight_map(gt_specular.detach())
            l_l1 = (torch.abs(pred_diffuse - target) * weight_map).mean()
        else:
            l_l1 = F.l1_loss(pred_diffuse, target)

        l_freq = self.frequency_loss(pred_diffuse.float(), target.float())

        if self.use_perceptual:
            l_perceptual = self.perceptual(pred_diffuse, target)
        else:
            l_perceptual = torch.tensor(0.0, device=pred_diffuse.device)

        gt_specular_clamped = gt_specular.clamp(0.0, 1.0)
        l_spec = F.l1_loss(est_b, gt_specular_clamped) + \
                 0.5 * (1.0 - self.ssim(est_b.float(), gt_specular_clamped.float()))

        l_stat = statistical_loss(pred_diffuse, target)
        l_grad = 1.0 - self.ssim(pred_grad.float(), gt_grad.float())

        total = (self.lambda_ssim * l_ssim + self.lambda_l1 * l_l1 +
                 self.lambda_freq * l_freq + self.lambda_perceptual * l_perceptual +
                 self.lambda_spec * l_spec + self.lambda_stat * l_stat +
                 self.lambda_grad * l_grad)

        loss_dict = {}
        for k, v in [('total', total), ('ssim', l_ssim), ('l1', l_l1),
                      ('freq', l_freq), ('perceptual', l_perceptual),
                      ('spec', l_spec), ('stat', l_stat), ('grad', l_grad)]:
            loss_dict[k] = v.item() if isinstance(v, torch.Tensor) else v

        return total, loss_dict

    def update_weights(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class EnhancedCompositeLoss(nn.Module):
    """
    增强版复合损失（V3 优化版 — 负数损失已修复）。
    [P3]  SSIM → 0.8*MS-SSIM + 0.2*SSIM 混合
    [P4]  频域损失加入相位一致性约束
    [P5]  移除冗余 statistical_loss（lambda_stat 默认 0）
          ★ 注意：不再使用可学习权重，这是导致负数损失的根源 ★
    [P12] SSIM as_loss=True 保留梯度
    [P13] L1 → Charbonnier 平滑损失
    """

    def __init__(self, lambda_ssim=1.0, lambda_l1=1.0, lambda_freq=0.1,
                 lambda_perceptual=0.05, lambda_spec=0.5, lambda_stat=0.0,
                 lambda_grad=0.5, use_edge_weight=True, use_perceptual=True,
                 lambda_focal=0.8, lambda_dice=0.3, lambda_gc=0.2,
                 use_ohem=True, ohem_ratio=0.3):
        super().__init__()

        self.ssim_fn = SSIM(as_loss=True)       # [P12]
        self.msssim_fn = MSSSIM(n_scales=3)     # [P3]
        self.use_edge_weight = use_edge_weight
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual = PerceptualLoss()
        self._ew_fn = compute_edge_weight_map

        self.focal = SpecularFocalLoss()
        self.dice = SpecularDiceLoss()
        self.gc = GradientConsistencyLoss()

        self.lambda_ssim = lambda_ssim
        self.lambda_l1 = lambda_l1
        self.lambda_freq = lambda_freq
        self.lambda_perceptual = lambda_perceptual
        self.lambda_spec = lambda_spec
        self.lambda_stat = lambda_stat     # [P5] 默认 0，移除冗余 stat
        self.lambda_grad = lambda_grad
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.lambda_gc = lambda_gc
        self.use_ohem = use_ohem
        self.ohem_ratio = ohem_ratio
        # ★ P5 修复：不再有可学习损失权重 ★
        # ★ 所有损失项的权重完全由 lambda_* 和 ImprovedWeightScheduler 控制 ★

    def frequency_loss(self, p, g):
        """[P4-FIX] 幅度谱 + 相位一致性联合损失（修复数值稳定性）。"""
        p_fft = torch.fft.rfft2(p, norm='ortho')
        g_fft = torch.fft.rfft2(g, norm='ortho')
        p_amp = torch.abs(p_fft)
        g_amp = torch.abs(g_fft)
        amp_loss = F.l1_loss(p_amp, g_amp)
        # 修复：用复数乘法代替 torch.angle() 的不连续梯度
        # cos(angle_p - angle_g) = Re(p_fft * conj(g_fft)) / (|p_fft| * |g_fft|)
        p_norm = p_amp.clamp(min=1e-8)
        g_norm = g_amp.clamp(min=1e-8)
        cross = p_fft * g_fft.conj()
        cos_diff = torch.clamp(cross.real / (p_norm * g_norm), -1.0, 1.0)
        phase_weight = (0.5 * (p_amp.detach() + g_amp.detach()))
        phase_weight = phase_weight / (phase_weight.mean(dim=(-2, -1), keepdim=True) + 1e-8)
        phase_loss = ((1.0 - cos_diff) * phase_weight).sum() / (phase_weight.sum() + 1e-8)
        return amp_loss + 0.5 * phase_loss

    def forward(self, pred_diffuse, target, est_b, gt_specular,
                pred_grad, gt_grad):
        gt_sc = gt_specular.clamp(0, 1)

        # [P3] MS-SSIM + SSIM 混合
        l_ssim_single = 1.0 - self.ssim_fn(pred_diffuse.float(), target.float())
        l_msssim = 1.0 - self.msssim_fn(pred_diffuse.float(), target.float())
        l_ssim = 0.8 * l_msssim + 0.2 * l_ssim_single

        # [P13] Charbonnier 替代 L1
        if self.use_edge_weight:
            wm = self._ew_fn(gt_specular.detach())
            l_l1 = (torch.sqrt((pred_diffuse - target)**2 + 1e-12) * wm).mean()
        else:
            l_l1 = charbonnier_loss(pred_diffuse, target)

        # [P4] 频域损失含相位
        l_freq = self.frequency_loss(pred_diffuse.float(), target.float())

        l_percep = (self.perceptual(pred_diffuse, target)
                    if self.use_perceptual
                    else torch.tensor(0.0, device=pred_diffuse.device))

        # [P13] Charbonnier for specular
        l_spec = (charbonnier_loss(est_b, gt_sc) +
                  0.5 * (1.0 - self.ssim_fn(est_b.float(), gt_sc.float())))

        # [P5] stat 默认关闭
        l_stat = torch.tensor(0.0, device=pred_diffuse.device)
        if self.lambda_stat > 0:
            l_stat = statistical_loss(pred_diffuse, target)

        l_grad = 1.0 - self.ssim_fn(pred_grad.float(), gt_grad.float())

        if self.use_ohem:
            l_focal = specular_ohem_loss(est_b, gt_sc, self.focal, self.ohem_ratio)
        else:
            l_focal = self.focal(est_b, gt_sc)
        l_dice = self.dice(est_b, gt_sc)
        l_gc = self.gc(est_b, gt_sc)

        # ★ 纯加权求和，所有项 ≥ 0，total 必定 ≥ 0 ★
        total = (self.lambda_ssim * l_ssim + self.lambda_l1 * l_l1 +
                 self.lambda_freq * l_freq + self.lambda_perceptual * l_percep +
                 self.lambda_spec * l_spec + self.lambda_stat * l_stat +
                 self.lambda_grad * l_grad + self.lambda_focal * l_focal +
                 self.lambda_dice * l_dice + self.lambda_gc * l_gc)

        d = {}
        for k, v in [('total', total), ('ssim', l_ssim_single),
                      ('msssim', l_msssim), ('ssim_mix', l_ssim), ('l1', l_l1),
                      ('freq', l_freq), ('perceptual', l_percep),
                      ('spec', l_spec), ('stat', l_stat), ('grad', l_grad),
                      ('focal', l_focal), ('dice', l_dice), ('gc', l_gc)]:
            d[k] = v.item() if isinstance(v, torch.Tensor) else v
        return total, d

    def update_weights(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║          第 4 层：高反光专用模块                                             ║
# ║                                                                          ║
# ║  这些模块直接嵌入到 Encoder 和 MainDecoder 中，                             ║
# ║  为高反光检测提供专门的注意力机制和高频增强能力。                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝


class SpecularAwareAttention(nn.Module):
    """
    镜面反射感知注意力模块。
    [P9] steepness 15→6；多级阈值(0.5/0.7/0.9)区分弱/中/强高光。
    """

    def __init__(self, in_channels, threshold=0.8, steepness=6.0):
        super().__init__()
        self.steepness = steepness
        self.thresholds = [0.5, 0.7, 0.9]
        self.gate_scale = 0.5
        n_masks = len(self.thresholds)

        mid = max(in_channels // 16, 4)
        self.adaptive_thresh = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, n_masks, 1, bias=True),
            nn.Sigmoid(),
        )

        self.modulation = nn.Sequential(
            nn.Conv2d(in_channels + n_masks, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Tanh(),
        )

    def forward(self, features, original_input):
        inp = original_input
        if inp.shape[-2:] != features.shape[-2:]:
            inp = F.interpolate(inp, size=features.shape[-2:],
                                mode='bilinear', align_corners=False)
        t_off = self.adaptive_thresh(features)
        masks = []
        for i, base_t in enumerate(self.thresholds):
            eff_t = base_t * (0.5 + 0.5 * t_off[:, i:i+1, :, :])
            masks.append(torch.sigmoid(self.steepness * (inp - eff_t)))
        multi_mask = torch.cat(masks, dim=1)
        att = self.modulation(torch.cat([features, multi_mask], dim=1))
        return features * (1.0 + self.gate_scale * att)


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


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║          第 5 层：数据集与数据增强                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝


class PairedTransform:
    """
    配对数据增强。
    对 input 和 target 施加完全相同的几何变换，光度变换仅施加在 input 上。
    """

    def __init__(self, p_flip=0.5, p_rotate=0.3, p_crop=0.5,
                 crop_scale=(0.7, 1.0), brightness_range=(0.9, 1.1)):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_crop = p_crop
        self.crop_scale = crop_scale
        self.brightness_range = brightness_range

    def __call__(self, input_img, target_img):
        _, orig_h, orig_w = input_img.shape
        if random.random() < self.p_flip:
            input_img = torch.flip(input_img, [-1])
            target_img = torch.flip(target_img, [-1])
        if random.random() < self.p_flip:
            input_img = torch.flip(input_img, [-2])
            target_img = torch.flip(target_img, [-2])
        if random.random() < self.p_rotate:
            k = random.choice([1, 2, 3]) if orig_h == orig_w else 2
            input_img = torch.rot90(input_img, k, [1, 2])
            target_img = torch.rot90(target_img, k, [1, 2])
        if random.random() < self.p_crop:
            _, h, w = input_img.shape
            scale = random.uniform(*self.crop_scale)
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h >= 32 and new_w >= 32:
                top = random.randint(0, h - new_h)
                left = random.randint(0, w - new_w)
                input_img = input_img[:, top:top + new_h, left:left + new_w]
                target_img = target_img[:, top:top + new_h, left:left + new_w]
                input_img = F.interpolate(input_img.unsqueeze(0), size=(h, w),
                                          mode='bilinear', align_corners=False).squeeze(0)
                target_img = F.interpolate(target_img.unsqueeze(0), size=(h, w),
                                           mode='bilinear', align_corners=False).squeeze(0)

        brightness_factor = random.uniform(*self.brightness_range)
        input_img = (input_img * brightness_factor).clamp(0.0, 1.0)
        return input_img, target_img


class HighlightAugmentation:
    """
    合成高反光数据增强。
    在 input 上随机叠加合成高斯 blob 高反光，同步更新 gt_specular。
    """

    def __init__(self, p=0.4, num_range=(1, 6),
                 radius_range=(3, 25), intensity_range=(0.2, 0.7)):
        self.p = p
        self.num_range = num_range
        self.radius_range = radius_range
        self.intensity_range = intensity_range

    def __call__(self, input_img, target_img, gt_specular):
        if random.random() > self.p:
            return input_img, target_img, gt_specular
        _, H, W = input_img.shape
        n = random.randint(*self.num_range)
        synth = torch.zeros(1, H, W, dtype=input_img.dtype, device=input_img.device)
        yy = torch.arange(H, dtype=input_img.dtype, device=input_img.device)
        xx = torch.arange(W, dtype=input_img.dtype, device=input_img.device)
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')
        for _ in range(n):
            cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
            r = random.uniform(*self.radius_range)
            I = random.uniform(*self.intensity_range)
            synth[0] += torch.exp(-((gy - cy) ** 2 + (gx - cx) ** 2) / (2 * r * r)) * I
        synth = synth.clamp(0, 1)
        input_img = (input_img + synth).clamp(0, 1)
        gt_specular = (gt_specular + synth).clamp(0, 1)
        return input_img, target_img, gt_specular


class EnhancedPairedTransform:
    """
    增强版配对变换。
    在 PairedTransform 基础上集成 HighlightAugmentation。
    返回 4 项：input, target, gt_specular, gt_gradient。
    """

    def __init__(self, p_flip=0.5, p_rotate=0.3, p_crop=0.5,
                 crop_scale=(0.7, 1.0), brightness_range=(0.9, 1.1),
                 highlight_aug=None):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_crop = p_crop
        self.crop_scale = crop_scale
        self.brightness_range = brightness_range
        self.hl_aug = highlight_aug or HighlightAugmentation()

    def __call__(self, input_img, target_img):
        _, orig_h, orig_w = input_img.shape

        if random.random() < self.p_flip:
            input_img = torch.flip(input_img, [-1])
            target_img = torch.flip(target_img, [-1])
        if random.random() < self.p_flip:
            input_img = torch.flip(input_img, [-2])
            target_img = torch.flip(target_img, [-2])
        if random.random() < self.p_rotate:
            if orig_h == orig_w:
                k = random.choice([1, 2, 3])
            else:
                k = 2
            input_img = torch.rot90(input_img, k, [1, 2])
            target_img = torch.rot90(target_img, k, [1, 2])
        if random.random() < self.p_crop:
            _, h, w = input_img.shape
            s = random.uniform(*self.crop_scale)
            nh, nw = int(h * s), int(w * s)
            if nh >= 32 and nw >= 32:
                t = random.randint(0, h - nh)
                l = random.randint(0, w - nw)
                input_img = input_img[:, t:t + nh, l:l + nw]
                target_img = target_img[:, t:t + nh, l:l + nw]
                input_img = F.interpolate(input_img.unsqueeze(0),
                    size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0)
                target_img = F.interpolate(target_img.unsqueeze(0),
                    size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0)

        # [P7] 先算 specular，再施加亮度增强（同步到 input+target+specular）
        gt_specular = (input_img - target_img).clamp(0, 1)
        bf = random.uniform(*self.brightness_range)
        input_img = (input_img * bf).clamp(0, 1)
        target_img = (target_img * bf).clamp(0, 1)
        gt_specular = (gt_specular * bf).clamp(0, 1)
        input_img, target_img, gt_specular = self.hl_aug(
            input_img, target_img, gt_specular)
        gt_specular = (input_img - target_img).clamp(0, 1)

        gt_gradient = compute_gradient_target(target_img)

        return input_img, target_img, gt_specular, gt_gradient


class FringeDataset(Dataset):
    """
    条纹图数据集。
    支持 PairedTransform 数据增强，gt_gradient 使用梯度幅值。
    """

    def __init__(self, input_dir, target_dir, transform=None, resize=None,
                 paired_transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.resize = resize
        self.paired_transform = paired_transform
        self.files = sorted(
            [f for f in os.listdir(input_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        input_path = os.path.join(self.input_dir, img_name)

        candidate_names = [
            img_name,
            img_name.replace("input_", "target_", 1) if img_name.startswith("input_") else img_name,
            img_name.replace("input", "target", 1) if img_name.startswith("input") else img_name,
        ]

        target_path = None
        for name in candidate_names:
            p = os.path.join(self.target_dir, name)
            if os.path.exists(p):
                target_path = p
                break

        if target_path is None:
            base, ext = os.path.splitext(img_name)
            digits = "".join(ch for ch in base if ch.isdigit())
            if digits:
                guessed = f"target_{digits}{ext}"
                p = os.path.join(self.target_dir, guessed)
                if os.path.exists(p):
                    target_path = p

        if target_path is None:
            raise FileNotFoundError(
                f"Could not find matching target for input '{img_name}' "
                f"in '{self.target_dir}'. Tried: {set(candidate_names)}")

        input_pil = Image.open(input_path).convert("L")
        target_pil = Image.open(target_path).convert("L")

        if self.resize is not None:
            if isinstance(self.resize, (int, float)):
                resize_size = (int(self.resize), int(self.resize))
            elif isinstance(self.resize, (tuple, list)) and len(self.resize) == 2:
                resize_size = (int(self.resize[1]), int(self.resize[0]))  # PIL expects (W, H)
            else:
                resize_size = None
            if resize_size:
                input_pil = input_pil.resize(resize_size, Image.LANCZOS)
                target_pil = target_pil.resize(resize_size, Image.LANCZOS)

        input_img = torch.from_numpy(np.array(input_pil, dtype=np.float32) / 255.0).unsqueeze(0)
        target_img = torch.from_numpy(np.array(target_pil, dtype=np.float32) / 255.0).unsqueeze(0)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        if self.paired_transform is not None:
            input_img, target_img = self.paired_transform(input_img, target_img)

        gt_specular = (input_img - target_img).clamp(0, 1)

        # 稳定边缘概率作为 gt_gradient，避免逐图 p995 归一化造成强度语义漂移。
        gt_gradient = compute_gradient_target(target_img)

        return input_img, target_img, gt_specular, gt_gradient


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║          第 6 层：模型结构                                                  ║
# ║                                                                          ║
# ║  网络总体架构（X 形结构）：                                                 ║
# ║    Encoder（ResNet50 + EnhancedViT + CrossModalFusion +                   ║
# ║            FFC bottleneck + HighFrequencyEnhancement）                    ║
# ║      ↓ 编码特征（5 级：x_bb_1~4, x_c）                                    ║
# ║      ├── GradientDecoder → 梯度图 x_g + 4 级增强特征                      ║
# ║      │   ├── EnhancementLayers → 对增强特征做频域精炼                      ║
# ║      │   └──┐                                                            ║
# ║      └────── MainDecoder（含 SpecularAwareAttention）→ 镜面分量 est_b     ║
# ║    最终输出 pred_diffuse = clamp(input - est_b, 0, 1)                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝


class Encoder(nn.Module):
    """
    编码器：ResNet50（dilated）+ 增强版 ViT-B/16 双分支，末尾接 FFC bottleneck
    和 HighFrequencyEnhancement。

    改进点（相比原始版本）：
      - ViT 分支升级为 EnhancedViTEncoderBranch（多层特征 + CPE + 自适应融合）
      - CNN-ViT 融合升级为 CrossModalFusion（双向交叉注意力）
    输出 5 级特征用于后续 Decoder 的 skip connection。
    """

    def __init__(self, vit_image_size=224):
        super().__init__()
        self.pretrained_loaded = False
        self.pretrained_status = "resnet50: not initialized"

        try:
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1,
                replace_stride_with_dilation=[False, False, True])
            self.pretrained_loaded = True
            self.pretrained_status = "resnet50: ImageNet pretrained"
        except Exception as exc:
            backbone = models.resnet50(
                weights=None,
                replace_stride_with_dilation=[False, False, True])
            self.pretrained_loaded = False
            self.pretrained_status = f"resnet50: random init ({type(exc).__name__})"

        old_conv1 = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight.copy_(old_conv1.weight.mean(1, keepdim=True))
        self.register_buffer(
            "resnet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).mean().view(1, 1, 1, 1),
            persistent=False)
        self.register_buffer(
            "resnet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).mean().view(1, 1, 1, 1),
            persistent=False)

        self.stem_conv = backbone.conv1
        self.stem_bn = backbone.bn1
        self.stem_relu = backbone.relu
        self.stem_pool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.reduce_l1 = Conv2D_BN_activa(256, 128, 1, 1, 0)
        self.reduce_l2 = Conv2D_BN_activa(512, 256, 1, 1, 0)
        self.reduce_l3 = Conv2D_BN_activa(1024, 512, 1, 1, 0)
        self.reduce_l4 = Conv2D_BN_activa(2048, 512, 1, 1, 0)
        self.fuse_l34 = Conv2D_BN_activa(1024, 512, 3, 1, 1)

        self.vit_branch = EnhancedViTEncoderBranch(vit_image_size=vit_image_size)
        # [V5] 交叉注意力融合替代简单拼接
        self.cross_modal_fusion = CrossModalFusion(
            cnn_dim=512, vit_dim=self.vit_branch.embed_dim,
            out_dim=512, num_heads=4, sr_ratio=2)
        self.cba_after_backbone = Conv2D_BN_activa(512, 256, 3, 1, 1)
        self.ffc_bottleneck = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)

        # 高频增强
        self.hf_enhance = HighFrequencyEnhancement(channels=256)

    def forward(self, x):
        x_in = x
        mean = self.resnet_mean.to(dtype=x.dtype, device=x.device)
        std = self.resnet_std.to(dtype=x.dtype, device=x.device)
        x = (x - mean) / std
        x = self.stem_conv(x); x = self.stem_bn(x)
        x_bb_1 = self.stem_relu(x)

        x = self.stem_pool(x_bb_1)
        x_l1 = self.layer1(x); x_bb_2 = self.reduce_l1(x_l1)
        x_l2 = self.layer2(x_l1); x_bb_3 = self.reduce_l2(x_l2)
        x_l3 = self.layer3(x_l2); x_l4 = self.layer4(x_l3)

        x_l3_512 = self.reduce_l3(x_l3)
        x_l4_512 = self.reduce_l4(x_l4)
        x_bb_4 = self.fuse_l34(torch.cat([x_l3_512, x_l4_512], 1))

        vit_feat = self.vit_branch(x_in)
        if vit_feat.shape[-2:] != x_bb_4.shape[-2:]:
            vit_feat = F.interpolate(vit_feat, size=x_bb_4.shape[-2:],
                                     mode='bilinear', align_corners=False)

        # [V5] 交叉注意力融合
        fused = self.cross_modal_fusion(x_bb_4, vit_feat)
        x_c = self.cba_after_backbone(fused)

        nl = 256 - int(256 * 0.5)
        out = self.ffc_bottleneck((x_c[:, :nl], x_c[:, nl:]))
        x_c = torch.cat([out[0], out[1]], 1) + x_c

        # 高频增强
        x_c = self.hf_enhance(x_c)

        return x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c


class GradientDecoder(nn.Module):
    """
    梯度解码器：接收编码器特征，输出梯度图 x_g 和 4 级待增强特征。
    各级都穿插 FFC 模块以捕获频域全局信息。
    """

    def __init__(self):
        super(GradientDecoder, self).__init__()
        self.fuse_global_context = Conv2D_BN_activa(512 + 256, 512, 1, 1, 0)

        self.block_1_conv_1 = Conv2D_BN_activa(512, 1024, 7, 1, 3)
        self.block_1_conv_2 = Conv2D_BN_activa(1024, 512, 1, 1, 0)
        self.block_1_conv_3 = Conv2D_BN_activa(512, 256, 3, 1, 1)
        self.ffc_grad_h16 = FFC_BN_ACT(256, 256, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.block_1_conv_transpose = Conv2D_BN_activa(256, 256, 5, 1, 2, transpose=False)

        self.block_2_conv = Conv2D_BN_activa(256 + 512, 128, 3, 1, 1)
        self.ffc_grad_h8 = FFC_BN_ACT(128, 128, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.block_2_conv_transpose = Conv2D_BN_activa(128, 128, 4, 2, 1, transpose=True)

        self.block_3_conv = Conv2D_BN_activa(128 + 256, 64, 3, 1, 1)
        self.ffc_grad_h4 = FFC_BN_ACT(64, 64, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.block_3_conv_transpose = Conv2D_BN_activa(64, 64, 4, 2, 1, transpose=True)

        self.block_4_conv = Conv2D_BN_activa(64 + 128, 32, 3, 1, 1)
        self.block_4_conv_transpose = Conv2D_BN_activa(32, 32, 4, 2, 1, transpose=True)

        self.block_5_conv_1 = Conv2D_BN_activa(32 + 64, 64, 4, 2, 1, transpose=True)
        self.block_5_conv_2 = Conv2D_BN_activa(64, 1, 5, 1, 2, activation=None, if_bn=False)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, backbone_features, x_c):
        x_bb_1, x_bb_2, x_bb_3, x_bb_4 = backbone_features

        if x_c.shape[-2:] != x_bb_4.shape[-2:]:
            x_c = F.interpolate(x_c, size=x_bb_4.shape[-2:],
                                mode="bilinear", align_corners=False)
        x_bb_4_fused = self.fuse_global_context(torch.cat([x_bb_4, x_c], dim=1))

        x = self.block_1_conv_1(x_bb_4_fused)
        x = self.block_1_conv_2(x)
        x = self.block_1_conv_3(x)

        _nl_h16 = 256 - int(256 * 0.25)
        _ffc_h16_out = self.ffc_grad_h16((x[:, :_nl_h16], x[:, _nl_h16:]))
        x = torch.cat([_ffc_h16_out[0], _ffc_h16_out[1]], dim=1) + x

        x_to_be_enhanced_1 = self.block_1_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_1, x_bb_4], dim=1)

        x = self.block_2_conv(x)
        _nl_h8 = 128 - int(128 * 0.25)
        _ffc_h8_out = self.ffc_grad_h8((x[:, :_nl_h8], x[:, _nl_h8:]))
        x = torch.cat([_ffc_h8_out[0], _ffc_h8_out[1]], dim=1) + x

        x_to_be_enhanced_2 = self.block_2_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_2, x_bb_3], dim=1)

        x = self.block_3_conv(x)
        _nl_h4 = 64 - int(64 * 0.25)
        _ffc_h4_out = self.ffc_grad_h4((x[:, :_nl_h4], x[:, _nl_h4:]))
        x = torch.cat([_ffc_h4_out[0], _ffc_h4_out[1]], dim=1) + x

        x_to_be_enhanced_3 = self.block_3_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_3, x_bb_2], dim=1)

        x = self.block_4_conv(x)
        x_to_be_enhanced_4 = self.block_4_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_4, x_bb_1], dim=1)

        x = self.block_5_conv_1(x)
        x = self.block_5_conv_2(x)
        x_g = self.sigmoid_layer(x)

        return x_to_be_enhanced_1, x_to_be_enhanced_2, x_to_be_enhanced_3, x_to_be_enhanced_4, x_g


class MainDecoder(nn.Module):
    """
    主解码器：融合编码器 skip connection + 梯度分支增强特征 + 梯度图，
    逐级上采样并最终输出镜面分量 est_b。
    每级 transposed_block 输出后使用 SpecularAwareAttention，
    利用原始输入亮度作为先验增强高光区域响应。
    """

    def __init__(self, in_channels):
        super().__init__()

        self.FFC1 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)
        self.FFC2 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)

        self.multiscale_conv = MultiscaleConvBlock(in_channels)
        self.transposed_block_1 = TransposedConvBlock_same(192 * 3, 256)
        self.transposed_block_2 = TransposedConvBlock(256 * 3 + 128 + 512, 128)
        self.feature_extraction_B = FeatureExtrationLayersB(128 * 3 + 64 + 256)
        self.transposed_block_3 = TransposedConvBlock(640, 64)
        self.transposed_block_4 = TransposedConvBlock(64 * 3 + 32 + 128, 32)
        self.transposed_block_5 = TransposedConvBlock(32 * 3 + 16 + 64, 16)

        self.att1 = Attention_block(768, 512, 64)
        self.att2 = Attention_block(384, 256, 64)
        self.att3 = Attention_block(192, 128, 32)
        self.att4 = Attention_block(96, 64, 16)

        self.ffc_h8 = FFC_BN_ACT(384, 384, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.ffc_h4 = FFC_BN_ACT(192, 192, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)

        # [P2] 扩展输出头：4 层 + 膨胀卷积 + skip connection
        _out_in = 16 * 3
        self.final_grad_scale = 0.25
        self.final_grad_proj = nn.Sequential(
            nn.Conv2d(1, _out_in, 3, 1, 1, bias=True),
            nn.Tanh(),
        )
        self.output_reduce = Conv2D_BN_activa(_out_in, 32, 1, 1, 0)
        self.output_dilated = Conv2D_BN_activa(32, 32, 3, 1, 2, dilation=2)
        self.output_conv3 = Conv2D_BN_activa(32, 16, 3, 1, 1)
        self.output_conv4 = Conv2D_BN_activa(16, 1, 3, 1, 1, activation=None, if_bn=False)
        self.output_sigmoid = nn.Sigmoid()
        self.output_skip = nn.Conv2d(32, 32, 1, 1, 0, bias=False)

        self.sa1 = SpecularAwareAttention(256 * 3, threshold=0.8)
        self.sa2 = SpecularAwareAttention(128 * 3, threshold=0.8)
        self.sa3 = SpecularAwareAttention(64 * 3, threshold=0.8)
        self.sa4 = SpecularAwareAttention(32 * 3, threshold=0.8)

        # [P10] 梯度图调制门控
        self.grad_gate_scale = 0.5
        self.grad_gate_2 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Tanh())
        self.grad_gate_3 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Tanh())
        self.grad_gate_4 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Tanh())

    def forward(self, c_features, enhanced_features, x_g, original_input):
        x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c = c_features
        xe1, xe2, xe3, xe4 = enhanced_features

        x0, x1 = x_c.chunk(2, dim=1)
        x = self.FFC1((x0, x1)); x = self.FFC2(x)
        x = torch.cat([x[0], x[1]], 1) + x_c

        x = self.multiscale_conv(x)

        # Level 1
        x = self.transposed_block_1(x)
        x = self.sa1(x, original_input)
        if x_bb_4.shape[-2:] != x.shape[-2:]:
            x_bb_4 = F.interpolate(x_bb_4, x.shape[-2:], mode='bilinear', align_corners=False)
        if xe1.shape[-2:] != x.shape[-2:]:
            xe1 = F.interpolate(xe1, x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att1(x, x_bb_4)
        x = torch.cat([x, bridge, xe1], 1)

        # Level 2
        x = self.transposed_block_2(x)
        nl = 384 - int(384 * 0.25)
        fo = self.ffc_h8((x[:, :nl], x[:, nl:]))
        x = torch.cat([fo[0], fo[1]], 1) + x
        # [P10] 梯度注入 Level 2
        _xg2 = F.interpolate(x_g, x.shape[-2:], mode='bilinear', align_corners=False) if x_g.shape[-2:] != x.shape[-2:] else x_g
        x = x * (1.0 + self.grad_gate_scale * self.grad_gate_2(_xg2))
        x = self.sa2(x, original_input)
        if x_bb_3.shape[-2:] != x.shape[-2:]:
            x_bb_3 = F.interpolate(x_bb_3, x.shape[-2:], mode='bilinear', align_corners=False)
        if xe2.shape[-2:] != x.shape[-2:]:
            xe2 = F.interpolate(xe2, x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att2(x, x_bb_3)
        x = torch.cat([x, bridge, xe2], 1)

        # Level 3
        x = self.feature_extraction_B(x)
        x = self.transposed_block_3(x)
        nl4 = 192 - int(192 * 0.25)
        fo4 = self.ffc_h4((x[:, :nl4], x[:, nl4:]))
        x = torch.cat([fo4[0], fo4[1]], 1) + x
        # [P10] 梯度注入 Level 3
        _xg3 = F.interpolate(x_g, x.shape[-2:], mode='bilinear', align_corners=False) if x_g.shape[-2:] != x.shape[-2:] else x_g
        x = x * (1.0 + self.grad_gate_scale * self.grad_gate_3(_xg3))
        x = self.sa3(x, original_input)
        if x_bb_2.shape[-2:] != x.shape[-2:]:
            x_bb_2 = F.interpolate(x_bb_2, x.shape[-2:], mode='bilinear', align_corners=False)
        if xe3.shape[-2:] != x.shape[-2:]:
            xe3 = F.interpolate(xe3, x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att3(x, x_bb_2)
        x = torch.cat([x, bridge, xe3], 1)

        # Level 4
        x = self.transposed_block_4(x)
        # [P10] 梯度注入 Level 4
        _xg4 = F.interpolate(x_g, x.shape[-2:], mode='bilinear', align_corners=False) if x_g.shape[-2:] != x.shape[-2:] else x_g
        x = x * (1.0 + self.grad_gate_scale * self.grad_gate_4(_xg4))
        x = self.sa4(x, original_input)
        if x_bb_1.shape[-2:] != x.shape[-2:]:
            x_bb_1 = F.interpolate(x_bb_1, x.shape[-2:], mode='bilinear', align_corners=False)
        if xe4.shape[-2:] != x.shape[-2:]:
            xe4 = F.interpolate(xe4, x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att4(x, x_bb_1)
        x = torch.cat([x, bridge, xe4], 1)

        # Output
        x = self.transposed_block_5(x)
        if x_g.shape[-2:] != x.shape[-2:]:
            x_g = F.interpolate(x_g, x.shape[-2:], mode='bilinear', align_corners=False)
        x = x * (1.0 + self.final_grad_scale * self.final_grad_proj(x_g))

        # [P2] 扩展输出头 with skip connection
        x = self.output_reduce(x)
        skip = self.output_skip(x)
        x = self.output_dilated(x)
        x = x + skip
        x = self.output_conv3(x)
        return self.output_sigmoid(self.output_conv4(x))


class _SpectralDecompositionUnit(nn.Module):
    """梯度分解单元，仅用于 EnhancementLayers 内部。"""

    def __init__(self, in_c: int, out_c: int):
        super(_SpectralDecompositionUnit, self).__init__()
        self.out_c = out_c
        self.projection_branch = Conv2D_BN_activa(in_c, out_c, 7, 1, 3)
        self.specular_branch = Conv2D_BN_activa(in_c, out_c, 3, 1, 1)
        self.ffc_specular = FFC_BN_ACT(out_c, out_c, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.gate_scale = 0.5
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = self.projection_branch(x)
        specular = self.specular_branch(x)

        _nl_spec = self.out_c - int(self.out_c * 0.25)
        _ffc_spec_out = self.ffc_specular((specular[:, :_nl_spec], specular[:, _nl_spec:]))
        specular = torch.cat([_ffc_spec_out[0], _ffc_spec_out[1]], dim=1) + specular

        gate = self.gate_conv(torch.cat([projection, specular], dim=1))
        specular_refined = specular * (1.0 + self.gate_scale * gate)
        return specular_refined


class EnhancementLayers(nn.Module):
    """增强层：对 GradientDecoder 输出的 4 级特征做频域精炼。"""

    def __init__(self):
        super(EnhancementLayers, self).__init__()
        self.decomp_unit_1 = _SpectralDecompositionUnit(in_c=256, out_c=128)
        self.decomp_unit_2 = _SpectralDecompositionUnit(in_c=128, out_c=64)
        self.decomp_unit_3 = _SpectralDecompositionUnit(in_c=64, out_c=32)
        self.decomp_unit_4 = _SpectralDecompositionUnit(in_c=32, out_c=16)

    def forward(self, to_be_enhanced_features):
        x1 = self.decomp_unit_1(to_be_enhanced_features[0])
        x2 = self.decomp_unit_2(to_be_enhanced_features[1])
        x3 = self.decomp_unit_3(to_be_enhanced_features[2])
        x4 = self.decomp_unit_4(to_be_enhanced_features[3])
        return x1, x2, x3, x4


class ResidualRefinementHead(nn.Module):
    """[P1] 残差精修头：提升硬减法后局部结构纠错能力，输出仍保持 [0, 1]。"""

    def __init__(self, scale=0.10, channels=32):
        super().__init__()
        self.scale = scale
        self.entry = nn.Sequential(
            nn.Conv2d(3, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 2, dilation=2, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.out = nn.Conv2d(channels, 1, 3, 1, 1, bias=True)
        self.direct = nn.Conv2d(channels, 1, 3, 1, 1, bias=True)
        self.blend = nn.Conv2d(channels, 1, 3, 1, 1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.direct.weight)
        nn.init.zeros_(self.direct.bias)
        nn.init.zeros_(self.blend.weight)
        nn.init.constant_(self.blend.bias, -12.0)

    def forward(self, coarse, original, specular):
        feat = self.entry(torch.cat([coarse, original, specular], dim=1))
        refined_feat = feat + self.refine(feat)
        delta = torch.tanh(self.out(refined_feat))
        residual = torch.clamp(coarse + self.scale * delta, 0.0, 1.0)
        direct = torch.sigmoid(self.direct(refined_feat))
        blend = torch.sigmoid(self.blend(refined_feat))
        return torch.clamp(residual * (1.0 - blend) + direct * blend, 0.0, 1.0)


class YFFC(nn.Module):
    """Y-FFC 主网络（V3 优化版）。含 P1 残差精修头。"""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.gradient_decoder = GradientDecoder()
        self.enhancement_layers = EnhancementLayers()
        self.main_decoder = MainDecoder(in_channels=256)
        self.refinement_head = ResidualRefinementHead(scale=0.10)  # [P1]

    def forward(self, x):
        c_feat = self.encoder(x)
        g_feat = self.gradient_decoder(c_feat[:-1], c_feat[-1])
        enh = self.enhancement_layers(g_feat[:-1])
        est_b = self.main_decoder(c_feat, enh, g_feat[-1], original_input=x)
        coarse = torch.clamp(x - est_b, 0.0, 1.0)
        pred_diffuse = self.refinement_head(coarse, x, est_b)  # [P1]
        return {"r": pred_diffuse, "g": g_feat[-1], "b": est_b}


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║          第 7 层：训练工具                                                  ║
# ║                                                                          ║
# ║  包括：EMA / 分层学习率优化器 / Warmup+余弦退火调度器 / 动态损失权重        ║
# ║        调度器 / BN→GN 替换 / 单 epoch 训练 / 验证 / 训练入口              ║
# ╚════════════════════════════════════════════════════════════════════════════╝


class EMA:
    """指数移动平均模型参数，提升推理稳定性。"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + \
                              (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class DynamicLossWeightScheduler:
    """根据训练进度动态调整各 loss 项的权重（修复版：60% 后冻结）。"""

    def __init__(self, total_epochs):
        self.total_epochs = max(total_epochs, 1)

    def get_weights(self, current_epoch):
        progress = min(current_epoch / self.total_epochs, 1.0)
        # 修复：在 60% 处截断变化，之后权重不再改变
        p = min(progress / 0.6, 1.0)
        weights = {
            'lambda_ssim':       1.0,
            'lambda_l1':         max(0.5, 1.0 - 0.5 * p),
            'lambda_freq':       min(0.15, 0.05 + 0.10 * p),
            'lambda_perceptual': min(0.10, 0.02 + 0.08 * p),
            'lambda_spec':       0.5,
            'lambda_stat':       0.0,     # 移除冗余 stat
            'lambda_grad':       max(0.3, 0.5 - 0.2 * p),
            'lambda_focal':      min(0.8, 0.4 + 0.4 * p),
            'lambda_dice':       min(0.3, 0.1 + 0.2 * p),
            'lambda_gc':         min(0.2, 0.1 + 0.1 * p),
        }
        return weights

    def step(self, criterion, current_epoch):
        weights = self.get_weights(current_epoch)
        criterion.update_weights(**weights)
        return weights


def build_optimizer(model, base_lr=2e-4, weight_decay=1e-4):
    """为 YFFC 模型构建分层学习率 AdamW 优化器。"""
    encoder_pretrained = []
    ffc_params = []
    decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ('encoder.vit_branch.vit' in name or
            'encoder.stem' in name or
            'encoder.layer' in name):
            encoder_pretrained.append(param)
        elif 'ffc' in name.lower():
            ffc_params.append(param)
        elif ('gradient_decoder' in name or
              'main_decoder' in name or
              'enhancement_layers' in name):
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': encoder_pretrained, 'lr': base_lr * 0.1, 'name': 'encoder_pretrained'},
        {'params': other_params, 'lr': base_lr * 0.5, 'name': 'fusion_layers'},
        {'params': decoder_params, 'lr': base_lr, 'name': 'decoders'},
        {'params': ffc_params, 'lr': base_lr * 0.5, 'name': 'ffc_modules'},
    ]
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    for g in param_groups:
        n_params = sum(p.numel() for p in g['params'])
        print(f"  优化器分组 [{g['name']}]: {n_params:,} 参数, lr={g['lr']:.2e}")

    return optimizer


def build_scheduler(optimizer, warmup_iters=500, total_epochs=200,
                    iters_per_epoch=100, eta_min=1e-6):
    """
    [P6-FIX] Warmup + 余弦退火调度器（修复版：防止 LR 回弹）。

    关键修复：
      原版使用 CosineAnnealingLR，当训练超过 T_max 迭代后，LR 会自动周期性
      回升，导致训练后期损失突然增大。

      修复方案：使用 LambdaLR + 自定义余弦衰减函数，超过 T_max 后 LR 钳位在
      eta_min，不再回弹。
    """
    from torch.optim.lr_scheduler import LambdaLR

    total_iters = total_epochs * iters_per_epoch
    cosine_iters = max(total_iters - warmup_iters, 1)

    # 自定义 warmup + 余弦衰减：超过 cosine_iters 后 LR 钳位在 eta_min，不循环
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def _make_schedule(base_lr):
        min_ratio = min(1.0, eta_min / base_lr) if base_lr > 0 else 1.0

        def _warmup_cosine_no_restart(current_iter):
            """每个参数组独立 warmup + 余弦退火，到达 T_max 后保持自己的 eta_min。"""
            if warmup_iters > 0 and current_iter < warmup_iters:
                warmup_progress = current_iter / warmup_iters
                return 0.01 + (1.0 - 0.01) * warmup_progress

            cosine_iter = current_iter - warmup_iters
            if cosine_iter >= cosine_iters:
                return min_ratio
            progress = cosine_iter / cosine_iters
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine_decay

        return _warmup_cosine_no_restart

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=[_make_schedule(base_lr) for base_lr in base_lrs])
    return scheduler


def replace_bn_with_gn(module, num_groups=8):
    """递归将模型中所有 BatchNorm2d 替换为 GroupNorm。"""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            groups = min(num_groups, num_channels)
            while num_channels % groups != 0 and groups > 1:
                groups -= 1
            gn = nn.GroupNorm(groups, num_channels, eps=child.eps)
            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
                gn.weight.requires_grad = child.weight.requires_grad
                gn.bias.requires_grad = child.bias.requires_grad
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler,
                    scaler, ema, device,
                    accumulation_steps=4, max_grad_norm=0.5,
                    epoch=0, total_epochs=200):
    """单个 epoch 的训练循环（AMP + 梯度累积 + EMA）。"""
    model.train()
    running_loss = 0.0
    running_ssim = 0.0
    n_batches = 0

    optimizer.zero_grad()
    valid_accum = 0

    for i, (input_img, target_img, gt_specular, gt_gradient) in enumerate(dataloader):
        input_img = input_img.to(device)
        target_img = target_img.to(device)
        gt_specular = gt_specular.to(device)
        gt_gradient = gt_gradient.to(device)

        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            output = model(input_img)
            pred_diffuse = output['r']
            pred_grad = output['g']
            est_b = output['b']

            with torch.amp.autocast(device_type='cuda', enabled=False):
                loss, loss_dict = criterion(
                    pred_diffuse.float(), target_img.float(),
                    est_b.float(), gt_specular.float(),
                    pred_grad.float(), gt_gradient.float())

            loss = loss / accumulation_steps

        # [FIX] NaN / Inf 保护
        if not torch.isfinite(loss):
            print(f'[WARNING] Epoch {epoch+1} Batch {i+1}: loss NaN/Inf, 跳过')
            continue

        if device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()
        valid_accum += 1

        if valid_accum >= accumulation_steps or ((i + 1) == len(dataloader) and valid_accum > 0):
            if device.type == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            if ema is not None:
                ema.update()
            if scheduler is not None:
                scheduler.step()
            valid_accum = 0

        running_loss += loss_dict['total']
        running_ssim += (1.0 - loss_dict['ssim'])
        n_batches += 1

    if valid_accum > 0:
        if device.type == 'cuda':
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
        optimizer.zero_grad()
        if ema is not None:
            ema.update()
        if scheduler is not None:
            scheduler.step()

    avg_loss = running_loss / max(n_batches, 1)
    avg_ssim = running_ssim / max(n_batches, 1)
    return avg_loss, avg_ssim


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """验证循环。"""
    model.eval()
    running_loss = 0.0
    running_ssim = 0.0
    n_batches = 0

    for input_img, target_img, gt_specular, gt_gradient in dataloader:
        input_img = input_img.to(device)
        target_img = target_img.to(device)
        gt_specular = gt_specular.to(device)
        gt_gradient = gt_gradient.to(device)

        output = model(input_img)
        loss, loss_dict = criterion(
            output['r'], target_img, output['b'], gt_specular,
            output['g'], gt_gradient)

        running_loss += loss_dict['total']
        running_ssim += (1.0 - loss_dict['ssim'])
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    avg_ssim = running_ssim / max(n_batches, 1)
    return avg_loss, avg_ssim


def train(args):
    """完整训练入口。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备: {device}")

    model = YFFC()
    print("使用 YFFC 模型")

    if args.use_gn:
        print("将 BatchNorm 替换为 GroupNorm...")
        replace_bn_with_gn(model, num_groups=8)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 数据集
    paired_aug = PairedTransform() if args.use_augmentation else None
    train_dataset = FringeDataset(args.input_dir, args.target_dir,
                                  resize=args.image_size, paired_transform=paired_aug)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == 'cuda'), drop_last=False)

    val_loader = None
    if args.val_input_dir and args.val_target_dir:
        val_dataset = FringeDataset(args.val_input_dir, args.val_target_dir,
                                    resize=args.image_size, paired_transform=None)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=(device.type == 'cuda'))

    # 损失函数
    criterion = EnhancedCompositeLoss(use_perceptual=True).to(device)
    loss_weight_scheduler = DynamicLossWeightScheduler(total_epochs=args.epochs)

    # 优化器 + 调度器
    optimizer = build_optimizer(model, base_lr=args.base_lr, weight_decay=1e-4)
    iters_per_epoch = max(1, math.ceil(len(train_loader) / args.accumulation_steps))
    scheduler = build_scheduler(optimizer, warmup_iters=min(500, iters_per_epoch * 3),
                                total_epochs=args.epochs, iters_per_epoch=iters_per_epoch)

    scaler = torch.amp.GradScaler(device='cuda', enabled=(device.type == 'cuda'))
    ema = EMA(model, decay=0.999) if args.use_ema else None

    best_val_ssim = 0.0
    print(f"\n开始训练: {args.epochs} epochs, batch_size={args.batch_size}")
    print("=" * 70)

    for epoch in range(args.epochs):
        t0 = time.time()
        weights = loss_weight_scheduler.step(criterion, epoch)

        train_loss, train_ssim = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, ema, device, accumulation_steps=args.accumulation_steps,
            epoch=epoch, total_epochs=args.epochs)

        elapsed = time.time() - t0

        val_loss, val_ssim = 0.0, 0.0
        if val_loader is not None:
            if ema is not None:
                ema.apply_shadow()
            val_loss, val_ssim = validate(model, val_loader, criterion, device)
            if ema is not None:
                ema.restore()

        current_lr = optimizer.param_groups[0]['lr']
        log_msg = f"Epoch [{epoch + 1:03d}/{args.epochs}] train_loss={train_loss:.4f} train_SSIM={train_ssim:.4f} "
        if val_loader is not None:
            log_msg += f"val_loss={val_loss:.4f} val_SSIM={val_ssim:.4f} "
        log_msg += f"lr={current_lr:.2e} time={elapsed:.1f}s"
        print(log_msg)

        save_ssim = val_ssim if val_loader is not None else train_ssim
        if save_ssim > best_val_ssim:
            best_val_ssim = save_ssim
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ssim': best_val_ssim,
            }
            if ema is not None:
                save_dict['ema'] = ema.shadow
                save_dict['ema_shadow'] = ema.shadow
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(save_dict, save_path)
            print(f"  -> 最佳模型已保存: SSIM={best_val_ssim:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch + 1}.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}
            if ema is not None:
                ckpt['ema'] = ema.shadow
                ckpt['ema_shadow'] = ema.shadow
            torch.save(ckpt, ckpt_path)

    print("=" * 70)
    print(f"训练完成！最佳 SSIM: {best_val_ssim:.4f}")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║          第 8 层：辅助工具 + 主入口                                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝


def display_basic_model_info(model):
    print("\n" + "=" * 60)
    print("模型层次结构")
    print("=" * 60)
    print("\n【主要模块】")
    for name, module in model.named_children():
        print(f"  * {name}: {module.__class__.__name__}")
    print("\n【详细结构】")
    for name, module in model.named_children():
        print(f"\n{name} ({module.__class__.__name__}):")
        for sub_name, sub_module in module.named_children():
            print(f"  -- {sub_name}: {sub_module.__class__.__name__}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数: {total_params:,}, 可训练: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    print("\n【各模块参数量】")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        pct = (module_params / total_params * 100) if total_params > 0 else 0
        print(f"  * {name}: {module_params:,} ({pct:.1f}%)")


def display_model_summary(model, input_size, device="cpu"):
    try:
        from torchinfo import summary as torchinfo_summary
        torchinfo_summary(model, input_size=input_size, device=device, verbose=1)
        return True
    except ImportError:
        try:
            from torchsummary import summary as torchsummary_summary
            torchsummary_summary(model, input_size=input_size[1:], device=device)
            return True
        except ImportError:
            display_basic_model_info(model)
            return False


def main():
    parser = argparse.ArgumentParser(description="Y-FFC 统一版——网络测试与训练")
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # 子命令 1: test
    test_parser = subparsers.add_parser('test', help='网络前向测试')
    test_parser.add_argument("--input_channels", type=int, default=1)
    test_parser.add_argument("--image_size", type=int, default=512)
    test_parser.add_argument("--batch_size", type=int, default=1)
    test_parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    test_parser.add_argument("--show_params", action="store_true", default=True)
    test_parser.add_argument("--show_summary", action="store_true", default=True)

    # 子命令 2: train
    train_parser = subparsers.add_parser('train', help='模型训练')
    train_parser.add_argument("--input_dir", type=str, required=True)
    train_parser.add_argument("--target_dir", type=str, required=True)
    train_parser.add_argument("--val_input_dir", type=str, default=None)
    train_parser.add_argument("--val_target_dir", type=str, default=None)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch_size", type=int, default=2)
    train_parser.add_argument("--image_size", type=int, default=512)
    train_parser.add_argument("--base_lr", type=float, default=2e-4)
    train_parser.add_argument("--accumulation_steps", type=int, default=4)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--save_dir", type=str, default="./checkpoints")
    train_parser.add_argument("--save_interval", type=int, default=10)
    train_parser.add_argument("--use_gn", action="store_true")
    train_parser.add_argument("--use_ema", action="store_true")
    train_parser.add_argument("--use_augmentation", action="store_true", default=True)
    train_parser.add_argument("--no_augmentation", action="store_true")

    args = parser.parse_args()
    if getattr(args, 'command', None) == 'train' and getattr(args, 'no_augmentation', False):
        args.use_augmentation = False

    if args.command is None:
        args.command = 'test'
        args.input_channels = 1
        args.image_size = 512
        args.batch_size = 1
        args.device = 'cpu'
        args.show_params = True
        args.show_summary = True

    if args.command == 'test':
        print("=" * 60)
        print("Y-FFC 统一版 网络前向测试")
        print("=" * 60)

        device = torch.device(args.device)
        if args.device == "cuda" and not torch.cuda.is_available():
            print("警告: CUDA 不可用，切换到 CPU")
            device = torch.device("cpu")

        # 创建模型
        model = YFFC()
        print("测试模型: YFFC")
        model.to(device)

        input_size = (args.batch_size, args.input_channels, args.image_size, args.image_size)
        test_input = torch.randn(input_size).to(device)

        # 前向传播测试
        try:
            with torch.no_grad():
                out = model(test_input)
            print(f"前向传播成功: r={out['r'].shape}, g={out['g'].shape}, b={out['b'].shape}")
            assert out["r"].shape == test_input.shape
            assert out["g"].shape == test_input.shape
            assert out["b"].shape == test_input.shape
            print("输出 shape 校验通过")
        except Exception as e:
            print(f"前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # 损失函数测试
        print("\n--- 损失函数测试 ---")
        try:
            criterion = EnhancedCompositeLoss(use_perceptual=False).to(device)
            target = torch.rand_like(out['r'])
            gt_specular = (test_input - target).clamp(0, 1)
            gt_gradient = torch.rand_like(out['g'])
            with torch.no_grad():
                loss, loss_dict = criterion(
                    out['r'], target, out['b'], gt_specular, out['g'], gt_gradient)
            print(f"EnhancedCompositeLoss 前向成功, total={loss.item():.4f}")
        except Exception as e:
            print(f"损失函数测试失败: {e}")

        # EMA 测试
        print("\n--- EMA 测试 ---")
        try:
            ema = EMA(model, decay=0.999)
            ema.update(); ema.apply_shadow(); ema.restore()
            print("EMA 创建/更新/切换/恢复 成功")
        except Exception as e:
            print(f"EMA 测试失败: {e}")

        # BN → GN 替换测试
        print("\n--- BN -> GN 替换测试 ---")
        try:
            model_gn = YFFC()
            bn_count = sum(1 for m in model_gn.modules() if isinstance(m, nn.BatchNorm2d))
            replace_bn_with_gn(model_gn)
            gn_count = sum(1 for m in model_gn.modules() if isinstance(m, nn.GroupNorm))
            bn_after = sum(1 for m in model_gn.modules() if isinstance(m, nn.BatchNorm2d))
            print(f"替换前 BN: {bn_count}, 替换后 GN: {gn_count}, 剩余 BN: {bn_after}")
            model_gn.to(device)
            with torch.no_grad():
                out_gn = model_gn(test_input)
            assert out_gn["r"].shape == test_input.shape
            print("BN->GN 替换后前向传播正常")
            del model_gn
        except Exception as e:
            print(f"BN->GN 测试失败: {e}")

        if args.show_params:
            display_basic_model_info(model)

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)

    elif args.command == 'train':
        if hasattr(args, 'no_augmentation') and args.no_augmentation:
            args.use_augmentation = False
        train(args)


if __name__ == "__main__":
    main()
