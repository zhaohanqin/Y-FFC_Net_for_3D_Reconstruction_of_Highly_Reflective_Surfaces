"""
X_FFC_unified.py - Y-FFC / X-FFC-Swin 去高反光网络
=====================================================

本文件包含 Y-FFC / X-FFC-Swin 网络的完整实现，包括以下核心组件：

  • FFC 核心模块（FourierUnit / SpectralTransform / FFC_BN_ACT / FFCResnetBlock）
  • 网络基础构件（Conv2D_BN_activa / Attention_block / MultiscaleConvBlock 等）
  • Swin-Tiny 编码器分支（SwinEncoderBranch）与 CrossModalFusion 融合模块
  • 高反光专用模块（SpecularAwareAttention / HighFrequencyEnhancement）
  • 损失函数与评价指标（SSIM / PerceptualLoss / CompositeLoss / Specular 损失等）
  • 数据集与数据增强（FringeDataset / PairedTransform / EnhancedPairedTransform 等）
  • 模型结构（Encoder / GradientDecoder / MainDecoder / EnhancementLayers / XFFC）
  • 训练工具（EMA / build_optimizer / build_scheduler / train_one_epoch 等）

第一阶段 Swin 编码器改造：
─────────────────────────────────────────────────
  S1  使用 torchvision.models.swin_t 作为窗口上下文分支，默认加载 ImageNet 预训练权重
  S2  将 Swin patch embedding 改为单通道输入，用 RGB 预训练权重均值初始化灰度卷积核
  S3  Swin 分支直接处理训练/推理 patch，不再强制 resize 到 224x224
  S4  对不能被 32 整除的输入做右/下 padding，保证 Swin 分层下采样可以正常前向
  S5  只取 Swin H/16 stage 特征，NHWC 转 NCHW 后投影到 512 通道
  S6  ResNet x_bb_4 与 Swin H/16 特征通过 CrossModalFusion 做瓶颈层融合
─────────────────────────────────────────────────

文件组织结构（自上而下）：
─────────────────────────────────────────────────
  第 0 层  外部依赖导入 + 通用工具函数
  第 1 层  FFC 核心模块（FourierUnit / SpectralTransform / FFC_BN_ACT / FFCResnetBlock）
  第 2 层  网络基础构件（Conv2D_BN_activa / Attention_block / SwinEncoderBranch / ...）
  第 3 层  损失函数与评价指标（SSIM / PerceptualLoss / CompositeLoss / Specular 损失 ...）
  第 4 层  高反光专用模块（SpecularAwareAttention / HighFrequencyEnhancement）
  第 5 层  数据集与数据增强（FringeDataset / PairedTransform / EnhancedPairedTransform / ...）
  第 6 层  模型结构（Encoder / GradientDecoder / MainDecoder / EnhancementLayers / XFFC）
  第 7 层  训练工具（EMA / build_optimizer / build_scheduler / train_one_epoch / ...）
  第 8 层  主入口（main）
─────────────────────────────────────────────────

输入保持灰度图 [B, 1, H, W]，输出保持：
{"r": pred_diffuse, "g": input_gradient_map, "b": est_specular}
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


def _safe_tensor(tensor, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=None):
    """Numerical guard used only at high-risk boundaries.

    It prevents one unstable FFT/attention activation from turning the whole
    training run into persistent NaN/Inf while keeping normal finite values
    unchanged.
    """
    tensor = torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)
    if clamp is not None:
        tensor = tensor.clamp(clamp[0], clamp[1])
    return tensor


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
                 ffc3d=False, fft_norm='ortho'):
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

        # [F6][P11] Tukey 窗（α=0.25）：减少频谱泄漏，保留边缘 75% 能量
        h, w = x.shape[-2], x.shape[-1]
        win_h = _tukey_window(h, alpha=0.25, device=x.device, dtype=x.dtype)
        win_w = _tukey_window(w, alpha=0.25, device=x.device, dtype=x.dtype)
        tukey_2d = torch.outer(win_h, win_w).unsqueeze(0).unsqueeze(0)
        x_win = x * tukey_2d

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

        output = _safe_tensor(output, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=(-1.0e4, 1.0e4))

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
# ║  Swin 编码器分支、交叉注意力融合模块等。                                    ║
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
        return x * psi


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
    """三路 resize-conv 上采样块，替代转置卷积以降低棋盘伪影。"""

    def __init__(self, in_channels, out_channels_branch=256):
        super(TransposedConvBlock, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1))
        self.branch_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv2D_BN_activa(in_channels, out_channels_branch, 5, 1, 2))
        self.branch_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv2D_BN_activa(in_channels, out_channels_branch, 7, 1, 3))

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


class SwinEncoderBranch(nn.Module):
    """
    Swin-Tiny 单通道多尺度上下文分支，输出 H/4、H/8、H/16 三个层级特征。

    输入为 [B, 1, H, W]，不再像旧 ViT 分支那样 resize 到 224x224。
    Swin 直接在训练/推理 patch 的原始空间采样上建模；如果 H/W 不能被 32 整除，
    只在右侧和下侧做必要 padding，后续在 Encoder 中插值对齐到 ResNet 对应尺度。
    torchvision Swin 中间特征为 NHWC，这里统一转回 NCHW。

    对结构光条纹图像，这一点非常关键：强制 resize 会改变条纹频率、局部相位关系
    和高光边界形态，可能破坏后续相位恢复所依赖的空间采样一致性。Swin 的窗口注意力
    支持可变 patch 尺寸，因此可以保留原始 patch 内的相位、周期和细条纹位置关系。

    在整个模型中的位置：
        被 Encoder 调用，与 ResNet50 并行提取窗口上下文特征。
        H/4 用于增强 x_bb_2，H/8 用于增强 x_bb_3，H/16 用于瓶颈层 CrossModalFusion。

    训练/推理建议：
        第三阶段加入 AdaptiveSkipFusion 后显存会高于第一阶段。
        推荐输入 320x256、384x384、512x512 等 patch，尽量让 H/W 被 32 整除。
        大图不建议直接整图训练，推理时建议 sliding-window + overlap 融合。
    """

    def __init__(self, pretrained=True):
        super().__init__()
        if not hasattr(models, "swin_t"):
            raise RuntimeError("当前 torchvision 版本不包含 swin_t，请升级 torchvision。")

        weights = None
        if pretrained:
            try:
                weights = models.Swin_T_Weights.IMAGENET1K_V1
            except Exception:
                weights = None

        try:
            self.swin = models.swin_t(weights=weights)
        except Exception:
            self.swin = models.swin_t(weights=None)

        # 将 Swin patch embedding 从 RGB 输入改为单通道输入。
        # 若加载了 ImageNet 预训练权重，用 RGB 卷积核的通道均值初始化灰度卷积核。
        patch_embed = self.swin.features[0][0]
        old_weight = patch_embed.weight.detach().clone()
        old_bias = patch_embed.bias.detach().clone() if patch_embed.bias is not None else None
        new_patch_embed = nn.Conv2d(
            1,
            patch_embed.out_channels,
            kernel_size=patch_embed.kernel_size,
            stride=patch_embed.stride,
            padding=patch_embed.padding,
            bias=patch_embed.bias is not None,
        )
        with torch.no_grad():
            new_patch_embed.weight.copy_(old_weight.mean(dim=1, keepdim=True))
            if old_bias is not None:
                new_patch_embed.bias.copy_(old_bias)
        self.swin.features[0][0] = new_patch_embed

        # Swin-Tiny 三个层级的原始输出通道数。
        # H/4=96、H/8=192、H/16=384，具体投影和融合在 Encoder 中完成。
        self.channels_4 = 96
        self.channels_8 = 192
        self.channels_16 = 384

    @staticmethod
    def _pad_to_multiple(x, multiple=32):
        h, w = x.shape[-2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return x
        can_reflect = pad_h < h and pad_w < w and h > 1 and w > 1
        pad_mode = "reflect" if can_reflect else "replicate"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)

    def forward(self, x):
        # 推荐训练/推理使用 320x256、384x384、512x512 等 patch。
        # 这里不做 224x224 resize，只做右/下 padding 到 32 的倍数，保护结构光条纹相位关系。
        # 不建议直接整图训练 2448x2048；大图推理建议 sliding-window + overlap 融合。
        x = self._pad_to_multiple(x, multiple=32)

        feat = x
        swin_4 = swin_8 = swin_16 = None
        # torchvision Swin features:
        #   0/1 -> H/4,  C=96
        #   2/3 -> H/8,  C=192
        #   4/5 -> H/16, C=384
        # Swin 内部特征为 NHWC，返回前统一转换为 NCHW。
        for idx in range(6):
            feat = self.swin.features[idx](feat)
            if idx == 1:
                swin_4 = feat.permute(0, 3, 1, 2).contiguous()
            elif idx == 3:
                swin_8 = feat.permute(0, 3, 1, 2).contiguous()
            elif idx == 5:
                swin_16 = feat.permute(0, 3, 1, 2).contiguous()

        return {
            "swin_4": swin_4,
            "swin_8": swin_8,
            "swin_16": swin_16,
        }


class AdaptiveSkipFusion(nn.Module):
    """
    轻量级自适应 Swin skip 融合模块。

    作用：
      - 先把 Swin skip 投影到 ResNet skip 的通道数；
      - 根据 concat([res_feat, swin_proj]) 生成逐位置、逐通道 gate；
      - 使用 fused = res_feat + beta * gate * swin_proj 做内容自适应残差注入。

    beta 是整体强度控制参数，初始为 0.0，让模型初始时接近纯 ResNet skip，
    避免突然改变 Decoder 已经适配的 skip 特征分布。gate 负责控制不同空间位置、
    不同通道上 Swin 上下文注入多少。H/4、H/8 分辨率较高，因此这里使用轻量门控，
    不直接使用重型 CrossModalFusion；H/16 仍保留 CrossModalFusion 做瓶颈强交互。
    """

    def __init__(self, res_channels, swin_channels, init_beta=0.03, gate_reduction=2):
        super().__init__()
        hidden_channels = max(res_channels // gate_reduction, 16)
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.last_gate_mean = None

        self.swin_proj = nn.Sequential(
            nn.Conv2d(swin_channels, res_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.GELU(),
        )
        self.gate_net = nn.Sequential(
            nn.Conv2d(res_channels * 2, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, res_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        # 初始 gate 偏小，配合 beta=0.0，使训练初期尽量保持 ResNet skip 分布稳定。
        final_gate_conv = self.gate_net[-2]
        if isinstance(final_gate_conv, nn.Conv2d) and final_gate_conv.bias is not None:
            nn.init.constant_(final_gate_conv.bias, -2.0)

    def forward(self, res_feat, swin_feat):
        if swin_feat.shape[-2:] != res_feat.shape[-2:]:
            swin_feat = F.interpolate(
                swin_feat, size=res_feat.shape[-2:],
                mode='bilinear', align_corners=False)

        swin_proj = self.swin_proj(swin_feat)
        gate = self.gate_net(torch.cat([res_feat, swin_proj], dim=1))
        self.last_gate_mean = gate.detach().mean()
        return res_feat + self.beta * gate * swin_proj


class CrossModalFusion(nn.Module):
    """
    轻量级 CNN-Transformer 双向交叉注意力融合模块。

    核心思想：
      - CNN → Swin 方向：ResNet 深层特征作为 Query，查询 Swin 的窗口上下文 Key/Value，
        为局部纹理注入更大范围的反射区域和条纹结构上下文。
      - Swin → CNN 方向：Swin H/16 特征作为 Query，查询 ResNet 的局部细节 Key/Value，
        为窗口上下文补充高光边界、细条纹和局部反射形态。
      - 双向交叉后通过门控机制融合，输出维度为 out_dim。

    使用 sr_ratio 对键值对做空间降采样，降低高分辨率特征上的注意力开销。
    本阶段只用于瓶颈层 H/16 融合，不用于 H/4、H/8 skip 融合。

    输入含义：
      - cnn_feat: ResNet 深层局部纹理特征
      - context_feat: Swin H/16 窗口上下文特征
    """

    def __init__(self, cnn_dim, context_dim, out_dim, num_heads=4, sr_ratio=2):
        super(CrossModalFusion, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.out_dim = out_dim

        self.q_c2ctx = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.k_c2ctx = nn.Conv2d(context_dim, out_dim, 1, bias=False)
        self.v_c2ctx = nn.Conv2d(context_dim, out_dim, 1, bias=False)
        self.proj_c2ctx = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        self.q_ctx2c = nn.Conv2d(context_dim, out_dim, 1, bias=False)
        self.k_ctx2c = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.v_ctx2c = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.proj_ctx2c = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_pool = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)

        # 门控融合：自适应平衡 CNN 查询 Swin 与 Swin 查询 CNN 两个方向的贡献。
        self.gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid(),
        )

        # 最终投影：融合后再做一次局部卷积精炼。
        self.final_proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    @staticmethod
    def _modality_norm(feat):
        channels = feat.shape[1]
        groups = min(32, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return F.group_norm(feat, num_groups=groups)

    def _cross_attn(self, q, k, v, proj):
        """执行单方向的交叉注意力。"""
        B, C, H, W = q.shape
        Hk, Wk = k.shape[-2:]
        nh = self.num_heads
        hd = self.head_dim

        q = q.reshape(B, nh, hd, H * W).permute(0, 1, 3, 2)
        k = k.reshape(B, nh, hd, Hk * Wk).permute(0, 1, 3, 2)
        v = v.reshape(B, nh, hd, Hk * Wk).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return proj(out)

    def forward(self, cnn_feat, context_feat):
        """
        Args:
            cnn_feat: (B, cnn_dim, H, W) — ResNet50 深层输出 x_bb_4
            context_feat: (B, context_dim, H, W) — Swin H/16 输出，已对齐到 x_bb_4
        Returns:
            (B, out_dim, H, W) — 融合后的瓶颈输入特征
        """
        cnn_feat = self._modality_norm(cnn_feat)
        context_feat = self._modality_norm(context_feat)

        if self.sr_ratio > 1:
            context_kv = self.sr_pool(context_feat)
            cnn_kv = self.sr_pool(cnn_feat)
        else:
            context_kv = context_feat
            cnn_kv = cnn_feat

        # CNN 查询 Swin：为 ResNet 局部纹理补充窗口上下文。
        q_c = self.q_c2ctx(cnn_feat)
        k_ctx = self.k_c2ctx(context_kv)
        v_ctx = self.v_c2ctx(context_kv)
        c2ctx_out = self._cross_attn(q_c, k_ctx, v_ctx, self.proj_c2ctx)

        # Swin 查询 CNN：为 Swin 上下文补充高光边界和细条纹局部细节。
        q_ctx = self.q_ctx2c(context_feat)
        k_c = self.k_ctx2c(cnn_kv)
        v_c = self.v_ctx2c(cnn_kv)
        ctx2c_out = self._cross_attn(q_ctx, k_c, v_c, self.proj_ctx2c)

        # 门控融合两个方向的结果。
        gate = self.gate(torch.cat([c2ctx_out, ctx2c_out], dim=1))
        fused = gate * c2ctx_out + (1.0 - gate) * ctx2c_out
        return self.final_proj(fused)


class CrossScaleTopDownRefinement(nn.Module):
    """轻量 H/16 -> H/8 -> H/4 特征回流，使用受限残差避免破坏已稳定的 skip。"""

    def __init__(self, h4_channels=128, h8_channels=256, h16_channels=512, init_gamma=0.05):
        super().__init__()
        self.h16_refine = Conv2D_BN_activa(h16_channels, h16_channels, 3, 1, 1)
        self.h16_to_h8 = Conv2D_BN_activa(h16_channels, h8_channels, 1, 1, 0)
        self.h8_gate = nn.Sequential(
            nn.Conv2d(h8_channels * 2, h8_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.h8_refine = Conv2D_BN_activa(h8_channels, h8_channels, 3, 1, 1)
        self.h8_to_h4 = Conv2D_BN_activa(h8_channels, h4_channels, 1, 1, 0)
        self.h4_gate = nn.Sequential(
            nn.Conv2d(h4_channels * 2, h4_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, h4, h8, h16):
        gamma = self.gamma.clamp(0.0, 0.2)

        h16_ctx = self.h16_refine(h16)
        h16_out = h16 + gamma * h16_ctx

        h8_from_h16 = self.h16_to_h8(h16_ctx)
        h8_from_h16 = F.interpolate(h8_from_h16, size=h8.shape[-2:], mode='bilinear', align_corners=False)
        h8_gate = self.h8_gate(torch.cat([h8, h8_from_h16], dim=1))
        h8_out = h8 + gamma * h8_gate * h8_from_h16

        h8_ctx = self.h8_refine(h8_out)
        h4_from_h8 = self.h8_to_h4(h8_ctx)
        h4_from_h8 = F.interpolate(h4_from_h8, size=h4.shape[-2:], mode='bilinear', align_corners=False)
        h4_gate = self.h4_gate(torch.cat([h4, h4_from_h8], dim=1))
        h4_out = h4 + gamma * h4_gate * h4_from_h8

        return h4_out, h8_out, h16_out


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

        denom = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)).clamp_min(eps)
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / denom
        ssim_map = _safe_tensor(ssim_map, nan=0.0, posinf=1.0, neginf=-0.5)
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
        lum = (2 * mu12 + C1) / (mu1_sq + mu2_sq + C1).clamp_min(1e-8)
        cs = (2 * s12 + C2) / (s1_sq + s2_sq + C2).clamp_min(1e-8)
        lum = _safe_tensor(lum, nan=0.0, posinf=1.0, neginf=-0.5, clamp=(-0.5, 1.0))
        cs = _safe_tensor(cs, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))
        return lum.mean(), cs.mean()

    def forward(self, img1, img2):
        cs_list = []
        for i in range(self.n_scales):
            if i > 0:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
                if img1.shape[-1] < self.window_size or img1.shape[-2] < self.window_size:
                    break
            lum, cs = self._ssim_components(img1, img2)
            cs_list.append(cs)
        n_actual = len(cs_list)
        w = self.scale_weights[:n_actual].to(img1.device)
        w = w / w.sum()
        msssim = 1.0
        for i in range(n_actual):
            msssim = msssim * (torch.clamp(cs_list[i], min=1e-8) ** w[i])
        return msssim


def charbonnier_loss(pred, gt, eps=1e-3):
    """[P13][SAFE] Charbonnier 损失：误差 0 处光滑可导，后期梯度更稳定。

    eps 由 1e-6 提升至 1e-3：在 fp16 下 1e-6 已接近下溢，反向梯度被 1/eps 放大
    可达 1e6 量级；1e-3 在精度损失可忽略（典型误差 0.01~0.5）的同时显著稳定反向。
    """
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


def _as_batched_4d(tensor):
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0), 2
    if tensor.dim() == 3:
        return tensor.unsqueeze(0), 3
    if tensor.dim() == 4:
        return tensor, 4
    raise ValueError(f"Expected a 2D, 3D, or 4D tensor, got shape {tuple(tensor.shape)}")


def _restore_tensor_dim(tensor, original_dim):
    if original_dim == 2:
        return tensor.squeeze(0).squeeze(0)
    if original_dim == 3:
        return tensor.squeeze(0)
    return tensor


def sobel_magnitude(image):
    """[SAFE] Sobel 幅值，反向稳定。

    原版 sqrt(gx²+gy²) 在零梯度区域 sqrt(0) 的反向导数是 ∞。
    加入 eps=1e-6 保护：sqrt(gx²+gy²+1e-6) 在 0 处梯度为 0，反向安全。
    """
    image_b, original_dim = _as_batched_4d(image)
    channels = image_b.size(1)
    dtype = image_b.dtype
    device = image_b.device

    sx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    sy = sx.transpose(-1, -2)
    sx = sx.repeat(channels, 1, 1, 1)
    sy = sy.repeat(channels, 1, 1, 1)

    padded = F.pad(image_b, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, sx, groups=channels)
    gy = F.conv2d(padded, sy, groups=channels)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)  # [SAFE] eps=1e-6 防 sqrt(0) 反向爆炸
    return _restore_tensor_dim(mag, original_dim)


def normalize_per_sample(image, quantile=0.995):
    """[SAFE] 按样本对幅值图按分位数归一化到 [0,1]。

    数值安全要点：
      原版 scale.clamp_min(1e-8) 在反向时，对 `values / scale` 求导会得到
      `-values / scale²` 量级的梯度；当 scale ≈ 1e-8 时反向梯度会被放大 1e16 倍。
      改成 clamp_min(1e-3) 是相对安全的（典型 Sobel 幅值在 [0, sqrt(8)] 之间，
      1e-3 远小于正常 scale，但远大于致命的反向爆炸阈值）。
    """
    image_b, original_dim = _as_batched_4d(image)
    dtype = image_b.dtype
    values = image_b.float()
    flat = values.flatten(1)
    scale = torch.quantile(flat, quantile, dim=1, keepdim=True)
    scale = scale.view(-1, 1, 1, 1).clamp_min(1e-3)
    normalized = (values / scale).clamp(0.0, 1.0).to(dtype)
    return _restore_tensor_dim(normalized, original_dim)


def normalized_sobel_magnitude(image, quantile=0.995):
    return normalize_per_sample(sobel_magnitude(image), quantile=quantile)


def compute_input_gradient_target(input_img):
    """Input-image Sobel gradient target normalized per sample to [0, 1]."""
    return normalized_sobel_magnitude(input_img, quantile=0.995)


def compute_input_gradient_weight_map(gt_gradient, base_weight=1.0,
                                      edge_weight=5.0):
    grad_b, original_dim = _as_batched_4d(gt_gradient)
    weight = base_weight + (edge_weight - base_weight) * grad_b.clamp(0.0, 1.0)
    return _restore_tensor_dim(weight, original_dim)


def compute_highlight_prior(input_img, threshold=0.7, softness=0.1):
    """Soft highlight prior from input brightness, smoothed for stable weighting."""
    img_b, original_dim = _as_batched_4d(input_img)
    img_b = img_b.clamp(0.0, 1.0)
    softness = max(float(softness), 1e-6)
    prior = torch.sigmoid((img_b - threshold) / softness)
    prior = F.avg_pool2d(prior, kernel_size=5, stride=1, padding=2)
    return _restore_tensor_dim(prior.clamp(0.0, 1.0), original_dim)


class PerceptualLoss(nn.Module):
    """
    感知损失（VGG16 特征匹配）。
    使用预训练 VGG16 的浅层/中层特征计算感知距离，参数冻结不参与训练。
    """

    def __init__(self, layers=('relu1_2', 'relu2_2', 'relu3_3'),
                 weights=(1.0, 0.5, 0.25)):
        super(PerceptualLoss, self).__init__()
        self.is_pretrained = False
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
            self.is_pretrained = True
            print("[PerceptualLoss] 已加载 ImageNet 预训练 VGG16，并启用 ImageNet 标准化")
        except Exception as exc:
            print(f"[PerceptualLoss] 预训练 VGG16 加载失败，感知损失将自动关闭: {exc}")
            vgg = models.vgg16(weights=None).features

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
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('imagenet_mean', mean)
        self.register_buffer('imagenet_std', std)

    def forward(self, pred, gt):
        # ★[D8 V3.2] 小尺寸输入 (H<48 或 W<48) 时 relu3_3 层特征几乎无意义，
        # 直接返回 0 避免不稳定梯度污染主任务。
        # 训练分辨率 (128,160) 走全路径；后续 stage 也安全。
        if pred.shape[-1] < 48 or pred.shape[-2] < 48:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
            gt = gt.repeat(1, 3, 1, 1)
        pred = (pred - self.imagenet_mean) / self.imagenet_std
        gt = (gt - self.imagenet_mean) / self.imagenet_std
        loss = 0.0
        x, y = pred, gt
        for i, slice_module in enumerate(self.slices):
            x = slice_module(x)
            y = slice_module(y)
            loss += self.weights[i] * F.l1_loss(x, y)
        return loss


def compute_edge_weight_map(gt_specular, sigma=3.0, base_weight=1.0, edge_weight=3.0):
    """根据镜面反射分量的梯度生成边缘权重图，用于边缘感知加权 L1 损失。

    [SAFE] sqrt eps 由 1e-8 提升至 1e-4，避免低梯度区反向爆炸；
    输入通常已 detach（gt_specular），但保持稳定性约定。
    """
    gx = gt_specular[:, :, :, 1:] - gt_specular[:, :, :, :-1]
    gy = gt_specular[:, :, 1:, :] - gt_specular[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0), mode='replicate')
    gy = F.pad(gy, (0, 0, 0, 1), mode='replicate')
    edge_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-4)

    k = max(3, int(6 * sigma + 1) | 1)
    pad_size = k // 2
    edge_mag = F.avg_pool2d(
        F.pad(edge_mag, [pad_size] * 4, mode='replicate'),
        kernel_size=k, stride=1, padding=0)

    batch_max = edge_mag.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    edge_mag = edge_mag / batch_max.clamp_min(1e-4)
    weight_map = base_weight + (edge_weight - base_weight) * edge_mag
    return weight_map


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
        """[SAFE] 反向稳定的梯度一致性损失。

        原版 eps=1e-8 在 fp32 下也接近下溢边缘；改为 1e-4 大大提升反向稳定性，
        同时对损失的精度影响可忽略（典型梯度幅值在 1e-2~1.0 量级）。
        """
        sx = self.sobel_x.to(pred.dtype)
        sy = self.sobel_y.to(pred.dtype)

        pgx = F.conv2d(pred, sx, padding=1)
        pgy = F.conv2d(pred, sy, padding=1)
        ggx = F.conv2d(gt, sx, padding=1)
        ggy = F.conv2d(gt, sy, padding=1)

        # [SAFE] eps=1e-4：sqrt(0+1e-4)=0.01，远离反向爆炸点
        pm = torch.sqrt(pgx ** 2 + pgy ** 2 + 1e-4)
        gm = torch.sqrt(ggx ** 2 + ggy ** 2 + 1e-4)

        mag_loss = F.l1_loss(pm, gm)

        # [SAFE] 用 detach mask + 较大下界，反向只走 cos 数值通道
        with torch.no_grad():
            mask = (gm > gm.mean() * 0.5).float()
            mask_sum = mask.sum().clamp_min(1.0)
        cos = (pgx * ggx + pgy * ggy) / (pm * gm + 1e-4)
        cos = _safe_tensor(cos, nan=1.0, posinf=1.0, neginf=-1.0, clamp=(-1.0, 1.0))
        dir_loss = ((1.0 - cos) * mask).sum() / mask_sum

        return mag_loss + 0.3 * dir_loss


def specular_ohem_loss(pred, gt, base_fn, ratio=0.3):
    """在线难样本挖掘（OHEM）：只对误差最大的 top-k 像素额外加权。

    [SAFE] mask.sum() 实际至少为 k = ratio*H*W (远大于 1)，
    但保持显式 clamp_min(1.0) 防御性下界，避免极小 patch 上的数值病态。
    """
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
    hard = (torch.abs(pred - gt) * mask).sum() / mask.sum().clamp_min(1.0)
    return 0.5 * base + 0.5 * hard


# --- 复合损失函数 ---

class CompositeLoss(nn.Module):
    """
    多目标复合损失。
    整合 SSIM / L1 / 频域 / 感知 / 镜面直接监督 / 统计 / 梯度 / 边缘加权。

    [NEW] 新增 spec_likelihood 监督项 l_s_lik：
        对 GradientDecoder 输出的高光概率图 pred_s_lik 用 gt_specular 软监督，
        使梯度分支的中间特征获得"任务化"语义信号。
    """

    def __init__(self, lambda_ssim=1.0, lambda_l1=1.0, lambda_freq=0.1,
                 lambda_perceptual=0.05, lambda_spec=0.5, lambda_stat=0.1,
                 lambda_grad=0.5, lambda_s_lik=0.3,
                 use_edge_weight=True, use_perceptual=True):
        super(CompositeLoss, self).__init__()
        self.ssim = SSIM()
        self.lambda_ssim = lambda_ssim
        self.lambda_l1 = lambda_l1
        self.lambda_freq = lambda_freq
        self.lambda_perceptual = lambda_perceptual
        self.lambda_spec = lambda_spec
        self.lambda_stat = lambda_stat
        self.lambda_grad = lambda_grad
        self.lambda_s_lik = lambda_s_lik
        self.use_edge_weight = use_edge_weight
        self.use_perceptual = use_perceptual
        if self.use_perceptual:
            self.perceptual = PerceptualLoss()

    def frequency_loss(self, pred, gt):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        gt_fft = torch.fft.rfft2(gt, norm='ortho')
        return F.l1_loss(torch.abs(pred_fft), torch.abs(gt_fft))

    def forward(self, pred_diffuse, target, est_b, gt_specular,
                pred_grad, gt_grad, pred_s_lik=None):
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

        # [NEW] spec_likelihood 头损失
        if pred_s_lik is None:
            l_s_lik = torch.tensor(0.0, device=pred_diffuse.device)
        else:
            l_s_lik = F.l1_loss(pred_s_lik, gt_specular_clamped) + \
                      0.5 * (1.0 - self.ssim(pred_s_lik.float(), gt_specular_clamped.float()))

        total = (self.lambda_ssim * l_ssim + self.lambda_l1 * l_l1 +
                 self.lambda_freq * l_freq + self.lambda_perceptual * l_perceptual +
                 self.lambda_spec * l_spec + self.lambda_stat * l_stat +
                 self.lambda_grad * l_grad + self.lambda_s_lik * l_s_lik)

        loss_dict = {}
        for k, v in [('total', total), ('ssim', l_ssim), ('l1', l_l1),
                      ('freq', l_freq), ('perceptual', l_perceptual),
                      ('spec', l_spec), ('stat', l_stat), ('grad', l_grad),
                      ('s_lik', l_s_lik)]:
            loss_dict[k] = v.item() if isinstance(v, torch.Tensor) else v

        return total, loss_dict

    def update_weights(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class EnhancedCompositeLoss(nn.Module):
    """
    增强版复合损失（V3.2 优化版 — 针对漫反射 r 增加直接监督，根治"靠减法链"的脆弱）。

    [P3]  SSIM → 0.8*MS-SSIM + 0.2*SSIM 混合
    [P4]  频域损失加入相位一致性约束 (V3.2: phase 项权重 0.25 → 0.15)
    [P5]  移除冗余 statistical_loss（lambda_stat 默认 0）
    [P12] SSIM as_loss=True 保留梯度
    [P13] L1 → Charbonnier 平滑损失

    [V3.1 新增]
      l_s_lik   : spec_lik_head 与 gt_specular 的 BCE+SSIM
      l_s_cons  : pred_s_lik 与 est_b 在高光区域的双向一致性
      l_spec_peak: 高光区 est_b 欠预测单侧惩罚

    ★[V3.2 新增 — 针对漫反射 r 的直接监督]★
      [D3] l_r_focal_hl : 在 gt_specular 高光区，对 pred_r vs target 加 focal 重点
                          监督。意义：原版"高光区是否去得干净"完全由 b 决定，
                          这里给 r 一个直接强约束，迫使去高光残留无处可逃。
      [D4] l_r_dual_cons: r_direct（主解码器直接预测）与最终 r 一致性。
                          意义：让 dual-head 的两条路径相互校准、互为正则。
      [D5] l_r_smooth   : 在 gt_specular 高光区域，对 pred_r 施加局部平滑先验。
                          意义：漫反射应低频为主；高光被去除后该区域应平滑。
    """

    def __init__(self, lambda_ssim=1.0, lambda_l1=1.0, lambda_freq=0.1,
                 lambda_perceptual=0.05, lambda_spec=0.5, lambda_stat=0.0,
                 lambda_grad=0.5, lambda_grad_edge=0.5,
                 use_edge_weight=True, use_perceptual=True,
                 lambda_focal=0.8, lambda_dice=0.3, lambda_gc=0.2,
                 lambda_cs=0.05, lambda_grad_spec=0.05,
                 lambda_s_lik=0.5, lambda_s_cons=0.05,
                 lambda_spec_peak=0.3,
                 # ★[V3.2 新增权重] ★
                 lambda_r_focal_hl=0.5, lambda_r_dual_cons=0.1, lambda_r_smooth=0.05,
                 highlight_prior_gamma=1.0,
                 # ★[FIX-A 问题2] SpecLik 监督可配置超参（均有默认值，可后续调整）
                 spec_lik_pos_thresh=0.10,      # 高光前景二值化阈值（gt_sc>此值=前景）
                 spec_lik_pos_weight_max=50.0,  # pos_weight 上限，平衡前景/背景
                 spec_lik_dice_weight=0.5,      # Dice 项权重
                 use_ohem=True, ohem_ratio=0.3):
        super().__init__()

        self.ssim_fn = SSIM(as_loss=True)       # [P12]
        self.msssim_fn = MSSSIM(n_scales=3)     # [P3]
        self.use_edge_weight = use_edge_weight
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual = PerceptualLoss()
            if not self.perceptual.is_pretrained:
                self.use_perceptual = False
        self._ew_fn = compute_edge_weight_map

        self.focal = SpecularFocalLoss()
        self.dice = SpecularDiceLoss()
        self.gc = GradientConsistencyLoss()

        self.lambda_ssim = lambda_ssim
        self.lambda_l1 = lambda_l1
        self.lambda_freq = lambda_freq
        self.lambda_perceptual = lambda_perceptual
        self.lambda_spec = lambda_spec
        self.lambda_stat = lambda_stat
        self.lambda_grad = lambda_grad
        self.lambda_grad_edge = lambda_grad_edge
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.lambda_gc = lambda_gc
        self.lambda_cs = lambda_cs
        self.lambda_grad_spec = lambda_grad_spec
        self.lambda_s_lik = lambda_s_lik
        self.lambda_s_cons = lambda_s_cons
        self.lambda_spec_peak = lambda_spec_peak
        # ★[V3.2 新增 — 针对 r 的直接监督] ★
        self.lambda_r_focal_hl = lambda_r_focal_hl
        self.lambda_r_dual_cons = lambda_r_dual_cons
        self.lambda_r_smooth = lambda_r_smooth
        self.highlight_prior_gamma = highlight_prior_gamma
        # ★[FIX-A 问题2] SpecLik 监督超参
        self.spec_lik_pos_thresh = spec_lik_pos_thresh
        self.spec_lik_pos_weight_max = spec_lik_pos_weight_max
        self.spec_lik_dice_weight = spec_lik_dice_weight
        self.use_ohem = use_ohem
        self.ohem_ratio = ohem_ratio

    def frequency_loss(self, p, g):
        """[P4-FIX2] 幅度谱 + 能量加权相位一致性。

        ★[V3.2 微调] phase 系数 0.25 → 0.15。
        原因：训练前期 pred 与 gt 相位差大，phase_loss 量级远大于 amp_loss，
              即便 [V3.1 NaN-Fix] 解决了反向爆炸问题，前向数值仍偏大干扰主任务。
        """
        p = _safe_tensor(p.float(), nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))
        g = _safe_tensor(g.float(), nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))
        p_fft = torch.fft.rfft2(p, norm='ortho')
        g_fft = torch.fft.rfft2(g, norm='ortho')
        p_abs = torch.abs(p_fft)
        g_abs = torch.abs(g_fft)
        amp_loss = F.l1_loss(p_abs, g_abs)

        # 能量加权"准相位"差（无除法，反向稳定）
        re_inner = (p_fft * g_fft.conj()).real
        amp_prod = p_abs * g_abs
        with torch.no_grad():
            energy_mask = (g_abs > g_abs.mean() * 0.01).float()
            denom_mask = energy_mask.sum().clamp_min(1.0)
        phase_loss = (torch.abs(amp_prod - re_inner) * energy_mask).sum() / denom_mask
        phase_loss = _safe_tensor(phase_loss, nan=0.0, posinf=0.0, neginf=0.0)
        # ★[V3.2] 0.25 → 0.15
        return amp_loss + 0.15 * phase_loss

    def forward(self, pred_diffuse, target, est_b, gt_specular,
                pred_grad, gt_grad, cross_scale_consistency=None,
                input_img=None, pred_s_lik=None,
                pred_r_direct=None):
        """
        ★[V3.2] 新增 pred_r_direct 参数：MainDecoder 直接预测的漫反射输出。
        若为 None（旧 forward 调用），相关损失项为 0，保持向后兼容。
        """
        gt_sc = gt_specular.clamp(0, 1)

        # [P3] MS-SSIM + SSIM 混合
        l_ssim_single = 1.0 - self.ssim_fn(pred_diffuse.float(), target.float())
        l_msssim = 1.0 - self.msssim_fn(pred_diffuse.float(), target.float())
        l_ssim = 0.8 * l_msssim + 0.2 * l_ssim_single

        # [P13][SAFE] Charbonnier 替代 L1
        if self.use_edge_weight:
            wm = self._ew_fn(gt_specular.detach())
            l_l1 = (torch.sqrt((pred_diffuse - target)**2 + 1e-6) * wm).mean()
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

        l_stat = torch.tensor(0.0, device=pred_diffuse.device)
        if self.lambda_stat > 0:
            l_stat = statistical_loss(pred_diffuse, target)

        l_grad = 1.0 - self.ssim_fn(pred_grad.float(), gt_grad.float())
        grad_w = compute_input_gradient_weight_map(gt_grad.detach()).to(
            device=pred_grad.device, dtype=pred_grad.dtype)
        highlight_prior = None
        if input_img is not None:
            highlight_prior = compute_highlight_prior(input_img.detach()).to(
                device=pred_grad.device, dtype=pred_grad.dtype)
            if highlight_prior.shape[-2:] != pred_grad.shape[-2:]:
                highlight_prior = F.interpolate(
                    highlight_prior, size=pred_grad.shape[-2:],
                    mode='bilinear', align_corners=False)
            grad_w = grad_w * (1.0 + self.highlight_prior_gamma * highlight_prior)
        l_grad_edge = (
            torch.sqrt((pred_grad - gt_grad) ** 2 + 1e-6) * grad_w
        ).mean()

        if highlight_prior is None:
            l_grad_spec = torch.tensor(0.0, device=pred_grad.device)
        else:
            # ★[FIX-D 问题2] 把动态目标 Sobel(est_b) 换成稳定目标 Sobel(gt_specular)。
            #   旧版用 est_b.detach() 作目标:est_b 是模型自己的预测,在整个训练过程
            #   中持续演化 → Sobel(est_b) 是一个"会变的靶子",pred_grad 永远在追
            #   一个移动目标,体现在 val 曲线上就是中期的 U 形回弹(epoch 7~15)。
            #   现在用 gt_sc.detach() 的 Sobel 作目标:目标固定不变,pred_grad
            #   只需要单调收敛,U 形彻底消失。
            #   语义上仍然合理:highlight_prior 在亮区为正,Sobel(input) ≈
            #   Sobel(diffuse) + Sobel(specular) 且亮区主导项是 specular,所以
            #   Sobel(gt_sc) 在亮区与 Sobel(input) 接近,不会与主 grad 监督冲突。
            gt_sc_grad = normalized_sobel_magnitude(
                gt_sc.detach().float(), quantile=0.995).to(
                device=pred_grad.device, dtype=pred_grad.dtype)
            if gt_sc_grad.shape[-2:] != pred_grad.shape[-2:]:
                gt_sc_grad = F.interpolate(
                    gt_sc_grad, size=pred_grad.shape[-2:],
                    mode='bilinear', align_corners=False)
            l_grad_spec = (
                torch.abs(pred_grad - gt_sc_grad) * highlight_prior
            ).mean()

        # ── spec_likelihood 头主监督 ──
        # ★[FIX-A 问题2] 改为「二值高光标签 + pos_weight 加权 BCE + Dice」：
        #   旧版用连续 gt_sc(0~1) 作 BCE 目标，模型最优解=匹配偏低的连续值，
        #   永远到不了 0.5；且背景像素海量 + reduction='mean' 把整图往 0 压。
        #   现用二值前景标签 (gt_sc > spec_lik_pos_thresh) 并以 pos_weight 平衡
        #   前景/背景，再加 Dice 推前景重叠，使高光区跨过 0.5、二值图更完整。
        if pred_s_lik is None:
            l_s_lik = torch.tensor(0.0, device=pred_diffuse.device)
        else:
            if pred_s_lik.shape[-2:] != gt_sc.shape[-2:]:
                pred_s_lik = F.interpolate(
                    pred_s_lik, size=gt_sc.shape[-2:],
                    mode='bilinear', align_corners=False)
            pred_s_lik_safe = pred_s_lik.clamp(1e-4, 1.0 - 1e-4)
            with torch.no_grad():
                gt_bin = (gt_sc > self.spec_lik_pos_thresh).float()       # 高光前景二值标签
                pos = gt_bin.mean().clamp(1e-3, 0.5)                      # 前景占比
                pos_weight = ((1.0 - pos) / pos).clamp(1.0, self.spec_lik_pos_weight_max)
            # 带 pos_weight 的 BCE（对 sigmoid 输出）
            bce_map = -(pos_weight * gt_bin * torch.log(pred_s_lik_safe)
                        + (1.0 - gt_bin) * torch.log(1.0 - pred_s_lik_safe))
            l_bce = bce_map.mean()
            # Dice：缓解极端类别不平衡，进一步推前景重叠
            inter = (pred_s_lik_safe * gt_bin).sum(dim=(1, 2, 3))
            union = pred_s_lik_safe.sum(dim=(1, 2, 3)) + gt_bin.sum(dim=(1, 2, 3))
            l_dice_slik = (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()
            l_s_lik = l_bce + self.spec_lik_dice_weight * l_dice_slik

        # ── pred_s_lik 与 est_b 双向一致性 (V3.1) ──
        if pred_s_lik is None:
            l_s_cons = torch.tensor(0.0, device=pred_diffuse.device)
        else:
            soft_mask = gt_sc.detach()
            cons_fwd = (torch.abs(pred_s_lik - est_b.detach()) * soft_mask).mean()
            cons_bwd = (torch.abs(est_b - pred_s_lik.detach()) * soft_mask).mean()
            l_s_cons = cons_fwd + cons_bwd

        if self.use_ohem:
            l_focal = specular_ohem_loss(est_b, gt_sc, self.focal, self.ohem_ratio)
        else:
            l_focal = self.focal(est_b, gt_sc)
        l_dice = self.dice(est_b, gt_sc)
        l_gc = self.gc(est_b, gt_sc)

        # ── 高光区 est_b 欠预测单侧惩罚 (V3.1) ──
        # ★[FIX-D 问题3] 把硬阈值 (gt_sc > 0.3).float() 换成 soft_high_mask:
        #   - 硬阈值在阈值附近不连续:gt_sc 从 0.299 → 0.301 时掩码值从 0 跳到 1,
        #     损失值会有微小阶跃。叠加 lambda_spec_peak 在旧调度里 0.1 → 0.5 的
        #     爬升,加权贡献先升后降,呈现小幅回升的形态。
        #   - 现在用 sigmoid(12*(gt_sc - 0.30)) 软掩码:阈值附近连续过渡,像素的
        #     "高光程度"以平滑 [0,1] 值参与加权,损失随训练进度的演化也更平滑。
        #   - 配合 FIX-D 常量权重(lambda_spec_peak = 0.5 锁定不变),小幅回升消失。
        with torch.no_grad():
            soft_high_mask = torch.sigmoid(12.0 * (gt_sc - 0.30))  # [0,1] 软掩码
        under = F.relu(gt_sc - est_b) * soft_high_mask
        l_spec_peak = under.mean()

        # ═════════════════════════════════════════════════════════════════
        # ★[V3.2 新增 — 针对漫反射 r 的直接监督] ★
        # ═════════════════════════════════════════════════════════════════

        # [D3] l_r_focal_hl：高光区域对 pred_r vs target 的 focal-style 监督
        # 关键意义：旧版"r 在高光区是否去除干净"完全由 b 决定，间接监督薄弱。
        # 这里给 r 直接增加一个高光区强约束损失，迫使 r 不允许残留高光痕迹。
        # 配方：error * (1 + 4*soft_high_mask) * (1 + |error|)^1.5
        #   - 在 soft_high_mask≈1（强高光区）权重 ×5
        #   - focal 形式 (1+|err|)^1.5 让误差大的像素额外加权
        r_err = torch.abs(pred_diffuse - target)
        r_hl_weight = 1.0 + 4.0 * soft_high_mask
        r_focal_term = (1.0 + r_err.detach()) ** 1.5
        l_r_focal_hl = (r_err * r_hl_weight * r_focal_term).mean()

        # [D4] l_r_dual_cons：r_direct 与最终 pred_diffuse 一致性
        # 让 dual-head 的两条路径相互校准、互为正则。
        # 注意：pred_r_direct 是 MainDecoder 直接头输出（grad 流），pred_diffuse 是融合 + 精修后输出。
        # 不 detach 任何一方，让梯度可以双向流通，但用 Charbonnier 防止前期不稳定。
        if pred_r_direct is None:
            l_r_dual_cons = torch.tensor(0.0, device=pred_diffuse.device)
        else:
            l_r_dual_cons = charbonnier_loss(pred_r_direct, pred_diffuse)

        # [D5] l_r_smooth：在高光区，pred_r 的局部平滑先验
        # ★[FIX-A 问题1] 改为「单侧」约束：仅惩罚比局部均值更亮的残留（亮点/高光残留），
        #   不再惩罚模型正常恢复的暗向条纹细节。旧版双侧惩罚与主任务(SSIM/L1)直接对抗，
        #   导致后期细节恢复时 l_r_smooth 反弹上升、顶起总 loss。
        pred_r_smooth = F.avg_pool2d(pred_diffuse, kernel_size=3, stride=1, padding=1)
        bright_residual = F.relu(pred_diffuse - pred_r_smooth)
        l_r_smooth = (bright_residual * soft_high_mask).mean()

        if cross_scale_consistency is None:
            l_cs = torch.tensor(0.0, device=pred_diffuse.device)
        else:
            l_cs = cross_scale_consistency.to(device=pred_diffuse.device, dtype=pred_diffuse.dtype)

        # ★ 纯加权求和，所有项 ≥ 0
        total = (self.lambda_ssim * l_ssim + self.lambda_l1 * l_l1 +
                 self.lambda_freq * l_freq + self.lambda_perceptual * l_percep +
                 self.lambda_spec * l_spec + self.lambda_stat * l_stat +
                 self.lambda_grad * l_grad +
                 self.lambda_grad_edge * l_grad_edge +
                 self.lambda_grad_spec * l_grad_spec +
                 self.lambda_focal * l_focal +
                 self.lambda_dice * l_dice + self.lambda_gc * l_gc +
                 self.lambda_cs * l_cs +
                 self.lambda_s_lik * l_s_lik +
                 self.lambda_s_cons * l_s_cons +
                 self.lambda_spec_peak * l_spec_peak +
                 # ★[V3.2 新增] ★
                 self.lambda_r_focal_hl * l_r_focal_hl +
                 self.lambda_r_dual_cons * l_r_dual_cons +
                 self.lambda_r_smooth * l_r_smooth)

        d = {}
        for k, v in [('total', total), ('ssim', l_ssim_single),
                      ('ssim_single_loss', l_ssim_single),
                      ('msssim_loss', l_msssim),
                      ('ssim_mixed_loss', l_ssim),
                      ('l1', l_l1),
                      ('freq', l_freq), ('perceptual', l_percep),
                      ('spec', l_spec), ('stat', l_stat), ('grad', l_grad),
                      ('grad_edge', l_grad_edge), ('grad_spec', l_grad_spec),
                      ('focal', l_focal), ('dice', l_dice), ('gc', l_gc),
                      ('cs', l_cs),
                      ('s_lik', l_s_lik), ('s_cons', l_s_cons),
                      ('spec_peak', l_spec_peak),
                      # ★[V3.2 新增] ★
                      ('r_focal_hl', l_r_focal_hl),
                      ('r_dual_cons', l_r_dual_cons),
                      ('r_smooth', l_r_smooth)]:
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
            nn.Sigmoid(),
        )
        # Keep the brightness-aware attention as a residual hint rather than
        # multiplying every decoder stage by up to 2× at initialization.
        self.att_scale = nn.Parameter(torch.tensor(0.20))

    def forward(self, features, original_input):
        inp = original_input
        if inp.shape[-2:] != features.shape[-2:]:
            inp = F.interpolate(inp, size=features.shape[-2:],
                                mode='bilinear', align_corners=False)
        local_mean = F.avg_pool2d(inp, kernel_size=3, stride=1, padding=1)
        local_contrast = (inp - local_mean).abs().clamp(0, 1)
        spec_prior = (0.75 * inp + 0.25 * local_contrast).clamp(0, 1)
        t_off = self.adaptive_thresh(features)
        masks = []
        for i, base_t in enumerate(self.thresholds):
            eff_t = base_t * (0.5 + 0.5 * t_off[:, i:i+1, :, :])
            masks.append(torch.sigmoid(self.steepness * (spec_prior - eff_t)))
        multi_mask = torch.cat(masks, dim=1)
        att = self.modulation(torch.cat([features, multi_mask], dim=1))
        scale = self.att_scale.clamp(0.0, 0.5)
        return _safe_tensor(features * (1.0 + scale * att), nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=(-1.0e4, 1.0e4))


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
        if random.random() < self.p_flip:
            input_img = torch.flip(input_img, [-1])
            target_img = torch.flip(target_img, [-1])
        if random.random() < self.p_flip:
            input_img = torch.flip(input_img, [-2])
            target_img = torch.flip(target_img, [-2])
        if random.random() < self.p_rotate:
            k = random.choice([1, 2, 3])
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

    def __init__(self, p=1.0, num_range=(1, 6),
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
    返回 4 项：input, target, gt_specular, gt_gradient（输入图像 Sobel 梯度）。
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

        gt_gradient = compute_input_gradient_target(input_img)

        return input_img, target_img, gt_specular, gt_gradient


class FringeDataset(Dataset):
    """
    条纹图数据集。
    支持 PairedTransform 数据增强，gt_gradient 使用输入图像 Sobel 梯度幅值。
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
                resize_size = (int(self.resize[1]), int(self.resize[0]))
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
            transformed = self.paired_transform(input_img, target_img)
            if len(transformed) == 4:
                return transformed
            input_img, target_img = transformed

        gt_specular = (input_img - target_img).clamp(0, 1)

        gt_gradient = compute_input_gradient_target(input_img)

        return input_img, target_img, gt_specular, gt_gradient


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║          第 6 层：模型结构                                                  ║
# ║                                                                          ║
# ║  网络总体架构（X 形结构）：                                                 ║
# ║    Encoder（ResNet50 + Swin-Tiny H/16 + CrossModalFusion +                ║
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
    编码器：ResNet50 局部纹理分支 + Swin-Tiny 多尺度上下文分支。

    结构流程：
      1. ResNet50 分支提取多尺度局部纹理特征：
         x_bb_1: [B, 64,  H/2,  W/2]
         x_bb_2: [B, 128, H/4,  W/4]
         x_bb_3: [B, 256, H/8,  W/8]
         x_bb_4: [B, 512, H/16, W/16]
      2. SwinEncoderBranch 从同一灰度输入提取 H/4、H/8、H/16 三个层级特征：
         swin_4:  [B, 96,  H/4,  W/4]
         swin_8:  [B, 192, H/8,  W/8]
         swin_16: [B, 384, H/16, W/16]
      3. H/4 和 H/8 使用 AdaptiveSkipFusion 做内容自适应 skip 融合，增强 x_bb_2 / x_bb_3。
      4. H/16 仍沿用第一阶段 CrossModalFusion(x_bb_4, swin_feat_16) 做瓶颈层融合。
      5. 融合结果进入 cba_after_backbone、FFC bottleneck 和 HighFrequencyEnhancement。

    本阶段不改 Decoder 接口，不引入 H/32 特征，不做多尺度 CrossModalFusion。
    返回 5 级特征，保持后续 Decoder 的 skip connection 约定不变。
    """

    def __init__(self, swin_pretrained=True, debug_shapes=False):
        super().__init__()
        self.debug_shapes = debug_shapes

        try:
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1,
                replace_stride_with_dilation=[False, False, True])
        except Exception:
            backbone = models.resnet50(
                weights=None,
                replace_stride_with_dilation=[False, False, True])

        old_conv1 = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight.copy_(old_conv1.weight.mean(1, keepdim=True))

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

        self.swin_branch = SwinEncoderBranch(pretrained=swin_pretrained)

        # 第三阶段自适应多尺度 skip 融合：
        # H/4 和 H/8 使用 AdaptiveSkipFusion，根据内容、位置和通道动态控制 Swin 注入比例。
        # ★[D7 V3.2] init_beta 由 0.03/0.05 调整为 0.10/0.15。
        # 原因：旧版初始 beta 极小，前 50 epoch Swin 信号几乎不参与，等价于
        # 浪费了 ImageNet 预训练 Swin-Tiny 的能力。提高初值让双编码器更早协作。
        self.fuse_h4 = AdaptiveSkipFusion(
            res_channels=128,
            swin_channels=self.swin_branch.channels_4,
            init_beta=0.10)
        self.fuse_h8 = AdaptiveSkipFusion(
            res_channels=256,
            swin_channels=self.swin_branch.channels_8,
            init_beta=0.15)
        self.swin16_proj = nn.Sequential(
            nn.Conv2d(self.swin_branch.channels_16, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        # H/16 仍沿用第一阶段 CrossModalFusion，不额外 residual add，避免重复注入上下文。
        self.cross_modal_fusion = CrossModalFusion(
            cnn_dim=512, context_dim=512,
            out_dim=512, num_heads=4, sr_ratio=2)
        self.top_down_refine = CrossScaleTopDownRefinement(
            h4_channels=128, h8_channels=256, h16_channels=512, init_gamma=0.05)
        self.cba_after_backbone = Conv2D_BN_activa(512, 256, 3, 1, 1)
        self.ffc_bottleneck = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)
        self.hf_enhance = HighFrequencyEnhancement(channels=256)

    def forward(self, x):
        x_in = x
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x_bb_1 = self.stem_relu(x)

        x = self.stem_pool(x_bb_1)
        x_l1 = self.layer1(x)
        x_bb_2 = self.reduce_l1(x_l1)
        x_l2 = self.layer2(x_l1)
        x_bb_3 = self.reduce_l2(x_l2)
        x_l3 = self.layer3(x_l2)
        x_l4 = self.layer4(x_l3)

        x_l3_512 = self.reduce_l3(x_l3)
        x_l4_512 = self.reduce_l4(x_l4)
        x_bb_4 = self.fuse_l34(torch.cat([x_l3_512, x_l4_512], 1))

        swin_feats = self.swin_branch(x_in)
        swin_4 = swin_feats["swin_4"]
        swin_8 = swin_feats["swin_8"]
        swin_16 = swin_feats["swin_16"]

        if self.debug_shapes:
            print(
                f"[Swin debug] x_bb_2={tuple(x_bb_2.shape)}, swin_4={tuple(swin_4.shape)}, "
                f"x_bb_3={tuple(x_bb_3.shape)}, swin_8={tuple(swin_8.shape)}, "
                f"x_bb_4={tuple(x_bb_4.shape)}, swin_16={tuple(swin_16.shape)}"
            )

        # H/4 自适应融合：保留细条纹和浅层边缘，同时按内容注入 Swin 局部窗口上下文。
        x_bb_2 = self.fuse_h4(x_bb_2, swin_4)

        # H/8 自适应融合：对高光边界、局部反射区域和弱高光结构进行内容自适应增强。
        x_bb_3 = self.fuse_h8(x_bb_3, swin_8)

        if self.debug_shapes:
            gate4 = self.fuse_h4.last_gate_mean
            gate8 = self.fuse_h8.last_gate_mean
            gate4_str = f"{gate4.item():.6f}" if gate4 is not None else "n/a"
            gate8_str = f"{gate8.item():.6f}" if gate8 is not None else "n/a"
            print(
                f"[Adaptive skip debug] beta_h4={self.fuse_h4.beta.item():.6f}, "
                f"gate_h4_mean={gate4_str}, beta_h8={self.fuse_h8.beta.item():.6f}, "
                f"gate_h8_mean={gate8_str}"
            )

        # H/16 瓶颈融合保持第一阶段设计：投影到 512 后与 ResNet x_bb_4 做 CrossModalFusion。
        if swin_16.shape[-2:] != x_bb_4.shape[-2:]:
            swin_16 = F.interpolate(swin_16, size=x_bb_4.shape[-2:], mode='bilinear', align_corners=False)
        swin_feat_16 = self.swin16_proj(swin_16)

        fused = self.cross_modal_fusion(x_bb_4, swin_feat_16)
        x_bb_2, x_bb_3, x_bb_4 = self.top_down_refine(x_bb_2, x_bb_3, fused)
        x_c = self.cba_after_backbone(x_bb_4)

        nl = 256 - int(256 * 0.5)
        out = self.ffc_bottleneck((x_c[:, :nl], x_c[:, nl:]))
        x_c = torch.cat([out[0], out[1]], 1) + x_c
        x_c = self.hf_enhance(x_c)

        return x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c


class GradientDecoder(nn.Module):
    """
    梯度解码器：接收编码器特征，输出
        - 梯度图 x_g（监督目标 = Sobel(I)，输入图像梯度幅值）
        - [新增] 高反光软概率图 x_s（监督目标 = gt_specular，[0, 1] 软标签）
        - 4 级待增强特征 xe1, xe2, xe3, xe4

    任务化辅助头设计动机：
    ─────────────────────────────────────────────────
    梯度分支的监督已经从 `Sobel(Id)` 改为 `Sobel(I)`，但纯梯度语义会包含大量
    "非高光"边缘（条纹边、物体轮廓、噪声）。仅靠 `pred_grad ≈ Sobel(I)` 这一
    监督，无法保证梯度分支学到的中间特征对"高反光分量重建"任务有效。

    解决方案：在 GradientDecoder 尾部并列一个 `specular_likelihood` 辅助头，
    与梯度头共享前面所有特征，但用 `gt_specular` 直接监督。这强制梯度分支
    的深层特征同时编码两种语义：
        - "哪里是边界"（梯度头）
        - "哪里是高光"（高光概率头）

    这个 `x_s` 还会被馈送到 EnhancementLayers 的 `specular_gate` 和
    MainDecoder 的 `joint_gate/spec_lik_gate`，作为显式监督下的高光先验，
    替代原来仅依赖"亮度+梯度"启发式的隐式估计。
    ─────────────────────────────────────────────────
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
        # 梯度头（保留原结构）：输入 = 64ch 上采特征 + 1ch 原图 = 65ch
        self.block_5_conv_2 = Conv2D_BN_activa(65, 1, 5, 1, 2, activation=None, if_bn=False)
        self.sigmoid_layer = nn.Sigmoid()

        # ── [NEW] 高反光软概率辅助头（与梯度头并列）──
        # 输入：64ch 上采特征 + 1ch 原图 = 65ch；输出：1ch 全分辨率高光概率图 [0, 1]
        # 该头比梯度头稍微深一些，因为"是否高光"比"是否边缘"判别难度更高，
        # 需要同时利用亮度、边界形态与高频纹理。
        self.spec_lik_conv_a = Conv2D_BN_activa(65, 32, 3, 1, 1)
        self.spec_lik_conv_b = Conv2D_BN_activa(32, 16, 3, 1, 1)
        self.spec_lik_conv_c = Conv2D_BN_activa(16, 1, 5, 1, 2, activation=None, if_bn=False)
        self.spec_lik_sigmoid = nn.Sigmoid()
        self._init_spec_lik_head()

    def _init_spec_lik_head(self):
        """让 spec_lik 头初始预测 ≈ 0（即"无高光"），避免训练初期向下游注入噪声。

        通过将最后一个 conv 的权重清零、bias 设为 -2.0，使 pre-sigmoid 输出 ≈ -2，
        sigmoid(-2) ≈ 0.12。随着 `l_s_lik` 监督的引导，模型会逐步把高光区域抬起。
        """
        last = self.spec_lik_conv_c.conv2d
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            # ★[FIX-C 问题2] bias -2.0→-1.0：sigmoid(-1)≈0.27，初始预测略高，
            #   配合二值监督更快把高光区抬过 0.5。
            nn.init.constant_(last.bias, -1.0)

    def forward(self, backbone_features, x_c, original_input=None):
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
        if x_bb_4.shape[-2:] != x_to_be_enhanced_1.shape[-2:]:
            x_bb_4 = F.interpolate(
                x_bb_4, size=x_to_be_enhanced_1.shape[-2:],
                mode='bilinear', align_corners=False)
        x = torch.cat([x_to_be_enhanced_1, x_bb_4], dim=1)

        x = self.block_2_conv(x)
        _nl_h8 = 128 - int(128 * 0.25)
        _ffc_h8_out = self.ffc_grad_h8((x[:, :_nl_h8], x[:, _nl_h8:]))
        x = torch.cat([_ffc_h8_out[0], _ffc_h8_out[1]], dim=1) + x

        x_to_be_enhanced_2 = self.block_2_conv_transpose(x)
        if x_bb_3.shape[-2:] != x_to_be_enhanced_2.shape[-2:]:
            x_bb_3 = F.interpolate(
                x_bb_3, size=x_to_be_enhanced_2.shape[-2:],
                mode='bilinear', align_corners=False)
        x = torch.cat([x_to_be_enhanced_2, x_bb_3], dim=1)

        x = self.block_3_conv(x)
        _nl_h4 = 64 - int(64 * 0.25)
        _ffc_h4_out = self.ffc_grad_h4((x[:, :_nl_h4], x[:, _nl_h4:]))
        x = torch.cat([_ffc_h4_out[0], _ffc_h4_out[1]], dim=1) + x

        x_to_be_enhanced_3 = self.block_3_conv_transpose(x)
        if x_bb_2.shape[-2:] != x_to_be_enhanced_3.shape[-2:]:
            x_bb_2 = F.interpolate(
                x_bb_2, size=x_to_be_enhanced_3.shape[-2:],
                mode='bilinear', align_corners=False)
        x = torch.cat([x_to_be_enhanced_3, x_bb_2], dim=1)

        x = self.block_4_conv(x)
        x_to_be_enhanced_4 = self.block_4_conv_transpose(x)
        if x_bb_1.shape[-2:] != x_to_be_enhanced_4.shape[-2:]:
            x_bb_1 = F.interpolate(
                x_bb_1, size=x_to_be_enhanced_4.shape[-2:],
                mode='bilinear', align_corners=False)
        x = torch.cat([x_to_be_enhanced_4, x_bb_1], dim=1)

        x = self.block_5_conv_1(x)
        if original_input is None:
            original_input = torch.zeros(
                x.size(0), 1, x.size(2), x.size(3),
                device=x.device, dtype=x.dtype)
        else:
            if original_input.shape[-2:] != x.shape[-2:]:
                original_input = F.interpolate(
                    original_input, size=x.shape[-2:],
                    mode='bilinear', align_corners=False)
            original_input = original_input.to(device=x.device, dtype=x.dtype)
        # 梯度头与 spec_likelihood 头共享 [block_5_conv_1 特征 + 原图] 输入
        x_concat = torch.cat([x, original_input], dim=1)

        # 梯度头：预测 Sobel(I)（即输入图像梯度幅值）
        x_g_logits = self.block_5_conv_2(x_concat)
        x_g = self.sigmoid_layer(x_g_logits)

        # [NEW] 高光概率头：预测 gt_specular 的软概率图
        x_s_feat = self.spec_lik_conv_a(x_concat)
        x_s_feat = self.spec_lik_conv_b(x_s_feat)
        x_s_logits = self.spec_lik_conv_c(x_s_feat)
        x_s = self.spec_lik_sigmoid(x_s_logits)

        return x_to_be_enhanced_1, x_to_be_enhanced_2, x_to_be_enhanced_3, x_to_be_enhanced_4, x_g, x_s


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
        #
        # ★[D1] 双头升级（V3.2 关键改进）★
        # 旧版只输出 est_b（镜面分量），通过 r = clamp(x - est_b, 0, 1) 减法链得到漫反射。
        # 该链条对 est_b 的小幅误差极敏感：
        #   - est_b 高估 → x-b < 0 被 clamp 为 0，暗区信息永久丢失（不可逆）
        #   - est_b 低估 → x-b 仍含高光残留，refinement_head ±0.08 不足以补救
        # 双头方案：增加 r_direct_head，让网络直接预测漫反射；
        # 最终 r 由可学习 alpha 融合 [direct, subtraction_path]，更稳健。
        _out_in = 16 * 3 + 1  # 49
        self.output_reduce = Conv2D_BN_activa(_out_in, 32, 1, 1, 0)
        self.output_dilated = Conv2D_BN_activa(32, 32, 3, 1, 2, dilation=2)
        self.output_conv3 = Conv2D_BN_activa(32, 16, 3, 1, 1)
        # 镜面分量头（保留旧名 output_conv4 兼容 V3 checkpoint）
        self.output_conv4 = Conv2D_BN_activa(16, 1, 3, 1, 1, activation=None, if_bn=False)
        self.output_sigmoid = nn.Sigmoid()
        self.output_skip = nn.Conv2d(32, 32, 1, 1, 0, bias=False)
        # ★[D1] 直接漫反射预测头（与镜面头共享前几层特征）
        # 该头有自己的输出 conv，可独立从 16-ch 特征解码 r_direct
        self.output_r_direct_conv = Conv2D_BN_activa(16, 1, 3, 1, 1, activation=None, if_bn=False)
        self.output_r_direct_sigmoid = nn.Sigmoid()

        self.sa1 = SpecularAwareAttention(256 * 3, threshold=0.8)
        self.sa2 = SpecularAwareAttention(128 * 3, threshold=0.8)
        self.sa3 = SpecularAwareAttention(64 * 3, threshold=0.8)
        self.sa4 = SpecularAwareAttention(32 * 3, threshold=0.8)

        # [P10 + NEW] 三条并行调制通道：
        #   grad_gate_i    ← 仅 x_g（梯度先验：哪里是边界）
        #   joint_gate_i   ← concat(x_g, input, x_s) 3 通道：高光相关联合判别
        #   spec_lik_gate_i← 仅 x_s（监督下的高光概率，独立 scale 控制）
        # 三条通道相加再乘到特征上，让主解码器既保留泛边缘提示，也获得显式高光定位。
        self.grad_gate_2 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        self.grad_gate_3 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        self.grad_gate_4 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        # joint_gate: 输入由原来 2 通道(g, input) → 3 通道(g, input, s)
        self.joint_gate_2 = nn.Sequential(nn.Conv2d(3, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        self.joint_gate_3 = nn.Sequential(nn.Conv2d(3, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        self.joint_gate_4 = nn.Sequential(nn.Conv2d(3, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        # [NEW] spec_lik_gate: 1 通道(s)，独立 scale，专门反映监督下的高光概率
        self.spec_lik_gate_2 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        self.spec_lik_gate_3 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        self.spec_lik_gate_4 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        # ★[D6] V3.2 调整：让三路门控初始就有显著调制作用。
        # 旧版 bias=-1.0 → sigmoid(-1)≈0.27，scale=0.15 clamp[0,0.3]
        # → 初始调制 1+3×0.15×0.27=1.12 几乎无作用
        # 新版 bias=0.0 → sigmoid(0)=0.5, scale=0.25 clamp[0,0.4]
        # → 初始调制 1+3×0.25×0.5=1.375，前期就能发挥作用
        self.grad_gate_scale = nn.Parameter(torch.tensor(0.25))
        self.joint_gate_scale = nn.Parameter(torch.tensor(0.25))
        self.spec_lik_gate_scale = nn.Parameter(torch.tensor(0.25))
        self._init_aux_gates()
        # ★[D1] dual-head 融合系数（logit 形式，sigmoid 后约 0.5，初始平衡两路）
        # alpha = sigmoid(logit_alpha)，最终 r = alpha * r_direct + (1-alpha) * r_subtraction
        # 初始 logit_alpha = 0.0 → alpha = 0.5（两路等权）
        self.dual_head_alpha_logit = nn.Parameter(torch.tensor(0.0))

    def _init_aux_gates(self):
        # ★[D6] bias 由 -1.0 调整为 0.0，让初始 sigmoid 输出≈0.5（中性而非偏低）
        for gate in (self.grad_gate_2, self.grad_gate_3, self.grad_gate_4,
                     self.joint_gate_2, self.joint_gate_3, self.joint_gate_4,
                     self.spec_lik_gate_2, self.spec_lik_gate_3, self.spec_lik_gate_4):
            conv = gate[0]
            nn.init.zeros_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)  # 由 -1.0 → 0.0

    @staticmethod
    def _resize_prior(prior, size, device, dtype):
        if prior.shape[1] != 1:
            prior = prior.mean(dim=1, keepdim=True)
        if prior.shape[-2:] != size:
            prior = F.interpolate(prior, size=size, mode='bilinear', align_corners=False)
        return prior.to(device=device, dtype=dtype)

    def forward(self, c_features, enhanced_features, x_g, x_s, original_input):
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
        # [P10 + NEW] 三通道联合调制：grad + joint(g,i,s) + spec_lik
        _xg2 = F.interpolate(x_g, x.shape[-2:], mode='bilinear', align_corners=False) if x_g.shape[-2:] != x.shape[-2:] else x_g
        _xs2 = F.interpolate(x_s, x.shape[-2:], mode='bilinear', align_corners=False) if x_s.shape[-2:] != x.shape[-2:] else x_s
        _inp2 = self._resize_prior(original_input, x.shape[-2:], x.device, x.dtype)
        x = x * (1.0 +
                 self.grad_gate_scale.clamp(0.0, 0.3) * self.grad_gate_2(_xg2) +
                 self.joint_gate_scale.clamp(0.0, 0.3) * self.joint_gate_2(torch.cat([_xg2, _inp2, _xs2], dim=1)) +
                 self.spec_lik_gate_scale.clamp(0.0, 0.3) * self.spec_lik_gate_2(_xs2))
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
        # [P10 + NEW] 三通道联合调制 Level 3
        _xg3 = F.interpolate(x_g, x.shape[-2:], mode='bilinear', align_corners=False) if x_g.shape[-2:] != x.shape[-2:] else x_g
        _xs3 = F.interpolate(x_s, x.shape[-2:], mode='bilinear', align_corners=False) if x_s.shape[-2:] != x.shape[-2:] else x_s
        _inp3 = self._resize_prior(original_input, x.shape[-2:], x.device, x.dtype)
        x = x * (1.0 +
                 self.grad_gate_scale.clamp(0.0, 0.3) * self.grad_gate_3(_xg3) +
                 self.joint_gate_scale.clamp(0.0, 0.3) * self.joint_gate_3(torch.cat([_xg3, _inp3, _xs3], dim=1)) +
                 self.spec_lik_gate_scale.clamp(0.0, 0.3) * self.spec_lik_gate_3(_xs3))
        x = self.sa3(x, original_input)
        if x_bb_2.shape[-2:] != x.shape[-2:]:
            x_bb_2 = F.interpolate(x_bb_2, x.shape[-2:], mode='bilinear', align_corners=False)
        if xe3.shape[-2:] != x.shape[-2:]:
            xe3 = F.interpolate(xe3, x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att3(x, x_bb_2)
        x = torch.cat([x, bridge, xe3], 1)

        # Level 4
        x = self.transposed_block_4(x)
        # [P10 + NEW] 三通道联合调制 Level 4
        _xg4 = F.interpolate(x_g, x.shape[-2:], mode='bilinear', align_corners=False) if x_g.shape[-2:] != x.shape[-2:] else x_g
        _xs4 = F.interpolate(x_s, x.shape[-2:], mode='bilinear', align_corners=False) if x_s.shape[-2:] != x.shape[-2:] else x_s
        _inp4 = self._resize_prior(original_input, x.shape[-2:], x.device, x.dtype)
        x = x * (1.0 +
                 self.grad_gate_scale.clamp(0.0, 0.3) * self.grad_gate_4(_xg4) +
                 self.joint_gate_scale.clamp(0.0, 0.3) * self.joint_gate_4(torch.cat([_xg4, _inp4, _xs4], dim=1)) +
                 self.spec_lik_gate_scale.clamp(0.0, 0.3) * self.spec_lik_gate_4(_xs4))
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
        x = torch.cat([x, x_g], 1)

        # [P2] 扩展输出头 with skip connection
        x = self.output_reduce(x)
        skip = self.output_skip(x)
        x = self.output_dilated(x)
        x = x + skip
        x = self.output_conv3(x)
        # ★[D1] V3.2 双头输出
        # 旧版只输出 est_b：return self.output_sigmoid(self.output_conv4(x))
        # 新版从共享的 16-ch 特征 x 同时解码两个头：
        #   est_b      : 镜面反射分量（与旧版一致，保留兼容性）
        #   r_direct   : 直接预测的漫反射分量（绕过减法链，避免暗区被 clamp 抹掉）
        est_b = self.output_sigmoid(self.output_conv4(x))
        r_direct = self.output_r_direct_sigmoid(self.output_r_direct_conv(x))
        # 返回元组：(est_b, r_direct, alpha_logit)
        # XFFC.forward 会接住元组并做下游融合 + 残差精修
        return est_b, r_direct, self.dual_head_alpha_logit


class _SpectralDecompositionUnit(nn.Module):
    """梯度分解单元，仅用于 EnhancementLayers 内部。

    本单元的职责是：把"通用梯度特征"转化为"高反光相关增强特征"。

    门控设计（[NEW] 引入 s_map 后的 3 输入版本）：
    ─────────────────────────────────────────────────
        - edge_gate     ← g_map（输入梯度图）：定位"哪里有结构突变"
        - specular_gate ← concat(g_map, original_input, s_map)：
                          联合判别"哪些突变是高反光相关"
        - final_gate    = edge_gate * specular_gate
        - feature_out   = specular_refined * (1 + beta * final_gate)

    `s_map` 由 GradientDecoder 的高光概率辅助头提供（gt_specular 监督）。
    相比旧版仅靠"亮度+梯度"启发式估计，新版获得了显式监督下的高光语义。
    ─────────────────────────────────────────────────
    """

    def __init__(self, in_c: int, out_c: int, beta: float = 0.3):
        super(_SpectralDecompositionUnit, self).__init__()
        self.out_c = out_c
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.projection_branch = Conv2D_BN_activa(in_c, out_c, 7, 1, 3)
        self.specular_branch = Conv2D_BN_activa(in_c, out_c, 3, 1, 1)
        self.ffc_specular = FFC_BN_ACT(out_c, out_c, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        # edge_gate：1 通道（梯度图） → out_c 通道门控
        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, out_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )
        # specular_gate：[NEW] 3 通道输入（g_down + input_down + s_down）
        # 旧版只有 g_down + input_down，是靠 "亮+强梯度=高光" 的启发式；
        # 现在直接接入监督下的高光概率 s_down，使高光语义判别更可靠。
        self.specular_gate = nn.Sequential(
            nn.Conv2d(3, out_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self._init_prior_gates()

    def _init_prior_gates(self):
        # Start as a conservative high-reflection prior and learn stronger
        # responses only when useful for specular reconstruction.
        for gate in (self.edge_gate, self.specular_gate):
            conv = gate[0]
            nn.init.zeros_(conv.weight)
            nn.init.constant_(conv.bias, -1.0)

    @staticmethod
    def _prepare_prior(prior: torch.Tensor, size, device, dtype) -> torch.Tensor:
        if prior.shape[1] != 1:
            prior = prior.mean(dim=1, keepdim=True)
        if prior.shape[-2:] != size:
            prior = F.interpolate(prior, size=size, mode='bilinear', align_corners=False)
        return prior.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, g_map: torch.Tensor,
                original_input: torch.Tensor, s_map: torch.Tensor) -> torch.Tensor:
        projection = self.projection_branch(x)
        specular = self.specular_branch(x)

        _nl_spec = self.out_c - int(self.out_c * 0.25)
        _ffc_spec_out = self.ffc_specular((specular[:, :_nl_spec], specular[:, _nl_spec:]))
        specular = torch.cat([_ffc_spec_out[0], _ffc_spec_out[1]], dim=1) + specular

        gate = self.gate_conv(torch.cat([projection, specular], dim=1))
        specular_refined = specular + specular * gate

        size = specular_refined.shape[-2:]
        device, dtype = specular_refined.device, specular_refined.dtype
        g_down = self._prepare_prior(g_map, size, device, dtype)
        input_down = self._prepare_prior(original_input, size, device, dtype)
        s_down = self._prepare_prior(s_map, size, device, dtype)
        edge_gate = self.edge_gate(g_down)
        # 3 输入：梯度 + 亮度 + 高光概率（显式监督先验）
        spec_gate = self.specular_gate(torch.cat([g_down, input_down, s_down], dim=1))
        final_gate = edge_gate * spec_gate
        return specular_refined + self.beta.clamp(0.0, 1.0) * specular_refined * final_gate


class EnhancementLayers(nn.Module):
    """增强层：把梯度分支的多尺度特征转化为"高反光相关增强特征"。

    [改造] 接口由 (features, g_map, original_input) 升级为
            (features, g_map, original_input, s_map)。
    `s_map` 即 GradientDecoder 输出的高光软概率图（gt_specular 监督），
    会进入每一级 _SpectralDecompositionUnit 的 specular_gate 联合判别。
    """

    def __init__(self):
        super(EnhancementLayers, self).__init__()
        self.decomp_unit_1 = _SpectralDecompositionUnit(in_c=256, out_c=128)
        self.decomp_unit_2 = _SpectralDecompositionUnit(in_c=128, out_c=64)
        self.decomp_unit_3 = _SpectralDecompositionUnit(in_c=64, out_c=32)
        self.decomp_unit_4 = _SpectralDecompositionUnit(in_c=32, out_c=16)

    def forward(self, to_be_enhanced_features, g_map, original_input, s_map):
        x1 = self.decomp_unit_1(to_be_enhanced_features[0], g_map, original_input, s_map)
        x2 = self.decomp_unit_2(to_be_enhanced_features[1], g_map, original_input, s_map)
        x3 = self.decomp_unit_3(to_be_enhanced_features[2], g_map, original_input, s_map)
        x4 = self.decomp_unit_4(to_be_enhanced_features[3], g_map, original_input, s_map)
        return x1, x2, x3, x4


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


def cross_scale_consistency_loss(features):
    """约束 H/4、H/8、H/16 融合特征对结构区域的响应保持一致。"""
    maps = []
    for feat in features:
        feat = _safe_tensor(feat.float(), nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=(-1.0e4, 1.0e4))
        response = feat.abs().mean(dim=1, keepdim=True)
        mean = response.mean(dim=(2, 3), keepdim=True)
        std = response.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
        maps.append((response - mean) / std)

    target_size = maps[0].shape[-2:]
    aligned = [
        maps[0],
        F.interpolate(maps[1], size=target_size, mode='bilinear', align_corners=False),
        F.interpolate(maps[2], size=target_size, mode='bilinear', align_corners=False),
    ]
    return 0.5 * (
        F.l1_loss(aligned[0], aligned[1]) +
        F.l1_loss(aligned[1], aligned[2]))


class XFFC(nn.Module):
    """X-FFC 主网络（V3.2 优化版）。含 P1 残差精修头 + [D1] 双头漫反射预测。

    [V3.1] GradientDecoder 现在并列输出梯度图 `g` 和高光软概率图 `s`；
           EnhancementLayers 和 MainDecoder 都接收 `s` 作为显式监督下的高光先验。

    [V3.2 ★关键升级] MainDecoder 升级为双头输出：
      - est_b   : 镜面反射分量（与 V3.1 一致）
      - r_direct: 直接预测的漫反射分量
      - alpha   : 可学习融合系数

    最终漫反射 r 的形成路径（替代旧的纯减法链）：
      r_sub      = clamp(input - est_b, 0, 1)              # 减法链（保留物理含义）
      r_fused    = alpha * r_direct + (1 - alpha) * r_sub  # 双路加权融合
      r_final    = refinement_head(r_fused, input, r_direct)  # 残差精修

    返回 {r, g, b, s, r_direct, alpha, cs}。其中 r_direct/alpha 仅用于训练损失。
    """

    def __init__(self, swin_pretrained=True, debug_shapes=False):
        super().__init__()
        self.debug_shapes = debug_shapes
        self.encoder = Encoder(swin_pretrained=swin_pretrained, debug_shapes=debug_shapes)
        self.gradient_decoder = GradientDecoder()
        self.enhancement_layers = EnhancementLayers()
        self.main_decoder = MainDecoder(in_channels=256)
        # ★[D2] scale 0.08 → 0.15
        self.refinement_head = ResidualRefinementHead(scale=0.15)

    def forward(self, x):
        c_feat = self.encoder(x)
        # GradientDecoder 返回 6-tuple: (xe1, xe2, xe3, xe4, x_g, x_s)
        g_feat = self.gradient_decoder(c_feat[:-1], c_feat[-1], original_input=x)
        xe_feats = g_feat[:-2]              # (xe1, xe2, xe3, xe4)
        g_raw = g_feat[-2]                  # 梯度预测图
        s_raw = g_feat[-1]                  # 高光软概率图
        enh = self.enhancement_layers(
            xe_feats, g_map=g_raw, original_input=x, s_map=s_raw)
        # ★[D1] MainDecoder 现在返回 3-tuple: (est_b, r_direct, alpha_logit)
        # 向后兼容: 若旧版返回单 Tensor (历史 ckpt 假设)，自动包装为 3-tuple
        decoder_out = self.main_decoder(c_feat, enh, g_raw, s_raw, original_input=x)
        if isinstance(decoder_out, tuple) and len(decoder_out) == 3:
            est_b, r_direct, alpha_logit = decoder_out
        else:
            # 极端兜底：旧版本只输出 est_b
            est_b = decoder_out
            r_direct = torch.clamp(x - est_b, 0.0, 1.0)
            alpha_logit = torch.tensor(0.0, device=est_b.device)

        # 对齐到原图分辨率
        if est_b.shape[-2:] != x.shape[-2:]:
            est_b = F.interpolate(est_b, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if r_direct.shape[-2:] != x.shape[-2:]:
            r_direct = F.interpolate(r_direct, size=x.shape[-2:], mode='bilinear', align_corners=False)
        est_b = _safe_tensor(est_b, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))
        r_direct = _safe_tensor(r_direct, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))

        # 梯度图与高光概率图上采到原图尺寸
        if g_raw.shape[-2:] != x.shape[-2:]:
            g_map = F.interpolate(g_raw, size=x.shape[-2:], mode='bilinear', align_corners=False)
        else:
            g_map = g_raw
        g_map = _safe_tensor(g_map, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))
        if s_raw.shape[-2:] != x.shape[-2:]:
            s_map = F.interpolate(s_raw, size=x.shape[-2:], mode='bilinear', align_corners=False)
        else:
            s_map = s_raw
        s_map = _safe_tensor(s_map, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))

        # ★[D1] 双路融合：物理减法路径 + 直接预测路径
        r_sub = torch.clamp(x - est_b, 0.0, 1.0)
        alpha = torch.sigmoid(alpha_logit)  # [0, 1]，初始 0.5
        r_fused = alpha * r_direct + (1.0 - alpha) * r_sub
        # 残差精修（3 输入：fused, original, r_direct）
        pred_diffuse = self.refinement_head(r_fused, x, r_direct)
        pred_diffuse = _safe_tensor(pred_diffuse, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))

        cs_loss = cross_scale_consistency_loss((c_feat[1], c_feat[2], c_feat[3]))
        cs_loss = _safe_tensor(cs_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "r": pred_diffuse,
            "g": g_map,
            "b": est_b,
            "s": s_map,
            # 训练辅助输出（推理时可忽略）
            "r_direct": r_direct,
            "alpha": alpha.detach() if alpha.dim() == 0 else alpha,
            "cs": cs_loss,
        }


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
    """根据训练进度动态调整各 loss 项的权重（FIX-D:常量权重 + 例外两项退火）。

    ★[FIX-D] 同步 train.py 的 ImprovedWeightScheduler:
      所有 aux 权重锁 plateau,从 epoch 0 起不变,只有 `grad_spec` 和 `r_smooth`
      保留后期退火,`s_cons` 直接置 0。详细背景见 train.py 中的同名调度器注释。
    """

    def __init__(self, total_epochs):
        self.total_epochs = max(total_epochs, 1)

    def get_weights(self, current_epoch):
        progress = min(current_epoch / self.total_epochs, 1.0)
        # 后期退火因子:仅 grad_spec 和 r_smooth 用
        anneal = 1.0 if progress <= 0.6 else max(0.2, 1.0 - (progress - 0.6) / 0.4 * 0.8)
        weights = {
            'lambda_ssim':        1.0,
            'lambda_l1':          0.5,
            'lambda_freq':        0.15,
            'lambda_perceptual':  0.10,
            'lambda_spec':        0.5,
            'lambda_stat':        0.0,
            'lambda_grad':        0.3,
            'lambda_grad_edge':   0.8,
            'lambda_grad_spec':   0.03 * anneal,
            'lambda_focal':       0.8,
            'lambda_dice':        0.3,
            'lambda_gc':          0.2,
            'lambda_cs':          0.05,
            'lambda_s_lik':       1.0,
            # ★[FIX-D] s_cons 锁 0:FIX-A 之后结构性抗任务,直接禁用
            'lambda_s_cons':      0.0,
            'lambda_spec_peak':   0.5,
            'lambda_r_focal_hl':  0.8,
            'lambda_r_dual_cons': 0.15,
            'lambda_r_smooth':    0.05 * anneal,
        }
        return weights

    def step(self, criterion, current_epoch):
        weights = self.get_weights(current_epoch)
        criterion.update_weights(**weights)
        return weights


def build_optimizer(model, base_lr=2e-4, weight_decay=1e-4):
    """为 XFFC 模型构建分层学习率 AdamW 优化器。"""
    encoder_pretrained = []
    swin_skip_params = []
    ffc_params = []
    decoder_params = []
    new_module_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ('sa1' in name or 'sa2' in name or 'sa3' in name or
            'sa4' in name or 'hf_enhance' in name or
            'refinement_head' in name or
            'top_down_refine' in name):
            new_module_params.append(param)
        elif ('fuse_h4' in name or 'fuse_h8' in name or
              'swin16_proj' in name):
            # 第三阶段 AdaptiveSkipFusion 与 H/16 投影层使用 base_lr，避免 beta 初始为 0 后学习过慢。
            swin_skip_params.append(param)
        elif ('encoder.swin_branch.swin' in name or
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
        {'params': swin_skip_params, 'lr': base_lr, 'name': 'adaptive_skip_fusion'},
        {'params': other_params, 'lr': base_lr * 0.5, 'name': 'fusion_layers'},
        {'params': decoder_params, 'lr': base_lr, 'name': 'decoders'},
        {'params': ffc_params, 'lr': base_lr * 0.5, 'name': 'ffc_modules'},
        {'params': new_module_params, 'lr': base_lr, 'name': 'new_modules'},
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
    from torch.optim.lr_scheduler import LinearLR, LambdaLR, SequentialLR

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
    total_iters = total_epochs * iters_per_epoch
    cosine_iters = max(total_iters - warmup_iters, 1)

    # 自定义余弦衰减：超过 cosine_iters 后 LR 钳位在 eta_min，不循环
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def _cosine_no_restart(current_iter):
        """余弦退火 lambda，到达 T_max 后保持 eta_min，不回弹。"""
        if current_iter >= cosine_iters:
            # 钳位在最低学习率比例
            return eta_min / base_lrs[0] if base_lrs[0] > 0 else eta_min
        progress = current_iter / cosine_iters
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # 从 1.0 衰减到 eta_min / base_lr
        min_ratio = eta_min / base_lrs[0] if base_lrs[0] > 0 else eta_min
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    cosine_scheduler = LambdaLR(optimizer, lr_lambda=_cosine_no_restart)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_iters])
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
            pred_s_lik = output.get('s')

            with torch.amp.autocast(device_type='cuda', enabled=False):
                loss, loss_dict = criterion(
                    pred_diffuse.float(), target_img.float(),
                    est_b.float(), gt_specular.float(),
                    pred_grad.float(), gt_gradient.float(),
                    output.get('cs'), input_img=input_img.float(),
                    pred_s_lik=pred_s_lik.float() if pred_s_lik is not None else None)

            loss = loss / accumulation_steps

        # [FIX] NaN / Inf 保护
        if not torch.isfinite(loss):
            print(f'[WARNING] Epoch {epoch+1} Batch {i+1}: loss NaN/Inf, 跳过')
            optimizer.zero_grad()
            continue

        if device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            if device.type == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update()

        running_loss += loss_dict['total']
        running_ssim += (1.0 - loss_dict['ssim'])
        n_batches += 1

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
            output['g'], gt_gradient, output.get('cs'), input_img=input_img,
            pred_s_lik=output.get('s'))

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

    model = XFFC()
    print("使用 XFFC 模型")

    if args.use_gn:
        print("将 BatchNorm 替换为 GroupNorm...")
        replace_bn_with_gn(model, num_groups=8)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 数据集
    if args.use_augmentation:
        img_size = args.image_size
        if isinstance(img_size, (tuple, list)):
            min_side = min(int(img_size[0]), int(img_size[1]))
        else:
            min_side = int(img_size)
        hl_aug = HighlightAugmentation(
            p=0.5, num_range=(1, 6),
            radius_range=(3, max(5, min_side // 20)),
            intensity_range=(0.2, 0.7))
        paired_aug = EnhancedPairedTransform(highlight_aug=hl_aug)
        print("[Aug] 已启用 EnhancedPairedTransform + HighlightAugmentation(p=0.5)")
    else:
        paired_aug = None
    full_train_dataset = FringeDataset(args.input_dir, args.target_dir,
                                       resize=args.image_size, paired_transform=paired_aug)
    if args.val_input_dir and args.val_target_dir:
        train_dataset = full_train_dataset
        val_dataset = FringeDataset(args.val_input_dir, args.val_target_dir,
                                    resize=args.image_size, paired_transform=None)
    else:
        full_val_dataset = FringeDataset(args.input_dir, args.target_dir,
                                         resize=args.image_size, paired_transform=None)
        total_size = len(full_train_dataset)
        if total_size < 2:
            raise ValueError("自动划分训练/验证至少需要 2 张图像；请增加数据或提供独立验证集。")
        train_size = int(args.train_split_ratio * total_size)
        train_size = max(1, min(train_size, total_size - 1))
        val_size = total_size - train_size
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_size, generator=generator).tolist()
        train_dataset = torch.utils.data.Subset(full_train_dataset, indices[:train_size])
        val_dataset = torch.utils.data.Subset(full_val_dataset, indices[train_size:])
        print(f"[Dataset] 自动划分训练/验证: train={train_size}, val={val_size}, seed=42")

    if len(train_dataset) < args.batch_size:
        raise ValueError(
            f"训练样本数({len(train_dataset)}) 小于 batch_size({args.batch_size})，"
            "drop_last=True 会导致没有可训练 batch；请调小 batch_size 或增加数据。")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == 'cuda'), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type == 'cuda'))

    # 损失函数
    criterion = EnhancedCompositeLoss(
        use_perceptual=True, use_ohem=args.use_ohem).to(device)
    loss_weight_scheduler = DynamicLossWeightScheduler(total_epochs=args.epochs)
    print(f"[Loss] 已启用 EnhancedCompositeLoss(use_ohem={args.use_ohem})")

    # 优化器 + 调度器
    optimizer = build_optimizer(model, base_lr=args.base_lr, weight_decay=args.weight_decay)
    iters_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    if iters_per_epoch == 0:
        raise ValueError("训练 DataLoader 为空，请检查数据量、batch_size 和 drop_last 设置。")
    scheduler = build_scheduler(optimizer, warmup_iters=min(500, iters_per_epoch * 3),
                                total_epochs=args.epochs, iters_per_epoch=iters_per_epoch)

    scaler = torch.amp.GradScaler(device='cuda', enabled=(device.type == 'cuda'))
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

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
            raw_state_dict = model.state_dict()
            save_state_dict = dict(raw_state_dict)
            if ema is not None:
                for name, tensor in ema.shadow.items():
                    if name in save_state_dict:
                        save_state_dict[name] = tensor
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': save_state_dict,
                'raw_model_state_dict': raw_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ssim': best_val_ssim,
                'image_size': args.image_size,
                'uses_ema_weights': bool(ema),
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
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)

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
    train_parser.add_argument("--train_split_ratio", type=float, default=0.8)
    train_parser.add_argument("--base_lr", type=float, default=2e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--accumulation_steps", type=int, default=4)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--save_dir", type=str, default="./checkpoints")
    train_parser.add_argument("--save_interval", type=int, default=10)
    train_parser.add_argument("--use_gn", action="store_true")
    train_parser.add_argument("--use_ema", action="store_true")
    train_parser.add_argument("--ema_decay", type=float, default=0.9995)
    train_parser.add_argument("--use_ohem", action="store_true", default=True)
    train_parser.add_argument("--no_ohem", action="store_true")
    train_parser.add_argument("--use_augmentation", action="store_true", default=True)
    train_parser.add_argument("--no_augmentation", action="store_true")

    args = parser.parse_args()

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
        model = XFFC()
        print("测试模型: XFFC")
        model.to(device)

        input_size = (args.batch_size, args.input_channels, args.image_size, args.image_size)
        test_input = torch.randn(input_size).to(device)

        # 前向传播测试
        try:
            with torch.no_grad():
                out = model(test_input)
            print(f"前向传播成功: r={out['r'].shape}, g={out['g'].shape}, "
                  f"b={out['b'].shape}, s={out['s'].shape}")
            assert out["r"].shape == test_input.shape
            assert out["g"].shape == test_input.shape
            assert out["b"].shape == test_input.shape
            assert out["s"].shape == test_input.shape
            print("输出 shape 校验通过")
        except Exception as e:
            print(f"前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # 损失函数测试
        print("\n--- 损失函数测试 ---")
        try:
            criterion = CompositeLoss(use_perceptual=False).to(device)
            target = torch.rand_like(out['r'])
            gt_specular = (test_input - target).clamp(0, 1)
            gt_gradient = torch.rand_like(out['g'])
            with torch.no_grad():
                loss, loss_dict = criterion(
                    out['r'], target, out['b'], gt_specular,
                    out['g'], gt_gradient, pred_s_lik=out['s'])
            print(f"CompositeLoss 前向成功, total={loss.item():.4f}, "
                  f"keys={list(loss_dict.keys())}")
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
            model_gn = XFFC()
            bn_count = sum(1 for m in model_gn.modules() if isinstance(m, nn.BatchNorm2d))
            replace_bn_with_gn(model_gn)
            gn_count = sum(1 for m in model_gn.modules() if isinstance(m, nn.GroupNorm))
            bn_after = sum(1 for m in model_gn.modules() if isinstance(m, nn.BatchNorm2d))
            print(f"替换前 BN: {bn_count}, 替换后 GN: {gn_count}, 剩余 BN: {bn_after}")
            model_gn.to(device)
            with torch.no_grad():
                out_gn = model_gn(test_input)
            assert out_gn["r"].shape == test_input.shape
            assert out_gn["s"].shape == test_input.shape
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
        if hasattr(args, 'no_ohem') and args.no_ohem:
            args.use_ohem = False
        train(args)


if __name__ == "__main__":
    main()
