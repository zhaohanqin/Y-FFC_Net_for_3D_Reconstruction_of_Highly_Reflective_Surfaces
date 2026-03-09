"""
Y-FFC 去高反光网络 —— 优化版
=============================

基于《Y-FFC_训练优化分析报告》，在 **不改动网络结构** 的前提下，落地以下改进：

  改进 1  ─ 多目标复合损失函数 CompositeLoss（SSIM + L1 + 频域 + 镜面直接监督 + 统计 + 梯度）
  改进 2  ─ 感知损失 PerceptualLoss（VGG16 浅/中层特征匹配）
  改进 3  ─ 分层学习率 + Warmup + 余弦退火（build_optimizer / build_scheduler）
  改进 4  ─ 梯度分支监督目标改为幅值（FringeDataset）
  改进 5  ─ 数据增强 PairedTransform（翻转/旋转/裁剪/亮度扰动）
  改进 6  ─ 损失权重动态调度 DynamicLossWeightScheduler
  改进 7  ─ EMA 模型平均
  改进 8  ─ 混合精度 + 梯度累积（train_one_epoch 中实现）
  改进 9  ─ 边缘感知加权损失（CompositeLoss 内部）
  改进 10 ─ BN → GN 替换工具 replace_bn_with_gn

所有网络定义代码（YFFC / Encoder / GradientDecoder / MainDecoder / EnhancementLayers 等）
与原版完全一致，仅在文件后半部分新增训练工具类与训练入口 train()。
"""

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

from ffc import *


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                     原始网络结构（完全不变）                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)


class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        base = gaussian_window(window_size, sigma).unsqueeze(0).unsqueeze(0)
        self.register_buffer("base_window", base)

    def forward(self, img1, img2):
        c = img1.size(1)
        window = self.base_window.to(dtype=img1.dtype, device=img1.device).repeat(c, 1, 1, 1)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=c)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=c)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1**2, window, padding=self.window_size // 2, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(img2**2, window, padding=self.window_size // 2, groups=c) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=c) - mu12

        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        eps = 1e-8
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps
        )
        ssim_map = torch.clamp(ssim_map, 0.0, 1.0)
        return ssim_map.mean()


def statistical_loss(pred, gt):
    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  [修复 BUG-4]  statistical_loss 改为逐样本计算                    ║
    # ╚══════════════════════════════════════════════════════════════════╝
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    gt_flat   = gt.view(B, -1)
    mean_loss = torch.abs(pred_flat.mean(dim=1) - gt_flat.mean(dim=1)).mean()
    pred_std = pred_flat.std(dim=1, unbiased=False)
    gt_std   = gt_flat.std(dim=1, unbiased=False)
    std_loss = torch.abs(pred_std - gt_std).mean()
    return mean_loss + std_loss


class Conv2D_BN_activa(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        if_bn=True,
        activation="relu",
        bias=None,
        initializer=None,
        transpose=False,
    ):
        super(Conv2D_BN_activa, self).__init__()
        if transpose:
            self.conv2d = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                bias=(not if_bn) if bias is None else bias,
            )
        else:
            self.conv2d = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                bias=(not if_bn) if bias is None else bias,
            )

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
    改进点 E：注意力块的两项修改。

    修改 1 — channel-wise 注意力掩码：
      原设计 psi 输出 1ch 空间掩码，F_l 个通道共享同一权重，无法对镜面高光区域
      的不同语义通道（边缘/纹理/高光）进行差异化处理。
      改为 psi 输出 F_l ch（每通道独立注意力权重），BatchNorm(1) → BatchNorm(F_l)，
      使各通道根据自身语义独立决定融入比例，输出 shape 与原版完全一致（x * psi）。

    修改 2 — F_int 大幅缩小：
      原设计 F_int 接近 F_g（如 att1: F_int=640 vs F_g=768），中间层仅略有压缩，
      绝大部分信息在最后的 F_int→1 卷积中被丢弃，参数利用率极低。
      F_int 缩小后（调用时指定），W_g、W_x 先强制压缩到低维，让网络学习更紧凑的
      注意力表达，同时大幅降低参数量、减少高反光少样本场景的过拟合风险。
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
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        return torch.cat((x1, x2, x3), 1)


class FeatureExtrationLayersB(nn.Module):
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
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        p4 = self.path4(x)
        return torch.cat((p1, p2, p3, p4), 1)


class TransposedConvBlock(nn.Module):
    """
    改进点 C：三路转置卷积改为差异化 kernel size。
    """
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
    def __init__(self, in_channels, out_channels_branch=256):
        super(TransposedConvBlock_same, self).__init__()
        self.branch_1 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.branch_2 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.branch_3 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat([x1, x2, x3], dim=1)


class ViTEncoderBranch(nn.Module):
    def __init__(self, vit_image_size=224):
        super(ViTEncoderBranch, self).__init__()
        self.vit_image_size = vit_image_size

        vit_weights = None
        try:
            vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        except Exception:
            vit_weights = None

        if not hasattr(models, "vit_b_16"):
            raise RuntimeError("当前 torchvision 版本不包含 vit_b_16，请升级 torchvision 或改用 timm。")

        try:
            self.vit = models.vit_b_16(weights=vit_weights)
        except Exception:
            self.vit = models.vit_b_16(weights=None)

        if hasattr(self.vit, "heads"):
            self.vit.heads = nn.Identity()
        elif hasattr(self.vit, "head"):
            self.vit.head = nn.Identity()

        self.embed_dim = getattr(self.vit, "hidden_dim", self.vit.conv_proj.out_channels)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x = F.interpolate(
            x,
            size=(self.vit_image_size, self.vit_image_size),
            mode="bilinear",
            align_corners=False,
        )

        batch_size = x.size(0)

        x = self.vit.conv_proj(x)
        grid_h, grid_w = x.shape[-2], x.shape[-1]

        x = x.flatten(2).transpose(1, 2)

        if hasattr(self.vit, "class_token"):
            cls_token = self.vit.class_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = self.vit.encoder(x)

        if x.shape[1] == grid_h * grid_w + 1:
            x = x[:, 1:, :]

        x = x.transpose(1, 2).reshape(batch_size, x.shape[2], grid_h, grid_w)
        return x


class Encoder(nn.Module):
    def __init__(self, vit_image_size=224):
        super(Encoder, self).__init__()

        try:
            backbone_model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1,
                replace_stride_with_dilation=[False, False, True],
            )
        except Exception:
            backbone_model = models.resnet50(
                weights=None,
                replace_stride_with_dilation=[False, False, True],
            )

        old_conv1 = backbone_model.conv1
        backbone_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone_model.conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))

        self.stem_conv = backbone_model.conv1
        self.stem_bn = backbone_model.bn1
        self.stem_relu = backbone_model.relu
        self.stem_pool = backbone_model.maxpool

        self.layer1 = backbone_model.layer1
        self.layer2 = backbone_model.layer2
        self.layer3 = backbone_model.layer3
        self.layer4 = backbone_model.layer4

        self.reduce_l1 = Conv2D_BN_activa(256, 128, 1, 1, 0)
        self.reduce_l2 = Conv2D_BN_activa(512, 256, 1, 1, 0)

        self.reduce_l3 = Conv2D_BN_activa(1024, 512, 1, 1, 0)
        self.reduce_l4 = Conv2D_BN_activa(2048, 512, 1, 1, 0)
        self.fuse_l34 = Conv2D_BN_activa(1024, 512, 3, 1, 1)

        self.vit_branch = ViTEncoderBranch(vit_image_size=vit_image_size)

        self.vit_resnet_fuse_1x1 = Conv2D_BN_activa(512 + self.vit_branch.embed_dim, 512, 1, 1, 0)

        self.cba_after_backbone = Conv2D_BN_activa(512, 256, 3, 1, 1)

        self.ffc_bottleneck = FFC_BN_ACT(256, 256, 3, ratio_gin=0.5, ratio_gout=0.5, padding=1)

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
        x_bb_4 = self.fuse_l34(torch.cat([x_l3_512, x_l4_512], dim=1))

        vit_feat = self.vit_branch(x_in)
        if vit_feat.shape[-2:] != x_bb_4.shape[-2:]:
            vit_feat = F.interpolate(vit_feat, size=x_bb_4.shape[-2:], mode="bilinear", align_corners=False)

        fused = torch.cat([x_bb_4, vit_feat], dim=1)
        fused = self.vit_resnet_fuse_1x1(fused)
        x_c = self.cba_after_backbone(fused)

        _nl_btn = 256 - int(256 * 0.5)
        _ffc_btn_out = self.ffc_bottleneck((x_c[:, :_nl_btn], x_c[:, _nl_btn:]))
        x_c = torch.cat([_ffc_btn_out[0], _ffc_btn_out[1]], dim=1) + x_c

        return x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c


class GradientDecoder(nn.Module):
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
            x_c = F.interpolate(x_c, size=x_bb_4.shape[-2:], mode="bilinear", align_corners=False)
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
    def __init__(self, in_channels):
        super(MainDecoder, self).__init__()
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
        self.att4 = Attention_block(96,  64,  16)

        self.ffc_h8 = FFC_BN_ACT(384, 384, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)

        self.ffc_h4 = FFC_BN_ACT(192, 192, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)

        self.output_conv1 = Conv2D_BN_activa(16 * 3 + 1, 16, 3, 1, 1)
        self.output_conv2 = Conv2D_BN_activa(16, 1, 3, 1, 1, activation=None, if_bn=False)
        self.output_sigmoid = nn.Sigmoid()

    def forward(self, c_features, enhanced_features, x_g):
        x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c = c_features
        x_enhanced_1, x_enhanced_2, x_enhanced_3, x_enhanced_4 = enhanced_features

        x0, x1 = x_c.chunk(2, dim=1)
        x = self.FFC1(tuple([x0, x1]))
        x = self.FFC2(x)
        x = torch.cat([x[0], x[1]], dim=1)
        x = x + x_c

        x = self.multiscale_conv(x)
        x = self.transposed_block_1(x)

        if x_bb_4.shape[-2:] != x.shape[-2:]:
            x_bb_4 = F.interpolate(x_bb_4, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if x_enhanced_1.shape[-2:] != x.shape[-2:]:
            x_enhanced_1 = F.interpolate(x_enhanced_1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        bridge = self.att1(x, x_bb_4)
        x = torch.cat([x, bridge, x_enhanced_1], dim=1)

        x = self.transposed_block_2(x)
        _n_l = 384 - int(384 * 0.25)
        _ffc_out = self.ffc_h8((x[:, :_n_l], x[:, _n_l:]))
        x = torch.cat([_ffc_out[0], _ffc_out[1]], dim=1) + x
        if x_bb_3.shape[-2:] != x.shape[-2:]:
            x_bb_3 = F.interpolate(x_bb_3, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if x_enhanced_2.shape[-2:] != x.shape[-2:]:
            x_enhanced_2 = F.interpolate(x_enhanced_2, size=x.shape[-2:], mode="bilinear", align_corners=False)
        bridge = self.att2(x, x_bb_3)
        x = torch.cat([x, bridge, x_enhanced_2], dim=1)

        x = self.feature_extraction_B(x)
        x = self.transposed_block_3(x)

        _n_l_h4 = 192 - int(192 * 0.25)
        _ffc_h4_out = self.ffc_h4((x[:, :_n_l_h4], x[:, _n_l_h4:]))
        x = torch.cat([_ffc_h4_out[0], _ffc_h4_out[1]], dim=1) + x

        if x_bb_2.shape[-2:] != x.shape[-2:]:
            x_bb_2 = F.interpolate(x_bb_2, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if x_enhanced_3.shape[-2:] != x.shape[-2:]:
            x_enhanced_3 = F.interpolate(x_enhanced_3, size=x.shape[-2:], mode="bilinear", align_corners=False)
        bridge = self.att3(x, x_bb_2)
        x = torch.cat([x, bridge, x_enhanced_3], dim=1)

        x = self.transposed_block_4(x)
        if x_bb_1.shape[-2:] != x.shape[-2:]:
            x_bb_1 = F.interpolate(x_bb_1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if x_enhanced_4.shape[-2:] != x.shape[-2:]:
            x_enhanced_4 = F.interpolate(x_enhanced_4, size=x.shape[-2:], mode="bilinear", align_corners=False)
        bridge = self.att4(x, x_bb_1)
        x = torch.cat([x, bridge, x_enhanced_4], dim=1)

        x = self.transposed_block_5(x)
        if x_g.shape[-2:] != x.shape[-2:]:
            x_g = F.interpolate(x_g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x_g], dim=1)

        x = self.output_conv1(x)
        est_b = self.output_sigmoid(self.output_conv2(x))
        return est_b


# ==================== 梯度分解单元（仅用于 EnhancementLayers 内部） ====================

class _SpectralDecompositionUnit(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super(_SpectralDecompositionUnit, self).__init__()

        self.out_c = out_c

        self.projection_branch = Conv2D_BN_activa(in_c, out_c, 7, 1, 3)
        self.specular_branch = Conv2D_BN_activa(in_c, out_c, 3, 1, 1)

        self.ffc_specular = FFC_BN_ACT(out_c, out_c, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)

        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = self.projection_branch(x)
        specular = self.specular_branch(x)

        _nl_spec = self.out_c - int(self.out_c * 0.25)
        _ffc_spec_out = self.ffc_specular((specular[:, :_nl_spec], specular[:, _nl_spec:]))
        specular = torch.cat([_ffc_spec_out[0], _ffc_spec_out[1]], dim=1) + specular

        gate = self.gate_conv(torch.cat([projection, specular], dim=1))
        specular_refined = specular + specular * gate

        return specular_refined


class EnhancementLayers(nn.Module):
    def __init__(self):
        super(EnhancementLayers, self).__init__()
        self.decomp_unit_1 = _SpectralDecompositionUnit(in_c=256, out_c=128)
        self.decomp_unit_2 = _SpectralDecompositionUnit(in_c=128, out_c=64)
        self.decomp_unit_3 = _SpectralDecompositionUnit(in_c=64,  out_c=32)
        self.decomp_unit_4 = _SpectralDecompositionUnit(in_c=32,  out_c=16)

    def forward(self, to_be_enhanced_features):
        x1 = self.decomp_unit_1(to_be_enhanced_features[0])
        x2 = self.decomp_unit_2(to_be_enhanced_features[1])
        x3 = self.decomp_unit_3(to_be_enhanced_features[2])
        x4 = self.decomp_unit_4(to_be_enhanced_features[3])
        return x1, x2, x3, x4


class YFFC(nn.Module):
    def __init__(self):
        super(YFFC, self).__init__()
        self.encoder = Encoder()
        self.gradient_decoder = GradientDecoder()
        self.enhancement_layers = EnhancementLayers()
        self.main_decoder = MainDecoder(in_channels=256)

    def forward(self, x):
        c_features = self.encoder(x)
        g_features = self.gradient_decoder(c_features[:-1], c_features[-1])
        enhanced_features = self.enhancement_layers(g_features[:-1])
        est_b = self.main_decoder(c_features, enhanced_features, g_features[-1])
        pred_diffuse = torch.clamp(x - est_b, 0.0, 1.0)
        return {"r": pred_diffuse, "g": g_features[-1], "b": est_b}


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║               以下为报告建议的全部新增优化代码                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════
# 【改进 2】感知损失 PerceptualLoss（VGG16 浅/中层特征匹配）
# ───────────────────────────────────────────────────────────────────────────
# 修改了什么：新增 PerceptualLoss 类
# 为什么：SSIM + L1 仅在像素空间操作，无法约束深层语义特征一致性；
#         使用预训练 VGG 中间层特征匹配可有效恢复高光下方纹理细节。
# 解决问题：报告「不足 2」—— 缺少感知损失，高频细节恢复能力不足。
# ═══════════════════════════════════════════════════════════════════════════

class PerceptualLoss(nn.Module):
    """
    使用预训练 VGG16 的浅层/中层特征计算感知距离。
    参数冻结，不参与梯度更新，仅作为损失的特征提取器。
    """
    def __init__(self, layers=('relu1_2', 'relu2_2', 'relu3_3'),
                 weights=(1.0, 0.5, 0.25)):
        super(PerceptualLoss, self).__init__()

        # 尝试加载预训练权重；若无网络，使用随机权重
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        except Exception:
            vgg = models.vgg16(weights=None).features

        layer_map = {'relu1_2': 4, 'relu2_2': 9, 'relu3_3': 16}
        self.slices = nn.ModuleList()
        prev = 0
        for layer_name in layers:
            end = layer_map[layer_name]
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:end]))
            prev = end

        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False

        self.weights = weights

    def forward(self, pred, gt):
        # 单通道 → 3通道以匹配 VGG 输入
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
            gt = gt.repeat(1, 3, 1, 1)
        loss = 0.0
        x, y = pred, gt
        for i, slice_module in enumerate(self.slices):
            x = slice_module(x)
            y = slice_module(y)
            loss += self.weights[i] * F.l1_loss(x, y)
        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 【改进 9】边缘感知加权图生成
# ───────────────────────────────────────────────────────────────────────────
# 修改了什么：新增 compute_edge_weight_map 函数
# 为什么：高光边缘（高光与非高光的过渡区）是恢复质量对 SSIM 影响最大的区域，
#         通过空间加权使模型更关注这些关键区域。
# 解决问题：报告「改进 9」—— 高光边缘精细恢复不足。
# ═══════════════════════════════════════════════════════════════════════════

def compute_edge_weight_map(gt_specular, sigma=3.0, base_weight=1.0, edge_weight=3.0):
    """根据镜面反射分量的梯度生成边缘权重图。"""
    # 有限差分计算梯度
    gx = gt_specular[:, :, :, 1:] - gt_specular[:, :, :, :-1]
    gy = gt_specular[:, :, 1:, :] - gt_specular[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0), mode='replicate')
    gy = F.pad(gy, (0, 0, 0, 1), mode='replicate')
    edge_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    # 使用平均池化近似高斯模糊（避免依赖 torchvision.transforms.functional）
    k = max(3, int(6 * sigma + 1) | 1)  # 保证奇数
    pad_size = k // 2
    edge_mag = F.avg_pool2d(
        F.pad(edge_mag, [pad_size] * 4, mode='replicate'),
        kernel_size=k, stride=1, padding=0
    )

    # 归一化 → 映射到 [base_weight, edge_weight]
    batch_max = edge_mag.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    edge_mag = edge_mag / (batch_max + 1e-8)
    weight_map = base_weight + (edge_weight - base_weight) * edge_mag
    return weight_map


# ═══════════════════════════════════════════════════════════════════════════
# 【改进 1 + 9】多目标复合损失函数 CompositeLoss
# ───────────────────────────────────────────────────────────────────────────
# 修改了什么：新增 CompositeLoss 类，整合 SSIM / L1 / 频域 / 感知 / 镜面
#            直接监督 / 统计 / 梯度 / 边缘加权 共 8 种损失项。
# 为什么：单一 SSIM 损失在后期梯度消失，缺乏像素级和频域约束，
#         镜面反射分量无直接监督。复合损失覆盖不同误差维度。
# 解决问题：报告「不足 1、2、8」—— 损失函数单薄 / 缺少感知损失 / 缺少镜面直接监督。
# ═══════════════════════════════════════════════════════════════════════════

class CompositeLoss(nn.Module):
    """
    多目标复合损失函数。

    Args:
        lambda_ssim:  漫反射 SSIM loss 权重（始终保持）
        lambda_l1:    漫反射 L1 loss 权重（像素级精确约束）
        lambda_freq:  频域 L1 loss 权重（约束全局频谱一致性）
        lambda_perceptual: 感知损失权重（VGG 特征空间匹配）
        lambda_spec:  镜面分量直接监督权重（L1 + SSIM 对 est_b）
        lambda_stat:  统计损失权重（均值/标准差匹配）
        lambda_grad:  梯度分支 SSIM loss 权重
        use_edge_weight: 是否启用边缘感知加权（对 L1 项）
    """
    def __init__(self,
                 lambda_ssim=1.0,
                 lambda_l1=1.0,
                 lambda_freq=0.1,
                 lambda_perceptual=0.05,
                 lambda_spec=0.5,
                 lambda_stat=0.1,
                 lambda_grad=0.5,
                 use_edge_weight=True,
                 use_perceptual=True):
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
        """频域 L1 损失：约束 FFT 幅度谱分布的一致性。"""
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        gt_fft = torch.fft.rfft2(gt, norm='ortho')
        return F.l1_loss(torch.abs(pred_fft), torch.abs(gt_fft))

    def forward(self, pred_diffuse, target, est_b, gt_specular,
                pred_grad, gt_grad):
        """
        Args:
            pred_diffuse: 预测漫反射图 [B,1,H,W]（模型输出 'r'）
            target:       GT 漫反射图   [B,1,H,W]
            est_b:        预测镜面分量   [B,1,H,W]（模型输出 'b'）
            gt_specular:  GT 镜面分量    [B,1,H,W]
            pred_grad:    预测梯度图     [B,1,H,W]（模型输出 'g'）
            gt_grad:      GT 梯度图      [B,1,H,W]

        Returns:
            total_loss, loss_dict
        """
        # ── 1. 漫反射 SSIM loss ─────────────────────────────────────────────
        # AMP 下 SSIM 保持 float32 精度
        l_ssim = 1.0 - self.ssim(pred_diffuse.float(), target.float())

        # ── 2. 漫反射 L1 loss（可选边缘加权）────────────────────────────────
        if self.use_edge_weight:
            weight_map = compute_edge_weight_map(gt_specular.detach())
            l_l1 = (torch.abs(pred_diffuse - target) * weight_map).mean()
        else:
            l_l1 = F.l1_loss(pred_diffuse, target)

        # ── 3. 频域 L1 loss ─────────────────────────────────────────────────
        l_freq = self.frequency_loss(pred_diffuse.float(), target.float())

        # ── 4. 感知损失 ─────────────────────────────────────────────────────
        if self.use_perceptual:
            l_perceptual = self.perceptual(pred_diffuse, target)
        else:
            l_perceptual = torch.tensor(0.0, device=pred_diffuse.device)

        # ── 5. 镜面分量直接监督（L1 + SSIM）─────────────────────────────────
        # gt_specular 需 clamp 到 [0,1] 以匹配 est_b 的 Sigmoid 输出范围
        gt_specular_clamped = gt_specular.clamp(0.0, 1.0)
        l_spec = F.l1_loss(est_b, gt_specular_clamped) + \
                 0.5 * (1.0 - self.ssim(est_b.float(), gt_specular_clamped.float()))

        # ── 6. 统计损失 ─────────────────────────────────────────────────────
        l_stat = statistical_loss(pred_diffuse, target)

        # ── 7. 梯度分支 SSIM loss ───────────────────────────────────────────
        l_grad = 1.0 - self.ssim(pred_grad.float(), gt_grad.float())

        # ── 加权求和 ────────────────────────────────────────────────────────
        total = (self.lambda_ssim * l_ssim +
                 self.lambda_l1 * l_l1 +
                 self.lambda_freq * l_freq +
                 self.lambda_perceptual * l_perceptual +
                 self.lambda_spec * l_spec +
                 self.lambda_stat * l_stat +
                 self.lambda_grad * l_grad)

        loss_dict = {
            'total': total.item(),
            'ssim': l_ssim.item(),
            'l1': l_l1.item(),
            'freq': l_freq.item(),
            'perceptual': l_perceptual.item(),
            'spec': l_spec.item(),
            'stat': l_stat.item(),
            'grad': l_grad.item(),
        }

        return total, loss_dict

    def update_weights(self, **kwargs):
        """动态更新各损失项权重（配合 DynamicLossWeightScheduler 使用）。"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


# ═══════════════════════════════════════════════════════════════════════════
# 【改进 6】损失权重动态调度
# ───────────────────────────────────────────────────────────────────────────
# 修改了什么：新增 DynamicLossWeightScheduler 类
# 为什么：训练早期需快速收敛全局结构（L1 为主），后期需精细化高频细节
#         （频域 + 感知损失加大），不同阶段需要不同的损失权重配比。
# 解决问题：报告「改进 6」—— loss 后期下降停滞。
# ═══════════════════════════════════════════════════════════════════════════

class DynamicLossWeightScheduler:
    """根据训练进度动态调整各 loss 项的权重。"""

    def __init__(self, total_epochs):
        self.total_epochs = max(total_epochs, 1)

    def get_weights(self, current_epoch):
        progress = min(current_epoch / self.total_epochs, 1.0)
        weights = {
            'lambda_ssim':       1.0,                                     # 始终保持
            'lambda_l1':         max(0.5, 1.0 - 0.5 * progress),         # 1.0 → 0.5
            'lambda_freq':       min(0.2, 0.05 + 0.15 * progress),       # 0.05 → 0.2
            'lambda_perceptual': min(0.15, 0.02 + 0.13 * progress),      # 0.02 → 0.15
            'lambda_spec':       0.5,                                     # 保持不变
            'lambda_stat':       0.1,                                     # 保持不变
            'lambda_grad':       max(0.3, 0.5 - 0.2 * progress),         # 0.5 → 0.3
        }
        return weights

    def step(self, criterion, current_epoch):
        """将权重直接更新到 CompositeLoss 实例中。"""
        weights = self.get_weights(current_epoch)
        criterion.update_weights(**weights)
        return weights


# ═══════════════════════════════════════════════════════════════════════════
# 【改进 5】数据增强 PairedTransform
# ───────────────────────────────────────────────────────────────────────────
# 修改了什么：新增 PairedTransform 类
# 为什么：原始 FringeDataset 几乎无数据增强，模型容易过拟合；
#         通过对 input/target 同步施加几何变换、仅对 input 施加光度扰动，
#         扩充训练分布、提高泛化能力。
# 解决问题：报告「不足 6」—— 数据增强为空白。
# ═══════════════════════════════════════════════════════════════════════════

class PairedTransform:
    """
    对 input 和 target 施加完全相同的几何变换；
    光度变换仅施加在 input 上。

    注意：gt_specular 和 gt_gradient 需要在增强后重新计算，
          因此本类仅在 FringeDataset.__getitem__() 的 PIL → tensor 之后、
          gt 计算之前 使用。
    """
    def __init__(self, p_flip=0.5, p_rotate=0.3, p_crop=0.5,
                 crop_scale=(0.7, 1.0), brightness_range=(0.9, 1.1)):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_crop = p_crop
        self.crop_scale = crop_scale
        self.brightness_range = brightness_range

    def __call__(self, input_img, target_img):
        """
        Args:
            input_img:  [1, H, W] float32 tensor ∈ [0, 1]
            target_img: [1, H, W] float32 tensor ∈ [0, 1]
        Returns:
            input_img, target_img（增强后，shape 可能不变也可能因裁剪后 resize 回原尺寸）
        """
        # 1. 随机水平翻转
        if random.random() < self.p_flip:
            input_img = torch.flip(input_img, [-1])
            target_img = torch.flip(target_img, [-1])

        # 2. 随机垂直翻转
        if random.random() < self.p_flip:
            input_img = torch.flip(input_img, [-2])
            target_img = torch.flip(target_img, [-2])

        # 3. 随机 90° 旋转
        if random.random() < self.p_rotate:
            k = random.choice([1, 2, 3])
            input_img = torch.rot90(input_img, k, [1, 2])
            target_img = torch.rot90(target_img, k, [1, 2])

        # 4. 随机裁剪 + resize 回原尺寸
        if random.random() < self.p_crop:
            _, h, w = input_img.shape
            scale = random.uniform(*self.crop_scale)
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h >= 32 and new_w >= 32:  # 安全下界
                top = random.randint(0, h - new_h)
                left = random.randint(0, w - new_w)
                input_img = input_img[:, top:top+new_h, left:left+new_w]
                target_img = target_img[:, top:top+new_h, left:left+new_w]
                input_img = F.interpolate(
                    input_img.unsqueeze(0), size=(h, w),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
                target_img = F.interpolate(
                    target_img.unsqueeze(0), size=(h, w),
                    mode='bilinear', align_corners=False
                ).squeeze(0)

        # 5. 轻微亮度扰动（仅 input）
        brightness_factor = random.uniform(*self.brightness_range)
        input_img = (input_img * brightness_factor).clamp(0.0, 1.0)

        return input_img, target_img


# ═══════════════════════════════════════════════════════════════════════════
# 【改进 4 + 5 整合】修改后的 FringeDataset
# ───────────────────────────────────────────────────────────────────────────
# 修改了什么：
#   1. gt_gradient 改为基于梯度幅值 sqrt(gx^2 + gy^2)，替代原始 (gx+gy)/2+0.5
#   2. 支持 PairedTransform 数据增强（在 gt 计算之前应用）
#   3. gt_specular / gt_gradient 均在增强后重新计算
# 为什么：
#   原始梯度方向信息丢失、动态范围极窄、与 Sigmoid 输出语义不匹配。
#   幅值表示物理含义清晰，零值即无边缘，充分利用 [0,1] 范围。
# 解决问题：报告「不足 3」—— 梯度分支监督信号的物理歧义。
# ═══════════════════════════════════════════════════════════════════════════

class FringeDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None, resize=None,
                 paired_transform=None):
        """
        Args:
            input_dir:        输入图像目录
            target_dir:       目标（无高光参考）图像目录
            transform:        传统 torchvision transform（兼容旧接口）
            resize:           目标尺寸
            paired_transform: PairedTransform 实例（新增，推荐使用）
        """
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.resize = resize
        self.paired_transform = paired_transform
        self.files = sorted(
            [
                f
                for f in os.listdir(input_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            ]
        )

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
                f"Could not find matching target for input '{img_name}' in '{self.target_dir}'. Tried: {set(candidate_names)}"
            )

        input_pil = Image.open(input_path).convert("L")
        target_pil = Image.open(target_path).convert("L")

        if self.resize is not None:
            if isinstance(self.resize, (int, float)):
                resize_size = (int(self.resize), int(self.resize))
            elif isinstance(self.resize, (tuple, list)) and len(self.resize) == 2:
                resize_size = (int(self.resize[0]), int(self.resize[1]))
            else:
                resize_size = None

            if resize_size:
                input_pil = input_pil.resize(resize_size, Image.LANCZOS)
                target_pil = target_pil.resize(resize_size, Image.LANCZOS)

        input_img = np.array(input_pil, dtype=np.float32) / 255.0
        target_img = np.array(target_pil, dtype=np.float32) / 255.0

        input_img = torch.from_numpy(input_img).unsqueeze(0)
        target_img = torch.from_numpy(target_img).unsqueeze(0)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        # ── 【改进 5】数据增强（在 gt 计算之前）──────────────────────────────
        if self.paired_transform is not None:
            input_img, target_img = self.paired_transform(input_img, target_img)
        # ────────────────────────────────────────────────────────────────────

        # gt_specular 和 gt_gradient 在增强后重新计算
        gt_specular = input_img - target_img

        # ── 【改进 4】梯度分支监督目标改为幅值 ────────────────────────────────
        # 原始：gt_gradient_raw = (gx + gy) / 2.0; gt_gradient = (raw + 0.5).clamp(0, 1)
        # 问题：方向信息丢失、动态范围窄、0.5 歧义
        # 修改：使用梯度幅值 sqrt(gx^2 + gy^2)，自适应归一化到 [0, 1]
        gx, gy = torch.gradient(target_img, dim=(1, 2))
        gt_gradient_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        # 使用 99.5 百分位数归一化（避免极端值影响）
        flat = gt_gradient_mag.flatten()
        if flat.numel() > 0:
            p995 = torch.quantile(flat, 0.995)
        else:
            p995 = torch.tensor(1.0)
        gt_gradient = (gt_gradient_mag / (p995 + 1e-8)).clamp(0.0, 1.0)
        # ────────────────────────────────────────────────────────────────────

        return input_img, target_img, gt_specular, gt_gradient


# ═══════════════════════════════════════════════════════════════════════════
# 【改进 7】EMA 模型参数平均
# ───────────────────────────────────────────────────────────────────────────
# 修改了什么：新增 EMA 类
# 为什么：小 batch size + 复杂多分支架构 → 参数震荡明显；
#         EMA 通过指数滑动平均平滑参数，提升推理稳定性。
# 解决问题：报告「改进 7」—— 训练 loss 曲线震荡、验证指标不稳定。
# ═══════════════════════════════════════════════════════════════════════════

class EMA:
    """指数移动平均模型参数。"""
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
        """切换到 EMA 参数（用于验证/推理）。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复训练参数。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ═══════════════════════════════════════════════════════════════════════════
# 【改进 3】分层学习率构建器
# ───────────────────────────────────────────────────────────────────────────
# 修改了什么：新增 build_optimizer 函数
# 为什么：预训练编码器（ResNet50 + ViT）需低学习率微调，随机初始化的
#         解码器/FFC 需要较大学习率，统一学习率会导致其中一方训练不充分。
# 解决问题：报告「不足 7」—— 学习率策略和优化器配置不足。
# ═══════════════════════════════════════════════════════════════════════════

def build_optimizer(model, base_lr=2e-4, weight_decay=1e-4):
    """
    为 YFFC 模型构建分层学习率 AdamW 优化器。

    参数分组（4 组）：
      - 预训练编码器（ResNet stem + layer1~4, ViT）：base_lr * 0.1
      - 融合层（reduce, fuse, cba 等）：              base_lr * 0.5
      - 解码器（GradientDecoder + MainDecoder + EnhancementLayers）：base_lr
      - FFC 模块：                                     base_lr * 0.5
    """
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
        {'params': encoder_pretrained, 'lr': base_lr * 0.1,  'name': 'encoder_pretrained'},
        {'params': other_params,       'lr': base_lr * 0.5,  'name': 'fusion_layers'},
        {'params': decoder_params,     'lr': base_lr,         'name': 'decoders'},
        {'params': ffc_params,         'lr': base_lr * 0.5,  'name': 'ffc_modules'},
    ]

    # 过滤空组（防止某些参数组无参数时 AdamW 报错）
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    # 打印分组信息
    for g in param_groups:
        n_params = sum(p.numel() for p in g['params'])
        print(f"  优化器分组 [{g['name']}]: {n_params:,} 参数, lr={g['lr']:.2e}")

    return optimizer


# ═══════════════════════════════════════════════════════════════════════════
# 【改进 3 续】Warmup + 余弦退火调度器
# ───────────────────────────────────────────────────────────────────────────
# 修改了什么：新增 build_scheduler 函数
# 为什么：Warmup 保护预训练权重在初期不被大梯度破坏；
#         余弦退火避免后期陷入局部极小。
# 解决问题：报告「不足 4、7」—— 训练策略缺失、学习率策略不足。
# ═══════════════════════════════════════════════════════════════════════════

def build_scheduler(optimizer, warmup_iters=500, total_epochs=200,
                    iters_per_epoch=100, eta_min=1e-6):
    """
    构建 Warmup + 余弦退火调度器。

    前 warmup_iters 步：线性升温 0.01*lr → lr
    之后：余弦退火带周期重启（T_0=50, T_mult=2）
    """
    from torch.optim.lr_scheduler import (
        LinearLR, CosineAnnealingWarmRestarts, SequentialLR
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_iters
    )

    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=eta_min
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iters]
    )

    return scheduler


# ═══════════════════════════════════════════════════════════════════════════
# 【改进 10】BN → GN 替换工具
# ───────────────────────────────────────────────────────────────────────────
# 修改了什么：新增 replace_bn_with_gn 函数
# 为什么：小 batch size 下 BatchNorm 统计量噪声极大，
#         GroupNorm 不依赖 batch 统计，天然稳定。
# 解决问题：报告「不足 5」—— BatchNorm 在小 batch size 下统计不稳定。
# ═══════════════════════════════════════════════════════════════════════════

def replace_bn_with_gn(module, num_groups=8):
    """
    递归将模型中所有 BatchNorm2d 替换为 GroupNorm。
    不修改网络定义文件，仅在训练脚本中对实例化后的模型执行替换。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # groups 不能超过 channels
            groups = min(num_groups, num_channels)
            # channels 必须能被 groups 整除
            while num_channels % groups != 0 and groups > 1:
                groups -= 1
            gn = nn.GroupNorm(groups, num_channels, eps=child.eps)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)


# ═══════════════════════════════════════════════════════════════════════════
# 【改进 8 整合】完整训练函数（AMP + 梯度累积 + EMA + 动态权重调度）
# ───────────────────────────────────────────────────────────────────────────
# 修改了什么：新增 train_one_epoch / validate / train 三个函数
# 为什么：将报告中所有训练策略改进统一落地到可执行的训练循环中，
#         包括混合精度、梯度累积、梯度裁剪、EMA、分项 loss 记录等。
# 解决问题：报告「改进 8」—— 混合精度 + 梯度累积；以及所有改进的集成。
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler,
                    scaler, ema, device,
                    accumulation_steps=4, max_grad_norm=1.0,
                    epoch=0, total_epochs=200):
    """单个 epoch 的训练循环。"""
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

        # ── 混合精度前向传播 ──────────────────────────────────────────────
        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            output = model(input_img)
            pred_diffuse = output['r']
            pred_grad = output['g']
            est_b = output['b']

            # SSIM 在 float32 下计算（AMP 精度保护）
            with torch.amp.autocast(device_type='cuda', enabled=False):
                loss, loss_dict = criterion(
                    pred_diffuse.float(), target_img.float(),
                    est_b.float(), gt_specular.float(),
                    pred_grad.float(), gt_gradient.float()
                )

            loss = loss / accumulation_steps

        # ── 反向传播 ──────────────────────────────────────────────────────
        if device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ── 梯度累积 + 梯度裁剪 + 参数更新 ───────────────────────────────
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

            # EMA 更新
            if ema is not None:
                ema.update()

        # 学习率更新（每 iteration）
        if scheduler is not None:
            scheduler.step()

        running_loss += loss_dict['total']
        running_ssim += (1.0 - loss_dict['ssim'])  # SSIM 值 = 1 - loss
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
        pred_diffuse = output['r']
        pred_grad = output['g']
        est_b = output['b']

        loss, loss_dict = criterion(
            pred_diffuse, target_img,
            est_b, gt_specular,
            pred_grad, gt_gradient
        )

        running_loss += loss_dict['total']
        running_ssim += (1.0 - loss_dict['ssim'])
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    avg_ssim = running_ssim / max(n_batches, 1)
    return avg_loss, avg_ssim


def train(args):
    """
    完整训练入口，整合全部 10 项改进。

    用法示例：
        python Y_FFC_optimized.py train \\
            --input_dir ./data/input \\
            --target_dir ./data/target \\
            --epochs 200 --batch_size 2 --base_lr 2e-4 \\
            --use_gn --use_ema --accumulation_steps 4
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备: {device}")

    # ── 1. 模型创建 ──────────────────────────────────────────────────────
    model = YFFC()

    # 【改进 10】可选 BN → GN 替换
    if args.use_gn:
        print("【改进 10】将 BatchNorm 替换为 GroupNorm...")
        replace_bn_with_gn(model, num_groups=8)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # ── 2. 数据集 + 数据增强 ──────────────────────────────────────────────
    # 【改进 5】PairedTransform 数据增强
    paired_aug = PairedTransform(
        p_flip=0.5, p_rotate=0.3, p_crop=0.5,
        crop_scale=(0.7, 1.0), brightness_range=(0.9, 1.1)
    ) if args.use_augmentation else None

    train_dataset = FringeDataset(
        args.input_dir, args.target_dir,
        resize=args.image_size,
        paired_transform=paired_aug,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )

    # 验证集（如指定）
    val_loader = None
    if args.val_input_dir and args.val_target_dir:
        val_dataset = FringeDataset(
            args.val_input_dir, args.val_target_dir,
            resize=args.image_size,
            paired_transform=None,  # 验证集不做增强
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
        )

    # ── 3. 损失函数 ──────────────────────────────────────────────────────
    # 【改进 1 + 2 + 9】复合损失（SSIM + L1 + 频域 + 感知 + 镜面直接 + 统计 + 梯度 + 边缘加权）
    criterion = CompositeLoss(
        lambda_ssim=1.0,
        lambda_l1=1.0,
        lambda_freq=0.1,
        lambda_perceptual=0.05,
        lambda_spec=0.5,
        lambda_stat=0.1,
        lambda_grad=0.5,
        use_edge_weight=True,
        use_perceptual=True,
    ).to(device)

    # 【改进 6】动态权重调度器
    loss_weight_scheduler = DynamicLossWeightScheduler(total_epochs=args.epochs)

    # ── 4. 优化器 + 学习率调度 ────────────────────────────────────────────
    # 【改进 3】分层学习率 + AdamW
    print("\n【改进 3】构建分层学习率优化器...")
    optimizer = build_optimizer(model, base_lr=args.base_lr, weight_decay=1e-4)

    # 【改进 3 续】Warmup + 余弦退火
    iters_per_epoch = len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        warmup_iters=min(500, iters_per_epoch * 3),  # warmup 不超过 3 个 epoch
        total_epochs=args.epochs,
        iters_per_epoch=iters_per_epoch,
    )

    # ── 5. AMP + EMA ─────────────────────────────────────────────────────
    # 【改进 8】混合精度
    scaler = torch.amp.GradScaler(device='cuda', enabled=(device.type == 'cuda'))

    # 【改进 7】EMA
    ema = EMA(model, decay=0.999) if args.use_ema else None
    if ema:
        print("【改进 7】EMA 已启用 (decay=0.999)")

    # ── 6. 训练循环 ──────────────────────────────────────────────────────
    best_val_ssim = 0.0
    print(f"\n开始训练: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"accumulation_steps={args.accumulation_steps}")
    print(f"等效 batch size = {args.batch_size * args.accumulation_steps}")
    print("=" * 70)

    for epoch in range(args.epochs):
        t0 = time.time()

        # 【改进 6】动态调整损失权重
        weights = loss_weight_scheduler.step(criterion, epoch)

        # 训练
        train_loss, train_ssim = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, ema, device,
            accumulation_steps=args.accumulation_steps,
            max_grad_norm=1.0,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        elapsed = time.time() - t0

        # 验证
        val_loss, val_ssim = 0.0, 0.0
        if val_loader is not None:
            if ema is not None:
                ema.apply_shadow()
            val_loss, val_ssim = validate(model, val_loader, criterion, device)
            if ema is not None:
                ema.restore()

        # 打印日志
        current_lr = optimizer.param_groups[0]['lr']
        log_msg = (
            f"Epoch [{epoch+1:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_SSIM={train_ssim:.4f} "
        )
        if val_loader is not None:
            log_msg += f"val_loss={val_loss:.4f} val_SSIM={val_ssim:.4f} "
        log_msg += f"lr={current_lr:.2e} time={elapsed:.1f}s"
        print(log_msg)

        # 保存最佳模型
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
                save_dict['ema_shadow'] = ema.shadow
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(save_dict, save_path)
            print(f"  ✓ 最佳模型已保存: SSIM={best_val_ssim:.4f}")

        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)

    print("=" * 70)
    print(f"训练完成！最佳 SSIM: {best_val_ssim:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 辅助工具函数（原有，保持不变）
# ═══════════════════════════════════════════════════════════════════════════

def display_basic_model_info(model):
    print("\n" + "=" * 60)
    print("模型层次结构")
    print("=" * 60)

    print("\n【主要模块】")
    for name, module in model.named_children():
        print(f"  • {name}: {module.__class__.__name__}")

    print("\n【详细结构】")
    for name, module in model.named_children():
        print(f"\n{name} ({module.__class__.__name__}):")
        for sub_name, sub_module in module.named_children():
            print(f"  └─ {sub_name}: {sub_module.__class__.__name__}")

    print("\n" + "=" * 60)
    print("模型参数信息")
    print("=" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    print("\n【各模块参数量】")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        percentage = (module_params / total_params * 100) if total_params > 0 else 0
        print(f"  • {name}: {module_params:,} ({percentage:.1f}%)")


def display_model_summary(model, input_size, device="cpu"):
    try:
        from torchinfo import summary as torchinfo_summary
        print("使用 torchinfo 显示详细模型结构...")
        print("=" * 60)
        torchinfo_summary(model, input_size=input_size, device=device, verbose=1)
        return True
    except ImportError:
        try:
            from torchsummary import summary as torchsummary_summary
            print("使用 torchsummary 显示详细模型结构...")
            print("=" * 60)
            torchsummary_summary(model, input_size=input_size[1:], device=device)
            return True
        except ImportError:
            print("\n提示: 未安装 torchinfo 或 torchsummary，改为显示基本信息。\n")
            display_basic_model_info(model)
            return False


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Y-FFC 优化版——网络测试与训练")
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # ── 子命令 1: test（原有功能，网络结构测试）──────────────────────────
    test_parser = subparsers.add_parser('test', help='网络前向测试')
    test_parser.add_argument("--input_channels", type=int, default=1)
    test_parser.add_argument("--image_size", type=int, default=512)
    test_parser.add_argument("--batch_size", type=int, default=1)
    test_parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    test_parser.add_argument("--show_params", action="store_true", default=True)
    test_parser.add_argument("--no-show_params", action="store_true")
    test_parser.add_argument("--show_summary", action="store_true", default=True)
    test_parser.add_argument("--no-show_summary", action="store_true")

    # ── 子命令 2: train（新增训练入口）──────────────────────────────────
    train_parser = subparsers.add_parser('train', help='模型训练')
    train_parser.add_argument("--input_dir", type=str, required=True, help="训练输入图像目录")
    train_parser.add_argument("--target_dir", type=str, required=True, help="训练目标图像目录")
    train_parser.add_argument("--val_input_dir", type=str, default=None, help="验证输入图像目录")
    train_parser.add_argument("--val_target_dir", type=str, default=None, help="验证目标图像目录")
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch_size", type=int, default=2)
    train_parser.add_argument("--image_size", type=int, default=512)
    train_parser.add_argument("--base_lr", type=float, default=2e-4)
    train_parser.add_argument("--accumulation_steps", type=int, default=4)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--save_dir", type=str, default="./checkpoints")
    train_parser.add_argument("--save_interval", type=int, default=10)
    train_parser.add_argument("--use_gn", action="store_true", help="启用 BN→GN 替换")
    train_parser.add_argument("--use_ema", action="store_true", help="启用 EMA")
    train_parser.add_argument("--use_augmentation", action="store_true", default=True,
                              help="启用数据增强")
    train_parser.add_argument("--no_augmentation", action="store_true", help="禁用数据增强")

    args = parser.parse_args()

    # 如果没有指定子命令，默认运行 test
    if args.command is None:
        args.command = 'test'
        args.input_channels = 1
        args.image_size = 512
        args.batch_size = 1
        args.device = 'cpu'
        args.show_params = True
        args.show_summary = True
        setattr(args, 'no_show_params', False)
        setattr(args, 'no_show_summary', False)

    if args.command == 'test':
        if getattr(args, 'no_show_params', False):
            args.show_params = False
        if getattr(args, 'no_show_summary', False):
            args.show_summary = False

        print("=" * 60)
        print("Y-FFC 优化版 网络前向测试")
        print("=" * 60)
        print(f"input_channels={args.input_channels}, image_size={args.image_size}, "
              f"batch_size={args.batch_size}, device={args.device}")

        try:
            model = YFFC()
            print("✓ 模型创建成功")
        except Exception as e:
            print(f"✗ 模型创建失败: {e}")
            import traceback
            traceback.print_exc()
            return

        device = torch.device(args.device)
        if args.device == "cuda" and not torch.cuda.is_available():
            print("警告: CUDA 不可用，切换到 CPU")
            device = torch.device("cpu")

        model.to(device)

        input_size = (args.batch_size, args.input_channels, args.image_size, args.image_size)
        test_input = torch.randn(input_size).to(device)

        try:
            with torch.no_grad():
                out = model(test_input)
            print("✓ 前向传播成功")
            print(f"r: {out['r'].shape}, g: {out['g'].shape}, b: {out['b'].shape}")
            assert out["r"].shape == test_input.shape
            assert out["g"].shape == test_input.shape
            assert out["b"].shape == test_input.shape
            print("✓ 输出 shape 校验通过")
        except Exception as e:
            print(f"✗ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # ── 额外测试：损失函数前向传播 ──────────────────────────────────
        print("\n" + "=" * 60)
        print("损失函数测试")
        print("=" * 60)
        try:
            criterion = CompositeLoss(use_perceptual=False).to(device)  # 跳过 VGG 下载
            pred_diffuse = out['r']
            pred_grad = out['g']
            est_b = out['b']
            target = torch.rand_like(pred_diffuse)
            gt_specular = (test_input - target).clamp(0, 1)
            gt_gradient = torch.rand_like(pred_grad)

            with torch.no_grad():
                loss, loss_dict = criterion(
                    pred_diffuse, target, est_b, gt_specular,
                    pred_grad, gt_gradient
                )
            print(f"✓ CompositeLoss 前向成功, total_loss={loss.item():.4f}")
            for k, v in loss_dict.items():
                print(f"    {k}: {v:.4f}")
        except Exception as e:
            print(f"✗ 损失函数测试失败: {e}")
            import traceback
            traceback.print_exc()

        # ── 额外测试：优化器 + 调度器构建 ─────────────────────────────────
        print("\n" + "=" * 60)
        print("优化器 + 调度器测试")
        print("=" * 60)
        try:
            optimizer = build_optimizer(model, base_lr=2e-4)
            scheduler = build_scheduler(optimizer, warmup_iters=10, total_epochs=10)
            print("✓ 分层学习率优化器构建成功")
            print("✓ Warmup + 余弦退火调度器构建成功")
        except Exception as e:
            print(f"✗ 构建失败: {e}")
            import traceback
            traceback.print_exc()

        # ── 额外测试：EMA ────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("EMA 测试")
        print("=" * 60)
        try:
            ema = EMA(model, decay=0.999)
            ema.update()
            ema.apply_shadow()
            ema.restore()
            print("✓ EMA 创建/更新/切换/恢复 全部成功")
        except Exception as e:
            print(f"✗ EMA 测试失败: {e}")
            import traceback
            traceback.print_exc()

        # ── 额外测试：数据增强 ───────────────────────────────────────────
        print("\n" + "=" * 60)
        print("数据增强测试")
        print("=" * 60)
        try:
            aug = PairedTransform()
            dummy_input = torch.rand(1, 64, 64)
            dummy_target = torch.rand(1, 64, 64)
            aug_input, aug_target = aug(dummy_input, dummy_target)
            assert aug_input.shape == dummy_input.shape, f"shape 不匹配: {aug_input.shape} vs {dummy_input.shape}"
            assert aug_target.shape == dummy_target.shape
            print(f"✓ PairedTransform 测试成功: {dummy_input.shape} → {aug_input.shape}")
        except Exception as e:
            print(f"✗ 数据增强测试失败: {e}")
            import traceback
            traceback.print_exc()

        # ── 额外测试：BN → GN 替换 ──────────────────────────────────────
        print("\n" + "=" * 60)
        print("BN → GN 替换测试")
        print("=" * 60)
        try:
            model_gn = YFFC()
            bn_count = sum(1 for m in model_gn.modules() if isinstance(m, nn.BatchNorm2d))
            replace_bn_with_gn(model_gn)
            gn_count = sum(1 for m in model_gn.modules() if isinstance(m, nn.GroupNorm))
            bn_after = sum(1 for m in model_gn.modules() if isinstance(m, nn.BatchNorm2d))
            print(f"✓ 替换前 BN 数量: {bn_count}, 替换后 GN 数量: {gn_count}, 剩余 BN: {bn_after}")

            # 验证替换后前向传播仍正常
            model_gn.to(device)
            with torch.no_grad():
                out_gn = model_gn(test_input)
            assert out_gn["r"].shape == test_input.shape
            print("✓ BN→GN 替换后前向传播正常")
            del model_gn
        except Exception as e:
            print(f"✗ BN→GN 测试失败: {e}")
            import traceback
            traceback.print_exc()

        # ── 额外测试：动态权重调度 ───────────────────────────────────────
        print("\n" + "=" * 60)
        print("动态权重调度测试")
        print("=" * 60)
        try:
            dyn_sched = DynamicLossWeightScheduler(total_epochs=100)
            w0 = dyn_sched.get_weights(0)
            w50 = dyn_sched.get_weights(50)
            w100 = dyn_sched.get_weights(100)
            print(f"✓ epoch 0:   lambda_l1={w0['lambda_l1']:.2f}, lambda_freq={w0['lambda_freq']:.3f}")
            print(f"  epoch 50:  lambda_l1={w50['lambda_l1']:.2f}, lambda_freq={w50['lambda_freq']:.3f}")
            print(f"  epoch 100: lambda_l1={w100['lambda_l1']:.2f}, lambda_freq={w100['lambda_freq']:.3f}")
        except Exception as e:
            print(f"✗ 动态权重调度测试失败: {e}")

        if args.show_params:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("\n" + "=" * 60)
            print("参数统计")
            print("=" * 60)
            print(f"总参数数量: {total_params:,}")
            print(f"可训练参数: {trainable_params:,}")
            print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

        if args.show_summary:
            print("=" * 60)
            print("模型结构摘要")
            print("=" * 60)
            display_model_summary(model, input_size, device=str(device))

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)

    elif args.command == 'train':
        if hasattr(args, 'no_augmentation') and args.no_augmentation:
            args.use_augmentation = False
        train(args)


if __name__ == "__main__":
    main()
