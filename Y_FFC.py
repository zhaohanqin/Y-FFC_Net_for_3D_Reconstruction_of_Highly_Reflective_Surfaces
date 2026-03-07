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

from ffc import *


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
    # ║  原代码：pred.mean() / pred.std() 在整个 batch 上统计，            ║
    # ║  批内多张图像的统计量互相抵消，掩盖了单样本的预测误差，             ║
    # ║  导致模型可以"让批内均值/方差一致"来最小化这个 loss，              ║
    # ║  而无需真正预测准确每张图。                                        ║
    # ║  修改后：对 batch 中每张图单独计算 mean/std 差异，再取均值。        ║
    # ╚══════════════════════════════════════════════════════════════════╝
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)   # [B, H*W*C]
    gt_flat   = gt.view(B, -1)     # [B, H*W*C]
    # 逐样本均值差异
    mean_loss = torch.abs(pred_flat.mean(dim=1) - gt_flat.mean(dim=1)).mean()
    # 逐样本标准差差异（unbiased=False 与原 .std() 行为保持一致，避免小 batch 时 NaN）
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
        # 改进点 E：psi 输出 F_l 通道（channel-wise），每通道独立空间注意力权重
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
        psi = self.psi(psi)   # [B, F_l, H, W]，channel-wise 注意力权重
        return x * psi        # 每通道独立加权，shape 与原版一致


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
    改进点 C：三路转置卷积改为差异化 kernel size，消除原设计三路完全相同导致的
    结构性冗余，使各路真正具备不同感受野的多尺度上采样能力。

    输出尺寸一致性验证（stride=2, input=H）：
      output = (H-1)*2 - 2*padding + kernel_size = 2H（需满足 kernel - 2*padding = 2）
      branch_1: k=2, p=0  → 2-0=2 ✓  最小感受野，保留精细边缘细节
      branch_2: k=4, p=1  → 4-2=2 ✓  中等感受野（原始设计），平衡纹理
      branch_3: k=6, p=2  → 6-4=2 ✓  大感受野，捕获低频结构与周期条纹
    三路输出均为 [B, out_channels_branch, 2H, 2W]，cat 后 3*out_channels_branch。
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
        # 若极端输入尺寸下三路有 1 像素偏差，对齐到 branch_2 的尺寸
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
    """
    ViT 全局特征分支：
    - 不参与 skip connections
    - 只输出最后层 patch tokens 的二维特征图 [B, Cv, Hv, Wv]
    """
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

        x = self.vit.conv_proj(x)  # [B, Cv, Gh, Gw]
        grid_h, grid_w = x.shape[-2], x.shape[-1]

        x = x.flatten(2).transpose(1, 2)  # [B, N, Cv]

        if hasattr(self.vit, "class_token"):
            cls_token = self.vit.class_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = self.vit.encoder(x)  # [B, N(+1), Cv]

        if x.shape[1] == grid_h * grid_w + 1:
            x = x[:, 1:, :]

        x = x.transpose(1, 2).reshape(batch_size, x.shape[2], grid_h, grid_w)  # [B, Cv, Gh, Gw]
        return x


class Encoder(nn.Module):
    """
    ResNet50 编码器（skip 来自 ResNet）+ ViT 分支（仅瓶颈融合）

    输出保持与原解码器兼容：
      x_bb_1: [B, 64,  H/2,  W/2]
      x_bb_2: [B, 128, H/4,  W/4]
      x_bb_3: [B, 256, H/8,  W/8]
      x_bb_4: [B, 512, H/16, W/16]   (仅 ResNet 特征，用于所有 skip connections)
      x_c:    [B, 256, H/16, W/16]   (ResNet+ViT 融合后的瓶颈输入再压到 256)
    """
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

        self.layer1 = backbone_model.layer1  # /4, 256
        self.layer2 = backbone_model.layer2  # /8, 512
        self.layer3 = backbone_model.layer3  # /16, 1024
        self.layer4 = backbone_model.layer4  # /16, 2048 (dilated)

        self.reduce_l1 = Conv2D_BN_activa(256, 128, 1, 1, 0)
        self.reduce_l2 = Conv2D_BN_activa(512, 256, 1, 1, 0)

        self.reduce_l3 = Conv2D_BN_activa(1024, 512, 1, 1, 0)
        self.reduce_l4 = Conv2D_BN_activa(2048, 512, 1, 1, 0)
        self.fuse_l34 = Conv2D_BN_activa(1024, 512, 3, 1, 1)

        self.vit_branch = ViTEncoderBranch(vit_image_size=vit_image_size)

        self.vit_resnet_fuse_1x1 = Conv2D_BN_activa(512 + self.vit_branch.embed_dim, 512, 1, 1, 0)

        self.cba_after_backbone = Conv2D_BN_activa(512, 256, 3, 1, 1)

        # ── 【新增 FFC-1】Encoder 瓶颈频域增强（H/16, 256ch） ────────────────
        # 位置：cba_after_backbone 输出后（ResNet+ViT 融合完成后的 x_c，256ch，H/16）。
        #
        # 问题：当前 x_c 仅经过卷积生成，ResNet 提供局部空间特征、ViT 提供全局语义，
        # 但两者均为空间域表示，缺乏频率维度的建模。镜面反射在频域中具有特定的
        # 高频尖锐特征与全局频谱分布（如宽带白色谱峰），纯空间卷积无法有效编码。
        #
        # 收益：x_c 同时输入 GradientDecoder 和 MainDecoder，在此处引入 FFC 意味着
        # 两条解码路径都能从"带频域感知的瓶颈特征"出发，对镜面高光区域的全局
        # 频率特征有整体性的感知基础，效率最高（一处改动惠及两路解码）。
        #
        # ratio=0.5：瓶颈层特征已高度抽象，局部/全局各占一半，均衡分配最合理。
        # 残差连接保留原有空间特征，FFC 贡献额外频域补充，不破坏已有信息。
        self.ffc_bottleneck = FFC_BN_ACT(256, 256, 3, ratio_gin=0.5, ratio_gout=0.5, padding=1)
        # ────────────────────────────────────────────────────────────────────

    def forward(self, x):
        x_in = x

        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x_bb_1 = self.stem_relu(x)  # /2, 64

        x = self.stem_pool(x_bb_1)  # /4
        x_l1 = self.layer1(x)       # /4, 256
        x_bb_2 = self.reduce_l1(x_l1)  # /4, 128

        x_l2 = self.layer2(x_l1)    # /8, 512
        x_bb_3 = self.reduce_l2(x_l2)  # /8, 256

        x_l3 = self.layer3(x_l2)    # /16, 1024
        x_l4 = self.layer4(x_l3)    # /16, 2048

        x_l3_512 = self.reduce_l3(x_l3)  # /16, 512
        x_l4_512 = self.reduce_l4(x_l4)  # /16, 512
        x_bb_4 = self.fuse_l34(torch.cat([x_l3_512, x_l4_512], dim=1))  # /16, 512

        vit_feat = self.vit_branch(x_in)  # [B, Cv, Gh, Gw]
        if vit_feat.shape[-2:] != x_bb_4.shape[-2:]:
            vit_feat = F.interpolate(vit_feat, size=x_bb_4.shape[-2:], mode="bilinear", align_corners=False)

        fused = torch.cat([x_bb_4, vit_feat], dim=1)        # [B, 512+Cv, H/16, W/16]
        fused = self.vit_resnet_fuse_1x1(fused)             # [B, 512,     H/16, W/16]
        x_c = self.cba_after_backbone(fused)                # [B, 256,     H/16, W/16]

        # ── 【新增 FFC-1】瓶颈频域增强（残差）──────────────────────────────
        # 将 256ch 按 ratio=0.5 拆分：局部路径 128ch + 全局路径 128ch，
        # FFC 后 cat 还原 256ch，残差叠加保留原有空间特征，不改变 x_c 的 shape。
        # x_c 同时作为两路解码器入口，此处一次 FFC 使两路解码均获益。
        _nl_btn = 256 - int(256 * 0.5)   # 128（局部路径通道数）
        _ffc_btn_out = self.ffc_bottleneck((x_c[:, :_nl_btn], x_c[:, _nl_btn:]))
        x_c = torch.cat([_ffc_btn_out[0], _ffc_btn_out[1]], dim=1) + x_c  # 256ch 残差
        # ────────────────────────────────────────────────────────────────────

        return x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c


class GradientDecoder(nn.Module):
    def __init__(self):
        super(GradientDecoder, self).__init__()
        # 改进点 A：在 GradientDecoder 入口融合 ViT 全局上下文特征 x_c。
        # x_bb_4: [B, 512, H/16, W/16]  —— ResNet 局部特征
        # x_c:    [B, 256, H/16, W/16]  —— ResNet+ViT 融合的全局瓶颈特征
        # cat 后 768ch，用 1×1 Conv 压回 512ch，保持 block_1_conv_1 输入通道不变，
        # 后续所有层无需任何修改。
        self.fuse_global_context = Conv2D_BN_activa(512 + 256, 512, 1, 1, 0)

        self.block_1_conv_1 = Conv2D_BN_activa(512, 1024, 7, 1, 3)
        self.block_1_conv_2 = Conv2D_BN_activa(1024, 512, 1, 1, 0)
        self.block_1_conv_3 = Conv2D_BN_activa(512, 256, 3, 1, 1)

        # ── 新增 FFC：GradientDecoder H/16 频域增强 ──────────────────────────
        # 位置：block_1_conv_3 输出后（256ch, H/16），block_1_conv_transpose 前。
        # 此处是 GradientDecoder 最深层表示（也是 x_to_be_enhanced_1 的生成前一步），
        # 完全由局部卷积（7×7+1×1+3×3）生成，感受野仅覆盖局部区域，
        # 对跨越整图的周期性条纹结构缺乏全局感知。
        #
        # ratio=0.25：256ch 中 192ch 保留空间局部路径，64ch 走频域全局路径。
        # 以残差方式叠加（不改变通道数），x_to_be_enhanced_1 生成过程及后续
        # 所有 cat 路径完全不变，EnhancementLayers 接口零影响。
        #
        # 物理收益：x_to_be_enhanced_1 进入 EnhancementLayers 后用于投影条纹
        # vs 镜面分离，加入全局条纹相位感知后，分离依据从纯局部跳变到全图周期约束。
        self.ffc_grad_h16 = FFC_BN_ACT(256, 256, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        # ────────────────────────────────────────────────────────────────────

        self.block_1_conv_transpose = Conv2D_BN_activa(256, 256, 5, 1, 2, transpose=False)

        self.block_2_conv = Conv2D_BN_activa(256 + 512, 128, 3, 1, 1)

        # ── 新增 FFC：GradientDecoder H/8 频域增强 ───────────────────────────
        # 位置：block_2_conv 输出后（128ch, H/8），block_2_conv_transpose 前。
        # H/8 分辨率是条纹边缘空间频率响应最清晰的尺度（条纹周期在此可完整表达），
        # 也是 x_to_be_enhanced_2 的生成前一步，同样全程无 FFC 覆盖。
        #
        # ratio=0.25：128ch 中 96ch 保留空间局部路径，32ch 走频域全局路径。
        # 以残差方式叠加，通道数不变，后续 cat 及 skip connection 零影响。
        #
        # 物理收益：x_to_be_enhanced_2 在 H/8 尺度感知到条纹频率峰值，
        # EnhancementLayers 第二尺度分解的依据从局部纹理升级到频率特征。
        self.ffc_grad_h8 = FFC_BN_ACT(128, 128, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        # ────────────────────────────────────────────────────────────────────

        self.block_2_conv_transpose = Conv2D_BN_activa(128, 128, 4, 2, 1, transpose=True)

        self.block_3_conv = Conv2D_BN_activa(128 + 256, 64, 3, 1, 1)

        # ── 【新增 FFC-2】GradientDecoder H/4 频域增强 ───────────────────────
        # 位置：block_3_conv 输出后（64ch, H/4），block_3_conv_transpose 前。
        # 此处生成 x_to_be_enhanced_3，即 EnhancementLayers 第三尺度分解的输入。
        #
        # 问题：H/4 是镜面高光边缘细节最丰富的尺度之一，当前该层仅有
        # 卷积处理（局部感受野），没有任何频域覆盖，对条纹与镜面反射高频
        # 混叠区域的辨别能力极弱。
        #
        # 收益：EnhancementLayers 第三尺度的分解依据（projection vs specular）
        # 在此之前就已获得频率特征支持，使 H/4 的镜面梯度分离更精准；
        # 同时高光边缘处的频率转变（低→高）在此尺度最为显著，FFC 能有效捕获。
        #
        # ratio=0.25：H/4 特征图较大，频域路径保守占 1/4（16ch），
        # 保留更多局部细节，避免大尺寸特征图的频域计算开销过大。
        self.ffc_grad_h4 = FFC_BN_ACT(64, 64, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        # ────────────────────────────────────────────────────────────────────

        self.block_3_conv_transpose = Conv2D_BN_activa(64, 64, 4, 2, 1, transpose=True)

        self.block_4_conv = Conv2D_BN_activa(64 + 128, 32, 3, 1, 1)
        self.block_4_conv_transpose = Conv2D_BN_activa(32, 32, 4, 2, 1, transpose=True)

        self.block_5_conv_1 = Conv2D_BN_activa(32 + 64, 64, 4, 2, 1, transpose=True)
        self.block_5_conv_2 = Conv2D_BN_activa(64, 1, 5, 1, 2, activation=None, if_bn=False)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, backbone_features, x_c):
        x_bb_1, x_bb_2, x_bb_3, x_bb_4 = backbone_features

        # 改进点 A：将 ViT 全局信息 x_c 与 ResNet 深层特征 x_bb_4 在空间上拼接，
        # 再用 fuse_global_context（1×1 Conv）压回 512ch，使梯度解码全程携带
        # 全图条纹相位感知，x_c 的 spatial 尺寸与 x_bb_4 相同（均为 H/16）。
        if x_c.shape[-2:] != x_bb_4.shape[-2:]:
            x_c = F.interpolate(x_c, size=x_bb_4.shape[-2:], mode="bilinear", align_corners=False)
        x_bb_4_fused = self.fuse_global_context(torch.cat([x_bb_4, x_c], dim=1))  # [B, 512, H/16, W/16]

        x = self.block_1_conv_1(x_bb_4_fused)
        x = self.block_1_conv_2(x)
        x = self.block_1_conv_3(x)   # [B, 256, H/16, W/16]

        # ── 新增 FFC：H/16 梯度频域增强（残差）───────────────────────────────
        # 将 256ch 按 ratio=0.25 拆分：局部路径 192ch + 全局路径 64ch，
        # FFC 后 cat 还原 256ch，残差叠加不改变 shape，后续流程不受影响。
        _nl_h16 = 256 - int(256 * 0.25)   # 192（局部路径通道数）
        _ffc_h16_out = self.ffc_grad_h16((x[:, :_nl_h16], x[:, _nl_h16:]))
        x = torch.cat([_ffc_h16_out[0], _ffc_h16_out[1]], dim=1) + x  # 256ch 残差
        # ────────────────────────────────────────────────────────────────────

        x_to_be_enhanced_1 = self.block_1_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_1, x_bb_4], dim=1)

        x = self.block_2_conv(x)   # [B, 128, H/8, W/8]

        # ── 新增 FFC：H/8 梯度频域增强（残差）───────────────────────────────
        # 将 128ch 按 ratio=0.25 拆分：局部路径 96ch + 全局路径 32ch，
        # FFC 后 cat 还原 128ch，残差叠加不改变 shape，后续流程不受影响。
        _nl_h8 = 128 - int(128 * 0.25)    # 96（局部路径通道数）
        _ffc_h8_out = self.ffc_grad_h8((x[:, :_nl_h8], x[:, _nl_h8:]))
        x = torch.cat([_ffc_h8_out[0], _ffc_h8_out[1]], dim=1) + x  # 128ch 残差
        # ────────────────────────────────────────────────────────────────────

        x_to_be_enhanced_2 = self.block_2_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_2, x_bb_3], dim=1)

        x = self.block_3_conv(x)

        # ── 【新增 FFC-2】H/4 梯度频域增强（残差）───────────────────────────
        # 将 64ch 按 ratio=0.25 拆分：局部路径 48ch + 全局路径 16ch，
        # FFC 后 cat 还原 64ch，残差叠加不改变 shape，x_to_be_enhanced_3 携带频率信息。
        _nl_h4 = 64 - int(64 * 0.25)    # 48（局部路径通道数）
        _ffc_h4_out = self.ffc_grad_h4((x[:, :_nl_h4], x[:, _nl_h4:]))
        x = torch.cat([_ffc_h4_out[0], _ffc_h4_out[1]], dim=1) + x  # 64ch 残差
        # ────────────────────────────────────────────────────────────────────

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

        # 改进点 E：F_int 大幅缩小，消除原设计中近满维中间层的参数浪费；
        # psi 已改为输出 F_l 通道（channel-wise），此处 F_int 仅控制压缩程度。
        # att1: F_g=768, F_l=512, F_int=64   (原 640)
        # att2: F_g=384, F_l=256, F_int=64   (原 320)
        # att3: F_g=192, F_l=128, F_int=32   (原 160)
        # att4: F_g=96,  F_l=64,  F_int=16   (原 80)
        self.att1 = Attention_block(768, 512, 64)
        self.att2 = Attention_block(384, 256, 64)
        self.att3 = Attention_block(192, 128, 32)
        self.att4 = Attention_block(96,  64,  16)

        # 改进点 D：在 H/8 分辨率插入轻量 FFC，对 transposed_block_2 的 384ch 输出
        # 进行频域感知。条纹边缘在此尺度响应最清晰，当前解码路径全程为局部卷积，
        # 缺少中频段的频域约束。ratio=0.25 保留 3/4 局部路径，避免过度频域化。
        # 以残差方式叠加，不改变通道数（384ch），对后续 cat 路径零影响。
        self.ffc_h8 = FFC_BN_ACT(384, 384, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)

        # ── 【新增 FFC-4】MainDecoder H/4 频域增强 ───────────────────────────
        # 位置：transposed_block_3 输出后（192ch, H/4），att3 注意力融合前。
        #
        # 问题：transposed_block_3 将 FeatureExtrationLayersB 的 640ch 输出上采样到
        # H/4 分辨率（192ch），此处是主解码器从中分辨率过渡到高分辨率的关键节点。
        # H/4 特征图携带了丰富的镜面边缘细节，但当前解码路径在 H/4 没有任何 FFC，
        # 仅依赖 H/8 的 ffc_h8 提供频域支撑，无法感知 H/4 尺度特有的更高频率成分。
        #
        # 收益：在 att3 执行 skip connection 融合之前注入 H/4 频域特征，使注意力
        # 权重的计算能够参考频域驱动的特征（而非纯卷积输出），att3 对高光区域的
        # skip connection 通道权重分配将更精准。同时 H/4 是条纹结构最接近实际
        # 条纹周期的尺度，频域约束在此处最能抑制条纹高频混叠入漫反射输出。
        #
        # ratio=0.25：H/4 特征图较大（192ch），保守分配频域路径，
        # 避免上采样阶段过度频域化损失空间精细结构。残差连接保留所有原有特征。
        self.ffc_h4 = FFC_BN_ACT(192, 192, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        # ────────────────────────────────────────────────────────────────────

        self.output_conv1 = Conv2D_BN_activa(16 * 3 + 1, 16, 3, 1, 1)
        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  [修复 BUG-1]  output_conv2 输出层改为裸卷积 + 独立 Sigmoid      ║
        # ║  原代码使用默认参数 (activation="relu", if_bn=True)，             ║
        # ║  导致 est_b = ReLU(BN(conv(x))) ∈ [0, +∞)，完全无上界。         ║
        # ║  与 GradientDecoder 的设计对齐：                                  ║
        # ║    block_5_conv_2: activation=None, if_bn=False                  ║
        # ║    sigmoid_layer : nn.Sigmoid()                                   ║
        # ║  修改后 est_b ∈ (0, 1)，物理含义正确（镜面反射分量不超过原图）。  ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.output_conv2 = Conv2D_BN_activa(16, 1, 3, 1, 1, activation=None, if_bn=False)
        self.output_sigmoid = nn.Sigmoid()  # 与 GradientDecoder.sigmoid_layer 对称

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
        # 改进点 D：在 H/8 分辨率对 384ch 特征做一次轻量 FFC，捕获条纹边缘的中频响应。
        # 按 ratio=0.25 将 384ch 分为 local(288ch) + global(96ch)，FFC 后 cat 还原 384ch，
        # 残差叠加保留原始局部信息，不改变 spatial 尺寸，不影响后续 cat 通道计算。
        _n_l = 384 - int(384 * 0.25)   # 288
        _ffc_out = self.ffc_h8((x[:, :_n_l], x[:, _n_l:]))
        x = torch.cat([_ffc_out[0], _ffc_out[1]], dim=1) + x   # 384ch 残差
        if x_bb_3.shape[-2:] != x.shape[-2:]:
            x_bb_3 = F.interpolate(x_bb_3, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if x_enhanced_2.shape[-2:] != x.shape[-2:]:
            x_enhanced_2 = F.interpolate(x_enhanced_2, size=x.shape[-2:], mode="bilinear", align_corners=False)
        bridge = self.att2(x, x_bb_3)
        x = torch.cat([x, bridge, x_enhanced_2], dim=1)

        x = self.feature_extraction_B(x)
        x = self.transposed_block_3(x)

        # ── 【新增 FFC-4】H/4 主解码频域增强（残差）─────────────────────────
        # 将 192ch 按 ratio=0.25 拆分：局部路径 144ch + 全局路径 48ch，
        # FFC 后 cat 还原 192ch，残差叠加不改变 shape，后续 att3 + cat 路径不受影响。
        # 在注意力 skip connection（att3）执行前引入频域特征，提升高光区域的通道权重精度。
        _n_l_h4 = 192 - int(192 * 0.25)   # 144（局部路径通道数）
        _ffc_h4_out = self.ffc_h4((x[:, :_n_l_h4], x[:, _n_l_h4:]))
        x = torch.cat([_ffc_h4_out[0], _ffc_h4_out[1]], dim=1) + x   # 192ch 残差
        # ────────────────────────────────────────────────────────────────────

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
        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  [修复 BUG-1 续]  对 output_conv2 输出应用 Sigmoid               ║
        # ║  确保 est_b ∈ (0, 1)，与 __init__ 中的修改配套。                 ║
        # ╚══════════════════════════════════════════════════════════════════╝
        est_b = self.output_sigmoid(self.output_conv2(x))  # est_b ∈ (0, 1)
        return est_b


# ==================== 梯度分解单元（仅用于 EnhancementLayers 内部） ====================

class _SpectralDecompositionUnit(nn.Module):
    """
    内部使用的梯度分解单元，不对外暴露。

    设计思路：
      - projection_branch：大感受野（7×7）卷积，捕获投影条纹的低频周期性梯度。
        该分支仅在内部产生门控信号，不作为最终输出传递给解码器。
      - specular_branch：小感受野（3×3）卷积，捕获镜面反射的高频局部梯度。
        该分支的输出经过投影分支门控调制后，作为最终镜面反射梯度输出。

    门控调制逻辑：
      gate = Sigmoid( 1×1 Conv( projection ) )
      specular_refined = specular * gate + specular   （残差保留原始镜面信息）

    输出 specular_refined 的 shape 与原 EnhancementLayers 输出完全一致：
      channel = out_c，spatial = 输入 spatial（无上/下采样）。
    """

    def __init__(self, in_c: int, out_c: int):
        super(_SpectralDecompositionUnit, self).__init__()

        # 保存 out_c 为实例属性，供 forward() 中拆分通道时使用
        self.out_c = out_c

        # 投影条纹分支：大核感受野，侧重于周期性条纹结构（仅内部使用）
        self.projection_branch = Conv2D_BN_activa(in_c, out_c, 7, 1, 3)

        # 镜面反射分支：小核感受野，侧重于高频局部高光特征
        self.specular_branch = Conv2D_BN_activa(in_c, out_c, 3, 1, 1)

        # ── 【新增 FFC-3】镜面反射分支频域增强 ──────────────────────────────
        # 位置：specular_branch 卷积输出后，gate 融合前。
        #
        # 问题：specular_branch 仅是一个 3×3 卷积，感受野有限，无法捕获镜面反射
        # 在频域中的全局特征——镜面高光区域通常在低频段呈现宽带增强（高亮大面积），
        # 在高频段呈现锐利边缘（高光→阴影的快速转变），这两种特性在空间域很难同时
        # 用单一小核卷积捕获，但在频域中可以通过全局 FFT 自然建模。
        #
        # 收益：FFC 作为残差叠加在 specular 上，不破坏 projection 分支和 gate 融合
        # 的原有逻辑；新增的频域路径专门强化镜面反射的频率特征表达，使后续 gate
        # （对比 projection 与 specular 的差异）能在更鲜明的特征对比下做出更准确
        # 的"镜面 vs 条纹"判断。
        #
        # ratio=0.25：EnhancementLayers 输出通道较小（16~128ch），频域路径保守占 1/4，
        # 避免小通道数时频域路径参数量过于饱和。残差连接保留原始空间特征。
        self.ffc_specular = FFC_BN_ACT(out_c, out_c, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        # ────────────────────────────────────────────────────────────────────

        # 改进点 B：gate_conv 的输入从原来的 out_c（仅 projection）改为 2*out_c
        # （projection 与 specular 拼接），使门控信号能对比两路特征的差异进行
        # 条件分离：specular 相对 projection 的独特激活越强，gate 越高，从而真正
        # 实现"有镜面的地方放行、有条纹的地方抑制"的语义分离，而非单纯幅度放大。
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 投影条纹梯度（仅内部使用，不输出）
        projection = self.projection_branch(x)

        # 镜面反射梯度（原始）
        specular = self.specular_branch(x)

        # ── 【新增 FFC-3】镜面分支频域增强（残差）──────────────────────────
        # 将 out_c 通道按 ratio=0.25 拆分：局部路径 (out_c*0.75)ch + 全局路径 (out_c*0.25)ch，
        # FFC 后 cat 还原 out_c，残差叠加到 specular 上，不改变通道数与 shape。
        # 频域路径捕获镜面高光的全局宽带增强和边缘锐化响应，作为 specular 的频域补充。
        _nl_spec = self.out_c - int(self.out_c * 0.25)   # 局部路径通道数（out_c * 0.75）
        _ffc_spec_out = self.ffc_specular((specular[:, :_nl_spec], specular[:, _nl_spec:]))
        specular = torch.cat([_ffc_spec_out[0], _ffc_spec_out[1]], dim=1) + specular  # 残差
        # ────────────────────────────────────────────────────────────────────

        # 改进点 B：将 projection 与 specular 拼接后送入 gate_conv，
        # gate 通过比对两路激活决定各位置以哪种成分为主：
        #   gate 接近 0 → 投影条纹主导，压制镜面分量
        #   gate 接近 1 → 镜面高光主导，增强镜面分量
        gate = self.gate_conv(torch.cat([projection, specular], dim=1))

        # 残差门控调制：保留原始镜面信息，再叠加门控筛选后的增强量
        specular_refined = specular + specular * gate

        return specular_refined


class EnhancementLayers(nn.Module):
    """
    梯度特征增强层（改造版）。

    接口与原版完全一致：
      输入：to_be_enhanced_features —— GradientDecoder 输出的 4 个中间特征（tuple）
            [0]: [B, 256, H/16, W/16]
            [1]: [B, 128, H/8,  W/8 ]
            [2]: [B,  64, H/4,  W/4 ]
            [3]: [B,  32, H/2,  W/2 ]
      输出：(x1, x2, x3, x4) —— 与原版输出 shape 完全相同
            x1: [B, 128, H/16, W/16]
            x2: [B,  64, H/8,  W/8 ]
            x3: [B,  32, H/4,  W/4 ]
            x4: [B,  16, H/2,  W/2 ]

    内部改造：
      每个尺度上由 _SpectralDecompositionUnit 完成梯度分离。
      - projection 分支（内部）：提取投影条纹梯度，生成门控信号后丢弃。
      - specular 分支（输出）：提取镜面反射梯度，经门控调制后作为最终输出。
      最终输出的仅为镜面反射梯度，通道数与 spatial 尺寸与原版完全一致。
    """

    def __init__(self):
        super(EnhancementLayers, self).__init__()
        # 4 个尺度的梯度分解单元
        # 输入/输出通道与原 enhancement_layers_1~4 完全对应
        self.decomp_unit_1 = _SpectralDecompositionUnit(in_c=256, out_c=128)
        self.decomp_unit_2 = _SpectralDecompositionUnit(in_c=128, out_c=64)
        self.decomp_unit_3 = _SpectralDecompositionUnit(in_c=64,  out_c=32)
        self.decomp_unit_4 = _SpectralDecompositionUnit(in_c=32,  out_c=16)

    def forward(self, to_be_enhanced_features):
        # 每个单元内部进行梯度分离，只输出镜面反射梯度
        x1 = self.decomp_unit_1(to_be_enhanced_features[0])  # [B, 128, H/16, W/16]
        x2 = self.decomp_unit_2(to_be_enhanced_features[1])  # [B,  64, H/8,  W/8 ]
        x3 = self.decomp_unit_3(to_be_enhanced_features[2])  # [B,  32, H/4,  W/4 ]
        x4 = self.decomp_unit_4(to_be_enhanced_features[3])  # [B,  16, H/2,  W/2 ]
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
        # 改进点 A：将编码器瓶颈特征 x_c（c_features[-1]，含 ViT 全局信息）
        # 同时传入 GradientDecoder，使梯度解码分支获得全图条纹相位感知。
        # c_features[:-1] 仍为 (x_bb_1, x_bb_2, x_bb_3, x_bb_4)，skip connection 不变。
        g_features = self.gradient_decoder(c_features[:-1], c_features[-1])
        enhanced_features = self.enhancement_layers(g_features[:-1])
        est_b = self.main_decoder(c_features, enhanced_features, g_features[-1])
        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  [修复 BUG-2]  pred_diffuse = x - est_b 需要 clamp 到 [0, 1]    ║
        # ║  原代码直接返回 x - est_b，当 est_b > x 时（即镜面反射超过        ║
        # ║  原图像素），差值为负，超出 SSIM 的工作量纲 [0, 1]，              ║
        # ║  导致 SSIM 计算完全失效（输出接近 0）。                           ║
        # ║  修复后 BUG-1 已保证 est_b ∈ (0,1)，clamp 进一步作为物理兜底：   ║
        # ║  漫反射分量在物理上不能为负、也不能超过原图亮度。                  ║
        # ╚══════════════════════════════════════════════════════════════════╝
        pred_diffuse = torch.clamp(x - est_b, 0.0, 1.0)
        return {"r": pred_diffuse, "g": g_features[-1], "b": est_b}


class FringeDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None, resize=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.resize = resize
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

        gt_specular = input_img - target_img

        gx, gy = torch.gradient(target_img, dim=(1, 2))
        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  [修复 BUG-3]  gt_gradient 归一化到 [0, 1] 以匹配 pred 量纲      ║
        # ║  原代码：gt_gradient = (gx + gy) / 2.0 ∈ 约 [-0.5, 0.5]，       ║
        # ║  含负值；而 GradientDecoder 输出经过 Sigmoid，                    ║
        # ║  pred_gradient ∈ (0, 1)，全为正值。                              ║
        # ║  量纲不匹配导致：                                                  ║
        # ║    ① SSIM(pred, gt) 无效（两者数值区间不同）                       ║
        # ║    ② 当两者均接近 0 时 SSIM 虚假为 1，lg_ssim ≈ 0，              ║
        # ║       梯度分支对训练完全失去监督作用。                              ║
        # ║  修复：将梯度图 +0.5 平移后 clamp 到 [0,1]，                      ║
        # ║  与 Sigmoid 输出 (0,1) 对齐，保留梯度方向信息的相对大小。          ║
        # ╚══════════════════════════════════════════════════════════════════╝
        gt_gradient_raw = (gx + gy) / 2.0                       # ∈ 约 [-0.5, 0.5]
        gt_gradient = (gt_gradient_raw + 0.5).clamp(0.0, 1.0)  # 平移到 [0, 1]

        return input_img, target_img, gt_specular, gt_gradient


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

def main():
    parser = argparse.ArgumentParser(description="Y-FFC 网络结构显示与前向测试")
    parser.add_argument("--input_channels", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--show_params", action="store_true", default=True)
    parser.add_argument("--no-show_params", action="store_true")
    parser.add_argument("--show_summary", action="store_true", default=True)
    parser.add_argument("--no-show_summary", action="store_true")
    args = parser.parse_args()

    if args.no_show_params:
        args.show_params = False
    if args.no_show_summary:
        args.show_summary = False

    print("=" * 60)
    print("Y-FFC 网络前向测试")
    print("=" * 60)
    print(f"input_channels={args.input_channels}, image_size={args.image_size}, batch_size={args.batch_size}, device={args.device}")

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

    if args.show_params:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("=" * 60)
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

    print("=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
