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
    mean_loss = torch.abs(pred.mean() - gt.mean())
    std_loss = torch.abs(pred.std() - gt.std())
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
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
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
    def __init__(self, in_channels, out_channels_branch=256):
        super(TransposedConvBlock, self).__init__()
        self.branch_1 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)
        self.branch_2 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)
        self.branch_3 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
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

        # 作用：ResNet bottleneck 特征与 ViT 全局特征拼接后，用 1×1 Conv 把通道压回 512，
        # 从而保证后续瓶颈层输入 shape 与原模型一致（解码器不需要改）
        self.vit_resnet_fuse_1x1 = Conv2D_BN_activa(512 + self.vit_branch.embed_dim, 512, 1, 1, 0)

        # 原模型瓶颈：输入 512 -> 输出 256（保持不变）
        self.cba_after_backbone = Conv2D_BN_activa(512, 256, 3, 1, 1)

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
        x_bb_4 = self.fuse_l34(torch.cat([x_l3_512, x_l4_512], dim=1))  # /16, 512 (仅 ResNet，用于 skip)

        vit_feat = self.vit_branch(x_in)  # [B, Cv, Gh, Gw]
        if vit_feat.shape[-2:] != x_bb_4.shape[-2:]:
            vit_feat = F.interpolate(vit_feat, size=x_bb_4.shape[-2:], mode="bilinear", align_corners=False)

        fused = torch.cat([x_bb_4, vit_feat], dim=1)        # [B, 512+Cv, H/16, W/16]
        fused = self.vit_resnet_fuse_1x1(fused)             # [B, 512,     H/16, W/16]
        x_c = self.cba_after_backbone(fused)                # [B, 256,     H/16, W/16]

        return x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c


class GradientDecoder(nn.Module):
    def __init__(self):
        super(GradientDecoder, self).__init__()
        self.block_1_conv_1 = Conv2D_BN_activa(512, 1024, 7, 1, 3)
        self.block_1_conv_2 = Conv2D_BN_activa(1024, 512, 1, 1, 0)
        self.block_1_conv_3 = Conv2D_BN_activa(512, 256, 3, 1, 1)
        self.block_1_conv_transpose = Conv2D_BN_activa(256, 256, 5, 1, 2, transpose=False)

        self.block_2_conv = Conv2D_BN_activa(256 + 512, 128, 3, 1, 1)
        self.block_2_conv_transpose = Conv2D_BN_activa(128, 128, 4, 2, 1, transpose=True)

        self.block_3_conv = Conv2D_BN_activa(128 + 256, 64, 3, 1, 1)
        self.block_3_conv_transpose = Conv2D_BN_activa(64, 64, 4, 2, 1, transpose=True)

        self.block_4_conv = Conv2D_BN_activa(64 + 128, 32, 3, 1, 1)
        self.block_4_conv_transpose = Conv2D_BN_activa(32, 32, 4, 2, 1, transpose=True)

        self.block_5_conv_1 = Conv2D_BN_activa(32 + 64, 64, 4, 2, 1, transpose=True)
        self.block_5_conv_2 = Conv2D_BN_activa(64, 1, 5, 1, 2, activation=None, if_bn=False)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, backbone_features):
        x_bb_1, x_bb_2, x_bb_3, x_bb_4 = backbone_features

        x = self.block_1_conv_1(x_bb_4)
        x = self.block_1_conv_2(x)
        x = self.block_1_conv_3(x)
        x_to_be_enhanced_1 = self.block_1_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_1, x_bb_4], dim=1)

        x = self.block_2_conv(x)
        x_to_be_enhanced_2 = self.block_2_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_2, x_bb_3], dim=1)

        x = self.block_3_conv(x)
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

        self.att1 = Attention_block(768, 512, 640)
        self.att2 = Attention_block(384, 256, 320)
        self.att3 = Attention_block(192, 128, 160)
        self.att4 = Attention_block(96, 64, 80)

        self.output_conv1 = Conv2D_BN_activa(16 * 3 + 1, 16, 3, 1, 1)
        self.output_conv2 = Conv2D_BN_activa(16, 1, 3, 1, 1)

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
        if x_bb_3.shape[-2:] != x.shape[-2:]:
            x_bb_3 = F.interpolate(x_bb_3, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if x_enhanced_2.shape[-2:] != x.shape[-2:]:
            x_enhanced_2 = F.interpolate(x_enhanced_2, size=x.shape[-2:], mode="bilinear", align_corners=False)
        bridge = self.att2(x, x_bb_3)
        x = torch.cat([x, bridge, x_enhanced_2], dim=1)

        x = self.feature_extraction_B(x)
        x = self.transposed_block_3(x)
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
        est_b = self.output_conv2(x)
        return est_b


class EnhancementLayers(nn.Module):
    def __init__(self):
        super(EnhancementLayers, self).__init__()
        self.enhancement_layers_1 = Conv2D_BN_activa(256, 128, 7, 1, 3)
        self.enhancement_layers_2 = Conv2D_BN_activa(128, 64, 7, 1, 3)
        self.enhancement_layers_3 = Conv2D_BN_activa(64, 32, 7, 1, 3)
        self.enhancement_layers_4 = Conv2D_BN_activa(32, 16, 7, 1, 3)

    def forward(self, to_be_enhanced_features):
        x1 = self.enhancement_layers_1(to_be_enhanced_features[0])
        x2 = self.enhancement_layers_2(to_be_enhanced_features[1])
        x3 = self.enhancement_layers_3(to_be_enhanced_features[2])
        x4 = self.enhancement_layers_4(to_be_enhanced_features[3])
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
        g_features = self.gradient_decoder(c_features[:-1])
        enhanced_features = self.enhancement_layers(g_features[:-1])
        est_b = self.main_decoder(c_features, enhanced_features, g_features[-1])
        return {"r": x - est_b, "g": g_features[-1], "b": est_b}


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
        gt_gradient = (gx + gy) / 2.0

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