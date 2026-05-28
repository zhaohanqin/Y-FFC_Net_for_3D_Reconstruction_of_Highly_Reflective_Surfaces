"""Swin Transformer encoder branch used by the X-FFC-Swin encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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
                mode="bilinear", align_corners=False)

        swin_proj = self.swin_proj(swin_feat)
        gate = self.gate_net(torch.cat([res_feat, swin_proj], dim=1))
        self.last_gate_mean = gate.detach().mean()
        return res_feat + self.beta * gate * swin_proj
