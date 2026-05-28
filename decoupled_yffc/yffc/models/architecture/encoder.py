"""Y-FFC encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..attention import CrossModalFusion
from ..blocks import Conv2D_BN_activa
from ..branches import AdaptiveSkipFusion, SwinEncoderBranch
from ..ffc import FFC_BN_ACT
from ..heads import HighFrequencyEnhancement


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

        # H/4 和 H/8 使用 AdaptiveSkipFusion，根据内容、位置和通道动态控制 Swin 注入比例。
        # beta 初始为 0，模型初始行为接近纯 ResNet skip，训练中再逐步学习 Swin skip 的整体强度。
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

        # H/16 仍使用 CrossModalFusion，不额外 residual add，避免重复注入上下文。
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

        # H/16 瓶颈融合：投影到 512 后与 ResNet x_bb_4 做 CrossModalFusion。
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
