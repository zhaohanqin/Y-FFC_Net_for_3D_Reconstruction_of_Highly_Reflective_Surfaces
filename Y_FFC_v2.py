"""
Y_FFC_v2.py — Y-FFC 改进版网络结构
===================================

在原始 Y_FFC.py 网络基础上，仅修改三个类：

  1. EncoderV2      — FFC bottleneck 后增加 HighFrequencyEnhancement
  2. MainDecoderV2  — 每级 skip connection 后增加 SpecularAwareAttention
  3. YFFC_V2        — 组装改进后的 Encoder/MainDecoder，传入原始输入

其余类（GradientDecoder, EnhancementLayers, Attention_block 等）
全部从原始 Y_FFC.py 导入，保持不变。

使用方法：
    from Y_FFC_v2 import YFFC_V2
    model = YFFC_V2()
    out = model(x)   # 输出格式与原版 YFFC 完全一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ── 从原始 Y_FFC.py 导入所有不变组件 ──
from Y_FFC import (
    Conv2D_BN_activa, Attention_block,
    MultiscaleConvBlock, FeatureExtrationLayersB,
    TransposedConvBlock, TransposedConvBlock_same,
    ViTEncoderBranch,
    GradientDecoder, EnhancementLayers,
    truncated_normal_,
)
from ffc import FFC_BN_ACT

# ── 从新增模块导入 ──
from specular_modules import SpecularAwareAttention, HighFrequencyEnhancement


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  EncoderV2 — 在 FFC bottleneck 后增加 HighFrequencyEnhancement      ║
# ╠════════════════════════════════════════════════════════════════════════╣
# ║  改动点（仅一处）：                                                    ║
# ║    self.hf_enhance = HighFrequencyEnhancement(256)                   ║
# ║    在 forward 末尾：x_c = self.hf_enhance(x_c)                      ║
# ║  其余代码与原版 Encoder 逐行一致。                                     ║
# ╚════════════════════════════════════════════════════════════════════════╝

class EncoderV2(nn.Module):
    def __init__(self, vit_image_size=224):
        super().__init__()

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
        self.stem_bn   = backbone.bn1
        self.stem_relu = backbone.relu
        self.stem_pool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.reduce_l1 = Conv2D_BN_activa(256,  128, 1, 1, 0)
        self.reduce_l2 = Conv2D_BN_activa(512,  256, 1, 1, 0)
        self.reduce_l3 = Conv2D_BN_activa(1024, 512, 1, 1, 0)
        self.reduce_l4 = Conv2D_BN_activa(2048, 512, 1, 1, 0)
        self.fuse_l34  = Conv2D_BN_activa(1024, 512, 3, 1, 1)

        self.vit_branch = ViTEncoderBranch(vit_image_size=vit_image_size)
        self.vit_resnet_fuse_1x1 = Conv2D_BN_activa(
            512 + self.vit_branch.embed_dim, 512, 1, 1, 0)
        self.cba_after_backbone = Conv2D_BN_activa(512, 256, 3, 1, 1)
        self.ffc_bottleneck = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)

        # 🆕 高频增强
        self.hf_enhance = HighFrequencyEnhancement(channels=256)

    def forward(self, x):
        x_in = x
        x = self.stem_conv(x); x = self.stem_bn(x)
        x_bb_1 = self.stem_relu(x)

        x = self.stem_pool(x_bb_1)
        x_l1 = self.layer1(x);  x_bb_2 = self.reduce_l1(x_l1)
        x_l2 = self.layer2(x_l1); x_bb_3 = self.reduce_l2(x_l2)
        x_l3 = self.layer3(x_l2); x_l4 = self.layer4(x_l3)

        x_l3_512 = self.reduce_l3(x_l3)
        x_l4_512 = self.reduce_l4(x_l4)
        x_bb_4 = self.fuse_l34(torch.cat([x_l3_512, x_l4_512], 1))

        vit_feat = self.vit_branch(x_in)
        if vit_feat.shape[-2:] != x_bb_4.shape[-2:]:
            vit_feat = F.interpolate(vit_feat, size=x_bb_4.shape[-2:],
                                      mode='bilinear', align_corners=False)

        fused = torch.cat([x_bb_4, vit_feat], 1)
        fused = self.vit_resnet_fuse_1x1(fused)
        x_c = self.cba_after_backbone(fused)

        nl = 256 - int(256 * 0.5)
        out = self.ffc_bottleneck((x_c[:, :nl], x_c[:, nl:]))
        x_c = torch.cat([out[0], out[1]], 1) + x_c

        # 🆕 高频增强
        x_c = self.hf_enhance(x_c)

        return x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  MainDecoderV2 — 每级增加 SpecularAwareAttention                    ║
# ╠════════════════════════════════════════════════════════════════════════╣
# ║  改动点：                                                            ║
# ║    每级 transposed_block 输出后调用 SpecularAwareAttention，          ║
# ║    接收 original_input 作为亮度先验。                                  ║
# ║    forward 签名新增参数 original_input。                               ║
# ╚════════════════════════════════════════════════════════════════════════╝

class MainDecoderV2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # 原版结构（完整复制）
        self.FFC1 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)
        self.FFC2 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)

        self.multiscale_conv = MultiscaleConvBlock(in_channels)
        self.transposed_block_1 = TransposedConvBlock_same(192*3, 256)
        self.transposed_block_2 = TransposedConvBlock(256*3+128+512, 128)
        self.feature_extraction_B = FeatureExtrationLayersB(128*3+64+256)
        self.transposed_block_3 = TransposedConvBlock(640, 64)
        self.transposed_block_4 = TransposedConvBlock(64*3+32+128, 32)
        self.transposed_block_5 = TransposedConvBlock(32*3+16+64, 16)

        self.att1 = Attention_block(768, 512, 64)
        self.att2 = Attention_block(384, 256, 64)
        self.att3 = Attention_block(192, 128, 32)
        self.att4 = Attention_block(96, 64, 16)

        self.ffc_h8 = FFC_BN_ACT(384, 384, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.ffc_h4 = FFC_BN_ACT(192, 192, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)

        self.output_conv1 = Conv2D_BN_activa(16*3+1, 16, 3, 1, 1)
        self.output_conv2 = Conv2D_BN_activa(16, 1, 3, 1, 1, activation=None, if_bn=False)
        self.output_sigmoid = nn.Sigmoid()

        # 🆕 SpecularAwareAttention（每级 transposed_block 输出通道数 = 3×branch）
        self.sa1 = SpecularAwareAttention(256*3, threshold=0.8)
        self.sa2 = SpecularAwareAttention(128*3, threshold=0.8)
        self.sa3 = SpecularAwareAttention(64*3,  threshold=0.8)
        self.sa4 = SpecularAwareAttention(32*3,  threshold=0.8)

    def forward(self, c_features, enhanced_features, x_g, original_input):
        """
        Args:
            c_features        : (x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c)
            enhanced_features : (x_enh_1, x_enh_2, x_enh_3, x_enh_4)
            x_g               : 梯度图 [B,1,H,W]
            original_input    : 🆕 原始输入 [B,1,H_in,W_in]
        Returns:
            est_b : 预测镜面分量 [B,1,H,W]
        """
        x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c = c_features
        xe1, xe2, xe3, xe4 = enhanced_features

        # FFC refinement
        x0, x1 = x_c.chunk(2, dim=1)
        x = self.FFC1((x0, x1)); x = self.FFC2(x)
        x = torch.cat([x[0], x[1]], 1) + x_c

        x = self.multiscale_conv(x)

        # Level 1
        x = self.transposed_block_1(x)
        x = self.sa1(x, original_input)                      # 🆕
        if x_bb_4.shape[-2:] != x.shape[-2:]:
            x_bb_4 = F.interpolate(x_bb_4, x.shape[-2:], mode='bilinear', align_corners=False)
        if xe1.shape[-2:] != x.shape[-2:]:
            xe1 = F.interpolate(xe1, x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att1(x, x_bb_4)
        x = torch.cat([x, bridge, xe1], 1)

        # Level 2
        x = self.transposed_block_2(x)
        nl = 384 - int(384*0.25)
        fo = self.ffc_h8((x[:,:nl], x[:,nl:])); x = torch.cat([fo[0],fo[1]],1) + x
        x = self.sa2(x, original_input)                      # 🆕
        if x_bb_3.shape[-2:] != x.shape[-2:]:
            x_bb_3 = F.interpolate(x_bb_3, x.shape[-2:], mode='bilinear', align_corners=False)
        if xe2.shape[-2:] != x.shape[-2:]:
            xe2 = F.interpolate(xe2, x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att2(x, x_bb_3)
        x = torch.cat([x, bridge, xe2], 1)

        # Level 3
        x = self.feature_extraction_B(x)
        x = self.transposed_block_3(x)
        nl4 = 192 - int(192*0.25)
        fo4 = self.ffc_h4((x[:,:nl4], x[:,nl4:])); x = torch.cat([fo4[0],fo4[1]],1) + x
        x = self.sa3(x, original_input)                      # 🆕
        if x_bb_2.shape[-2:] != x.shape[-2:]:
            x_bb_2 = F.interpolate(x_bb_2, x.shape[-2:], mode='bilinear', align_corners=False)
        if xe3.shape[-2:] != x.shape[-2:]:
            xe3 = F.interpolate(xe3, x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att3(x, x_bb_2)
        x = torch.cat([x, bridge, xe3], 1)

        # Level 4
        x = self.transposed_block_4(x)
        x = self.sa4(x, original_input)                      # 🆕
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

        x = self.output_conv1(x)
        return self.output_sigmoid(self.output_conv2(x))


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  YFFC_V2 — 改进版主网络                                               ║
# ╠════════════════════════════════════════════════════════════════════════╣
# ║  • EncoderV2  替换 Encoder  （+ HighFrequencyEnhancement）            ║
# ║  • MainDecoderV2 替换 MainDecoder（+ SpecularAwareAttention）         ║
# ║  • forward 将原始输入 x 传入 MainDecoderV2                            ║
# ║  • 输出格式与原版 YFFC 完全一致                                         ║
# ╚════════════════════════════════════════════════════════════════════════╝

class YFFC_V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder             = EncoderV2()
        self.gradient_decoder    = GradientDecoder()
        self.enhancement_layers  = EnhancementLayers()
        self.main_decoder        = MainDecoderV2(in_channels=256)

    def forward(self, x):
        c_feat = self.encoder(x)
        g_feat = self.gradient_decoder(c_feat[:-1], c_feat[-1])
        enh    = self.enhancement_layers(g_feat[:-1])
        est_b  = self.main_decoder(c_feat, enh, g_feat[-1],
                                   original_input=x)          # 🆕
        pred_diffuse = torch.clamp(x - est_b, 0.0, 1.0)
        return {"r": pred_diffuse, "g": g_feat[-1], "b": est_b}
