# -*- coding: utf-8 -*-
"""
Y_FFC_no_gradient.py — 消融实验 B：去除梯度辅助分支（X 型解码 → 单解码器 U 型）
==============================================================================

【消融对象】
    解码侧 X 结构的辅助臂及其向主臂的全部注入通路：
      1. GradientDecoder（梯度图 g 头 + 高反光概率图 s 头 + xe1..xe4 中间特征）
      2. EnhancementLayers（enh1..enh4 增强桥接特征）
      3. MainDecoder 中所有依赖 g / s / enh 的结构：
         - Level 1~4 与 enh 特征的通道拼接
             Level2 输入 1408 → 1280
             Level3 输入  704 →  640
             Level4 输入  352 →  320
             Level5 输入  176 →  160
         - Level 2/3/4 的三路先验调制（grad_gate / joint_gate / spec_lik_gate）
         - Level 5 输出头中与梯度图 g 的拼接（49ch → 48ch）

【保留对象（关键：保证单变量）】
    双分支编码器（含 Swin 融合）/ 全部 FFC 频域模块 /
    SpecularAwareAttention（sa1~sa4，亮度先验注意力）/
    Attention_block skip 门控 / 双头输出 (est_b, r_direct, alpha) /
    ResidualRefinementHead / 跨尺度一致性 cs。

【训练侧配套（见 train_no_gradient.py）】
    lambda_grad / lambda_grad_edge / lambda_grad_spec /
    lambda_s_lik / lambda_s_cons 置 0 —— 分支不存在则其监督必然移除，
    这是消融的一部分而非混杂变量。其余损失与权重调度保持不变。

【接口兼容】
    forward 返回的 "g" 为全零占位图（仅用于兼容训练循环与可视化代码，
    其相关损失权重为 0，不影响优化）；不返回 "s"，criterion 中
    s_lik / s_cons 项会自动为 0。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Y_FFC_unified import (
    FFC_BN_ACT,
    Conv2D_BN_activa,
    Attention_block,
    MultiscaleConvBlock,
    FeatureExtrationLayersB,
    TransposedConvBlock,
    TransposedConvBlock_same,
    SpecularAwareAttention,
    Encoder,
    ResidualRefinementHead,
    cross_scale_consistency_loss,
    _safe_tensor,
    # 供训练脚本统一导入
    build_scheduler,            # noqa: F401
    EMA,                        # noqa: F401
    replace_bn_with_gn,         # noqa: F401
    EnhancedCompositeLoss,      # noqa: F401
    EnhancedPairedTransform,    # noqa: F401
    HighlightAugmentation,      # noqa: F401
    compute_input_gradient_target,  # noqa: F401
)


class MainDecoderNoGrad(nn.Module):
    """无梯度先验版主解码器。

    与原 MainDecoder 完全同构，仅做三处删除：
      ① 各级与 enh1..enh4 的拼接（对应通道数同步缩减）；
      ② Level 2/3/4 的三路先验门控调制；
      ③ Level 5 与梯度图 g 的拼接（输出头输入 49 → 48 通道）。
    其余（FFC、SpecularAwareAttention、Attention_block、双头输出、
    扩展输出头 + 膨胀卷积 + skip）全部保持原样。
    """

    def __init__(self, in_channels=256):
        super().__init__()

        self.FFC1 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)
        self.FFC2 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)

        self.multiscale_conv = MultiscaleConvBlock(in_channels)          # → 576
        self.transposed_block_1 = TransposedConvBlock_same(192 * 3, 256)  # → 768
        # 原: 256*3 + 128(enh1) + 512 = 1408；消融后: 768 + 512 = 1280
        self.transposed_block_2 = TransposedConvBlock(256 * 3 + 512, 128)  # → 384
        # 原: 128*3 + 64(enh2) + 256 = 704；消融后: 384 + 256 = 640
        self.feature_extraction_B = FeatureExtrationLayersB(128 * 3 + 256)  # → 640
        self.transposed_block_3 = TransposedConvBlock(640, 64)             # → 192
        # 原: 64*3 + 32(enh3) + 128 = 352；消融后: 192 + 128 = 320
        self.transposed_block_4 = TransposedConvBlock(64 * 3 + 128, 32)    # → 96
        # 原: 32*3 + 16(enh4) + 64 = 176；消融后: 96 + 64 = 160
        self.transposed_block_5 = TransposedConvBlock(32 * 3 + 64, 16)     # → 48

        self.att1 = Attention_block(768, 512, 64)
        self.att2 = Attention_block(384, 256, 64)
        self.att3 = Attention_block(192, 128, 32)
        self.att4 = Attention_block(96, 64, 16)

        self.ffc_h8 = FFC_BN_ACT(384, 384, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.ffc_h4 = FFC_BN_ACT(192, 192, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)

        # 扩展输出头：原版输入 16*3 + 1(g) = 49，消融后 16*3 = 48
        _out_in = 16 * 3
        self.output_reduce = Conv2D_BN_activa(_out_in, 32, 1, 1, 0)
        self.output_dilated = Conv2D_BN_activa(32, 32, 3, 1, 2, dilation=2)
        self.output_conv3 = Conv2D_BN_activa(32, 16, 3, 1, 1)
        self.output_conv4 = Conv2D_BN_activa(16, 1, 3, 1, 1, activation=None, if_bn=False)
        self.output_sigmoid = nn.Sigmoid()
        self.output_skip = nn.Conv2d(32, 32, 1, 1, 0, bias=False)
        self.output_r_direct_conv = Conv2D_BN_activa(16, 1, 3, 1, 1, activation=None, if_bn=False)
        self.output_r_direct_sigmoid = nn.Sigmoid()

        self.sa1 = SpecularAwareAttention(256 * 3, threshold=0.8)
        self.sa2 = SpecularAwareAttention(128 * 3, threshold=0.8)
        self.sa3 = SpecularAwareAttention(64 * 3, threshold=0.8)
        self.sa4 = SpecularAwareAttention(32 * 3, threshold=0.8)

        # 双头融合系数（与原版一致）
        self.dual_head_alpha_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, c_features, original_input):
        x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c = c_features

        x0, x1 = x_c.chunk(2, dim=1)
        x = self.FFC1((x0, x1))
        x = self.FFC2(x)
        x = torch.cat([x[0], x[1]], 1) + x_c

        x = self.multiscale_conv(x)

        # Level 1（无 enh1 拼接）
        x = self.transposed_block_1(x)
        x = self.sa1(x, original_input)
        if x_bb_4.shape[-2:] != x.shape[-2:]:
            x_bb_4 = F.interpolate(x_bb_4, x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att1(x, x_bb_4)
        x = torch.cat([x, bridge], 1)                       # 1280

        # Level 2（无三路先验调制、无 enh2 拼接）
        x = self.transposed_block_2(x)                      # 384
        nl = 384 - int(384 * 0.25)
        fo = self.ffc_h8((x[:, :nl], x[:, nl:]))
        x = torch.cat([fo[0], fo[1]], 1) + x
        x = self.sa2(x, original_input)
        if x_bb_3.shape[-2:] != x.shape[-2:]:
            x_bb_3 = F.interpolate(x_bb_3, x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att2(x, x_bb_3)
        x = torch.cat([x, bridge], 1)                       # 640

        # Level 3
        x = self.feature_extraction_B(x)                    # 640
        x = self.transposed_block_3(x)                      # 192
        nl4 = 192 - int(192 * 0.25)
        fo4 = self.ffc_h4((x[:, :nl4], x[:, nl4:]))
        x = torch.cat([fo4[0], fo4[1]], 1) + x
        x = self.sa3(x, original_input)
        if x_bb_2.shape[-2:] != x.shape[-2:]:
            x_bb_2 = F.interpolate(x_bb_2, x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att3(x, x_bb_2)
        x = torch.cat([x, bridge], 1)                       # 320

        # Level 4
        x = self.transposed_block_4(x)                      # 96
        x = self.sa4(x, original_input)
        if x_bb_1.shape[-2:] != x.shape[-2:]:
            x_bb_1 = F.interpolate(x_bb_1, x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att4(x, x_bb_1)
        x = torch.cat([x, bridge], 1)                       # 160

        # Output（无梯度图 g 拼接）
        x = self.transposed_block_5(x)                      # 48
        x = self.output_reduce(x)
        skip = self.output_skip(x)
        x = self.output_dilated(x)
        x = x + skip
        x = self.output_conv3(x)
        est_b = self.output_sigmoid(self.output_conv4(x))
        r_direct = self.output_r_direct_sigmoid(self.output_r_direct_conv(x))
        return est_b, r_direct, self.dual_head_alpha_logit


class YFFC_NoGradient(nn.Module):
    """无梯度辅助分支版 YFFC：编码器 → 单主解码器 → 双路融合 → 残差精修。

    与 YFFC 的差异仅为：删除 gradient_decoder / enhancement_layers，
    主解码器换为 MainDecoderNoGrad。融合与精修路径逐行保持一致。
    """

    def __init__(self, swin_pretrained=True, debug_shapes=False):
        super().__init__()
        self.debug_shapes = debug_shapes
        self.encoder = Encoder(swin_pretrained=swin_pretrained,
                               debug_shapes=debug_shapes)
        self.main_decoder = MainDecoderNoGrad(in_channels=256)
        self.refinement_head = ResidualRefinementHead(scale=0.15)

    def forward(self, x):
        c_feat = self.encoder(x)

        est_b, r_direct, alpha_logit = self.main_decoder(c_feat, original_input=x)

        if est_b.shape[-2:] != x.shape[-2:]:
            est_b = F.interpolate(est_b, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if r_direct.shape[-2:] != x.shape[-2:]:
            r_direct = F.interpolate(r_direct, size=x.shape[-2:], mode='bilinear', align_corners=False)
        est_b = _safe_tensor(est_b, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))
        r_direct = _safe_tensor(r_direct, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))

        # 与 YFFC.forward 末尾完全一致的双路融合 + 残差精修
        r_sub = torch.clamp(x - est_b, 0.0, 1.0)
        alpha = torch.sigmoid(alpha_logit)
        r_fused = alpha * r_direct + (1.0 - alpha) * r_sub
        pred_diffuse = self.refinement_head(r_fused, x, r_direct)
        pred_diffuse = _safe_tensor(pred_diffuse, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))

        cs_loss = cross_scale_consistency_loss((c_feat[1], c_feat[2], c_feat[3]))
        cs_loss = _safe_tensor(cs_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "r": pred_diffuse,
            # 全零占位：兼容训练循环 / 可视化接口，相关损失权重已置 0
            "g": torch.zeros_like(x),
            "b": est_b,
            # 不返回 "s" → criterion 中 s_lik / s_cons 自动为 0
            "r_direct": r_direct,
            "alpha": alpha.detach() if alpha.dim() == 0 else alpha,
            "cs": cs_loss,
        }


def build_model(swin_pretrained=True):
    return YFFC_NoGradient(swin_pretrained=swin_pretrained)


if __name__ == "__main__":
    print("=" * 60)
    print("Y_FFC_no_gradient 前向自测")
    print("=" * 60)
    model = YFFC_NoGradient(swin_pretrained=False)
    assert not hasattr(model, "gradient_decoder")
    assert not hasattr(model, "enhancement_layers")
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    print(f"前向成功: r={tuple(out['r'].shape)}, b={tuple(out['b'].shape)}, "
          f"g(占位)={tuple(out['g'].shape)}, 's' in out = {'s' in out}")
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total:,}")
