# -*- coding: utf-8 -*-
"""
X_FFC_single_encoder.py — 消融实验 C：去除 Swin 分支（X 型编码 → 单编码器）
==========================================================================

【消融对象】
    编码侧 X 结构：Swin-Tiny 窗口上下文分支及其全部三处融合模块：
      1. SwinEncoderBranch（H/4 / H/8 / H/16 三级上下文特征）
      2. fuse_h4 / fuse_h8（AdaptiveSkipFusion，H/4、H/8 skip 注入）
      3. swin16_proj + CrossModalFusion（H/16 双向交叉注意力瓶颈融合）

【保留对象（关键：保证单变量）】
    ResNet50 局部纹理分支（含空洞卷积 layer4、灰度初始化）/
    CrossScaleTopDownRefinement（只作用于 ResNet 特征，与 Swin 无关，
        保留以隔离变量；其 H/16 输入由"融合特征"改为 x_bb_4 本身）/
    cba_after_backbone / FFC 瓶颈 / HighFrequencyEnhancement /
    解码侧全部结构（GradientDecoder / EnhancementLayers / MainDecoder /
        双头输出 / ResidualRefinementHead）。

【接口兼容】
    EncoderSingleBranch 输出 5 级特征的通道与分辨率与原 Encoder 完全一致
    (x_bb_1: 64, x_bb_2: 128, x_bb_3: 256, x_bb_4: 512, x_c: 256)，
    因此两条解码臂一行代码都不需要改。
    优化器分组中 Swin 相关参数组自然为空，build_optimizer_v2 会自动过滤。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from X_FFC_unified import (
    Conv2D_BN_activa,
    FFC_BN_ACT,
    HighFrequencyEnhancement,
    CrossScaleTopDownRefinement,
    GradientDecoder,
    EnhancementLayers,
    MainDecoder,
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


class EncoderSingleBranch(nn.Module):
    """单分支编码器：仅保留 ResNet50 局部纹理路径。

    与原 Encoder 的差异：删除 swin_branch / fuse_h4 / fuse_h8 /
    swin16_proj / cross_modal_fusion；top_down_refine 的 H/16 输入
    由 CrossModalFusion 输出改为 x_bb_4 本身。其余逐行一致。
    """

    def __init__(self, debug_shapes=False):
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

        # 保留 top-down 回流（仅作用于 ResNet 特征，与 Swin 无关）
        self.top_down_refine = CrossScaleTopDownRefinement(
            h4_channels=128, h8_channels=256, h16_channels=512, init_gamma=0.05)
        self.cba_after_backbone = Conv2D_BN_activa(512, 256, 3, 1, 1)
        self.ffc_bottleneck = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)
        self.hf_enhance = HighFrequencyEnhancement(channels=256)

    def forward(self, x):
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

        # 无 Swin 融合：top-down 回流直接以 x_bb_4 作为 H/16 输入
        x_bb_2, x_bb_3, x_bb_4 = self.top_down_refine(x_bb_2, x_bb_3, x_bb_4)
        x_c = self.cba_after_backbone(x_bb_4)

        nl = 256 - int(256 * 0.5)
        out = self.ffc_bottleneck((x_c[:, :nl], x_c[:, nl:]))
        x_c = torch.cat([out[0], out[1]], 1) + x_c
        x_c = self.hf_enhance(x_c)

        return x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c


class YFFC_SingleEncoder(nn.Module):
    """单编码器版 XFFC：仅替换编码器，解码侧与融合/精修路径与 XFFC 逐行一致。

    swin_pretrained 参数保留以兼容 `model = XFFC()` 的调用签名，实际被忽略。
    """

    def __init__(self, swin_pretrained=True, debug_shapes=False):
        super().__init__()
        self.debug_shapes = debug_shapes
        self.encoder = EncoderSingleBranch(debug_shapes=debug_shapes)
        self.gradient_decoder = GradientDecoder()
        self.enhancement_layers = EnhancementLayers()
        self.main_decoder = MainDecoder(in_channels=256)
        self.refinement_head = ResidualRefinementHead(scale=0.15)

    def forward(self, x):
        # ── 与 X_FFC_unified.XFFC.forward 完全一致 ──
        c_feat = self.encoder(x)
        g_feat = self.gradient_decoder(c_feat[:-1], c_feat[-1], original_input=x)
        xe_feats = g_feat[:-2]
        g_raw = g_feat[-2]
        s_raw = g_feat[-1]
        enh = self.enhancement_layers(
            xe_feats, g_map=g_raw, original_input=x, s_map=s_raw)
        decoder_out = self.main_decoder(c_feat, enh, g_raw, s_raw, original_input=x)
        if isinstance(decoder_out, tuple) and len(decoder_out) == 3:
            est_b, r_direct, alpha_logit = decoder_out
        else:
            est_b = decoder_out
            r_direct = torch.clamp(x - est_b, 0.0, 1.0)
            alpha_logit = torch.tensor(0.0, device=est_b.device)

        if est_b.shape[-2:] != x.shape[-2:]:
            est_b = F.interpolate(est_b, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if r_direct.shape[-2:] != x.shape[-2:]:
            r_direct = F.interpolate(r_direct, size=x.shape[-2:], mode='bilinear', align_corners=False)
        est_b = _safe_tensor(est_b, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))
        r_direct = _safe_tensor(r_direct, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))

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

        r_sub = torch.clamp(x - est_b, 0.0, 1.0)
        alpha = torch.sigmoid(alpha_logit)
        r_fused = alpha * r_direct + (1.0 - alpha) * r_sub
        pred_diffuse = self.refinement_head(r_fused, x, r_direct)
        pred_diffuse = _safe_tensor(pred_diffuse, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))

        cs_loss = cross_scale_consistency_loss((c_feat[1], c_feat[2], c_feat[3]))
        cs_loss = _safe_tensor(cs_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "r": pred_diffuse,
            "g": g_map,
            "b": est_b,
            "s": s_map,
            "r_direct": r_direct,
            "alpha": alpha.detach() if alpha.dim() == 0 else alpha,
            "cs": cs_loss,
        }


def build_model(swin_pretrained=True):
    return YFFC_SingleEncoder(swin_pretrained=swin_pretrained)


if __name__ == "__main__":
    print("=" * 60)
    print("X_FFC_single_encoder 前向自测")
    print("=" * 60)
    model = YFFC_SingleEncoder()
    assert not hasattr(model.encoder, "swin_branch")
    assert not hasattr(model.encoder, "cross_modal_fusion")
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    print(f"前向成功: r={tuple(out['r'].shape)}, b={tuple(out['b'].shape)}, "
          f"g={tuple(out['g'].shape)}, s={tuple(out['s'].shape)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total:,}")
