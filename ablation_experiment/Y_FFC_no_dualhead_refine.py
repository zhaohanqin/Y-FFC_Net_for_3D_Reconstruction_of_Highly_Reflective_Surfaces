# -*- coding: utf-8 -*-
"""
Y_FFC_no_dualhead_refine.py — 消融实验 D：去除双头输出 + 残差精修头
====================================================================

【消融对象】
    V3.2 输出端核心设计，共三件事：
      1. MainDecoder 的直接漫反射头 r_direct
         （output_r_direct_conv / output_r_direct_sigmoid）
      2. 可学习双路融合系数 alpha（dual_head_alpha_logit）
         及融合式 r_fused = α·r_direct + (1−α)·clamp(x−b)
      3. ResidualRefinementHead 残差精修头

    消融后输出端退化为旧式纯减法链：
        r = clamp(x − est_b, 0, 1)

【检验假设】
    纯减法链对 est_b 的小幅误差极度敏感：
      - est_b 高估 → x−b < 0 被 clamp 为 0，暗区条纹信息永久丢失（不可逆）
      - est_b 低估 → x−b 仍含高光残留，且无任何后续修正通路
    若完整模型显著占优，即证明"双头 + 可学习融合 + 残差精修"
    是去高光质量与训练稳健性的关键来源，而非可有可无的附件。

【保留对象（关键：保证单变量）】
    双分支编码器 / 全部 FFC / 梯度辅助臂（g、s、enh1..4、三路先验门控
    全部保留，主解码器特征干逐行不变）/ SpecularAwareAttention /
    Attention_block / 扩展输出头（output_reduce/dilated/skip/conv3）/
    跨尺度一致性 cs。

【损失侧说明（train_no_dualhead_refine.py 无需改调度器）】
    - 输出字典不含 "r_direct"，criterion 收到 pred_r_direct=None，
      l_r_dual_cons 自动为 0 —— 其结构前提消失，属于消融本身；
    - l_r_focal_hl / l_r_smooth 直接监督 r 本身，与 r 的形成方式无关，
      原样保留，保证损失函数与 baseline 一致。

【可选二级实验（用于进一步拆分贡献，本文件不实现）】
    D-1 只删 ResidualRefinementHead（保留双头融合）：
        YFFC.forward 中 pred_diffuse = r_fused 即可；
    D-2 只删双头（保留精修头）：
        用本文件的 MainDecoderSingleHead + 原 refinement_head(r_sub, x)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Y_FFC_unified import (
    Encoder,
    GradientDecoder,
    EnhancementLayers,
    MainDecoder,
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


class MainDecoderSingleHead(MainDecoder):
    """单头版主解码器：删除 r_direct 头与 alpha 融合系数，仅输出 est_b。

    继承 MainDecoder 复用全部构件（FFC / sa1~4 / att1~4 / 三路先验门控 /
    enh 拼接 / 扩展输出头），__init__ 中删除双头相关参数；
    forward 与父类逐行一致，仅末尾改为只返回 est_b。
    """

    def __init__(self, in_channels=256):
        super().__init__(in_channels)
        # 删除双头相关模块与参数（nn.Module.__delattr__ 会同步清理注册表）
        del self.output_r_direct_conv
        del self.output_r_direct_sigmoid
        del self.dual_head_alpha_logit

    def forward(self, c_features, enhanced_features, x_g, x_s, original_input):
        # ── 与 Y_FFC_unified.MainDecoder.forward 逐行一致，仅末尾输出不同 ──
        x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c = c_features
        xe1, xe2, xe3, xe4 = enhanced_features

        x0, x1 = x_c.chunk(2, dim=1)
        x = self.FFC1((x0, x1))
        x = self.FFC2(x)
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

        x = self.output_reduce(x)
        skip = self.output_skip(x)
        x = self.output_dilated(x)
        x = x + skip
        x = self.output_conv3(x)

        # ★ 消融点：仅单一镜面头，无 r_direct / alpha
        est_b = self.output_sigmoid(self.output_conv4(x))
        return est_b


class YFFC_NoDualRefine(nn.Module):
    """无双头 + 无精修头版 YFFC：输出端退化为纯减法链 r = clamp(x − b, 0, 1)。

    编码器 / 梯度辅助臂 / 增强层 / 主解码器特征干与完整 YFFC 逐行一致，
    唯一差异是输出端结构（消融对象）。
    """

    def __init__(self, swin_pretrained=True, debug_shapes=False):
        super().__init__()
        self.debug_shapes = debug_shapes
        self.encoder = Encoder(swin_pretrained=swin_pretrained,
                               debug_shapes=debug_shapes)
        self.gradient_decoder = GradientDecoder()
        self.enhancement_layers = EnhancementLayers()
        self.main_decoder = MainDecoderSingleHead(in_channels=256)
        # ★ 消融点：无 ResidualRefinementHead

    def forward(self, x):
        c_feat = self.encoder(x)
        g_feat = self.gradient_decoder(c_feat[:-1], c_feat[-1], original_input=x)
        xe_feats = g_feat[:-2]
        g_raw = g_feat[-2]
        s_raw = g_feat[-1]
        enh = self.enhancement_layers(
            xe_feats, g_map=g_raw, original_input=x, s_map=s_raw)

        est_b = self.main_decoder(c_feat, enh, g_raw, s_raw, original_input=x)

        if est_b.shape[-2:] != x.shape[-2:]:
            est_b = F.interpolate(est_b, size=x.shape[-2:], mode='bilinear', align_corners=False)
        est_b = _safe_tensor(est_b, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))

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

        # ★ 消融点：纯减法链，无双路融合、无残差精修
        pred_diffuse = torch.clamp(x - est_b, 0.0, 1.0)
        pred_diffuse = _safe_tensor(pred_diffuse, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))

        cs_loss = cross_scale_consistency_loss((c_feat[1], c_feat[2], c_feat[3]))
        cs_loss = _safe_tensor(cs_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "r": pred_diffuse,
            "g": g_map,
            "b": est_b,
            "s": s_map,
            # 不返回 "r_direct"/"alpha" → criterion 中 l_r_dual_cons 自动为 0
            "cs": cs_loss,
        }


def build_model(swin_pretrained=True):
    return YFFC_NoDualRefine(swin_pretrained=swin_pretrained)


if __name__ == "__main__":
    print("=" * 60)
    print("Y_FFC_no_dualhead_refine 前向自测")
    print("=" * 60)
    model = YFFC_NoDualRefine(swin_pretrained=False)
    assert not hasattr(model, "refinement_head")
    assert not hasattr(model.main_decoder, "output_r_direct_conv")
    assert "dual_head_alpha_logit" not in dict(model.main_decoder.named_parameters())
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    print(f"前向成功: r={tuple(out['r'].shape)}, b={tuple(out['b'].shape)}, "
          f"g={tuple(out['g'].shape)}, s={tuple(out['s'].shape)}, "
          f"'r_direct' in out = {'r_direct' in out}")
    # 数值自检：r 必须严格等于 clamp(x - b)
    assert torch.allclose(out['r'], torch.clamp(x - out['b'], 0.0, 1.0))
    print("纯减法链数值校验通过: r == clamp(x - b, 0, 1)")
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total:,}")
