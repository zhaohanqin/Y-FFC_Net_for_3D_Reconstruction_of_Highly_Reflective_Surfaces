"""Top-level YFFC model wrapper."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoders import GradientDecoder, MainDecoder
from .enhancement import EnhancementLayers
from ..common import _safe_tensor
from ..heads import ResidualRefinementHead


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

class YFFC(nn.Module):
    """Y-FFC 主网络（V3.2 优化版）。含 P1 残差精修头 + [D1] 双头漫反射预测。

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
