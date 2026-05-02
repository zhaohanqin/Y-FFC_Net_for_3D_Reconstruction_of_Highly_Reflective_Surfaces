"""Top-level YFFC model wrapper."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoders import GradientDecoder, MainDecoder
from .enhancement import EnhancementLayers
from ..heads import ResidualRefinementHead


def cross_scale_consistency_loss(features):
    """约束 H/4、H/8、H/16 融合特征对结构区域的响应保持一致。"""
    maps = []
    for feat in features:
        response = feat.float().abs().mean(dim=1, keepdim=True)
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
    """Y-FFC 主网络（V3 优化版）。含 P1 残差精修头。"""

    def __init__(self, swin_pretrained=True, debug_shapes=False):
        super().__init__()
        self.debug_shapes = debug_shapes
        self.encoder = Encoder(swin_pretrained=swin_pretrained, debug_shapes=debug_shapes)
        self.gradient_decoder = GradientDecoder()
        self.enhancement_layers = EnhancementLayers()
        self.main_decoder = MainDecoder(in_channels=256)
        self.refinement_head = ResidualRefinementHead(scale=0.08)  # [P1]

    def forward(self, x):
        c_feat = self.encoder(x)
        g_feat = self.gradient_decoder(c_feat[:-1], c_feat[-1])
        enh = self.enhancement_layers(g_feat[:-1])
        est_b = self.main_decoder(c_feat, enh, g_feat[-1], original_input=x)
        if est_b.shape[-2:] != x.shape[-2:]:
            est_b = F.interpolate(est_b, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if g_feat[-1].shape[-2:] != x.shape[-2:]:
            g_map = F.interpolate(g_feat[-1], size=x.shape[-2:], mode='bilinear', align_corners=False)
        else:
            g_map = g_feat[-1]
        coarse = torch.clamp(x - est_b, 0.0, 1.0)
        pred_diffuse = self.refinement_head(coarse, x)  # [P1]
        cs_loss = cross_scale_consistency_loss((c_feat[1], c_feat[2], c_feat[3]))
        return {"r": pred_diffuse, "g": g_map, "b": est_b, "cs": cs_loss}
