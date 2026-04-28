"""Top-level YFFC model wrapper."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoders import GradientDecoder, MainDecoder
from .enhancement import EnhancementLayers
from ..heads import ResidualRefinementHead


class YFFC(nn.Module):
    """Y-FFC 主网络（V3 优化版）。含 P1 残差精修头。"""

    def __init__(self, swin_pretrained=True, debug_shapes=False):
        super().__init__()
        self.debug_shapes = debug_shapes
        self.encoder = Encoder(swin_pretrained=swin_pretrained, debug_shapes=debug_shapes)
        self.gradient_decoder = GradientDecoder()
        self.enhancement_layers = EnhancementLayers()
        self.main_decoder = MainDecoder(in_channels=256)
        self.refinement_head = ResidualRefinementHead(scale=0.05)  # [P1]

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
        return {"r": pred_diffuse, "g": g_map, "b": est_b}
