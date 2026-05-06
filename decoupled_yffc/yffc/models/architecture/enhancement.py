"""Feature enhancement layers used between decoders."""

import torch
import torch.nn as nn

from ..blocks import Conv2D_BN_activa
from ..ffc import FFC_BN_ACT


class _SpectralDecompositionUnit(nn.Module):
    """梯度分解单元，仅用于 EnhancementLayers 内部。"""

    def __init__(self, in_c: int, out_c: int):
        super(_SpectralDecompositionUnit, self).__init__()
        self.out_c = out_c
        self.projection_branch = Conv2D_BN_activa(in_c, out_c, 7, 1, 3)
        self.specular_branch = Conv2D_BN_activa(in_c, out_c, 3, 1, 1)
        self.ffc_specular = FFC_BN_ACT(out_c, out_c, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = self.projection_branch(x)
        specular = self.specular_branch(x)

        _nl_spec = self.out_c - int(self.out_c * 0.25)
        _ffc_spec_out = self.ffc_specular((specular[:, :_nl_spec], specular[:, _nl_spec:]))
        specular = torch.cat([_ffc_spec_out[0], _ffc_spec_out[1]], dim=1) + specular

        gate = self.gate_conv(torch.cat([projection, specular], dim=1))
        specular_refined = specular + specular * gate
        return specular_refined


class EnhancementLayers(nn.Module):
    """增强层：对 GradientDecoder 输出的 4 级特征做频域精炼。"""

    def __init__(self):
        super(EnhancementLayers, self).__init__()
        self.decomp_unit_1 = _SpectralDecompositionUnit(in_c=256, out_c=128)
        self.decomp_unit_2 = _SpectralDecompositionUnit(in_c=128, out_c=64)
        self.decomp_unit_3 = _SpectralDecompositionUnit(in_c=64, out_c=32)
        self.decomp_unit_4 = _SpectralDecompositionUnit(in_c=32, out_c=16)

    def forward(self, to_be_enhanced_features):
        x1 = self.decomp_unit_1(to_be_enhanced_features[0])
        x2 = self.decomp_unit_2(to_be_enhanced_features[1])
        x3 = self.decomp_unit_3(to_be_enhanced_features[2])
        x4 = self.decomp_unit_4(to_be_enhanced_features[3])
        return x1, x2, x3, x4
