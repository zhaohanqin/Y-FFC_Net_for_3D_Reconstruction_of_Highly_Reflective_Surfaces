"""Feature enhancement layers used between decoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv2D_BN_activa
from ..ffc import FFC_BN_ACT


class _SpectralDecompositionUnit(nn.Module):
    """梯度分解单元，仅用于 EnhancementLayers 内部。

    本单元的职责是：把"通用梯度特征"转化为"高反光相关增强特征"。

    门控设计（[NEW] 引入 s_map 后的 3 输入版本）：
    ─────────────────────────────────────────────────
        - edge_gate     ← g_map（输入梯度图）：定位"哪里有结构突变"
        - specular_gate ← concat(g_map, original_input, s_map)：
                          联合判别"哪些突变是高反光相关"
        - final_gate    = edge_gate * specular_gate
        - feature_out   = specular_refined * (1 + beta * final_gate)

    `s_map` 由 GradientDecoder 的高光概率辅助头提供（gt_specular 监督）。
    相比旧版仅靠"亮度+梯度"启发式估计，新版获得了显式监督下的高光语义。
    ─────────────────────────────────────────────────
    """

    def __init__(self, in_c: int, out_c: int, beta: float = 0.3):
        super(_SpectralDecompositionUnit, self).__init__()
        self.out_c = out_c
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.projection_branch = Conv2D_BN_activa(in_c, out_c, 7, 1, 3)
        self.specular_branch = Conv2D_BN_activa(in_c, out_c, 3, 1, 1)
        self.ffc_specular = FFC_BN_ACT(out_c, out_c, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        # edge_gate：1 通道（梯度图） → out_c 通道门控
        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, out_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )
        # specular_gate：[NEW] 3 通道输入（g_down + input_down + s_down）
        # 旧版只有 g_down + input_down，是靠 "亮+强梯度=高光" 的启发式；
        # 现在直接接入监督下的高光概率 s_down，使高光语义判别更可靠。
        self.specular_gate = nn.Sequential(
            nn.Conv2d(3, out_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self._init_prior_gates()

    def _init_prior_gates(self):
        # Start as a conservative high-reflection prior and learn stronger
        # responses only when useful for specular reconstruction.
        for gate in (self.edge_gate, self.specular_gate):
            conv = gate[0]
            nn.init.zeros_(conv.weight)
            nn.init.constant_(conv.bias, -1.0)

    @staticmethod
    def _prepare_prior(prior: torch.Tensor, size, device, dtype) -> torch.Tensor:
        if prior.shape[1] != 1:
            prior = prior.mean(dim=1, keepdim=True)
        if prior.shape[-2:] != size:
            prior = F.interpolate(prior, size=size, mode='bilinear', align_corners=False)
        return prior.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, g_map: torch.Tensor,
                original_input: torch.Tensor, s_map: torch.Tensor) -> torch.Tensor:
        projection = self.projection_branch(x)
        specular = self.specular_branch(x)

        _nl_spec = self.out_c - int(self.out_c * 0.25)
        _ffc_spec_out = self.ffc_specular((specular[:, :_nl_spec], specular[:, _nl_spec:]))
        specular = torch.cat([_ffc_spec_out[0], _ffc_spec_out[1]], dim=1) + specular

        gate = self.gate_conv(torch.cat([projection, specular], dim=1))
        specular_refined = specular + specular * gate

        size = specular_refined.shape[-2:]
        device, dtype = specular_refined.device, specular_refined.dtype
        g_down = self._prepare_prior(g_map, size, device, dtype)
        input_down = self._prepare_prior(original_input, size, device, dtype)
        s_down = self._prepare_prior(s_map, size, device, dtype)
        edge_gate = self.edge_gate(g_down)
        # 3 输入：梯度 + 亮度 + 高光概率（显式监督先验）
        spec_gate = self.specular_gate(torch.cat([g_down, input_down, s_down], dim=1))
        final_gate = edge_gate * spec_gate
        return specular_refined + self.beta.clamp(0.0, 1.0) * specular_refined * final_gate

class EnhancementLayers(nn.Module):
    """增强层：把梯度分支的多尺度特征转化为"高反光相关增强特征"。

    [改造] 接口由 (features, g_map, original_input) 升级为
            (features, g_map, original_input, s_map)。
    `s_map` 即 GradientDecoder 输出的高光软概率图（gt_specular 监督），
    会进入每一级 _SpectralDecompositionUnit 的 specular_gate 联合判别。
    """

    def __init__(self):
        super(EnhancementLayers, self).__init__()
        self.decomp_unit_1 = _SpectralDecompositionUnit(in_c=256, out_c=128)
        self.decomp_unit_2 = _SpectralDecompositionUnit(in_c=128, out_c=64)
        self.decomp_unit_3 = _SpectralDecompositionUnit(in_c=64, out_c=32)
        self.decomp_unit_4 = _SpectralDecompositionUnit(in_c=32, out_c=16)

    def forward(self, to_be_enhanced_features, g_map, original_input, s_map):
        x1 = self.decomp_unit_1(to_be_enhanced_features[0], g_map, original_input, s_map)
        x2 = self.decomp_unit_2(to_be_enhanced_features[1], g_map, original_input, s_map)
        x3 = self.decomp_unit_3(to_be_enhanced_features[2], g_map, original_input, s_map)
        x4 = self.decomp_unit_4(to_be_enhanced_features[3], g_map, original_input, s_map)
        return x1, x2, x3, x4
