"""Gradient and main decoders for Y-FFC."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention import Attention_block, SpecularAwareAttention
from ..blocks import (
    Conv2D_BN_activa,
    FeatureExtrationLayersB,
    MultiscaleConvBlock,
    TransposedConvBlock,
    TransposedConvBlock_same,
)
from ..ffc import FFC_BN_ACT


class GradientDecoder(nn.Module):
    """
    梯度解码器：接收编码器特征，输出
        - 梯度图 x_g（监督目标 = Sobel(I)，输入图像梯度幅值）
        - [新增] 高反光软概率图 x_s（监督目标 = gt_specular，[0, 1] 软标签）
        - 4 级待增强特征 xe1, xe2, xe3, xe4

    任务化辅助头设计动机：
    ─────────────────────────────────────────────────
    梯度分支的监督已经从 `Sobel(Id)` 改为 `Sobel(I)`，但纯梯度语义会包含大量
    "非高光"边缘（条纹边、物体轮廓、噪声）。仅靠 `pred_grad ≈ Sobel(I)` 这一
    监督，无法保证梯度分支学到的中间特征对"高反光分量重建"任务有效。

    解决方案：在 GradientDecoder 尾部并列一个 `specular_likelihood` 辅助头，
    与梯度头共享前面所有特征，但用 `gt_specular` 直接监督。这强制梯度分支
    的深层特征同时编码两种语义：
        - "哪里是边界"（梯度头）
        - "哪里是高光"（高光概率头）

    这个 `x_s` 还会被馈送到 EnhancementLayers 的 `specular_gate` 和
    MainDecoder 的 `joint_gate/spec_lik_gate`，作为显式监督下的高光先验，
    替代原来仅依赖"亮度+梯度"启发式的隐式估计。
    ─────────────────────────────────────────────────
    """

    def __init__(self):
        super(GradientDecoder, self).__init__()
        self.fuse_global_context = Conv2D_BN_activa(512 + 256, 512, 1, 1, 0)

        self.block_1_conv_1 = Conv2D_BN_activa(512, 1024, 7, 1, 3)
        self.block_1_conv_2 = Conv2D_BN_activa(1024, 512, 1, 1, 0)
        self.block_1_conv_3 = Conv2D_BN_activa(512, 256, 3, 1, 1)
        self.ffc_grad_h16 = FFC_BN_ACT(256, 256, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.block_1_conv_transpose = Conv2D_BN_activa(256, 256, 5, 1, 2, transpose=False)

        self.block_2_conv = Conv2D_BN_activa(256 + 512, 128, 3, 1, 1)
        self.ffc_grad_h8 = FFC_BN_ACT(128, 128, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.block_2_conv_transpose = Conv2D_BN_activa(128, 128, 4, 2, 1, transpose=True)

        self.block_3_conv = Conv2D_BN_activa(128 + 256, 64, 3, 1, 1)
        self.ffc_grad_h4 = FFC_BN_ACT(64, 64, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.block_3_conv_transpose = Conv2D_BN_activa(64, 64, 4, 2, 1, transpose=True)

        self.block_4_conv = Conv2D_BN_activa(64 + 128, 32, 3, 1, 1)
        self.block_4_conv_transpose = Conv2D_BN_activa(32, 32, 4, 2, 1, transpose=True)

        self.block_5_conv_1 = Conv2D_BN_activa(32 + 64, 64, 4, 2, 1, transpose=True)
        # 梯度头（保留原结构）：输入 = 64ch 上采特征 + 1ch 原图 = 65ch
        self.block_5_conv_2 = Conv2D_BN_activa(65, 1, 5, 1, 2, activation=None, if_bn=False)
        self.sigmoid_layer = nn.Sigmoid()

        # ── [NEW] 高反光软概率辅助头（与梯度头并列）──
        # 输入：64ch 上采特征 + 1ch 原图 = 65ch；输出：1ch 全分辨率高光概率图 [0, 1]
        # 该头比梯度头稍微深一些，因为"是否高光"比"是否边缘"判别难度更高，
        # 需要同时利用亮度、边界形态与高频纹理。
        self.spec_lik_conv_a = Conv2D_BN_activa(65, 32, 3, 1, 1)
        self.spec_lik_conv_b = Conv2D_BN_activa(32, 16, 3, 1, 1)
        self.spec_lik_conv_c = Conv2D_BN_activa(16, 1, 5, 1, 2, activation=None, if_bn=False)
        self.spec_lik_sigmoid = nn.Sigmoid()
        self._init_spec_lik_head()

    def _init_spec_lik_head(self):
        """让 spec_lik 头初始预测 ≈ 0（即"无高光"），避免训练初期向下游注入噪声。

        通过将最后一个 conv 的权重清零、bias 设为 -2.0，使 pre-sigmoid 输出 ≈ -2，
        sigmoid(-2) ≈ 0.12。随着 `l_s_lik` 监督的引导，模型会逐步把高光区域抬起。
        """
        last = self.spec_lik_conv_c.conv2d
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            # ★[FIX-C 问题2] bias -2.0→-1.0：sigmoid(-1)≈0.27，初始预测略高，
            #   配合二值监督更快把高光区抬过 0.5。
            nn.init.constant_(last.bias, -1.0)

    def forward(self, backbone_features, x_c, original_input=None):
        x_bb_1, x_bb_2, x_bb_3, x_bb_4 = backbone_features

        if x_c.shape[-2:] != x_bb_4.shape[-2:]:
            x_c = F.interpolate(x_c, size=x_bb_4.shape[-2:],
                                mode="bilinear", align_corners=False)
        x_bb_4_fused = self.fuse_global_context(torch.cat([x_bb_4, x_c], dim=1))

        x = self.block_1_conv_1(x_bb_4_fused)
        x = self.block_1_conv_2(x)
        x = self.block_1_conv_3(x)

        _nl_h16 = 256 - int(256 * 0.25)
        _ffc_h16_out = self.ffc_grad_h16((x[:, :_nl_h16], x[:, _nl_h16:]))
        x = torch.cat([_ffc_h16_out[0], _ffc_h16_out[1]], dim=1) + x

        x_to_be_enhanced_1 = self.block_1_conv_transpose(x)
        if x_bb_4.shape[-2:] != x_to_be_enhanced_1.shape[-2:]:
            x_bb_4 = F.interpolate(
                x_bb_4, size=x_to_be_enhanced_1.shape[-2:],
                mode='bilinear', align_corners=False)
        x = torch.cat([x_to_be_enhanced_1, x_bb_4], dim=1)

        x = self.block_2_conv(x)
        _nl_h8 = 128 - int(128 * 0.25)
        _ffc_h8_out = self.ffc_grad_h8((x[:, :_nl_h8], x[:, _nl_h8:]))
        x = torch.cat([_ffc_h8_out[0], _ffc_h8_out[1]], dim=1) + x

        x_to_be_enhanced_2 = self.block_2_conv_transpose(x)
        if x_bb_3.shape[-2:] != x_to_be_enhanced_2.shape[-2:]:
            x_bb_3 = F.interpolate(
                x_bb_3, size=x_to_be_enhanced_2.shape[-2:],
                mode='bilinear', align_corners=False)
        x = torch.cat([x_to_be_enhanced_2, x_bb_3], dim=1)

        x = self.block_3_conv(x)
        _nl_h4 = 64 - int(64 * 0.25)
        _ffc_h4_out = self.ffc_grad_h4((x[:, :_nl_h4], x[:, _nl_h4:]))
        x = torch.cat([_ffc_h4_out[0], _ffc_h4_out[1]], dim=1) + x

        x_to_be_enhanced_3 = self.block_3_conv_transpose(x)
        if x_bb_2.shape[-2:] != x_to_be_enhanced_3.shape[-2:]:
            x_bb_2 = F.interpolate(
                x_bb_2, size=x_to_be_enhanced_3.shape[-2:],
                mode='bilinear', align_corners=False)
        x = torch.cat([x_to_be_enhanced_3, x_bb_2], dim=1)

        x = self.block_4_conv(x)
        x_to_be_enhanced_4 = self.block_4_conv_transpose(x)
        if x_bb_1.shape[-2:] != x_to_be_enhanced_4.shape[-2:]:
            x_bb_1 = F.interpolate(
                x_bb_1, size=x_to_be_enhanced_4.shape[-2:],
                mode='bilinear', align_corners=False)
        x = torch.cat([x_to_be_enhanced_4, x_bb_1], dim=1)

        x = self.block_5_conv_1(x)
        if original_input is None:
            original_input = torch.zeros(
                x.size(0), 1, x.size(2), x.size(3),
                device=x.device, dtype=x.dtype)
        else:
            if original_input.shape[-2:] != x.shape[-2:]:
                original_input = F.interpolate(
                    original_input, size=x.shape[-2:],
                    mode='bilinear', align_corners=False)
            original_input = original_input.to(device=x.device, dtype=x.dtype)
        # 梯度头与 spec_likelihood 头共享 [block_5_conv_1 特征 + 原图] 输入
        x_concat = torch.cat([x, original_input], dim=1)

        # 梯度头：预测 Sobel(I)（即输入图像梯度幅值）
        x_g_logits = self.block_5_conv_2(x_concat)
        x_g = self.sigmoid_layer(x_g_logits)

        # [NEW] 高光概率头：预测 gt_specular 的软概率图
        x_s_feat = self.spec_lik_conv_a(x_concat)
        x_s_feat = self.spec_lik_conv_b(x_s_feat)
        x_s_logits = self.spec_lik_conv_c(x_s_feat)
        x_s = self.spec_lik_sigmoid(x_s_logits)

        return x_to_be_enhanced_1, x_to_be_enhanced_2, x_to_be_enhanced_3, x_to_be_enhanced_4, x_g, x_s

class MainDecoder(nn.Module):
    """
    主解码器：融合编码器 skip connection + 梯度分支增强特征 + 梯度图，
    逐级上采样并最终输出镜面分量 est_b。
    每级 transposed_block 输出后使用 SpecularAwareAttention，
    利用原始输入亮度作为先验增强高光区域响应。
    """

    def __init__(self, in_channels):
        super().__init__()

        self.FFC1 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)
        self.FFC2 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)

        self.multiscale_conv = MultiscaleConvBlock(in_channels)
        self.transposed_block_1 = TransposedConvBlock_same(192 * 3, 256)
        self.transposed_block_2 = TransposedConvBlock(256 * 3 + 128 + 512, 128)
        self.feature_extraction_B = FeatureExtrationLayersB(128 * 3 + 64 + 256)
        self.transposed_block_3 = TransposedConvBlock(640, 64)
        self.transposed_block_4 = TransposedConvBlock(64 * 3 + 32 + 128, 32)
        self.transposed_block_5 = TransposedConvBlock(32 * 3 + 16 + 64, 16)

        self.att1 = Attention_block(768, 512, 64)
        self.att2 = Attention_block(384, 256, 64)
        self.att3 = Attention_block(192, 128, 32)
        self.att4 = Attention_block(96, 64, 16)

        self.ffc_h8 = FFC_BN_ACT(384, 384, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)
        self.ffc_h4 = FFC_BN_ACT(192, 192, 3, ratio_gin=0.25, ratio_gout=0.25, padding=1)

        # [P2] 扩展输出头：4 层 + 膨胀卷积 + skip connection
        #
        # ★[D1] 双头升级（V3.2 关键改进）★
        # 旧版只输出 est_b（镜面分量），通过 r = clamp(x - est_b, 0, 1) 减法链得到漫反射。
        # 该链条对 est_b 的小幅误差极敏感：
        #   - est_b 高估 → x-b < 0 被 clamp 为 0，暗区信息永久丢失（不可逆）
        #   - est_b 低估 → x-b 仍含高光残留，refinement_head ±0.08 不足以补救
        # 双头方案：增加 r_direct_head，让网络直接预测漫反射；
        # 最终 r 由可学习 alpha 融合 [direct, subtraction_path]，更稳健。
        _out_in = 16 * 3 + 1  # 49
        self.output_reduce = Conv2D_BN_activa(_out_in, 32, 1, 1, 0)
        self.output_dilated = Conv2D_BN_activa(32, 32, 3, 1, 2, dilation=2)
        self.output_conv3 = Conv2D_BN_activa(32, 16, 3, 1, 1)
        # 镜面分量头（保留旧名 output_conv4 兼容 V3 checkpoint）
        self.output_conv4 = Conv2D_BN_activa(16, 1, 3, 1, 1, activation=None, if_bn=False)
        self.output_sigmoid = nn.Sigmoid()
        self.output_skip = nn.Conv2d(32, 32, 1, 1, 0, bias=False)
        # ★[D1] 直接漫反射预测头（与镜面头共享前几层特征）
        # 该头有自己的输出 conv，可独立从 16-ch 特征解码 r_direct
        self.output_r_direct_conv = Conv2D_BN_activa(16, 1, 3, 1, 1, activation=None, if_bn=False)
        self.output_r_direct_sigmoid = nn.Sigmoid()

        self.sa1 = SpecularAwareAttention(256 * 3, threshold=0.8)
        self.sa2 = SpecularAwareAttention(128 * 3, threshold=0.8)
        self.sa3 = SpecularAwareAttention(64 * 3, threshold=0.8)
        self.sa4 = SpecularAwareAttention(32 * 3, threshold=0.8)

        # [P10 + NEW] 三条并行调制通道：
        #   grad_gate_i    ← 仅 x_g（梯度先验：哪里是边界）
        #   joint_gate_i   ← concat(x_g, input, x_s) 3 通道：高光相关联合判别
        #   spec_lik_gate_i← 仅 x_s（监督下的高光概率，独立 scale 控制）
        # 三条通道相加再乘到特征上，让主解码器既保留泛边缘提示，也获得显式高光定位。
        self.grad_gate_2 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        self.grad_gate_3 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        self.grad_gate_4 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        # joint_gate: 输入由原来 2 通道(g, input) → 3 通道(g, input, s)
        self.joint_gate_2 = nn.Sequential(nn.Conv2d(3, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        self.joint_gate_3 = nn.Sequential(nn.Conv2d(3, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        self.joint_gate_4 = nn.Sequential(nn.Conv2d(3, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        # [NEW] spec_lik_gate: 1 通道(s)，独立 scale，专门反映监督下的高光概率
        self.spec_lik_gate_2 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        self.spec_lik_gate_3 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        self.spec_lik_gate_4 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True), nn.Sigmoid())
        # ★[D6] V3.2 调整：让三路门控初始就有显著调制作用。
        # 旧版 bias=-1.0 → sigmoid(-1)≈0.27，scale=0.15 clamp[0,0.3]
        # → 初始调制 1+3×0.15×0.27=1.12 几乎无作用
        # 新版 bias=0.0 → sigmoid(0)=0.5, scale=0.25 clamp[0,0.4]
        # → 初始调制 1+3×0.25×0.5=1.375，前期就能发挥作用
        self.grad_gate_scale = nn.Parameter(torch.tensor(0.25))
        self.joint_gate_scale = nn.Parameter(torch.tensor(0.25))
        self.spec_lik_gate_scale = nn.Parameter(torch.tensor(0.25))
        self._init_aux_gates()
        # ★[D1] dual-head 融合系数（logit 形式，sigmoid 后约 0.5，初始平衡两路）
        # alpha = sigmoid(logit_alpha)，最终 r = alpha * r_direct + (1-alpha) * r_subtraction
        # 初始 logit_alpha = 0.0 → alpha = 0.5（两路等权）
        self.dual_head_alpha_logit = nn.Parameter(torch.tensor(0.0))

    def _init_aux_gates(self):
        # ★[D6] bias 由 -1.0 调整为 0.0，让初始 sigmoid 输出≈0.5（中性而非偏低）
        for gate in (self.grad_gate_2, self.grad_gate_3, self.grad_gate_4,
                     self.joint_gate_2, self.joint_gate_3, self.joint_gate_4,
                     self.spec_lik_gate_2, self.spec_lik_gate_3, self.spec_lik_gate_4):
            conv = gate[0]
            nn.init.zeros_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)  # 由 -1.0 → 0.0

    @staticmethod
    def _resize_prior(prior, size, device, dtype):
        if prior.shape[1] != 1:
            prior = prior.mean(dim=1, keepdim=True)
        if prior.shape[-2:] != size:
            prior = F.interpolate(prior, size=size, mode='bilinear', align_corners=False)
        return prior.to(device=device, dtype=dtype)

    def forward(self, c_features, enhanced_features, x_g, x_s, original_input):
        x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c = c_features
        xe1, xe2, xe3, xe4 = enhanced_features

        x0, x1 = x_c.chunk(2, dim=1)
        x = self.FFC1((x0, x1)); x = self.FFC2(x)
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
        # [P10 + NEW] 三通道联合调制：grad + joint(g,i,s) + spec_lik
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
        # [P10 + NEW] 三通道联合调制 Level 3
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
        # [P10 + NEW] 三通道联合调制 Level 4
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

        # [P2] 扩展输出头 with skip connection
        x = self.output_reduce(x)
        skip = self.output_skip(x)
        x = self.output_dilated(x)
        x = x + skip
        x = self.output_conv3(x)
        # ★[D1] V3.2 双头输出
        # 旧版只输出 est_b：return self.output_sigmoid(self.output_conv4(x))
        # 新版从共享的 16-ch 特征 x 同时解码两个头：
        #   est_b      : 镜面反射分量（与旧版一致，保留兼容性）
        #   r_direct   : 直接预测的漫反射分量（绕过减法链，避免暗区被 clamp 抹掉）
        est_b = self.output_sigmoid(self.output_conv4(x))
        r_direct = self.output_r_direct_sigmoid(self.output_r_direct_conv(x))
        # 返回元组：(est_b, r_direct, alpha_logit)
        # XFFC.forward 会接住元组并做下游融合 + 残差精修
        return est_b, r_direct, self.dual_head_alpha_logit
