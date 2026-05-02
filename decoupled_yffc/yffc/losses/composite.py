"""Composite Y-FFC training losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import PerceptualLoss, charbonnier_loss, compute_edge_weight_map, statistical_loss
from .metrics import MSSSIM, SSIM
from .specular import (
    GradientConsistencyLoss,
    SpecularDiceLoss,
    SpecularFocalLoss,
    specular_ohem_loss,
)


class CompositeLoss(nn.Module):
    """
    多目标复合损失。
    整合 SSIM / L1 / 频域 / 感知 / 镜面直接监督 / 统计 / 梯度 / 边缘加权。
    """

    def __init__(self, lambda_ssim=1.0, lambda_l1=1.0, lambda_freq=0.1,
                 lambda_perceptual=0.05, lambda_spec=0.5, lambda_stat=0.1,
                 lambda_grad=0.5, use_edge_weight=True, use_perceptual=True):
        super(CompositeLoss, self).__init__()
        self.ssim = SSIM()
        self.lambda_ssim = lambda_ssim
        self.lambda_l1 = lambda_l1
        self.lambda_freq = lambda_freq
        self.lambda_perceptual = lambda_perceptual
        self.lambda_spec = lambda_spec
        self.lambda_stat = lambda_stat
        self.lambda_grad = lambda_grad
        self.use_edge_weight = use_edge_weight
        self.use_perceptual = use_perceptual
        if self.use_perceptual:
            self.perceptual = PerceptualLoss()

    def frequency_loss(self, pred, gt):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        gt_fft = torch.fft.rfft2(gt, norm='ortho')
        return F.l1_loss(torch.abs(pred_fft), torch.abs(gt_fft))

    def forward(self, pred_diffuse, target, est_b, gt_specular,
                pred_grad, gt_grad):
        l_ssim = 1.0 - self.ssim(pred_diffuse.float(), target.float())

        if self.use_edge_weight:
            weight_map = compute_edge_weight_map(gt_specular.detach())
            l_l1 = (torch.abs(pred_diffuse - target) * weight_map).mean()
        else:
            l_l1 = F.l1_loss(pred_diffuse, target)

        l_freq = self.frequency_loss(pred_diffuse.float(), target.float())

        if self.use_perceptual:
            l_perceptual = self.perceptual(pred_diffuse, target)
        else:
            l_perceptual = torch.tensor(0.0, device=pred_diffuse.device)

        gt_specular_clamped = gt_specular.clamp(0.0, 1.0)
        l_spec = F.l1_loss(est_b, gt_specular_clamped) + \
                 0.5 * (1.0 - self.ssim(est_b.float(), gt_specular_clamped.float()))

        l_stat = statistical_loss(pred_diffuse, target)
        l_grad = 1.0 - self.ssim(pred_grad.float(), gt_grad.float())

        total = (self.lambda_ssim * l_ssim + self.lambda_l1 * l_l1 +
                 self.lambda_freq * l_freq + self.lambda_perceptual * l_perceptual +
                 self.lambda_spec * l_spec + self.lambda_stat * l_stat +
                 self.lambda_grad * l_grad)

        loss_dict = {}
        for k, v in [('total', total), ('ssim', l_ssim), ('l1', l_l1),
                      ('freq', l_freq), ('perceptual', l_perceptual),
                      ('spec', l_spec), ('stat', l_stat), ('grad', l_grad)]:
            loss_dict[k] = v.item() if isinstance(v, torch.Tensor) else v

        return total, loss_dict

    def update_weights(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class EnhancedCompositeLoss(nn.Module):
    """
    增强版复合损失（V3 优化版 — 负数损失已修复）。
    [P3]  SSIM → 0.8*MS-SSIM + 0.2*SSIM 混合
    [P4]  频域损失加入相位一致性约束
    [P5]  移除冗余 statistical_loss（lambda_stat 默认 0）
          ★ 注意：不再使用可学习权重，这是导致负数损失的根源 ★
    [P12] SSIM as_loss=True 保留梯度
    [P13] L1 → Charbonnier 平滑损失
    """

    def __init__(self, lambda_ssim=1.0, lambda_l1=1.0, lambda_freq=0.1,
                 lambda_perceptual=0.05, lambda_spec=0.5, lambda_stat=0.0,
                 lambda_grad=0.5, use_edge_weight=True, use_perceptual=True,
                 lambda_focal=0.8, lambda_dice=0.3, lambda_gc=0.2,
                 lambda_cs=0.05, use_ohem=True, ohem_ratio=0.3):
        super().__init__()

        self.ssim_fn = SSIM(as_loss=True)       # [P12]
        self.msssim_fn = MSSSIM(n_scales=3)     # [P3]
        self.use_edge_weight = use_edge_weight
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual = PerceptualLoss()
            if not self.perceptual.is_pretrained:
                self.use_perceptual = False
        self._ew_fn = compute_edge_weight_map

        self.focal = SpecularFocalLoss()
        self.dice = SpecularDiceLoss()
        self.gc = GradientConsistencyLoss()

        self.lambda_ssim = lambda_ssim
        self.lambda_l1 = lambda_l1
        self.lambda_freq = lambda_freq
        self.lambda_perceptual = lambda_perceptual
        self.lambda_spec = lambda_spec
        self.lambda_stat = lambda_stat     # [P5] 默认 0，移除冗余 stat
        self.lambda_grad = lambda_grad
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.lambda_gc = lambda_gc
        self.lambda_cs = lambda_cs
        self.use_ohem = use_ohem
        self.ohem_ratio = ohem_ratio
        # ★ P5 修复：不再有可学习损失权重 ★
        # ★ 所有损失项的权重完全由 lambda_* 和 ImprovedWeightScheduler 控制 ★

    def frequency_loss(self, p, g):
        """[P4-FIX] 幅度谱 + 相位一致性联合损失（修复数值稳定性）。"""
        p_fft = torch.fft.rfft2(p, norm='ortho')
        g_fft = torch.fft.rfft2(g, norm='ortho')
        amp_loss = F.l1_loss(torch.abs(p_fft), torch.abs(g_fft))
        # 修复：用复数乘法代替 torch.angle() 的不连续梯度
        # cos(angle_p - angle_g) = Re(p_fft * conj(g_fft)) / (|p_fft| * |g_fft|)
        p_norm = torch.abs(p_fft).clamp(min=1e-8)
        g_norm = torch.abs(g_fft).clamp(min=1e-8)
        cross = p_fft * g_fft.conj()
        cos_diff = cross.real / (p_norm * g_norm)
        phase_loss = (1.0 - cos_diff).mean()
        return amp_loss + 0.5 * phase_loss

    def forward(self, pred_diffuse, target, est_b, gt_specular,
                pred_grad, gt_grad, cross_scale_consistency=None):
        gt_sc = gt_specular.clamp(0, 1)

        # [P3] MS-SSIM + SSIM 混合
        l_ssim_single = 1.0 - self.ssim_fn(pred_diffuse.float(), target.float())
        l_msssim = 1.0 - self.msssim_fn(pred_diffuse.float(), target.float())
        l_ssim = 0.8 * l_msssim + 0.2 * l_ssim_single

        # [P13] Charbonnier 替代 L1
        if self.use_edge_weight:
            wm = self._ew_fn(gt_specular.detach())
            l_l1 = (torch.sqrt((pred_diffuse - target)**2 + 1e-12) * wm).mean()
        else:
            l_l1 = charbonnier_loss(pred_diffuse, target)

        # [P4] 频域损失含相位
        l_freq = self.frequency_loss(pred_diffuse.float(), target.float())

        l_percep = (self.perceptual(pred_diffuse, target)
                    if self.use_perceptual
                    else torch.tensor(0.0, device=pred_diffuse.device))

        # [P13] Charbonnier for specular
        l_spec = (charbonnier_loss(est_b, gt_sc) +
                  0.5 * (1.0 - self.ssim_fn(est_b.float(), gt_sc.float())))

        # [P5] stat 默认关闭
        l_stat = torch.tensor(0.0, device=pred_diffuse.device)
        if self.lambda_stat > 0:
            l_stat = statistical_loss(pred_diffuse, target)

        l_grad = 1.0 - self.ssim_fn(pred_grad.float(), gt_grad.float())

        if self.use_ohem:
            l_focal = specular_ohem_loss(est_b, gt_sc, self.focal, self.ohem_ratio)
        else:
            l_focal = self.focal(est_b, gt_sc)
        l_dice = self.dice(est_b, gt_sc)
        l_gc = self.gc(est_b, gt_sc)
        if cross_scale_consistency is None:
            l_cs = torch.tensor(0.0, device=pred_diffuse.device)
        else:
            l_cs = cross_scale_consistency.to(device=pred_diffuse.device, dtype=pred_diffuse.dtype)

        # ★ 纯加权求和，所有项 ≥ 0，total 必定 ≥ 0 ★
        total = (self.lambda_ssim * l_ssim + self.lambda_l1 * l_l1 +
                 self.lambda_freq * l_freq + self.lambda_perceptual * l_percep +
                 self.lambda_spec * l_spec + self.lambda_stat * l_stat +
                 self.lambda_grad * l_grad + self.lambda_focal * l_focal +
                 self.lambda_dice * l_dice + self.lambda_gc * l_gc +
                 self.lambda_cs * l_cs)

        d = {}
        for k, v in [('total', total), ('ssim', l_ssim_single),
                      ('ssim_single_loss', l_ssim_single),
                      ('msssim_loss', l_msssim),
                      ('ssim_mixed_loss', l_ssim),
                      ('l1', l_l1),
                      ('freq', l_freq), ('perceptual', l_percep),
                      ('spec', l_spec), ('stat', l_stat), ('grad', l_grad),
                      ('focal', l_focal), ('dice', l_dice), ('gc', l_gc),
                      ('cs', l_cs)]:
            d[k] = v.item() if isinstance(v, torch.Tensor) else v
        return total, d

    def update_weights(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)
