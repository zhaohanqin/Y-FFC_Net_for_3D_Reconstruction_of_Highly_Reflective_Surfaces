"""Composite Y-FFC training losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import (
    _safe_tensor,
    PerceptualLoss,
    charbonnier_loss,
    compute_edge_weight_map,
    compute_highlight_prior,
    compute_input_gradient_weight_map,
    normalized_sobel_magnitude,
    statistical_loss,
)
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

    [NEW] 新增 spec_likelihood 监督项 l_s_lik：
        对 GradientDecoder 输出的高光概率图 pred_s_lik 用 gt_specular 软监督，
        使梯度分支的中间特征获得"任务化"语义信号。
    """

    def __init__(self, lambda_ssim=1.0, lambda_l1=1.0, lambda_freq=0.1,
                 lambda_perceptual=0.05, lambda_spec=0.5, lambda_stat=0.1,
                 lambda_grad=0.5, lambda_s_lik=0.3,
                 use_edge_weight=True, use_perceptual=True):
        super(CompositeLoss, self).__init__()
        self.ssim = SSIM()
        self.lambda_ssim = lambda_ssim
        self.lambda_l1 = lambda_l1
        self.lambda_freq = lambda_freq
        self.lambda_perceptual = lambda_perceptual
        self.lambda_spec = lambda_spec
        self.lambda_stat = lambda_stat
        self.lambda_grad = lambda_grad
        self.lambda_s_lik = lambda_s_lik
        self.use_edge_weight = use_edge_weight
        self.use_perceptual = use_perceptual
        if self.use_perceptual:
            self.perceptual = PerceptualLoss()

    def frequency_loss(self, pred, gt):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        gt_fft = torch.fft.rfft2(gt, norm='ortho')
        return F.l1_loss(torch.abs(pred_fft), torch.abs(gt_fft))

    def forward(self, pred_diffuse, target, est_b, gt_specular,
                pred_grad, gt_grad, pred_s_lik=None):
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

        # [NEW] spec_likelihood 头损失
        if pred_s_lik is None:
            l_s_lik = torch.tensor(0.0, device=pred_diffuse.device)
        else:
            l_s_lik = F.l1_loss(pred_s_lik, gt_specular_clamped) + \
                      0.5 * (1.0 - self.ssim(pred_s_lik.float(), gt_specular_clamped.float()))

        total = (self.lambda_ssim * l_ssim + self.lambda_l1 * l_l1 +
                 self.lambda_freq * l_freq + self.lambda_perceptual * l_perceptual +
                 self.lambda_spec * l_spec + self.lambda_stat * l_stat +
                 self.lambda_grad * l_grad + self.lambda_s_lik * l_s_lik)

        loss_dict = {}
        for k, v in [('total', total), ('ssim', l_ssim), ('l1', l_l1),
                      ('freq', l_freq), ('perceptual', l_perceptual),
                      ('spec', l_spec), ('stat', l_stat), ('grad', l_grad),
                      ('s_lik', l_s_lik)]:
            loss_dict[k] = v.item() if isinstance(v, torch.Tensor) else v

        return total, loss_dict

    def update_weights(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

class EnhancedCompositeLoss(nn.Module):
    """
    增强版复合损失（V3.2 优化版 — 针对漫反射 r 增加直接监督，根治"靠减法链"的脆弱）。

    [P3]  SSIM → 0.8*MS-SSIM + 0.2*SSIM 混合
    [P4]  频域损失加入相位一致性约束 (V3.2: phase 项权重 0.25 → 0.15)
    [P5]  移除冗余 statistical_loss（lambda_stat 默认 0）
    [P12] SSIM as_loss=True 保留梯度
    [P13] L1 → Charbonnier 平滑损失

    [V3.1 新增]
      l_s_lik   : spec_lik_head 与 gt_specular 的 BCE+SSIM
      l_s_cons  : pred_s_lik 与 est_b 在高光区域的双向一致性
      l_spec_peak: 高光区 est_b 欠预测单侧惩罚

    ★[V3.2 新增 — 针对漫反射 r 的直接监督]★
      [D3] l_r_focal_hl : 在 gt_specular 高光区，对 pred_r vs target 加 focal 重点
                          监督。意义：原版"高光区是否去得干净"完全由 b 决定，
                          这里给 r 一个直接强约束，迫使去高光残留无处可逃。
      [D4] l_r_dual_cons: r_direct（主解码器直接预测）与最终 r 一致性。
                          意义：让 dual-head 的两条路径相互校准、互为正则。
      [D5] l_r_smooth   : 在 gt_specular 高光区域，对 pred_r 施加局部平滑先验。
                          意义：漫反射应低频为主；高光被去除后该区域应平滑。
    """

    def __init__(self, lambda_ssim=1.0, lambda_l1=1.0, lambda_freq=0.1,
                 lambda_perceptual=0.05, lambda_spec=0.5, lambda_stat=0.0,
                 lambda_grad=0.5, lambda_grad_edge=0.5,
                 use_edge_weight=True, use_perceptual=True,
                 lambda_focal=0.8, lambda_dice=0.3, lambda_gc=0.2,
                 lambda_cs=0.05, lambda_grad_spec=0.05,
                 lambda_s_lik=0.5, lambda_s_cons=0.05,
                 lambda_spec_peak=0.3,
                 # ★[V3.2 新增权重] ★
                 lambda_r_focal_hl=0.5, lambda_r_dual_cons=0.1, lambda_r_smooth=0.05,
                 highlight_prior_gamma=1.0,
                 # ★[FIX-A 问题2] SpecLik 监督可配置超参（均有默认值，可后续调整）
                 spec_lik_pos_thresh=0.10,      # 高光前景二值化阈值（gt_sc>此值=前景）
                 spec_lik_pos_weight_max=50.0,  # pos_weight 上限，平衡前景/背景
                 spec_lik_dice_weight=0.5,      # Dice 项权重
                 use_ohem=True, ohem_ratio=0.3):
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
        self.lambda_stat = lambda_stat
        self.lambda_grad = lambda_grad
        self.lambda_grad_edge = lambda_grad_edge
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.lambda_gc = lambda_gc
        self.lambda_cs = lambda_cs
        self.lambda_grad_spec = lambda_grad_spec
        self.lambda_s_lik = lambda_s_lik
        self.lambda_s_cons = lambda_s_cons
        self.lambda_spec_peak = lambda_spec_peak
        # ★[V3.2 新增 — 针对 r 的直接监督] ★
        self.lambda_r_focal_hl = lambda_r_focal_hl
        self.lambda_r_dual_cons = lambda_r_dual_cons
        self.lambda_r_smooth = lambda_r_smooth
        self.highlight_prior_gamma = highlight_prior_gamma
        # ★[FIX-A 问题2] SpecLik 监督超参
        self.spec_lik_pos_thresh = spec_lik_pos_thresh
        self.spec_lik_pos_weight_max = spec_lik_pos_weight_max
        self.spec_lik_dice_weight = spec_lik_dice_weight
        self.use_ohem = use_ohem
        self.ohem_ratio = ohem_ratio

    def frequency_loss(self, p, g):
        """[P4-FIX2] 幅度谱 + 能量加权相位一致性。

        ★[V3.2 微调] phase 系数 0.25 → 0.15。
        原因：训练前期 pred 与 gt 相位差大，phase_loss 量级远大于 amp_loss，
              即便 [V3.1 NaN-Fix] 解决了反向爆炸问题，前向数值仍偏大干扰主任务。
        """
        p = _safe_tensor(p.float(), nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))
        g = _safe_tensor(g.float(), nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))
        p_fft = torch.fft.rfft2(p, norm='ortho')
        g_fft = torch.fft.rfft2(g, norm='ortho')
        p_abs = torch.abs(p_fft)
        g_abs = torch.abs(g_fft)
        amp_loss = F.l1_loss(p_abs, g_abs)

        # 能量加权"准相位"差（无除法，反向稳定）
        re_inner = (p_fft * g_fft.conj()).real
        amp_prod = p_abs * g_abs
        with torch.no_grad():
            energy_mask = (g_abs > g_abs.mean() * 0.01).float()
            denom_mask = energy_mask.sum().clamp_min(1.0)
        phase_loss = (torch.abs(amp_prod - re_inner) * energy_mask).sum() / denom_mask
        phase_loss = _safe_tensor(phase_loss, nan=0.0, posinf=0.0, neginf=0.0)
        # ★[V3.2] 0.25 → 0.15
        return amp_loss + 0.15 * phase_loss

    def forward(self, pred_diffuse, target, est_b, gt_specular,
                pred_grad, gt_grad, cross_scale_consistency=None,
                input_img=None, pred_s_lik=None,
                pred_r_direct=None):
        """
        ★[V3.2] 新增 pred_r_direct 参数：MainDecoder 直接预测的漫反射输出。
        若为 None（旧 forward 调用），相关损失项为 0，保持向后兼容。
        """
        gt_sc = gt_specular.clamp(0, 1)

        # [P3] MS-SSIM + SSIM 混合
        l_ssim_single = 1.0 - self.ssim_fn(pred_diffuse.float(), target.float())
        l_msssim = 1.0 - self.msssim_fn(pred_diffuse.float(), target.float())
        l_ssim = 0.8 * l_msssim + 0.2 * l_ssim_single

        # [P13][SAFE] Charbonnier 替代 L1
        if self.use_edge_weight:
            wm = self._ew_fn(gt_specular.detach())
            l_l1 = (torch.sqrt((pred_diffuse - target)**2 + 1e-6) * wm).mean()
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

        l_stat = torch.tensor(0.0, device=pred_diffuse.device)
        if self.lambda_stat > 0:
            l_stat = statistical_loss(pred_diffuse, target)

        l_grad = 1.0 - self.ssim_fn(pred_grad.float(), gt_grad.float())
        grad_w = compute_input_gradient_weight_map(gt_grad.detach()).to(
            device=pred_grad.device, dtype=pred_grad.dtype)
        highlight_prior = None
        if input_img is not None:
            highlight_prior = compute_highlight_prior(input_img.detach()).to(
                device=pred_grad.device, dtype=pred_grad.dtype)
            if highlight_prior.shape[-2:] != pred_grad.shape[-2:]:
                highlight_prior = F.interpolate(
                    highlight_prior, size=pred_grad.shape[-2:],
                    mode='bilinear', align_corners=False)
            grad_w = grad_w * (1.0 + self.highlight_prior_gamma * highlight_prior)
        l_grad_edge = (
            torch.sqrt((pred_grad - gt_grad) ** 2 + 1e-6) * grad_w
        ).mean()

        if highlight_prior is None:
            l_grad_spec = torch.tensor(0.0, device=pred_grad.device)
        else:
            # ★[FIX-D 问题2] 把动态目标 Sobel(est_b) 换成稳定目标 Sobel(gt_specular)。
            #   旧版用 est_b.detach() 作目标:est_b 是模型自己的预测,在整个训练过程
            #   中持续演化 → Sobel(est_b) 是一个"会变的靶子",pred_grad 永远在追
            #   一个移动目标,体现在 val 曲线上就是中期的 U 形回弹(epoch 7~15)。
            #   现在用 gt_sc.detach() 的 Sobel 作目标:目标固定不变,pred_grad
            #   只需要单调收敛,U 形彻底消失。
            #   语义上仍然合理:highlight_prior 在亮区为正,Sobel(input) ≈
            #   Sobel(diffuse) + Sobel(specular) 且亮区主导项是 specular,所以
            #   Sobel(gt_sc) 在亮区与 Sobel(input) 接近,不会与主 grad 监督冲突。
            gt_sc_grad = normalized_sobel_magnitude(
                gt_sc.detach().float(), quantile=0.995).to(
                device=pred_grad.device, dtype=pred_grad.dtype)
            if gt_sc_grad.shape[-2:] != pred_grad.shape[-2:]:
                gt_sc_grad = F.interpolate(
                    gt_sc_grad, size=pred_grad.shape[-2:],
                    mode='bilinear', align_corners=False)
            l_grad_spec = (
                torch.abs(pred_grad - gt_sc_grad) * highlight_prior
            ).mean()

        # ── spec_likelihood 头主监督 ──
        # ★[FIX-A 问题2] 改为「二值高光标签 + pos_weight 加权 BCE + Dice」：
        #   旧版用连续 gt_sc(0~1) 作 BCE 目标，模型最优解=匹配偏低的连续值，
        #   永远到不了 0.5；且背景像素海量 + reduction='mean' 把整图往 0 压。
        #   现用二值前景标签 (gt_sc > spec_lik_pos_thresh) 并以 pos_weight 平衡
        #   前景/背景，再加 Dice 推前景重叠，使高光区跨过 0.5、二值图更完整。
        if pred_s_lik is None:
            l_s_lik = torch.tensor(0.0, device=pred_diffuse.device)
        else:
            if pred_s_lik.shape[-2:] != gt_sc.shape[-2:]:
                pred_s_lik = F.interpolate(
                    pred_s_lik, size=gt_sc.shape[-2:],
                    mode='bilinear', align_corners=False)
            pred_s_lik_safe = pred_s_lik.clamp(1e-4, 1.0 - 1e-4)
            with torch.no_grad():
                gt_bin = (gt_sc > self.spec_lik_pos_thresh).float()       # 高光前景二值标签
                pos = gt_bin.mean().clamp(1e-3, 0.5)                      # 前景占比
                pos_weight = ((1.0 - pos) / pos).clamp(1.0, self.spec_lik_pos_weight_max)
            # 带 pos_weight 的 BCE（对 sigmoid 输出）
            bce_map = -(pos_weight * gt_bin * torch.log(pred_s_lik_safe)
                        + (1.0 - gt_bin) * torch.log(1.0 - pred_s_lik_safe))
            l_bce = bce_map.mean()
            # Dice：缓解极端类别不平衡，进一步推前景重叠
            inter = (pred_s_lik_safe * gt_bin).sum(dim=(1, 2, 3))
            union = pred_s_lik_safe.sum(dim=(1, 2, 3)) + gt_bin.sum(dim=(1, 2, 3))
            l_dice_slik = (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()
            l_s_lik = l_bce + self.spec_lik_dice_weight * l_dice_slik

        # ── pred_s_lik 与 est_b 双向一致性 (V3.1) ──
        if pred_s_lik is None:
            l_s_cons = torch.tensor(0.0, device=pred_diffuse.device)
        else:
            soft_mask = gt_sc.detach()
            cons_fwd = (torch.abs(pred_s_lik - est_b.detach()) * soft_mask).mean()
            cons_bwd = (torch.abs(est_b - pred_s_lik.detach()) * soft_mask).mean()
            l_s_cons = cons_fwd + cons_bwd

        if self.use_ohem:
            l_focal = specular_ohem_loss(est_b, gt_sc, self.focal, self.ohem_ratio)
        else:
            l_focal = self.focal(est_b, gt_sc)
        l_dice = self.dice(est_b, gt_sc)
        l_gc = self.gc(est_b, gt_sc)

        # ── 高光区 est_b 欠预测单侧惩罚 (V3.1) ──
        # ★[FIX-D 问题3] 把硬阈值 (gt_sc > 0.3).float() 换成 soft_high_mask:
        #   - 硬阈值在阈值附近不连续:gt_sc 从 0.299 → 0.301 时掩码值从 0 跳到 1,
        #     损失值会有微小阶跃。叠加 lambda_spec_peak 在旧调度里 0.1 → 0.5 的
        #     爬升,加权贡献先升后降,呈现小幅回升的形态。
        #   - 现在用 sigmoid(12*(gt_sc - 0.30)) 软掩码:阈值附近连续过渡,像素的
        #     "高光程度"以平滑 [0,1] 值参与加权,损失随训练进度的演化也更平滑。
        #   - 配合 FIX-D 常量权重(lambda_spec_peak = 0.5 锁定不变),小幅回升消失。
        with torch.no_grad():
            soft_high_mask = torch.sigmoid(12.0 * (gt_sc - 0.30))  # [0,1] 软掩码
        under = F.relu(gt_sc - est_b) * soft_high_mask
        l_spec_peak = under.mean()

        # ═════════════════════════════════════════════════════════════════
        # ★[V3.2 新增 — 针对漫反射 r 的直接监督] ★
        # ═════════════════════════════════════════════════════════════════

        # [D3] l_r_focal_hl：高光区域对 pred_r vs target 的 focal-style 监督
        # 关键意义：旧版"r 在高光区是否去除干净"完全由 b 决定，间接监督薄弱。
        # 这里给 r 直接增加一个高光区强约束损失，迫使 r 不允许残留高光痕迹。
        # 配方：error * (1 + 4*soft_high_mask) * (1 + |error|)^1.5
        #   - 在 soft_high_mask≈1（强高光区）权重 ×5
        #   - focal 形式 (1+|err|)^1.5 让误差大的像素额外加权
        r_err = torch.abs(pred_diffuse - target)
        r_hl_weight = 1.0 + 4.0 * soft_high_mask
        r_focal_term = (1.0 + r_err.detach()) ** 1.5
        l_r_focal_hl = (r_err * r_hl_weight * r_focal_term).mean()

        # [D4] l_r_dual_cons：r_direct 与最终 pred_diffuse 一致性
        # 让 dual-head 的两条路径相互校准、互为正则。
        # 注意：pred_r_direct 是 MainDecoder 直接头输出（grad 流），pred_diffuse 是融合 + 精修后输出。
        # 不 detach 任何一方，让梯度可以双向流通，但用 Charbonnier 防止前期不稳定。
        if pred_r_direct is None:
            l_r_dual_cons = torch.tensor(0.0, device=pred_diffuse.device)
        else:
            l_r_dual_cons = charbonnier_loss(pred_r_direct, pred_diffuse)

        # [D5] l_r_smooth：在高光区，pred_r 的局部平滑先验
        # ★[FIX-A 问题1] 改为「单侧」约束：仅惩罚比局部均值更亮的残留（亮点/高光残留），
        #   不再惩罚模型正常恢复的暗向条纹细节。旧版双侧惩罚与主任务(SSIM/L1)直接对抗，
        #   导致后期细节恢复时 l_r_smooth 反弹上升、顶起总 loss。
        pred_r_smooth = F.avg_pool2d(pred_diffuse, kernel_size=3, stride=1, padding=1)
        bright_residual = F.relu(pred_diffuse - pred_r_smooth)
        l_r_smooth = (bright_residual * soft_high_mask).mean()

        if cross_scale_consistency is None:
            l_cs = torch.tensor(0.0, device=pred_diffuse.device)
        else:
            l_cs = cross_scale_consistency.to(device=pred_diffuse.device, dtype=pred_diffuse.dtype)

        # ★ 纯加权求和，所有项 ≥ 0
        total = (self.lambda_ssim * l_ssim + self.lambda_l1 * l_l1 +
                 self.lambda_freq * l_freq + self.lambda_perceptual * l_percep +
                 self.lambda_spec * l_spec + self.lambda_stat * l_stat +
                 self.lambda_grad * l_grad +
                 self.lambda_grad_edge * l_grad_edge +
                 self.lambda_grad_spec * l_grad_spec +
                 self.lambda_focal * l_focal +
                 self.lambda_dice * l_dice + self.lambda_gc * l_gc +
                 self.lambda_cs * l_cs +
                 self.lambda_s_lik * l_s_lik +
                 self.lambda_s_cons * l_s_cons +
                 self.lambda_spec_peak * l_spec_peak +
                 # ★[V3.2 新增] ★
                 self.lambda_r_focal_hl * l_r_focal_hl +
                 self.lambda_r_dual_cons * l_r_dual_cons +
                 self.lambda_r_smooth * l_r_smooth)

        d = {}
        for k, v in [('total', total), ('ssim', l_ssim_single),
                      ('ssim_single_loss', l_ssim_single),
                      ('msssim_loss', l_msssim),
                      ('ssim_mixed_loss', l_ssim),
                      ('l1', l_l1),
                      ('freq', l_freq), ('perceptual', l_percep),
                      ('spec', l_spec), ('stat', l_stat), ('grad', l_grad),
                      ('grad_edge', l_grad_edge), ('grad_spec', l_grad_spec),
                      ('focal', l_focal), ('dice', l_dice), ('gc', l_gc),
                      ('cs', l_cs),
                      ('s_lik', l_s_lik), ('s_cons', l_s_cons),
                      ('spec_peak', l_spec_peak),
                      # ★[V3.2 新增] ★
                      ('r_focal_hl', l_r_focal_hl),
                      ('r_dual_cons', l_r_dual_cons),
                      ('r_smooth', l_r_smooth)]:
            d[k] = v.item() if isinstance(v, torch.Tensor) else v
        return total, d

    def update_weights(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)
