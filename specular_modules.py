"""
specular_modules.py
====================
高反光检测专用模块集合，包含：

  1️⃣  SpecularAwareAttention   — 镜面反射感知注意力（利用输入亮度先验）
  2️⃣  HighFrequencyEnhancement — 拉普拉斯 + 可学习多尺度高频增强
  3️⃣  SpecularFocalLoss        — 正负样本不平衡 Focal 加权
  4️⃣  SpecularDiceLoss         — 高亮区域 Dice 损失
  5️⃣  GradientConsistencyLoss  — Specular 边界梯度一致性
  6️⃣  EnhancedCompositeLoss    — 整合全部损失的最终组合
  7️⃣  HighlightAugmentation    — 合成高反光数据增强
  8️⃣  EnhancedPairedTransform  — 增强版配对变换
  9️⃣  specular_ohem_loss       — 在线难样本挖掘

所有模块均为独立文件，通过 import 使用，不修改原始 ffc.py / Y_FFC.py。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math


# ════════════════════════════════════════════════════════════════════════
# 1️⃣ Specular-Aware Attention
# ════════════════════════════════════════════════════════════════════════
#
# 【作用】利用原始输入的亮度信息生成高反光空间先验掩码，与深层语义
#   特征融合后产生 specular-aware 通道+空间注意力权重。
#   高亮区域特征被放大 → 网络天然具有更高响应灵敏度。
#
# 【插入位置】MainDecoderV2 每级 transposed_block 输出之后
#   （forward 需接收 original_input 参数）
#
# 【结构设计】
#   original_input → resize → soft_threshold → highlight_mask(1ch)
#   features(Cch) ⊕ mask(1ch) → Conv3×3→BN→GELU→Conv3×3→Sigmoid → att(Cch)
#   output = features × (1 + att)            ← 残差增强
# ════════════════════════════════════════════════════════════════════════

class SpecularAwareAttention(nn.Module):
    """
    镜面反射感知注意力模块。

    参数:
        in_channels : 输入特征通道数
        threshold   : soft 阈值中心（0~1）
        steepness   : sigmoid 陡度，越大越接近硬阈值
    """
    def __init__(self, in_channels, threshold=0.8, steepness=15.0):
        super().__init__()
        self.threshold = threshold
        self.steepness = steepness

        # 自适应阈值：根据全局特征统计微调
        mid = max(in_channels // 16, 4)
        self.adaptive_thresh = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        # 融合网络：features(C) + highlight_mask(1) → 逐通道权重(C)
        self.modulation = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, features, original_input):
        """
        Args:
            features       : [B, C, H, W]
            original_input : [B, 1, H_in, W_in]  值域 [0,1]
        Returns:
            [B, C, H, W] — 注意力增强后的特征
        """
        inp = original_input
        if inp.shape[-2:] != features.shape[-2:]:
            inp = F.interpolate(inp, size=features.shape[-2:],
                                mode='bilinear', align_corners=False)

        # 自适应阈值
        t_off = self.adaptive_thresh(features)               # [B,1,1,1]
        eff_t = self.threshold * (0.5 + 0.5 * t_off)        # ∈ [0.5τ, τ]

        # soft 高亮掩码
        mask = torch.sigmoid(self.steepness * (inp - eff_t)) # [B,1,H,W]

        # 融合 → 注意力权重
        att = self.modulation(torch.cat([features, mask], dim=1))  # [B,C,H,W]

        # 残差增强：非高亮区 att≈0 → output≈features，高亮区 att≈1 → output≈2×features
        return features * (1.0 + att)


# ════════════════════════════════════════════════════════════════════════
# 2️⃣ High-Frequency Enhancement
# ════════════════════════════════════════════════════════════════════════
#
# 【作用】显式提取编码特征中的高频分量（边缘/跃变），为 specular 检测
#   提供清晰的边界信号。高反光区域核心特征 = 像素值急剧跃变。
#
# 【插入位置】EncoderV2 的 FFC bottleneck 之后（对 x_c 做高频增强）
#
# 【结构设计】
#   x → Laplacian(fixed, groups=C)                  → lap_hf(C)
#   x → Conv3×3→BN→GELU→DilatedConv3×3→BN→GELU    → learn_hf(C//2)
#   [x, lap_hf, learn_hf] → Conv1×1→BN→GELU→Conv3×3→BN → fused(C)
#   fused → SE_channel_attention → fused_weighted(C)
#   output = x + fused_weighted                      ← 残差
# ════════════════════════════════════════════════════════════════════════

class HighFrequencyEnhancement(nn.Module):
    """
    高频特征增强模块。

    参数:
        channels : 输入/输出通道数
    """
    def __init__(self, channels):
        super().__init__()

        # 固定 Laplacian 核
        lap = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32)
        self.register_buffer('lap_kernel', lap.unsqueeze(0).unsqueeze(0))

        # 可学习多尺度高频
        half = channels // 2
        self.hf_branch = nn.Sequential(
            nn.Conv2d(channels, half, 3, 1, 1, bias=False),
            nn.BatchNorm2d(half),
            nn.GELU(),
            nn.Conv2d(half, half, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(half),
            nn.GELU(),
        )

        # 融合：原始(C) + Laplacian(C) + 学习(C//2)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels + channels + half, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # SE 通道注意力
        se_mid = max(channels // 8, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, se_mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: [B, C, H, W] → [B, C, H, W]"""
        B, C, H, W = x.shape

        # Laplacian 高频
        lap_k = self.lap_kernel.expand(C, 1, 3, 3).to(x.dtype)
        lap_hf = F.conv2d(x, lap_k, padding=1, groups=C)

        # 学习高频
        learn_hf = self.hf_branch(x)

        # 融合 + SE
        fused = self.fusion(torch.cat([x, lap_hf, learn_hf], dim=1))
        fused = fused * self.se(fused)

        return x + fused


# ════════════════════════════════════════════════════════════════════════
# 3️⃣ Specular Focal Loss
# ════════════════════════════════════════════════════════════════════════
#
# 【作用】高亮像素占比极小（< 5%），MAE 被大面积背景零误差稀释。
#   本损失对 gt_specular > threshold 的像素给予 focal_weight 倍权重，
#   同时对误差大（难预测）的像素额外加权（Focal 调制）。
#
# loss = mean( |pred - gt| × highlight_weight × (1 + |pred-gt|^γ) )
# ════════════════════════════════════════════════════════════════════════

class SpecularFocalLoss(nn.Module):
    def __init__(self, threshold=0.05, focal_weight=5.0, gamma=2.0):
        super().__init__()
        self.threshold = threshold
        self.focal_weight = focal_weight
        self.gamma = gamma

    def forward(self, pred, gt):
        err = torch.abs(pred - gt)
        mask = torch.sigmoid(30.0 * (gt - self.threshold))
        w = 1.0 + (self.focal_weight - 1.0) * mask
        focal = (1.0 + err.detach()) ** self.gamma
        return (err * w * focal).mean()


# ════════════════════════════════════════════════════════════════════════
# 4️⃣ Specular Dice Loss
# ════════════════════════════════════════════════════════════════════════
#
# 【作用】从分割视角约束 specular map：将预测和 GT 转为"高亮概率图"
#   后计算 Dice 系数。Dice 不受背景零值稀释，直接衡量前景重叠度。
# ════════════════════════════════════════════════════════════════════════

class SpecularDiceLoss(nn.Module):
    def __init__(self, threshold=0.05, smooth=1.0):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, pred, gt):
        pred_p = torch.sigmoid(20.0 * (pred - self.threshold))
        gt_m = (gt > self.threshold).float()
        inter = (pred_p * gt_m).sum(dim=(1,2,3))
        union = pred_p.sum(dim=(1,2,3)) + gt_m.sum(dim=(1,2,3))
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


# ════════════════════════════════════════════════════════════════════════
# 5️⃣ Gradient Consistency Loss
# ════════════════════════════════════════════════════════════════════════
#
# 【作用】约束预测 specular map 的空间梯度与 GT 在幅值和方向上一致，
#   确保高反光区域边界精确对齐，避免边界模糊和位移。
#
# L = L1(|∇pred|, |∇gt|) + 0.3 × (1 - cos(∇pred, ∇gt))  [仅在 GT 梯度大处]
# ════════════════════════════════════════════════════════════════════════

class GradientConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sx.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sx.t().unsqueeze(0).unsqueeze(0))

    def forward(self, pred, gt):
        sx = self.sobel_x.to(pred.dtype)
        sy = self.sobel_y.to(pred.dtype)

        pgx = F.conv2d(pred, sx, padding=1)
        pgy = F.conv2d(pred, sy, padding=1)
        ggx = F.conv2d(gt, sx, padding=1)
        ggy = F.conv2d(gt, sy, padding=1)

        pm = torch.sqrt(pgx**2 + pgy**2 + 1e-8)
        gm = torch.sqrt(ggx**2 + ggy**2 + 1e-8)

        mag_loss = F.l1_loss(pm, gm)

        mask = (gm > gm.mean() * 0.5).float()
        cos = (pgx*ggx + pgy*ggy) / (pm * gm + 1e-8)
        dir_loss = ((1.0 - cos) * mask).sum() / (mask.sum() + 1e-8)

        return mag_loss + 0.3 * dir_loss


# ════════════════════════════════════════════════════════════════════════
# 6️⃣ Enhanced Composite Loss  —— 最终总损失
# ════════════════════════════════════════════════════════════════════════
#
# 【总损失公式】
#
#  L_total = λ_ssim  · (1 - SSIM_diffuse)
#          + λ_l1    · L1_diffuse             (边缘加权)
#          + λ_freq  · L1_freq_diffuse
#          + λ_percep· L_perceptual
#          + λ_spec  · (L1 + 0.5·SSIM)_spec  (基础镜面监督)
#          + λ_stat  · L_statistical
#          + λ_grad  · (1 - SSIM_gradient)
#          + λ_focal · SpecularFocal          🆕
#          + λ_dice  · SpecularDice           🆕
#          + λ_gc    · GradientConsistency    🆕
# ════════════════════════════════════════════════════════════════════════

class EnhancedCompositeLoss(nn.Module):
    def __init__(self,
                 lambda_ssim=1.0, lambda_l1=1.0, lambda_freq=0.1,
                 lambda_perceptual=0.05, lambda_spec=0.5,
                 lambda_stat=0.1, lambda_grad=0.5,
                 use_edge_weight=True, use_perceptual=True,
                 lambda_focal=0.8, lambda_dice=0.3, lambda_gc=0.2,
                 use_ohem=True, ohem_ratio=0.3):
        super().__init__()

        from Y_FFC import (SSIM, PerceptualLoss, compute_edge_weight_map,
                           statistical_loss)

        self.ssim_fn = SSIM()
        self.use_edge_weight = use_edge_weight
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual = PerceptualLoss()
        self._ew_fn = compute_edge_weight_map
        self._stat_fn = statistical_loss

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
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.lambda_gc = lambda_gc
        self.use_ohem = use_ohem
        self.ohem_ratio = ohem_ratio

    def frequency_loss(self, p, g):
        return F.l1_loss(torch.abs(torch.fft.rfft2(p, norm='ortho')),
                         torch.abs(torch.fft.rfft2(g, norm='ortho')))

    def forward(self, pred_diffuse, target, est_b, gt_specular,
                pred_grad, gt_grad):
        gt_sc = gt_specular.clamp(0,1)

        l_ssim = 1.0 - self.ssim_fn(pred_diffuse.float(), target.float())

        if self.use_edge_weight:
            wm = self._ew_fn(gt_specular.detach())
            l_l1 = (torch.abs(pred_diffuse - target) * wm).mean()
        else:
            l_l1 = F.l1_loss(pred_diffuse, target)

        l_freq = self.frequency_loss(pred_diffuse.float(), target.float())

        l_percep = (self.perceptual(pred_diffuse, target)
                    if self.use_perceptual
                    else torch.tensor(0.0, device=pred_diffuse.device))

        l_spec = (F.l1_loss(est_b, gt_sc) +
                  0.5*(1.0 - self.ssim_fn(est_b.float(), gt_sc.float())))

        l_stat = self._stat_fn(pred_diffuse, target)
        l_grad = 1.0 - self.ssim_fn(pred_grad.float(), gt_grad.float())

        if self.use_ohem:
            l_focal = specular_ohem_loss(est_b, gt_sc, self.focal, self.ohem_ratio)
        else:
            l_focal = self.focal(est_b, gt_sc)
        l_dice = self.dice(est_b, gt_sc)
        l_gc = self.gc(est_b, gt_sc)

        total = (self.lambda_ssim*l_ssim + self.lambda_l1*l_l1 +
                 self.lambda_freq*l_freq + self.lambda_perceptual*l_percep +
                 self.lambda_spec*l_spec + self.lambda_stat*l_stat +
                 self.lambda_grad*l_grad + self.lambda_focal*l_focal +
                 self.lambda_dice*l_dice + self.lambda_gc*l_gc)

        d = {}
        for k, v in [('total',total),('ssim',l_ssim),('l1',l_l1),
                      ('freq',l_freq),('perceptual',l_percep),
                      ('spec',l_spec),('stat',l_stat),('grad',l_grad),
                      ('focal',l_focal),('dice',l_dice),('gc',l_gc)]:
            d[k] = v.item() if isinstance(v, torch.Tensor) else v
        return total, d

    def update_weights(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ════════════════════════════════════════════════════════════════════════
# 7️⃣ Highlight Augmentation —— 合成高反光数据增强
# ════════════════════════════════════════════════════════════════════════
#
# 【作用】在 input 上随机叠加合成高斯 blob 高反光，
#   同步更新 gt_specular，增加训练多样性。
#
# 【实现方法】
#   随机生成 1~6 个高斯 blob（中心/半径/强度均随机），
#   叠加到 input 和 gt_specular 上。
# ════════════════════════════════════════════════════════════════════════

class HighlightAugmentation:
    def __init__(self, p=0.4, num_range=(1,6),
                 radius_range=(3,25), intensity_range=(0.2,0.7)):
        self.p = p
        self.num_range = num_range
        self.radius_range = radius_range
        self.intensity_range = intensity_range

    def __call__(self, input_img, target_img, gt_specular):
        if random.random() > self.p:
            return input_img, target_img, gt_specular
        _, H, W = input_img.shape
        n = random.randint(*self.num_range)
        synth = torch.zeros(1, H, W, dtype=input_img.dtype, device=input_img.device)
        yy = torch.arange(H, dtype=input_img.dtype, device=input_img.device)
        xx = torch.arange(W, dtype=input_img.dtype, device=input_img.device)
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')
        for _ in range(n):
            cy, cx = random.randint(0,H-1), random.randint(0,W-1)
            r = random.uniform(*self.radius_range)
            I = random.uniform(*self.intensity_range)
            synth[0] += torch.exp(-((gy-cy)**2+(gx-cx)**2)/(2*r*r)) * I
        synth = synth.clamp(0,1)
        input_img = (input_img + synth).clamp(0,1)
        gt_specular = (gt_specular + synth).clamp(0,1)
        return input_img, target_img, gt_specular


# ════════════════════════════════════════════════════════════════════════
# 8️⃣ Enhanced Paired Transform
# ════════════════════════════════════════════════════════════════════════
#
# 在 Y_FFC.py PairedTransform 基础上集成高反光模拟。
# 返回 4 项：input, target, gt_specular, gt_gradient。
# ════════════════════════════════════════════════════════════════════════

class EnhancedPairedTransform:
    def __init__(self, p_flip=0.5, p_rotate=0.3, p_crop=0.5,
                 crop_scale=(0.7,1.0), brightness_range=(0.9,1.1),
                 highlight_aug=None):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_crop = p_crop
        self.crop_scale = crop_scale
        self.brightness_range = brightness_range
        self.hl_aug = highlight_aug or HighlightAugmentation()

    def __call__(self, input_img, target_img):
        # 记录原始尺寸，保证增强后输出尺寸一致
        _, orig_h, orig_w = input_img.shape

        # 几何增强
        if random.random() < self.p_flip:
            input_img = torch.flip(input_img, [-1])
            target_img = torch.flip(target_img, [-1])
        if random.random() < self.p_flip:
            input_img = torch.flip(input_img, [-2])
            target_img = torch.flip(target_img, [-2])
        if random.random() < self.p_rotate:
            # 非正方形图像仅允许 180° 旋转（k=2），避免 90°/270° 导致的畸变
            if orig_h == orig_w:
                k = random.choice([1, 2, 3])
            else:
                k = 2
            input_img = torch.rot90(input_img, k, [1, 2])
            target_img = torch.rot90(target_img, k, [1, 2])
        if random.random() < self.p_crop:
            _, h, w = input_img.shape
            s = random.uniform(*self.crop_scale)
            nh, nw = int(h*s), int(w*s)
            if nh >= 32 and nw >= 32:
                t = random.randint(0, h-nh)
                l = random.randint(0, w-nw)
                input_img = input_img[:, t:t+nh, l:l+nw]
                target_img = target_img[:, t:t+nh, l:l+nw]
                input_img = F.interpolate(input_img.unsqueeze(0),
                    size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0)
                target_img = F.interpolate(target_img.unsqueeze(0),
                    size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0)

        bf = random.uniform(*self.brightness_range)
        input_img = (input_img * bf).clamp(0,1)

        gt_specular = (input_img - target_img).clamp(0,1)
        input_img, target_img, gt_specular = self.hl_aug(
            input_img, target_img, gt_specular)

        gx, gy = torch.gradient(target_img, dim=(1,2))
        mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
        flat = mag.flatten()
        p995 = torch.quantile(flat, 0.995) if flat.numel()>0 else torch.tensor(1.0)
        gt_gradient = (mag / (p995 + 1e-8)).clamp(0,1)

        return input_img, target_img, gt_specular, gt_gradient


# ════════════════════════════════════════════════════════════════════════
# 9️⃣ OHEM —— 在线难样本挖掘
# ════════════════════════════════════════════════════════════════════════
#
# 【思路】
#   1. 计算每个像素的逐点误差
#   2. 取误差最大的 top_k_ratio 比例像素
#   3. 只对这些"最难"像素额外计算加权损失
#   4. 最终 = 0.5×全局损失 + 0.5×难样本损失
# ════════════════════════════════════════════════════════════════════════

def specular_ohem_loss(pred, gt, base_fn, ratio=0.3):
    with torch.no_grad():
        err = torch.abs(pred - gt)
        B = err.shape[0]
        flat = err.view(B, -1)
        k = max(1, int(flat.shape[1] * ratio))
        _, idx = torch.topk(flat, k, dim=1)
        mask = torch.zeros_like(flat)
        mask.scatter_(1, idx, 1.0)
        mask = mask.view_as(err)
    base = base_fn(pred, gt)
    hard = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + 1e-8)
    return 0.5 * base + 0.5 * hard
