"""Specular-aware auxiliary losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import _safe_tensor


class SpecularFocalLoss(nn.Module):
    """高光像素 Focal 加权损失：对难预测的高亮像素给予更大权重。"""

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

class SpecularDiceLoss(nn.Module):
    """高光区域 Dice 损失：从分割视角约束高光图的前景重叠度。"""

    def __init__(self, threshold=0.05, smooth=1.0):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, pred, gt):
        pred_p = torch.sigmoid(20.0 * (pred - self.threshold))
        gt_m = (gt > self.threshold).float()
        inter = (pred_p * gt_m).sum(dim=(1, 2, 3))
        union = pred_p.sum(dim=(1, 2, 3)) + gt_m.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()

class GradientConsistencyLoss(nn.Module):
    """Specular 边界梯度一致性损失：约束预测与 GT 的空间梯度幅值和方向一致。"""

    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sx.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sx.t().unsqueeze(0).unsqueeze(0))

    def forward(self, pred, gt):
        """[SAFE] 反向稳定的梯度一致性损失。

        原版 eps=1e-8 在 fp32 下也接近下溢边缘；改为 1e-4 大大提升反向稳定性，
        同时对损失的精度影响可忽略（典型梯度幅值在 1e-2~1.0 量级）。
        """
        sx = self.sobel_x.to(pred.dtype)
        sy = self.sobel_y.to(pred.dtype)

        pgx = F.conv2d(pred, sx, padding=1)
        pgy = F.conv2d(pred, sy, padding=1)
        ggx = F.conv2d(gt, sx, padding=1)
        ggy = F.conv2d(gt, sy, padding=1)

        # [SAFE] eps=1e-4：sqrt(0+1e-4)=0.01，远离反向爆炸点
        pm = torch.sqrt(pgx ** 2 + pgy ** 2 + 1e-4)
        gm = torch.sqrt(ggx ** 2 + ggy ** 2 + 1e-4)

        mag_loss = F.l1_loss(pm, gm)

        # [SAFE] 用 detach mask + 较大下界，反向只走 cos 数值通道
        with torch.no_grad():
            mask = (gm > gm.mean() * 0.5).float()
            mask_sum = mask.sum().clamp_min(1.0)
        cos = (pgx * ggx + pgy * ggy) / (pm * gm + 1e-4)
        cos = _safe_tensor(cos, nan=1.0, posinf=1.0, neginf=-1.0, clamp=(-1.0, 1.0))
        dir_loss = ((1.0 - cos) * mask).sum() / mask_sum

        return mag_loss + 0.3 * dir_loss

def specular_ohem_loss(pred, gt, base_fn, ratio=0.3):
    """在线难样本挖掘（OHEM）：只对误差最大的 top-k 像素额外加权。

    [SAFE] mask.sum() 实际至少为 k = ratio*H*W (远大于 1)，
    但保持显式 clamp_min(1.0) 防御性下界，避免极小 patch 上的数值病态。
    """
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
    hard = (torch.abs(pred - gt) * mask).sum() / mask.sum().clamp_min(1.0)
    return 0.5 * base + 0.5 * hard
