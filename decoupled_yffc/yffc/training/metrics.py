"""Metric accumulation for enhanced training."""

import math
from collections import defaultdict

import torch


SUB_LOSS_KEYS = [
    'total', 'ssim', 'ssim_single_loss', 'msssim_loss', 'ssim_mixed_loss',
    'l1', 'freq', 'perceptual',
    'spec', 'stat', 'grad',
    'focal', 'dice', 'gc', 'cs',
]

SUB_LOSS_DISPLAY_NAMES = {
    'total':      'Total Loss',
    'ssim':       'SSIM Loss (1-SSIM)',
    'ssim_single_loss': 'SSIM Single Loss',
    'msssim_loss': 'MS-SSIM Loss',
    'ssim_mixed_loss': 'SSIM Mixed Loss',
    'l1':         'L1 / MAE Loss',
    'freq':       'Frequency Loss',
    'perceptual': 'Perceptual Loss',
    'spec':       'Specular Loss',
    'stat':       'Statistical Loss',
    'grad':       'Gradient Loss',
    'focal':      'Focal Loss',
    'dice':       'Dice Loss',
    'gc':         'Grad Consistency',
    'cs':         'Cross-scale Consistency',
}


class MetricTracker:
    """
    统一指标计算器，供 train_one_epoch 和 validate 共用。
    避免训练/验证阶段指标计算逻辑不一致或重复代码。

    追踪指标：
      - SSIM（漫反射分支，1 - ssim_loss，越高越好）
      - MAE （漫反射分支，L1 距离）
      - MSE （漫反射分支，均方误差）
      - PSNR（漫反射分支，峰值信噪比，dB）
      - 全部子损失项（来自 EnhancedCompositeLoss 的 loss_dict）

    PSNR 公式（图像范围 [0, 1]）：
      PSNR = 10 × log10(1 / MSE)
    当 MSE 极小时（< 1e-10），PSNR 上限设为 60 dB。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.tot_loss = 0.0
        self.tot_ssim = 0.0
        self.tot_mae = 0.0
        self.tot_mse = 0.0
        self.tot_psnr = 0.0
        self.sub_accum = defaultdict(float)

    def update(self, loss_dict, pred, target):
        """
        更新一个 batch 的指标。

        参数:
            loss_dict : dict，EnhancedCompositeLoss 返回的 loss_dict
                        必须包含 'total' 和 'ssim'，可选包含 'l1'/'mae' 等
            pred      : Tensor，漫反射分支预测输出 (B, 1, H, W)
            target    : Tensor，漫反射分支真值 (B, 1, H, W)
        """
        self.n += 1

        # 主指标
        self.tot_loss += loss_dict['total']
        self.tot_ssim += (1.0 - loss_dict['ssim'])
        self.tot_mae += loss_dict.get('l1', loss_dict.get('mae', 0.0))

        # MSE（漫反射分支）
        mse_val = torch.mean((pred - target) ** 2).item()
        self.tot_mse += mse_val

        # PSNR = 10 × log10(1 / MSE)，图像范围 [0, 1]
        if mse_val > 1e-10:
            self.tot_psnr += 10.0 * math.log10(1.0 / mse_val)
        else:
            self.tot_psnr += 60.0  # MSE 极小时设上限

        # 子损失项
        for k, v in loss_dict.items():
            self.sub_accum[k] += v

    def compute(self):
        """
        计算 epoch 平均指标。

        返回:
            avg_loss  : float
            avg_ssim  : float（越高越好）
            avg_mae   : float
            avg_mse   : float
            avg_psnr  : float（dB）
            avg_subs  : dict，所有子损失项的平均值
        """
        d = max(self.n, 1)
        return (
            self.tot_loss / d,
            self.tot_ssim / d,
            self.tot_mae / d,
            self.tot_mse / d,
            self.tot_psnr / d,
            {k: v / d for k, v in self.sub_accum.items()},
        )
