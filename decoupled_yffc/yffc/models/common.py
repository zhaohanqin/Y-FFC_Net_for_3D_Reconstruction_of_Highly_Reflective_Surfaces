"""Small shared helpers for model modules."""

import math

import torch


def truncated_normal_(tensor, mean=0.0, std=1.0):
    """
    截断正态分布初始化函数。
    生成截断在 [-2σ, 2σ] 范围内的正态分布随机数，避免极端值。
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def _tukey_window(size, alpha=0.25, device=None, dtype=None):
    """[P11] Tukey 窗：中间 75% 不衰减，两端 12.5% 余弦过渡。"""
    if size <= 1:
        return torch.ones(size, device=device, dtype=dtype or torch.float32)
    n = torch.arange(size, device=device, dtype=dtype or torch.float32)
    if alpha <= 0:
        return torch.ones(size, device=device, dtype=dtype or torch.float32)
    if alpha >= 1:
        return 0.5 * (1.0 - torch.cos(2.0 * math.pi * n / (size - 1)))
    width = int(alpha * (size - 1) / 2.0)
    w = torch.ones(size, device=device, dtype=dtype or torch.float32)
    left = n[:width + 1]
    w[:width + 1] = 0.5 * (1.0 + torch.cos(math.pi * (2.0 * left / (alpha * (size - 1)) - 1.0)))
    right = n[size - width - 1:]
    w[size - width - 1:] = 0.5 * (1.0 + torch.cos(math.pi * (2.0 * right / (alpha * (size - 1)) - 2.0 / alpha + 1.0)))
    return w
