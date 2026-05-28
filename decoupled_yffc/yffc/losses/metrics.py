"""Image quality metrics used by Y-FFC losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_tensor(tensor, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=None):
    """Numerical guard used only at high-risk boundaries.

    It prevents one unstable FFT/attention activation from turning the whole
    training run into persistent NaN/Inf while keeping normal finite values
    unchanged.
    """
    tensor = torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)
    if clamp is not None:
        tensor = tensor.clamp(clamp[0], clamp[1])
    return tensor

def gaussian_window(size, sigma):
    """用于 SSIM 计算的高斯窗口生成函数。"""
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)

class SSIM(nn.Module):
    """SSIM 结构相似性指标。[P12] as_loss=True 时不 clamp，保留梯度流。"""

    def __init__(self, window_size=11, sigma=1.5, as_loss=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.as_loss = as_loss
        base = gaussian_window(window_size, sigma).unsqueeze(0).unsqueeze(0)
        self.register_buffer("base_window", base)

    def forward(self, img1, img2):
        c = img1.size(1)
        window = self.base_window.to(dtype=img1.dtype, device=img1.device).repeat(c, 1, 1, 1)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=c)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=c)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 ** 2, window, padding=self.window_size // 2, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window, padding=self.window_size // 2, groups=c) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=c) - mu12

        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        eps = 1e-8

        denom = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)).clamp_min(eps)
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / denom
        ssim_map = _safe_tensor(ssim_map, nan=0.0, posinf=1.0, neginf=-0.5)
        # [P12-FIX] 训练时用 soft clamp 保护：允许梯度流但限制在合理范围
        # 原版 as_loss=True 完全不 clamp，可能出现 >1 的值使 1-ssim 变负
        if not self.as_loss:
            ssim_map = torch.clamp(ssim_map, 0.0, 1.0)
        else:
            # soft clamp：限制在 [-0.5, 1.0] 范围，保护梯度但阻止极端值
            ssim_map = torch.clamp(ssim_map, -0.5, 1.0)
        return ssim_map.mean()

class MSSSIM(nn.Module):
    """[P3] 多尺度 SSIM 损失：3 级降采样，加权融合对比度和结构分量。"""
    def __init__(self, window_size=11, sigma=1.5, n_scales=3, weights=None):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.n_scales = n_scales
        if weights is None:
            dw = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
            weights = dw[:n_scales]
            s = sum(weights)
            weights = [w_/s for w_ in weights]
        self.register_buffer("scale_weights", torch.tensor(weights, dtype=torch.float32))
        base = gaussian_window(window_size, sigma).unsqueeze(0).unsqueeze(0)
        self.register_buffer("base_window", base)

    def _ssim_components(self, img1, img2):
        c = img1.size(1)
        window = self.base_window.to(dtype=img1.dtype, device=img1.device).repeat(c,1,1,1)
        pad = self.window_size // 2
        mu1 = F.conv2d(img1, window, padding=pad, groups=c)
        mu2 = F.conv2d(img2, window, padding=pad, groups=c)
        mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1*mu2
        s1_sq = F.conv2d(img1**2, window, padding=pad, groups=c) - mu1_sq
        s2_sq = F.conv2d(img2**2, window, padding=pad, groups=c) - mu2_sq
        s12 = F.conv2d(img1*img2, window, padding=pad, groups=c) - mu12
        s1_sq = torch.clamp(s1_sq, min=0)
        s2_sq = torch.clamp(s2_sq, min=0)
        C1, C2 = 0.01**2, 0.03**2
        lum = (2 * mu12 + C1) / (mu1_sq + mu2_sq + C1).clamp_min(1e-8)
        cs = (2 * s12 + C2) / (s1_sq + s2_sq + C2).clamp_min(1e-8)
        lum = _safe_tensor(lum, nan=0.0, posinf=1.0, neginf=-0.5, clamp=(-0.5, 1.0))
        cs = _safe_tensor(cs, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))
        return lum.mean(), cs.mean()

    def forward(self, img1, img2):
        cs_list = []
        for i in range(self.n_scales):
            if i > 0:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
                if img1.shape[-1] < self.window_size or img1.shape[-2] < self.window_size:
                    break
            lum, cs = self._ssim_components(img1, img2)
            cs_list.append(cs)
        n_actual = len(cs_list)
        w = self.scale_weights[:n_actual].to(img1.device)
        w = w / w.sum()
        msssim = 1.0
        for i in range(n_actual):
            msssim = msssim * (torch.clamp(cs_list[i], min=1e-8) ** w[i])
        return msssim
