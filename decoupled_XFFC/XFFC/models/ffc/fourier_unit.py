"""Fourier transform unit used by FFC blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import _safe_tensor, _tukey_window


class FourierUnit(nn.Module):
    """
    傅里叶单元 — FFC 的核心组件。

    工作流程：
        空间特征 → [F6] Hann 窗 → FFT → 频域 [real, imag] 拼接 →
        1×1 Conv → [F1] BN + GELU → [F2] SE 注意力 → IFFT → 空间域

    在整个模型中的位置：
        被 SpectralTransform 调用，SpectralTransform 被 FFC_BN_ACT 调用，
        FFC_BN_ACT 被 Encoder/Decoder/EnhancementLayers 中大量使用。
    """

    def __init__(self, in_channels, out_channels, groups=1,
                 spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None,
                 ffc3d=False, fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        self.groups = groups

        # 频域卷积层：输入通道×2（实部+虚部），可选+2（位置编码）
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
            out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)

        # [F1] GELU 替代 ReLU：不 hard-zero 负的虚部，保护相位信息
        self.act = nn.GELU()

        # [F2] SE 通道注意力：在频谱上自适应选择有用频段
        _se_mid = max(out_channels // 4, 1)
        self.freq_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, _se_mid, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(_se_mid, out_channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor,
                              mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)

        # AMP 兼容
        orig_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()

        # [F6][P11] Tukey 窗（α=0.25）：减少频谱泄漏，保留边缘 75% 能量
        h, w = x.shape[-2], x.shape[-1]
        win_h = _tukey_window(h, alpha=0.25, device=x.device, dtype=x.dtype)
        win_w = _tukey_window(w, alpha=0.25, device=x.device, dtype=x.dtype)
        tukey_2d = torch.outer(win_h, win_w).unsqueeze(0).unsqueeze(0)
        x_win = x * tukey_2d

        # 空间域 → 频域
        ffted = torch.fft.rfftn(x_win, dim=fft_dim, norm=self.fft_norm)

        # 复数分离 → 通道拼接
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # 可选频谱位置编码
        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(
                batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(
                batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        # 频域卷积 + [F1] BN + GELU
        ffted = self.conv_layer(ffted)
        ffted = self.act(self.bn(ffted))

        # [F2] 频率 SE 通道注意力
        ffted = ffted * self.freq_se(ffted)

        # 复数合并
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        if ffted.dtype == torch.float16:
            ffted = ffted.float()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        # 频域 → 空间域
        if self.ffc3d:
            output = torch.fft.irfftn(ffted, s=r_size[-3:], dim=fft_dim, norm=self.fft_norm)
        else:
            output = torch.fft.irfftn(ffted, s=r_size[-2:], dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size,
                                   mode=self.spatial_scale_mode, align_corners=False)

        output = _safe_tensor(output, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=(-1.0e4, 1.0e4))

        if orig_dtype == torch.float16:
            output = output.half()

        return output
