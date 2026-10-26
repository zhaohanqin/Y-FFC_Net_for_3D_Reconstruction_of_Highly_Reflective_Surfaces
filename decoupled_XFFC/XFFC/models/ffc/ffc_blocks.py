"""Higher-level Fast Fourier Convolution blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral import SpectralTransform


class FFC_BN_ACT(nn.Module):
    """
    FFC 模块 — 快速傅里叶卷积（带 BN 和激活）。

    核心思想：将输入特征分为局部(L)/全局(G)两个分支，四路交互：
        L2L (空间 Conv) + G2L (SpectralTransform) → 局部输出
        L2G (空间 Conv) + G2G (SpectralTransform) → 全局输出
    [F5] 跨域分支门控融合防止频域噪声污染空间分支。

    在整个模型中的位置：
        Encoder 的 ffc_bottleneck、Decoder 各级的 ffc_grad/ffc_h 均使用此模块。
        是 FFC 在网络中的统一入口。
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 activation_type='relu', **kwargs):
        super(FFC_BN_ACT, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        # 四个分支
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, **kwargs)

        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, **kwargs)

        module = nn.Identity if in_cg == 0 or out_cl == 0 else SpectralTransform
        self.convg2l = module(in_cg, out_cl, stride,
                              1 if groups == 1 else groups // 2,
                              enable_lfu, **kwargs)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride,
                              1 if groups == 1 else groups // 2,
                              enable_lfu, **kwargs)

        self.bnl = nn.BatchNorm2d(out_cl) if out_cl != 0 else nn.Identity()
        self.bng = nn.BatchNorm2d(out_cg) if out_cg != 0 else nn.Identity()

        if activation_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation_type == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Identity()

        # [F5] 跨域门控
        self._has_gate_l = (out_cl > 0 and in_cg > 0
                            and not isinstance(self.convg2l, nn.Identity))
        self._has_gate_g = (out_cg > 0 and in_cg > 0
                            and not isinstance(self.convg2g, nn.Identity))

        if self._has_gate_l:
            self.gate_l = nn.Sequential(
                nn.Conv2d(out_cl, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        if self._has_gate_g:
            self.gate_g = nn.Sequential(
                nn.Conv2d(out_cg, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, None)
        out_xl, out_xg = 0, 0

        if self.global_in_num > 0 and x_g is not None:
            # 局部输出
            out_l2l = self.convl2l(x_l) if not isinstance(self.convl2l, nn.Identity) else x_l
            out_g2l = self.convg2l(x_g) if not isinstance(self.convg2l, nn.Identity) else x_g

            if (isinstance(out_l2l, torch.Tensor) and isinstance(out_g2l, torch.Tensor)
                    and out_l2l.shape[-2:] != out_g2l.shape[-2:]):
                out_g2l = F.interpolate(out_g2l, size=out_l2l.shape[-2:],
                                        mode='bilinear', align_corners=False)

            if isinstance(out_l2l, torch.Tensor) and isinstance(out_g2l, torch.Tensor):
                if self._has_gate_l:
                    gate_l = self.gate_l(out_g2l)
                    out_xl = out_l2l + gate_l * out_g2l
                else:
                    out_xl = out_l2l + out_g2l
            else:
                out_xl = out_l2l

            # 全局输出
            out_l2g = self.convl2g(x_l) if not isinstance(self.convl2g, nn.Identity) else x_l
            out_g2g = self.convg2g(x_g) if not isinstance(self.convg2g, nn.Identity) else x_g

            if (isinstance(out_l2g, torch.Tensor) and isinstance(out_g2g, torch.Tensor)
                    and out_l2g.shape[-2:] != out_g2g.shape[-2:]):
                out_g2g = F.interpolate(out_g2g, size=out_l2g.shape[-2:],
                                        mode='bilinear', align_corners=False)

            if isinstance(out_l2g, torch.Tensor) and isinstance(out_g2g, torch.Tensor):
                if self._has_gate_g:
                    gate_g = self.gate_g(out_g2g)
                    out_xg = out_l2g + gate_g * out_g2g
                else:
                    out_xg = out_l2g + out_g2g
            else:
                out_xg = out_l2g
        else:
            out_xl = self.convl2l(x_l) if not isinstance(self.convl2l, nn.Identity) else x_l
            out_xg = self.convl2g(x_l) if not isinstance(self.convl2g, nn.Identity) else x_l

        out_xl = self.act(self.bnl(out_xl))
        out_xg = self.act(self.bng(out_xg))

        return out_xl, out_xg


class FFCResnetBlock(nn.Module):
    """
    FFC 残差块 — 用于堆叠多层 FFC。
    结构：输入 → FFC1 → FFC2 → 残差连接 → 输出
    """

    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU,
                 dilation=1, spatial_transform_kwargs=None, inline=False,
                 **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3,
                                 padding=dilation, dilation=dilation,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3,
                                 padding=dilation, dilation=dilation,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **conv_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l = x[:, :-self.conv1.global_in_num]
            x_g = x[:, -self.conv1.global_in_num:]
        else:
            x_l, x_g = x if isinstance(x, tuple) else (x, 0)

        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g

        if self.inline:
            out = torch.cat(out, dim=1)
        return out
