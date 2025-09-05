import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def truncated_normal_(tensor, mean=0.0, std=1.0):
    """截断正态分布初始化"""
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

class SpectralTransform(nn.Module):
    """谱变换模块 - 傅里叶变换和逆变换"""
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **kwargs):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if groups == 1:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=1, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True)
            )
            self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **kwargs)
            if self.enable_lfu:
                self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
            self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=1, bias=False)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.fu = FourierUnit(out_channels, out_channels, groups, **kwargs)
            if self.enable_lfu:
                self.lfu = FourierUnit(out_channels, out_channels, groups)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = x[:, :, :split_s, :]
            xs2 = x[:, :, split_s:, :]
            xs = self.lfu(xs)
            xs2 = self.lfu(xs2)
            x = torch.cat([xs, xs2], dim=2)
        x = self.conv2(x + output)
        return x

class FourierUnit(nn.Module):
    """傅里叶单元"""
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                         out_channels=out_channels * 2,
                                         kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        if self.ffc3d:
            fft_dim = (-3, -2, -1)
        else:
            fft_dim = (-2, -1)

        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        if self.ffc3d:
            output = torch.fft.irfftn(ffted, s=x.shape[-3:], dim=fft_dim, norm=self.fft_norm)
        else:
            output = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output

class FFC_BN_ACT(nn.Module):
    """FFC with BatchNorm and Activation"""
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
        
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                             stride, padding, dilation, groups, bias, **kwargs)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                             stride, padding, dilation, groups, bias, **kwargs)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else SpectralTransform
        self.convg2l = module(in_cg, out_cl, stride, 1 if groups == 1 else groups // 2, enable_lfu, **kwargs)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **kwargs)
        
        self.bnl = nn.BatchNorm2d(out_cl) if not out_cl == 0 else nn.Identity()
        self.bng = nn.BatchNorm2d(out_cg) if not out_cg == 0 else nn.Identity()
        
        if activation_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation_type == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, None)
        out_xl, out_xg = 0, 0
        
        if self.global_in_num > 0 and x_g is not None:
            # 局部->局部 与 全局->局部
            out_l2l = self.convl2l(x_l) if not isinstance(self.convl2l, nn.Identity) else x_l
            out_g2l = self.convg2l(x_g) if not isinstance(self.convg2l, nn.Identity) else x_g
            if isinstance(out_l2l, torch.Tensor) and isinstance(out_g2l, torch.Tensor) and out_l2l.shape[-2:] != out_g2l.shape[-2:]:
                out_g2l = F.interpolate(out_g2l, size=out_l2l.shape[-2:], mode='bilinear', align_corners=False)
            out_xl = out_l2l + out_g2l if isinstance(out_l2l, torch.Tensor) and isinstance(out_g2l, torch.Tensor) else out_l2l

            # 局部->全局 与 全局->全局
            out_l2g = self.convl2g(x_l) if not isinstance(self.convl2g, nn.Identity) else x_l
            out_g2g = self.convg2g(x_g) if not isinstance(self.convg2g, nn.Identity) else x_g
            if isinstance(out_l2g, torch.Tensor) and isinstance(out_g2g, torch.Tensor) and out_l2g.shape[-2:] != out_g2g.shape[-2:]:
                out_g2g = F.interpolate(out_g2g, size=out_l2g.shape[-2:], mode='bilinear', align_corners=False)
            out_xg = out_l2g + out_g2g if isinstance(out_l2g, torch.Tensor) and isinstance(out_g2g, torch.Tensor) else out_l2g
        else:
            # 仅局部路径
            out_xl = self.convl2l(x_l) if not isinstance(self.convl2l, nn.Identity) else x_l
            out_xg = self.convl2g(x_l) if not isinstance(self.convl2g, nn.Identity) else x_l
        
        out_xl = self.act(self.bnl(out_xl))
        out_xg = self.act(self.bng(out_xg))
        
        return out_xl, out_xg

class FFCResnetBlock(nn.Module):
    """FFC ResNet Block"""
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                **conv_kwargs)
        if inline:
            self.inline = True
        else:
            self.inline = False

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.global_in_num], x[:, -self.conv1.global_in_num:]
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