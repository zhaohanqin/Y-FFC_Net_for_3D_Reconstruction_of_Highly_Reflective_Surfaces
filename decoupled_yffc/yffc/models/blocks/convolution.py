"""Reusable convolution and upsampling blocks."""

import torch.nn as nn


class Conv2D_BN_activa(nn.Module):
    """通用卷积/转置卷积 + BN + 激活 组合块。被网络各处大量使用。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=0, dilation=1, if_bn=True, activation="relu",
                 bias=None, initializer=None, transpose=False):
        super(Conv2D_BN_activa, self).__init__()
        if transpose:
            self.conv2d = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation,
                bias=(not if_bn) if bias is None else bias)
        else:
            self.conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation,
                bias=(not if_bn) if bias is None else bias)

        self.if_bn = if_bn
        if self.if_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

        self.initializer = initializer
        if self.initializer is not None and self.initializer == "truncated_norm":
            nn.init.normal_(self.conv2d.weight, std=0.02)
            self.conv2d.weight = truncated_normal_(self.conv2d.weight, std=0.02)

    def forward(self, x):
        x = self.conv2d(x)
        if self.activation:
            if self.if_bn:
                x = self.bn(x)
            x = self.activation(x)
        return x


class MultiscaleConvBlock(nn.Module):
    """多尺度卷积块（三路并行：7×7 / 3×3 / 3×3+3×3），用于 MainDecoder 特征提取。"""

    def __init__(self, in_channels, out_channels_branch=192):
        super(MultiscaleConvBlock, self).__init__()
        self.path1 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 96, 1, 1, 0),
            Conv2D_BN_activa(96, out_channels_branch, 7, 1, 3),
        )
        self.path2 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.path3 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 256, 1, 1, 0),
            Conv2D_BN_activa(256, 256, 3, 1, 1),
            Conv2D_BN_activa(256, out_channels_branch, 3, 1, 1),
        )

    def forward(self, x):
        return torch.cat((self.path1(x), self.path2(x), self.path3(x)), 1)


class FeatureExtrationLayersB(nn.Module):
    """四路特征提取块，用于 MainDecoder 中间层。"""

    def __init__(self, in_channels):
        super(FeatureExtrationLayersB, self).__init__()
        self.path1 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 192, 7, 1, 3),
        )
        self.path2 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 192, 3, 1, 1),
        )
        self.path3 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 128, 3, 1, 1),
        )
        self.path4 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 128, 3, 1, 1),
            Conv2D_BN_activa(128, 128, 3, 1, 1),
        )

    def forward(self, x):
        return torch.cat((self.path1(x), self.path2(x),
                          self.path3(x), self.path4(x)), 1)


class TransposedConvBlock(nn.Module):
    """三路差异化 kernel 转置卷积（上采样）块。"""

    def __init__(self, in_channels, out_channels_branch=256):
        super(TransposedConvBlock, self).__init__()
        self.branch_1 = Conv2D_BN_activa(in_channels, out_channels_branch, 2, 2, 0, transpose=True)
        self.branch_2 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)
        self.branch_3 = Conv2D_BN_activa(in_channels, out_channels_branch, 6, 2, 2, transpose=True)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        if x3.shape[-2:] != x2.shape[-2:]:
            x3 = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x1, x2, x3], dim=1)


class TransposedConvBlock_same(nn.Module):
    """三路同尺寸卷积块（不改变空间分辨率）。"""

    def __init__(self, in_channels, out_channels_branch=256):
        super(TransposedConvBlock_same, self).__init__()
        self.branch_1 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.branch_2 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.branch_3 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)

    def forward(self, x):
        return torch.cat([self.branch_1(x), self.branch_2(x), self.branch_3(x)], dim=1)
