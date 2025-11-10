"""
FFC (Fast Fourier Convolution) 模块实现

核心思想：
    FFC将特征分为局部分支和全局分支两部分：
    - 局部分支：使用传统卷积处理，捕获局部空间特征
    - 全局分支：使用傅里叶变换处理，捕获全局频域特征
    
    这种设计能够同时利用空间域和频域信息，特别适合处理：
    1. 需要全局感受野的任务（如去除条纹、去噪等）
    2. 具有周期性模式的图像（如本项目中的条纹图像）
    3. 需要捕获长距离依赖关系的场景

主要模块：
    - FourierUnit: 傅里叶单元，执行FFT→卷积→IFFT操作
    - SpectralTransform: 谱变换模块，包含局部傅里叶单元(LFU)
    - FFC_BN_ACT: 完整的FFC模块，包含四个分支的特征交互
    - FFCResnetBlock: FFC残差块，用于堆叠多层FFC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def truncated_normal_(tensor, mean=0.0, std=1.0):
    """
    截断正态分布初始化函数
    
    功能：生成截断在[-2σ, 2σ]范围内的正态分布随机数
    优点：避免极端值，使训练更稳定
    
    参数:
        tensor: 要初始化的张量
        mean: 均值，默认0.0
        std: 标准差，默认1.0
    
    返回:
        初始化后的张量
    """
    size = tensor.shape
    # 生成4倍大小的正态分布样本，增加有效值的选择范围
    tmp = tensor.new_empty(size + (4,)).normal_()
    # 筛选在[-2σ, 2σ]范围内的值
    valid = (tmp < 2) & (tmp > -2)
    # 选择每个位置第一个有效值的索引
    ind = valid.max(-1, keepdim=True)[1]
    # 将有效值复制到目标张量
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    # 调整均值和标准差
    tensor.data.mul_(std).add_(mean)
    return tensor

class SpectralTransform(nn.Module):
    """
    谱变换模块 - 在频域中进行特征变换
    
    核心流程：
        输入 → 1x1卷积降维 → 傅里叶单元(全局) → (可选)局部傅里叶单元 → 1x1卷积升维 → 输出
    
    设计特点：
        1. 全局傅里叶单元(FU): 处理整个特征图，捕获全局频域信息
        2. 局部傅里叶单元(LFU): 将特征图分割后分别处理，捕获局部频域信息
        3. 残差连接: FU和LFU的输出相加，增强特征表达
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 步长（当前实现中未使用）
        groups: 分组卷积数，1表示普通卷积
        enable_lfu: 是否启用局部傅里叶单元
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **kwargs):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        
        # groups=1时，使用通道压缩策略（降维到一半）
        if groups == 1:
            # 前置1x1卷积：降维 + BN + ReLU
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=1, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True)
            )
            # 全局傅里叶单元：处理整个特征图
            self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **kwargs)
            # 局部傅里叶单元：处理分割后的特征图
            if self.enable_lfu:
                self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
            # 后置1x1卷积：恢复通道数
            self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=1, bias=False)
        else:
            # groups>1时，使用分组卷积
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
        """
        前向传播
        
        流程：
            1. 1x1卷积降维
            2. 全局傅里叶单元处理
            3. (可选)局部傅里叶单元处理：将特征图水平分割为两部分，分别处理后拼接
            4. 残差连接：FU输出 + LFU输出
            5. 1x1卷积恢复通道数
        
        参数:
            x: 输入特征 [B, C, H, W]
        
        返回:
            变换后的特征 [B, C', H, W]
        """
        # 前置卷积：降维并激活
        x = self.conv1(x)
        # 全局傅里叶变换
        output = self.fu(x)
        
        # 局部傅里叶变换（如果启用）
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2  # 分割数量
            split_s = h // split_no  # 分割点
            # 水平分割特征图为上下两部分
            xs = x[:, :, :split_s, :]   # 上半部分
            xs2 = x[:, :, split_s:, :]  # 下半部分
            # 分别进行局部傅里叶变换
            xs = self.lfu(xs)
            xs2 = self.lfu(xs2)
            # 拼接回完整特征图
            x = torch.cat([xs, xs2], dim=2)
        
        # 残差连接 + 后置卷积
        x = self.conv2(x + output)
        return x

class FourierUnit(nn.Module):
    """
    傅里叶单元 - FFC的核心组件
    
    工作原理：
        1. 将空间域特征通过FFT变换到频域
        2. 在频域中使用1x1卷积进行特征变换（全局感受野）
        3. 通过IFFT变换回空间域
    
    优势：
        - 全局感受野：FFT后每个频率分量都包含全局信息
        - 计算效率：FFT的复杂度为O(N log N)，比大卷积核更高效
        - 适合周期性模式：频域表示对周期性结构（如条纹）特别有效
    
    数学流程：
        输入x → FFT(x) → 实部/虚部分离 → Conv1x1 → BN+ReLU → 实部/虚部合并 → IFFT → 输出
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        groups: 分组卷积数
        spatial_scale_factor: 空间缩放因子（用于多尺度处理）
        spatial_scale_mode: 空间缩放模式
        spectral_pos_encoding: 是否添加频谱位置编码
        ffc3d: 是否使用3D FFT
        fft_norm: FFT归一化模式，'ortho'表示正交归一化
    """
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        self.groups = groups
        # 频域卷积层：输入通道×2（实部+虚部），可选+2（位置编码）
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
        """
        前向传播：在频域中处理特征
        
        详细步骤：
            1. 空间域 → 频域：使用实数FFT (rfftn) 将特征变换到频域
            2. 复数分离：将复数FFT结果分离为实部和虚部两个通道
            3. 频域卷积：使用1x1卷积在频域中进行特征变换（全局感受野）
            4. 复数合并：将实部和虚部重新组合为复数
            5. 频域 → 空间域：使用逆FFT (irfftn) 变换回空间域
        
        为什么在频域处理？
            - 全局感受野：频域中的每个点都包含整个空间域的信息
            - 高效处理周期性：条纹等周期性模式在频域中表现为离散的峰值
            - 计算效率：对于大尺寸特征图，FFT比大卷积核更高效
        
        参数:
            x: 输入特征 [B, C, H, W]
        
        返回:
            频域处理后的特征 [B, C', H, W]
        """
        batch = x.shape[0]
        
        # 可选：多尺度处理（缩放输入）
        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # 确定FFT的维度：2D或3D
        if self.ffc3d:
            fft_dim = (-3, -2, -1)  # 3D FFT
        else:
            fft_dim = (-2, -1)  # 2D FFT（高度和宽度维度）

        # 数据类型处理：避免ComplexHalf警告
        orig_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()
        
        # ==================== 步骤1：空间域 → 频域 ====================
        # 使用实数FFT（rfftn）：利用实数输入的对称性，只计算一半的频率
        # 输出形状：[B, C, H, W//2+1] (复数)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        
        # ==================== 步骤2：复数分离 ====================
        # 将复数分离为实部和虚部，便于卷积处理
        # 形状：[B, C, H, W//2+1] → [B, C, H, W//2+1, 2]
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        # 调整维度顺序：[B, C, 2, H, W//2+1]
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        # 展平实部和虚部到通道维度：[B, C*2, H, W//2+1]
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # 可选：添加频谱位置编码
        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            # 创建垂直和水平位置编码
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            # 拼接位置编码到特征：[B, C*2+2, H, W//2+1]
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        # ==================== 步骤3：频域卷积 ====================
        # 1x1卷积：在频域中进行特征变换（全局感受野）
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        # ==================== 步骤4：复数合并 ====================
        # 重新分离实部和虚部：[B, C'*2, H, W//2+1] → [B, C', 2, H, W//2+1]
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        # 数据类型处理
        if ffted.dtype == torch.float16:
            ffted = ffted.float()
        # 合并为复数：[B, C', H, W//2+1] (复数)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        # ==================== 步骤5：频域 → 空间域 ====================
        # 使用逆实数FFT（irfftn）变换回空间域
        if self.ffc3d:
            output = torch.fft.irfftn(ffted, s=x.shape[-3:], dim=fft_dim, norm=self.fft_norm)
        else:
            output = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=fft_dim, norm=self.fft_norm)

        # 恢复原始尺寸（如果进行了缩放）
        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        # 恢复原始数据类型
        if orig_dtype == torch.float16:
            output = output.half()

        return output

class FFC_BN_ACT(nn.Module):
    """
    FFC模块 - 快速傅里叶卷积（带BN和激活）
    
    核心思想：
        将输入特征分为两个分支：
        - 局部分支(Local): 使用传统卷积，捕获局部空间特征
        - 全局分支(Global): 使用傅里叶变换，捕获全局频域特征
    
    四分支交互机制：
        输入: (x_local, x_global)
        
        1. Local → Local (L2L): 传统卷积，局部特征提取
        2. Local → Global (L2G): 传统卷积，将局部特征转换为全局特征
        3. Global → Local (G2L): 傅里叶变换，将全局特征转换为局部特征
        4. Global → Global (G2G): 傅里叶变换，全局特征提取
        
        输出: (x_local_out, x_global_out)
            x_local_out = L2L + G2L  (融合局部和全局信息)
            x_global_out = L2G + G2G (融合局部和全局信息)
    
    优势：
        - 同时利用空间域和频域信息
        - 局部分支保留细节，全局分支捕获长距离依赖
        - 特别适合处理条纹图像等具有周期性模式的任务
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        ratio_gin: 输入全局分支的通道比例（0-1）
        ratio_gout: 输出全局分支的通道比例（0-1）
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组卷积数
        bias: 是否使用偏置
        enable_lfu: 是否启用局部傅里叶单元
        activation_type: 激活函数类型
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 activation_type='relu', **kwargs):
        super(FFC_BN_ACT, self).__init__()
        
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        
        # 计算局部和全局分支的通道数
        in_cg = int(in_channels * ratio_gin)    # 输入全局通道数
        in_cl = in_channels - in_cg              # 输入局部通道数
        out_cg = int(out_channels * ratio_gout)  # 输出全局通道数
        out_cl = out_channels - out_cg           # 输出局部通道数
        
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg
        
        # ==================== 四个分支的定义 ====================
        # 分支1: Local → Local (传统卷积)
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                             stride, padding, dilation, groups, bias, **kwargs)
        
        # 分支2: Local → Global (传统卷积)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                             stride, padding, dilation, groups, bias, **kwargs)
        
        # 分支3: Global → Local (傅里叶变换)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else SpectralTransform
        self.convg2l = module(in_cg, out_cl, stride, 1 if groups == 1 else groups // 2, enable_lfu, **kwargs)
        
        # 分支4: Global → Global (傅里叶变换)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **kwargs)
        
        # 批归一化层
        self.bnl = nn.BatchNorm2d(out_cl) if not out_cl == 0 else nn.Identity()
        self.bng = nn.BatchNorm2d(out_cg) if not out_cg == 0 else nn.Identity()
        
        # 激活函数
        if activation_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation_type == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        """
        前向传播：四分支特征交互
        
        处理流程：
            1. 输入分离：将输入分为局部和全局两部分
            2. 四分支处理：
               - L2L: 局部→局部 (传统卷积)
               - G2L: 全局→局部 (傅里叶变换)
               - L2G: 局部→全局 (传统卷积)
               - G2G: 全局→全局 (傅里叶变换)
            3. 特征融合：
               - 局部输出 = L2L + G2L
               - 全局输出 = L2G + G2G
            4. BN + 激活
        
        参数:
            x: 输入特征，可以是：
               - 单个张量 [B, C, H, W]：自动分割为局部和全局
               - 元组 (x_local, x_global)：已分割的局部和全局特征
        
        返回:
            (out_local, out_global): 局部和全局输出特征的元组
        """
        # 输入分离：如果是元组则直接使用，否则分割
        x_l, x_g = x if isinstance(x, tuple) else (x, None)
        out_xl, out_xg = 0, 0
        
        # 情况1：有全局输入，执行完整的四分支交互
        if self.global_in_num > 0 and x_g is not None:
            # ==================== 局部输出分支 ====================
            # L2L: 局部→局部 (传统卷积)
            out_l2l = self.convl2l(x_l) if not isinstance(self.convl2l, nn.Identity) else x_l
            # G2L: 全局→局部 (傅里叶变换)
            out_g2l = self.convg2l(x_g) if not isinstance(self.convg2l, nn.Identity) else x_g
            # 对齐空间尺寸（如果不同）
            if isinstance(out_l2l, torch.Tensor) and isinstance(out_g2l, torch.Tensor) and out_l2l.shape[-2:] != out_g2l.shape[-2:]:
                out_g2l = F.interpolate(out_g2l, size=out_l2l.shape[-2:], mode='bilinear', align_corners=False)
            # 融合：局部输出 = L2L + G2L
            out_xl = out_l2l + out_g2l if isinstance(out_l2l, torch.Tensor) and isinstance(out_g2l, torch.Tensor) else out_l2l

            # ==================== 全局输出分支 ====================
            # L2G: 局部→全局 (传统卷积)
            out_l2g = self.convl2g(x_l) if not isinstance(self.convl2g, nn.Identity) else x_l
            # G2G: 全局→全局 (傅里叶变换)
            out_g2g = self.convg2g(x_g) if not isinstance(self.convg2g, nn.Identity) else x_g
            # 对齐空间尺寸（如果不同）
            if isinstance(out_l2g, torch.Tensor) and isinstance(out_g2g, torch.Tensor) and out_l2g.shape[-2:] != out_g2g.shape[-2:]:
                out_g2g = F.interpolate(out_g2g, size=out_l2g.shape[-2:], mode='bilinear', align_corners=False)
            # 融合：全局输出 = L2G + G2G
            out_xg = out_l2g + out_g2g if isinstance(out_l2g, torch.Tensor) and isinstance(out_g2g, torch.Tensor) else out_l2g
        else:
            # 情况2：无全局输入，仅使用局部路径
            out_xl = self.convl2l(x_l) if not isinstance(self.convl2l, nn.Identity) else x_l
            out_xg = self.convl2g(x_l) if not isinstance(self.convl2g, nn.Identity) else x_l
        
        # BN + 激活
        out_xl = self.act(self.bnl(out_xl))
        out_xg = self.act(self.bng(out_xg))
        
        return out_xl, out_xg

class FFCResnetBlock(nn.Module):
    """
    FFC残差块 - 用于堆叠多层FFC
    
    结构：
        输入 → FFC1 → FFC2 → 残差连接 → 输出
        
        残差连接：
            out_local = input_local + FFC2(FFC1(input_local))
            out_global = input_global + FFC2(FFC1(input_global))
    
    优势：
        - 残差连接：缓解梯度消失，支持更深的网络
        - 双层FFC：增强特征表达能力
        - 保持局部-全局分离：持续利用双分支优势
    
    参数:
        dim: 特征维度（通道数）
        padding_type: 填充类型
        norm_layer: 归一化层类型
        activation_layer: 激活函数类型
        dilation: 膨胀率
        spatial_transform_kwargs: 空间变换参数
        inline: 是否使用内联模式（将局部和全局拼接在通道维度）
        conv_kwargs: 其他卷积参数
    """
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        # 第一个FFC层
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                **conv_kwargs)
        # 第二个FFC层
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                **conv_kwargs)
        if inline:
            self.inline = True
        else:
            self.inline = False

    def forward(self, x):
        """
        前向传播：双层FFC + 残差连接
        
        流程：
            1. 输入分离（如果使用inline模式）
            2. 保存输入作为残差
            3. 通过两层FFC
            4. 添加残差连接
            5. 输出（如果使用inline模式则拼接）
        
        参数:
            x: 输入特征，格式取决于inline模式
        
        返回:
            输出特征，格式与输入相同
        """
        # 输入处理：根据inline模式分离局部和全局
        if self.inline:
            # inline模式：从通道维度分离
            x_l, x_g = x[:, :-self.conv1.global_in_num], x[:, -self.conv1.global_in_num:]
        else:
            # 标准模式：元组输入或单个张量
            x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        
        # 保存输入作为残差
        id_l, id_g = x_l, x_g
        
        # 通过两层FFC
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        
        # 残差连接
        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        
        # 输出处理：根据inline模式拼接或返回元组
        if self.inline:
            out = torch.cat(out, dim=1)
        return out 