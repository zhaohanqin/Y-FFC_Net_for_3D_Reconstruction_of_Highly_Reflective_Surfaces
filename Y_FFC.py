import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器
from torchvision import models  # 预训练模型
import os  # 操作系统接口
from PIL import Image  # 图像处理
import numpy as np  # 数值计算
import argparse  # 命令行参数解析
import sys  # 系统相关功能

# 导入FFC（Fourier Feature Convolution）模块
from ffc import *

def truncated_normal_(tensor, mean=0.0, std=1.0):
    """
    截断正态分布初始化函数
    
    参数:
        tensor: 要初始化的张量
        mean: 均值，默认0.0
        std: 标准差，默认1.0
    
    返回:
        初始化后的张量
    """
    size = tensor.shape
    # 生成4倍大小的正态分布随机数
    tmp = tensor.new_empty(size + (4,)).normal_()
    # 筛选在[-2, 2]范围内的值
    valid = (tmp < 2) & (tmp > -2)
    # 选择第一个有效值的索引
    ind = valid.max(-1, keepdim=True)[1]
    # 复制有效值到目标张量
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    # 调整均值和标准差
    tensor.data.mul_(std).add_(mean)
    return tensor

# ==================== SSIM损失函数相关 ====================
def gaussian_window(size, sigma):
    """
    生成2D高斯窗口，用于SSIM计算
    
    参数:
        size: 窗口大小
        sigma: 高斯函数的标准差
    
    返回:
        2D高斯窗口张量，形状为 [1, 1, size, size]
    """
    coords = torch.arange(size, dtype=torch.float32)  # 创建坐标序列
    coords -= size // 2  # 将坐标中心化到0
    # 计算1D高斯函数
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()  # 归一化
    # 扩展为2D窗口
    return g.unsqueeze(0) * g.unsqueeze(1)

class SSIM(nn.Module):
    """
    结构相似性指数（Structural Similarity Index）损失函数
    
    用于衡量两个图像之间的结构相似性，值越大表示越相似
    基于亮度、对比度和结构三个方面的比较
    """
    def __init__(self, window_size=11, sigma=1.5):
        """
        初始化SSIM模块
        
        参数:
            window_size: 高斯窗口大小，默认11
            sigma: 高斯函数标准差，默认1.5
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        # 创建高斯窗口并注册为缓冲区（不参与梯度更新）
        base = gaussian_window(window_size, sigma).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
        self.register_buffer('base_window', base)

    def forward(self, img1, img2):
        """
        计算两个图像的SSIM值
        
        参数:
            img1, img2: 输入图像，形状为 [B,C,H,W]
        
        返回:
            SSIM值，标量
        """
        c = img1.size(1)  # 获取通道数
        # 将高斯窗口扩展到所有通道
        window = self.base_window.to(dtype=img1.dtype, device=img1.device).repeat(c, 1, 1, 1)
        
        # 计算均值（亮度）
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=c)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=c)
        
        # 计算均值平方
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2  # 交叉均值
        
        # 计算方差（对比度）
        sigma1_sq = F.conv2d(img1**2, window, padding=self.window_size//2, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(img2**2, window, padding=self.window_size//2, groups=c) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=c) - mu12  # 协方差
        
        # SSIM常数，防止除零
        C1 = (0.01)**2
        C2 = (0.03)**2
        
        # 计算SSIM映射
        ssim_map = ((2*mu12 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()  # 返回平均SSIM值

def statistical_loss(pred, gt):
    """
    统计损失函数：确保预测结果与真实值在统计特性上保持一致
    
    参数:
        pred: 预测值
        gt: 真实值
    
    返回:
        统计损失值（均值差异 + 标准差差异）
    """
    # 计算均值差异（亮度一致性）
    mean_loss = torch.abs(pred.mean() - gt.mean())
    # 计算标准差差异（对比度一致性）
    std_loss = torch.abs(pred.std() - gt.std())
    return mean_loss + std_loss

class Conv2D_BN_activa(nn.Module):
    """
    通用卷积模块：Conv2d/ConvTranspose2d + BatchNorm2d(可选) + 激活(可选)。

    - 通过 `transpose` 控制是否为转置卷积，实现上采样或保持/降采样。
    - 通过 `if_bn`、`activation` 控制是否插入 BN 与激活，常用于快速搭建网络骨干。

    输入/输出:
        输入: [B, in_channels, H, W]
        输出: [B, out_channels, H', W'] 由卷积参数决定。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 dilation=1, if_bn=True, activation='relu', bias=None, initializer=None, transpose=False):
        super(Conv2D_BN_activa, self).__init__()
        if transpose:
            self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                            dilation=dilation, bias=(not if_bn) if bias is None else bias)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                   dilation=dilation, bias=(not if_bn) if bias is None else bias)
        self.if_bn = if_bn
        if self.if_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None
        self.initializer = initializer
        if self.initializer is not None:
            if self.initializer == 'truncated_norm':
                nn.init.normal_(self.conv2d.weight, std=0.02)
                self.conv2d.weight = truncated_normal_(self.conv2d.weight, std=0.02)

    def forward(self, x):
        """
        前向传播：卷积 → (可选)BN → (可选)激活。

        参数:
            x: 输入特征图 [B, C, H, W]

        返回:
            处理后的特征图，形状取决于卷积/转置卷积设置。
        """
        x = self.conv2d(x)
        if self.activation:
            if self.if_bn:
                x = self.bn(x)
            x = self.activation(x)
        return x

class Attention_block(nn.Module):
    """
    注意力块：实现门控注意力机制
    
    功能：根据全局特征g的信息，对局部特征x进行加权，实现自适应特征融合
    设计灵感来自U-Net++的注意力门控机制
    """
    def __init__(self, F_g, F_l, F_int):
        """
        初始化注意力块
        
        参数:
            F_g: 全局特征通道数（来自深层特征）
            F_l: 局部特征通道数（来自浅层特征）
            F_int: 中间特征通道数
        """
        super(Attention_block, self).__init__()
        
        # 全局特征处理分支：F_g → F_int
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # 局部特征处理分支：F_l → F_int
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # 注意力权重生成：F_int → 1，输出0-1之间的权重
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # 确保权重在[0,1]范围内
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        前向传播
        
        参数:
            g: 全局特征（深层特征）
            x: 局部特征（浅层特征）
        
        返回:
            加权后的局部特征
        """
        # 将全局和局部特征映射到相同的特征空间
        g1 = self.W_g(g)  # 全局特征变换
        x1 = self.W_x(x)  # 局部特征变换
        
        # 融合特征并生成注意力权重
        psi = self.relu(g1 + x1)  # 特征融合
        psi = self.psi(psi)       # 生成注意力权重
        
        # 应用注意力权重到局部特征
        return x * psi  # 加权特征

class MultiscaleConvBlock(nn.Module):
    """
    多尺度卷积块：并行处理不同感受野的特征
    
    设计理念：同时使用不同大小的卷积核捕获不同尺度的特征
    三个并行分支：7x7大感受野、3x3中等感受野、深层3x3小感受野
    """
    def __init__(self, in_channels, out_channels_branch=192):
        """
        初始化多尺度卷积块
        
        参数:
            in_channels: 输入通道数
            out_channels_branch: 每个分支的输出通道数，默认192
        """
        super(MultiscaleConvBlock, self).__init__()
        
        # 路径1：1x1降维 + 7x7大感受野
        # 适合捕获大范围的空间依赖关系
        self.path1 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 96, 1, 1, 0),           # 1x1降维
            Conv2D_BN_activa(96, out_channels_branch, 7, 1, 3)     # 7x7大感受野
        )
        
        # 路径2：3x3中等感受野
        # 平衡计算效率和感受野大小
        self.path2 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        
        # 路径3：1x1降维 + 3x3 + 3x3深层处理
        # 通过多层卷积提取更抽象的特征
        self.path3 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 256, 1, 1, 0),          # 1x1降维
            Conv2D_BN_activa(256, 256, 3, 1, 1),                   # 3x3卷积
            Conv2D_BN_activa(256, out_channels_branch, 3, 1, 1)    # 3x3卷积
        )

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征，形状 [B, C, H, W]
        
        返回:
            多尺度特征融合结果，形状 [B, 3*out_channels_branch, H, W]
        """
        # 并行处理三个分支
        x1 = self.path1(x)  # 大感受野特征
        x2 = self.path2(x)  # 中等感受野特征
        x3 = self.path3(x)  # 深层特征
        
        # 在通道维度上拼接三个分支的输出
        out = torch.cat((x1, x2, x3), 1)
        return out

class FeatureExtrationLayersB(nn.Module):
    """
    特征提取层B：四分支并行特征提取模块
    
    设计理念：通过不同深度的卷积路径提取多尺度特征
    四个分支分别使用不同数量的卷积层，捕获不同抽象级别的特征
    """
    def __init__(self, in_channels):
        """
        初始化特征提取层B
        
        参数:
            in_channels: 输入通道数
        """
        super(FeatureExtrationLayersB, self).__init__()
        
        # 路径1：1x1降维 + 7x7大感受野
        # 适合捕获大范围的空间依赖关系
        self.path1 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),          # 1x1降维
            Conv2D_BN_activa(128, 192, 7, 1, 3)                    # 7x7大感受野
        )
        
        # 路径2：1x1降维 + 3x3中等感受野
        # 平衡计算效率和感受野大小
        self.path2 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),          # 1x1降维
            Conv2D_BN_activa(128, 192, 3, 1, 1)                    # 3x3卷积
        )
        
        # 路径3：1x1降维 + 3x3卷积
        # 标准卷积路径
        self.path3 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),          # 1x1降维
            Conv2D_BN_activa(128, 128, 3, 1, 1)                    # 3x3卷积
        )
        
        # 路径4：1x1降维 + 3x3 + 3x3深层处理
        # 通过多层卷积提取更抽象的特征
        self.path4 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),          # 1x1降维
            Conv2D_BN_activa(128, 128, 3, 1, 1),                   # 3x3卷积
            Conv2D_BN_activa(128, 128, 3, 1, 1)                    # 3x3卷积
        )

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征，形状 [B, C, H, W]
        
        返回:
            多尺度特征融合结果，形状 [B, 640, H, W]
        """
        # 并行处理四个分支
        path1 = self.path1(x)  # 大感受野特征 (192通道)
        path2 = self.path2(x)  # 中等感受野特征 (192通道)
        path3 = self.path3(x)  # 标准特征 (128通道)
        path4 = self.path4(x)  # 深层特征 (128通道)
        
        # 在通道维度上拼接四个分支的输出
        # 总通道数: 192 + 192 + 128 + 128 = 640
        out = torch.cat((path1, path2, path3, path4), 1)
        return out

class TransposedConvBlock(nn.Module):
    """
    转置卷积块：三分支并行上采样模块。

    功能：将特征图分辨率放大 2 倍，同时通过三分支拼接提升通道表达能力。

    输入/输出：
        输入: [B, in_channels, H, W]
        输出: [B, 3*out_channels_branch, 2H, 2W]
    """
    def __init__(self, in_channels, out_channels_branch=256):
        """
        初始化转置卷积块
        
        参数:
            in_channels: 输入通道数
            out_channels_branch: 每个分支的输出通道数，默认256
        """
        super(TransposedConvBlock, self).__init__()
        
        # 三个并行分支，每个都使用4x4转置卷积进行2倍上采样
        # 4x4卷积核，stride=2，padding=1，输出尺寸为输入的2倍
        self.branch_1 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)
        self.branch_2 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)
        self.branch_3 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)

    def forward(self, x):
        """
        前向传播：三分支 2× 上采样并在通道维拼接。

        参数:
            x: 输入特征 [B, C, H, W]

        返回:
            [B, 3*out_channels_branch, 2H, 2W]
        """
        # 并行处理三个分支
        x_1 = self.branch_1(x)  # 分支1上采样
        x_2 = self.branch_2(x)  # 分支2上采样
        x_3 = self.branch_3(x)  # 分支3上采样
        
        # 在通道维度上拼接三个分支的输出
        # 输出通道数 = 3 * out_channels_branch
        x_123 = torch.cat([x_1, x_2, x_3], dim=1)
        return x_123

class TransposedConvBlock_same(nn.Module):
    """
    三分支并行特征处理模块（保持分辨率不变）。

    功能：通过三个 3×3 卷积分支提取互补特征后拼接，增强通道表达；
    与 TransposedConvBlock 不同，此模块不改变空间尺寸。
    """
    def __init__(self, in_channels, out_channels_branch=256):
        """
        初始化转置卷积块（保持尺寸）
        
        参数:
            in_channels: 输入通道数
            out_channels_branch: 每个分支的输出通道数，默认256
        """
        super(TransposedConvBlock_same, self).__init__()
        
        # 三个并行分支，每个都使用3x3卷积保持尺寸不变
        # 3x3卷积核，stride=1，padding=1，输出尺寸与输入相同
        self.branch_1 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.branch_2 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.branch_3 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)

    def forward(self, x):
        """
        参数:
            x: 输入特征 [B, C, H, W]

        返回:
            拼接后的特征 [B, 3*out_channels_branch, H, W]
        """
        # 并行处理三个分支
        x_1 = self.branch_1(x)  # 分支1特征变换
        x_2 = self.branch_2(x)  # 分支2特征变换
        x_3 = self.branch_3(x)  # 分支3特征变换
        
        # 在通道维度上拼接三个分支的输出
        # 输出通道数 = 3 * out_channels_branch，空间尺寸保持不变
        x_123 = torch.cat([x_1, x_2, x_3], dim=1)
        return x_123

class Encoder(nn.Module):
    """
    编码器：基于VGG16-BN的特征提取网络
    
    负责从输入图像中提取多尺度特征，为后续解码器提供丰富的特征信息
    采用跳跃连接设计，保留不同层级的特征细节。
    
    输入/输出：
        输入:  单通道灰度图 [B, 1, H, W]
        输出:  (x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c)
               其中分辨率依次为 1/2, 1/4, 1/8, 1/16(或1/32)；x_c 为 256 通道瓶颈特征。
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # 使用预训练的VGG16-BN作为backbone，利用ImageNet预训练权重
        backbone_model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        backbone_model_list = list(backbone_model.features.children())
        
        # 5层结构定义，但仅使用前4层作为跳连，x_c来自第4层输出
        # 每层输出不同的特征尺度和通道数
        self.backbone_1 = nn.Sequential(*backbone_model_list[0:7])   # 输出: 64通道, 1/2分辨率
        self.backbone_2 = nn.Sequential(*backbone_model_list[7:14])  # 输出: 128通道, 1/4分辨率
        self.backbone_3 = nn.Sequential(*backbone_model_list[14:24]) # 输出: 256通道, 1/8分辨率
        self.backbone_4 = nn.Sequential(*backbone_model_list[24:34]) # 输出: 512通道, 1/16->1/32分辨率（包含池化层）
        self.backbone_5 = nn.Sequential(*backbone_model_list[34:44]) # 未直接使用，仅保留
        
        # 输入适配层：将单通道输入转换为3通道（适配VGG16）
        self.conv1 = Conv2D_BN_activa(1, 3, 3, 1, 1)
        # 瓶颈层：将512通道压缩到256通道，减少计算量
        self.cba_after_backbone = Conv2D_BN_activa(512, 256, 3, 1, 1)

    def forward(self, x):
        """
        参数:
            x: 输入图像 [B, 1, H, W]

        返回:
            x_bb_1..x_bb_4: VGG 各阶段多尺度特征
            x_c: 瓶颈变换后的 256 通道特征
        """
        x1 = self.conv1(x)
        x_bb_1 = self.backbone_1(x1)      # 1/2
        x_bb_2 = self.backbone_2(x_bb_1)  # 1/4
        x_bb_3 = self.backbone_3(x_bb_2)  # 1/8
        x_bb_4 = self.backbone_4(x_bb_3)  # 1/16 -> 1/32 取决于切片是否含pool（与原实现一致）
        x_c = self.cba_after_backbone(x_bb_4)
        
        return x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c

class GradientDecoder(nn.Module):
    """
    梯度解码器：从深层特征逐步上采样生成梯度图
    
    功能：将编码器的深层特征逐步上采样到原始分辨率，生成梯度信息
    采用跳跃连接设计，融合不同层级的特征信息
    """
    def __init__(self):
        super(GradientDecoder, self).__init__()
        
        # ==================== 第一层：特征提炼 ====================
        # 对应原实现：7x7/1x1/3x3 提炼 + 特征变换
        self.block_1_conv_1 = Conv2D_BN_activa(512, 1024, 7, 1, 3)  # 7x7大感受野特征提取
        self.block_1_conv_2 = Conv2D_BN_activa(1024, 512, 1, 1, 0)  # 1x1降维
        self.block_1_conv_3 = Conv2D_BN_activa(512, 256, 3, 1, 1)   # 3x3特征精炼
        
        # 注意：第一"transpose"块在原代码中并不改变分辨率（5x5, stride=1）
        # 这里只是特征变换，不进行上采样
        self.block_1_conv_transpose = Conv2D_BN_activa(256, 256, 5, 1, 2, transpose=False)

        # ==================== 第二层：第一次上采样 ====================
        # 输入：256(提炼特征) + 512(跳跃连接) = 768通道
        self.block_2_conv = Conv2D_BN_activa(256 + 512, 128, 3, 1, 1)
        self.block_2_conv_transpose = Conv2D_BN_activa(128, 128, 4, 2, 1, transpose=True)  # ×2上采样

        # ==================== 第三层：第二次上采样 ====================
        # 输入：128(上采样特征) + 256(跳跃连接) = 384通道
        self.block_3_conv = Conv2D_BN_activa(128 + 256, 64, 3, 1, 1)
        self.block_3_conv_transpose = Conv2D_BN_activa(64, 64, 4, 2, 1, transpose=True)    # ×2上采样

        # ==================== 第四层：第三次上采样 ====================
        # 输入：64(上采样特征) + 128(跳跃连接) = 192通道
        self.block_4_conv = Conv2D_BN_activa(64 + 128, 32, 3, 1, 1)
        self.block_4_conv_transpose = Conv2D_BN_activa(32, 32, 4, 2, 1, transpose=True)    # ×2上采样

        # ==================== 第五层：最终上采样到输入分辨率 ====================
        # 输入：32(上采样特征) + 64(跳跃连接) = 96通道
        self.block_5_conv_1 = Conv2D_BN_activa(32 + 64, 64, 4, 2, 1, transpose=True)       # ×2上采样
        self.block_5_conv_2 = Conv2D_BN_activa(64, 1, 5, 1, 2, activation=None, if_bn=False)  # 输出层
        self.sigmoid_layer = nn.Sigmoid()  # 将输出限制在[0,1]范围内

    def forward(self, backbone_features):
        """
        前向传播：逐步上采样生成梯度图
        
        参数:
            backbone_features: 编码器的多尺度特征 [x_bb_1, x_bb_2, x_bb_3, x_bb_4]
                - x_bb_1: 1/2分辨率, 64通道
                - x_bb_2: 1/4分辨率, 128通道  
                - x_bb_3: 1/8分辨率, 256通道
                - x_bb_4: 1/16分辨率, 512通道
        
        返回:
            x_to_be_enhanced_1..4: 中间增强特征，用于主解码器
            x_g: 最终梯度图，与输入同分辨率
        """
        x_bb_1, x_bb_2, x_bb_3, x_bb_4 = backbone_features
        
        # ==================== 第一层：特征提炼（不改变分辨率） ====================
        x = self.block_1_conv_1(x_bb_4)      # 7x7大感受野特征提取
        x = self.block_1_conv_2(x)            # 1x1降维
        x = self.block_1_conv_3(x)            # 3x3特征精炼
        x_to_be_enhanced_1 = self.block_1_conv_transpose(x)  # 与 x_bb_4 同分辨率
        x = torch.cat([x_to_be_enhanced_1, x_bb_4], dim=1)  # 跳跃连接融合

        # ==================== 第二层：第一次上采样 ====================
        x = self.block_2_conv(x)              # 3x3卷积处理融合特征
        x_to_be_enhanced_2 = self.block_2_conv_transpose(x)  # ×2上采样，与 x_bb_3 对齐
        x = torch.cat([x_to_be_enhanced_2, x_bb_3], dim=1)  # 跳跃连接融合

        # ==================== 第三层：第二次上采样 ====================
        x = self.block_3_conv(x)              # 3x3卷积处理融合特征
        x_to_be_enhanced_3 = self.block_3_conv_transpose(x)  # ×2上采样，与 x_bb_2 对齐
        x = torch.cat([x_to_be_enhanced_3, x_bb_2], dim=1)  # 跳跃连接融合

        # ==================== 第四层：第三次上采样 ====================
        x = self.block_4_conv(x)              # 3x3卷积处理融合特征
        x_to_be_enhanced_4 = self.block_4_conv_transpose(x)  # ×2上采样，与 x_bb_1 对齐
        x = torch.cat([x_to_be_enhanced_4, x_bb_1], dim=1)  # 跳跃连接融合

        # ==================== 第五层：最终上采样到输入分辨率 ====================
        x = self.block_5_conv_1(x)            # ×2上采样到原始分辨率
        x = self.block_5_conv_2(x)            # 输出层，生成单通道特征
        x_g = self.sigmoid_layer(x)           # Sigmoid激活，输出范围[0,1]
        
        return x_to_be_enhanced_1, x_to_be_enhanced_2, x_to_be_enhanced_3, x_to_be_enhanced_4, x_g

class MainDecoder(nn.Module):
    """
    主解码器：融合编码器多尺度特征与梯度分支增强特征，预测镜面反射分量 b。

    结构特点：
        - 两个 FFC 模块对瓶颈特征进行频域增强并残差连接
        - 多尺度卷积后进入五层级联上采样模块
        - 四个注意力门控桥接，与编码器各层特征对齐并融合
        - 与梯度分支的四级增强特征逐层级联，末端再与梯度图 x_g 融合输出

    前向输入：
        c_features: (x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c)
        enhanced_features: (x_enhanced_1..4)
        x_g: 梯度图 [B, 1, H, W]

    前向输出：
        est_b: 镜面反射分量 [B, 1, H, W]
    """
    def __init__(self, in_channels):
        super(MainDecoder, self).__init__()
        # FFC模块（使用 padding=1 保持空间尺寸）
        self.FFC1 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)
        self.FFC2 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)
        
        # 多尺度卷积块
        self.multiscale_conv = MultiscaleConvBlock(in_channels)
        
        # 5层转置卷积块
        self.transposed_block_1 = TransposedConvBlock_same(192 * 3, 256)  # in: 768 -> out: 256*3
        self.transposed_block_2 = TransposedConvBlock(256 * 3 + 128 + 512, 128)  # in: 1408 -> out: 128*3
        # FeatureExtrationLayersB 输出通道为 192+192+128+128 = 640
        self.feature_extraction_B = FeatureExtrationLayersB(128 * 3 + 64 + 256)  # in: 704
        self.transposed_block_3 = TransposedConvBlock(640, 64)  # 修正：输入640通道
        self.transposed_block_4 = TransposedConvBlock(64 * 3 + 32 + 128, 32)  # in: 352 -> out: 32*3
        self.transposed_block_5 = TransposedConvBlock(32 * 3 + 16 + 64, 16)   # in: 176 -> out: 16*3
        
        # 注意力块
        self.att1 = Attention_block(768, 512, 640)
        self.att2 = Attention_block(384, 256, 320)
        self.att3 = Attention_block(192, 128, 160)
        self.att4 = Attention_block(96, 64, 80)
        
        # 输出层
        self.output_conv1 = Conv2D_BN_activa(16 * 3 + 1, 16, 3, 1, 1)
        self.output_conv2 = Conv2D_BN_activa(16, 1, 3, 1, 1)

    def forward(self, c_features, enhanced_features, x_g):
        """
        参数:
            c_features: (x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c)
            enhanced_features: (x_enhanced_1, x_enhanced_2, x_enhanced_3, x_enhanced_4)
            x_g: 梯度图 [B, 1, H, W]

        返回:
            est_b: 镜面反射分量 [B, 1, H, W]
        """
        x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c = c_features
        x_enhanced_1, x_enhanced_2, x_enhanced_3, x_enhanced_4 = enhanced_features
        
        # FFC处理
        x0, x1 = x_c.chunk(2, dim=1)
        x = self.FFC1(tuple([x0, x1]))
        x = self.FFC2(x)
        x = torch.cat([x[0], x[1]], dim=1)
        x = x + x_c  # 残差连接
        
        # 多尺度卷积
        x = self.multiscale_conv(x)
        x = self.transposed_block_1(x)
        
        # 注意力桥接与级联上采样
        # 对齐分辨率后再做注意力和拼接
        if x_bb_4.shape[-2:] != x.shape[-2:]:
            x_bb_4 = F.interpolate(x_bb_4, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if x_enhanced_1.shape[-2:] != x.shape[-2:]:
            x_enhanced_1 = F.interpolate(x_enhanced_1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att1(x, x_bb_4)
        x = torch.cat([x, bridge, x_enhanced_1], dim=1)
        
        x = self.transposed_block_2(x)
        if x_bb_3.shape[-2:] != x.shape[-2:]:
            x_bb_3 = F.interpolate(x_bb_3, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if x_enhanced_2.shape[-2:] != x.shape[-2:]:
            x_enhanced_2 = F.interpolate(x_enhanced_2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att2(x, x_bb_3)
        x = torch.cat([x, bridge, x_enhanced_2], dim=1)
        
        x = self.feature_extraction_B(x)
        x = self.transposed_block_3(x)
        if x_bb_2.shape[-2:] != x.shape[-2:]:
            x_bb_2 = F.interpolate(x_bb_2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if x_enhanced_3.shape[-2:] != x.shape[-2:]:
            x_enhanced_3 = F.interpolate(x_enhanced_3, size=x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att3(x, x_bb_2)
        x = torch.cat([x, bridge, x_enhanced_3], dim=1)
        
        x = self.transposed_block_4(x)
        if x_bb_1.shape[-2:] != x.shape[-2:]:
            x_bb_1 = F.interpolate(x_bb_1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if x_enhanced_4.shape[-2:] != x.shape[-2:]:
            x_enhanced_4 = F.interpolate(x_enhanced_4, size=x.shape[-2:], mode='bilinear', align_corners=False)
        bridge = self.att4(x, x_bb_1)
        x = torch.cat([x, bridge, x_enhanced_4], dim=1)
        
        x = self.transposed_block_5(x)
        if x_g.shape[-2:] != x.shape[-2:]:
            x_g = F.interpolate(x_g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x_g], dim=1)
        
        # 输出
        x = self.output_conv1(x)
        est_b = self.output_conv2(x)
        
        return est_b

class EnhancementLayers(nn.Module):
    """
    梯度分支增强层：对梯度解码器产生的四个中间特征进行逐级卷积增强，
    提升其表达能力后再送入主解码器进行融合。
    
    输入/输出：
        输入:  (x_to_be_enhanced_1..4)
        输出:  (x_enhanced_1..4)
    """
    def __init__(self):
        super(EnhancementLayers, self).__init__()
        self.enhancement_layers_1 = Conv2D_BN_activa(256, 128, 7, 1, 3)
        self.enhancement_layers_2 = Conv2D_BN_activa(128, 64, 7, 1, 3)
        self.enhancement_layers_3 = Conv2D_BN_activa(64, 32, 7, 1, 3)
        self.enhancement_layers_4 = Conv2D_BN_activa(32, 16, 7, 1, 3)

    def forward(self, to_be_enhanced_features):
        """
        参数:
            to_be_enhanced_features: 4 个待增强的中间特征元组

        返回:
            4 个增强后的特征元组
        """
        x_enhanced_1 = self.enhancement_layers_1(to_be_enhanced_features[0])
        x_enhanced_2 = self.enhancement_layers_2(to_be_enhanced_features[1])
        x_enhanced_3 = self.enhancement_layers_3(to_be_enhanced_features[2])
        x_enhanced_4 = self.enhancement_layers_4(to_be_enhanced_features[3])
        return x_enhanced_1, x_enhanced_2, x_enhanced_3, x_enhanced_4

class YFFC(nn.Module):
    """
    Y-FFC网络：Y形快速傅立叶卷积网络
    
    主要功能：从高反射条纹图像中分离漫反射和镜面反射分量
    网络架构：Y形设计，共享编码器，双解码器分支
    
    输出:
        - r: 漫反射分量（输入 - 镜面反射）
        - g: 梯度分量（辅助信息）
        - b: 镜面反射分量
    """
    def __init__(self):
        super(YFFC, self).__init__()
        # 编码器：提取多尺度特征
        self.encoder = Encoder()
        # 梯度解码器：生成梯度信息分支
        self.gradient_decoder = GradientDecoder()
        # 增强层：对梯度分支特征进行增强
        self.enhancement_layers = EnhancementLayers()
        # 主解码器：生成镜面反射分量
        self.main_decoder = MainDecoder(in_channels=256)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像，形状 [B, 1, H, W]
        
        返回:
            estimations: 包含三个分量的字典
                - 'r': 漫反射分量
                - 'g': 梯度分量  
                - 'b': 镜面反射分量
        """
        # 编码器输出：x_bb_1..x_bb_4（多尺度特征）, x_c（瓶颈特征）
        c_features = self.encoder(x)
        
        # 梯度解码器输入：x_bb_1..x_bb_4（不包括瓶颈特征）
        g_features = self.gradient_decoder(c_features[:-1])
        
        # 增强层输入：去掉最终梯度图，只处理中间特征
        enhanced_features = self.enhancement_layers(g_features[:-1])
        
        # 主解码器：融合所有特征，输出镜面反射分量
        est_b = self.main_decoder(c_features, enhanced_features, g_features[-1])
        
        # 构建输出字典：
        #   r = x - b  表示漫反射分量；
        #   g          表示梯度分量（辅助监督/先验）；
        #   b          表示镜面反射分量。
        estimations = {
            'r': x - est_b,        # 漫反射 = 输入 - 镜面反射
            'g': g_features[-1],   # 梯度分量
            'b': est_b             # 镜面反射分量
        }
        return estimations

class FringeDataset(Dataset):
    """
    条纹数据集类：管理高反射条纹图像和对应的漫反射目标图像
    
    功能：
    1. 自动匹配输入和目标图像文件
    2. 加载和预处理图像数据
    3. 生成训练所需的标签数据（镜面反射、梯度等）
    
    支持多种文件命名约定和自动匹配策略
    
    使用方式：
        dataset = FringeDataset('data/inputs', 'data/targets')
        input_img, target_img, gt_specular, gt_gradient = dataset[i]
    
    返回张量形状：
        - input_img:   [1, H, W], 值域[0,1]
        - target_img:  [1, H, W], 值域[0,1]
        - gt_specular: [1, H, W] = input - target
        - gt_gradient: [1, H, W] = target 在 (x,y) 方向的梯度（此处使用 torch.gradient）
    """
    def __init__(self, input_dir, target_dir, transform=None):
        """
        初始化数据集
        
        参数:
            input_dir: 输入图像目录（高反射条纹）
            target_dir: 目标图像目录（漫反射条纹）
            transform: 数据增强变换（可选）
        """
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        # 获取所有支持的图像文件，按名称排序
        self.files = sorted([f for f in os.listdir(input_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        获取单个数据样本，并生成对应标签。
        
        参数:
            idx: 样本索引
        
        返回:
            input_img: 输入图像（高反射条纹）
            target_img: 目标图像（漫反射条纹）
            gt_specular: 真实镜面反射（输入 - 目标）
            gt_gradient: 真实梯度（目标图像的x方向梯度）
        """
        img_name = self.files[idx]
        input_path = os.path.join(self.input_dir, img_name)
        
        # ==================== 智能文件名匹配 ====================
        # 推导目标文件名：支持多种命名约定
        candidate_names = [
            img_name,  # 优先同名匹配
            img_name.replace('input_', 'target_', 1) if img_name.startswith('input_') else img_name,  # input_001.png → target_001.png
            img_name.replace('input', 'target', 1) if img_name.startswith('input') else img_name,    # input001.png → target001.png
        ]
        
        # 尝试找到匹配的目标文件
        target_path = None
        for name in candidate_names:
            p = os.path.join(self.target_dir, name)
            if os.path.exists(p):
                target_path = p
                break
        
        # 如果上述方法都失败，尝试数字匹配
        if target_path is None:
            base, ext = os.path.splitext(img_name)
            digits = ''.join(ch for ch in base if ch.isdigit())
            if digits:
                guessed = f'target_{digits}{ext}'  # 提取数字部分构造目标文件名
                p = os.path.join(self.target_dir, guessed)
                if os.path.exists(p):
                    target_path = p
        
        # 如果仍然找不到匹配文件，抛出错误
        if target_path is None:
            raise FileNotFoundError(f"Could not find matching target for input '{img_name}' in '{self.target_dir}'. Tried: {set(candidate_names)}")
        
        # ==================== 图像加载与预处理 ====================
        # 加载图像并转换为灰度，归一化到[0,1]范围
        input_img = np.array(Image.open(input_path).convert('L'), dtype=np.float32) / 255.0
        target_img = np.array(Image.open(target_path).convert('L'), dtype=np.float32) / 255.0
        
        # 转换为PyTorch张量，添加通道维度 [H,W] → [1,H,W]
        input_img = torch.from_numpy(input_img).unsqueeze(0)
        target_img = torch.from_numpy(target_img).unsqueeze(0)
        
        # 应用数据增强变换（如果提供）
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        
        # ==================== 标签数据生成 ====================
        # 生成真实镜面反射标签：输入 - 目标
        gt_specular = input_img - target_img
        
        # 生成真实梯度标签：目标图像梯度（同时考虑 x、y 方向）
        # torch.gradient 返回 (grad_dim1, grad_dim2)，此处合成为幅值近似
        gx, gy = torch.gradient(target_img, dim=(1, 2))
        gt_gradient = (gx + gy) / 2.0
        
        return input_img, target_img, gt_specular, gt_gradient

# ==================== 工具函数 ====================
def display_basic_model_info(model):
    """
    显示基本的模型信息（不需要额外库）
    
    参数:
        model: 要显示的模型
    """
    print("\n" + "=" * 60)
    print("模型层次结构")
    print("=" * 60)
    
    # 显示主要模块
    print("\n【主要模块】")
    for name, module in model.named_children():
        print(f"  • {name}: {module.__class__.__name__}")
    
    # 显示各模块的子模块（第二层）
    print("\n【详细结构】")
    for name, module in model.named_children():
        print(f"\n{name} ({module.__class__.__name__}):")
        for sub_name, sub_module in module.named_children():
            print(f"  └─ {sub_name}: {sub_module.__class__.__name__}")
    
    # 显示参数信息
    print("\n" + "=" * 60)
    print("模型参数信息")
    print("=" * 60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")
    
    # 按模块显示参数量
    print("\n【各模块参数量】")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        percentage = (module_params / total_params * 100) if total_params > 0 else 0
        print(f"  • {name}: {module_params:,} ({percentage:.1f}%)")

def display_model_summary(model, input_size, device='cpu'):
    """
    显示模型结构摘要，优先使用 torchinfo，回退到 torchsummary，最后使用基本信息显示
    
    参数:
        model: 要显示的模型
        input_size: 输入数据的形状，格式为 (batch_size, channels, height, width)
        device: 运行设备，'cpu' 或 'cuda'
    
    返回:
        bool: 成功返回 True，失败返回 False
    """
    try:
        from torchinfo import summary
        print("使用 torchinfo 显示详细模型结构...")
        print("=" * 60)
        summary(model, input_size=input_size, device=device, verbose=1)
        return True
    except ImportError:
        try:
            from torchsummary import summary
            print("使用 torchsummary 显示详细模型结构...")
            print("=" * 60)
            summary(model, input_size[1:], device=device)  # torchsummary 不需要 batch 维度
            return True
        except ImportError:
            print("\n⚠️  提示: 未安装 torchinfo 或 torchsummary")
            print("   建议安装以获得更详细的模型结构信息:")
            print("   pip install torchinfo")
            print("\n   现在将显示基本模型信息...\n")
            display_basic_model_info(model)
            return True

def main():
    """
    主函数：解析命令行参数并显示模型结构
    
    功能：
    1. 解析命令行参数（图像大小、批次大小、设备等）
    2. 创建 Y-FFC 模型实例
    3. 测试模型前向传播
    4. 显示模型结构摘要（可选）
    5. 显示参数统计（可选）
    """
    parser = argparse.ArgumentParser(description='Y-FFC网络模型结构显示工具')
    parser.add_argument('--input_channels', type=int, default=1, 
                       help='输入图像通道数 (默认: 1)')
    parser.add_argument('--image_size', type=int, default=512, 
                       help='输入图像尺寸 (默认: 512)')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='批次大小 (默认: 1)')
    parser.add_argument('--device', type=str, default='cpu', 
                       choices=['cpu', 'cuda'], help='运行设备 (默认: cpu)')
    parser.add_argument('--show_params', action='store_true', default=True,
                       help='显示详细参数统计（默认开启，使用 --no-show_params 关闭）')
    parser.add_argument('--no-show_params', action='store_true',
                       help='不显示参数统计')
    parser.add_argument('--show_summary', action='store_true', default=True,
                       help='显示模型结构摘要（默认开启，使用 --no-show_summary 关闭）')
    parser.add_argument('--no-show_summary', action='store_true',
                       help='不显示模型结构摘要')
    
    args = parser.parse_args()
    
    # 处理 --no-* 参数
    if args.no_show_params:
        args.show_params = False
    if args.no_show_summary:
        args.show_summary = False
    
    print("=" * 60)
    print("Y-FFC网络模型结构显示")
    print("=" * 60)
    print(f"输入通道数: {args.input_channels}")
    print(f"图像尺寸: {args.image_size}x{args.image_size}")
    print(f"批次大小: {args.batch_size}")
    print(f"运行设备: {args.device}")
    print("=" * 60)
    
    # 创建模型
    try:
        model = YFFC()
        print("✓ Y-FFC模型创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 设置设备
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU")
        device = torch.device('cpu')
    
    model.to(device)
    print(f"✓ 模型已移动到设备: {device}")
    
    # 准备输入数据
    input_size = (args.batch_size, args.input_channels, args.image_size, args.image_size)
    test_input = torch.randn(input_size).to(device)
    print(f"✓ 测试输入数据生成成功，形状: {test_input.shape}")
    
    # 测试前向传播
    try:
        with torch.no_grad():
            output = model(test_input)
            print("\n" + "=" * 30)
            print("模型前向传播测试成功!")
            print("=" * 30)
            print(f"输入形状: {test_input.shape}")
            print(f"漫反射输出形状: {output['r'].shape}")
            print(f"梯度输出形状: {output['g'].shape}")
            print(f"镜面反射输出形状: {output['b'].shape}")
            
            # 验证输出形状
            assert output['r'].shape == test_input.shape, "漫反射输出形状不匹配"
            assert output['g'].shape == test_input.shape, "梯度输出形状不匹配"
            assert output['b'].shape == test_input.shape, "镜面反射输出形状不匹配"
            print("✓ 所有输出形状验证通过")
    except Exception as e:
        print(f"\n❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 显示参数统计
    if args.show_params:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n" + "=" * 60)
        print("参数统计")
        print("=" * 60)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")
    
    # 显示模型结构
    if args.show_summary:
        print("\n" + "=" * 60)
        print("模型结构详情")
        print("=" * 60)
        display_model_summary(model, input_size, device)
    
    print("\n" + "=" * 60)
    print("✅ 模型分析完成!")
    print("=" * 60)
    
    # 提示信息
    if not args.show_summary and not args.show_params:
        print("\n💡 提示: 使用以下参数获取更多信息:")
        print("   --show_params   : 显示参数统计")
        print("   --show_summary  : 显示模型结构")
        print("   或直接运行 'python Y_FFC.py' 查看所有信息")

if __name__ == '__main__':
    main() 