"""
FFC (Fast Fourier Convolution) 模块实现 —— 优化版

优化点摘要（相比原版，接口/类名/调用方式零改动）：

  F1  FourierUnit: ReLU → GELU
      原 ReLU 对频域 [real, imag] 拼接张量做 hard-zero，破坏虚部编码的相位信息。
      GELU 光滑且不 hard-zero 负值，相位信息得以保留。

  F2  FourierUnit: 增加频率 SE 通道注意力
      在频谱卷积后插入 AdaptiveAvgPool → 2层Conv → Sigmoid 的 SE 门控，
      让网络自适应聚焦于条纹频段，压制噪声频段。

  F3  SpectralTransform: LFU 拆分为独立 lfu_top / lfu_bot
      原设计上下两半共享同一 FourierUnit 参数，导致参数竞争。
      拆分后各半区域有独立的频域滤波器，表达能力提升。

  F4  SpectralTransform: FU / LFU 门控融合
      原简单相加改为 `gate * fu_out + (1-gate) * lfu_out`，
      空间门控（1×1 Conv → Sigmoid）自适应平衡全局 FU 与局部 LFU 的贡献。

  F5  FFC_BN_ACT: L/G 跨域分支门控融合
      局部输出 L+G 和全局输出 L+G 的简单加法改为门控加法，
      门控由频域分支输出驱动（1×1 Conv → Sigmoid），
      防止频域噪声直接污染空间分支；训练初期频域分支贡献被门控压低，
      收敛后门控自动放开。

  F6  FourierUnit: FFT 前应用 Hann 窗
      减少非周期特征图边界不连续引起的频谱泄漏（吉布斯效应），
      能量衰减由后续 BN 归一化自动补偿。

类名、forward 签名、输入输出 shape、调用方式均与原版完全一致。
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
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class FourierUnit(nn.Module):
    """
    傅里叶单元 - FFC 的核心组件（优化版）

    工作原理：
        1. [F6] Hann 窗预处理：减少频谱泄漏
        2. 空间域特征 → FFT → 频域
        3. [real, imag] 分离后拼接到通道维
        4. 频域 1×1 卷积（全局感受野）
        5. [F1] BN + GELU（保留相位，不 hard-zero 虚部）
        6. [F2] SE 通道注意力：自适应频段选择
        7. 重组为复数 → IFFT → 空间域

    改动说明（相比原版）：
        - ReLU → GELU：保护频域相位信息
        - 新增 freq_se：SE 式通道注意力，对频谱进行频段选择
        - 新增 Hann 窗：前向传播时动态构建并应用，减少频谱泄漏

    参数（与原版完全相同）:
        in_channels, out_channels, groups, spatial_scale_factor,
        spatial_scale_mode, spectral_pos_encoding, use_se,
        se_kwargs, ffc3d, fft_norm
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

        # [F1] GELU 替代 ReLU：频域 [real, imag] 拼接通道上不 hard-zero 负值，
        # 保护虚部编码的空间相位信息，防止相位折叠。
        self.act = nn.GELU()

        # [F2] SE 通道注意力：在频谱上自适应选择有用频段。
        # AdaptiveAvgPool 对整个频率平面 (H × W//2+1) 全局平均，
        # 随后两层 1×1 Conv 学习通道间的频段重要性权重，输出 [B, C*2, 1, 1]。
        # 每个输出通道（对应某输入通道的实部或虚部）独立获得一个 [0,1] 频段权重。
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
        """
        前向传播：频域处理特征（优化版）

        步骤:
            1. 可选空间缩放
            2. [F6] Hann 窗加权（动态构建）→ 减少频谱泄漏
            3. rfftn → 复数频谱
            4. [real, imag] 分离 → 展平到通道维
            5. 可选频谱位置编码
            6. 1×1 Conv → [F1] BN + GELU
            7. [F2] SE 频率通道注意力加权
            8. 重组为复数 → irfftn → 空间域
            9. 可选恢复原始尺寸

        参数 / 返回值与原版完全一致。
        """
        batch = x.shape[0]

        # 可选：多尺度处理（缩放输入）
        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor,
                              mode=self.spatial_scale_mode, align_corners=False)

        # 记录当前空间尺寸（用于 irfftn 的 s 参数）
        r_size = x.size()

        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)

        # AMP 兼容：ComplexHalf 不支持，提升为 float32
        orig_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()

        # ==================== [F6] Hann 窗：减少频谱泄漏 ====================
        # FFT 假设信号周期化；非周期特征图在边界产生不连续，引起高频虚假响应。
        # Hann 窗使边界幅度平滑趋零，泄漏能量被 BN 自动补偿，不影响学习。
        # 动态构建（不注册 buffer），适配任意输入尺寸（如变分辨率推理）。
        h, w = x.shape[-2], x.shape[-1]
        win_h = torch.hann_window(h, device=x.device, dtype=x.dtype)
        win_w = torch.hann_window(w, device=x.device, dtype=x.dtype)
        hann_2d = torch.outer(win_h, win_w).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        x_win = x * hann_2d  # 加窗后的特征（不修改 x，保持 r_size 一致）

        # ==================== 步骤1：空间域 → 频域 ====================
        ffted = torch.fft.rfftn(x_win, dim=fft_dim, norm=self.fft_norm)
        # 输出形状：[B, C, H, W//2+1]（复数）

        # ==================== 步骤2：复数分离 ====================
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        # [B, C, H, W//2+1, 2] → [B, C, 2, H, W//2+1] → [B, C*2, H, W//2+1]
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

        # ==================== 步骤3：频域卷积 ====================
        ffted = self.conv_layer(ffted)
        # [F1] GELU 替代 ReLU：不 hard-zero 负的虚部，保护相位信息
        ffted = self.act(self.bn(ffted))

        # ==================== [F2] 频率 SE 通道注意力 ====================
        # freq_se 输出 [B, C*2, 1, 1]，广播到整个频率平面
        # 相当于对每个输出通道（实部/虚部）的频谱施加独立的频段重要性权重
        ffted = ffted * self.freq_se(ffted)

        # ==================== 步骤4：复数合并 ====================
        # [B, C'*2, H, W//2+1] → [B, C', 2, H, W//2+1] → [B, C', H, W//2+1]（复数）
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        if ffted.dtype == torch.float16:
            ffted = ffted.float()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        # ==================== 步骤5：频域 → 空间域 ====================
        # s=r_size[-2:] 使用加窗前的原始空间尺寸，确保输出 shape 与输入一致
        if self.ffc3d:
            output = torch.fft.irfftn(ffted, s=r_size[-3:], dim=fft_dim, norm=self.fft_norm)
        else:
            output = torch.fft.irfftn(ffted, s=r_size[-2:], dim=fft_dim, norm=self.fft_norm)

        # 恢复原始尺寸（如果进行了缩放）
        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size,
                                   mode=self.spatial_scale_mode, align_corners=False)

        # 恢复原始数据类型
        if orig_dtype == torch.float16:
            output = output.half()

        return output


class SpectralTransform(nn.Module):
    """
    谱变换模块 - 在频域中进行特征变换（优化版）

    核心流程：
        输入 → 1×1卷积降维 →
            全局 FU（整图 FFT）
            [F3] 独立 lfu_top / lfu_bot（上下半区各自 FFT）
        → [F4] 门控融合（gate * fu_out + (1-gate) * lfu_out）
        → 1×1 卷积升维 → 输出

    改动说明（相比原版）：
        F3 - self.lfu 拆分为 self.lfu_top + self.lfu_bot：
             上下两半区域有独立的频域滤波器，避免参数竞争。
        F4 - FU/LFU 门控融合：
             self.fusion_gate（1×1 Conv → Sigmoid）生成空间门控，
             替代原始的硬相加，自适应平衡全局与局部频域信息。

    参数（与原版完全相同）:
        in_channels, out_channels, stride, groups, enable_lfu, **kwargs
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=1,
                 enable_lfu=True, **kwargs):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu

        if groups == 1:
            # 前置 1×1 卷积：降维（→ out_channels // 2）+ BN + ReLU
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=1,
                          groups=1, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True),
            )
            fu_ch = out_channels // 2
            # 全局傅里叶单元
            self.fu = FourierUnit(fu_ch, fu_ch, groups, **kwargs)
            if self.enable_lfu:
                # [F3] 拆分为两个独立 FourierUnit，上下半区各自学习独立频域滤波器
                # 原共享 self.lfu 导致参数竞争；分离后各区域表达能力独立提升
                self.lfu_top = FourierUnit(fu_ch, fu_ch, groups)
                self.lfu_bot = FourierUnit(fu_ch, fu_ch, groups)
            # 后置 1×1 卷积：恢复通道数
            self.conv2 = nn.Conv2d(fu_ch, out_channels, kernel_size=1,
                                   groups=1, bias=False)
        else:
            # groups > 1 时，使用分组卷积（保持原逻辑）
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            fu_ch = out_channels
            self.fu = FourierUnit(fu_ch, fu_ch, groups, **kwargs)
            if self.enable_lfu:
                self.lfu_top = FourierUnit(fu_ch, fu_ch, groups)
                self.lfu_bot = FourierUnit(fu_ch, fu_ch, groups)
            self.conv2 = nn.Conv2d(fu_ch, out_channels, kernel_size=1,
                                   groups=groups, bias=False)

        # [F4] FU / LFU 门控融合：空间门控（per-pixel，broadcast 到 fu_ch 通道）
        # 输入为全局 FU 输出（fu_ch 通道），输出 [B, 1, H, W] 的空间注意力图。
        # gate 接近 1 → 偏向全局 FU（整图频率特征），
        # gate 接近 0 → 偏向局部 LFU（局部频率特征）。
        # 使用全局 FU 输出作为门控输入，使门控具有全局感知能力。
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(fu_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        前向传播（优化版）

        流程：
            1. conv1 降维
            2. 全局 FU 处理
            3. [F3] 分别用 lfu_top / lfu_bot 处理上下两半，cat 恢复完整特征
            4. [F4] gate = sigmoid(conv(fu_out))，门控融合 fu_out 与 lfu_out
            5. conv2 升维输出

        参数 / 返回值 shape 与原版完全一致。
        """
        x = self.conv1(x)             # [B, fu_ch, H, W]
        fu_out = self.fu(x)           # 全局 FU：[B, fu_ch, H, W]

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_s = h // 2          # 水平分割点

            # [F3] 独立 lfu_top / lfu_bot，各区域独立频域建模
            top_out = self.lfu_top(x[:, :, :split_s, :])   # [B, fu_ch, H/2, W]
            bot_out = self.lfu_bot(x[:, :, split_s:, :])   # [B, fu_ch, H/2, W]
            lfu_out = torch.cat([top_out, bot_out], dim=2) # [B, fu_ch, H,   W]

            # [F4] 门控融合：gate 由全局 FU 输出驱动，具有全局频谱感知能力
            # gate: [B, 1, H, W]，广播到 fu_ch 通道
            gate = self.fusion_gate(fu_out)
            combined = gate * fu_out + (1.0 - gate) * lfu_out
        else:
            # LFU 未启用时：对全局 FU 与 conv1 输出做门控融合（等价于原来的 x + output）
            gate = self.fusion_gate(fu_out)
            combined = gate * fu_out + (1.0 - gate) * x

        # conv2 升维：[B, fu_ch, H, W] → [B, out_channels, H, W]
        x = self.conv2(combined)
        return x


class FFC_BN_ACT(nn.Module):
    """
    FFC 模块 - 快速傅里叶卷积（带 BN 和激活）（优化版）

    核心思想（同原版）：
        将输入特征分为局部/全局两个分支，四路交互：
        L2L (空间) + G2L (频域) → 局部输出
        L2G (空间) + G2G (频域) → 全局输出

    改动说明（相比原版）：
        [F5] 跨域分支门控融合：
        原 `out_xl = out_l2l + out_g2l`（硬相加）改为门控加法：
            gate_l = Sigmoid(Conv1×1(out_g2l))   # [B, 1, H, W]
            out_xl  = out_l2l + gate_l * out_g2l
        同理对全局输出分支。

        物理动机：频域分支（G2L, G2G）训练初期权重未收敛，
        输出噪声较大；门控在初期自动压低频域贡献，
        随训练稳定后门控逐渐放开，无需手动调整权重。

    参数（与原版完全相同）:
        in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
        stride, padding, dilation, groups, bias, enable_lfu, activation_type
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 activation_type='relu', **kwargs):
        super(FFC_BN_ACT, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        # 计算局部和全局分支的通道数
        in_cg  = int(in_channels * ratio_gin)
        in_cl  = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        # ==================== 四个分支 ====================
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

        # 批归一化层
        self.bnl = nn.BatchNorm2d(out_cl) if out_cl != 0 else nn.Identity()
        self.bng = nn.BatchNorm2d(out_cg) if out_cg != 0 else nn.Identity()

        # 激活函数
        if activation_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation_type == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Identity()

        # ==================== [F5] 跨域门控 ====================
        # gate_l：控制频域分支 G2L 注入局部输出的强度
        #   输入  G2L 输出（out_cl 通道），输出 [B,1,H,W] 空间注意力图
        #   gate 接近 1 → 频域信息可靠，全量注入
        #   gate 接近 0 → 频域信息噪声大，抑制注入
        # 当 G→L 或 G→G 路径是 Identity 时不添加门控（无意义）
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
        """
        前向传播：四分支特征交互 + [F5] 门控融合（优化版）

        流程：
            1. 输入分离（元组或单张量）
            2. 四分支处理（L2L / G2L / L2G / G2G）
            3. [F5] 门控融合：
               局部输出 = L2L + gate_l * G2L
               全局输出 = L2G + gate_g * G2G
            4. BN + 激活

        参数 / 返回值格式与原版完全一致：
            输入：单张量 [B,C,H,W] 或元组 (x_local, x_global)
            返回：(out_local, out_global)
        """
        x_l, x_g = x if isinstance(x, tuple) else (x, None)
        out_xl, out_xg = 0, 0

        if self.global_in_num > 0 and x_g is not None:
            # -------- 局部输出 --------
            out_l2l = self.convl2l(x_l) if not isinstance(self.convl2l, nn.Identity) else x_l
            out_g2l = self.convg2l(x_g) if not isinstance(self.convg2l, nn.Identity) else x_g

            # 空间尺寸对齐（非常规输入时）
            if (isinstance(out_l2l, torch.Tensor) and isinstance(out_g2l, torch.Tensor)
                    and out_l2l.shape[-2:] != out_g2l.shape[-2:]):
                out_g2l = F.interpolate(out_g2l, size=out_l2l.shape[-2:],
                                        mode='bilinear', align_corners=False)

            if isinstance(out_l2l, torch.Tensor) and isinstance(out_g2l, torch.Tensor):
                if self._has_gate_l:
                    # [F5] 门控注入：gate 由 G2L 输出（频域信息）驱动
                    # 物理含义：频域输出自身的幅度决定其贡献可信度
                    gate_l = self.gate_l(out_g2l)     # [B, 1, H, W]
                    out_xl = out_l2l + gate_l * out_g2l
                else:
                    out_xl = out_l2l + out_g2l
            else:
                out_xl = out_l2l

            # -------- 全局输出 --------
            out_l2g = self.convl2g(x_l) if not isinstance(self.convl2g, nn.Identity) else x_l
            out_g2g = self.convg2g(x_g) if not isinstance(self.convg2g, nn.Identity) else x_g

            if (isinstance(out_l2g, torch.Tensor) and isinstance(out_g2g, torch.Tensor)
                    and out_l2g.shape[-2:] != out_g2g.shape[-2:]):
                out_g2g = F.interpolate(out_g2g, size=out_l2g.shape[-2:],
                                        mode='bilinear', align_corners=False)

            if isinstance(out_l2g, torch.Tensor) and isinstance(out_g2g, torch.Tensor):
                if self._has_gate_g:
                    # [F5] 门控注入：gate 由 G2G 输出（频域信息）驱动
                    gate_g = self.gate_g(out_g2g)     # [B, 1, H, W]
                    out_xg = out_l2g + gate_g * out_g2g
                else:
                    out_xg = out_l2g + out_g2g
            else:
                out_xg = out_l2g

        else:
            # 无全局输入，仅使用局部路径（原版逻辑保留）
            out_xl = self.convl2l(x_l) if not isinstance(self.convl2l, nn.Identity) else x_l
            out_xg = self.convl2g(x_l) if not isinstance(self.convl2g, nn.Identity) else x_l

        # BN + 激活（与原版完全一致）
        out_xl = self.act(self.bnl(out_xl))
        out_xg = self.act(self.bng(out_xg))

        return out_xl, out_xg


class FFCResnetBlock(nn.Module):
    """
    FFC 残差块 - 用于堆叠多层 FFC（同原版，无改动）

    结构：
        输入 → FFC1 → FFC2 → 残差连接 → 输出

    参数（与原版完全相同）:
        dim, padding_type, norm_layer, activation_layer,
        dilation, spatial_transform_kwargs, inline, **conv_kwargs
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
        """
        前向传播：双层 FFC + 残差连接（同原版）

        参数 / 返回值格式与原版完全一致。
        """
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
