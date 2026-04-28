"""Attention and cross-modal fusion modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_block(nn.Module):
    """
    注意力门控块。
    用于 Decoder skip connection 中对编码器特征进行注意力加权。

    特点：
      - psi 输出 F_l 通道（每通道独立注意力权重），而非 1 通道。
      - F_int 缩小以降低参数量并减轻过拟合。
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class CrossModalFusion(nn.Module):
    """
    轻量级 CNN-Transformer 双向交叉注意力融合模块。

    核心思想：
      - CNN → Swin 方向：ResNet 深层特征作为 Query，查询 Swin 的窗口上下文 Key/Value，
        为局部纹理注入更大范围的反射区域和条纹结构上下文。
      - Swin → CNN 方向：Swin H/16 特征作为 Query，查询 ResNet 的局部细节 Key/Value，
        为窗口上下文补充高光边界、细条纹和局部反射形态。
      - 双向交叉后通过门控机制融合，输出维度为 out_dim。

    使用 sr_ratio 对键值对做空间降采样，降低高分辨率特征上的注意力开销。
    本阶段只用于瓶颈层 H/16 融合，不用于 H/4、H/8 skip 融合。
    """

    def __init__(self, cnn_dim, context_dim, out_dim, num_heads=4, sr_ratio=2):
        super(CrossModalFusion, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.out_dim = out_dim

        self.q_c2ctx = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.k_c2ctx = nn.Conv2d(context_dim, out_dim, 1, bias=False)
        self.v_c2ctx = nn.Conv2d(context_dim, out_dim, 1, bias=False)
        self.proj_c2ctx = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        self.q_ctx2c = nn.Conv2d(context_dim, out_dim, 1, bias=False)
        self.k_ctx2c = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.v_ctx2c = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.proj_ctx2c = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_pool = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)

        # 门控融合：自适应平衡 CNN 查询 Swin 与 Swin 查询 CNN 两个方向的贡献。
        self.gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid(),
        )

        # 最终投影
        self.final_proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def _cross_attn(self, q, k, v, proj):
        """执行单方向的交叉注意力。"""
        B, C, H, W = q.shape
        Hk, Wk = k.shape[-2:]
        nh = self.num_heads
        hd = self.head_dim

        q = q.reshape(B, nh, hd, H * W).permute(0, 1, 3, 2)   # (B, nh, HW, hd)
        k = k.reshape(B, nh, hd, Hk * Wk).permute(0, 1, 3, 2) # (B, nh, HkWk, hd)
        v = v.reshape(B, nh, hd, Hk * Wk).permute(0, 1, 3, 2) # (B, nh, HkWk, hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale           # (B, nh, HW, HkWk)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return proj(out)

    def forward(self, cnn_feat, context_feat):
        """
        Args:
            cnn_feat: (B, cnn_dim, H, W) — ResNet50 深层输出 x_bb_4
            context_feat: (B, context_dim, H, W) — Swin H/16 输出，已对齐到 x_bb_4
        Returns:
            (B, out_dim, H, W) — 融合后的瓶颈输入特征
        """
        if self.sr_ratio > 1:
            context_kv = self.sr_pool(context_feat)
            cnn_kv = self.sr_pool(cnn_feat)
        else:
            context_kv = context_feat
            cnn_kv = cnn_feat

        # CNN 查询 Swin：为 ResNet 局部纹理补充窗口上下文。
        q_c = self.q_c2ctx(cnn_feat)
        k_ctx = self.k_c2ctx(context_kv)
        v_ctx = self.v_c2ctx(context_kv)
        c2ctx_out = self._cross_attn(q_c, k_ctx, v_ctx, self.proj_c2ctx)

        # Swin 查询 CNN：为 Swin 上下文补充高光边界和细条纹局部细节。
        q_ctx = self.q_ctx2c(context_feat)
        k_c = self.k_ctx2c(cnn_kv)
        v_c = self.v_ctx2c(cnn_kv)
        ctx2c_out = self._cross_attn(q_ctx, k_c, v_c, self.proj_ctx2c)

        # 门控融合两个方向的结果。
        gate = self.gate(torch.cat([c2ctx_out, ctx2c_out], dim=1))
        fused = gate * c2ctx_out + (1.0 - gate) * ctx2c_out

        return self.final_proj(fused)


class SpecularAwareAttention(nn.Module):
    """
    镜面反射感知注意力模块。
    [P9] steepness 15→6；多级阈值(0.5/0.7/0.9)区分弱/中/强高光。
    """

    def __init__(self, in_channels, threshold=0.8, steepness=6.0):
        super().__init__()
        self.steepness = steepness
        self.thresholds = [0.5, 0.7, 0.9]
        n_masks = len(self.thresholds)

        mid = max(in_channels // 16, 4)
        self.adaptive_thresh = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, n_masks, 1, bias=True),
            nn.Sigmoid(),
        )

        self.modulation = nn.Sequential(
            nn.Conv2d(in_channels + n_masks, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, features, original_input):
        inp = original_input
        if inp.shape[-2:] != features.shape[-2:]:
            inp = F.interpolate(inp, size=features.shape[-2:],
                                mode='bilinear', align_corners=False)
        t_off = self.adaptive_thresh(features)
        masks = []
        for i, base_t in enumerate(self.thresholds):
            eff_t = base_t * (0.5 + 0.5 * t_off[:, i:i+1, :, :])
            masks.append(torch.sigmoid(self.steepness * (inp - eff_t)))
        multi_mask = torch.cat(masks, dim=1)
        att = self.modulation(torch.cat([features, multi_mask], dim=1))
        return features * (1.0 + att)
