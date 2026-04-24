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
    [V5] 轻量级 CNN-ViT 交叉注意力融合模块。

    替代原始的简单 concat + 1×1 Conv 融合方式。
    核心思想：
      - CNN → ViT 方向：CNN 特征作为 Query 去查询 ViT 的全局语义 Key/Value，
        为局部细节注入全局上下文（例如：判断某个高亮区域是局部反射还是全局照明）
      - ViT → CNN 方向：ViT 特征作为 Query 去查询 CNN 的局部细节 Key/Value，
        为全局语义补充精确的空间结构信息（例如：高光边界的精确位置）
      - 双向交叉后通过门控机制融合，输出维度为 out_dim

    使用降采样键值对减少计算量（sr_ratio），适合高分辨率特征图。

    在整个模型中的位置：
        Encoder 中，在 ResNet50 backbone 输出 x_bb_4 和 ViT 输出 vit_feat
        之后进行融合，替代原来的 cat + 1×1 Conv。
    """

    def __init__(self, cnn_dim, vit_dim, out_dim, num_heads=4, sr_ratio=2):
        super(CrossModalFusion, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.out_dim = out_dim

        # CNN 关注 ViT（获取全局上下文）
        self.q_c2v = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.k_c2v = nn.Conv2d(vit_dim, out_dim, 1, bias=False)
        self.v_c2v = nn.Conv2d(vit_dim, out_dim, 1, bias=False)
        self.proj_c2v = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        # ViT 关注 CNN（获取局部细节）
        self.q_v2c = nn.Conv2d(vit_dim, out_dim, 1, bias=False)
        self.k_v2c = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.v_v2c = nn.Conv2d(cnn_dim, out_dim, 1, bias=False)
        self.proj_v2c = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        # 键值对空间降采样（减少计算量）
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_pool = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)

        # 门控融合：自适应平衡两个方向的贡献
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

    def forward(self, cnn_feat, vit_feat):
        """
        Args:
            cnn_feat: (B, cnn_dim, H, W) — ResNet50 输出
            vit_feat: (B, vit_dim, H, W) — ViT 输出（已对齐到相同空间尺寸）
        Returns:
            (B, out_dim, H, W) — 融合后的特征
        """
        # 降采样键值对以减少计算量
        if self.sr_ratio > 1:
            vit_kv = self.sr_pool(vit_feat)
            cnn_kv = self.sr_pool(cnn_feat)
        else:
            vit_kv = vit_feat
            cnn_kv = cnn_feat

        # CNN → ViT 方向：CNN 查询 ViT 的全局语义
        q_c = self.q_c2v(cnn_feat)
        k_v = self.k_c2v(vit_kv)
        v_v = self.v_c2v(vit_kv)
        c2v_out = self._cross_attn(q_c, k_v, v_v, self.proj_c2v)

        # ViT → CNN 方向：ViT 查询 CNN 的局部细节
        q_v = self.q_v2c(vit_feat)
        k_c = self.k_v2c(cnn_kv)
        v_c = self.v_v2c(cnn_kv)
        v2c_out = self._cross_attn(q_v, k_c, v_c, self.proj_v2c)

        # 门控融合
        gate = self.gate(torch.cat([c2v_out, v2c_out], dim=1))
        fused = gate * c2v_out + (1.0 - gate) * v2c_out

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
