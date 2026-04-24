"""ViT encoder branch used by the Y-FFC encoder."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..common import truncated_normal_


class ConvPosEncoding(nn.Module):
    """
    [V2] 卷积位置编码（Convolutional Position Encoding）。

    用深度可分离卷积为 Transformer token 序列注入局部空间结构信息。
    相比标准 ViT 的固定/可学习位置编码：
      - 能感知局部邻域关系（3×3 卷积窗口）
      - 对不同分辨率输入具有天然的泛化能力
      - 保留了平移等变性
    参考：CPVT（Conditional Position encoding for Vision Transformers, ICLR 2023）
    """

    def __init__(self, dim, k=3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=True)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, N, C) — token 序列
            H, W: patch grid 的空间尺寸
        Returns:
            (B, N, C) — 加入局部位置信息后的 token 序列
        """
        B, N, C = x.shape
        feat = x.transpose(1, 2).reshape(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        return x + feat


class EnhancedViTEncoderBranch(nn.Module):
    """
    增强版 ViT-B/16 编码器分支（多层特征提取 + CPE + 自适应层融合）。

    相比原始 ViTEncoderBranch 的改进：
    ─────────────────────────────────────────────
      V1  多层特征提取：手动遍历 Transformer 的各层，从第 3/6/9/12 层
          提取中间特征。浅层保留纹理/边缘信息（对高光边界至关重要），
          深层提供全局语义（对大面积高光区域识别至关重要）。

      V2  卷积位置编码（CPE）：在 patch embedding 之后、进入 Transformer
          之前注入局部空间先验。弥补标准 ViT 对局部结构感知不足的缺陷，
          使模型能更精确地定位高光边界。

      V3  自适应层注意力融合：使用通道注意力机制（SE-style）自适应加权
          不同 Transformer 层的特征贡献。训练过程中模型会自动学习：
          浅层特征在强高光边界区域的权重更高，深层特征在大面积高光
          区域的权重更高。

      V4  残差特征精炼：融合后通过两层 3×3 Conv + BN + GELU 进行
          任务相关的特征精炼，加残差连接避免信息丢失。
    ─────────────────────────────────────────────

    在整个模型中的位置：
        被 Encoder 调用，与 ResNet50 并行提取全局语义特征。
        输出通过 CrossModalFusion 与 CNN 特征进行交叉注意力融合。
    """

    def __init__(self, vit_image_size=224, extract_layers=(2, 5, 8, 11)):
        super(EnhancedViTEncoderBranch, self).__init__()
        self.vit_image_size = vit_image_size
        self.extract_layers = list(extract_layers)

        # ---------- 加载 ViT-B/16 ----------
        vit_weights = None
        try:
            vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        except Exception:
            vit_weights = None

        if not hasattr(models, "vit_b_16"):
            raise RuntimeError("当前 torchvision 版本不包含 vit_b_16")

        try:
            self.vit = models.vit_b_16(weights=vit_weights)
        except Exception:
            self.vit = models.vit_b_16(weights=None)

        if hasattr(self.vit, "heads"):
            self.vit.heads = nn.Identity()
        elif hasattr(self.vit, "head"):
            self.vit.head = nn.Identity()

        self.embed_dim = getattr(self.vit, "hidden_dim", self.vit.conv_proj.out_channels)

        # [V2] 卷积位置编码
        self.cpe = ConvPosEncoding(self.embed_dim, k=3)

        # [V3] 多层特征的层归一化 + 自适应注意力融合
        n_layers = len(self.extract_layers)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.embed_dim) for _ in range(n_layers)
        ])

        fused_dim = self.embed_dim * n_layers
        se_mid = max(n_layers * 2, 4)
        self.layer_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fused_dim, se_mid),
            nn.GELU(),
            nn.Linear(se_mid, n_layers),
            nn.Softmax(dim=-1),
        )

        # [V4] 残差特征精炼
        self.refine = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
        )

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x = F.interpolate(x, size=(self.vit_image_size, self.vit_image_size),
                          mode="bilinear", align_corners=False)
        B = x.size(0)

        # Patch embedding
        x = self.vit.conv_proj(x)
        grid_h, grid_w = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # [V2] 注入局部空间位置先验
        x = self.cpe(x, grid_h, grid_w)

        # 添加 class token
        has_cls = hasattr(self.vit, "class_token")
        if has_cls:
            cls_token = self.vit.class_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # 添加 ViT 原始位置编码 + dropout
        if hasattr(self.vit.encoder, "pos_embedding"):
            x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)

        # [V1] 手动遍历 Transformer 层，收集中间特征
        multi_feats = []
        for i, layer in enumerate(self.vit.encoder.layers):
            x = layer(x)
            if i in self.extract_layers:
                multi_feats.append(x)

        # 确保收集到足够的特征（如果 ViT 层数不够则用最后的输出补齐）
        while len(multi_feats) < len(self.extract_layers):
            multi_feats.append(x)

        # [V3] 多层特征处理 + 自适应加权融合
        layer_feats_2d = []
        for i, feat in enumerate(multi_feats):
            feat = self.layer_norms[i](feat)
            # 移除 class token
            if has_cls and feat.shape[1] == grid_h * grid_w + 1:
                feat = feat[:, 1:, :]
            elif feat.shape[1] > grid_h * grid_w:
                feat = feat[:, 1:, :]
            feat_2d = feat.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)
            layer_feats_2d.append(feat_2d)

        # 通道注意力加权：自适应选择各层贡献
        concat_feats = torch.cat(layer_feats_2d, dim=1)  # (B, embed_dim*n, H, W)
        attn_weights = self.layer_attention(concat_feats)  # (B, n_layers)

        fused = torch.zeros_like(layer_feats_2d[0])
        for i, feat in enumerate(layer_feats_2d):
            w = attn_weights[:, i].view(B, 1, 1, 1)
            fused = fused + w * feat

        # [V4] 残差精炼
        fused = fused + self.refine(fused)

        return fused
