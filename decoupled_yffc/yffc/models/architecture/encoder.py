"""Y-FFC encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..attention import CrossModalFusion
from ..blocks import Conv2D_BN_activa
from ..branches import EnhancedViTEncoderBranch
from ..ffc import FFC_BN_ACT
from ..heads import HighFrequencyEnhancement


class Encoder(nn.Module):
    """
    编码器：ResNet50（dilated）+ 增强版 ViT-B/16 双分支，末尾接 FFC bottleneck
    和 HighFrequencyEnhancement。

    改进点（相比原始版本）：
      - ViT 分支升级为 EnhancedViTEncoderBranch（多层特征 + CPE + 自适应融合）
      - CNN-ViT 融合升级为 CrossModalFusion（双向交叉注意力）
    输出 5 级特征用于后续 Decoder 的 skip connection。
    """

    def __init__(self, vit_image_size=224):
        super().__init__()

        try:
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1,
                replace_stride_with_dilation=[False, False, True])
        except Exception:
            backbone = models.resnet50(
                weights=None,
                replace_stride_with_dilation=[False, False, True])

        old_conv1 = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight.copy_(old_conv1.weight.mean(1, keepdim=True))

        self.stem_conv = backbone.conv1
        self.stem_bn = backbone.bn1
        self.stem_relu = backbone.relu
        self.stem_pool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.reduce_l1 = Conv2D_BN_activa(256, 128, 1, 1, 0)
        self.reduce_l2 = Conv2D_BN_activa(512, 256, 1, 1, 0)
        self.reduce_l3 = Conv2D_BN_activa(1024, 512, 1, 1, 0)
        self.reduce_l4 = Conv2D_BN_activa(2048, 512, 1, 1, 0)
        self.fuse_l34 = Conv2D_BN_activa(1024, 512, 3, 1, 1)

        self.vit_branch = EnhancedViTEncoderBranch(vit_image_size=vit_image_size)
        # [V5] 交叉注意力融合替代简单拼接
        self.cross_modal_fusion = CrossModalFusion(
            cnn_dim=512, vit_dim=self.vit_branch.embed_dim,
            out_dim=512, num_heads=4, sr_ratio=2)
        self.cba_after_backbone = Conv2D_BN_activa(512, 256, 3, 1, 1)
        self.ffc_bottleneck = FFC_BN_ACT(256, 256, 3, 0.5, 0.5, padding=1)

        # 高频增强
        self.hf_enhance = HighFrequencyEnhancement(channels=256)

    def forward(self, x):
        x_in = x
        x = self.stem_conv(x); x = self.stem_bn(x)
        x_bb_1 = self.stem_relu(x)

        x = self.stem_pool(x_bb_1)
        x_l1 = self.layer1(x); x_bb_2 = self.reduce_l1(x_l1)
        x_l2 = self.layer2(x_l1); x_bb_3 = self.reduce_l2(x_l2)
        x_l3 = self.layer3(x_l2); x_l4 = self.layer4(x_l3)

        x_l3_512 = self.reduce_l3(x_l3)
        x_l4_512 = self.reduce_l4(x_l4)
        x_bb_4 = self.fuse_l34(torch.cat([x_l3_512, x_l4_512], 1))

        vit_feat = self.vit_branch(x_in)
        if vit_feat.shape[-2:] != x_bb_4.shape[-2:]:
            vit_feat = F.interpolate(vit_feat, size=x_bb_4.shape[-2:],
                                     mode='bilinear', align_corners=False)

        # [V5] 交叉注意力融合
        fused = self.cross_modal_fusion(x_bb_4, vit_feat)
        x_c = self.cba_after_backbone(fused)

        nl = 256 - int(256 * 0.5)
        out = self.ffc_bottleneck((x_c[:, :nl], x_c[:, nl:]))
        x_c = torch.cat([out[0], out[1]], 1) + x_c

        # 高频增强
        x_c = self.hf_enhance(x_c)

        return x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c
