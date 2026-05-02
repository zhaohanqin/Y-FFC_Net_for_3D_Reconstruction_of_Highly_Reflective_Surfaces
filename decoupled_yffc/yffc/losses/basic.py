"""Basic reconstruction losses and helpers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def charbonnier_loss(pred, gt, eps=1e-6):
    """[P13] Charbonnier 损失：误差 0 处光滑可导，后期梯度更稳定。"""
    return torch.sqrt((pred - gt)**2 + eps**2).mean()


def statistical_loss(pred, gt):
    """统计损失：约束预测与 GT 的均值/标准差一致（逐样本计算）。"""
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    gt_flat = gt.view(B, -1)
    mean_loss = torch.abs(pred_flat.mean(dim=1) - gt_flat.mean(dim=1)).mean()
    pred_std = pred_flat.std(dim=1, unbiased=False)
    gt_std = gt_flat.std(dim=1, unbiased=False)
    std_loss = torch.abs(pred_std - gt_std).mean()
    return mean_loss + std_loss


class PerceptualLoss(nn.Module):
    """
    感知损失（VGG16 特征匹配）。
    使用预训练 VGG16 的浅层/中层特征计算感知距离，参数冻结不参与训练。
    """

    def __init__(self, layers=('relu1_2', 'relu2_2', 'relu3_3'),
                 weights=(1.0, 0.5, 0.25)):
        super(PerceptualLoss, self).__init__()
        self.is_pretrained = False
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
            self.is_pretrained = True
            print("[PerceptualLoss] 已加载 ImageNet 预训练 VGG16，并启用 ImageNet 标准化")
        except Exception as exc:
            print(f"[PerceptualLoss] 预训练 VGG16 加载失败，感知损失将自动关闭: {exc}")
            vgg = models.vgg16(weights=None).features

        layer_map = {'relu1_2': 4, 'relu2_2': 9, 'relu3_3': 16}
        self.slices = nn.ModuleList()
        prev = 0
        for layer_name in layers:
            end = layer_map[layer_name]
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:end]))
            prev = end

        for param in self.parameters():
            param.requires_grad = False

        self.weights = weights
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('imagenet_mean', mean)
        self.register_buffer('imagenet_std', std)

    def forward(self, pred, gt):
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
            gt = gt.repeat(1, 3, 1, 1)
        pred = (pred - self.imagenet_mean) / self.imagenet_std
        gt = (gt - self.imagenet_mean) / self.imagenet_std
        loss = 0.0
        x, y = pred, gt
        for i, slice_module in enumerate(self.slices):
            x = slice_module(x)
            y = slice_module(y)
            loss += self.weights[i] * F.l1_loss(x, y)
        return loss


def compute_edge_weight_map(gt_specular, sigma=3.0, base_weight=1.0, edge_weight=3.0):
    """根据镜面反射分量的梯度生成边缘权重图，用于边缘感知加权 L1 损失。"""
    gx = gt_specular[:, :, :, 1:] - gt_specular[:, :, :, :-1]
    gy = gt_specular[:, :, 1:, :] - gt_specular[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0), mode='replicate')
    gy = F.pad(gy, (0, 0, 0, 1), mode='replicate')
    edge_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    k = max(3, int(6 * sigma + 1) | 1)
    pad_size = k // 2
    edge_mag = F.avg_pool2d(
        F.pad(edge_mag, [pad_size] * 4, mode='replicate'),
        kernel_size=k, stride=1, padding=0)

    batch_max = edge_mag.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    edge_mag = edge_mag / (batch_max + 1e-8)
    weight_map = base_weight + (edge_weight - base_weight) * edge_mag
    return weight_map
