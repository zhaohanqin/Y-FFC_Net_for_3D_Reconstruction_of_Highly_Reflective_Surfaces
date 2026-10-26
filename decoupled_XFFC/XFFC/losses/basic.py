"""Basic reconstruction losses and gradient/highlight helpers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _safe_tensor(tensor, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=None):
    """Numerical guard used only at high-risk boundaries.

    It prevents one unstable FFT/attention activation from turning the whole
    training run into persistent NaN/Inf while keeping normal finite values
    unchanged.
    """
    tensor = torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)
    if clamp is not None:
        tensor = tensor.clamp(clamp[0], clamp[1])
    return tensor

def charbonnier_loss(pred, gt, eps=1e-3):
    """[P13][SAFE] Charbonnier 损失：误差 0 处光滑可导，后期梯度更稳定。

    eps 由 1e-6 提升至 1e-3：在 fp16 下 1e-6 已接近下溢，反向梯度被 1/eps 放大
    可达 1e6 量级；1e-3 在精度损失可忽略（典型误差 0.01~0.5）的同时显著稳定反向。
    """
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

def _as_batched_4d(tensor):
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0), 2
    if tensor.dim() == 3:
        return tensor.unsqueeze(0), 3
    if tensor.dim() == 4:
        return tensor, 4
    raise ValueError(f"Expected a 2D, 3D, or 4D tensor, got shape {tuple(tensor.shape)}")

def _restore_tensor_dim(tensor, original_dim):
    if original_dim == 2:
        return tensor.squeeze(0).squeeze(0)
    if original_dim == 3:
        return tensor.squeeze(0)
    return tensor

def sobel_magnitude(image):
    """[SAFE] Sobel 幅值，反向稳定。

    原版 sqrt(gx²+gy²) 在零梯度区域 sqrt(0) 的反向导数是 ∞。
    加入 eps=1e-6 保护：sqrt(gx²+gy²+1e-6) 在 0 处梯度为 0，反向安全。
    """
    image_b, original_dim = _as_batched_4d(image)
    channels = image_b.size(1)
    dtype = image_b.dtype
    device = image_b.device

    sx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    sy = sx.transpose(-1, -2)
    sx = sx.repeat(channels, 1, 1, 1)
    sy = sy.repeat(channels, 1, 1, 1)

    padded = F.pad(image_b, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, sx, groups=channels)
    gy = F.conv2d(padded, sy, groups=channels)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)  # [SAFE] eps=1e-6 防 sqrt(0) 反向爆炸
    return _restore_tensor_dim(mag, original_dim)

def normalize_per_sample(image, quantile=0.995):
    """[SAFE] 按样本对幅值图按分位数归一化到 [0,1]。

    数值安全要点：
      原版 scale.clamp_min(1e-8) 在反向时，对 `values / scale` 求导会得到
      `-values / scale²` 量级的梯度；当 scale ≈ 1e-8 时反向梯度会被放大 1e16 倍。
      改成 clamp_min(1e-3) 是相对安全的（典型 Sobel 幅值在 [0, sqrt(8)] 之间，
      1e-3 远小于正常 scale，但远大于致命的反向爆炸阈值）。
    """
    image_b, original_dim = _as_batched_4d(image)
    dtype = image_b.dtype
    values = image_b.float()
    flat = values.flatten(1)
    scale = torch.quantile(flat, quantile, dim=1, keepdim=True)
    scale = scale.view(-1, 1, 1, 1).clamp_min(1e-3)
    normalized = (values / scale).clamp(0.0, 1.0).to(dtype)
    return _restore_tensor_dim(normalized, original_dim)

def normalized_sobel_magnitude(image, quantile=0.995):
    return normalize_per_sample(sobel_magnitude(image), quantile=quantile)

def compute_input_gradient_target(input_img):
    """Input-image Sobel gradient target normalized per sample to [0, 1]."""
    return normalized_sobel_magnitude(input_img, quantile=0.995)

def compute_input_gradient_weight_map(gt_gradient, base_weight=1.0,
                                      edge_weight=5.0):
    grad_b, original_dim = _as_batched_4d(gt_gradient)
    weight = base_weight + (edge_weight - base_weight) * grad_b.clamp(0.0, 1.0)
    return _restore_tensor_dim(weight, original_dim)

def compute_highlight_prior(input_img, threshold=0.7, softness=0.1):
    """Soft highlight prior from input brightness, smoothed for stable weighting."""
    img_b, original_dim = _as_batched_4d(input_img)
    img_b = img_b.clamp(0.0, 1.0)
    softness = max(float(softness), 1e-6)
    prior = torch.sigmoid((img_b - threshold) / softness)
    prior = F.avg_pool2d(prior, kernel_size=5, stride=1, padding=2)
    return _restore_tensor_dim(prior.clamp(0.0, 1.0), original_dim)

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
        # ★[D8 V3.2] 小尺寸输入 (H<48 或 W<48) 时 relu3_3 层特征几乎无意义，
        # 直接返回 0 避免不稳定梯度污染主任务。
        # 训练分辨率 (128,160) 走全路径；后续 stage 也安全。
        if pred.shape[-1] < 48 or pred.shape[-2] < 48:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
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
    """根据镜面反射分量的梯度生成边缘权重图，用于边缘感知加权 L1 损失。

    [SAFE] sqrt eps 由 1e-8 提升至 1e-4，避免低梯度区反向爆炸；
    输入通常已 detach（gt_specular），但保持稳定性约定。
    """
    gx = gt_specular[:, :, :, 1:] - gt_specular[:, :, :, :-1]
    gy = gt_specular[:, :, 1:, :] - gt_specular[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0), mode='replicate')
    gy = F.pad(gy, (0, 0, 0, 1), mode='replicate')
    edge_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-4)

    k = max(3, int(6 * sigma + 1) | 1)
    pad_size = k // 2
    edge_mag = F.avg_pool2d(
        F.pad(edge_mag, [pad_size] * 4, mode='replicate'),
        kernel_size=k, stride=1, padding=0)

    batch_max = edge_mag.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    edge_mag = edge_mag / batch_max.clamp_min(1e-4)
    weight_map = base_weight + (edge_weight - base_weight) * edge_mag
    return weight_map
