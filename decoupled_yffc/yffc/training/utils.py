"""Reusable training utilities split from ../../Y_FFC_unified.py."""

import math

import torch
import torch.nn as nn


class EMA:
    """指数移动平均模型参数，提升推理稳定性。"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + \
                              (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class DynamicLossWeightScheduler:
    """根据训练进度动态调整各 loss 项的权重（修复版：60% 后冻结）。"""

    def __init__(self, total_epochs):
        self.total_epochs = max(total_epochs, 1)

    def get_weights(self, current_epoch):
        progress = min(current_epoch / self.total_epochs, 1.0)
        # 修复：在 60% 处截断变化，之后权重不再改变
        p = min(progress / 0.6, 1.0)
        weights = {
            'lambda_ssim':       1.0,
            'lambda_l1':         max(0.5, 1.0 - 0.5 * p),
            'lambda_freq':       min(0.15, 0.05 + 0.10 * p),
            'lambda_perceptual': min(0.10, 0.02 + 0.08 * p),
            'lambda_spec':       0.5,
            'lambda_stat':       0.0,     # 移除冗余 stat
            'lambda_grad':       max(0.3, 0.5 - 0.2 * p),
        }
        return weights

    def step(self, criterion, current_epoch):
        weights = self.get_weights(current_epoch)
        criterion.update_weights(**weights)
        return weights


def build_optimizer(model, base_lr=2e-4, weight_decay=1e-4):
    """为 YFFC 模型构建分层学习率 AdamW 优化器。"""
    encoder_pretrained = []
    ffc_params = []
    decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ('encoder.vit_branch.vit' in name or
            'encoder.stem' in name or
            'encoder.layer' in name):
            encoder_pretrained.append(param)
        elif 'ffc' in name.lower():
            ffc_params.append(param)
        elif ('gradient_decoder' in name or
              'main_decoder' in name or
              'enhancement_layers' in name):
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': encoder_pretrained, 'lr': base_lr * 0.1, 'name': 'encoder_pretrained'},
        {'params': other_params, 'lr': base_lr * 0.5, 'name': 'fusion_layers'},
        {'params': decoder_params, 'lr': base_lr, 'name': 'decoders'},
        {'params': ffc_params, 'lr': base_lr * 0.5, 'name': 'ffc_modules'},
    ]
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    for g in param_groups:
        n_params = sum(p.numel() for p in g['params'])
        print(f"  优化器分组 [{g['name']}]: {n_params:,} 参数, lr={g['lr']:.2e}")

    return optimizer


def build_scheduler(optimizer, warmup_iters=500, total_epochs=200,
                    iters_per_epoch=100, eta_min=1e-6):
    """
    [P6-FIX] Warmup + 余弦退火调度器（修复版：防止 LR 回弹）。

    关键修复：
      原版使用 CosineAnnealingLR，当训练超过 T_max 迭代后，LR 会自动周期性
      回升，导致训练后期损失突然增大。

      修复方案：使用 LambdaLR + 自定义余弦衰减函数，超过 T_max 后 LR 钳位在
      eta_min，不再回弹。
    """
    from torch.optim.lr_scheduler import LinearLR, LambdaLR, SequentialLR

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
    total_iters = total_epochs * iters_per_epoch
    cosine_iters = max(total_iters - warmup_iters, 1)

    # 自定义余弦衰减：超过 cosine_iters 后 LR 钳位在 eta_min，不循环
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def _cosine_no_restart(current_iter):
        """余弦退火 lambda，到达 T_max 后保持 eta_min，不回弹。"""
        if current_iter >= cosine_iters:
            # 钳位在最低学习率比例
            return eta_min / base_lrs[0] if base_lrs[0] > 0 else eta_min
        progress = current_iter / cosine_iters
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # 从 1.0 衰减到 eta_min / base_lr
        min_ratio = eta_min / base_lrs[0] if base_lrs[0] > 0 else eta_min
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    cosine_scheduler = LambdaLR(optimizer, lr_lambda=_cosine_no_restart)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_iters])
    return scheduler


def replace_bn_with_gn(module, num_groups=8):
    """递归将模型中所有 BatchNorm2d 替换为 GroupNorm。"""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            groups = min(num_groups, num_channels)
            while num_channels % groups != 0 and groups > 1:
                groups -= 1
            gn = nn.GroupNorm(groups, num_channels, eps=child.eps)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)
