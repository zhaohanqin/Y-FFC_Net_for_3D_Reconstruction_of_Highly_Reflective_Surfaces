"""Reusable training utilities split from ../../X_FFC_unified.py."""

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
    """根据训练进度动态调整各 loss 项的权重（FIX-D:常量权重 + 例外两项退火）。

    ★[FIX-D] 同步 train.py 的 ImprovedWeightScheduler:
      所有 aux 权重锁 plateau,从 epoch 0 起不变,只有 `grad_spec` 和 `r_smooth`
      保留后期退火,`s_cons` 直接置 0。详细背景见 train.py 中的同名调度器注释。
    """

    def __init__(self, total_epochs):
        self.total_epochs = max(total_epochs, 1)

    def get_weights(self, current_epoch):
        progress = min(current_epoch / self.total_epochs, 1.0)
        # 后期退火因子:仅 grad_spec 和 r_smooth 用
        anneal = 1.0 if progress <= 0.6 else max(0.2, 1.0 - (progress - 0.6) / 0.4 * 0.8)
        weights = {
            'lambda_ssim':        1.0,
            'lambda_l1':          0.5,
            'lambda_freq':        0.15,
            'lambda_perceptual':  0.10,
            'lambda_spec':        0.5,
            'lambda_stat':        0.0,
            'lambda_grad':        0.3,
            'lambda_grad_edge':   0.8,
            'lambda_grad_spec':   0.03 * anneal,
            'lambda_focal':       0.8,
            'lambda_dice':        0.3,
            'lambda_gc':          0.2,
            'lambda_cs':          0.05,
            'lambda_s_lik':       1.0,
            # ★[FIX-D] s_cons 锁 0:FIX-A 之后结构性抗任务,直接禁用
            'lambda_s_cons':      0.0,
            'lambda_spec_peak':   0.5,
            'lambda_r_focal_hl':  0.8,
            'lambda_r_dual_cons': 0.15,
            'lambda_r_smooth':    0.05 * anneal,
        }
        return weights

    def step(self, criterion, current_epoch):
        weights = self.get_weights(current_epoch)
        criterion.update_weights(**weights)
        return weights


def build_optimizer(model, base_lr=2e-4, weight_decay=1e-4):
    """为 XFFC 模型构建分层学习率 AdamW 优化器。"""
    encoder_pretrained = []
    swin_skip_params = []
    ffc_params = []
    decoder_params = []
    new_module_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ('sa1' in name or 'sa2' in name or 'sa3' in name or
            'sa4' in name or 'hf_enhance' in name or
            'refinement_head' in name or
            'top_down_refine' in name):
            new_module_params.append(param)
        elif ('fuse_h4' in name or 'fuse_h8' in name or
              'swin16_proj' in name):
            # 第三阶段 AdaptiveSkipFusion 与 H/16 投影层使用 base_lr，避免 beta 初始为 0 后学习过慢。
            swin_skip_params.append(param)
        elif ('encoder.swin_branch.swin' in name or
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
        {'params': swin_skip_params, 'lr': base_lr, 'name': 'adaptive_skip_fusion'},
        {'params': other_params, 'lr': base_lr * 0.5, 'name': 'fusion_layers'},
        {'params': decoder_params, 'lr': base_lr, 'name': 'decoders'},
        {'params': ffc_params, 'lr': base_lr * 0.5, 'name': 'ffc_modules'},
        {'params': new_module_params, 'lr': base_lr, 'name': 'new_modules'},
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
