"""Enhanced optimizer and dynamic loss-weight scheduling."""

import torch


class ImprovedWeightScheduler:
    """
    策略（修复版）：
      早期(0~30%)   以 L1 快速收敛
      中期(30~60%)  加大 focal/dice/freq
      后期(60~100%) 权重冻结在平台值，不再变化 ← 关键修复

    原版问题：权重持续线性增大直到最后一轮，导致后期优化目标不断漂移，
    与学习率衰减产生冲突，叠加 LR 回弹造成损失突增。
    """
    def __init__(self, total_epochs):
        self.T = max(total_epochs, 1)

    def get_weights(self, ep):
        # 在 60% 训练进度处截断，之后权重不再变化
        p = min(ep / self.T, 1.0)
        p_clamped = min(p / 0.6, 1.0)  # 在 60% 处达到最终值

        return {
            'lambda_ssim':       1.0,
            'lambda_l1':         max(0.5, 1.0 - 0.5 * p_clamped),
            'lambda_freq':       min(0.15, 0.05 + 0.10 * p_clamped),      # 上限降低
            'lambda_perceptual': min(0.10, 0.02 + 0.08 * p_clamped),      # 上限降低
            'lambda_spec':       0.5,
            'lambda_stat':       0.0,    # [P5] 移除冗余 stat 损失
            'lambda_grad':       max(0.3, 0.5 - 0.2 * p_clamped),
            'lambda_focal':      min(0.8, 0.3 + 0.5 * p_clamped),         # 上限降低
            'lambda_dice':       min(0.3, 0.1 + 0.2 * p_clamped),         # 上限降低
            'lambda_gc':         min(0.2, 0.05 + 0.15 * p_clamped),       # 上限降低
            'lambda_cs':         min(0.05, 0.02 + 0.03 * p_clamped),
        }

    def step(self, criterion, ep):
        w = self.get_weights(ep)
        criterion.update_weights(**w)
        return w


def build_optimizer_v2(model, base_lr=2e-4, weight_decay=1e-4):
    """分层学习率：预训练编码器 0.1×, FFC/融合 0.5×, 解码器/新模块 1.0×"""
    enc_pre, swin_skip_p, ffc_p, dec_p, new_p, other_p = [], [], [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ('sa1' in name or 'sa2' in name or 'sa3' in name or
            'sa4' in name or 'hf_enhance' in name or
            'refinement_head' in name or
            'top_down_refine' in name):
            new_p.append(param)
        elif ('fuse_h4' in name or 'fuse_h8' in name or
              'swin16_proj' in name):
            # AdaptiveSkipFusion 与 H/16 投影层使用 base_lr，避免 beta 初始为 0 后学习过慢。
            swin_skip_p.append(param)
        elif 'encoder.swin_branch.swin' in name or 'encoder.stem' in name or 'encoder.layer' in name:
            enc_pre.append(param)
        elif 'ffc' in name.lower():
            ffc_p.append(param)
        elif 'gradient_decoder' in name or 'main_decoder' in name or 'enhancement_layers' in name:
            dec_p.append(param)
        else:
            other_p.append(param)

    groups = [
        {'params': enc_pre, 'lr': base_lr * 0.1, 'name': 'encoder_pretrained'},
        {'params': swin_skip_p, 'lr': base_lr, 'name': 'adaptive_skip_fusion'},
        {'params': other_p, 'lr': base_lr * 0.5, 'name': 'fusion'},
        {'params': dec_p,   'lr': base_lr,        'name': 'decoders'},
        {'params': ffc_p,   'lr': base_lr * 0.5, 'name': 'ffc'},
        {'params': new_p,   'lr': base_lr,        'name': 'new_modules'},
    ]
    groups = [g for g in groups if g['params']]
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    for g in groups:
        n = sum(p.numel() for p in g['params'])
        print(f"  [{g['name']}] {n:,} params, lr={g['lr']:.8f}")
    return opt


def format_lr_groups(optimizer):
    """按参数组输出学习率，便于检查分层 LR 是否按预期运行。"""
    parts = []
    for idx, group in enumerate(optimizer.param_groups):
        name = group.get('name', f'group_{idx}')
        parts.append(f"{name}={group['lr']:.2e}")
    return ", ".join(parts)
