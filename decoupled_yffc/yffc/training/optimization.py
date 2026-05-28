"""Enhanced optimizer and dynamic loss-weight scheduling."""

import torch


class ImprovedWeightScheduler:
    """
    ★[FIX-D] 全量改用「常量权重 + 例外两项退火」,根治"权重曲线驱动的中期回升"。

    旧版策略(已废弃):
      早期(0~30%)以 L1 快速收敛 → 中期(30~60%)加大 focal/dice/freq →
      后期(60~100%) 权重冻结。
    旧版问题(实测):
      多个 aux 损失(dice / focal / grad_edge / perceptual / s_lik / spec_peak)
      的数值在 epoch ≈ 10 后就已经进入 plateau,但权重还在沿 30%~60% 进度
      持续线性爬升。"恒定的子损失值 × 单调增大的权重 = 加权贡献单调增大",
      在你 50-epoch 训练里仅 dice 一项贡献就从 0.080 涨到 0.165,叠加其他
      三项一共净增 ≈ +0.10,正好等于观察到的总 loss 中期回升幅度。这不是
      模型变差,纯粹是调度策略本身的算术现象。

    新版策略:
      ① 所有 aux 权重直接锁 plateau,从 epoch 0 起不变。理由:
          - 你的模型 SSIM 在 epoch ~13 就到 0.98+,主任务收敛根本不需要
            "先抑后扬"的课程;
          - 锁定权重后,总 loss 的演化只反映"sub_loss 数值变化"本身,
            曲线天然单调下降。
      ② 仅 `grad_spec` 和 `r_smooth` 保留 60% 之后的退火(沿用 FIX-B 的设计):
          这两项确实与主任务的细节恢复冲突,后期必须让路。
      ③ `lambda_s_cons` 直接置 0:FIX-A 把 pred_s_lik 改成二值监督后,
         它的值域已经从连续 [0,1] 收缩到 ≈{0,1},而 est_b 仍预测连续值,
         再让它们做 L1 一致性是结构性抗任务,实测 s_cons 单调上涨
         从 0.003 → 0.04 就是证据。
    """
    def __init__(self, total_epochs):
        self.T = max(total_epochs, 1)

    def get_weights(self, ep):
        p = min(ep / self.T, 1.0)
        # 后期退火因子:仅 grad_spec 和 r_smooth 用(主任务冲突项,细节阶段必须让路)
        anneal = 1.0 if p <= 0.6 else max(0.2, 1.0 - (p - 0.6) / 0.4 * 0.8)

        return {
            'lambda_ssim':        1.0,
            'lambda_l1':          0.5,
            'lambda_freq':        0.15,
            'lambda_perceptual':  0.10,
            'lambda_spec':        0.5,
            'lambda_stat':        0.0,   # [P5] 移除冗余 stat 损失
            'lambda_grad':        0.3,
            'lambda_grad_edge':   0.8,
            # ★[FIX-D] grad_spec 保留后期退火:仍是主任务冲突项,后期细节恢复阶段让路
            'lambda_grad_spec':   0.03 * anneal,
            'lambda_focal':       0.8,
            'lambda_dice':        0.3,
            'lambda_gc':          0.2,
            'lambda_cs':          0.05,
            'lambda_s_lik':       1.0,
            # ★[FIX-D 关键] s_cons 锁 0:FIX-A 把 pred_s_lik 改二值监督后,
            #   它和 est_b 已经是结构性不同的两种预测(二值 vs 连续),
            #   再做一致性是抗任务。实测训练曲线中 s_cons 从 0.003
            #   单调涨到 0.04,直接禁用。
            'lambda_s_cons':      0.0,
            'lambda_spec_peak':   0.5,
            'lambda_r_focal_hl':  0.8,
            'lambda_r_dual_cons': 0.15,
            # ★[FIX-D] r_smooth 保留后期退火(沿用 FIX-B)
            'lambda_r_smooth':    0.05 * anneal,
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
            # 第三阶段 AdaptiveSkipFusion 与 H/16 投影层使用 base_lr，避免 beta 初始为 0 后学习过慢。
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
