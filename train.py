"""
train_v2.py — Y-FFC V2 工程化训练脚本（增强可观测性版）
=========================================================

整合全部改进方案，同时遵循 train.py 的工程规范：
  - 顶部集中参数配置
  - tqdm 进度条 + 结构化日志
  - 多指标最优模型自动保存（综合 / SSIM / Loss）
  - 训练曲线自动生成（Loss / SSIM / MAE / 子损失）
  - 训练总结文件（training_summary.txt）

核心训练逻辑（V2 增强部分）：
  ⭐⭐⭐⭐⭐ 分层 LR / EMA / GN / Warmup / 动态权重
  ⭐⭐⭐⭐⭐ 输入分辨率自定义 (H, W)，如 (160, 128) / (384, 384)
  ⭐⭐⭐⭐  EnhancedCompositeLoss（SpecularFocal + Dice + GradConsistency）
  ⭐⭐⭐⭐  高反光模拟数据增强 HighlightAugmentation
  ⭐⭐⭐   网络内嵌 SpecularAwareAttention + HighFrequencyEnhancement
  ⭐⭐     梯度一致性损失 GradientConsistencyLoss
  ⭐⭐     OHEM 难样本挖掘

增强可观测性：
  ✅ 终端日志完整展示全部子损失项（SSIM / L1 / Freq / Perceptual / Spec / Stat / Grad / Focal / Dice / GC）
  ✅ 验证阶段额外计算 MSE / PSNR 指标
  ✅ 验证样本可视化（输入 / GT / 预测 / 镜面反射 / 梯度 / 差异图）
  ✅ 子损失历史曲线独立绘制
  ✅ 训练总结文件包含完整子损失信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import time
import math
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 从统一文件导入所有模型相关模块 ──
# 原分散在 Y_FFC_v2.py / Y_FFC.py / specular_modules.py 中，
# 现已整合到 X_FFC_unified.py 单文件。
from X_FFC_unified import (
    XFFC,
    build_scheduler,
    EMA,
    replace_bn_with_gn,
    # 高反光专用模块（原 specular_modules.py）
    EnhancedCompositeLoss,
    EnhancedPairedTransform,
    HighlightAugmentation,
    compute_input_gradient_target,
)


# ════════════════════════════════════════════════════════════════════════
# 改进版数据集
# ════════════════════════════════════════════════════════════════════════

class ImprovedFringeDataset(Dataset):
    """
    改进版数据集：
      - 支持自定义分辨率 (H, W)，如 (160, 128)、(384, 384)
      - EnhancedPairedTransform（含高反光模拟）
      - gt_gradient 使用输入图像 Sobel 梯度幅值

    resize 参数说明：
      - int         : 正方形，如 384 → (384, 384)
      - (H, W) 元组 : 任意矩形，如 (160, 128)
      - None        : 不 resize，保持原图大小
    """
    def __init__(self, input_dir, target_dir, resize=(160, 128), paired_transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.paired_transform = paired_transform

        # 统一 resize 格式为 (H, W) 或 None
        if resize is None:
            self.resize_hw = None
        elif isinstance(resize, (int, float)):
            self.resize_hw = (int(resize), int(resize))
        elif isinstance(resize, (tuple, list)) and len(resize) == 2:
            self.resize_hw = (int(resize[0]), int(resize[1]))
        else:
            raise ValueError(f"resize 参数格式不正确: {resize}，应为 int / (H, W) / None")

        self.files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])

        resize_str = f"{self.resize_hw[0]}×{self.resize_hw[1]}" if self.resize_hw else "原始尺寸"
        print(f"[Dataset] {len(self.files)} 样本, resize={resize_str}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        inp_path = os.path.join(self.input_dir, name)

        # 查找匹配的目标图
        candidates = [
            name,
            name.replace("input_", "target_", 1) if name.startswith("input_") else name,
            name.replace("input", "target", 1) if name.startswith("input") else name,
        ]
        tgt_path = None
        for c in candidates:
            p = os.path.join(self.target_dir, c)
            if os.path.exists(p):
                tgt_path = p
                break
        if tgt_path is None:
            base, ext = os.path.splitext(name)
            digits = ''.join(ch for ch in base if ch.isdigit())
            if digits:
                p = os.path.join(self.target_dir, f"target_{digits}{ext}")
                if os.path.exists(p):
                    tgt_path = p
        if tgt_path is None:
            raise FileNotFoundError(f"找不到 '{name}' 的匹配目标, 目录: '{self.target_dir}'")

        inp_pil = Image.open(inp_path).convert('L')
        tgt_pil = Image.open(tgt_path).convert('L')

        # resize: self.resize_hw = (H, W)，PIL.resize 需要 (W, H)
        if self.resize_hw is not None:
            pil_size = (self.resize_hw[1], self.resize_hw[0])  # (W, H)
            inp_pil = inp_pil.resize(pil_size, Image.LANCZOS)
            tgt_pil = tgt_pil.resize(pil_size, Image.LANCZOS)

        inp = torch.from_numpy(np.array(inp_pil, dtype=np.float32) / 255.0).unsqueeze(0)
        tgt = torch.from_numpy(np.array(tgt_pil, dtype=np.float32) / 255.0).unsqueeze(0)

        if self.paired_transform is not None:
            inp, tgt, gt_spec, gt_grad = self.paired_transform(inp, tgt)
        else:
            gt_spec = (inp - tgt).clamp(0, 1)
            gt_grad = compute_input_gradient_target(inp)

        return inp, tgt, gt_spec, gt_grad


# ════════════════════════════════════════════════════════════════════════
# 扩展版动态权重调度（新增 focal/dice/gc 权重调度）
# ════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════
# 扩展版分层学习率
# ════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════
# EnhancedCompositeLoss 返回的 loss_dict 中可能包含的键名列表
# 用于统一累积、日志输出和历史追踪
# ════════════════════════════════════════════════════════════════════════

# 主要的子损失键名（EnhancedCompositeLoss 的 loss_dict 中可能出现的所有键）
# 'total' 和 'ssim' 是必有的；其余根据配置可能存在也可能为 0
SUB_LOSS_KEYS = [
    'total', 'ssim', 'ssim_single_loss', 'msssim_loss', 'ssim_mixed_loss',
    'l1', 'freq', 'perceptual',
    'spec', 'stat', 'grad', 'grad_edge', 'grad_spec',
    'focal', 'dice', 'gc', 'cs',
    # [NEW] spec_likelihood 监督相关
    's_lik', 's_cons',
    # [NEW S3] 高光峰值欠预测惩罚
    'spec_peak',
    # ★[V3.2 新增] 漫反射 r 直接监督系列
    'r_focal_hl', 'r_dual_cons', 'r_smooth',
]

# 用于日志输出时的友好名称映射
SUB_LOSS_DISPLAY_NAMES = {
    'total':      'Total Loss',
    'ssim':       'SSIM Loss (1-SSIM)',
    'ssim_single_loss': 'SSIM Single Loss',
    'msssim_loss': 'MS-SSIM Loss',
    'ssim_mixed_loss': 'SSIM Mixed Loss',
    'l1':         'L1 / MAE Loss',
    'freq':       'Frequency Loss',
    'perceptual': 'Perceptual Loss',
    'spec':       'Specular Loss',
    'stat':       'Statistical Loss',
    'grad':       'Gradient SSIM Loss',
    'grad_edge':  'Gradient Edge Loss',
    'grad_spec':  'Gradient-Specular Consistency',
    'focal':      'Focal Loss',
    'dice':       'Dice Loss',
    'gc':         'Grad Consistency',
    'cs':         'Cross-scale Consistency',
    's_lik':      'Spec-Likelihood Sup (BCE)',
    's_cons':     'Spec-Likelihood Bi-Cons',
    'spec_peak':  'Spec Peak (Under-Pred)',
    # ★[V3.2 新增] 漫反射 r 直接监督
    'r_focal_hl':  'R Focal Highlight',
    'r_dual_cons': 'R Dual-Head Consistency',
    'r_smooth':    'R Highlight Smooth',
}


# ════════════════════════════════════════════════════════════════════════
# 统一指标计算模块 MetricTracker
# ════════════════════════════════════════════════════════════════════════

class MetricTracker:
    """
    统一指标计算器，供 train_one_epoch 和 validate 共用。
    避免训练/验证阶段指标计算逻辑不一致或重复代码。

    追踪指标：
      - SSIM（漫反射分支，1 - ssim_loss，越高越好）
      - MAE （漫反射分支，L1 距离）
      - MSE （漫反射分支，均方误差）
      - PSNR（漫反射分支，峰值信噪比，dB）
      - 全部子损失项（来自 EnhancedCompositeLoss 的 loss_dict）

    PSNR 公式（图像范围 [0, 1]）：
      PSNR = 10 × log10(1 / MSE)
    当 MSE 极小时（< 1e-10），PSNR 上限设为 60 dB。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.tot_loss = 0.0
        self.tot_ssim = 0.0
        self.tot_mae = 0.0
        self.tot_mse = 0.0
        self.tot_psnr = 0.0
        self.sub_accum = defaultdict(float)

    def update(self, loss_dict, pred, target):
        """
        更新一个 batch 的指标。

        参数:
            loss_dict : dict，EnhancedCompositeLoss 返回的 loss_dict
                        必须包含 'total' 和 'ssim'，可选包含 'l1'/'mae' 等
            pred      : Tensor，漫反射分支预测输出 (B, 1, H, W)
            target    : Tensor，漫反射分支真值 (B, 1, H, W)
        """
        self.n += 1

        # 主指标
        self.tot_loss += loss_dict['total']
        self.tot_ssim += (1.0 - loss_dict['ssim'])
        self.tot_mae += loss_dict.get('l1', loss_dict.get('mae', 0.0))

        # MSE（漫反射分支）
        mse_val = torch.mean((pred - target) ** 2).item()
        self.tot_mse += mse_val

        # PSNR = 10 × log10(1 / MSE)，图像范围 [0, 1]
        if mse_val > 1e-10:
            self.tot_psnr += 10.0 * math.log10(1.0 / mse_val)
        else:
            self.tot_psnr += 60.0  # MSE 极小时设上限

        # 子损失项
        for k, v in loss_dict.items():
            self.sub_accum[k] += v

    def compute(self):
        """
        计算 epoch 平均指标。

        返回:
            avg_loss  : float
            avg_ssim  : float（越高越好）
            avg_mae   : float
            avg_mse   : float
            avg_psnr  : float（dB）
            avg_subs  : dict，所有子损失项的平均值
        """
        d = max(self.n, 1)
        return (
            self.tot_loss / d,
            self.tot_ssim / d,
            self.tot_mae / d,
            self.tot_mse / d,
            self.tot_psnr / d,
            {k: v / d for k, v in self.sub_accum.items()},
        )


# ════════════════════════════════════════════════════════════════════════
# 数值稳定性辅助：防止一次非有限梯度污染后续所有 batch
# ════════════════════════════════════════════════════════════════════════

def _tensors_are_finite(*items):
    for item in items:
        if isinstance(item, torch.Tensor) and not torch.isfinite(item).all():
            return False
        if isinstance(item, dict):
            for value in item.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    return False
    return True


def _repair_nonfinite_model(model, ema=None):
    """Restore non-finite parameters from EMA when possible, else sanitize.

    If one optimizer step corrupts parameters, simply skipping later NaN losses
    is not enough.  This function makes the next batch recoverable.
    """
    repaired = 0
    ema_shadow = getattr(ema, 'shadow', None) if ema is not None else None
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                if ema_shadow is not None and name in ema_shadow and torch.isfinite(ema_shadow[name]).all():
                    param.data.copy_(ema_shadow[name].to(device=param.device, dtype=param.dtype))
                else:
                    param.data.copy_(torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0))
                repaired += 1
        for buffer in model.buffers():
            if torch.is_floating_point(buffer) and not torch.isfinite(buffer).all():
                buffer.copy_(torch.nan_to_num(buffer, nan=0.0, posinf=1.0, neginf=-1.0))
                repaired += 1
    return repaired


def _sanitize_optimizer_state(optimizer):
    """Clear non-finite Adam/AdamW state tensors after a repaired step."""
    repaired = 0
    with torch.no_grad():
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value) and not torch.isfinite(value).all():
                    value.copy_(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
                    repaired += 1
    return repaired


def _format_loss_diag(loss_dict, top_k=8):
    """[DIAG] 把 loss_dict 格式化为紧凑诊断字符串，按值降序排前 top_k 项。

    grad NaN/Inf 时调用，帮助定位是哪个具体子损失项数值异常。
    若某个项是 nan/inf，会用 'NaN' / 'Inf' 标记。
    """
    if not isinstance(loss_dict, dict) or not loss_dict:
        return '<empty>'
    items = []
    for k, v in loss_dict.items():
        try:
            fv = float(v)
            if math.isnan(fv):
                items.append((k, float('inf'), 'NaN'))
            elif math.isinf(fv):
                items.append((k, float('inf'), 'Inf'))
            else:
                items.append((k, abs(fv), f'{fv:.4g}'))
        except (TypeError, ValueError):
            items.append((k, 0.0, str(v)))
    # 按绝对值降序，并把 NaN/Inf 排在最前
    items.sort(key=lambda t: t[1], reverse=True)
    parts = [f'{k}={s}' for k, _, s in items[:top_k]]
    return '{' + ', '.join(parts) + '}'


# ════════════════════════════════════════════════════════════════════════
# 训练核心（一个 epoch）—— 增强版：返回全部子损失分量 + MSE/PSNR
# ════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    scaler, ema, device, accum=4, max_norm=1.0, epoch=0, total_epochs=0,
                    use_amp=True):
    """
    执行一个 epoch 的训练。

    返回:
        avg_loss  : float，平均总损失
        avg_ssim  : float，平均 SSIM（漫反射分支，越高越好）
        avg_mae   : float，平均 MAE（漫反射分支）
        avg_mse   : float，平均 MSE（漫反射分支）
        avg_psnr  : float，平均 PSNR（漫反射分支，dB）
        avg_subs  : dict，所有子损失项的 epoch 平均值
    """
    model.train()
    tracker = MetricTracker()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f'Train {epoch+1}/{total_epochs}', leave=False)
    for i, (inp, tgt, gt_s, gt_g) in pbar:
        inp, tgt = inp.to(device), tgt.to(device)
        gt_s, gt_g = gt_s.to(device), gt_g.to(device)

        if not _tensors_are_finite(inp, tgt, gt_s, gt_g):
            tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: input/target NaN/Inf, 跳过')
            optimizer.zero_grad(set_to_none=True)
            continue

        amp_enabled = (device.type == 'cuda' and bool(use_amp))
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            out = model(inp)

        if not _tensors_are_finite(out):
            repaired = _repair_nonfinite_model(model, ema)
            repaired += _sanitize_optimizer_state(optimizer)
            tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: output NaN/Inf, 跳过; repaired={repaired}')
            optimizer.zero_grad(set_to_none=True)
            continue

        with torch.amp.autocast('cuda', enabled=False):
            # [NEW] 取出 spec_likelihood 头的输出 pred_s_lik (可能不存在, 用 .get 兼容)
            pred_s_lik = out.get('s')
            # ★[V3.2 新增] 取出 MainDecoder 直接漫反射头输出 pred_r_direct
            pred_r_direct = out.get('r_direct')
            loss, ld = criterion(
                out['r'].float(), tgt.float(),
                out['b'].float(), gt_s.float(),
                out['g'].float(), gt_g.float(),
                out.get('cs'), input_img=inp.float(),
                pred_s_lik=pred_s_lik.float() if pred_s_lik is not None else None,
                pred_r_direct=pred_r_direct.float() if pred_r_direct is not None else None)
            loss = loss / accum

        # NaN / Inf 保护
        if not torch.isfinite(loss):
            repaired = _repair_nonfinite_model(model, ema)
            repaired += _sanitize_optimizer_state(optimizer)
            tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: loss NaN/Inf, 跳过; repaired={repaired}')
            optimizer.zero_grad(set_to_none=True)
            continue

        # [FIX] 损失尖峰保护：如果单批损失超过历史均值的 10 倍，跳过该批次
        # 防止异常样本或数值不稳定导致的梯度爆炸
        _current_loss_val = ld['total']
        if tracker.n > 5:  # 至少积累 5 批数据后才判断
            _avg_so_far = tracker.tot_loss / tracker.n
            if _avg_so_far > 0 and _current_loss_val > _avg_so_far * 10.0:
                tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: '
                           f'loss spike {_current_loss_val:.4f} >> avg {_avg_so_far:.4f}, 跳过')
                optimizer.zero_grad(set_to_none=True)
                continue

        if device.type == 'cuda' and bool(use_amp):
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            if device.type == 'cuda' and bool(use_amp):
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=False)
                if not torch.isfinite(grad_norm):
                    repaired = _repair_nonfinite_model(model, ema)
                    repaired += _sanitize_optimizer_state(optimizer)
                    # [DIAG] 打印各子损失值,辅助定位 NaN 源
                    _diag = _format_loss_diag(ld)
                    tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: grad NaN/Inf, '
                               f'跳过 step; repaired={repaired}; sub_losses={_diag}')
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=False)
                if not torch.isfinite(grad_norm):
                    repaired = _repair_nonfinite_model(model, ema)
                    repaired += _sanitize_optimizer_state(optimizer)
                    _diag = _format_loss_diag(ld)
                    tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: grad NaN/Inf, '
                               f'跳过 step; repaired={repaired}; sub_losses={_diag}')
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
            repaired = _repair_nonfinite_model(model, ema)
            repaired += _sanitize_optimizer_state(optimizer)
            if repaired:
                tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: optimizer 后发现非有限参数/状态，已修复 {repaired} 项')
            if scheduler:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update()

        # 使用 MetricTracker 统一累积全部指标（含 MSE / PSNR）
        with torch.no_grad():
            tracker.update(ld, out['r'].float(), tgt.float())

        # 更新进度条
        pbar.set_postfix({
            'loss': f"{ld['total']:.4f}",
            'ssim': f"{1 - ld['ssim']:.4f}",
            'l1':   f"{ld.get('l1', ld.get('mae', 0.0)):.4f}",
        })

    avg_loss, avg_ssim, avg_mae, avg_mse, avg_psnr, avg_subs = tracker.compute()
    return avg_loss, avg_ssim, avg_mae, avg_mse, avg_psnr, avg_subs


def _tta_forward(model, inp):
    """[P15] Test-Time Augmentation：4 次翻转推理取平均。

    ★[V3.2] 兼容 dual-head：除 'b' 外，'r_direct'、's' 也参与翻转累积，
    再走相同的"双路融合 + 精修"路径得到最终 r。
    若模型未输出 'r_direct'（旧 V3 ckpt），退回旧版 r = clamp(x - avg_b, 0, 1) → refinement。
    """
    preds_b = []
    preds_r_direct = []
    preds_g = []
    preds_s = []
    last_alpha = None
    for flip_h in [False, True]:
        for flip_v in [False, True]:
            x = inp
            if flip_h:
                x = torch.flip(x, [-1])
            if flip_v:
                x = torch.flip(x, [-2])
            out = model(x)
            # 把 b/r_direct/g/s 各通道按翻转回原方向，再累加
            for key, bucket in (('b', preds_b),
                                ('r_direct', preds_r_direct),
                                ('g', preds_g),
                                ('s', preds_s)):
                v = out.get(key)
                if v is None:
                    continue
                if flip_v:
                    v = torch.flip(v, [-2])
                if flip_h:
                    v = torch.flip(v, [-1])
                bucket.append(v)
            last_alpha = out.get('alpha')
    avg_b = torch.stack(preds_b, dim=0).mean(0) if preds_b else None
    avg_r_direct = torch.stack(preds_r_direct, dim=0).mean(0) if preds_r_direct else None
    avg_g = torch.stack(preds_g, dim=0).mean(0) if preds_g else out.get('g')
    avg_s = torch.stack(preds_s, dim=0).mean(0) if preds_s else None

    # 复刻 XFFC.forward 末尾的双路融合 + 精修，保证 TTA 输出 r 与正常 forward 路径一致
    r_sub = torch.clamp(inp - avg_b, 0.0, 1.0)
    if avg_r_direct is not None:
        # 用最后一次 forward 的 alpha（与训练保持一致；推理时 alpha 已稳定）
        if last_alpha is not None and last_alpha.dim() == 0:
            alpha_val = last_alpha
        else:
            alpha_val = torch.tensor(0.5, device=inp.device)
        r_fused = alpha_val * avg_r_direct + (1.0 - alpha_val) * r_sub
    else:
        r_fused = r_sub
    if hasattr(model, 'refinement_head'):
        r_final = model.refinement_head(r_fused, inp, avg_r_direct if avg_r_direct is not None else r_fused)
    else:
        r_final = r_fused
    result = {'r': r_final, 'b': avg_b, 'g': avg_g}
    if avg_s is not None:
        result['s'] = avg_s
    if avg_r_direct is not None:
        result['r_direct'] = avg_r_direct
    return result


@torch.no_grad()
def validate(model, loader, criterion, device, epoch=0, total_epochs=0,
             use_tta=False):
    """
    执行验证。

    返回:
        avg_loss  : float，平均总损失
        avg_ssim  : float，平均 SSIM（漫反射分支，越高越好）
        avg_mae   : float，平均 MAE（漫反射分支）
        avg_mse   : float，平均 MSE（漫反射分支）
        avg_psnr  : float，平均 PSNR（漫反射分支，dB）
        avg_subs  : dict，所有子损失项的 epoch 平均值
    """
    model.eval()
    tracker = MetricTracker()

    pbar = tqdm(loader, total=len(loader),
                desc=f'Val   {epoch+1}/{total_epochs}', leave=False)
    for inp, tgt, gt_s, gt_g in pbar:
        inp, tgt = inp.to(device), tgt.to(device)
        gt_s, gt_g = gt_s.to(device), gt_g.to(device)

        out = _tta_forward(model, inp) if use_tta else model(inp)  # [P15]
        if not _tensors_are_finite(out):
            continue
        # [NEW] spec_likelihood 头输出
        pred_s_lik = out.get('s')
        # ★[V3.2 新增] MainDecoder 直接漫反射头输出
        pred_r_direct = out.get('r_direct')
        loss, ld = criterion(
            out['r'], tgt, out['b'], gt_s, out['g'], gt_g,
            out.get('cs'), input_img=inp,
            pred_s_lik=pred_s_lik,
            pred_r_direct=pred_r_direct)

        if not torch.isfinite(loss):
            continue

        tracker.update(ld, out['r'].float(), tgt.float())

    return tracker.compute()


# ════════════════════════════════════════════════════════════════════════
# 验证样本可视化（参考 train.py 的 visualize_sample_during_training）
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def visualize_sample_during_training(model, vis_loader, device, output_folder,
                                      current_epoch, num_samples=2):
    """
    训练过程中可视化验证样本结果。

    布局：每个样本一行 3×4 子图
        第一行：输入图像 | GT 漫反射 | 预测漫反射 | |Pred - GT| 差异图
        第二行：GT 镜面反射 | 预测镜面反射 | GT 输入梯度 | 预测梯度
        第三行：Pred SpecLikelihood (新辅助头) | |SpecLik - GT_Spec| 差异图 | 与 Pred Specular 叠加比较 | (空)

    参数:
        model          : 当前模型
        vis_loader     : 本次训练固定的可视化样本 DataLoader
        device         : 计算设备
        output_folder  : 输出目录
        current_epoch  : 当前 epoch 编号
        num_samples    : 可视化样本数（默认 2）
    """
    model.eval()
    vis_dir = os.path.join(output_folder, 'val_visualization')
    os.makedirs(vis_dir, exist_ok=True)

    batch = next(iter(vis_loader))
    inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]

    estimations = model(inputs)
    pred_diffuse = estimations['r']
    pred_specular = estimations['b']
    pred_gradient = estimations['g']
    # [NEW] spec_likelihood 头输出 (可能为 None, 用 .get 兼容旧检查点)
    pred_spec_lik = estimations.get('s')

    actual_samples = min(num_samples, inputs.size(0))

    for s in range(actual_samples):
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        input_img = inputs[s, 0].cpu().numpy()
        gt_diff_img = gt_diffuse[s, 0].cpu().numpy()
        gt_spec_img = gt_specular[s, 0].cpu().numpy()
        gt_grad_img = gt_gradient[s, 0].cpu().numpy()
        pred_diff_img = pred_diffuse[s, 0].cpu().numpy()
        pred_spec_img = pred_specular[s, 0].cpu().numpy()
        pred_grad_img = pred_gradient[s, 0].cpu().numpy()
        pred_slik_img = (pred_spec_lik[s, 0].cpu().numpy()
                         if pred_spec_lik is not None else None)

        # 第一行
        axes[0, 0].imshow(input_img, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Input (Highly Reflective)', fontsize=11)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gt_diff_img, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('GT Diffuse', fontsize=11)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(pred_diff_img, cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title('Pred Diffuse', fontsize=11)
        axes[0, 2].axis('off')
        diff = np.abs(pred_diff_img - gt_diff_img)
        im = axes[0, 3].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[0, 3].set_title('|Pred - GT| Difference', fontsize=11)
        axes[0, 3].axis('off')
        plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

        # 第二行
        axes[1, 0].imshow(gt_spec_img, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('GT Specular', fontsize=11)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(pred_spec_img, cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('Pred Specular', fontsize=11)
        axes[1, 1].axis('off')

        axes[1, 2].imshow(gt_grad_img, cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title('GT Input Gradient', fontsize=11)
        axes[1, 2].axis('off')

        axes[1, 3].imshow(pred_grad_img, cmap='gray', vmin=0, vmax=1)
        axes[1, 3].set_title('Pred Gradient', fontsize=11)
        axes[1, 3].axis('off')

        # 第三行（新增）：spec_likelihood 相关
        if pred_slik_img is not None:
            # (3,0) Pred SpecLikelihood 热力图 - 显示梯度分支辅助头的预测
            im_slik = axes[2, 0].imshow(pred_slik_img, cmap='hot', vmin=0, vmax=1)
            axes[2, 0].set_title('Pred SpecLikelihood (Aux)', fontsize=11)
            axes[2, 0].axis('off')
            plt.colorbar(im_slik, ax=axes[2, 0], fraction=0.046)

            # (3,1) SpecLikelihood 与 GT Specular 的绝对误差
            slik_err = np.abs(pred_slik_img - gt_spec_img)
            im_err = axes[2, 1].imshow(slik_err, cmap='hot', vmin=0, vmax=0.3)
            axes[2, 1].set_title('|SpecLik - GT_Spec|', fontsize=11)
            axes[2, 1].axis('off')
            plt.colorbar(im_err, ax=axes[2, 1], fraction=0.046)

            # (3,2) SpecLikelihood 与 Pred Specular 的对比（两个估计的差异）
            slik_vs_pred = np.abs(pred_slik_img - pred_spec_img)
            im_cmp = axes[2, 2].imshow(slik_vs_pred, cmap='hot', vmin=0, vmax=0.3)
            axes[2, 2].set_title('|SpecLik - Pred_Spec|', fontsize=11)
            axes[2, 2].axis('off')
            plt.colorbar(im_cmp, ax=axes[2, 2], fraction=0.046)

            # (3,3) SpecLikelihood 阈值化 (0.5) 后的二值化图, 直观看高光区
            slik_binary = (pred_slik_img > 0.5).astype(np.float32)
            axes[2, 3].imshow(slik_binary, cmap='gray', vmin=0, vmax=1)
            axes[2, 3].set_title('SpecLik > 0.5 (Binary)', fontsize=11)
            axes[2, 3].axis('off')
        else:
            # 模型未输出 s 字段 (旧版兼容): 整行隐藏
            for j in range(4):
                axes[2, j].axis('off')
                axes[2, j].set_title('SpecLik (N/A)', fontsize=9, color='gray')

        # 计算当前样本指标
        sample_mae = np.mean(diff)
        sample_mse = np.mean(diff ** 2)
        sample_max_err = np.max(diff)

        title_parts = [
            f'Epoch {current_epoch} - Sample {s+1}',
            f'MAE={sample_mae:.4f}',
            f'MSE={sample_mse:.6f}',
            f'MaxErr={sample_max_err:.4f}',
        ]
        if pred_slik_img is not None:
            slik_mae = float(np.mean(np.abs(pred_slik_img - gt_spec_img)))
            title_parts.append(f'SpecLik MAE={slik_mae:.4f}')
            # ★[FIX-C 问题2] 打印 SpecLik 分布，便于定标二值化阈值
            tqdm.write(
                f'[SpecLik dist] epoch={current_epoch} sample{s+1} '
                f'min={pred_slik_img.min():.3f} max={pred_slik_img.max():.3f} '
                f'mean={pred_slik_img.mean():.3f} '
                f'p95={np.percentile(pred_slik_img, 95):.3f} '
                f'>0.5_ratio={np.mean(pred_slik_img > 0.5):.4f}')
        fig.suptitle('  |  '.join(title_parts),
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(vis_dir, f'epoch_{current_epoch:03d}_sample{s+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    model.train()
    return vis_dir


def build_fixed_visualization_loader(val_dataset, num_samples, num_workers,
                                     pin_memory=False, seed=None):
    """
    为本次训练固定抽取一批验证样本。

    seed=None 时使用当前时间生成种子，因此不同训练文件夹通常会选到不同图像；
    训练开始后样本索引固定，保证同一次训练中各 epoch 的可视化可直接对比。
    """
    dataset_size = len(val_dataset)
    if dataset_size <= 0:
        raise ValueError("验证集为空，无法抽取可视化样本")

    sample_count = min(max(int(num_samples), 1), dataset_size)
    if seed is None:
        seed = time.time_ns() % (2**63 - 1)
    seed = int(seed)

    generator = torch.Generator().manual_seed(seed)
    vis_indices = torch.randperm(dataset_size, generator=generator)[:sample_count].tolist()
    vis_dataset = torch.utils.data.Subset(val_dataset, vis_indices)
    vis_loader = DataLoader(
        vis_dataset,
        batch_size=sample_count,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)
    return vis_loader, vis_indices, seed


# ════════════════════════════════════════════════════════════════════════
# 训练曲线绘制
# ════════════════════════════════════════════════════════════════════════

def plot_training_history(history, output_folder, total_epochs):
    """
    绘制完整训练历史曲线（Loss / SSIM / MAE / MSE），保存为 training_history.png。

    布局：4 行 × 2 列（参考 train.py 风格）
        左列：Train vs Val 折线图
        右列：Gap 填充对比图
    """
    epochs_range = range(1, len(history['train_losses']) + 1)

    # 固定 4 组指标
    pairs = [
        ('train_losses', 'val_losses',  'Loss',  'purple'),
        ('train_ssims',  'val_ssims',   'SSIM',  'green'),
        ('train_maes',   'val_maes',    'MAE',   'orange'),
        ('train_mses',   'val_mses',    'MSE',   'teal'),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))

    # 左列标签（Diffuse 分支相关的加 "(Diffuse)" 后缀）
    left_titles = [
        'Train vs Val Loss',
        'Train vs Val SSIM (Diffuse)',
        'Train vs Val MAE (Diffuse)',
        'Train vs Val MSE (Diffuse)',
    ]
    right_titles = [
        'Loss Gap (filled)',
        'SSIM Gap (filled)',
        'MAE Gap (filled)',
        'MSE Gap (filled)',
    ]

    for row, (tk, vk, label, fill_color) in enumerate(pairs):
        t_data = history.get(tk, [])
        v_data = history.get(vk, [])
        if not t_data or not v_data:
            continue

        # 左：折线图
        axes[row, 0].plot(epochs_range, t_data, 'b-o',
                          label=f'Train {label}' + (' (Diffuse)' if label in ('MAE', 'MSE') else ''),
                          linewidth=2, markersize=3)
        axes[row, 0].plot(epochs_range, v_data, 'r-s',
                          label=f'Val {label}' + (' (Diffuse)' if label in ('MAE', 'MSE') else ''),
                          linewidth=2, markersize=3)
        axes[row, 0].set_xlabel('Epoch', fontsize=11)
        axes[row, 0].set_ylabel(label, fontsize=11)
        axes[row, 0].set_title(left_titles[row], fontsize=13, fontweight='bold')
        axes[row, 0].legend(fontsize=10)
        axes[row, 0].grid(True, alpha=0.3)
        if label == 'SSIM':
            axes[row, 0].set_ylim([0, 1])

        # 右：填充对比图
        axes[row, 1].plot(epochs_range, t_data, 'b-',
                          label=f'Train {label}', linewidth=2, alpha=0.8)
        axes[row, 1].plot(epochs_range, v_data, 'r-',
                          label=f'Val {label}', linewidth=2, alpha=0.8)
        axes[row, 1].fill_between(epochs_range, t_data, v_data, alpha=0.15, color=fill_color)
        axes[row, 1].set_xlabel('Epoch', fontsize=11)
        axes[row, 1].set_ylabel(label, fontsize=11)
        axes[row, 1].set_title(right_titles[row], fontsize=13, fontweight='bold')
        axes[row, 1].legend(fontsize=10)
        axes[row, 1].grid(True, alpha=0.3)
        if label == 'SSIM':
            axes[row, 1].set_ylim([0, 1])

    fig.suptitle(f'Y-FFC V2 Net  Training History  ({total_epochs} epochs)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_individual_curves(history, output_folder, total_epochs):
    """
    将 Loss / SSIM / MAE / MSE 四条训练曲线分别保存为独立图片，
    放在 output_folder/training_curves/ 文件夹中。

    每张图包含 Train vs Val 两条曲线。
    """
    curves_dir = os.path.join(output_folder, 'training_curves')
    os.makedirs(curves_dir, exist_ok=True)

    epochs_range = range(1, len(history['train_losses']) + 1)

    items = [
        ('train_losses', 'val_losses', 'Loss',  '损失值 (Loss)', 'loss'),
        ('train_ssims',  'val_ssims',  'SSIM',  'SSIM',          'ssim'),
        ('train_maes',   'val_maes',   'MAE',   'MAE',           'mae'),
        ('train_mses',   'val_mses',   'MSE',   'MSE',           'mse'),
    ]

    saved = []
    for tk, vk, label, ylabel, fname in items:
        t_data = history.get(tk, [])
        v_data = history.get(vk, [])
        if not t_data or not v_data:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs_range, t_data, 'b-o', label=f'Train {label}', linewidth=2, markersize=3)
        ax.plot(epochs_range, v_data, 'r-s', label=f'Val {label}',   linewidth=2, markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Train vs Val {label}  ({total_epochs} epochs)',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        if label == 'SSIM':
            ax.set_ylim([0, 1])

        plt.tight_layout()
        p = os.path.join(curves_dir, f'{fname}_curve.png')
        plt.savefig(p, dpi=200, bbox_inches='tight')
        plt.close()
        saved.append(p)

    return curves_dir, saved


def plot_sub_loss_history(history, output_folder, total_epochs):
    """
    绘制子损失项历史曲线，保存为 sub_loss_history.png。
    从 history['train_sub_losses'] 和 history['val_sub_losses'] 中读取。

    每个子损失项一张子图，左右排布 Train vs Val。
    """
    train_subs = history.get('train_sub_losses', {})
    val_subs = history.get('val_sub_losses', {})

    if not train_subs:
        return None

    # 收集所有出现过的子损失键名（排除 'total'，因为已在主图中绘制）
    all_keys = sorted(set(train_subs.keys()) | set(val_subs.keys()))
    plot_keys = [k for k in all_keys if k != 'total']

    if not plot_keys:
        return None

    n_cols = 3
    n_rows = math.ceil(len(plot_keys) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    epochs_range = range(1, len(history['train_losses']) + 1)

    for idx, key in enumerate(plot_keys):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        t_data = train_subs.get(key, [])
        v_data = val_subs.get(key, [])
        display_name = SUB_LOSS_DISPLAY_NAMES.get(key, key)

        if t_data:
            ax.plot(epochs_range, t_data, 'b-', label=f'Train', linewidth=1.5, alpha=0.8)
        if v_data:
            ax.plot(epochs_range, v_data, 'r-', label=f'Val', linewidth=1.5, alpha=0.8)

        ax.set_title(display_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(len(plot_keys), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    fig.suptitle(f'Sub-Loss History  ({total_epochs} epochs)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'sub_loss_history.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    return save_path


def plot_training_curves_realtime(history, output_folder, current_epoch):
    """
    轻量实时曲线（仅 Loss + SSIM），参考 train.py 中文标签风格。
    保存为 training_curves_realtime.png。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(history['train_losses']) + 1)

    # Loss
    ax1.plot(epochs, history['train_losses'], 'b-o', label='训练损失', linewidth=2, markersize=3)
    ax1.plot(epochs, history['val_losses'],   'r-s', label='验证损失', linewidth=2, markersize=3)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('损失值 (Loss)', fontsize=11)
    ax1.set_title(f'训练和验证损失 (Epoch {current_epoch})', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # SSIM
    ax2.plot(epochs, history['train_ssims'], 'b-o', label='训练精度 (SSIM)', linewidth=2, markersize=3)
    ax2.plot(epochs, history['val_ssims'],   'r-s', label='验证精度 (SSIM)', linewidth=2, markersize=3)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('精度 (Accuracy)', fontsize=11)
    ax2.set_title(f'训练和验证精度 (Epoch {current_epoch})', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    save_path = os.path.join(output_folder, 'training_curves_realtime.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# 结构化日志输出辅助函数
# ════════════════════════════════════════════════════════════════════════

def print_epoch_log(ep, total_epochs, epoch_time, avg_epoch_time, eta,
                    tl, ts, tm, train_mse, train_psnr,
                    vl, vs, vm, val_mse, val_psnr,
                    train_subs, val_subs, weights, current_lr, device,
                    composite, best_val_ssim, best_val_loss):
    """紧凑版 epoch 日志输出。"""
    sep = '=' * 78
    sep2 = '-' * 78

    print(f"\n{sep}")
    print(f"  Epoch [{ep+1:03d}/{total_epochs}]  耗时: {epoch_time:.1f}s | 均: {avg_epoch_time:.1f}s/ep | ETA: {eta}")
    print(f"  LR: {current_lr}")
    print(f"  {sep2}")
    print(f"  {'指标':<16s}  {'Train':>12s}  {'Val':>12s}  {'Gap':>12s}")
    print(f"  {sep2}")
    print(f"  {'Total Loss':<16s}  {tl:>12.6f}  {vl:>12.6f}  {abs(tl-vl):>12.6f}")
    print(f"  {'SSIM':<16s}  {ts:>12.4f}  {vs:>12.4f}  {abs(ts-vs):>12.4f}")
    print(f"  {'MAE':<16s}  {tm:>12.6f}  {vm:>12.6f}  {abs(tm-vm):>12.6f}")
    print(f"  {'MSE':<16s}  {train_mse:>12.6f}  {val_mse:>12.6f}  {abs(train_mse-val_mse):>12.6f}")
    print(f"  {'PSNR (dB)':<16s}  {train_psnr:>12.2f}  {val_psnr:>12.2f}  {abs(train_psnr-val_psnr):>12.2f}")
    if 'ssim_mixed_loss' in train_subs or 'ssim_mixed_loss' in val_subs:
        train_mixed = 1.0 - train_subs.get('ssim_mixed_loss', 1.0)
        val_mixed = 1.0 - val_subs.get('ssim_mixed_loss', 1.0)
        print(f"  {'SSIM mixed':<16s}  {train_mixed:>12.4f}  {val_mixed:>12.4f}  {abs(train_mixed-val_mixed):>12.4f}")

    # ── 子损失项分解（完整显示） ──
    all_sub_keys = sorted(set(train_subs.keys()) | set(val_subs.keys()))
    display_keys = [k for k in all_sub_keys if k != 'total']
    if display_keys:
        print(f"  {sep2}")
        print(f"  {'子损失项':<20s}  {'Train':>10s}  {'Val':>10s}  {'Weight':>10s}")
        print(f"  {sep2}")
        for k in display_keys:
            t_val = train_subs.get(k, 0.0)
            v_val = val_subs.get(k, 0.0)
            w_val = weights.get(f'lambda_{k}', None)
            w_str = f'{w_val:.4f}' if w_val is not None else '--'
            name = SUB_LOSS_DISPLAY_NAMES.get(k, k)
            print(f"  {name:<20s}  {t_val:>10.6f}  {v_val:>10.6f}  {w_str:>10s}")

    # ── 最优 + 显存（合并为一行） ──
    mem_str = ''
    if device.type == 'cuda':
        cur_mem = torch.cuda.memory_allocated(device) / 1024**3
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
        mem_str = f'  显存: {cur_mem:.1f}/{peak_mem:.1f}GB'
    print(f"  {sep2}")
    print(f"  最优: SSIM={best_val_ssim:.4f} Loss={best_val_loss:.6f} 综合={composite:.4f}{mem_str}")
    print(sep)


# ════════════════════════════════════════════════════════════════════════
# 训练总结文件
# ════════════════════════════════════════════════════════════════════════

def save_training_summary(output_folder, cfg, history, best_info, total_time_str):
    """
    生成 training_summary.txt，包含：
      ① 训练配置
      ② 三种最优模型指标
      ③ 最终 epoch 指标（含子损失分解）
      ④ 输出文件列表
    """
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, 'training_summary.txt')

    actual_epochs = len(history['train_losses'])
    final_tr_loss = history['train_losses'][-1] if history['train_losses'] else float('nan')
    final_vl_loss = history['val_losses'][-1]   if history['val_losses']   else float('nan')
    final_tr_ssim = history['train_ssims'][-1]  if history['train_ssims']  else float('nan')
    final_vl_ssim = history['val_ssims'][-1]    if history['val_ssims']    else float('nan')
    final_tr_mae  = history['train_maes'][-1]   if history['train_maes']   else float('nan')
    final_vl_mae  = history['val_maes'][-1]     if history['val_maes']     else float('nan')
    final_vl_mse  = history['val_mses'][-1]     if history.get('val_mses') else None
    final_vl_psnr = history['val_psnrs'][-1]    if history.get('val_psnrs') else None
    final_tr_mse  = history['train_mses'][-1]   if history.get('train_mses') else None
    final_tr_psnr = history['train_psnrs'][-1]  if history.get('train_psnrs') else None

    def _fmt(val, spec, suffix=''):
        """安全格式化：None / NaN 显示为 N/A，否则按 spec 格式并附加 suffix。"""
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return 'N/A'
        return format(val, spec) + suffix

    sep = '=' * 64 + '\n'
    sep2 = '-' * 64 + '\n'

    with open(save_path, 'w', encoding='utf-8') as f:
        # ① 训练配置
        f.write(sep)
        f.write('Training Summary  (Y-FFC V2)\n')
        f.write(sep)
        f.write(f'  Generated at      : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'  Total Train Time  : {total_time_str}\n\n')
        f.write('  [Hyperparameters]\n')
        f.write(f'  Total Epochs      : {actual_epochs}\n')
        f.write(f'  Batch Size        : {cfg["BATCH_SIZE"]}\n')
        f.write(f'  Learning Rate     : {cfg["BASE_LR"]}\n')
        f.write(f'  Grad Accum Steps  : {cfg["ACCUMULATION_STEPS"]}'
                f'  (effective batch = {cfg["BATCH_SIZE"] * cfg["ACCUMULATION_STEPS"]})\n')
        # 格式化图像尺寸
        img_sz = cfg['IMAGE_SIZE']
        if img_sz is None:
            sz_str = 'Original (no resize)'
        elif isinstance(img_sz, (int, float)):
            sz_str = f'{int(img_sz)} x {int(img_sz)}'
        elif isinstance(img_sz, (tuple, list)) and len(img_sz) == 2:
            sz_str = f'{img_sz[0]} x {img_sz[1]} (H x W)'
        else:
            sz_str = str(img_sz)
        f.write(f'  Image Size        : {sz_str}\n')
        f.write(f'  Use GN            : {cfg["USE_GN"]}\n')
        f.write(f'  Use EMA           : {cfg["USE_EMA"]}\n')
        f.write(f'  Use OHEM          : {cfg["USE_OHEM"]}\n')
        f.write(f'  Mixed Precision   : Enabled (AMP)\n')
        f.write(f'  Val Vis Interval  : {cfg.get("VIS_EVERY_N_EPOCHS", 10)}\n')
        f.write(f'  Val Vis Samples   : {cfg.get("VIS_NUM_SAMPLES", 2)}\n')
        f.write(f'  Val Vis Seed Used : {cfg.get("VIS_SAMPLE_SEED_USED")}\n')
        f.write(f'  Val Vis Indices   : {cfg.get("VIS_SAMPLE_INDICES", [])}\n\n')

        # ② 最优模型
        f.write(sep)
        f.write('Best Models\n')
        f.write(sep)
        for tag, info in best_info.items():
            f.write(f'  {tag}\n')
            f.write(sep2)
            f.write(f'  Epoch             : {info["epoch"]}\n')
            f.write(f'  Train Loss        : {_fmt(info.get("train_loss"), ".6f")}\n')
            f.write(f'  Val   Loss        : {_fmt(info.get("loss"), ".6f")}\n')
            f.write(f'  Train SSIM        : {_fmt(info.get("train_ssim"), ".4f")}\n')
            f.write(f'  Val   SSIM        : {_fmt(info.get("ssim"), ".4f")}\n')
            f.write(f'  Train MAE         : {_fmt(info.get("train_mae"), ".6f")}\n')
            f.write(f'  Val   MAE         : {_fmt(info.get("mae"), ".6f")}\n')
            f.write(f'  Train MSE         : {_fmt(info.get("train_mse"), ".6f")}\n')
            f.write(f'  Val   MSE         : {_fmt(info.get("mse"), ".6f")}\n')
            f.write(f'  Train PSNR        : {_fmt(info.get("train_psnr"), ".2f", " dB")}\n')
            f.write(f'  Val   PSNR        : {_fmt(info.get("psnr"), ".2f", " dB")}\n')
            f.write(f'  Saved Path        : {info["path"]}\n\n')

        # ③ 最终 epoch（含子损失分解）
        f.write(sep)
        f.write('Final Epoch Result\n')
        f.write(sep)
        f.write(f'  Epoch             : {actual_epochs}\n')
        f.write(f'  Train Loss        : {final_tr_loss:.6f}\n')
        f.write(f'  Val   Loss        : {final_vl_loss:.6f}\n')
        f.write(f'  Train SSIM        : {final_tr_ssim:.4f}\n')
        f.write(f'  Val   SSIM        : {final_vl_ssim:.4f}\n')
        f.write(f'  Train MAE         : {final_tr_mae:.6f}\n')
        f.write(f'  Val   MAE         : {final_vl_mae:.6f}\n')
        f.write(f'  Train MSE         : {_fmt(final_tr_mse, ".6f")}\n')
        f.write(f'  Val   MSE         : {_fmt(final_vl_mse, ".6f")}\n')
        f.write(f'  Train PSNR        : {_fmt(final_tr_psnr, ".2f", " dB")}\n')
        f.write(f'  Val   PSNR        : {_fmt(final_vl_psnr, ".2f", " dB")}\n')

        # 子损失分解（最终 epoch）
        val_sub = history.get('val_sub_losses', {})
        train_sub = history.get('train_sub_losses', {})
        if val_sub:
            f.write(f'\n  [Final Epoch Sub-Loss Breakdown]\n')
            f.write(f'  {"Loss Item":<20s}  {"Train":>10s}  {"Val":>10s}\n')
            f.write(f'  {"-"*50}\n')
            all_keys = sorted(set(train_sub.keys()) | set(val_sub.keys()))
            for k in all_keys:
                if k == 'total':
                    continue
                t_list = train_sub.get(k, [])
                v_list = val_sub.get(k, [])
                t_val = t_list[-1] if t_list else 0.0
                v_val = v_list[-1] if v_list else 0.0
                display_name = SUB_LOSS_DISPLAY_NAMES.get(k, k)
                f.write(f'  {display_name:<20s}  {t_val:>10.6f}  {v_val:>10.6f}\n')
        f.write('\n')

        # ④ 文件列表
        f.write(sep)
        f.write('Output Files\n')
        f.write(sep)
        f.write(f'  Training History  : {os.path.join(output_folder, "training_history.png")}\n')
        f.write(f'  Individual Curves : {os.path.join(output_folder, "training_curves/")}\n')
        f.write(f'  Sub-Loss History  : {os.path.join(output_folder, "sub_loss_history.png")}\n')
        f.write(f'  Realtime Curves   : {os.path.join(output_folder, "training_curves_realtime.png")}\n')
        f.write(f'  Val Visualization : {os.path.join(output_folder, "val_visualization/")}\n')
        f.write(f'  This Summary      : {save_path}\n')
        f.write(sep)

    return save_path


# ════════════════════════════════════════════════════════════════════════
# 训练主函数
# ════════════════════════════════════════════════════════════════════════

def train_model(cfg):
    """
    执行完整训练流程：
      1. 构建模型、数据集、损失、优化器
      2. 多 epoch 训练 + 验证（含完整子损失追踪）
      3. 多指标最优模型保存
      4. 验证样本可视化
      5. 训练曲线 + 子损失曲线 + 总结文件

    参数:
        cfg: dict，所有训练配置集中于此

    返回:
        history: dict，包含全部训练历史与最优模型信息
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"已清理 CUDA 缓存，当前显存: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

    # 规范化 IMAGE_SIZE 为 (H, W) 元组
    img_sz = cfg['IMAGE_SIZE']
    if isinstance(img_sz, (int, float)):
        img_hw = (int(img_sz), int(img_sz))
    elif isinstance(img_sz, (tuple, list)) and len(img_sz) == 2:
        img_hw = (int(img_sz[0]), int(img_sz[1]))
    else:
        raise ValueError(f"IMAGE_SIZE 格式不正确: {img_sz}，应为 int 或 (H, W)")

    print(f"\n{'=' * 78}")
    print(f"Y-FFC V2 训练 | 设备={device} | 分辨率={img_hw[0]}×{img_hw[1]}")
    print(f"{'=' * 78}")

    # ── 输出目录 ──
    output_folder = cfg['OUTPUT_FOLDER']
    models_dir = os.path.join(output_folder, 'models')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'val_visualization'), exist_ok=True)
    if cfg['SAVE_EVERY_N_EPOCHS'] > 0:
        os.makedirs(os.path.join(output_folder, 'checkpoints'), exist_ok=True)

    # ══════════ 1. 模型 ══════════
    model = XFFC()
    if cfg['USE_GN']:
        print("[GN] BN -> GN 替换")
        replace_bn_with_gn(model, 8)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")
    print(f"  可训练: {trainable_params:,}  冻结: {total_params - trainable_params:,}")

    # ══════════ 2. 数据集 ══════════
    # 默认开启合成高反光增强：只作用于训练集，验证集保持原图，避免评估被增强扰动。
    min_side = min(img_hw)  # 取短边计算高光半径
    if cfg.get('USE_HIGHLIGHT_AUG', True):
        hl_aug = HighlightAugmentation(
            p=float(cfg.get('HIGHLIGHT_AUG_P', 1.0)), num_range=(1, 6),
            radius_range=(3, max(5, min_side // 20)),
            intensity_range=(0.2, 0.7))
        aug = EnhancedPairedTransform(highlight_aug=hl_aug)
        print(f"[Aug] 训练集已启用 EnhancedPairedTransform + HighlightAugmentation(p={hl_aug.p})")
    else:
        aug = EnhancedPairedTransform(highlight_aug=HighlightAugmentation(p=0.0))
        print("[Aug] 训练集已启用几何/亮度增强，高反光合成已关闭")

    full_ds = ImprovedFringeDataset(
        cfg['INPUT_DIR'], cfg['TARGET_DIR'],
        resize=img_hw, paired_transform=aug)

    # 数据集划分
    if cfg['VAL_INPUT_DIR'] and cfg['VAL_TARGET_DIR']:
        train_ds = full_ds
        val_ds = ImprovedFringeDataset(
            cfg['VAL_INPUT_DIR'], cfg['VAL_TARGET_DIR'],
            resize=img_hw)
    else:
        # [P8] 验证集不应有数据增强
        full_ds_no_aug = ImprovedFringeDataset(
            cfg['INPUT_DIR'], cfg['TARGET_DIR'],
            resize=img_hw, paired_transform=None)
        train_size = int(cfg['TRAIN_SPLIT_RATIO'] * len(full_ds))
        val_size = len(full_ds) - train_size
        g = torch.Generator().manual_seed(42)
        rand_idx = torch.randperm(len(full_ds), generator=g).tolist()
        train_ds = torch.utils.data.Subset(full_ds, rand_idx[:train_size])
        val_ds = torch.utils.data.Subset(full_ds_no_aug, rand_idx[train_size:])

    train_loader = DataLoader(
        train_ds, batch_size=cfg['BATCH_SIZE'], shuffle=True,
        num_workers=cfg['NUM_WORKERS'],
        pin_memory=(device.type == 'cuda'), drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg['BATCH_SIZE'], shuffle=False,
        num_workers=cfg['NUM_WORKERS'],
        pin_memory=(device.type == 'cuda'))

    print(f"数据集: 训练={len(train_ds)}, 验证={len(val_ds)}")
    print(f"批次大小: {cfg['BATCH_SIZE']} × 累积{cfg['ACCUMULATION_STEPS']}"
          f" = 等效 {cfg['BATCH_SIZE'] * cfg['ACCUMULATION_STEPS']}")

    # 可视化使用单独固定样本集：每次训练随机抽一次，同一次训练全程不变
    vis_interval = int(cfg.get('VIS_EVERY_N_EPOCHS', 10))
    vis_num_samples = int(cfg.get('VIS_NUM_SAMPLES', 2))
    vis_loader = None
    if vis_interval > 0 and vis_num_samples > 0:
        vis_loader, vis_indices, vis_seed = build_fixed_visualization_loader(
            val_ds,
            num_samples=vis_num_samples,
            num_workers=cfg['NUM_WORKERS'],
            pin_memory=(device.type == 'cuda'),
            seed=cfg.get('VIS_SAMPLE_SEED'))
        cfg['VIS_SAMPLE_SEED_USED'] = vis_seed
        cfg['VIS_SAMPLE_INDICES'] = vis_indices
        print(f"可视化样本: 本次训练固定抽取 {len(vis_indices)} 张 "
              f"(seed={vis_seed}, val_indices={vis_indices})")
    else:
        cfg['VIS_SAMPLE_SEED_USED'] = None
        cfg['VIS_SAMPLE_INDICES'] = []
        print("可视化样本: 已禁用")

    # ══════════ 3. 损失函数 ══════════
    criterion = EnhancedCompositeLoss(
        use_perceptual=True, use_ohem=cfg['USE_OHEM']).to(device)
    ws = ImprovedWeightScheduler(cfg['EPOCHS'])

    # 打印损失函数结构
    print(f"\n损失函数: EnhancedCompositeLoss")
    print(f"  use_perceptual={True}, use_ohem={cfg['USE_OHEM']}, use_amp={cfg['USE_AMP']}")
    print(f"  子损失项: {', '.join(SUB_LOSS_KEYS)}")

    # ══════════ 4. 优化器 + 调度器 ══════════
    print("\n分层学习率配置:")
    optimizer = build_optimizer_v2(model, cfg['BASE_LR'], cfg['WEIGHT_DECAY'])
    scheduler = build_scheduler(
        optimizer,
        warmup_iters=min(500, math.ceil(len(train_loader) / cfg['ACCUMULATION_STEPS']) * 3),
        total_epochs=cfg['EPOCHS'],
        iters_per_epoch=math.ceil(len(train_loader) / cfg['ACCUMULATION_STEPS']))

    # ══════════ 5. AMP + EMA ══════════
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and cfg['USE_AMP']))
    ema = EMA(model, cfg['EMA_DECAY']) if cfg['USE_EMA'] else None

    # ══════════ 6. 历史记录 + 最优模型跟踪 ══════════
    history = {
        'train_losses': [], 'val_losses': [],
        'train_ssims':  [], 'val_ssims':  [],
        'train_maes':   [], 'val_maes':   [],
        'val_mses':     [], 'val_psnrs':  [],
        # 训练阶段不额外计算 MSE/PSNR，用 placeholder 对齐
        'train_mses':   [], 'train_psnrs': [],
        # 子损失历史：{key: [epoch_1_avg, epoch_2_avg, ...]}
        'train_sub_losses': defaultdict(list),
        'val_sub_losses':   defaultdict(list),
    }
    epoch_times = []

    # 多指标最优模型保存路径
    best_composite_path = os.path.join(models_dir, 'best_model.pth')
    best_ssim_path      = os.path.join(models_dir, 'best_model_best_ssim.pth')
    best_loss_path      = os.path.join(models_dir, 'best_model_best_loss.pth')
    last_model_path     = os.path.join(models_dir, 'last_model.pth')

    COMPOSITE_ALPHA = 0.7
    best_composite  = 0.0
    best_val_ssim   = 0.0
    best_val_loss   = float('inf')

    # 每个最优模型同时记录训练 (train_*) 与验证 (loss/ssim/mae/mse/psnr) 指标，
    # 便于在总结中同步展示该 epoch 的 Train / Val 全量指标。
    _best_info_init = lambda path: {
        'epoch': 0, 'loss': float('inf'), 'ssim': 0.0, 'mae': float('inf'),
        'mse': None, 'psnr': None,
        'train_loss': None, 'train_ssim': None, 'train_mae': None,
        'train_mse': None, 'train_psnr': None,
        'path': path}
    best_info = {
        'Best Overall (SSIM×0.7 + NormLoss×0.3)': _best_info_init(best_composite_path),
        'Best SSIM': _best_info_init(best_ssim_path),
        'Best Loss': _best_info_init(best_loss_path),
    }

    # 早停
    early_stopping_counter = 0

    # ══════════ 打印完整配置 ══════════
    print(f"\n{'=' * 78}")
    print("训练配置总览:")
    print(f"  训练轮数    : {cfg['EPOCHS']}")
    print(f"  学习率      : {cfg['BASE_LR']}")
    print(f"  图像分辨率  : {img_hw[0]} × {img_hw[1]} (H × W)")
    print(f"  Use GN      : {cfg['USE_GN']}")
    print(f"  Use EMA     : {cfg['USE_EMA']} (decay={cfg['EMA_DECAY']})")
    print(f"  Use OHEM    : {cfg['USE_OHEM']}")
    print(f"  早停策略    : {'启用 (patience=' + str(cfg['EARLY_STOPPING_PATIENCE']) + ')' if cfg['USE_EARLY_STOPPING'] else '禁用'}")
    print(f"  输出目录    : {os.path.abspath(output_folder)}")
    print(f"\n模型保存路径:")
    print(f"  综合最优 -> {best_composite_path}")
    print(f"  SSIM最优 -> {best_ssim_path}")
    print(f"  Loss最优 -> {best_loss_path}")
    print(f"  最新模型 -> {last_model_path}")
    print(f"  综合评分权重: SSIM×{COMPOSITE_ALPHA:.1f} + Loss归一化×{1 - COMPOSITE_ALPHA:.1f}")

    # ══════════ 7. 训练循环 ══════════
    start_time_total = time.time()
    print(f"\n训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 78}\n")

    for ep in range(cfg['EPOCHS']):
        epoch_start = time.time()

        # 动态权重调度
        w = ws.step(criterion, ep)

        # --- 训练 ---
        tl, ts, tm, train_mse, train_psnr, train_subs = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, ema, device,
            accum=cfg['ACCUMULATION_STEPS'],
            max_norm=cfg['GRAD_CLIP_VALUE'],
            epoch=ep, total_epochs=cfg['EPOCHS'],
            use_amp=cfg['USE_AMP'])

        # --- 验证 ---
        if ema:
            ema.apply_shadow()
        # [P15-FIX] TTA 会导致验证指标突变，训练过程中禁用以保持一致性
        # TTA 仅在最终评估/推理时使用，不影响训练中的模型选择
        _use_tta = False
        vl, vs, vm, val_mse, val_psnr, val_subs = validate(
            model, val_loader, criterion, device,
            epoch=ep, total_epochs=cfg['EPOCHS'], use_tta=_use_tta)
        if ema:
            ema.restore()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # 记录主指标历史
        history['train_losses'].append(tl)
        history['val_losses'].append(vl)
        history['train_ssims'].append(ts)
        history['val_ssims'].append(vs)
        history['train_maes'].append(tm)
        history['val_maes'].append(vm)
        history['val_mses'].append(val_mse)
        history['val_psnrs'].append(val_psnr)
        history['train_mses'].append(train_mse)
        history['train_psnrs'].append(train_psnr)

        # 记录子损失历史
        for k, v in train_subs.items():
            history['train_sub_losses'][k].append(v)
        for k, v in val_subs.items():
            history['val_sub_losses'][k].append(v)

        # ETA
        avg_epoch_time = np.mean(epoch_times)
        remaining = avg_epoch_time * (cfg['EPOCHS'] - ep - 1)
        eta = timedelta(seconds=int(remaining))
        current_lr_value = optimizer.param_groups[0]['lr']
        current_lr = format_lr_groups(optimizer)

        # [FIX] LR 回弹检测：若 LR 比上一 epoch 增加超过 50%，发出警告
        if ep > 0 and hasattr(train_model, '_prev_lr'):
            if current_lr_value > train_model._prev_lr * 1.5 and current_lr_value > 1e-5:
                print(f"  ⚠️ [LR 异常] LR 从 {train_model._prev_lr:.2e} 跳升至 {current_lr_value:.2e}！")
        train_model._prev_lr = current_lr_value

        # ── 多指标最优模型保存 ──
        composite = COMPOSITE_ALPHA * vs + (1.0 - COMPOSITE_ALPHA) * (1.0 / (1.0 + vl))

        # ── 结构化日志 ──
        print_epoch_log(
            ep, cfg['EPOCHS'], epoch_time, avg_epoch_time, eta,
            tl, ts, tm, train_mse, train_psnr,
            vl, vs, vm, val_mse, val_psnr,
            train_subs, val_subs, w, current_lr, device,
            composite, best_val_ssim, best_val_loss)
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'fuse_h4') and hasattr(model.encoder, 'fuse_h8'):
            gate4 = model.encoder.fuse_h4.last_gate_mean
            gate8 = model.encoder.fuse_h8.last_gate_mean
            gate4_str = f"{gate4.item():.6f}" if gate4 is not None else "n/a"
            gate8_str = f"{gate8.item():.6f}" if gate8 is not None else "n/a"
            print(f"  [Adaptive skip] beta_h4={model.encoder.fuse_h4.beta.item():.6f}, "
                  f"beta_h8={model.encoder.fuse_h8.beta.item():.6f}, "
                  f"gate_h4_mean={gate4_str}, gate_h8_mean={gate8_str}")

        # ── 模型保存逻辑 ──
        improved_flags = []

        def _build_ema_state_dict(raw_state_dict):
            if not ema:
                return raw_state_dict
            ema_state_dict = dict(raw_state_dict)
            for name, tensor in ema.shadow.items():
                if name in ema_state_dict:
                    ema_state_dict[name] = tensor
            return ema_state_dict

        def _save_best(path):
            raw_state_dict = model.state_dict()
            sd = {
                'epoch': ep + 1,
                'model_state_dict': _build_ema_state_dict(raw_state_dict),
                'raw_model_state_dict': raw_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ssim': vs,
                'val_loss': vl,
                'image_size': cfg['IMAGE_SIZE'],
                'uses_ema_weights': bool(ema),
            }
            if ema:
                sd['ema'] = ema.shadow
                sd['ema_shadow'] = ema.shadow
            if scaler is not None:
                sd['scaler_state_dict'] = scaler.state_dict()
            torch.save(sd, path)

        # 策略 A：综合最优
        if composite > best_composite:
            best_composite = composite
            early_stopping_counter = 0
            _save_best(best_composite_path)
            improved_flags.append(f'综合↑{composite:.4f}')
            best_info['Best Overall (SSIM×0.7 + NormLoss×0.3)'].update(
                {'epoch': ep + 1, 'loss': vl, 'ssim': vs, 'mae': vm,
                 'mse': val_mse, 'psnr': val_psnr,
                 'train_loss': tl, 'train_ssim': ts, 'train_mae': tm,
                 'train_mse': train_mse, 'train_psnr': train_psnr})
        else:
            early_stopping_counter += 1

        # 策略 B：SSIM 最优
        if vs > best_val_ssim:
            best_val_ssim = vs
            _save_best(best_ssim_path)
            improved_flags.append(f'SSIM↑{vs:.4f}')
            best_info['Best SSIM'].update(
                {'epoch': ep + 1, 'loss': vl, 'ssim': vs, 'mae': vm,
                 'mse': val_mse, 'psnr': val_psnr,
                 'train_loss': tl, 'train_ssim': ts, 'train_mae': tm,
                 'train_mse': train_mse, 'train_psnr': train_psnr})

        # 策略 C：Loss 最优
        if vl < best_val_loss:
            best_val_loss = vl
            _save_best(best_loss_path)
            improved_flags.append(f'Loss↓{vl:.6f}')
            best_info['Best Loss'].update(
                {'epoch': ep + 1, 'loss': vl, 'ssim': vs, 'mae': vm,
                 'mse': val_mse, 'psnr': val_psnr,
                 'train_loss': tl, 'train_ssim': ts, 'train_mae': tm,
                 'train_mse': train_mse, 'train_psnr': train_psnr})

        if improved_flags:
            print(f"  ✅ 模型更新: {' | '.join(improved_flags)}")
        else:
            if cfg['USE_EARLY_STOPPING']:
                print(f"  ⚠️ 未改善 ({early_stopping_counter}/{cfg['EARLY_STOPPING_PATIENCE']})")
            else:
                print(f"  -- 本轮无改善")

        # ── 实时曲线 ──
        if (ep + 1) % 5 == 0 or ep == 0:
            plot_training_curves_realtime(history, output_folder, ep + 1)

        # ── 验证样本可视化 ──
        if vis_loader is not None and ((ep + 1) % vis_interval == 0 or ep == 0):
            if ema:
                ema.apply_shadow()
            vis_path = visualize_sample_during_training(
                model, vis_loader, device, output_folder,
                current_epoch=ep + 1, num_samples=vis_num_samples)
            if ema:
                ema.restore()
            print(f"  样本可视化已保存: {vis_path}")

        # ── 定期检查点 ──
        if cfg['SAVE_EVERY_N_EPOCHS'] > 0 and (ep + 1) % cfg['SAVE_EVERY_N_EPOCHS'] == 0:
            ckpt_path = os.path.join(output_folder, 'checkpoints', f'checkpoint_epoch_{ep+1}.pth')
            ckpt = {
                'epoch': ep + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_composite': best_composite,
                'best_val_ssim': best_val_ssim,
                'best_val_loss': best_val_loss,
                'image_size': cfg['IMAGE_SIZE'],
            }
            if scheduler is not None:
                ckpt['scheduler_state_dict'] = scheduler.state_dict()
            if scaler is not None:
                ckpt['scaler_state_dict'] = scaler.state_dict()
            torch.save(ckpt, ckpt_path)
            print(f"  检查点已保存: {ckpt_path}")

        # ── 早停 ──
        if cfg['USE_EARLY_STOPPING'] and early_stopping_counter >= cfg['EARLY_STOPPING_PATIENCE']:
            print(f"\n⛔ 早停触发！综合评分在 {cfg['EARLY_STOPPING_PATIENCE']} 轮内未改善。")
            break

    # ══════════ 8. 保存 last_model ══════════
    torch.save({
        'epoch': ep + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'image_size': cfg['IMAGE_SIZE'],
    }, last_model_path)

    # ══════════ 9. 训练总结 ══════════
    total_time = time.time() - start_time_total
    total_time_str = str(timedelta(seconds=int(total_time)))
    actual_epochs = len(history['train_losses'])

    print(f"\n{'=' * 78}")
    print("训练完成！")
    print(f"{'=' * 78}")
    print(f"  总训练时间   : {total_time_str}")
    print(f"  实际完成轮数 : {actual_epochs}")
    print(f"  最佳综合评分 : {best_composite:.4f}"
          f"  (Epoch {best_info['Best Overall (SSIM×0.7 + NormLoss×0.3)']['epoch']})")
    print(f"  最佳 SSIM    : {best_val_ssim:.4f}"
          f"  (Epoch {best_info['Best SSIM']['epoch']})")
    print(f"  最佳 Loss    : {best_val_loss:.6f}"
          f"  (Epoch {best_info['Best Loss']['epoch']})")

    # 最终 epoch 子损失总览
    print(f"\n  最终 Epoch 子损失分解 (Val):")
    val_sub = history['val_sub_losses']
    for k in sorted(val_sub.keys()):
        if k == 'total':
            continue
        vals = val_sub[k]
        if vals:
            display_name = SUB_LOSS_DISPLAY_NAMES.get(k, k)
            print(f"    {display_name:<20s}: {vals[-1]:.6f}")

    # 训练历史曲线
    print("\n生成训练历史曲线...")
    history_path = plot_training_history(history, output_folder, actual_epochs)
    print(f"  -> {history_path}")

    # 单独指标曲线
    print("生成单独指标曲线...")
    curves_dir, curves_list = plot_individual_curves(history, output_folder, actual_epochs)
    for cp in curves_list:
        print(f"  -> {cp}")

    # 子损失历史曲线
    print("生成子损失历史曲线...")
    sub_loss_path = plot_sub_loss_history(history, output_folder, actual_epochs)
    if sub_loss_path:
        print(f"  -> {sub_loss_path}")

    # 训练总结文件
    print("生成训练总结文件...")
    summary_path = save_training_summary(output_folder, cfg, history, best_info, total_time_str)
    print(f"  -> {summary_path}")

    # 文件结构
    print(f"\n{'=' * 78}")
    print(f"📁 所有结果已保存到: {os.path.abspath(output_folder)}")
    print(f"{'=' * 78}")
    print(f"  {output_folder}/")
    print(f"  ├── models/")
    print(f"  │   ├── best_model.pth             (综合评分最优)")
    print(f"  │   ├── best_model_best_ssim.pth   (SSIM 最优)")
    print(f"  │   ├── best_model_best_loss.pth   (Loss 最优)")
    print(f"  │   └── last_model.pth             (最后一轮)")
    print(f"  ├── training_curves/               (各指标独立曲线)")
    print(f"  │   ├── loss_curve.png")
    print(f"  │   ├── ssim_curve.png")
    print(f"  │   ├── mae_curve.png")
    print(f"  │   └── mse_curve.png")
    print(f"  ├── val_visualization/              (验证样本可视化)")
    print(f"  ├── training_history.png            (Loss / SSIM / MAE / MSE 总览)")
    print(f"  ├── sub_loss_history.png            (子损失项历史曲线)")
    print(f"  ├── training_curves_realtime.png    (实时 Loss + SSIM)")
    print(f"  └── training_summary.txt            (配置 / 最优 Epoch / 总结)")
    if cfg['SAVE_EVERY_N_EPOCHS'] > 0:
        print(f"  └── checkpoints/                   (定期检查点)")
    print(f"{'=' * 78}")

    history['best_info'] = best_info
    history['total_time_str'] = total_time_str
    history['actual_epochs'] = actual_epochs
    return history


# ════════════════════════════════════════════════════════════════════════
# 主函数：参数集中配置区
# ════════════════════════════════════════════════════════════════════════

def main():
    """
    主函数：所有训练参数在此集中配置。
    修改实验设置时只需修改这一个区域。
    """
    print("=" * 60)
    print("Y-FFC V2 Net 训练程序启动（增强可观测性版）")
    print("=" * 60)

    # ╔══════════════════════════════════════════════════════════════╗
    # ║              ★ 训练参数集中配置区 ★                          ║
    # ║  修改实验配置时，只需修改此区域内的变量                         ║
    # ╚══════════════════════════════════════════════════════════════╝

    # ========== 基础训练参数 ==========
    EPOCHS                = 300       # 训练总轮数
    BATCH_SIZE            = 3         # 384 分辨率推荐 2~4
    # ★[V3.2 调整] BASE_LR 1e-4 → 2e-4
    # 原因：V3.2 新增 dual-head 与 r 系列损失，新结构从头训练，1e-4 收敛过慢。
    # 配合 V3.1 反向稳定性修复，2e-4 在 dual-head + 三路调制下更平衡。
    BASE_LR               = 2e-4
    WEIGHT_DECAY          = 1e-4     # AdamW 权重衰减
    ACCUMULATION_STEPS    = 3        # 梯度累积步数（等效 batch = BATCH_SIZE × 此值）
    # ★[V3.2 调整] GRAD_CLIP 0.5 → 1.0
    # 原因：V3.1 频域损失反向稳定性已修复（无除法相位项 + sqrt eps 提升），
    # 旧版 0.5 过紧会压制 dual-head 早期梯度信号。1.0 是更标准的设定。
    GRAD_CLIP_VALUE       = 1.0

    # ========== 图像与数据参数 ==========
    IMAGE_SIZE            = (128, 160)  # ⭐ 输入分辨率 (H, W)，支持任意矩形
    TRAIN_SPLIT_RATIO     = 0.8      # 自动划分时训练集比例（有独立验证集时忽略）
    NUM_WORKERS           = 5        # DataLoader 工作进程数

    # ========== V2 特有开关 ==========
    USE_GN                = True     # BN → GN（小 batch 必备）
    USE_EMA               = True     # 指数移动平均
    EMA_DECAY             = 0.9995   # [P16] EMA 衰减系数（调高增加平滑力度）
    USE_OHEM              = True     # 在线难样本挖掘
    USE_AMP               = False    # 稳定版默认关闭 AMP；确认稳定后可改 True
    USE_HIGHLIGHT_AUG     = True     # 默认开启训练集高反光合成增强
    HIGHLIGHT_AUG_P       = 0.5      # 降低合成高反光占比，避免训练分布过度偏向合成模式

    # ========== 数据目录 ==========
    INPUT_DIR             = 'data2/inputs'
    TARGET_DIR            = 'data2/targets'
    VAL_INPUT_DIR         = None     # 独立验证集输入目录（None 则自动划分）
    VAL_TARGET_DIR        = None     # 独立验证集目标目录

    # ========== 模型保存与检查点 ==========
    SAVE_EVERY_N_EPOCHS   = 10       # 每 N 轮保存检查点（0=不保存）

    # ========== 可视化间隔 ==========
    VIS_EVERY_N_EPOCHS    = 10       # 每 N 轮保存验证样本可视化（0=不保存）
    VIS_NUM_SAMPLES       = 2        # 每次保存几张固定验证样本
    VIS_SAMPLE_SEED       = None     # None=每次训练随机抽样；填整数可复现同一批图像

    # ========== 早停策略 ==========
    USE_EARLY_STOPPING    = False
    EARLY_STOPPING_PATIENCE = 30

    # ========== 输出目录（自动带时间戳） ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FOLDER = f'experiments/yffc_v2_{timestamp}'

    # ╔══════════════════════════════════════════════════════════════╗
    # ║              ★ 配置区结束 ★                                  ║
    # ╚══════════════════════════════════════════════════════════════╝

    # 将所有配置打包为字典，统一传递
    cfg = {
        'EPOCHS':                 EPOCHS,
        'BATCH_SIZE':             BATCH_SIZE,
        'BASE_LR':                BASE_LR,
        'WEIGHT_DECAY':           WEIGHT_DECAY,
        'ACCUMULATION_STEPS':     ACCUMULATION_STEPS,
        'GRAD_CLIP_VALUE':        GRAD_CLIP_VALUE,
        'IMAGE_SIZE':             IMAGE_SIZE,
        'TRAIN_SPLIT_RATIO':      TRAIN_SPLIT_RATIO,
        'NUM_WORKERS':            NUM_WORKERS,
        'USE_GN':                 USE_GN,
        'USE_EMA':                USE_EMA,
        'EMA_DECAY':              EMA_DECAY,
        'USE_OHEM':               USE_OHEM,
        'USE_AMP':                USE_AMP,
        'USE_HIGHLIGHT_AUG':      USE_HIGHLIGHT_AUG,
        'HIGHLIGHT_AUG_P':        HIGHLIGHT_AUG_P,
        'INPUT_DIR':              INPUT_DIR,
        'TARGET_DIR':             TARGET_DIR,
        'VAL_INPUT_DIR':          VAL_INPUT_DIR,
        'VAL_TARGET_DIR':         VAL_TARGET_DIR,
        'SAVE_EVERY_N_EPOCHS':    SAVE_EVERY_N_EPOCHS,
        'VIS_EVERY_N_EPOCHS':     VIS_EVERY_N_EPOCHS,
        'VIS_NUM_SAMPLES':        VIS_NUM_SAMPLES,
        'VIS_SAMPLE_SEED':        VIS_SAMPLE_SEED,
        'USE_EARLY_STOPPING':     USE_EARLY_STOPPING,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'OUTPUT_FOLDER':          OUTPUT_FOLDER,
    }

    # 打印配置总览
    print(f"\n实验输出目录: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"\n当前训练配置:")
    for k, v in cfg.items():
        print(f"  {k:30s}: {v}")
    print("=" * 60)

    # 执行训练
    history = train_model(cfg)

    print("\n✅ 所有流程完成！")


if __name__ == '__main__':
    # Matplotlib 字体设置（支持中文）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    main()
