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
  ⭐⭐⭐⭐  EnhancedCompositeLoss（Charbonnier + MS-SSIM + 幅度谱 + 镜面梯度 + 高光 BCE）
  ⭐⭐⭐⭐  高反光模拟数据增强 HighlightAugmentation
  ⭐⭐⭐   网络内嵌 SpecularAwareAttention + HighFrequencyEnhancement
  ⭐⭐     梯度一致性损失 GradientConsistencyLoss

增强可观测性：
  ✅ 终端日志完整展示全部子损失项（SSIM / L1 / Freq / Perceptual / Spec / Stat / Grad / Focal / Dice / GC）
  ✅ 验证阶段额外计算 MSE / PSNR 指标
  ✅ 验证样本可视化（输入 / GT / 预测 / 镜面反射 / 梯度 / 差异图）
  ✅ 子损失历史曲线独立绘制
  ✅ 训练总结文件包含完整子损失信息
  ✅ 每个 epoch 的损失 / SSIM / MAE / MSE / PSNR / 子损失写入 CSV 与 Excel
  ✅ 每轮写入断点，中断后可续训，并继续更新同一份 CSV、Excel 和训练曲线
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
import os
import random
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
# 五项损失的固定权重
# ════════════════════════════════════════════════════════════════════════

class ImprovedWeightScheduler:
    """五项损失的固定权重。停用项保持 0，不再按 epoch 做课程或退火。"""

    def __init__(self, total_epochs):
        self.T = max(total_epochs, 1)

    def get_weights(self, ep):
        del ep
        return {
            'lambda_ssim':        1.0,   # MS-SSIM
            'lambda_l1':          0.5,   # Charbonnier(r, target)
            'lambda_freq':        0.15,  # 幅度谱
            'lambda_perceptual':  0.0,
            'lambda_spec':        0.5,   # Charbonnier(b, gt_specular)
            'lambda_stat':        0.0,
            'lambda_grad':        0.0,
            'lambda_grad_edge':   0.8,   # 边缘加权 Charbonnier(g, Sobel(input))
            'lambda_grad_spec':   0.0,
            'lambda_focal':       0.0,
            'lambda_dice':        0.0,
            'lambda_gc':          0.2,   # 镜面梯度一致性
            'lambda_cs':          0.0,
            'lambda_s_lik':       1.0,   # 高光 BCE
            'lambda_s_cons':      0.0,
            'lambda_spec_peak':   0.0,
            'lambda_r_focal_hl':  0.0,
            'lambda_r_dual_cons': 0.0,
            'lambda_r_smooth':    0.0,
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


def build_visualization_loader_from_indices(val_dataset, indices, num_workers,
                                            pin_memory=False):
    """按已保存的验证集索引重建可视化样本，续训时仍画同一批图像。"""
    dataset_size = len(val_dataset)
    usable = [int(idx) for idx in indices if 0 <= int(idx) < dataset_size]
    if not usable:
        raise ValueError("保存的可视化索引已全部失效")
    vis_dataset = torch.utils.data.Subset(val_dataset, usable)
    vis_loader = DataLoader(
        vis_dataset,
        batch_size=len(usable),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)
    return vis_loader, usable


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


def refresh_training_figures(history, output_folder, current_epoch, include_full=True):
    """把训练曲线覆盖写回原路径，续训和中断后看到的仍是同一套图。"""
    if not history.get('train_losses'):
        return
    plot_training_curves_realtime(history, output_folder, current_epoch)
    if not include_full:
        return
    total = len(history['train_losses'])
    plot_training_history(history, output_folder, total)
    plot_individual_curves(history, output_folder, total)
    plot_sub_loss_history(history, output_folder, total)


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
# 每个 epoch 的指标表（CSV + Excel）
# ════════════════════════════════════════════════════════════════════════

EPOCH_METRICS_CSV = 'epoch_metrics.csv'
EPOCH_METRICS_XLSX = 'epoch_metrics.xlsx'

# 主指标列的固定顺序。子损失、学习率和权重列按前缀追加在后面。
_METRIC_LEADING_COLUMNS = [
    'epoch', 'epoch_time_s',
    'train_loss', 'val_loss', 'loss_gap',
    'train_ssim', 'val_ssim', 'ssim_gap',
    'train_mae', 'val_mae', 'mae_gap',
    'train_mse', 'val_mse', 'mse_gap',
    'train_psnr', 'val_psnr', 'psnr_gap',
    'composite',
    'is_best_overall', 'is_best_ssim', 'is_best_loss',
]

_METRIC_COLUMN_NOTES = [
    ('epoch', '轮次，从 1 开始'),
    ('epoch_time_s', '该轮训练加验证耗时，单位秒'),
    ('train_loss / val_loss', '训练 / 验证总损失，越低越好'),
    ('train_ssim / val_ssim', '漫反射分支 SSIM 分数，越高越好，由 1 - ssim 子损失得到'),
    ('train_mae / val_mae', '漫反射分支 MAE'),
    ('train_mse / val_mse', '漫反射分支 MSE'),
    ('train_psnr / val_psnr', '漫反射分支 PSNR，单位 dB，越高越好'),
    ('*_gap', '训练值与验证值之差的绝对值'),
    ('composite', '综合评分：0.7×val_ssim + 0.3/(1+val_loss)，越高越好'),
    ('is_best_*', '该轮是否刷新对应最优记录，1 表示是，0 表示否'),
    ('lr_*', '该轮结束时各参数组学习率'),
    ('train_sub_* / val_sub_*', '损失函数字典中的子损失原值。其中 ssim 是 1-SSIM 损失，不是 SSIM 分数'),
    ('weight_*', '该轮使用的损失权重'),
    ('net_*', '网络结构读数，如自适应跳跃连接的 beta 与 gate 均值'),
]


def _as_metric_number(value):
    """把张量或 numpy 标量转成可写入表格的 Python 数。非有限值返回 None。"""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return int(value)
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    return value


def _ordered_metric_columns(records):
    """主指标在前，学习率、子损失、权重和网络读数按稳定顺序排在后面。"""
    present = set()
    for record in records:
        present.update(record.keys())

    ordered = [name for name in _METRIC_LEADING_COLUMNS if name in present]

    def _append_group(prefix, key_order):
        grouped = []
        for key in key_order:
            name = f'{prefix}{key}'
            if name in present:
                grouped.append(name)
        extras = sorted(
            name for name in present
            if name.startswith(prefix) and name not in grouped)
        ordered.extend(grouped)
        ordered.extend(extras)

    _append_group('lr_', [])
    _append_group('train_sub_', SUB_LOSS_KEYS)
    _append_group('val_sub_', SUB_LOSS_KEYS)
    _append_group('weight_', [])
    _append_group('net_', ['beta_h4', 'beta_h8', 'gate_h4_mean', 'gate_h8_mean'])

    used = set(ordered)
    ordered.extend(sorted(present - used))
    return ordered


def build_epoch_metric_record(epoch, epoch_time, learning_rates, weights,
                              train_loss, train_ssim, train_mae, train_mse, train_psnr,
                              val_loss, val_ssim, val_mae, val_mse, val_psnr,
                              train_subs, val_subs, composite,
                              is_best_overall, is_best_ssim, is_best_loss,
                              net_stats=None):
    """把一个 epoch 的全部标量指标收成一行。SSIM 主列是分数，子损失列保留损失原值。"""
    def _gap(left, right):
        left_num = _as_metric_number(left)
        right_num = _as_metric_number(right)
        if left_num is None or right_num is None:
            return None
        return abs(left_num - right_num)

    record = {
        'epoch': int(epoch),
        'epoch_time_s': _as_metric_number(epoch_time),
        'train_loss': _as_metric_number(train_loss),
        'val_loss': _as_metric_number(val_loss),
        'loss_gap': _gap(train_loss, val_loss),
        'train_ssim': _as_metric_number(train_ssim),
        'val_ssim': _as_metric_number(val_ssim),
        'ssim_gap': _gap(train_ssim, val_ssim),
        'train_mae': _as_metric_number(train_mae),
        'val_mae': _as_metric_number(val_mae),
        'mae_gap': _gap(train_mae, val_mae),
        'train_mse': _as_metric_number(train_mse),
        'val_mse': _as_metric_number(val_mse),
        'mse_gap': _gap(train_mse, val_mse),
        'train_psnr': _as_metric_number(train_psnr),
        'val_psnr': _as_metric_number(val_psnr),
        'psnr_gap': _gap(train_psnr, val_psnr),
        'composite': _as_metric_number(composite),
        'is_best_overall': int(bool(is_best_overall)),
        'is_best_ssim': int(bool(is_best_ssim)),
        'is_best_loss': int(bool(is_best_loss)),
    }
    for name, value in (learning_rates or {}).items():
        record[f'lr_{name}'] = _as_metric_number(value)
    for key, value in (train_subs or {}).items():
        record[f'train_sub_{key}'] = _as_metric_number(value)
    for key, value in (val_subs or {}).items():
        record[f'val_sub_{key}'] = _as_metric_number(value)
    for key, value in (weights or {}).items():
        record[f'weight_{key}'] = _as_metric_number(value)
    for key, value in (net_stats or {}).items():
        record[f'net_{key}'] = _as_metric_number(value)
    return record


def _replace_metrics_file(temp_path, dest_path):
    """覆盖目标文件。Excel 正打开目标文件时改写到同名 .new 文件，避免中断训练。"""
    try:
        os.replace(temp_path, dest_path)
        return dest_path
    except PermissionError:
        fallback = dest_path + '.new'
        os.replace(temp_path, fallback)
        print(f"  [metrics] 无法覆盖 {dest_path}（文件可能正被打开），已写入 {fallback}")
        return fallback


def _write_metrics_csv(records, columns, csv_path):
    temp_path = csv_path + '.tmp'
    with open(temp_path, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            row = {}
            for name in columns:
                value = record.get(name, None)
                if isinstance(value, float):
                    row[name] = f'{value:.10g}'
                elif value is None:
                    row[name] = ''
                else:
                    row[name] = value
            writer.writerow(row)
    return _replace_metrics_file(temp_path, csv_path)


def _style_metrics_sheet(worksheet, columns):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    header_font = Font(bold=True, color='FFFFFF')
    header_fill = PatternFill('solid', fgColor='1F4E79')
    for cell in worksheet[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', vertical='center')

    for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row,
                                   min_col=1, max_col=worksheet.max_column):
        for cell in row:
            header = columns[cell.column - 1]
            if not isinstance(cell.value, float):
                continue
            if header.startswith('lr_') or header.startswith('weight_'):
                cell.number_format = '0.00E+00'
            elif header in ('train_ssim', 'val_ssim', 'ssim_gap', 'composite'):
                cell.number_format = '0.0000'
            elif header.startswith('is_best') or header == 'epoch':
                cell.number_format = '0'
            elif header == 'epoch_time_s':
                cell.number_format = '0.0'
            else:
                cell.number_format = '0.000000'

    worksheet.freeze_panes = 'B2'
    worksheet.auto_filter.ref = worksheet.dimensions
    for index, name in enumerate(columns, start=1):
        worksheet.column_dimensions[get_column_letter(index)].width = max(14, min(32, len(name) + 2))
    worksheet.row_dimensions[1].height = 22


def _write_metrics_xlsx(records, columns, xlsx_path):
    """优先用 openpyxl 写带表头样式的工作簿；否则用 pandas。两者都没有时返回 None。"""
    try:
        from openpyxl import Workbook
    except ImportError:
        try:
            import pandas as pd
        except ImportError:
            return None
        frame = pd.DataFrame(records, columns=columns)
        temp_path = xlsx_path + '.tmp.xlsx'
        frame.to_excel(temp_path, index=False, sheet_name='epoch_metrics')
        return _replace_metrics_file(temp_path, xlsx_path)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'epoch_metrics'
    sheet.append(columns)
    for record in records:
        sheet.append([record.get(name, None) for name in columns])
    _style_metrics_sheet(sheet, columns)

    notes = workbook.create_sheet('字段说明')
    notes.append(['列名', '含义'])
    for name, meaning in _METRIC_COLUMN_NOTES:
        notes.append([name, meaning])
    _style_metrics_sheet(notes, ['列名', '含义'])
    notes.column_dimensions['A'].width = 36
    notes.column_dimensions['B'].width = 78

    temp_path = xlsx_path + '.tmp.xlsx'
    workbook.save(temp_path)
    return _replace_metrics_file(temp_path, xlsx_path)


def save_epoch_metrics_table(records, output_folder):
    """
    将目前已完成的全部 epoch 重写到 CSV，并在环境支持时同时写 Excel。

    每个 epoch 都整表重写，训练中途停止后，已完成轮次的数值仍然保留，
    可直接用 Excel 或绘图脚本重画曲线、对比不同实验。
    """
    if not records:
        return None, None

    os.makedirs(output_folder, exist_ok=True)
    columns = _ordered_metric_columns(records)
    csv_path = _write_metrics_csv(
        records, columns, os.path.join(output_folder, EPOCH_METRICS_CSV))
    xlsx_path = _write_metrics_xlsx(
        records, columns, os.path.join(output_folder, EPOCH_METRICS_XLSX))
    return csv_path, xlsx_path


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
        f.write(f'  Epoch Metrics CSV : {os.path.join(output_folder, EPOCH_METRICS_CSV)}\n')
        f.write(f'  Epoch Metrics XLSX: {os.path.join(output_folder, EPOCH_METRICS_XLSX)}\n')
        f.write(f'  Resume State      : {os.path.join(output_folder, RESUME_STATE_NAME)}\n')
        f.write(f'  Val Visualization : {os.path.join(output_folder, "val_visualization/")}\n')
        f.write(f'  This Summary      : {save_path}\n')
        f.write(sep)

    return save_path


# ════════════════════════════════════════════════════════════════════════
# 断点续训
# ════════════════════════════════════════════════════════════════════════

RESUME_STATE_NAME = 'resume_state.pth'

_RESUME_CFG_KEYS = (
    'EPOCHS', 'BATCH_SIZE', 'BASE_LR', 'WEIGHT_DECAY', 'ACCUMULATION_STEPS',
    'GRAD_CLIP_VALUE', 'IMAGE_SIZE', 'TRAIN_SPLIT_RATIO', 'USE_GN', 'USE_EMA',
    'EMA_DECAY', 'USE_AMP', 'USE_HIGHLIGHT_AUG', 'HIGHLIGHT_AUG_P',
    'INPUT_DIR', 'TARGET_DIR', 'VAL_INPUT_DIR', 'VAL_TARGET_DIR',
)


def resolve_resume_target(resume_value, experiments_root='experiments'):
    """
    把续训配置解析成 (实验目录, resume_state.pth)。

    resume_value:
      None / ''        不续训
      'latest'         使用 experiments 下最近写入的断点
      目录或断点文件路径  从指定实验继续
    """
    if resume_value is None or resume_value is False or str(resume_value).strip() == '':
        return None, None

    text = str(resume_value).strip()
    if text.lower() in {'latest', 'auto', 'last'}:
        matches = []
        if os.path.isdir(experiments_root):
            for dirpath, _dirnames, filenames in os.walk(experiments_root):
                if RESUME_STATE_NAME in filenames:
                    matches.append(os.path.join(dirpath, RESUME_STATE_NAME))
        if not matches:
            raise FileNotFoundError(
                f"在 {experiments_root} 下没有找到 {RESUME_STATE_NAME}，无法自动续训")
        state_path = max(matches, key=os.path.getmtime)
        return os.path.abspath(os.path.dirname(state_path)), os.path.abspath(state_path)

    path = os.path.abspath(text)
    if os.path.isfile(path):
        return os.path.dirname(path), path
    if os.path.isdir(path):
        state_path = os.path.join(path, RESUME_STATE_NAME)
        if not os.path.isfile(state_path):
            raise FileNotFoundError(
                f"目录 {path} 中没有 {RESUME_STATE_NAME}。请确认这是同一次训练的实验目录。")
        return path, os.path.abspath(state_path)
    raise FileNotFoundError(f"续训路径不存在: {resume_value}")


def load_resume_state(path, device):
    """读取断点。文件里包含模型、优化器、学习率日程、历史指标和可视化样本。"""
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if not isinstance(state, dict) or 'model_state_dict' not in state:
        raise ValueError(f"{path} 不是可续训的断点文件")
    if 'completed_epochs' not in state or 'history' not in state:
        raise ValueError(f"{path} 缺少训练历史，无法在同一份表格和曲线上继续")
    return state


def _atomic_torch_save(payload, dest_path):
    """先写入临时文件再替换，避免保存过程中断把上一份断点写坏。"""
    temp_path = dest_path + '.tmp'
    torch.save(payload, temp_path)
    try:
        os.replace(temp_path, dest_path)
        return dest_path
    except PermissionError:
        fallback = dest_path + '.new'
        os.replace(temp_path, fallback)
        print(f"  [resume] 无法覆盖 {dest_path}，已写入 {fallback}")
        return fallback


def _history_to_plain(history):
    plain = {}
    for key, value in history.items():
        if isinstance(value, defaultdict):
            plain[key] = {name: list(items) for name, items in value.items()}
        elif isinstance(value, list):
            plain[key] = list(value)
        else:
            plain[key] = value
    return plain


def _history_from_plain(plain):
    history = {
        'train_losses': list(plain.get('train_losses', [])),
        'val_losses': list(plain.get('val_losses', [])),
        'train_ssims': list(plain.get('train_ssims', [])),
        'val_ssims': list(plain.get('val_ssims', [])),
        'train_maes': list(plain.get('train_maes', [])),
        'val_maes': list(plain.get('val_maes', [])),
        'val_mses': list(plain.get('val_mses', [])),
        'val_psnrs': list(plain.get('val_psnrs', [])),
        'train_mses': list(plain.get('train_mses', [])),
        'train_psnrs': list(plain.get('train_psnrs', [])),
        'train_sub_losses': defaultdict(list),
        'val_sub_losses': defaultdict(list),
        'epoch_records': list(plain.get('epoch_records', [])),
    }
    for key, values in plain.get('train_sub_losses', {}).items():
        history['train_sub_losses'][key] = list(values)
    for key, values in plain.get('val_sub_losses', {}).items():
        history['val_sub_losses'][key] = list(values)
    return history


def _capture_rng_state():
    state = {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
        'cuda': None,
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    try:
        torch_state = state['torch']
        if hasattr(torch_state, 'device') and torch_state.device.type != 'cpu':
            torch_state = torch_state.cpu()
        torch.set_rng_state(torch_state)
        np.random.set_state(state['numpy'])
        random.setstate(state['python'])
        cuda_state = state.get('cuda')
        if cuda_state is not None and torch.cuda.is_available():
            if len(cuda_state) != torch.cuda.device_count():
                print("  [续训] CUDA 设备数量与断点不一致，跳过 CUDA 随机状态恢复")
                return
            restored = []
            for index, tensor in enumerate(cuda_state):
                if tensor.device.type == 'cpu':
                    tensor = tensor.cuda(index)
                restored.append(tensor)
            torch.cuda.set_rng_state_all(restored)
    except Exception as exc:
        print(f"  [续训] 随机状态恢复失败，后续数据打乱顺序可能与未中断时不同: {exc}")


def _warn_resume_config(snapshot, cfg):
    if not snapshot:
        return
    for key in _RESUME_CFG_KEYS:
        old = snapshot.get(key, None)
        new = cfg.get(key, None)
        if old is None or old == new:
            continue
        if isinstance(old, (tuple, list)) and isinstance(new, (tuple, list)):
            if tuple(old) == tuple(new):
                continue
        print(f"  [续训] 配置与断点不同: {key}  断点={old}  当前={new}")


def _empty_history():
    return {
        'train_losses': [], 'val_losses': [],
        'train_ssims':  [], 'val_ssims':  [],
        'train_maes':   [], 'val_maes':   [],
        'val_mses':     [], 'val_psnrs':  [],
        'train_mses':   [], 'train_psnrs': [],
        'train_sub_losses': defaultdict(list),
        'val_sub_losses':   defaultdict(list),
        'epoch_records': [],
    }


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

    resume_file = cfg.get('RESUME_FILE')
    resume_state = None
    if resume_file:
        resume_state = load_resume_state(resume_file, device)
        cfg['OUTPUT_FOLDER'] = os.path.dirname(os.path.abspath(resume_file))
        completed_epochs = int(resume_state.get('completed_epochs', 0))
        print(f"断点续训: {os.path.abspath(resume_file)}")
        print(f"  已完成 {completed_epochs} 轮，目标 {cfg['EPOCHS']} 轮")
        _warn_resume_config(resume_state.get('cfg_snapshot'), cfg)
    else:
        completed_epochs = 0

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
    if resume_state is not None:
        try:
            model.load_state_dict(resume_state['model_state_dict'])
        except RuntimeError as exc:
            raise RuntimeError(
                "断点与当前模型结构不一致。续训请保持 USE_GN 等结构开关和当初训练时相同。"
            ) from exc
        print(f"[续训] 已恢复模型权重（完成至第 {int(resume_state['completed_epochs'])} 轮）")

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
    saved_vis_indices = resume_state.get('vis_indices') if resume_state else None
    saved_vis_seed = resume_state.get('vis_seed') if resume_state else None
    if vis_interval > 0 and vis_num_samples > 0:
        pin_memory = (device.type == 'cuda')
        if saved_vis_indices:
            try:
                vis_loader, vis_indices = build_visualization_loader_from_indices(
                    val_ds, saved_vis_indices, cfg['NUM_WORKERS'], pin_memory=pin_memory)
                vis_seed = saved_vis_seed
                print(f"可视化样本: 续训沿用原来的 {len(vis_indices)} 张 "
                      f"(seed={vis_seed}, val_indices={vis_indices})")
            except ValueError as exc:
                print(f"  [续训] 原可视化索引不可用，改为按原种子重抽: {exc}")
                saved_vis_indices = None
        if vis_loader is None:
            vis_loader, vis_indices, vis_seed = build_fixed_visualization_loader(
                val_ds,
                num_samples=vis_num_samples,
                num_workers=cfg['NUM_WORKERS'],
                pin_memory=pin_memory,
                seed=saved_vis_seed if saved_vis_seed is not None else cfg.get('VIS_SAMPLE_SEED'))
            print(f"可视化样本: 本次训练固定抽取 {len(vis_indices)} 张 "
                  f"(seed={vis_seed}, val_indices={vis_indices})")
        cfg['VIS_SAMPLE_SEED_USED'] = vis_seed
        cfg['VIS_SAMPLE_INDICES'] = vis_indices
    else:
        cfg['VIS_SAMPLE_SEED_USED'] = None
        cfg['VIS_SAMPLE_INDICES'] = []
        print("可视化样本: 已禁用")

    # ══════════ 3. 损失函数 ══════════
    criterion = EnhancedCompositeLoss(
        use_perceptual=False, use_ohem=False).to(device)
    ws = ImprovedWeightScheduler(cfg['EPOCHS'])

    # 打印损失函数结构
    print(f"\n损失函数: EnhancedCompositeLoss")
    print(f"  生效项: MS-SSIM, Charbonnier(r), 幅度谱, Charbonnier(b)+梯度一致性, 边缘加权梯度, 高光BCE")
    print(f"  use_amp={cfg['USE_AMP']}")
    print(f"  子损失项: {', '.join(SUB_LOSS_KEYS)}")

    # ══════════ 4. 优化器 + 调度器 ══════════
    print("\n分层学习率配置:")
    optimizer = build_optimizer_v2(model, cfg['BASE_LR'], cfg['WEIGHT_DECAY'])
    current_scheduler_build = {
        'warmup_iters': min(500, math.ceil(len(train_loader) / cfg['ACCUMULATION_STEPS']) * 3),
        'total_epochs': cfg['EPOCHS'],
        'iters_per_epoch': math.ceil(len(train_loader) / cfg['ACCUMULATION_STEPS']),
    }
    if resume_state is not None and resume_state.get('scheduler_build'):
        scheduler_build = dict(resume_state['scheduler_build'])
        if scheduler_build.get('iters_per_epoch') != current_scheduler_build['iters_per_epoch']:
            print("  [续训] 每个 epoch 的优化步数与断点不一致，学习率日程仍按断点里的步数恢复")
        if scheduler_build.get('total_epochs') != cfg['EPOCHS']:
            print(f"  [续训] 学习率日程仍按最初的 {scheduler_build.get('total_epochs')} 轮余弦退火；"
                  f"超过后保持最低学习率。当前目标轮数为 {cfg['EPOCHS']}")
    else:
        scheduler_build = current_scheduler_build
    scheduler = build_scheduler(optimizer, **scheduler_build)
    if resume_state is not None and resume_state.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(resume_state['optimizer_state_dict'])
    if resume_state is not None and resume_state.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(resume_state['scheduler_state_dict'])

    # ══════════ 5. AMP + EMA ══════════
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and cfg['USE_AMP']))
    if resume_state is not None and resume_state.get('scaler_state_dict') is not None:
        scaler.load_state_dict(resume_state['scaler_state_dict'])
    ema = EMA(model, cfg['EMA_DECAY']) if cfg['USE_EMA'] else None
    if ema is not None and resume_state is not None and resume_state.get('ema_shadow'):
        for name, tensor in resume_state['ema_shadow'].items():
            if name in ema.shadow:
                ema.shadow[name] = tensor.detach().to(device)
        print("[续训] 已恢复 EMA 权重")

    # ══════════ 6. 历史记录 + 最优模型跟踪 ══════════
    if resume_state is not None:
        history = _history_from_plain(resume_state.get('history', {}))
        epoch_times = list(resume_state.get('epoch_times', []))
    else:
        history = _empty_history()
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
    if resume_state is not None:
        best_composite = float(resume_state.get('best_composite', best_composite))
        best_val_ssim = float(resume_state.get('best_val_ssim', best_val_ssim))
        best_val_loss = float(resume_state.get('best_val_loss', best_val_loss))
        saved_best_info = resume_state.get('best_info') or {}
        for tag, info in best_info.items():
            loaded = saved_best_info.get(tag)
            if not loaded:
                continue
            info.update(loaded)
            info['path'] = {
                'Best Overall (SSIM×0.7 + NormLoss×0.3)': best_composite_path,
                'Best SSIM': best_ssim_path,
                'Best Loss': best_loss_path,
            }[tag]

    # 早停
    early_stopping_counter = int(resume_state.get('early_stopping_counter', 0)) if resume_state else 0
    elapsed_offset = float(resume_state.get('elapsed_seconds', 0.0)) if resume_state else 0.0
    start_epoch = completed_epochs

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
    print(f"  断点续训    : {os.path.abspath(resume_file) if resume_file else '关闭'}")
    if start_epoch:
        print(f"  续训起点    : 第 {start_epoch + 1} 轮（已完成 {start_epoch} 轮）")
    print(f"  输出目录    : {os.path.abspath(output_folder)}")
    print(f"\n模型保存路径:")
    print(f"  综合最优 -> {best_composite_path}")
    print(f"  SSIM最优 -> {best_ssim_path}")
    print(f"  Loss最优 -> {best_loss_path}")
    print(f"  最新模型 -> {last_model_path}")
    print(f"  续训断点 -> {os.path.join(output_folder, RESUME_STATE_NAME)}")
    print(f"  综合评分权重: SSIM×{COMPOSITE_ALPHA:.1f} + Loss归一化×{1 - COMPOSITE_ALPHA:.1f}")

    def _save_resume(completed):
        """保存可续训断点。调用时本轮指标表已经写入同一实验目录。"""
        payload = {
            'completed_epochs': int(completed),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'scheduler_build': scheduler_build,
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'ema_shadow': (
                {name: tensor.detach().cpu() for name, tensor in ema.shadow.items()}
                if ema is not None else None),
            'history': _history_to_plain(history),
            'epoch_times': list(epoch_times),
            'best_composite': best_composite,
            'best_val_ssim': best_val_ssim,
            'best_val_loss': best_val_loss,
            'best_info': best_info,
            'early_stopping_counter': early_stopping_counter,
            'vis_seed': cfg.get('VIS_SAMPLE_SEED_USED'),
            'vis_indices': list(cfg.get('VIS_SAMPLE_INDICES') or []),
            'elapsed_seconds': elapsed_offset + (time.time() - session_start),
            'rng_state': _capture_rng_state(),
            'cfg_snapshot': {key: cfg.get(key) for key in _RESUME_CFG_KEYS},
            'train_size': len(train_ds),
            'val_size': len(val_ds),
        }
        return _atomic_torch_save(payload, os.path.join(output_folder, RESUME_STATE_NAME))

    if start_epoch >= cfg['EPOCHS'] and start_epoch > 0:
        print(f"\n断点已完成 {start_epoch} 轮，当前 EPOCHS={cfg['EPOCHS']}，没有剩余轮次。")
        print("若要在同一份结果上继续加训，请把 EPOCHS 调大后再运行。")
        refresh_training_figures(history, output_folder, start_epoch, include_full=True)
        total_time_str = str(timedelta(seconds=int(elapsed_offset)))
        save_training_summary(output_folder, cfg, history, best_info, total_time_str)
        history['best_info'] = best_info
        history['total_time_str'] = total_time_str
        history['actual_epochs'] = start_epoch
        return history

    if start_epoch > 0 and history['train_losses']:
        if history.get('epoch_records'):
            try:
                csv_path, xlsx_path = save_epoch_metrics_table(
                    history['epoch_records'], output_folder)
                print(f"  指标表继续写入: {csv_path}")
                if xlsx_path:
                    print(f"  指标表继续写入: {xlsx_path}")
            except Exception as exc:
                print(f"  [metrics] 续训时重写指标表失败: {exc}")
        refresh_training_figures(history, output_folder, start_epoch, include_full=True)
        print("  已把训练曲线覆盖回原来的图片")
        saved_train_size = resume_state.get('train_size')
        saved_val_size = resume_state.get('val_size')
        if saved_train_size is not None and (
                len(train_ds) != saved_train_size or len(val_ds) != saved_val_size):
            print("  [续训] 当前训练集或验证集样本数与断点不同，可视化索引仍按原验证集位置使用")

    # ══════════ 7. 训练循环 ══════════
    session_start = time.time()
    _restore_rng_state(resume_state.get('rng_state') if resume_state else None)
    train_model._prev_lr = optimizer.param_groups[0]['lr']
    print(f"\n{'继续训练' if start_epoch else '训练开始'}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 78}\n")

    last_completed_epoch = start_epoch
    for ep in range(start_epoch, cfg['EPOCHS']):
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
        net_stats = {}
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'fuse_h4') and hasattr(model.encoder, 'fuse_h8'):
            gate4 = model.encoder.fuse_h4.last_gate_mean
            gate8 = model.encoder.fuse_h8.last_gate_mean
            net_stats['beta_h4'] = model.encoder.fuse_h4.beta.item()
            net_stats['beta_h8'] = model.encoder.fuse_h8.beta.item()
            if gate4 is not None:
                net_stats['gate_h4_mean'] = gate4.item()
            if gate8 is not None:
                net_stats['gate_h8_mean'] = gate8.item()
            gate4_str = f"{net_stats['gate_h4_mean']:.6f}" if 'gate_h4_mean' in net_stats else "n/a"
            gate8_str = f"{net_stats['gate_h8_mean']:.6f}" if 'gate_h8_mean' in net_stats else "n/a"
            print(f"  [Adaptive skip] beta_h4={net_stats['beta_h4']:.6f}, "
                  f"beta_h8={net_stats['beta_h8']:.6f}, "
                  f"gate_h4_mean={gate4_str}, gate_h8_mean={gate8_str}")

        # 在更新最优记录之前记下本轮是否刷新，供指标表使用。
        is_best_overall = composite > best_composite
        is_best_ssim = vs > best_val_ssim
        is_best_loss = vl < best_val_loss

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

        # ── 每个 epoch 的数值表（CSV，环境支持时同时写 Excel）──
        lr_groups = {
            group.get('name', f'group_{idx}'): group['lr']
            for idx, group in enumerate(optimizer.param_groups)
        }
        history['epoch_records'].append(build_epoch_metric_record(
            epoch=ep + 1,
            epoch_time=epoch_time,
            learning_rates=lr_groups,
            weights=w,
            train_loss=tl, train_ssim=ts, train_mae=tm,
            train_mse=train_mse, train_psnr=train_psnr,
            val_loss=vl, val_ssim=vs, val_mae=vm,
            val_mse=val_mse, val_psnr=val_psnr,
            train_subs=train_subs, val_subs=val_subs,
            composite=composite,
            is_best_overall=is_best_overall,
            is_best_ssim=is_best_ssim,
            is_best_loss=is_best_loss,
            net_stats=net_stats))
        try:
            csv_path, xlsx_path = save_epoch_metrics_table(
                history['epoch_records'], output_folder)
            if ep == start_epoch:
                print(f"  每轮指标表: {csv_path}")
                if xlsx_path:
                    print(f"  每轮指标表: {xlsx_path}")
                else:
                    print("  当前环境没有 openpyxl 或 pandas，未生成 Excel；CSV 已包含全部数值")
        except Exception as exc:
            print(f"  [metrics] 指标表写入失败: {exc}")

        last_completed_epoch = ep + 1
        try:
            resume_path = _save_resume(last_completed_epoch)
            if ep == start_epoch:
                print(f"  续训断点: {resume_path}")
        except Exception as exc:
            print(f"  [resume] 断点写入失败: {exc}")

        # ── 同一套训练曲线，覆盖原来的图片 ──
        try:
            refresh_training_figures(history, output_folder, ep + 1, include_full=True)
        except Exception as exc:
            print(f"  [curves] 训练曲线更新失败: {exc}")

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
    if last_completed_epoch > 0:
        torch.save({
            'epoch': last_completed_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'image_size': cfg['IMAGE_SIZE'],
        }, last_model_path)

    # ══════════ 9. 训练总结 ══════════
    total_time = elapsed_offset + (time.time() - session_start)
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
    print(f"  ├── {EPOCH_METRICS_CSV:<31s} (每轮损失 / SSIM / 子损失)")
    print(f"  ├── {EPOCH_METRICS_XLSX:<31s} (同上，Excel；含字段说明)")
    print(f"  ├── {RESUME_STATE_NAME:<31s} (中断后续训用的断点)")
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

    # ========== 断点续训 ==========
    # None     : 新开一次训练，结果写到下面带时间戳的新目录
    # 'latest' : 从 experiments 里最近一次的断点接着训
    # 目录路径 : 从该实验目录接着训，CSV、Excel 和训练曲线都继续写回这个目录
    RESUME_FOLDER         = None

    # ========== 输出目录（仅新训练时使用；续训时自动改回原实验目录） ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FOLDER = f'experiments/yffc_v2_{timestamp}'
    resume_file = None
    if RESUME_FOLDER:
        OUTPUT_FOLDER, resume_file = resolve_resume_target(RESUME_FOLDER)

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
        'RESUME_FILE':            resume_file,
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
