# -*- coding: utf-8 -*-
"""
compare_common.py — 对比模型（DC-UNet / HDRNet）训练共享工具
=============================================================

本模块把 DC-UNet 与 HDRNet 两个对比模型训练脚本中**公共的、且必须保持一致**
的部分集中实现，确保两个基线之间、以及与主模型 train.py / Y_FFC_unified.py
之间的评价指标口径完全对齐。

提供：
  • EvalSSIM         —— 标准 SSIM 指标（不做逐像素下限截断，杜绝 SSIM 虚高）
  • compute_psnr     —— PSNR（与主模型同一公式：10*log10(1/MSE)，MSE 极小封顶 60dB）
  • MetricTracker    —— 统一累计 Loss / SSIM / MAE / MSE / PSNR
  • FiniteStateGuard —— 参数/优化器 NaN 恢复（根治训练后期崩溃为 nan 不可逆的问题）
  • build_fixed_vis_loader / visualize_single_output —— 按指定轮数做验证集结果可视化
  • plot_training_history —— Loss/SSIM/MAE/MSE/PSNR 五行训练曲线
  • save_training_summary —— 训练总结（含 PSNR）
  • setup_matplotlib_fonts

设计要点（为什么这样改）：
─────────────────────────────────────────────────────────────
[SSIM 虚高] HDRNet 旧版用 `ssim_map.clamp(0,1).mean()`，在逐像素层面把负的
            SSIM 像素抬到 0，系统性高估；DCUNet 旧版用的又是另一套（Y_FFC 的
            软截断），两个基线口径不一致、互相不可比。这里统一改为对“原始
            ssim_map”求均值，仅在“最终标量上”为显示截断到 [0,1]，与教科书
            SSIM 一致，且两个基线、主模型三者口径统一。
[后期崩溃] 旧版只在 loss 非有限时跳过该 batch，但一旦参数/BN running stats 被
            写成 NaN，之后所有 forward 恒为 NaN → 全程跳过 → 指标永久 nan。
            这里用 FiniteStateGuard 维护“最近一次全有限”的参数快照，检测到污染
            就回滚参数 + 清理 Adam 动量，使训练可恢复（与主模型 _repair 思路一致）。
─────────────────────────────────────────────────────────────
"""

import os
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams


# ════════════════════════════════════════════════════════════════════════
# 字体
# ════════════════════════════════════════════════════════════════════════

def setup_matplotlib_fonts():
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                   'WenQuanYi Micro Hei', 'Arial Unicode MS']
    rcParams['axes.unicode_minus'] = False


# ════════════════════════════════════════════════════════════════════════
# 标准 SSIM 指标（统一口径，杜绝 SSIM 虚高）
# ════════════════════════════════════════════════════════════════════════

def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)          # (W, W)
    return window_2d.unsqueeze(0).unsqueeze(0)           # (1, 1, W, W)


class EvalSSIM(nn.Module):
    """标准 SSIM（data_range=1.0）。

    关键区别（修复 SSIM 虚高）：
      - 对 *原始* ssim_map 求均值（不在逐像素层面把负值截断到 0/-0.5）；
      - 只在返回的“标量”上为显示稳定截断到 [0, 1]。
    这样得到的 SSIM 与教科书定义一致，且 DC-UNet / HDRNet 两个基线、主模型
    三者口径完全统一，可直接横向比较。

    forward 返回可反传的标量（用于 1-SSIM 损失）；做指标时取 .clamp(0,1).item()。
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 channels: int = 1, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.data_range = data_range
        win = _gaussian_window(window_size, sigma)
        self.register_buffer('window', win.expand(channels, 1, window_size, window_size).contiguous())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        pad = self.window_size // 2
        c = pred.size(1)
        w = self.window.to(dtype=pred.dtype, device=pred.device)
        if c != self.channels:
            w = w[:1].expand(c, 1, self.window_size, self.window_size).contiguous()

        mu1 = F.conv2d(pred, w, padding=pad, groups=c)
        mu2 = F.conv2d(target, w, padding=pad, groups=c)
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, w, padding=pad, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(target * target, w, padding=pad, groups=c) - mu2_sq
        sigma12 = F.conv2d(pred * target, w, padding=pad, groups=c) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
        # 注意：对“原始 map”求均值，不做逐像素 clamp（这正是去虚高的关键）
        return ssim_map.mean()


def ssim_to_metric(ssim_raw: torch.Tensor) -> float:
    """把 EvalSSIM 的原始标量转成用于报告的 SSIM 指标（截断到 [0,1]）。"""
    return float(torch.clamp(ssim_raw.detach(), 0.0, 1.0).item())


# ════════════════════════════════════════════════════════════════════════
# PSNR（与主模型同一公式）
# ════════════════════════════════════════════════════════════════════════

def compute_psnr(mse_val: float, max_db: float = 60.0) -> float:
    """图像范围 [0,1]：PSNR = 10*log10(1/MSE)，MSE 极小封顶 max_db。"""
    if mse_val is None or not math.isfinite(mse_val):
        return float('nan')
    if mse_val > 1e-10:
        return 10.0 * math.log10(1.0 / mse_val)
    return max_db


# ════════════════════════════════════════════════════════════════════════
# 统一指标累计器
# ════════════════════════════════════════════════════════════════════════

class MetricTracker:
    """累计 Loss / SSIM / MAE / MSE / PSNR，口径与主模型 train.py 对齐。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.tot_loss = 0.0
        self.tot_ssim = 0.0
        self.tot_mae = 0.0
        self.tot_mse = 0.0
        self.tot_psnr = 0.0

    def update(self, loss_val, ssim_val, mae_val, mse_val):
        self.n += 1
        self.tot_loss += float(loss_val)
        self.tot_ssim += float(ssim_val)
        self.tot_mae += float(mae_val)
        self.tot_mse += float(mse_val)
        self.tot_psnr += compute_psnr(float(mse_val))

    def compute(self):
        d = max(self.n, 1)
        return (self.tot_loss / d, self.tot_ssim / d, self.tot_mae / d,
                self.tot_mse / d, self.tot_psnr / d)

    @property
    def valid_batches(self):
        return self.n


# ════════════════════════════════════════════════════════════════════════
# NaN / Inf 恢复（根治训练后期不可逆崩溃）
# ════════════════════════════════════════════════════════════════════════

def model_is_finite(model) -> bool:
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return False
    for b in model.buffers():
        if torch.is_floating_point(b) and not torch.isfinite(b).all():
            return False
    return True


def sanitize_optimizer_state(optimizer) -> int:
    """清理 Adam/AdamW 中非有限的动量等状态张量。"""
    repaired = 0
    with torch.no_grad():
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v) and not torch.isfinite(v).all():
                    v.copy_(torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
                    repaired += 1
    return repaired


class FiniteStateGuard:
    """维护“最近一次全有限”的模型参数快照，污染时回滚。

    用法：
      guard = FiniteStateGuard(model)
      ...每个 epoch 训练开始前若模型有限：guard.snapshot()
      ...检测到 output/loss/grad 非有限：guard.restore(model, optimizer)
    """

    def __init__(self, model):
        self._state = None
        if model_is_finite(model):
            self.snapshot(model)

    def snapshot(self, model):
        # 存 CPU 副本，省显存；包含 BN running stats（buffers）
        self._state = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}

    def has_snapshot(self) -> bool:
        return self._state is not None

    def restore(self, model, optimizer=None) -> int:
        """回滚到最近的有限快照；若无快照则直接 nan_to_num 当前参数。返回修复项数。"""
        repaired = 0
        if self._state is not None:
            with torch.no_grad():
                cur = model.state_dict()
                for k, v in self._state.items():
                    if k in cur:
                        cur[k].copy_(v.to(device=cur[k].device, dtype=cur[k].dtype))
                        repaired += 1
        else:
            with torch.no_grad():
                for p in model.parameters():
                    if not torch.isfinite(p).all():
                        p.copy_(torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=-1.0))
                        repaired += 1
                for b in model.buffers():
                    if torch.is_floating_point(b) and not torch.isfinite(b).all():
                        b.copy_(torch.nan_to_num(b, nan=0.0, posinf=1.0, neginf=-1.0))
                        repaired += 1
        if optimizer is not None:
            repaired += sanitize_optimizer_state(optimizer)
        return repaired


# ════════════════════════════════════════════════════════════════════════
# 可视化：按指定轮数对验证集结果作图（适配单输出对比模型）
# ════════════════════════════════════════════════════════════════════════

def build_fixed_vis_loader(val_dataset, num_samples, num_workers=0,
                           pin_memory=False, seed=None):
    """从验证集固定抽取一批样本用于可视化（本次训练全程不变）。"""
    from torch.utils.data import DataLoader, Subset
    import time as _time
    n = len(val_dataset)
    if n <= 0:
        raise ValueError("验证集为空，无法抽取可视化样本")
    k = min(max(int(num_samples), 1), n)
    if seed is None:
        seed = _time.time_ns() % (2 ** 63 - 1)
    seed = int(seed)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k].tolist()
    loader = DataLoader(Subset(val_dataset, idx), batch_size=k, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
    return loader, idx, seed


@torch.no_grad()
def visualize_single_output(model, vis_loader, device, output_folder,
                            current_epoch, pred_map_fn, model_title='Model',
                            num_samples=2):
    """对单输出去高反光模型（DC-UNet / HDRNet）作验证样本可视化。

    这些模型只输出“去反光后的漫反射图”，没有镜面/梯度/高光概率头，因此
    可视化只保留一行四张图，与模型实际输出严格对应，不再派生 specular / 梯度：
      Input | GT Diffuse | Pred Diffuse | |Pred-GT|

    参数:
        pred_map_fn : 把模型原始输出映射到 [0,1] 的函数（tanh / sigmoid 各自不同）
    """
    model.eval()
    vis_dir = os.path.join(output_folder, 'val_visualization')
    os.makedirs(vis_dir, exist_ok=True)

    batch = next(iter(vis_loader))
    # 兼容 2 元组 (inp, tgt) 与 4 元组 (inp, tgt, gt_spec, gt_grad)
    inputs = batch[0].to(device)
    targets = batch[1].to(device)

    raw = model(inputs)
    preds = pred_map_fn(raw).clamp(0.0, 1.0)

    actual = min(num_samples, inputs.size(0))
    for s in range(actual):
        inp = inputs[s, 0].detach().cpu().numpy()
        gt = targets[s, 0].detach().cpu().numpy()
        pr = preds[s, 0].detach().cpu().numpy()
        diff = np.abs(pr - gt)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(inp, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Input (Highly Reflective)', fontsize=11)
        axes[0].axis('off')
        axes[1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('GT Diffuse', fontsize=11)
        axes[1].axis('off')
        axes[2].imshow(pr, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Pred Diffuse', fontsize=11)
        axes[2].axis('off')
        im = axes[3].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[3].set_title('|Pred - GT| Difference', fontsize=11)
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], fraction=0.046)

        mae = float(np.mean(diff))
        mse = float(np.mean(diff ** 2))
        psnr = compute_psnr(mse)
        max_err = float(np.max(diff))
        fig.suptitle(
            f'{model_title}  |  Epoch {current_epoch} - Sample {s + 1}  |  '
            f'MAE={mae:.4f}  |  MSE={mse:.6f}  |  PSNR={psnr:.2f}dB  |  MaxErr={max_err:.4f}',
            fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(vis_dir, f'epoch_{current_epoch:03d}_sample{s + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    model.train()
    return vis_dir


# ════════════════════════════════════════════════════════════════════════
# 训练历史曲线（5 行：Loss / SSIM / MAE / MSE / PSNR）
# ════════════════════════════════════════════════════════════════════════

def plot_training_history(history, output_folder, total_epochs, model_title='Model'):
    """绘制 Loss / SSIM / MAE / MSE / PSNR 五行训练曲线，保存 training_history.png。"""
    epochs_range = range(1, len(history['train_losses']) + 1)
    pairs = [
        ('train_losses', 'val_losses', 'Loss',  'purple', None),
        ('train_ssims',  'val_ssims',  'SSIM',  'green',   (0, 1)),
        ('train_maes',   'val_maes',   'MAE',   'orange',  None),
        ('train_mses',   'val_mses',   'MSE',   'teal',    None),
        ('train_psnrs',  'val_psnrs',  'PSNR (dB)', 'brown', None),
    ]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(16, 5 * len(pairs)))

    for row, (tk, vk, label, color, ylim) in enumerate(pairs):
        t = history.get(tk, [])
        v = history.get(vk, [])
        if not t and not v:
            continue
        axes[row, 0].plot(epochs_range, t, 'b-o', label=f'Train {label}', linewidth=2, markersize=3)
        axes[row, 0].plot(epochs_range, v, 'r-s', label=f'Val {label}', linewidth=2, markersize=3)
        axes[row, 0].set_xlabel('Epoch', fontsize=11)
        axes[row, 0].set_ylabel(label, fontsize=11)
        axes[row, 0].set_title(f'Train vs Val {label}', fontsize=13, fontweight='bold')
        axes[row, 0].legend(fontsize=10)
        axes[row, 0].grid(True, alpha=0.3)
        if ylim:
            axes[row, 0].set_ylim(list(ylim))

        axes[row, 1].plot(epochs_range, t, 'b-', label=f'Train {label}', linewidth=2, alpha=0.8)
        axes[row, 1].plot(epochs_range, v, 'r-', label=f'Val {label}', linewidth=2, alpha=0.8)
        try:
            axes[row, 1].fill_between(epochs_range, t, v, alpha=0.15, color=color)
        except Exception:
            pass
        axes[row, 1].set_xlabel('Epoch', fontsize=11)
        axes[row, 1].set_ylabel(label, fontsize=11)
        axes[row, 1].set_title(f'{label} Gap (filled)', fontsize=13, fontweight='bold')
        axes[row, 1].legend(fontsize=10)
        axes[row, 1].grid(True, alpha=0.3)
        if ylim:
            axes[row, 1].set_ylim(list(ylim))

    fig.suptitle(f'{model_title}  Training History  ({total_epochs} epochs)',
                 fontsize=16, fontweight='bold', y=1.005)
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_training_curves_realtime(history, output_folder, current_epoch):
    """轻量实时曲线（Loss + SSIM）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-o', label='训练损失', linewidth=2, markersize=3)
    ax1.plot(epochs, history['val_losses'], 'r-s', label='验证损失', linewidth=2, markersize=3)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('损失值 (Loss)')
    ax1.set_title(f'训练和验证损失 (Epoch {current_epoch})', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history['train_ssims'], 'b-o', label='训练 SSIM', linewidth=2, markersize=3)
    ax2.plot(epochs, history['val_ssims'], 'r-s', label='验证 SSIM', linewidth=2, markersize=3)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('SSIM')
    ax2.set_title(f'训练和验证 SSIM (Epoch {current_epoch})', fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim([0, 1])
    plt.tight_layout()
    save_path = os.path.join(output_folder, 'training_curves_realtime.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ════════════════════════════════════════════════════════════════════════
# 训练总结文件（含 PSNR）
# ════════════════════════════════════════════════════════════════════════

def save_training_summary(output_folder, model_title, cfg, history,
                          best_info, total_time_str):
    """生成 training_summary.txt（含 PSNR，口径与主模型对齐）。

    best_info: dict 含 epoch/loss/ssim/mae/mse/psnr/path
    history  : 含 train_*/val_* 列表，新增 *_psnrs
    """
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, 'training_summary.txt')

    def last(key):
        v = history.get(key, [])
        return v[-1] if v else float('nan')

    sep = '=' * 64 + '\n'
    sep2 = '-' * 64 + '\n'
    img_sz = cfg.get('IMAGE_RESIZE')
    if img_sz is None:
        sz_str = 'Original (no resize)'
    elif isinstance(img_sz, (int, float)):
        sz_str = f'{int(img_sz)} x {int(img_sz)}'
    else:
        sz_str = f'{img_sz[0]} x {img_sz[1]} (H x W)'

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(sep)
        f.write(f'Training Summary  ({model_title})\n')
        f.write(sep)
        f.write(f'  Generated at      : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'  Total Train Time  : {total_time_str}\n\n')
        f.write('  [Hyperparameters]\n')
        f.write(f'  Total Epochs      : {cfg.get("EPOCHS")}\n')
        f.write(f'  Batch Size        : {cfg.get("BATCH_SIZE")}\n')
        f.write(f'  Learning Rate     : {cfg.get("LR")}\n')
        f.write(f'  Image Resize      : {sz_str}\n')
        f.write(f'  LR Scheduler      : {cfg.get("SCHEDULER")}\n')
        f.write(f'  Mixed Precision   : {"Enabled (AMP)" if cfg.get("USE_AMP") else "Disabled"}\n')
        f.write(f'  Grad Clip         : {cfg.get("GRAD_CLIP")}\n')
        f.write(f'  Vis Every N Epoch : {cfg.get("VIS_EVERY_N_EPOCHS")}\n')
        f.write(f'  Vis Num Samples   : {cfg.get("VIS_NUM_SAMPLES")}\n\n')

        def _fmt(val, spec, suffix=''):
            """安全格式化：None / NaN 显示为 N/A，否则按 spec 格式并附加 suffix。"""
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return 'N/A'
            return format(val, spec) + suffix

        f.write(sep)
        f.write('Best Model Information  (lowest Val Loss)\n')
        f.write(sep2)
        f.write(f'  Best Epoch        : {best_info["epoch"]}\n')
        f.write(f'  Train Loss        : {_fmt(best_info.get("train_loss"), ".6f")}\n')
        f.write(f'  Val   Loss        : {_fmt(best_info.get("loss"), ".6f")}\n')
        f.write(f'  Train SSIM        : {_fmt(best_info.get("train_ssim"), ".4f")}\n')
        f.write(f'  Val   SSIM        : {_fmt(best_info.get("ssim"), ".4f")}\n')
        f.write(f'  Train MAE         : {_fmt(best_info.get("train_mae"), ".6f")}\n')
        f.write(f'  Val   MAE         : {_fmt(best_info.get("mae"), ".6f")}\n')
        f.write(f'  Train MSE         : {_fmt(best_info.get("train_mse"), ".6f")}\n')
        f.write(f'  Val   MSE         : {_fmt(best_info.get("mse"), ".6f")}\n')
        f.write(f'  Train PSNR        : {_fmt(best_info.get("train_psnr"), ".2f", " dB")}\n')
        f.write(f'  Val   PSNR        : {_fmt(best_info.get("psnr"), ".2f", " dB")}\n')
        f.write(f'  Saved Path        : {best_info["path"]}\n\n')

        f.write(sep)
        f.write('Final Epoch Result\n')
        f.write(sep)
        f.write(f'  Epoch             : {len(history["train_losses"])}\n')
        f.write(f'  Train Loss        : {last("train_losses"):.6f}\n')
        f.write(f'  Val   Loss        : {last("val_losses"):.6f}\n')
        f.write(f'  Train SSIM        : {last("train_ssims"):.4f}\n')
        f.write(f'  Val   SSIM        : {last("val_ssims"):.4f}\n')
        f.write(f'  Train MAE         : {last("train_maes"):.6f}\n')
        f.write(f'  Val   MAE         : {last("val_maes"):.6f}\n')
        f.write(f'  Train MSE         : {last("train_mses"):.6f}\n')
        f.write(f'  Val   MSE         : {last("val_mses"):.6f}\n')
        f.write(f'  Train PSNR        : {last("train_psnrs"):.2f} dB\n')
        f.write(f'  Val   PSNR        : {last("val_psnrs"):.2f} dB\n\n')

        f.write(sep)
        f.write('Output Files\n')
        f.write(sep)
        f.write(f'  Best Model        : {best_info["path"]}\n')
        f.write(f'  Training History  : {os.path.join(output_folder, "training_history.png")}\n')
        f.write(f'  Val Visualization : {os.path.join(output_folder, "val_visualization/")}\n')
        f.write(f'  Training Summary  : {save_path}\n')
        f.write('\n' + sep)

    return save_path
