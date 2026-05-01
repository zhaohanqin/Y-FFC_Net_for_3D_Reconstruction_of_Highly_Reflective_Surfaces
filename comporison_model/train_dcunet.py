"""
train_dcunet.py
===============
DC-UNet 高反光去除模型训练脚本

数据读取方式与 train.py 保持一致（FringeDataset + DataLoader）。
DC_Unet 只输出去反光后的图像（无镜面分量输出），
因此训练损失仅针对预测图像与目标漫反射图像之间的差异计算。

评估指标：Loss / MAE / MSE / SSIM

训练结束后自动生成（参考 train.py 实现方式）：
    training_history.png   -- 训练曲线（Loss / SSIM / MAE / MSE 各 2 张子图，共 4×2）
    training_summary.txt   -- 训练总结（配置 / 最优 Epoch / 最终指标）
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
from PIL import Image

# ---------- 从项目模块导入 ----------
from DCUNet import DC_Unet                              # 模型
from Y_FFC_unified import FringeDataset, SSIM                  # 数据集 + SSIM 损失


# ==============================================================================
# 损失函数
# ==============================================================================

def compute_loss(pred, target, ssim_fn, mae_fn):
    """
    计算单次前向的全部损失与指标。

    参数:
        pred   : 模型输出，tanh 激活，值域 [-1, 1]，shape (B, 1, H, W)
        target : 漫反射真值，值域 [0, 1]，         shape (B, 1, H, W)
        ssim_fn: SSIM 损失模块（越大越好 → 1 - SSIM 作为损失项）
        mae_fn : nn.L1Loss 实例

    返回:
        total_loss : 加权总损失（用于反向传播）
        ssim_val   : 漫反射 SSIM 值（float，统计用）
        mae_val    : 漫反射 MAE 值（float，统计用）
        mse_val    : 漫反射 MSE 值（float，统计用）

    注意:
        DC_Unet 使用 tanh 激活，输出范围 [-1, 1]。
        目标图像值域为 [0, 1]。
        在计算指标/损失前将预测值线性映射到 [0, 1]：
            pred_01 = (pred + 1) / 2
        此映射不改变梯度信息，仅使两者量纲对齐。
    """
    # 将 tanh 输出映射到 [0, 1]，与目标值域对齐
    pred_01 = ((pred + 1.0) / 2.0).clamp(0.0, 1.0)

    # SSIM 损失（1 - SSIM，越小越好）
    ssim_score = ssim_fn(pred_01, target).clamp(0.0, 1.0)
    loss_ssim  = 1.0 - ssim_score

    # MAE 损失
    loss_mae = mae_fn(pred_01, target)              # scalar tensor

    # 总损失 = SSIM 损失 + MAE 损失（等权，简洁为主）
    total_loss = loss_ssim + loss_mae

    # ---------- 统计指标（从计算图中分离，不触发 UserWarning） ----------
    ssim_val = ssim_score.detach().item()
    mae_val  = loss_mae.detach().item()
    mse_val  = torch.mean((pred_01 - target) ** 2).detach().item()

    return total_loss, ssim_val, mae_val, mse_val


# ==============================================================================
# 训练 + 验证单轮
# ==============================================================================

def train_one_epoch(model, loader, optimizer, ssim_fn, mae_fn, device, use_amp, scaler,
                    grad_clip=1.0):
    """执行一个 epoch 的训练，返回各指标均值。"""
    model.train()

    total_loss = total_ssim = total_mae = total_mse = 0.0

    for inputs, gt_diffuse, _gt_specular, _gt_gradient in tqdm(loader, leave=False, desc='Train'):
        # FringeDataset 返回 4 项，DCUNet 只用 inputs 和 gt_diffuse
        inputs     = inputs.to(device)
        gt_diffuse = gt_diffuse.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                pred = model(inputs)
                loss, ssim_val, mae_val, mse_val = compute_loss(
                    pred, gt_diffuse, ssim_fn, mae_fn)
        else:
            pred = model(inputs)
            loss, ssim_val, mae_val, mse_val = compute_loss(
                pred, gt_diffuse, ssim_fn, mae_fn)

        # NaN 保护
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.detach().item()
        total_ssim += ssim_val
        total_mae  += mae_val
        total_mse  += mse_val

    n = len(loader)
    return total_loss / n, total_ssim / n, total_mae / n, total_mse / n


@torch.no_grad()
def validate_one_epoch(model, loader, ssim_fn, mae_fn, device, use_amp, scaler):
    """执行一个 epoch 的验证，返回各指标均值。"""
    model.eval()

    total_loss = total_ssim = total_mae = total_mse = 0.0
    valid_batches = 0

    for inputs, gt_diffuse, _gt_specular, _gt_gradient in tqdm(loader, leave=False, desc='Val  '):
        inputs     = inputs.to(device)
        gt_diffuse = gt_diffuse.to(device)

        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                pred = model(inputs)
                loss, ssim_val, mae_val, mse_val = compute_loss(
                    pred, gt_diffuse, ssim_fn, mae_fn)
        else:
            pred = model(inputs)
            loss, ssim_val, mae_val, mse_val = compute_loss(
                pred, gt_diffuse, ssim_fn, mae_fn)

        if not torch.isfinite(loss):
            continue

        total_loss += loss.detach().item()
        total_ssim += ssim_val
        total_mae  += mae_val
        total_mse  += mse_val
        valid_batches += 1

    if valid_batches == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')

    return (total_loss / valid_batches,
            total_ssim / valid_batches,
            total_mae  / valid_batches,
            total_mse  / valid_batches)


# ==============================================================================
# 【参考 train.py】plot_training_history
# 训练结束后调用，绘制完整的训练历史曲线（Loss / SSIM / MAE / MSE）。
# 布局：4 行 × 2 列子图；左列为折线图，右列为填充面积对比图。
# 保存为 output_folder/training_history.png。
# ==============================================================================

def plot_training_history(train_losses, val_losses,
                          train_ssims,  val_ssims,
                          train_maes,   val_maes,
                          train_mses,   val_mses,
                          output_folder, total_epochs):
    """
    绘制完整训练历史曲线，包含 Loss、SSIM、MAE、MSE 四组指标，共 8 条曲线。

    布局（4×2 子图）：
        左列（折线图）  ：Loss / SSIM / MAE / MSE 各一张
        右列（填充面积）：对应 Train vs Val 差距可视化

    参数:
        train_losses / val_losses : 每 epoch 平均 Loss 列表
        train_ssims  / val_ssims  : 每 epoch 平均 SSIM 列表
        train_maes   / val_maes   : 每 epoch 平均 MAE 列表
        train_mses   / val_mses   : 每 epoch 平均 MSE 列表
        output_folder             : 图像保存目录
        total_epochs              : 实际完成的训练轮数（用于标题）

    返回:
        save_path: 保存路径字符串
    """
    epochs_range = range(1, len(train_losses) + 1)

    # 4 行 × 2 列子图：Left = 折线，Right = 填充对比
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))

    # ────────── 第一行：Loss ──────────
    # 左：折线图
    axes[0, 0].plot(epochs_range, train_losses, 'b-o', label='Train Loss',
                    linewidth=2, markersize=3)
    axes[0, 0].plot(epochs_range, val_losses,   'r-s', label='Val Loss',
                    linewidth=2, markersize=3)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss',  fontsize=11)
    axes[0, 0].set_title('Train vs Val Loss', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 右：填充对比图
    axes[0, 1].plot(epochs_range, train_losses, 'b-', label='Train Loss',
                    linewidth=2, alpha=0.8)
    axes[0, 1].plot(epochs_range, val_losses,   'r-', label='Val Loss',
                    linewidth=2, alpha=0.8)
    axes[0, 1].fill_between(epochs_range, train_losses, val_losses,
                             alpha=0.15, color='purple')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss',  fontsize=11)
    axes[0, 1].set_title('Loss Gap (filled)', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # ────────── 第二行：SSIM ──────────
    # 左：折线图
    axes[1, 0].plot(epochs_range, train_ssims, 'b-o', label='Train SSIM',
                    linewidth=2, markersize=3)
    axes[1, 0].plot(epochs_range, val_ssims,   'r-s', label='Val SSIM',
                    linewidth=2, markersize=3)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('SSIM',  fontsize=11)
    axes[1, 0].set_title('Train vs Val SSIM', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # 右：填充对比图
    axes[1, 1].plot(epochs_range, train_ssims, 'b-', label='Train SSIM',
                    linewidth=2, alpha=0.8)
    axes[1, 1].plot(epochs_range, val_ssims,   'r-', label='Val SSIM',
                    linewidth=2, alpha=0.8)
    axes[1, 1].fill_between(epochs_range, train_ssims, val_ssims,
                             alpha=0.15, color='green')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('SSIM',  fontsize=11)
    axes[1, 1].set_title('SSIM Gap (filled)', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])

    # ────────── 第三行：MAE ──────────
    # 左：折线图
    axes[2, 0].plot(epochs_range, train_maes, 'b-o', label='Train MAE',
                    linewidth=2, markersize=3)
    axes[2, 0].plot(epochs_range, val_maes,   'r-s', label='Val MAE',
                    linewidth=2, markersize=3)
    axes[2, 0].set_xlabel('Epoch', fontsize=11)
    axes[2, 0].set_ylabel('MAE',   fontsize=11)
    axes[2, 0].set_title('Train vs Val MAE', fontsize=13, fontweight='bold')
    axes[2, 0].legend(fontsize=10)
    axes[2, 0].grid(True, alpha=0.3)

    # 右：填充对比图
    axes[2, 1].plot(epochs_range, train_maes, 'b-', label='Train MAE',
                    linewidth=2, alpha=0.8)
    axes[2, 1].plot(epochs_range, val_maes,   'r-', label='Val MAE',
                    linewidth=2, alpha=0.8)
    axes[2, 1].fill_between(epochs_range, train_maes, val_maes,
                             alpha=0.15, color='orange')
    axes[2, 1].set_xlabel('Epoch', fontsize=11)
    axes[2, 1].set_ylabel('MAE',   fontsize=11)
    axes[2, 1].set_title('MAE Gap (filled)', fontsize=13, fontweight='bold')
    axes[2, 1].legend(fontsize=10)
    axes[2, 1].grid(True, alpha=0.3)

    # ────────── 第四行：MSE ──────────
    # 左：折线图
    axes[3, 0].plot(epochs_range, train_mses, 'b-o', label='Train MSE',
                    linewidth=2, markersize=3)
    axes[3, 0].plot(epochs_range, val_mses,   'r-s', label='Val MSE',
                    linewidth=2, markersize=3)
    axes[3, 0].set_xlabel('Epoch', fontsize=11)
    axes[3, 0].set_ylabel('MSE',   fontsize=11)
    axes[3, 0].set_title('Train vs Val MSE', fontsize=13, fontweight='bold')
    axes[3, 0].legend(fontsize=10)
    axes[3, 0].grid(True, alpha=0.3)

    # 右：填充对比图
    axes[3, 1].plot(epochs_range, train_mses, 'b-', label='Train MSE',
                    linewidth=2, alpha=0.8)
    axes[3, 1].plot(epochs_range, val_mses,   'r-', label='Val MSE',
                    linewidth=2, alpha=0.8)
    axes[3, 1].fill_between(epochs_range, train_mses, val_mses,
                             alpha=0.15, color='teal')
    axes[3, 1].set_xlabel('Epoch', fontsize=11)
    axes[3, 1].set_ylabel('MSE',   fontsize=11)
    axes[3, 1].set_title('MSE Gap (filled)', fontsize=13, fontweight='bold')
    axes[3, 1].legend(fontsize=10)
    axes[3, 1].grid(True, alpha=0.3)

    # 总标题 & 保存
    fig.suptitle(f'DC-UNet  Training History  ({total_epochs} epochs)',
                 fontsize=16, fontweight='bold', y=1.005)
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


# ==============================================================================
# 【参考 train.py】save_training_summary
# 训练结束后调用，将训练配置、最优模型信息、最终 epoch 指标写入
# output_folder/training_summary.txt。
# ==============================================================================

def save_training_summary(output_folder,
                          # ── 训练基本信息 ──
                          total_epochs, batch_size, learning_rate,
                          image_resize, scheduler_type, use_amp,
                          # ── 最优模型信息 ──
                          best_epoch, best_val_loss, best_val_mae,
                          best_val_mse, best_val_ssim, model_save_path,
                          # ── 每 epoch 历史（用于最终 epoch 报告） ──
                          train_losses, val_losses,
                          train_ssims,  val_ssims,
                          train_maes,   val_maes,
                          train_mses,   val_mses,
                          # ── 总训练时长 ──
                          total_time_str):
    """
    生成并保存训练总结文件 training_summary.txt。

    文件结构（与 train.py 保持一致）：
        ① Training Summary（基本信息）
        ② Best Model Information（最优模型 epoch / Loss / SSIM / MAE / MSE）
        ③ Final Epoch Result（最后一个 epoch 的全部指标）
        ④ Output Files（生成文件一览）

    参数:
        output_folder    : 文件保存目录（会自动创建）
        total_epochs     : 实际完成的训练轮数
        batch_size       : 训练批次大小
        learning_rate    : 初始学习率
        image_resize     : 图像尺寸设置（int / tuple / None）
        scheduler_type   : 学习率调度器类型字符串
        use_amp          : 是否启用混合精度训练
        best_epoch       : 验证集 Loss 最低时所处的 epoch
        best_val_loss    : 最优验证集 Loss
        best_val_mae     : 对应 MAE
        best_val_mse     : 对应 MSE
        best_val_ssim    : 对应 SSIM
        model_save_path  : 最优模型保存路径
        train_losses / val_losses : 每 epoch 平均 Loss 列表
        train_ssims  / val_ssims  : 每 epoch 平均 SSIM 列表
        train_maes   / val_maes   : 每 epoch 平均 MAE  列表
        train_mses   / val_mses   : 每 epoch 平均 MSE  列表
        total_time_str   : 格式化总训练时长字符串

    返回:
        save_path: 保存路径字符串
    """
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, 'training_summary.txt')

    # 图像尺寸格式化
    if image_resize is None:
        resize_str = 'Original (no resize)'
    elif isinstance(image_resize, int):
        resize_str = f'{image_resize} x {image_resize}'
    else:
        resize_str = f'{image_resize[0]} x {image_resize[1]}'

    # 最终 epoch 各项指标
    final_tr_loss = train_losses[-1] if train_losses else float('nan')
    final_vl_loss = val_losses[-1]   if val_losses   else float('nan')
    final_tr_ssim = train_ssims[-1]  if train_ssims  else float('nan')
    final_vl_ssim = val_ssims[-1]    if val_ssims    else float('nan')
    final_tr_mae  = train_maes[-1]   if train_maes   else float('nan')
    final_vl_mae  = val_maes[-1]     if val_maes     else float('nan')
    final_tr_mse  = train_mses[-1]   if train_mses   else float('nan')
    final_vl_mse  = val_mses[-1]     if val_mses     else float('nan')

    with open(save_path, 'w', encoding='utf-8') as f:
        sep  = '=' * 64 + '\n'
        sep2 = '-' * 64 + '\n'

        # ──────────────────────────────────────────────────────
        # ① Training Summary（基本信息）
        # ──────────────────────────────────────────────────────
        f.write(sep)
        f.write('Training Summary\n')
        f.write(sep)
        f.write(f'  Generated at      : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'  Total Train Time  : {total_time_str}\n')
        f.write('\n')
        f.write('  [Hyperparameters]\n')
        f.write(f'  Total Epochs      : {total_epochs}\n')
        f.write(f'  Batch Size        : {batch_size}\n')
        f.write(f'  Learning Rate     : {learning_rate}\n')
        f.write(f'  Image Resize      : {resize_str}\n')
        f.write(f'  LR Scheduler      : {scheduler_type}\n')
        f.write(f'  Mixed Precision   : {"Enabled (AMP)" if use_amp else "Disabled"}\n')
        f.write('\n')

        # ──────────────────────────────────────────────────────
        # ② Best Model Information
        # ──────────────────────────────────────────────────────
        f.write(sep)
        f.write('Best Model Information  (lowest Val Loss)\n')
        f.write(sep2)
        f.write(f'  Best Epoch        : {best_epoch}\n')
        f.write(f'  Best Val Loss     : {best_val_loss:.6f}\n')
        f.write(f'  Best Val SSIM     : {best_val_ssim:.4f}\n')
        f.write(f'  Best Val MAE      : {best_val_mae:.6f}\n')
        f.write(f'  Best Val MSE      : {best_val_mse:.6f}\n')
        f.write(f'  Saved Path        : {model_save_path}\n')
        f.write('\n')

        # ──────────────────────────────────────────────────────
        # ③ Final Epoch Result（最后一个 epoch 的全部指标）
        # ──────────────────────────────────────────────────────
        f.write(sep)
        f.write('Final Epoch Result\n')
        f.write(sep)
        f.write(f'  Epoch             : {total_epochs}\n')
        f.write(f'  Train Loss        : {final_tr_loss:.6f}\n')
        f.write(f'  Val   Loss        : {final_vl_loss:.6f}\n')
        f.write(f'  Train SSIM        : {final_tr_ssim:.4f}\n')
        f.write(f'  Val   SSIM        : {final_vl_ssim:.4f}\n')
        f.write(f'  Train MAE         : {final_tr_mae:.6f}\n')
        f.write(f'  Val   MAE         : {final_vl_mae:.6f}\n')
        f.write(f'  Train MSE         : {final_tr_mse:.6f}\n')
        f.write(f'  Val   MSE         : {final_vl_mse:.6f}\n')
        f.write('\n')

        # ──────────────────────────────────────────────────────
        # ④ Output Files（文件一览）
        # ──────────────────────────────────────────────────────
        f.write(sep)
        f.write('Output Files\n')
        f.write(sep)
        f.write(f'  Best Model        : {model_save_path}\n')
        f.write(f'  Training History  : {os.path.join(output_folder, "training_history.png")}\n')
        f.write(f'  Training Summary  : {save_path}\n')
        f.write('\n' + sep)

    return save_path


# ==============================================================================
# 主训练函数
# ==============================================================================

def train(
    # ── 数据 ──────────────────────────────
    input_dir,
    target_dir,
    image_resize   = (160, 128),
    train_split    = 0.8,
    batch_size     = 8,
    num_workers    = 0,
    # ── 模型 ──────────────────────────────
    in_channels    = 1,
    out_channels   = 1,
    nf             = 32,
    use_dropout    = False,
    # ── 训练超参数 ─────────────────────────
    epochs         = 100,
    lr             = 1e-3,
    use_amp        = True,
    grad_clip      = 1.0,
    # ── 学习率调度 ─────────────────────────
    scheduler_type = 'cosine',   # 'step' | 'cosine' | 'plateau' | None
    step_size      = 30,
    gamma          = 0.5,
    # ── 保存路径 ──────────────────────────
    model_save_path = 'best_model.pth',
    output_folder   = 'experiments/dcunet',
):
    """
    完整训练流程。

    数据集目录结构（与 train.py 一致）：
        input_dir/   input_0001.png, input_0002.png, ...
        target_dir/  target_0001.png, target_0002.png, ...

    训练结束后自动生成：
        output_folder/training_history.png   -- 训练曲线（Loss/SSIM/MAE/MSE）
        output_folder/training_summary.txt   -- 训练总结（配置/最优/最终指标）
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 设备
    # ──────────────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────────
    # 数据集（与 train.py 保持完全一致的加载方式）
    # ──────────────────────────────────────────────────────────────────────────
    dataset = FringeDataset(input_dir, target_dir, resize=image_resize)
    train_size = int(train_split * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    print(f"数据集: 训练 {train_size} / 验证 {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(
        val_dataset,   batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    # ──────────────────────────────────────────────────────────────────────────
    # 模型
    # ──────────────────────────────────────────────────────────────────────────
    model = DC_Unet(
        in_channels  = in_channels,
        out_channels = out_channels,
        nf           = nf,
        use_dropout  = use_dropout,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # ──────────────────────────────────────────────────────────────────────────
    # 损失函数
    # ──────────────────────────────────────────────────────────────────────────
    ssim_fn = SSIM(window_size=11, sigma=1.5).to(device)
    mae_fn  = nn.L1Loss()

    # ──────────────────────────────────────────────────────────────────────────
    # 优化器 + 学习率调度器
    # ──────────────────────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=gamma, patience=10)
    else:
        scheduler = None

    # 混合精度 scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    # ──────────────────────────────────────────────────────────────────────────
    # 历史记录列表（每 epoch 追加，用于绘制训练曲线和写入 summary）
    # ──────────────────────────────────────────────────────────────────────────
    history = {
        'train_losses': [], 'val_losses': [],
        'train_ssims':  [], 'val_ssims':  [],
        'train_maes':   [], 'val_maes':   [],
        'train_mses':   [], 'val_mses':   [],
    }

    # ── 最优模型跟踪 ──
    best_val_loss  = float('inf')
    best_epoch     = 0
    best_val_mae   = float('nan')
    best_val_mse   = float('nan')
    best_val_ssim  = float('nan')
    epoch_times    = []

    print(f"\n{'='*70}")
    print(f"开始训练  epochs={epochs}  lr={lr}  batch={batch_size}  amp={use_amp}")
    print(f"{'='*70}\n")
    start_total = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ---------- 训练 ----------
        tr_loss, tr_ssim, tr_mae, tr_mse = train_one_epoch(
            model, train_loader, optimizer,
            ssim_fn, mae_fn, device, use_amp, scaler,
            grad_clip=grad_clip)

        # ---------- 验证 ----------
        vl_loss, vl_ssim, vl_mae, vl_mse = validate_one_epoch(
            model, val_loader,
            ssim_fn, mae_fn, device, use_amp, scaler)

        # ---------- 学习率调度 ----------
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(vl_loss)
            else:
                scheduler.step()

        # ---------- 记录历史（用于绘图与 summary） ----------
        history['train_losses'].append(tr_loss)
        history['val_losses'].append(vl_loss)
        history['train_ssims'].append(tr_ssim)
        history['val_ssims'].append(vl_ssim)
        history['train_maes'].append(tr_mae)
        history['val_maes'].append(vl_mae)
        history['train_mses'].append(tr_mse)
        history['val_mses'].append(vl_mse)

        # ---------- 统计时间 ----------
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_time       = np.mean(epoch_times)
        remaining      = int(avg_time * (epochs - epoch))
        eta_str        = str(timedelta(seconds=remaining))
        current_lr     = optimizer.param_groups[0]['lr']

        # ---------- 打印 ----------
        print(f"Epoch [{epoch:>3d}/{epochs}]  "
              f"耗时 {epoch_time:.1f}s  剩余 {eta_str}  lr={current_lr:.2e}")
        print(f"  Train | Loss: {tr_loss:.6f}  MAE: {tr_mae:.6f}  "
              f"MSE: {tr_mse:.6f}  SSIM: {tr_ssim:.4f}")
        print(f"  Val   | Loss: {vl_loss:.6f}  MAE: {vl_mae:.6f}  "
              f"MSE: {vl_mse:.6f}  SSIM: {vl_ssim:.4f}")

        # ---------- 保存最优模型 ----------
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_epoch    = epoch
            best_val_mae  = vl_mae
            best_val_mse  = vl_mse
            best_val_ssim = vl_ssim
            torch.save({
                'epoch': epoch,
                'model_name': 'dcunet',
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'image_resize': image_resize,
                'val_loss': vl_loss,
                'val_ssim': vl_ssim,
                'val_mae': vl_mae,
                'val_mse': vl_mse,
                'model_config': {
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'nf': nf,
                    'use_dropout': use_dropout,
                },
            }, model_save_path)
            print(f"  ✅ 最优模型已保存 → {model_save_path}  "
                  f"(Val Loss↓ {best_val_loss:.6f})")
        else:
            print(f"     无改善（当前最优 Epoch {best_epoch}, "
                  f"Loss {best_val_loss:.6f}）")

        print()

    # ──────────────────────────────────────────────────────────────────────────
    # 训练结束汇总
    # ──────────────────────────────────────────────────────────────────────────
    total_time = str(timedelta(seconds=int(time.time() - start_total)))
    print(f"{'='*70}")
    print(f"训练完成！总时长: {total_time}")
    print(f"最佳验证结果:")
    print(f"  Epoch : {best_epoch}")
    print(f"  Loss  : {best_val_loss:.6f}")
    print(f"  MAE   : {best_val_mae:.6f}")
    print(f"  MSE   : {best_val_mse:.6f}")
    print(f"  SSIM  : {best_val_ssim:.4f}")
    print(f"  Path  : {model_save_path}")
    print(f"{'='*70}")

    # ──────────────────────────────────────────────────────────────────────────
    # 【参考 train.py】生成训练曲线图 training_history.png
    # 包含 Loss / SSIM / MAE / MSE 四组指标，共 4×2 = 8 张子图。
    # 保证即使前面步骤抛出异常也能在最终阶段独立调用此函数重新生成。
    # ──────────────────────────────────────────────────────────────────────────
    print("\n保存训练历史曲线（Loss / SSIM / MAE / MSE）...")
    history_path = plot_training_history(
        train_losses  = history['train_losses'],
        val_losses    = history['val_losses'],
        train_ssims   = history['train_ssims'],
        val_ssims     = history['val_ssims'],
        train_maes    = history['train_maes'],
        val_maes      = history['val_maes'],
        train_mses    = history['train_mses'],
        val_mses      = history['val_mses'],
        output_folder = output_folder,
        total_epochs  = epochs,
    )
    print(f"训练历史曲线已保存: {history_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # 【参考 train.py】生成训练总结文件 training_summary.txt
    # 包含训练配置、最优 Epoch 信息（含 epoch 编号）、最终 Epoch 全部指标。
    # ──────────────────────────────────────────────────────────────────────────
    print("生成训练总结文件（training_summary.txt）...")
    summary_path = save_training_summary(
        output_folder   = output_folder,
        total_epochs    = epochs,
        batch_size      = batch_size,
        learning_rate   = lr,
        image_resize    = image_resize,
        scheduler_type  = scheduler_type if scheduler_type else 'None',
        use_amp         = use_amp and (device.type == 'cuda'),
        best_epoch      = best_epoch,
        best_val_loss   = best_val_loss,
        best_val_mae    = best_val_mae,
        best_val_mse    = best_val_mse,
        best_val_ssim   = best_val_ssim,
        model_save_path = model_save_path,
        train_losses    = history['train_losses'],
        val_losses      = history['val_losses'],
        train_ssims     = history['train_ssims'],
        val_ssims       = history['val_ssims'],
        train_maes      = history['train_maes'],
        val_maes        = history['val_maes'],
        train_mses      = history['train_mses'],
        val_mses        = history['val_mses'],
        total_time_str  = total_time,
    )
    print(f"训练总结文件已保存: {summary_path}")

    # ── 输出文件结构 ──
    print("\n" + "=" * 60)
    print("✅ 所有流程完成！")
    print("=" * 60)
    print(f"\n📁 所有结果已保存到: {os.path.abspath(output_folder)}")
    print("\n文件结构:")
    print(f"  {output_folder}/")
    print(f"  ├── models/")
    print(f"  │   └── best_model.pth           (验证 Loss 最优模型)")
    print(f"  ├── training_history.png          (训练曲线: Loss/SSIM/MAE/MSE)")
    print(f"  └── training_summary.txt          (训练总结: 配置/最优Epoch/最终指标)")
    print("=" * 60)

    return history


# ==============================================================================
# 虚拟数据生成（测试用，与 train.py 保持一致）
# ==============================================================================

def create_dummy_data(input_dir, target_dir, num_samples=100):
    """生成灰度虚拟图像对，用于快速验证训练流程。"""
    os.makedirs(input_dir,  exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    print(f"生成 {num_samples} 个虚拟样本...")
    for i in range(num_samples):
        h, w   = 128, 160
        base   = np.random.rand(h, w).astype(np.float32)
        spec   = np.clip(np.random.rand(h, w).astype(np.float32) * 0.3, 0, 1)
        inp    = np.clip(base + spec, 0, 1)
        Image.fromarray((inp  * 255).astype(np.uint8)).save(
            os.path.join(input_dir,  f'input_{i:04d}.png'))
        Image.fromarray((base * 255).astype(np.uint8)).save(
            os.path.join(target_dir, f'target_{i:04d}.png'))
    print(f"虚拟数据已保存: input→{input_dir}  target→{target_dir}")


# ==============================================================================
# 参数配置 & 入口
# ==============================================================================

def main():
    print("=" * 70)
    print("DC-UNet 训练启动")
    print("=" * 70)

    # ══════════════════════════════════════════════════
    # 在此处修改训练参数，无需深入代码内部
    # ══════════════════════════════════════════════════

    # ── 数据路径 ──
    INPUT_DIR  = 'data2/inputs'
    TARGET_DIR = 'data2/targets'

    # ── 图像与数据 ──
    IMAGE_RESIZE  = (128, 160)   # (H, W)，None 表示不缩放
    TRAIN_SPLIT   = 0.8
    BATCH_SIZE    = 8
    NUM_WORKERS   = 0

    # ── 模型结构 ──
    IN_CHANNELS   = 1
    OUT_CHANNELS  = 1
    NF            = 32           # 基础特征图通道数
    USE_DROPOUT   = False

    # ── 训练超参数 ──
    EPOCHS        = 100
    LR            = 1e-3
    USE_AMP       = True         # 混合精度（仅 CUDA 有效）
    GRAD_CLIP     = 1.0          # 梯度裁剪阈值（0 表示不裁剪）

    # ── 学习率调度 ──
    SCHEDULER     = 'cosine'     # 'step' | 'cosine' | 'plateau' | None
    STEP_SIZE     = 30           # 仅 'step' 调度器使用
    GAMMA         = 0.5          # StepLR 衰减因子 / Plateau 衰减因子

    # ── 输出路径 ──
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FOLDER   = f'experiments/dcunet_{timestamp}'
    MODEL_SAVE_PATH = os.path.join(OUTPUT_FOLDER, 'models', 'best_model.pth')

    # 打印配置
    print(f"\n当前配置:")
    print(f"  数据路径     : {INPUT_DIR} / {TARGET_DIR}")
    print(f"  图像尺寸     : {IMAGE_RESIZE}")
    print(f"  训练轮数     : {EPOCHS}")
    print(f"  学习率       : {LR}")
    print(f"  批大小       : {BATCH_SIZE}")
    print(f"  混合精度     : {USE_AMP}")
    print(f"  调度器       : {SCHEDULER}")
    print(f"  输出目录     : {os.path.abspath(OUTPUT_FOLDER)}")
    print()

    # 若数据目录不存在，自动生成虚拟数据（测试用）
    if not os.path.exists(INPUT_DIR) or not os.path.exists(TARGET_DIR):
        print("数据目录不存在，自动生成虚拟数据集...")
        create_dummy_data(INPUT_DIR, TARGET_DIR, num_samples=500)

    # 启动训练
    train(
        input_dir        = INPUT_DIR,
        target_dir       = TARGET_DIR,
        image_resize     = IMAGE_RESIZE,
        train_split      = TRAIN_SPLIT,
        batch_size       = BATCH_SIZE,
        num_workers      = NUM_WORKERS,
        in_channels      = IN_CHANNELS,
        out_channels     = OUT_CHANNELS,
        nf               = NF,
        use_dropout      = USE_DROPOUT,
        epochs           = EPOCHS,
        lr               = LR,
        use_amp          = USE_AMP,
        grad_clip        = GRAD_CLIP,
        scheduler_type   = SCHEDULER,
        step_size        = STEP_SIZE,
        gamma            = GAMMA,
        model_save_path  = MODEL_SAVE_PATH,
        output_folder    = OUTPUT_FOLDER,
    )


if __name__ == '__main__':
    # Matplotlib 字体设置（解决中文乱码和负号显示问题，与 train.py 保持一致）
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                   'WenQuanYi Micro Hei', 'Arial Unicode MS']
    rcParams['axes.unicode_minus'] = False
    main()
