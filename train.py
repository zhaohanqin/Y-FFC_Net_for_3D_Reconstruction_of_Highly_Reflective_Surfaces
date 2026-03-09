import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
from torch.utils.data import DataLoader  # 数据加载器
import os  # 文件与路径操作
import numpy as np  # 数值计算
from PIL import Image  # 图像读写
import matplotlib.pyplot as plt  # 可视化
from matplotlib import rcParams  # 字体/渲染设置
from tqdm import tqdm  # 进度条
import time  # 时间跟踪
from datetime import datetime, timedelta  # 日期时间处理
from Y_FFC import YFFC, FringeDataset, SSIM, statistical_loss  # 导入模型、数据集与损失

def _compute_loss(pred_diffuse, pred_specular, pred_gradient,
                  gt_diffuse,  gt_specular,  gt_gradient,
                  ssim_loss, mae_loss,
                  weight_diffuse, weight_specular, weight_gradient):
    """
    [修复 P2/P3/P7/P8]
    统一 loss 计算入口，消除原代码中训练/验证 × AMP/非AMP 共 4 份完全相同的复制代码。
    所有分支统一走此函数，逻辑只需维护一处。

    同时修复：
      P3 - 用已计算的 ld_ssim 推导 SSIM 值（避免重复前向）
      P7 - 镜面分支补充 MAE，与漫反射分支结构对称（SSIM + MAE + stat）
      P8 - 分支项数对齐后，权重含义更清晰

    返回:
        total_loss: 加权总损失（已归一化，便于梯度累积）
        ssim_val:   漫反射 SSIM 值（float，用于精度统计，不再触发额外前向）
        mae_val:    漫反射 MAE 值（float，用于 MAE 指标历史记录）
        loss_components: dict，记录各子损失方便诊断
    """
    # ---------- 漫反射分支：SSIM + MAE + 统计损失 ----------
    ld_ssim = 1.0 - ssim_loss(pred_diffuse, gt_diffuse)
    ld_mae  = mae_loss(pred_diffuse, gt_diffuse)
    ld_sl   = statistical_loss(pred_diffuse, gt_diffuse)
    ld      = ld_ssim + ld_mae + ld_sl

    # ---------- 镜面反射分支：SSIM + MAE + 统计损失 ----------
    # [修复 P7] 原设计缺 MAE，与漫反射分支不对称；
    # 镜面高光区域像素级精度对 MAE 敏感，补充后物理逻辑更完整。
    ls_ssim = 1.0 - ssim_loss(pred_specular, gt_specular)
    ls_mae  = mae_loss(pred_specular, gt_specular)
    ls_sl   = statistical_loss(pred_specular, gt_specular)
    ls      = ls_ssim + ls_mae + ls_sl

    # ---------- 梯度分支：SSIM ----------
    lg_ssim = 1.0 - ssim_loss(pred_gradient, gt_gradient)
    lg      = lg_ssim

    total_loss = weight_diffuse * ld + weight_specular * ls + weight_gradient * lg

    # [修复 P3] ssim_val 直接由 ld_ssim 推导，无需第二次前向：
    #   ld_ssim = 1 - ssim(pred, gt)  =>  ssim = 1 - ld_ssim
    # [修复 UserWarning] .detach() 从计算图安全分离后再转标量，
    # 消除"Converting a tensor with requires_grad=True to a scalar"警告，
    # total_loss 本身不受影响，反向传播正常进行。
    ssim_val = (1.0 - ld_ssim).detach().item()

    # 漫反射 MAE 值直接取 ld_mae，供调用方按 epoch 累积
    mae_val = ld_mae.detach().item()

    # 【新增 MSE】计算漫反射分支的均方误差（Mean Squared Error）。
    # 与 MAE 同样只统计漫反射分支，保持与 SSIM / MAE 指标口径一致。
    # detach() 确保不触发 UserWarning，不影响 total_loss 的梯度流。
    mse_val = torch.mean((pred_diffuse - gt_diffuse) ** 2).detach().item()

    loss_components = {
        'ld':     ld.detach().item(),
        'ld_ssim': ld_ssim.detach().item(),
        'ld_mae':  ld_mae.detach().item(),
        'ld_sl':   ld_sl.detach().item(),
        'ls':      ls.detach().item(),
        'ls_ssim': ls_ssim.detach().item(),
        'ls_mae':  ls_mae.detach().item(),
        'ls_sl':   ls_sl.detach().item(),
        'lg':      lg.detach().item(),
    }
    # 返回值新增第 5 项 mse_val（漫反射 MSE），调用方同步更新解包语句
    return total_loss, ssim_val, mae_val, mse_val, loss_components


def _optimizer_step(optimizer, scaler, use_amp, use_grad_clip, grad_clip_value, model):
    """
    [修复 P4]
    将"梯度裁剪 + optimizer.step + scaler.update + zero_grad"封装为单函数，
    确保正常累积批次与末尾不完整批次走完全相同的逻辑，消除原代码末尾批次
    缺少梯度裁剪的 Bug。
    """
    if use_grad_clip:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)   # 必须在 clip 前 unscale
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

    if use_amp and scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad()


# ==============================================================================
# 【新增】plot_training_history
# 训练结束后调用，绘制完整的训练历史曲线（Loss / SSIM / MAE），
# 并将图像保存为 training_history.png。
# 与训练过程中的实时曲线（training_curves_realtime.png）独立存在，
# 确保即使训练中途出现异常也能在最终阶段独立重绘。
# ==============================================================================
def plot_training_history(train_losses, val_losses,
                          train_ssims,  val_ssims,
                          train_maes,   val_maes,
                          output_folder, total_epochs,
                          # 【新增 MSE】MSE 历史列表，默认为 None 保持向后兼容
                          train_mses=None, val_mses=None):
    """
    绘制完整训练历史曲线，包含 Loss、SSIM、MAE、MSE 四组指标，共 8 条曲线。

    布局说明（4×2 子图）：
        左列（train vs val）：Loss / SSIM / MAE / MSE 独立折线图
        右列（对比填充区域）：Loss / SSIM / MAE / MSE 填充面积图
    【注】MSE 为新增第四行；当 train_mses/val_mses 为 None 时退化为原 3×2 布局。

    参数:
        train_losses  : 每 epoch 平均训练 Loss 列表
        val_losses    : 每 epoch 平均验证 Loss 列表
        train_ssims   : 每 epoch 平均训练 SSIM 列表
        val_ssims     : 每 epoch 平均验证 SSIM 列表
        train_maes    : 每 epoch 平均训练 MAE（漫反射分支）列表
        val_maes      : 每 epoch 平均验证 MAE（漫反射分支）列表
        output_folder : 图像保存目录
        total_epochs  : 实际完成的训练轮数（用于标题）

    返回:
        save_path: 保存路径字符串
    """
    epochs_range = range(1, len(train_losses) + 1)

    # 【新增 MSE】根据是否传入 MSE 数据动态决定行数（兼容旧调用）
    has_mse = (train_mses is not None and val_mses is not None
               and len(train_mses) > 0 and len(val_mses) > 0)
    n_rows = 4 if has_mse else 3

    # 创建 n_rows 行 × 2 列子图：左列折线，右列填充对比
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows))

    # ────────── 第一行：Loss ──────────
    # 左：折线图
    axes[0, 0].plot(epochs_range, train_losses, 'b-o', label='Train Loss',
                    linewidth=2, markersize=3)
    axes[0, 0].plot(epochs_range, val_losses,   'r-s', label='Val Loss',
                    linewidth=2, markersize=3)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Train vs Val Loss', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 右：填充对比图
    axes[0, 1].plot(epochs_range, train_losses, 'b-', label='Train Loss',
                    linewidth=2, alpha=0.8)
    axes[0, 1].plot(epochs_range, val_losses,   'r-', label='Val Loss',
                    linewidth=2, alpha=0.8)
    axes[0, 1].fill_between(epochs_range, train_losses, val_losses, alpha=0.15, color='purple')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
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
    axes[1, 0].set_ylabel('SSIM', fontsize=11)
    axes[1, 0].set_title('Train vs Val SSIM (Diffuse)', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # 右：填充对比图
    axes[1, 1].plot(epochs_range, train_ssims, 'b-', label='Train SSIM',
                    linewidth=2, alpha=0.8)
    axes[1, 1].plot(epochs_range, val_ssims,   'r-', label='Val SSIM',
                    linewidth=2, alpha=0.8)
    axes[1, 1].fill_between(epochs_range, train_ssims, val_ssims, alpha=0.15, color='green')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('SSIM', fontsize=11)
    axes[1, 1].set_title('SSIM Gap (filled)', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])

    # ────────── 第三行：MAE（漫反射分支） ──────────
    # 左：折线图
    axes[2, 0].plot(epochs_range, train_maes, 'b-o', label='Train MAE (Diffuse)',
                    linewidth=2, markersize=3)
    axes[2, 0].plot(epochs_range, val_maes,   'r-s', label='Val MAE (Diffuse)',
                    linewidth=2, markersize=3)
    axes[2, 0].set_xlabel('Epoch', fontsize=11)
    axes[2, 0].set_ylabel('MAE', fontsize=11)
    axes[2, 0].set_title('Train vs Val MAE (Diffuse)', fontsize=13, fontweight='bold')
    axes[2, 0].legend(fontsize=10)
    axes[2, 0].grid(True, alpha=0.3)

    # 右：填充对比图
    axes[2, 1].plot(epochs_range, train_maes, 'b-', label='Train MAE',
                    linewidth=2, alpha=0.8)
    axes[2, 1].plot(epochs_range, val_maes,   'r-', label='Val MAE',
                    linewidth=2, alpha=0.8)
    axes[2, 1].fill_between(epochs_range, train_maes, val_maes, alpha=0.15, color='orange')
    axes[2, 1].set_xlabel('Epoch', fontsize=11)
    axes[2, 1].set_ylabel('MAE', fontsize=11)
    axes[2, 1].set_title('MAE Gap (filled)', fontsize=13, fontweight='bold')
    axes[2, 1].legend(fontsize=10)
    axes[2, 1].grid(True, alpha=0.3)

    fig.suptitle(f'Y-FFC Net  Training History  ({total_epochs} epochs)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    # ────────── 【新增 MSE】第四行：MSE（漫反射分支） ──────────
    if has_mse:
        # 左：折线图
        axes[3, 0].plot(epochs_range, train_mses, 'b-o', label='Train MSE (Diffuse)',
                        linewidth=2, markersize=3)
        axes[3, 0].plot(epochs_range, val_mses,   'r-s', label='Val MSE (Diffuse)',
                        linewidth=2, markersize=3)
        axes[3, 0].set_xlabel('Epoch', fontsize=11)
        axes[3, 0].set_ylabel('MSE', fontsize=11)
        axes[3, 0].set_title('Train vs Val MSE (Diffuse)', fontsize=13, fontweight='bold')
        axes[3, 0].legend(fontsize=10)
        axes[3, 0].grid(True, alpha=0.3)

        # 右：填充对比图
        axes[3, 1].plot(epochs_range, train_mses, 'b-', label='Train MSE',
                        linewidth=2, alpha=0.8)
        axes[3, 1].plot(epochs_range, val_mses,   'r-', label='Val MSE',
                        linewidth=2, alpha=0.8)
        axes[3, 1].fill_between(epochs_range, train_mses, val_mses, alpha=0.15, color='teal')
        axes[3, 1].set_xlabel('Epoch', fontsize=11)
        axes[3, 1].set_ylabel('MSE', fontsize=11)
        axes[3, 1].set_title('MSE Gap (filled)', fontsize=13, fontweight='bold')
        axes[3, 1].legend(fontsize=10)
        axes[3, 1].grid(True, alpha=0.3)

        plt.tight_layout()  # MSE 行添加后重新调整布局

    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


# ==============================================================================
# 【新增】save_training_summary
# 训练结束后调用，将训练配置、最佳模型指标（含 epoch 编号）、
# 最终 epoch 各项指标写入 training_summary.txt。
# 与 train_model 的返回值配合使用，不依赖任何全局变量。
# ==============================================================================
def save_training_summary(output_folder,
                          # ── 训练基本信息 ──
                          total_epochs, batch_size, learning_rate,
                          gradient_accumulation_steps, image_resize,
                          optimizer_type, scheduler_type, use_amp,
                          loss_weight_diffuse, loss_weight_specular, loss_weight_gradient,
                          # ── 最佳模型信息（train_model 返回） ──
                          best_composite_info,   # {'epoch': int, 'loss': float, 'ssim': float, 'mae': float, 'path': str}
                          best_ssim_info,        # {'epoch': int, 'loss': float, 'ssim': float, 'mae': float, 'path': str}
                          best_loss_info,        # {'epoch': int, 'loss': float, 'ssim': float, 'mae': float, 'path': str}
                          # ── 每 epoch 历史（用于最终 epoch 报告） ──
                          train_losses, val_losses,
                          train_ssims,  val_ssims,
                          train_maes,   val_maes,
                          # ── 总训练时长 ──
                          total_time_str,
                          # ── 可选附加路径 ──
                          checkpoint_dir='', save_every_n_epochs=0,
                          # 【新增 MSE】MSE 历史列表，默认 None 保持向后兼容
                          train_mses=None, val_mses=None):
    """
    生成并保存训练总结文件 training_summary.txt。

    文件结构：
        ① Training Summary（基本信息）
        ② Best Models（三种最优模型各自的 epoch / Loss / SSIM / MAE）
        ③ Final Epoch Result（最后一个 epoch 的各项指标）
        ④ Output Files（生成文件一览）

    参数:
        output_folder        : 文件保存目录（会自动创建）
        total_epochs         : 实际完成的训练轮数
        batch_size           : 训练批次大小
        learning_rate        : 初始学习率
        gradient_accumulation_steps : 梯度累积步数
        image_resize         : 图像尺寸设置（int / tuple / None）
        optimizer_type       : 优化器类型字符串
        scheduler_type       : 学习率调度器类型字符串
        use_amp              : 是否启用混合精度训练
        loss_weight_diffuse  : 漫反射损失权重
        loss_weight_specular : 镜面反射损失权重
        loss_weight_gradient : 梯度损失权重
        best_composite_info  : 综合评分最佳模型信息字典
        best_ssim_info       : SSIM 最佳模型信息字典
        best_loss_info       : Loss 最佳模型信息字典
        train_losses / val_losses : 每 epoch 平均 Loss 列表
        train_ssims  / val_ssims  : 每 epoch 平均 SSIM 列表
        train_maes   / val_maes   : 每 epoch 平均 MAE  列表
        total_time_str       : 格式化总训练时长字符串
        checkpoint_dir       : 检查点保存目录（可选）
        save_every_n_epochs  : 检查点保存间隔（0 表示不保存）

    返回:
        save_path: 保存路径字符串
    """
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, 'training_summary.txt')

    # 最终 epoch 各项指标
    final_epoch   = total_epochs
    final_tr_loss = train_losses[-1] if train_losses else float('nan')
    final_vl_loss = val_losses[-1]   if val_losses   else float('nan')
    final_tr_ssim = train_ssims[-1]  if train_ssims  else float('nan')
    final_vl_ssim = val_ssims[-1]    if val_ssims    else float('nan')
    final_tr_mae  = train_maes[-1]   if train_maes   else float('nan')
    final_vl_mae  = val_maes[-1]     if val_maes     else float('nan')
    # 【新增 MSE】读取最终 epoch 的 MSE 均值
    final_tr_mse  = train_mses[-1]   if (train_mses and len(train_mses) > 0) else None
    final_vl_mse  = val_mses[-1]     if (val_mses   and len(val_mses)   > 0) else None

    # 图像尺寸格式化
    if image_resize is None:
        resize_str = 'Original (no resize)'
    elif isinstance(image_resize, int):
        resize_str = f'{image_resize} × {image_resize}'
    else:
        resize_str = f'{image_resize[0]} × {image_resize[1]}'

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
        f.write(f'  Grad Accum Steps  : {gradient_accumulation_steps}'
                f'  (effective batch = {batch_size * gradient_accumulation_steps})\n')
        f.write(f'  Image Resize      : {resize_str}\n')
        f.write(f'  Optimizer         : {optimizer_type}\n')
        f.write(f'  LR Scheduler      : {scheduler_type}\n')
        f.write(f'  Mixed Precision   : {"Enabled (AMP)" if use_amp else "Disabled"}\n')
        f.write('\n')
        f.write('  [Loss Weights]\n')
        f.write(f'  Diffuse           : {loss_weight_diffuse}\n')
        f.write(f'  Specular          : {loss_weight_specular}\n')
        f.write(f'  Gradient          : {loss_weight_gradient}\n')
        f.write('\n')

        # ──────────────────────────────────────────────────────
        # ② Best Models（三个最优模型，每个包含 epoch / Loss / SSIM / MAE）
        # ──────────────────────────────────────────────────────
        f.write(sep)
        f.write('Best Models\n')
        f.write(sep)

        # 综合评分最优
        f.write('  Best Overall Model  (SSIM×0.7 + NormLoss×0.3)\n')
        f.write(sep2)
        f.write(f'  Epoch             : {best_composite_info["epoch"]}\n')
        f.write(f'  Val Loss          : {best_composite_info["loss"]:.6f}\n')
        f.write(f'  Val SSIM          : {best_composite_info["ssim"]:.4f}\n')
        f.write(f'  Val MAE           : {best_composite_info["mae"]:.6f}\n')
        f.write(f'  Saved Path        : {best_composite_info["path"]}\n')
        f.write('\n')

        # SSIM 最优
        f.write('  Best SSIM Model  (highest diffuse SSIM)\n')
        f.write(sep2)
        f.write(f'  Epoch             : {best_ssim_info["epoch"]}\n')
        f.write(f'  Val SSIM          : {best_ssim_info["ssim"]:.4f}\n')
        f.write(f'  Val Loss          : {best_ssim_info["loss"]:.6f}\n')
        f.write(f'  Val MAE           : {best_ssim_info["mae"]:.6f}\n')
        f.write(f'  Saved Path        : {best_ssim_info["path"]}\n')
        f.write('\n')

        # Loss 最优
        f.write('  Best Loss Model  (lowest val loss, fallback)\n')
        f.write(sep2)
        f.write(f'  Epoch             : {best_loss_info["epoch"]}\n')
        f.write(f'  Val Loss          : {best_loss_info["loss"]:.6f}\n')
        f.write(f'  Val SSIM          : {best_loss_info["ssim"]:.4f}\n')
        f.write(f'  Val MAE           : {best_loss_info["mae"]:.6f}\n')
        f.write(f'  Saved Path        : {best_loss_info["path"]}\n')
        f.write('\n')

        # ──────────────────────────────────────────────────────
        # ③ Final Epoch Result（最后一个 epoch 的全部指标）
        # ──────────────────────────────────────────────────────
        f.write(sep)
        f.write('Final Epoch Result\n')
        f.write(sep)
        f.write(f'  Epoch             : {final_epoch}\n')
        f.write(f'  Train Loss        : {final_tr_loss:.6f}\n')
        f.write(f'  Val   Loss        : {final_vl_loss:.6f}\n')
        f.write(f'  Train SSIM        : {final_tr_ssim:.4f}\n')
        f.write(f'  Val   SSIM        : {final_vl_ssim:.4f}\n')
        f.write(f'  Train MAE         : {final_tr_mae:.6f}\n')
        f.write(f'  Val   MAE         : {final_vl_mae:.6f}\n')
        # 【新增 MSE】在 MAE 后追加 MSE，仅当有数据时写入
        if final_tr_mse is not None:
            f.write(f'  Train MSE         : {final_tr_mse:.6f}\n')
            f.write(f'  Val   MSE         : {final_vl_mse:.6f}\n')
        f.write('\n')

        # ──────────────────────────────────────────────────────
        # ④ Output Files（文件一览）
        # ──────────────────────────────────────────────────────
        f.write(sep)
        f.write('Output Files\n')
        f.write(sep)
        history_path = os.path.join(output_folder, 'training_history.png')
        vis_path     = os.path.join(output_folder, 'visualizations')
        f.write(f'  Training History  : {history_path}\n')
        f.write(f'  Visualizations    : {vis_path}\n')
        if save_every_n_epochs > 0 and checkpoint_dir:
            f.write(f'  Checkpoints       : {checkpoint_dir}\n')
        f.write('\n' + sep)

    return save_path


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda',
                use_amp=True, gradient_accumulation_steps=1, optimizer_config=None,
                scheduler_config=None, loss_config=None, training_config=None):
    """
    训练模型：执行多轮（epoch）训练与验证，并保存最佳模型权重

    参数:
        model: Y-FFC网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数，默认100轮
        lr: 学习率，默认0.001
        device: 计算设备，可传字符串或 torch.device 对象
        use_amp: 是否使用混合精度训练，默认True（节省显存）
        gradient_accumulation_steps: 梯度累积步数，默认1（增大此值可模拟更大的batch_size）
        optimizer_config: 优化器配置字典
        scheduler_config: 学习率调度器配置字典
        loss_config: 损失函数配置字典
        training_config: 训练策略配置字典

    返回:
        history: dict，包含以下键：
            'train_losses'       : 每 epoch 平均训练 Loss
            'val_losses'         : 每 epoch 平均验证 Loss
            'train_ssims'        : 每 epoch 平均训练 SSIM（漫反射）
            'val_ssims'          : 每 epoch 平均验证 SSIM（漫反射）
            'train_maes'         : 每 epoch 平均训练 MAE（漫反射）  【新增】
            'val_maes'           : 每 epoch 平均验证 MAE（漫反射）  【新增】
            'train_mses'         : 每 epoch 平均训练 MSE（漫反射）  【新增 MSE】
            'val_mses'           : 每 epoch 平均验证 MSE（漫反射）  【新增 MSE】
            'best_composite_info': 综合评分最优模型信息字典         【新增】
            'best_ssim_info'     : SSIM 最优模型信息字典             【新增】
            'best_loss_info'     : Loss 最优模型信息字典             【新增】
            'total_time_str'     : 总训练时长格式化字符串            【新增】

        同时保留向后兼容的 4-tuple 展开方式（为不影响已有调用方）：
            train_losses, val_losses, train_accuracies, val_accuracies
        通过 history['train_losses'] 等访问扩展字段。
    """
    # [修复 P1] 统一 device 类型：函数签名允许传字符串或 torch.device，
    # 内部统一转为 torch.device 后使用 .type 属性，避免 AttributeError 崩溃。
    if isinstance(device, str):
        device = torch.device(device)
    # 设置默认配置
    if optimizer_config is None:
        optimizer_config = {'type': 'adam', 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0}
    if scheduler_config is None:
        scheduler_config = {'type': 'step', 'step_size': 30, 'gamma': 0.5}
    if loss_config is None:
        loss_config = {'ssim_window_size': 11, 'ssim_sigma': 1.5,
                      'weight_diffuse': 1.0, 'weight_specular': 1.0, 'weight_gradient': 1.0}
    if training_config is None:
        # [修复 P5] use_grad_clip 默认改为 True：
        # Y-FFC 含 ViT+ResNet50+FFC 多分支结构，训练初期梯度分布不稳定，
        # 高反光场景的 loss 变化剧烈，梯度爆炸风险高，应默认开启裁剪保护。
        training_config = {'use_grad_clip': True, 'grad_clip_value': 1.0,
                          'use_early_stopping': False, 'early_stopping_patience': 20,
                          'model_save_path': 'best_model.pth', 'checkpoint_dir': 'checkpoints',
                          'save_every_n_epochs': 0}
    # 清理CUDA缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"已清理CUDA缓存，当前显存使用: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
    
    model.to(device)  # 将模型移动到指定设备（CPU/GPU）
    
    # ==================== 初始化优化器 ====================
    opt_type = optimizer_config.get('type', 'adam').lower()
    if opt_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=lr,
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
        print(f"使用Adam优化器: lr={lr}, betas={optimizer_config.get('betas', (0.9, 0.999))}, weight_decay={optimizer_config.get('weight_decay', 0.0)}")
    elif opt_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8),
            weight_decay=optimizer_config.get('weight_decay', 0.01)
        )
        print(f"使用AdamW优化器: lr={lr}, weight_decay={optimizer_config.get('weight_decay', 0.01)}")
    elif opt_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 0.0001)
        )
        print(f"使用SGD优化器: lr={lr}, momentum={optimizer_config.get('momentum', 0.9)}")
    else:
        raise ValueError(f"不支持的优化器类型: {opt_type}")
    
    # ==================== 初始化学习率调度器 ====================
    sched_type = scheduler_config.get('type', 'step').lower()
    if sched_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.5)
        )
        print(f"使用StepLR调度器: step_size={scheduler_config.get('step_size', 30)}, gamma={scheduler_config.get('gamma', 0.5)}")
    elif sched_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', epochs),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
        print(f"使用CosineAnnealingLR调度器: T_max={scheduler_config.get('T_max', epochs)}")
    elif sched_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 10),
            verbose=True
        )
        print(f"使用ReduceLROnPlateau调度器: patience={scheduler_config.get('patience', 10)}")
    elif sched_type == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.get('gamma', 0.95)
        )
        print(f"使用ExponentialLR调度器: gamma={scheduler_config.get('gamma', 0.95)}")
    else:
        scheduler = None
        print("未使用学习率调度器")
    
    # 混合精度训练scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    
    # ==================== 定义损失函数 ====================
    ssim_window_size = loss_config.get('ssim_window_size', 11)
    ssim_sigma = loss_config.get('ssim_sigma', 1.5)
    ssim_loss = SSIM(window_size=ssim_window_size, sigma=ssim_sigma)
    mae_loss = nn.L1Loss()
    print(f"SSIM损失参数: window_size={ssim_window_size}, sigma={ssim_sigma}")
    
    # 损失权重
    weight_diffuse = loss_config.get('weight_diffuse', 1.0)
    weight_specular = loss_config.get('weight_specular', 1.0)
    weight_gradient = loss_config.get('weight_gradient', 1.0)
    print(f"损失权重: 漫反射={weight_diffuse}, 镜面反射={weight_specular}, 梯度={weight_gradient}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # 历史记录列表：每个 epoch 追加一条均值，训练结束后传给 plot/summary 函数使用
    # ──────────────────────────────────────────────────────────────────────────
    train_losses     = []   # 每轮训练平均 Loss
    val_losses       = []   # 每轮验证平均 Loss
    train_accuracies = []   # 每轮训练平均 SSIM（兼容旧接口，与 train_ssims 同值）
    val_accuracies   = []   # 每轮验证平均 SSIM（兼容旧接口，与 val_ssims 同值）
    train_ssims      = []   # 每轮训练平均 SSIM（漫反射分支）
    val_ssims        = []   # 每轮验证平均 SSIM（漫反射分支）
    # 【新增】每 epoch 按批次累积漫反射 MAE，epoch 结束时除以批次数得到均值
    train_maes       = []   # 每轮训练平均 MAE（漫反射分支）
    val_maes         = []   # 每轮验证平均 MAE（漫反射分支）
    # 【新增 MSE】每 epoch 按批次累积漫反射 MSE，与 MAE 同步维护
    train_mses       = []   # 每轮训练平均 MSE（漫反射分支）
    val_mses         = []   # 每轮验证平均 MSE（漫反射分支）
    epoch_times      = []   # 每轮训练时间

    # ==================== 最佳模型多指标跟踪初始化 ====================
    # 策略说明：
    #   综合评分 = α × SSIM + (1-α) × 1/(1+loss)
    #   SSIM 占主导（α=0.7），体现主任务（漫反射分离）的感知质量优先；
    #   loss 归一化后作为辅助约束，防止感知质量相近时选出训练不稳定的模型。
    #
    #   同时独立跟踪 SSIM 最优和 Loss 最优，三类模型各保存一份互为备份：
    #     - best_composite.pth : 综合评分最高（推荐用于最终测试）
    #     - best_ssim.pth      : 漫反射 SSIM 最高（视觉质量最优）
    #     - best_loss.pth      : 验证 loss 最低（兜底备份）
    COMPOSITE_ALPHA = 0.7   # SSIM 在综合评分中的权重，(1-α) 分配给归一化 loss
    best_val_loss      = float('inf')   # 验证 loss 历史最低（越小越好）
    best_val_ssim      = 0.0            # 验证 SSIM 历史最高（越大越好）
    best_composite     = 0.0            # 综合评分历史最高（越大越好）

    # 【新增】记录三种最优模型对应的 epoch 及各项指标，供 save_training_summary 使用
    best_composite_info = {'epoch': 0, 'loss': float('inf'), 'ssim': 0.0, 'mae': float('inf'), 'path': ''}
    best_ssim_info      = {'epoch': 0, 'loss': float('inf'), 'ssim': 0.0, 'mae': float('inf'), 'path': ''}
    best_loss_info      = {'epoch': 0, 'loss': float('inf'), 'ssim': 0.0, 'mae': float('inf'), 'path': ''}

    # 训练策略配置
    use_grad_clip = training_config.get('use_grad_clip', False)
    grad_clip_value = training_config.get('grad_clip_value', 1.0)
    use_early_stopping = training_config.get('use_early_stopping', False)
    early_stopping_patience = training_config.get('early_stopping_patience', 20)
    early_stopping_counter = 0
    model_save_path = training_config.get('model_save_path', 'best_model.pth')
    checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints')
    save_every_n_epochs = training_config.get('save_every_n_epochs', 0)

    # 从 model_save_path 派生出三个模型文件路径，存放在同一目录下
    _model_dir  = os.path.dirname(os.path.abspath(model_save_path))
    _model_stem = os.path.splitext(os.path.basename(model_save_path))[0]
    os.makedirs(_model_dir, exist_ok=True)
    best_composite_save_path = model_save_path                                            # 综合最优（=原 model_save_path，保持向后兼容）
    best_ssim_save_path      = os.path.join(_model_dir, f'{_model_stem}_best_ssim.pth')  # SSIM 最优
    best_loss_save_path      = os.path.join(_model_dir, f'{_model_stem}_best_loss.pth')  # Loss 最优

    # 将保存路径写入 info 字典
    best_composite_info['path'] = best_composite_save_path
    best_ssim_info['path']      = best_ssim_save_path
    best_loss_info['path']      = best_loss_save_path

    print(f"\n模型保存路径:")
    print(f"  综合最优 → {best_composite_save_path}")
    print(f"  SSIM最优 → {best_ssim_save_path}")
    print(f"  Loss最优 → {best_loss_save_path}")
    print(f"  综合评分权重: SSIM×{COMPOSITE_ALPHA:.1f} + Loss归一化×{1 - COMPOSITE_ALPHA:.1f}")

    # 创建检查点目录
    if save_every_n_epochs > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    if use_grad_clip:
        print(f"启用梯度裁剪: max_norm={grad_clip_value}")
    if use_early_stopping:
        print(f"启用早停策略: patience={early_stopping_patience}（由综合评分驱动）")
    
    print(f"开始训练，共{epochs}轮，学习率{lr}")
    if use_amp and device.type == 'cuda':
        print("使用混合精度训练（AMP）以节省显存")
    if gradient_accumulation_steps > 1:
        print(f"使用梯度累积，累积步数: {gradient_accumulation_steps}")
    
    # 打印训练开始时间
    start_time_total = time.time()
    print(f"\n 训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    for epoch in range(epochs):  # 逐轮训练
        epoch_start_time = time.time()  # 记录本轮开始时间
        
        # ==================== 训练阶段 ====================
        model.train()  # 切换到训练模式（启用BN/Dropout等正则化）
        train_loss       = 0   # 本轮训练 Loss 累计
        train_ssim_total = 0   # 本轮训练 SSIM 累计
        train_mae_total  = 0   # 【新增】本轮训练 MAE 累计（漫反射分支）
        # 【新增 MSE】本轮训练 MSE 累计（漫反射分支），epoch 结束后除批次数得均值
        train_mse_total  = 0
        
        # 使用 tqdm 展示训练批次进度
        optimizer.zero_grad()  # 在循环开始前清空梯度
        
        for batch_idx, batch in enumerate(tqdm(train_loader, total=len(train_loader), desc=f'Train {epoch+1}/{epochs}', leave=False)):
            # 将数据移到指定设备
            inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]

            # ==================== 前向传播（使用混合精度） ====================
            # [修复 P2] AMP/非AMP 路径合并：前向传播 + loss 计算统一走 _compute_loss()，
            # 消除原代码中 4 份完全相同的 loss 计算逻辑，任何修改只需改一处。
            if use_amp and scaler is not None:
                with torch.amp.autocast('cuda'):
                    estimations = model(inputs)
                    pred_diffuse  = estimations['r']
                    pred_specular = estimations['b']
                    pred_gradient = estimations['g']
                    loss_raw, ssim_val, mae_val, mse_val, loss_comps = _compute_loss(
                        pred_diffuse, pred_specular, pred_gradient,
                        gt_diffuse, gt_specular, gt_gradient,
                        ssim_loss, mae_loss,
                        weight_diffuse, weight_specular, weight_gradient,
                    )
            else:
                estimations = model(inputs)
                pred_diffuse  = estimations['r']
                pred_specular = estimations['b']
                pred_gradient = estimations['g']
                loss_raw, ssim_val, mae_val, mse_val, loss_comps = _compute_loss(
                    pred_diffuse, pred_specular, pred_gradient,
                    gt_diffuse, gt_specular, gt_gradient,
                    ssim_loss, mae_loss,
                    weight_diffuse, weight_specular, weight_gradient,
                )

            # [修复 P6] NaN 保护：高反光过曝区域可能导致 statistical_loss std→0，
            # 一旦产生 NaN 会悄悄污染所有后续梯度直至模型崩溃。
            # 检测到 NaN 时跳过本 batch，打印警告，不执行反向传播。
            if not torch.isfinite(loss_raw):
                tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {batch_idx+1}: '
                           f'loss 出现 NaN/Inf（{loss_raw.item():.4g}），跳过本批次。'
                           f' loss_comps={loss_comps}')
                optimizer.zero_grad()
                continue

            loss = loss_raw / gradient_accumulation_steps  # 梯度累积时缩放损失

            # [修复 P3] SSIM/MAE 精度统计直接使用 _compute_loss 返回值，
            # 无需再次前向计算。
            train_ssim_total += ssim_val
            train_mae_total  += mae_val   # 【新增】累积训练 MAE
            # 【新增 MSE】累积训练 MSE，与 MAE 保持同步
            train_mse_total  += mse_val

            # ==================== 反向传播与参数更新 ====================
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积：每 gradient_accumulation_steps 步更新一次参数
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # [修复 P4] 统一走 _optimizer_step，确保裁剪/step/update/zero_grad 顺序一致
                _optimizer_step(optimizer, scaler, use_amp, use_grad_clip, grad_clip_value, model)

            train_loss += loss_raw.item()  # 累计原始量级损失（不除以累积步数）

            # 在进度条上显示当前损失
            if (batch_idx + 1) % 10 == 0:
                tqdm.write(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss_raw.item():.6f}'
                           f'  [ld={loss_comps["ld"]:.4f} ls={loss_comps["ls"]:.4f} lg={loss_comps["lg"]:.4f}]')

        # [修复 P4] 末尾不完整累积批次：走相同的 _optimizer_step，
        # 确保梯度裁剪逻辑与正常路径完全一致（原代码此处缺少梯度裁剪）。
        if len(train_loader) > 0 and (len(train_loader)) % gradient_accumulation_steps != 0:
            _optimizer_step(optimizer, scaler, use_amp, use_grad_clip, grad_clip_value, model)
        
        # ==================== 验证阶段 ====================
        model.eval()  # 切换到评估模式（禁用BN/Dropout等）
        val_loss       = 0   # 本轮验证 Loss 累计
        val_ssim_total = 0   # 本轮验证 SSIM 累计
        val_mae_total  = 0   # 【新增】本轮验证 MAE 累计（漫反射分支）
        # 【新增 MSE】本轮验证 MSE 累计（漫反射分支）
        val_mse_total  = 0
        
        with torch.no_grad():  # 关闭梯度计算以节省显存与加速
            # 使用 tqdm 展示验证批次进度
            for batch in tqdm(val_loader, total=len(val_loader), desc=f'Val {epoch+1}/{epochs}', leave=False):
                inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]

                # [修复 P2] 验证路径同样统一走 _compute_loss，AMP/非AMP 合并。
                # 验证阶段 with torch.no_grad() 已禁用梯度，autocast 仅影响精度，
                # 不影响正确性；非AMP 时直接调用同一函数。
                if use_amp and scaler is not None:
                    with torch.amp.autocast('cuda'):
                        estimations = model(inputs)
                        pred_diffuse  = estimations['r']
                        pred_specular = estimations['b']
                        pred_gradient = estimations['g']
                        loss_raw, ssim_val, mae_val, mse_val, _ = _compute_loss(
                            pred_diffuse, pred_specular, pred_gradient,
                            gt_diffuse, gt_specular, gt_gradient,
                            ssim_loss, mae_loss,
                            weight_diffuse, weight_specular, weight_gradient,
                        )
                else:
                    estimations = model(inputs)
                    pred_diffuse  = estimations['r']
                    pred_specular = estimations['b']
                    pred_gradient = estimations['g']
                    loss_raw, ssim_val, mae_val, mse_val, _ = _compute_loss(
                        pred_diffuse, pred_specular, pred_gradient,
                        gt_diffuse, gt_specular, gt_gradient,
                        ssim_loss, mae_loss,
                        weight_diffuse, weight_specular, weight_gradient,
                    )

                # 验证阶段同样做 NaN 检测，异常 batch 不纳入统计
                if not torch.isfinite(loss_raw):
                    continue

                val_ssim_total += ssim_val
                val_mae_total  += mae_val   # 【新增】累积验证 MAE
                # 【新增 MSE】累积验证 MSE
                val_mse_total  += mse_val
                val_loss += loss_raw.item()
        
        # ==================== 统计与记录 ====================
        # 计算本轮的平均训练/验证损失和精度
        avg_train_loss = train_loss       / len(train_loader)
        avg_val_loss   = val_loss         / len(val_loader)
        avg_train_acc  = train_ssim_total / len(train_loader)   # SSIM 作为精度指标
        avg_val_acc    = val_ssim_total   / len(val_loader)
        avg_train_mae  = train_mae_total  / len(train_loader)   # 【新增】训练 MAE 均值
        avg_val_mae    = val_mae_total    / len(val_loader)      # 【新增】验证 MAE 均值
        # 【新增 MSE】计算漫反射分支 MSE 均值（各 batch MSE 之和除以批次数）
        avg_train_mse  = train_mse_total  / len(train_loader)
        avg_val_mse    = val_mse_total    / len(val_loader)
        
        # 记录损失、SSIM、MAE 历史
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(avg_train_acc)      # 兼容旧接口
        val_accuracies.append(avg_val_acc)          # 兼容旧接口
        train_ssims.append(avg_train_acc)           # 与 train_accuracies 同值，语义更清晰
        val_ssims.append(avg_val_acc)
        train_maes.append(avg_train_mae)            # 【新增】
        val_maes.append(avg_val_mae)                # 【新增】
        # 【新增 MSE】追加 MSE 历史，供训练结束后绘图与汇总使用
        train_mses.append(avg_train_mse)
        val_mses.append(avg_val_mse)
        
        # 计算本轮耗时和预估剩余时间
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta = timedelta(seconds=int(eta_seconds))
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # ==================== 打印详细训练信息 ====================
        print("\n" + "=" * 80)
        print(f"Epoch [{epoch+1}/{epochs}] 训练统计")
        print("=" * 80)
        print(f"本轮耗时: {epoch_time:.2f}秒 | 平均耗时: {avg_epoch_time:.2f}秒/轮 | 预计剩余: {eta}")
        print(f"损失值:   训练损失 = {avg_train_loss:.6f} | 验证损失 = {avg_val_loss:.6f}")
        print(f"SSIM:     训练SSIM = {avg_train_acc:.4f}  | 验证SSIM = {avg_val_acc:.4f}")
        print(f"MAE:      训练MAE  = {avg_train_mae:.6f} | 验证MAE  = {avg_val_mae:.6f}")
        # 【新增 MSE】在 MAE 下方新增一行，格式与 MAE 保持对齐，方便对比
        print(f"MSE:      训练MSE  = {avg_train_mse:.6f} | 验证MSE  = {avg_val_mse:.6f}")
        print(f"超参数:   学习率 = {current_lr:.6f} | 批大小 = {train_loader.batch_size} | 累积步数 = {gradient_accumulation_steps}")
        if device.type == 'cuda':
            print(f"显存使用: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB / {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB (当前/峰值)")

        # ==================== 最佳模型保存（多指标策略） ====================
        # 综合评分公式：
        #   composite = α × SSIM + (1-α) × 1/(1+loss)
        # 其中 1/(1+loss) 将 loss ∈ [0, +∞) 映射到 (0, 1]，与 SSIM 量纲对齐。
        # α=0.7 使得 SSIM（直接衡量漫反射视觉质量）占主导地位。
        composite_score = (COMPOSITE_ALPHA * avg_val_acc
                           + (1.0 - COMPOSITE_ALPHA) * (1.0 / (1.0 + avg_val_loss)))

        improved_flags = []  # 记录本轮改善了哪些指标，打印时展示

        # --- 策略 A：综合评分最佳 → 保存为主模型（覆盖 model_save_path）---
        # 同时驱动早停计数器：综合评分改善时重置，否则累加。
        # 这比单纯用 loss 驱动早停更合理，因为它同时考虑了感知质量。
        if composite_score > best_composite:
            best_composite = composite_score
            early_stopping_counter = 0   # 综合评分改善 → 重置早停
            torch.save(model.state_dict(), best_composite_save_path)
            improved_flags.append(f'综合↑{composite_score:.4f}')
            # 【新增】更新综合最优 epoch 及指标
            best_composite_info.update({
                'epoch': epoch + 1,
                'loss':  avg_val_loss,
                'ssim':  avg_val_acc,
                'mae':   avg_val_mae,
            })
        else:
            early_stopping_counter += 1

        # --- 策略 B：漫反射 SSIM 单独最佳 → 额外保存，独立于早停 ---
        # 即使综合评分未改善，只要漫反射 SSIM 创历史新高就保存。
        # 此模型代表主任务（高反光表面漫反射分离）的视觉质量峰值。
        if avg_val_acc > best_val_ssim:
            best_val_ssim = avg_val_acc
            torch.save(model.state_dict(), best_ssim_save_path)
            improved_flags.append(f'SSIM↑{avg_val_acc:.4f}')
            # 【新增】更新 SSIM 最优 epoch 及指标
            best_ssim_info.update({
                'epoch': epoch + 1,
                'loss':  avg_val_loss,
                'ssim':  avg_val_acc,
                'mae':   avg_val_mae,
            })

        # --- 策略 C：验证 loss 单独最佳 → 额外保存，兜底备份 ---
        # 即使综合评分和 SSIM 未改善，只要 loss 创历史新低就保存。
        # 为极端情况（如 SSIM 计算异常）提供保底模型。
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_loss_save_path)
            improved_flags.append(f'Loss↓{avg_val_loss:.6f}')
            # 【新增】更新 Loss 最优 epoch 及指标
            best_loss_info.update({
                'epoch': epoch + 1,
                'loss':  avg_val_loss,
                'ssim':  avg_val_acc,
                'mae':   avg_val_mae,
            })

        # 打印本轮模型保存情况
        if improved_flags:
            print(f'✅ 模型已更新 → {" | ".join(improved_flags)}')
            print(f'   综合最优: {best_composite_save_path}')
            print(f'   SSIM最优: {best_ssim_save_path}')
            print(f'   Loss最优: {best_loss_save_path}')
        else:
            if use_early_stopping:
                print(f'⚠️  指标未改善 ({early_stopping_counter}/{early_stopping_patience})'
                      f'  综合={composite_score:.4f} SSIM={avg_val_acc:.4f} Loss={avg_val_loss:.6f}')
            else:
                print(f'   本轮无改善  综合={composite_score:.4f} SSIM={avg_val_acc:.4f} Loss={avg_val_loss:.6f}')

        # ==================== 实时绘制训练曲线 ====================
        # [修复 P10] 原代码用 os.path.dirname(model_save_path) 推导输出目录，
        # 若路径不含目录分隔符则返回空串，曲线会保存到错误位置。
        # 改为直接取 checkpoint_dir 的父目录（即 OUTPUT_FOLDER），更可靠。
        curves_output_folder = os.path.dirname(os.path.abspath(model_save_path))
        if (epoch + 1) % 5 == 0 or epoch == 0:  # 每5轮或第一轮绘制一次
            plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies,
                               curves_output_folder, epoch + 1)
            print(f'训练曲线已更新: {os.path.join(curves_output_folder, "training_curves_realtime.png")}')

        # ==================== 定期可视化样本结果 ====================
        if (epoch + 1) % 10 == 0 or epoch == 0:  # 每10轮或第一轮可视化一次
            visualize_sample_during_training(model, val_loader, device, curves_output_folder, epoch + 1)
            print(f'样本可视化已更新: {os.path.join(curves_output_folder, f"sample_epoch_{epoch+1}.png")}')
        
        # ==================== 定期保存检查点 ====================
        if save_every_n_epochs > 0 and (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            # [修复 P9] 补充 scheduler_state_dict 和 scaler_state_dict：
            # 原代码缺失这两项，断点续训时 scheduler 从初始 LR 重新开始，
            # AMP scaler 的历史缩放因子也会重置，导致续训行为与完整训练不一致。
            # 同时记录三个最佳指标值，便于续训时正确初始化跟踪变量。
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': avg_train_acc,
                'val_acc': avg_val_acc,
                # 多指标跟踪状态（断点续训时可恢复）
                'best_val_loss': best_val_loss,
                'best_val_ssim': best_val_ssim,
                'best_composite': best_composite,
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint, checkpoint_path)
            print(f'检查点已保存: {checkpoint_path}')
            print("=" * 80 + "\n")
        
        # ==================== 早停检查 ====================
        # 早停由综合评分驱动（而非单纯 loss），与主任务目标一致
        if use_early_stopping and early_stopping_counter >= early_stopping_patience:
            print(f'\n⛔ 早停触发！综合评分在 {early_stopping_patience} 轮内未改善，停止训练。')
            break
        
        # ==================== 学习率调度 ====================
        if scheduler is not None:
            if sched_type == 'plateau':
                scheduler.step(avg_val_loss)  # ReduceLROnPlateau需要验证损失
            else:
                scheduler.step()  # 其他调度器按epoch步进
    
    # ==================== 训练总结 ====================
    total_time = time.time() - start_time_total
    total_time_str = str(timedelta(seconds=int(total_time)))
    actual_epochs = len(train_losses)   # 实际完成轮数（早停时 < epochs）

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"总训练时间:   {total_time_str}")
    print(f"最佳综合评分: {best_composite:.4f}  (Epoch {best_composite_info['epoch']}) → {best_composite_save_path}")
    print(f"最佳验证SSIM: {best_val_ssim:.4f}   (Epoch {best_ssim_info['epoch']}) → {best_ssim_save_path}")
    print(f"最佳验证Loss: {best_val_loss:.6f} (Epoch {best_loss_info['epoch']}) → {best_loss_save_path}")
    print(f"最终训练精度: {train_accuracies[-1]:.4f} | 最终验证精度: {val_accuracies[-1]:.4f}")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # 将所有历史数据和最优模型信息打包为 history 字典返回，
    # 供 main() 中的 plot_training_history / save_training_summary 使用。
    # 同时保持 4-tuple 返回接口向后兼容（train_losses, val_losses,
    # train_accuracies, val_accuracies）——history 对象附加在 train_losses 上，
    # 通过 train_losses.history 访问（无需修改已有调用代码）。
    # ──────────────────────────────────────────────────────────────────────────
    history = {
        'train_losses':        train_losses,
        'val_losses':          val_losses,
        'train_ssims':         train_ssims,
        'val_ssims':           val_ssims,
        'train_maes':          train_maes,
        'val_maes':            val_maes,
        # 【新增 MSE】漫反射分支 MSE 每 epoch 均值历史
        'train_mses':          train_mses,
        'val_mses':            val_mses,
        'best_composite_info': best_composite_info,
        'best_ssim_info':      best_ssim_info,
        'best_loss_info':      best_loss_info,
        'total_time_str':      total_time_str,
        'actual_epochs':       actual_epochs,
    }

    return train_losses, val_losses, train_accuracies, val_accuracies, history


def test_model(model, test_loader, device='cuda'):
    """
    测试模型：在测试集上计算评估指标（SSIM/MAE/PSNR），不更新参数
    
    参数:
        model: 训练好的Y-FFC网络模型
        test_loader: 测试数据加载器
        device: 计算设备
    
    返回:
        avg_ssim: 平均结构相似性指数
        avg_mae: 平均绝对误差
        avg_psnr: 平均峰值信噪比
    """
    model.eval()  # 切换到评估模式
    
    # 初始化损失函数（用于计算指标）
    ssim_loss = SSIM()  # 用于计算 SSIM 指标
    mae_loss = nn.L1Loss()  # 用于计算 MAE 指标
    
    # 累计指标统计
    total_ssim = 0  # SSIM 累计
    total_mae = 0   # MAE 累计
    total_psnr = 0  # PSNR 累计
    
    print("开始模型测试...")
    
    with torch.no_grad():  # 不计算梯度，节省内存
        for batch_idx, batch in enumerate(tqdm(test_loader, total=len(test_loader), desc='Testing', leave=True)):
            inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]
            
            # 前向传播
            estimations = model(inputs)
            pred_diffuse = estimations['r']     # 漫反射预测
            pred_specular = estimations['b']    # 镜面反射预测（这里未直接用于指标）
            pred_gradient = estimations['g']    # 梯度预测（这里未直接用于指标）
            
            # ==================== 计算评估指标 ====================
            ssim_val = ssim_loss(pred_diffuse, gt_diffuse)  # 结构相似性
            mae_val = mae_loss(pred_diffuse, gt_diffuse)    # 平均绝对误差
            
            # PSNR 计算：基于均方误差，值越大表示图像质量越好
            mse = nn.MSELoss()(pred_diffuse, gt_diffuse)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))  # PSNR = 20*log10(MAX/√MSE)
            
            # 累计指标
            total_ssim += ssim_val.item()
            total_mae += mae_val.item()
            total_psnr += psnr.item()
            
            # 在进度条上输出每个 batch 的指标
            tqdm.write(f'Batch {batch_idx}: SSIM: {ssim_val:.4f}, MAE: {mae_val:.4f}, PSNR: {psnr:.2f}')
    
    # ==================== 计算平均指标 ====================
    avg_ssim = total_ssim / len(test_loader)
    avg_mae = total_mae / len(test_loader)
    avg_psnr = total_psnr / len(test_loader)
    
    print(f'\n========== 测试结果汇总 ==========')
    print(f'SSIM: {avg_ssim:.4f} (越接近1越好)')
    print(f'MAE:  {avg_mae:.4f} (越接近0越好)')
    print(f'PSNR: {avg_psnr:.2f} dB (越大越好)')
    
    return avg_ssim, avg_mae, avg_psnr  # 返回测试集上的平均指标

def visualize_results(model, test_loader, device='cuda', num_samples=5, output_folder=''):
    """
    可视化测试结果：显示输入、预测与差异图，帮助理解模型效果
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        num_samples: 要可视化的样本数量
        output_folder: 输出文件夹路径
    """
    model.eval()  # 评估模式
    
    # 创建可视化结果子文件夹
    if output_folder:
        vis_folder = os.path.join(output_folder, 'visualizations')
        os.makedirs(vis_folder, exist_ok=True)
    else:
        vis_folder = ''
    
    # 处理num_samples为None的情况：可视化所有测试样本
    if num_samples is None:
        num_samples = len(test_loader)
    
    print(f"开始可视化{num_samples}个测试样本...")
    
    with torch.no_grad():  # 不进行梯度计算
        for i, batch in enumerate(tqdm(test_loader, total=min(len(test_loader), num_samples), desc='Visualizing', leave=True)):
            if i >= num_samples:  # 只可视化指定数量样本
                break
                
            inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]
            
            # 前向传播获取预测结果
            estimations = model(inputs)
            pred_diffuse = estimations['r']     # 预测漫反射
            pred_specular = estimations['b']    # 预测镜面反射
            pred_gradient = estimations['g']    # 预测梯度
            
            # ==================== 数据转换 ====================
            # 转为 numpy 数组便于显示（取第一个样本）
            input_img = inputs[0, 0].cpu().numpy()           # 输入图像
            gt_diffuse_img = gt_diffuse[0, 0].cpu().numpy() # 真实漫反射
            pred_diffuse_img = pred_diffuse[0, 0].cpu().numpy() # 预测漫反射
            pred_specular_img = pred_specular[0, 0].cpu().numpy() # 预测镜面反射
            pred_gradient_img = pred_gradient[0, 0].cpu().numpy() # 预测梯度
            
            # ==================== 创建可视化布局 ====================
            # 2×3 网格布局，显示所有相关图像
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 第一行：输入、真实漫反射、预测漫反射
            axes[0, 0].imshow(input_img, cmap='gray')
            axes[0, 0].set_title('Input (Highly Reflective)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(gt_diffuse_img, cmap='gray')
            axes[0, 1].set_title('Ground Truth (Diffuse)')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(pred_diffuse_img, cmap='gray')
            axes[0, 2].set_title('Predicted (Diffuse)')
            axes[0, 2].axis('off')
            
            # 第二行：预测镜面反射、预测梯度、差异图
            axes[1, 0].imshow(pred_specular_img, cmap='gray')
            axes[1, 0].set_title('Predicted (Specular)')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(pred_gradient_img, cmap='gray')
            axes[1, 1].set_title('Predicted (Gradient)')
            axes[1, 1].axis('off')
            
            # 显示与真实漫反射的差异（用于评估预测精度）
            diff = np.abs(pred_diffuse_img - gt_diffuse_img)
            axes[1, 2].imshow(diff, cmap='hot')  # 热力图显示差异
            axes[1, 2].set_title('Difference (|Pred - GT|)')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # 保存到输出文件夹
            if vis_folder:
                save_path = os.path.join(vis_folder, f'results_sample_{i+1}.png')
            else:
                save_path = f'results_sample_{i+1}.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存高分辨率结果
            plt.close()  # 关闭图形，节省内存
            
            print(f"样本 {i+1} 可视化完成，已保存为 {save_path}")

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_folder, current_epoch):
    """
    实时绘制训练曲线（损失和精度）——每 N 轮更新一次的轻量版本。
    与训练结束后调用的 plot_training_history() 不同，此函数仅绘制
    Loss 和 SSIM 两组指标，图像较小，用于训练过程中的快速监控。
    
    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练精度列表
        val_accs: 验证精度列表
        output_folder: 输出文件夹
        current_epoch: 当前epoch数
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-o', label='训练损失', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'r-s', label='验证损失', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('损失值 (Loss)', fontsize=12)
    ax1.set_title(f'训练和验证损失 (Epoch {current_epoch})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 绘制精度曲线
    ax2.plot(epochs, train_accs, 'b-o', label='训练精度 (SSIM)', linewidth=2, markersize=4)
    ax2.plot(epochs, val_accs, 'r-s', label='验证精度 (SSIM)', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('精度 (Accuracy)', fontsize=12)
    ax2.set_title(f'训练和验证精度 (Epoch {current_epoch})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])  # SSIM范围为0-1
    
    plt.tight_layout()
    save_path = os.path.join(output_folder, 'training_curves_realtime.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_sample_during_training(model, val_loader, device, output_folder, current_epoch):
    """
    训练过程中可视化样本结果
    
    参数:
        model: 模型
        val_loader: 验证数据加载器
        device: 计算设备
        output_folder: 输出文件夹
        current_epoch: 当前epoch数
    """
    model.eval()
    with torch.no_grad():
        # 获取一个批次的数据
        batch = next(iter(val_loader))
        inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]
        
        # 前向传播
        estimations = model(inputs)
        pred_diffuse = estimations['r']
        pred_specular = estimations['b']
        pred_gradient = estimations['g']
        
        # 转换为numpy
        input_img = inputs[0, 0].cpu().numpy()
        gt_diffuse_img = gt_diffuse[0, 0].cpu().numpy()
        pred_diffuse_img = pred_diffuse[0, 0].cpu().numpy()
        pred_specular_img = pred_specular[0, 0].cpu().numpy()
        pred_gradient_img = pred_gradient[0, 0].cpu().numpy()
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(input_img, cmap='gray')
        axes[0, 0].set_title('输入图像', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gt_diffuse_img, cmap='gray')
        axes[0, 1].set_title('真实漫反射', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred_diffuse_img, cmap='gray')
        axes[0, 2].set_title('预测漫反射', fontsize=12)
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(pred_specular_img, cmap='gray')
        axes[1, 0].set_title('预测镜面反射', fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred_gradient_img, cmap='gray')
        axes[1, 1].set_title('预测梯度', fontsize=12)
        axes[1, 1].axis('off')
        
        diff = np.abs(pred_diffuse_img - gt_diffuse_img)
        im = axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('差异图', fontsize=12)
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
        
        fig.suptitle(f'Epoch {current_epoch} - 样本可视化', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_folder, f'sample_epoch_{current_epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    model.train()  # 恢复训练模式

def create_dummy_data(input_dir, target_dir, num_samples=100):
    """
    创建虚拟数据集：生成带反射的输入与无反射的目标，用于快速测试训练流程
    
    参数:
        input_dir: 输入图像保存目录
        target_dir: 目标图像保存目录
        num_samples: 生成的样本数量
    """
    os.makedirs(input_dir, exist_ok=True)  # 创建输入目录
    os.makedirs(target_dir, exist_ok=True)  # 创建目标目录
    
    print(f"开始生成{num_samples}个虚拟样本...")
    
    for i in range(num_samples):  # 生成指定数量的样本
        # ==================== 生成基础条纹图案 ====================
        size = 512  # 图像尺寸
        x, y = np.meshgrid(np.linspace(0, 2*np.pi, size), np.linspace(0, 2*np.pi, size))  # 创建坐标网格
        
        # 基础正弦条纹图案（模拟真实条纹投影）
        base_pattern = 0.5 + 0.3 * np.sin(10*x + 5*y)  # 基础正弦条纹
        noise = np.random.normal(0, 0.05, (size, size))  # 添加高斯噪声（模拟真实环境）
        
        # 稀疏反射效果（模拟高反射表面的镜面反射）
        reflection = np.random.uniform(0, 0.4, (size, size)) * (np.random.random((size, size)) > 0.8)
        
        # ==================== 生成输入和目标图像 ====================
        input_pattern = np.clip(base_pattern + noise + reflection, 0, 1)  # 高反射输入（包含镜面反射）
        target_pattern = np.clip(base_pattern + noise, 0, 1)              # 漫反射目标（无镜面反射）
        
        # 保存图像
        input_img = Image.fromarray((input_pattern * 255).astype(np.uint8))
        target_img = Image.fromarray((target_pattern * 255).astype(np.uint8))
        input_img.save(os.path.join(input_dir, f'input_{i:04d}.png'))
        target_img.save(os.path.join(target_dir, f'target_{i:04d}.png'))
    
    print(f"虚拟数据集生成完成！共{num_samples}个样本")
    print(f"输入图像保存在: {input_dir}")
    print(f"目标图像保存在: {target_dir}")

def main():
    """
    主函数：执行完整的训练、测试和可视化流程
    """
    print("=" * 60)
    print("Y-FFC Net 训练程序启动")
    print("=" * 60)
    
    # ==================== 训练参数配置区域 ====================
    # 在这里修改训练参数，无需深入代码内部
    
    # ========== 基础训练参数 ==========
    EPOCHS = 300
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 30
    GRADIENT_ACCUMULATION_STEPS = 3
    
    # ========== 优化器参数 ==========
    OPTIMIZER_TYPE = 'adam'
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8
    ADAM_WEIGHT_DECAY = 0.0
    SGD_MOMENTUM = 0.9
    SGD_WEIGHT_DECAY = 0.0001
    
    # ========== 学习率调度器参数 ==========
    SCHEDULER_TYPE = 'step'
    STEP_SIZE = 30
    GAMMA = 0.5
    T_MAX = 50
    ETA_MIN = 1e-6
    PATIENCE = 10
    PLATEAU_FACTOR = 0.5
    
    # ========== 损失函数参数 ==========
    SSIM_WINDOW_SIZE = 11
    SSIM_SIGMA = 1.5
    # [修复 P8]
    LOSS_WEIGHT_DIFFUSE  = 1.0
    LOSS_WEIGHT_SPECULAR = 0.8
    LOSS_WEIGHT_GRADIENT = 0.5
    
    # ========== 图像和数据参数 ==========
    IMAGE_RESIZE = (160, 128)
    TRAIN_SPLIT_RATIO = 0.8
    NUM_WORKERS = 0
    PIN_MEMORY = True
    SHUFFLE_TRAIN = True
    
    # ========== 训练策略参数 ==========
    USE_AMP = True
    # [修复 P5]
    USE_GRAD_CLIP = True
    GRAD_CLIP_VALUE = 1.0
    USE_EARLY_STOPPING = False
    EARLY_STOPPING_PATIENCE = 20
    
    # ==================== 模型保存和可视化 ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FOLDER = f'experiments/original_yffc_{timestamp}'
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/models", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/logs", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/visualizations", exist_ok=True)
    
    print(f"\n 实验输出目录: {os.path.abspath(OUTPUT_FOLDER)}")
    print("   所有训练结果将保存在此文件夹中\n")
    
    MODEL_SAVE_PATH     = 'best_model.pth'
    CHECKPOINT_DIR      = 'checkpoints'
    SAVE_EVERY_N_EPOCHS = 5
    VISUALIZATION_SAMPLES = None
    
    MODEL_SAVE_PATH = os.path.join(OUTPUT_FOLDER, 'models', MODEL_SAVE_PATH)
    CHECKPOINT_DIR  = os.path.join(OUTPUT_FOLDER, 'checkpoints')

    _model_dir  = os.path.dirname(os.path.abspath(MODEL_SAVE_PATH))
    _model_stem = os.path.splitext(os.path.basename(MODEL_SAVE_PATH))[0]
    MODEL_SAVE_PATH_SSIM = os.path.join(_model_dir, f'{_model_stem}_best_ssim.pth')
    MODEL_SAVE_PATH_LOSS = os.path.join(_model_dir, f'{_model_stem}_best_loss.pth')
    
    print("\n当前训练配置:")
    print(f"  - 训练轮数: {EPOCHS}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - 梯度累积步数: {GRADIENT_ACCUMULATION_STEPS} (等效batch_size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  - 图像尺寸: {IMAGE_RESIZE if IMAGE_RESIZE else '原始尺寸'}")
    print(f"  - 训练集比例: {TRAIN_SPLIT_RATIO}")
    print(f"  - 输出文件夹: {OUTPUT_FOLDER}")
    print(f"  - 混合精度训练: {'启用' if USE_AMP else '禁用'}")
    print("=" * 60)
    
    # ==================== 设备设置 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # ==================== 数据路径设置 ====================
    input_dir  = 'data/inputs'
    target_dir = 'data/targets'
    
    if not os.path.exists(input_dir) or not os.path.exists(target_dir):
        print("检测到数据目录不存在，正在生成虚拟数据集...")
        create_dummy_data(input_dir, target_dir, num_samples=500)
    
    # ==================== 数据集构建与划分 ====================
    print("构建数据集...")
    train_dataset = FringeDataset(input_dir, target_dir, resize=IMAGE_RESIZE)
    train_size = int(TRAIN_SPLIT_RATIO * len(train_dataset))
    val_size   = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])
    
    print(f"数据集大小: 训练集 {train_size}, 验证集 {val_size}")
    if IMAGE_RESIZE:
        print(f"图像已resize到: {IMAGE_RESIZE if isinstance(IMAGE_RESIZE, int) else f'{IMAGE_RESIZE[0]}x{IMAGE_RESIZE[1]}'}")
    
    # ==================== 数据加载器创建 ====================
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(val_dataset,   batch_size=1,          shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    print(f"批次大小: {BATCH_SIZE}, 梯度累积步数: {GRADIENT_ACCUMULATION_STEPS} (等效batch_size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    
    # ==================== 数据维度检查 ====================
    print("检查数据维度...")
    try:
        sample_batch = next(iter(train_loader))
        print('样本批次形状:', [tuple(x.shape) for x in sample_batch])
        print('数据格式: [batch_size, channels, height, width]')
    except Exception as e:
        print('获取样本批次失败:', e)
        return
    
    # ==================== 模型创建 ====================
    print("创建Y-FFC网络模型...")
    model = YFFC()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    # ==================== 准备配置字典 ====================
    optimizer_config = {
        'type':         OPTIMIZER_TYPE,
        'betas':        ADAM_BETAS,
        'eps':          ADAM_EPS,
        'weight_decay': ADAM_WEIGHT_DECAY,
        'momentum':     SGD_MOMENTUM,
    }
    
    scheduler_config = {
        'type':      SCHEDULER_TYPE,
        'step_size': STEP_SIZE,
        'gamma':     GAMMA,
        'T_max':     T_MAX,
        'eta_min':   ETA_MIN,
        'patience':  PATIENCE,
        'factor':    PLATEAU_FACTOR,
    }
    
    loss_config = {
        'ssim_window_size': SSIM_WINDOW_SIZE,
        'ssim_sigma':       SSIM_SIGMA,
        'weight_diffuse':   LOSS_WEIGHT_DIFFUSE,
        'weight_specular':  LOSS_WEIGHT_SPECULAR,
        'weight_gradient':  LOSS_WEIGHT_GRADIENT,
    }
    
    training_config = {
        'use_grad_clip':          USE_GRAD_CLIP,
        'grad_clip_value':        GRAD_CLIP_VALUE,
        'use_early_stopping':     USE_EARLY_STOPPING,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'model_save_path':        MODEL_SAVE_PATH,
        'checkpoint_dir':         CHECKPOINT_DIR,
        'save_every_n_epochs':    SAVE_EVERY_N_EPOCHS,
    }
    
    # ==================== 模型训练 ====================
    print("\n" + "=" * 40)
    print("开始模型训练...")
    print("=" * 40)
    # train_model 现在返回 5 个值：前 4 个与原来相同（向后兼容），
    # 第 5 个 history 字典包含 MAE 历史及最优模型 epoch 信息。
    train_losses, val_losses, train_accuracies, val_accuracies, history = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        use_amp=USE_AMP,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        loss_config=loss_config,
        training_config=training_config,
    )
    
    # ==================== 模型测试 ====================
    print("\n" + "=" * 40)
    print("开始模型测试...")
    print("=" * 40)
    test_load_path = MODEL_SAVE_PATH_SSIM if os.path.exists(MODEL_SAVE_PATH_SSIM) else MODEL_SAVE_PATH
    print(f"加载测试模型（SSIM最优）: {test_load_path}")
    model.load_state_dict(torch.load(test_load_path, map_location=device))
    test_model(model, test_loader, device=device)
    
    # ==================== 结果可视化 ====================
    print("\n" + "=" * 40)
    print("开始结果可视化...")
    print("=" * 40)
    visualize_results(model, test_loader, device=device,
                      num_samples=VISUALIZATION_SAMPLES, output_folder=OUTPUT_FOLDER)

    # ──────────────────────────────────────────────────────────────────────────
    # 【新增】训练历史曲线：调用独立函数 plot_training_history()
    # 包含 Loss / SSIM / MAE 三组指标，共 3×2 = 6 张子图。
    # 保证即使前面步骤（测试、可视化）抛出异常也能单独调用此函数重新生成。
    # ──────────────────────────────────────────────────────────────────────────
    print("\n保存最终训练历史曲线（Loss / SSIM / MAE）...")
    history_path = plot_training_history(
        train_losses = history['train_losses'],
        val_losses   = history['val_losses'],
        train_ssims  = history['train_ssims'],
        val_ssims    = history['val_ssims'],
        train_maes   = history['train_maes'],
        val_maes     = history['val_maes'],
        output_folder = OUTPUT_FOLDER,
        total_epochs  = history['actual_epochs'],
        # 【新增 MSE】传入 MSE 历史，plot_training_history 会在第 4 行绘制 MSE 曲线
        train_mses   = history['train_mses'],
        val_mses     = history['val_mses'],
    )
    print(f"训练历史曲线已保存: {history_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # 【新增】训练总结文件：调用独立函数 save_training_summary()
    # 包含训练配置、三个最优模型的 epoch/Loss/SSIM/MAE、最终 epoch 指标。
    # ──────────────────────────────────────────────────────────────────────────
    print("生成训练总结文件（training_summary.txt）...")
    summary_path = save_training_summary(
        output_folder = OUTPUT_FOLDER,
        # 训练基本信息
        total_epochs                = history['actual_epochs'],
        batch_size                  = BATCH_SIZE,
        learning_rate               = LEARNING_RATE,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
        image_resize                = IMAGE_RESIZE,
        optimizer_type              = OPTIMIZER_TYPE,
        scheduler_type              = SCHEDULER_TYPE,
        use_amp                     = USE_AMP,
        loss_weight_diffuse         = LOSS_WEIGHT_DIFFUSE,
        loss_weight_specular        = LOSS_WEIGHT_SPECULAR,
        loss_weight_gradient        = LOSS_WEIGHT_GRADIENT,
        # 最优模型信息（来自 history，含 epoch 编号）
        best_composite_info = history['best_composite_info'],
        best_ssim_info      = history['best_ssim_info'],
        best_loss_info      = history['best_loss_info'],
        # 每 epoch 历史
        train_losses = history['train_losses'],
        val_losses   = history['val_losses'],
        train_ssims  = history['train_ssims'],
        val_ssims    = history['val_ssims'],
        train_maes   = history['train_maes'],
        val_maes     = history['val_maes'],
        # 【新增 MSE】传入 MSE 历史，summary 文件的 Final Epoch Result 段会写入 MSE
        train_mses   = history['train_mses'],
        val_mses     = history['val_mses'],
        # 总训练时长
        total_time_str = history['total_time_str'],
        # 检查点配置
        checkpoint_dir      = CHECKPOINT_DIR,
        save_every_n_epochs = SAVE_EVERY_N_EPOCHS,
    )
    print(f"训练总结文件已保存: {summary_path}")
    
    # ==================== 打印最终文件结构 ====================
    print("\n" + "=" * 60)
    print("✅ 所有流程完成！")
    print("=" * 60)
    print(f"\n📁 所有结果已保存到: {os.path.abspath(OUTPUT_FOLDER)}")
    print("\n文件结构:")
    print(f"  {OUTPUT_FOLDER}/")
    print(f"  ├── models/                        (模型文件)")
    print(f"  │   ├── best_model.pth             (综合评分最优模型)")
    print(f"  │   ├── best_model_best_ssim.pth   (SSIM最优模型)")
    print(f"  │   └── best_model_best_loss.pth   (Loss最优模型)")
    print(f"  ├── logs/                          (日志文件)")
    print(f"  ├── visualizations/                (可视化结果)")
    print(f"  ├── training_history.png           (训练曲线: Loss/SSIM/MAE)")
    print(f"  └── training_summary.txt           (训练总结: 配置/最优Epoch/最终指标)")
    if SAVE_EVERY_N_EPOCHS > 0:
        print(f"  └── checkpoints/                  (训练检查点)")
    print("=" * 60)

if __name__ == '__main__':  # 脚本入口
    # ========== Matplotlib 字体设置 ==========
    # 解决中文显示为乱码以及坐标轴负号显示问题
    # Windows 常见可用中文字体：Microsoft YaHei / SimHei
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
    rcParams['axes.unicode_minus'] = False
    main()  # 运行主流程
