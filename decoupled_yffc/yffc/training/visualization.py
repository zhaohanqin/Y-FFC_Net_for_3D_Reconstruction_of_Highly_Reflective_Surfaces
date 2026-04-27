"""Training visualization helpers."""

import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import SUB_LOSS_DISPLAY_NAMES


@torch.no_grad()
def visualize_sample_during_training(model, val_loader, device, output_folder,
                                      current_epoch, num_samples=2):
    """
    训练过程中可视化验证样本结果。

    布局：每个样本一行 2×3 子图
        第一行：输入图像 | GT 漫反射 | 预测漫反射
        第二行：预测镜面反射 | 预测梯度 | 差异图（热力图）

    参数:
        model          : 当前模型
        val_loader     : 验证数据加载器
        device         : 计算设备
        output_folder  : 输出目录
        current_epoch  : 当前 epoch 编号
        num_samples    : 可视化样本数（默认 2）
    """
    model.eval()
    vis_dir = os.path.join(output_folder, 'val_visualization')
    os.makedirs(vis_dir, exist_ok=True)

    batch = next(iter(val_loader))
    inputs, gt_diffuse, gt_specular, gt_gradient = [x.to(device) for x in batch]

    estimations = model(inputs)
    pred_diffuse = estimations['r']
    pred_specular = estimations['b']
    pred_gradient = estimations['g']

    actual_samples = min(num_samples, inputs.size(0))

    for s in range(actual_samples):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        input_img = inputs[s, 0].cpu().numpy()
        gt_diff_img = gt_diffuse[s, 0].cpu().numpy()
        pred_diff_img = pred_diffuse[s, 0].cpu().numpy()
        pred_spec_img = pred_specular[s, 0].cpu().numpy()
        pred_grad_img = pred_gradient[s, 0].cpu().numpy()

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

        # 第二行
        axes[1, 0].imshow(pred_spec_img, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Pred Specular', fontsize=11)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(pred_grad_img, cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('Pred Gradient', fontsize=11)
        axes[1, 1].axis('off')

        diff = np.abs(pred_diff_img - gt_diff_img)
        im = axes[1, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[1, 2].set_title('|Pred - GT| Difference', fontsize=11)
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

        # 计算当前样本指标
        sample_mae = np.mean(diff)
        sample_mse = np.mean(diff ** 2)
        sample_max_err = np.max(diff)

        fig.suptitle(
            f'Epoch {current_epoch} - Sample {s+1}  |  '
            f'MAE={sample_mae:.4f}  MSE={sample_mse:.6f}  MaxErr={sample_max_err:.4f}',
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

    seed=None 时使用当前时间生成种子；训练开始后样本索引固定，
    便于同一次训练中跨 epoch 对比可视化结果。
    """
    dataset_size = len(val_dataset)
    if dataset_size <= 0:
        raise ValueError("验证集为空，无法抽取可视化样本。")

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
