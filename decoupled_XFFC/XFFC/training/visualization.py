"""Training visualization helpers."""

import os
import time
import math
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import SUB_LOSS_DISPLAY_NAMES


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
