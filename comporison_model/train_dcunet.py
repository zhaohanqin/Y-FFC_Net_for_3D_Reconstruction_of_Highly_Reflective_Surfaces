"""
train_dcunet.py — DC-UNet 高反光去除模型训练脚本（改进版）
==========================================================

相对旧版的改进（解决你提出的全部问题）：
  ① SSIM 虚高     → 统一改用 compare_common.EvalSSIM（标准 SSIM，不做逐像素下限
                     截断），与 HDRNet、主模型口径完全一致，可横向比较。
  ② 训练后期崩溃   → 引入 FiniteStateGuard：检测到 输出/损失/梯度 非有限时，回滚到
                     最近一次“全有限”的参数快照并清理 Adam 动量，使训练可恢复，
                     不再出现“一旦 NaN 就永久 nan”的不可逆崩溃；AMP 默认关闭。
  ③ 结果可视化     → 支持 VIS_EVERY_N_EPOCHS / VIS_NUM_SAMPLES，按指定轮数对验证集
                     结果作图（按 DC-UNet 实际输出：去反光漫反射图）。
  ④ 新增 PSNR     → 训练/验证逐轮输出 PSNR，并写入曲线与总结，MSE/SSIM/PSNR 与主模型对齐。

DC-UNet 只输出去反光后的图像（tanh 激活，范围 [-1,1]），无镜面分量输出，
训练损失仅针对预测图像与目标漫反射图像之间的差异计算：loss = (1-SSIM) + MAE。
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from DCUNet import DC_Unet
from Y_FFC_unified import FringeDataset
import compare_common as cc


# ==============================================================================
# 模型输出 → [0,1] 映射（DC-UNet 为 tanh 输出）
# ==============================================================================

def dcunet_pred_map(raw_output):
    """tanh 输出 [-1,1] → [0,1]，与 [0,1] 目标对齐。"""
    return ((raw_output + 1.0) / 2.0).clamp(0.0, 1.0)


def compute_loss(pred_raw, target, ssim_fn, mae_fn):
    """返回 (total_loss, ssim_metric, mae, mse)。

    ssim_metric 使用标准 SSIM（截断 [0,1] 仅作显示），与主模型 / HDRNet 口径一致。
    """
    pred_01 = dcunet_pred_map(pred_raw)
    ssim_raw = ssim_fn(pred_01, target)            # 可反传标量（原始 map 均值）
    loss = (1.0 - ssim_raw) + mae_fn(pred_01, target)
    with torch.no_grad():
        ssim_val = cc.ssim_to_metric(ssim_raw)
        mae_val = torch.mean(torch.abs(pred_01 - target)).item()
        mse_val = torch.mean((pred_01 - target) ** 2).item()
    return loss, ssim_val, mae_val, mse_val


# ==============================================================================
# 单轮训练 / 验证
# ==============================================================================

def train_one_epoch(model, loader, optimizer, ssim_fn, mae_fn, device,
                    use_amp, scaler, guard, grad_clip=1.0):
    model.train()
    tracker = cc.MetricTracker()

    for inputs, gt_diffuse, *_ in tqdm(loader, leave=False, desc='Train'):
        inputs = inputs.to(device)
        gt_diffuse = gt_diffuse.to(device)
        if not (torch.isfinite(inputs).all() and torch.isfinite(gt_diffuse).all()):
            continue

        optimizer.zero_grad(set_to_none=True)

        amp_on = bool(use_amp) and scaler is not None
        with torch.amp.autocast('cuda', enabled=amp_on):
            pred = model(inputs)
        if not torch.isfinite(pred).all():
            r = guard.restore(model, optimizer)
            tqdm.write(f'[WARN] output NaN/Inf, 回滚 {r} 项参数并跳过')
            optimizer.zero_grad(set_to_none=True)
            continue

        with torch.amp.autocast('cuda', enabled=False):
            loss, ssim_val, mae_val, mse_val = compute_loss(
                pred.float(), gt_diffuse.float(), ssim_fn, mae_fn)

        if not torch.isfinite(loss):
            r = guard.restore(model, optimizer)
            tqdm.write(f'[WARN] loss NaN/Inf, 回滚 {r} 项参数并跳过')
            optimizer.zero_grad(set_to_none=True)
            continue

        # 损失尖峰保护
        if tracker.n > 5:
            avg = tracker.tot_loss / tracker.n
            if avg > 0 and loss.item() > avg * 10.0:
                tqdm.write(f'[WARN] loss spike {loss.item():.4f} >> avg {avg:.4f}, 跳过')
                optimizer.zero_grad(set_to_none=True)
                continue

        if amp_on:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                                 grad_clip if grad_clip else 1e9,
                                                 error_if_nonfinite=False)
            if not torch.isfinite(grad_norm):
                r = guard.restore(model, optimizer)
                tqdm.write(f'[WARN] grad NaN/Inf, 回滚 {r} 项参数并跳过 step')
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                                 grad_clip if grad_clip else 1e9,
                                                 error_if_nonfinite=False)
            if not torch.isfinite(grad_norm):
                r = guard.restore(model, optimizer)
                tqdm.write(f'[WARN] grad NaN/Inf, 回滚 {r} 项参数并跳过 step')
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

        tracker.update(loss.item(), ssim_val, mae_val, mse_val)

    return tracker.compute()


@torch.no_grad()
def validate_one_epoch(model, loader, ssim_fn, mae_fn, device, use_amp, scaler):
    model.eval()
    tracker = cc.MetricTracker()
    amp_on = bool(use_amp) and scaler is not None
    for inputs, gt_diffuse, *_ in tqdm(loader, leave=False, desc='Val  '):
        inputs = inputs.to(device)
        gt_diffuse = gt_diffuse.to(device)
        with torch.amp.autocast('cuda', enabled=amp_on):
            pred = model(inputs)
        if not torch.isfinite(pred).all():
            continue
        loss, ssim_val, mae_val, mse_val = compute_loss(
            pred.float(), gt_diffuse.float(), ssim_fn, mae_fn)
        if not torch.isfinite(loss):
            continue
        tracker.update(loss.item(), ssim_val, mae_val, mse_val)
    if tracker.valid_batches == 0:
        return (float('nan'),) * 5
    return tracker.compute()


# ==============================================================================
# 主训练流程
# ==============================================================================

def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    output_folder = cfg['OUTPUT_FOLDER']
    model_save_path = os.path.join(output_folder, 'models', 'best_model.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # ── 数据集 ──
    dataset = FringeDataset(cfg['INPUT_DIR'], cfg['TARGET_DIR'], resize=cfg['IMAGE_RESIZE'])
    train_size = int(cfg['TRAIN_SPLIT'] * len(dataset))
    val_size = len(dataset) - train_size
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g)
    print(f"数据集: 训练 {train_size} / 验证 {val_size}")

    train_loader = DataLoader(train_ds, batch_size=cfg['BATCH_SIZE'], shuffle=True,
                              num_workers=cfg['NUM_WORKERS'], pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=cfg['BATCH_SIZE'], shuffle=False,
                            num_workers=cfg['NUM_WORKERS'], pin_memory=(device.type == 'cuda'))

    # ── 可视化固定样本 ──
    vis_loader = None
    if cfg['VIS_EVERY_N_EPOCHS'] > 0 and cfg['VIS_NUM_SAMPLES'] > 0:
        vis_loader, vis_idx, vis_seed = cc.build_fixed_vis_loader(
            val_ds, cfg['VIS_NUM_SAMPLES'], cfg['NUM_WORKERS'],
            pin_memory=(device.type == 'cuda'), seed=cfg['VIS_SAMPLE_SEED'])
        print(f"可视化样本: 固定抽取 {len(vis_idx)} 张 (seed={vis_seed}, idx={vis_idx})")

    # ── 模型 / 损失 / 优化器 ──
    model = DC_Unet(in_channels=cfg['IN_CHANNELS'], out_channels=cfg['OUT_CHANNELS'],
                    nf=cfg['NF'], use_dropout=cfg['USE_DROPOUT']).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    ssim_fn = cc.EvalSSIM(window_size=11, sigma=1.5, channels=cfg['OUT_CHANNELS']).to(device)
    mae_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['LR'], betas=(0.9, 0.999))

    if cfg['SCHEDULER'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['STEP_SIZE'], gamma=cfg['GAMMA'])
    elif cfg['SCHEDULER'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['EPOCHS'], eta_min=1e-6)
    elif cfg['SCHEDULER'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg['GAMMA'], patience=10)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler('cuda') if cfg['USE_AMP'] and device.type == 'cuda' else None
    guard = cc.FiniteStateGuard(model)

    history = {k: [] for k in ['train_losses', 'val_losses', 'train_ssims', 'val_ssims',
                               'train_maes', 'val_maes', 'train_mses', 'val_mses',
                               'train_psnrs', 'val_psnrs']}
    best = {'epoch': 0, 'loss': float('inf'), 'ssim': float('nan'),
            'mae': float('nan'), 'mse': float('nan'), 'psnr': float('nan'),
            'train_loss': float('nan'), 'train_ssim': float('nan'),
            'train_mae': float('nan'), 'train_mse': float('nan'),
            'train_psnr': float('nan'),
            'path': model_save_path}
    epoch_times = []

    print(f"\n{'='*70}\n开始训练 epochs={cfg['EPOCHS']} lr={cfg['LR']} batch={cfg['BATCH_SIZE']} amp={cfg['USE_AMP']}\n{'='*70}\n")
    start = time.time()

    for epoch in range(1, cfg['EPOCHS'] + 1):
        t0 = time.time()
        if cc.model_is_finite(model):
            guard.snapshot(model)   # 每轮训练前刷新“最近全有限”快照

        tr = train_one_epoch(model, train_loader, optimizer, ssim_fn, mae_fn,
                             device, cfg['USE_AMP'], scaler, guard, grad_clip=cfg['GRAD_CLIP'])
        vl = validate_one_epoch(model, val_loader, ssim_fn, mae_fn, device, cfg['USE_AMP'], scaler)

        if scheduler is not None:
            if cfg['SCHEDULER'] == 'plateau':
                scheduler.step(vl[0])
            else:
                scheduler.step()

        for key, val in zip(['train_losses', 'train_ssims', 'train_maes', 'train_mses', 'train_psnrs'], tr):
            history[key].append(val)
        for key, val in zip(['val_losses', 'val_ssims', 'val_maes', 'val_mses', 'val_psnrs'], vl):
            history[key].append(val)

        dt = time.time() - t0
        epoch_times.append(dt)
        eta = str(timedelta(seconds=int(np.mean(epoch_times) * (cfg['EPOCHS'] - epoch))))
        lr_now = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch:>3d}/{cfg['EPOCHS']}]  耗时 {dt:.1f}s  剩余 {eta}  lr={lr_now:.2e}")
        print(f"  Train | Loss {tr[0]:.6f}  SSIM {tr[1]:.4f}  MAE {tr[2]:.6f}  MSE {tr[3]:.6f}  PSNR {tr[4]:.2f}dB")
        print(f"  Val   | Loss {vl[0]:.6f}  SSIM {vl[1]:.4f}  MAE {vl[2]:.6f}  MSE {vl[3]:.6f}  PSNR {vl[4]:.2f}dB")

        if np.isfinite(vl[0]) and vl[0] < best['loss']:
            best.update({'epoch': epoch, 'loss': vl[0], 'ssim': vl[1],
                         'mae': vl[2], 'mse': vl[3], 'psnr': vl[4],
                         'train_loss': tr[0], 'train_ssim': tr[1],
                         'train_mae': tr[2], 'train_mse': tr[3], 'train_psnr': tr[4]})
            torch.save({'epoch': epoch, 'model_name': 'dcunet',
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'image_resize': cfg['IMAGE_RESIZE'],
                        'val_loss': vl[0], 'val_ssim': vl[1], 'val_mae': vl[2],
                        'val_mse': vl[3], 'val_psnr': vl[4],
                        'model_config': {'in_channels': cfg['IN_CHANNELS'],
                                         'out_channels': cfg['OUT_CHANNELS'],
                                         'nf': cfg['NF'], 'use_dropout': cfg['USE_DROPOUT']}},
                       model_save_path)
            print(f"  ✅ 最优模型已保存 (Val Loss↓ {best['loss']:.6f})")
        else:
            print(f"     无改善（最优 Epoch {best['epoch']}, Loss {best['loss']:.6f}）")

        if (epoch % 5 == 0 or epoch == 1):
            cc.plot_training_curves_realtime(history, output_folder, epoch)

        if vis_loader is not None and (epoch % cfg['VIS_EVERY_N_EPOCHS'] == 0 or epoch == 1):
            vis_dir = cc.visualize_single_output(
                model, vis_loader, device, output_folder, epoch,
                pred_map_fn=dcunet_pred_map, model_title='DC-UNet',
                num_samples=cfg['VIS_NUM_SAMPLES'])
            print(f"  样本可视化已保存: {vis_dir}")
        print()

    total_time = str(timedelta(seconds=int(time.time() - start)))
    print(f"{'='*70}\n训练完成！总时长: {total_time}")
    print(f"最佳: Epoch {best['epoch']}  Loss {best['loss']:.6f}  SSIM {best['ssim']:.4f}  "
          f"MAE {best['mae']:.6f}  MSE {best['mse']:.6f}  PSNR {best['psnr']:.2f}dB\n{'='*70}")

    print("\n保存训练历史曲线 (Loss/SSIM/MAE/MSE/PSNR)...")
    print(f"  -> {cc.plot_training_history(history, output_folder, cfg['EPOCHS'], model_title='DC-UNet')}")
    print("生成训练总结文件...")
    print(f"  -> {cc.save_training_summary(output_folder, 'DC-UNet', cfg, history, best, total_time)}")
    return history


def create_dummy_data(input_dir, target_dir, num_samples=100):
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    print(f"生成 {num_samples} 个虚拟样本...")
    for i in range(num_samples):
        h, w = 128, 160
        base = np.random.rand(h, w).astype(np.float32)
        spec = np.clip(np.random.rand(h, w).astype(np.float32) * 0.3, 0, 1)
        inp = np.clip(base + spec, 0, 1)
        Image.fromarray((inp * 255).astype(np.uint8)).save(os.path.join(input_dir, f'input_{i:04d}.png'))
        Image.fromarray((base * 255).astype(np.uint8)).save(os.path.join(target_dir, f'target_{i:04d}.png'))


def main():
    print("=" * 70)
    print("DC-UNet 训练启动（改进版）")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = {
        # ── 数据 ──
        'INPUT_DIR':  'data2/inputs',
        'TARGET_DIR': 'data2/targets',
        'IMAGE_RESIZE': (128, 160),   # (H, W)
        'TRAIN_SPLIT': 0.8,
        'BATCH_SIZE': 8,
        'NUM_WORKERS': 0,
        # ── 模型 ──
        'IN_CHANNELS': 1, 'OUT_CHANNELS': 1, 'NF': 32, 'USE_DROPOUT': False,
        # ── 训练 ──
        'EPOCHS': 100,
        'LR': 1e-3,
        'USE_AMP': False,          # ★ 稳定起见默认关闭 AMP（与主模型一致）；确认稳定后可改 True
        'GRAD_CLIP': 1.0,
        # ── 调度 ──
        'SCHEDULER': 'cosine', 'STEP_SIZE': 30, 'GAMMA': 0.5,
        # ── 可视化（★ 新增：按指定轮数对验证集结果作图） ──
        'VIS_EVERY_N_EPOCHS': 10,
        'VIS_NUM_SAMPLES': 2,
        'VIS_SAMPLE_SEED': None,
        # ── 输出 ──
        'OUTPUT_FOLDER': f'experiments/dcunet_{timestamp}',
    }

    print("\n当前配置:")
    for k, v in cfg.items():
        print(f"  {k:18s}: {v}")
    print()

    if not os.path.exists(cfg['INPUT_DIR']) or not os.path.exists(cfg['TARGET_DIR']):
        print("数据目录不存在，自动生成虚拟数据集...")
        create_dummy_data(cfg['INPUT_DIR'], cfg['TARGET_DIR'], num_samples=500)

    train(cfg)


if __name__ == '__main__':
    cc.setup_matplotlib_fonts()
    main()
