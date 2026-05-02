"""Console and file reporting helpers for training."""

import os
from datetime import datetime

import torch

from .metrics import SUB_LOSS_DISPLAY_NAMES


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
            f.write(f'  Val Loss          : {info["loss"]:.6f}\n')
            f.write(f'  Val SSIM          : {info["ssim"]:.4f}\n')
            f.write(f'  Val MAE           : {info["mae"]:.6f}\n')
            if 'mse' in info and info['mse'] is not None:
                f.write(f'  Val MSE           : {info["mse"]:.6f}\n')
            if 'psnr' in info and info['psnr'] is not None:
                f.write(f'  Val PSNR          : {info["psnr"]:.2f} dB\n')
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
        if final_vl_mse is not None:
            f.write(f'  Val   MSE         : {final_vl_mse:.6f}\n')
        if final_vl_psnr is not None:
            f.write(f'  Val   PSNR        : {final_vl_psnr:.2f} dB\n')

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
