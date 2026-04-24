"""Enhanced Y-FFC training orchestration."""

import os
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import EnhancedPairedTransform, HighlightAugmentation, ImprovedFringeDataset
from ..losses import EnhancedCompositeLoss
from ..models import YFFC
from .loops import train_one_epoch, validate
from .metrics import SUB_LOSS_DISPLAY_NAMES, SUB_LOSS_KEYS
from .optimization import ImprovedWeightScheduler, build_optimizer_v2
from .reporting import print_epoch_log, save_training_summary
from .utils import EMA, build_scheduler, replace_bn_with_gn
from .visualization import (
    plot_individual_curves,
    plot_sub_loss_history,
    plot_training_curves_realtime,
    plot_training_history,
    visualize_sample_during_training,
)


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
    model = YFFC()
    if cfg['USE_GN']:
        print("[GN] BN -> GN 替换")
        replace_bn_with_gn(model, 8)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")
    print(f"  可训练: {trainable_params:,}  冻结: {total_params - trainable_params:,}")

    # ══════════ 2. 数据集 ══════════
    min_side = min(img_hw)  # 取短边计算高光半径
    hl_aug = HighlightAugmentation(
        p=0.4, num_range=(1, 6),
        radius_range=(3, max(5, min_side // 20)),
        intensity_range=(0.2, 0.7))
    aug = EnhancedPairedTransform(highlight_aug=hl_aug)

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

    # ══════════ 3. 损失函数 ══════════
    criterion = EnhancedCompositeLoss(
        use_perceptual=True, use_ohem=cfg['USE_OHEM']).to(device)
    ws = ImprovedWeightScheduler(cfg['EPOCHS'])

    # 打印损失函数结构
    print(f"\n损失函数: EnhancedCompositeLoss")
    print(f"  use_perceptual={True}, use_ohem={cfg['USE_OHEM']}")
    print(f"  子损失项: {', '.join(SUB_LOSS_KEYS)}")

    # ══════════ 4. 优化器 + 调度器 ══════════
    print("\n分层学习率配置:")
    optimizer = build_optimizer_v2(model, cfg['BASE_LR'], cfg['WEIGHT_DECAY'])
    scheduler = build_scheduler(
        optimizer,
        warmup_iters=min(500, len(train_loader) * 3),
        total_epochs=cfg['EPOCHS'],
        iters_per_epoch=len(train_loader))

    # ══════════ 5. AMP + EMA ══════════
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
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

    best_info = {
        'Best Overall (SSIM×0.7 + NormLoss×0.3)': {
            'epoch': 0, 'loss': float('inf'), 'ssim': 0.0, 'mae': float('inf'),
            'mse': None, 'psnr': None, 'path': best_composite_path},
        'Best SSIM': {
            'epoch': 0, 'loss': float('inf'), 'ssim': 0.0, 'mae': float('inf'),
            'mse': None, 'psnr': None, 'path': best_ssim_path},
        'Best Loss': {
            'epoch': 0, 'loss': float('inf'), 'ssim': 0.0, 'mae': float('inf'),
            'mse': None, 'psnr': None, 'path': best_loss_path},
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

    # ══════════ 可视化间隔配置 ══════════
    vis_interval = cfg.get('VIS_EVERY_N_EPOCHS', 10)  # 默认每 10 轮可视化一次

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
            epoch=ep, total_epochs=cfg['EPOCHS'])

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
        current_lr = optimizer.param_groups[0]['lr']

        # [FIX] LR 回弹检测：若 LR 比上一 epoch 增加超过 50%，发出警告
        if ep > 0 and hasattr(train_model, '_prev_lr'):
            if current_lr > train_model._prev_lr * 1.5 and current_lr > 1e-5:
                print(f"  ⚠️ [LR 异常] LR 从 {train_model._prev_lr:.2e} 跳升至 {current_lr:.2e}！")
        train_model._prev_lr = current_lr

        # ── 多指标最优模型保存 ──
        composite = COMPOSITE_ALPHA * vs + (1.0 - COMPOSITE_ALPHA) * (1.0 / (1.0 + vl))

        # ── 结构化日志 ──
        print_epoch_log(
            ep, cfg['EPOCHS'], epoch_time, avg_epoch_time, eta,
            tl, ts, tm, train_mse, train_psnr,
            vl, vs, vm, val_mse, val_psnr,
            train_subs, val_subs, w, current_lr, device,
            composite, best_val_ssim, best_val_loss)

        # ── 模型保存逻辑 ──
        improved_flags = []

        def _save_best(state_dict, path):
            sd = {
                'epoch': ep + 1,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ssim': vs,
                'val_loss': vl,
            }
            if ema:
                sd['ema'] = ema.shadow
            if scaler is not None:
                sd['scaler_state_dict'] = scaler.state_dict()
            torch.save(sd, path)

        # 策略 A：综合最优
        if composite > best_composite:
            best_composite = composite
            early_stopping_counter = 0
            _save_best(model.state_dict(), best_composite_path)
            improved_flags.append(f'综合↑{composite:.4f}')
            best_info['Best Overall (SSIM×0.7 + NormLoss×0.3)'].update(
                {'epoch': ep + 1, 'loss': vl, 'ssim': vs, 'mae': vm,
                 'mse': val_mse, 'psnr': val_psnr})
        else:
            early_stopping_counter += 1

        # 策略 B：SSIM 最优
        if vs > best_val_ssim:
            best_val_ssim = vs
            _save_best(model.state_dict(), best_ssim_path)
            improved_flags.append(f'SSIM↑{vs:.4f}')
            best_info['Best SSIM'].update(
                {'epoch': ep + 1, 'loss': vl, 'ssim': vs, 'mae': vm,
                 'mse': val_mse, 'psnr': val_psnr})

        # 策略 C：Loss 最优
        if vl < best_val_loss:
            best_val_loss = vl
            _save_best(model.state_dict(), best_loss_path)
            improved_flags.append(f'Loss↓{vl:.6f}')
            best_info['Best Loss'].update(
                {'epoch': ep + 1, 'loss': vl, 'ssim': vs, 'mae': vm,
                 'mse': val_mse, 'psnr': val_psnr})

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
        if (ep + 1) % vis_interval == 0 or ep == 0:
            if ema:
                ema.apply_shadow()
            vis_path = visualize_sample_during_training(
                model, val_loader, device, output_folder,
                current_epoch=ep + 1, num_samples=2)
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
