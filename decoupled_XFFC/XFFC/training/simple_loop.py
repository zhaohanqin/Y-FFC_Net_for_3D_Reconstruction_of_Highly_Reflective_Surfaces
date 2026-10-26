"""Original lightweight train/test loop used by the legacy unified CLI."""

import os
import time
import math

import torch
from torch.utils.data import DataLoader

from ..data import EnhancedPairedTransform, FringeDataset, HighlightAugmentation
from ..losses import EnhancedCompositeLoss
from ..models import XFFC
from .utils import (
    DynamicLossWeightScheduler,
    EMA,
    build_optimizer,
    build_scheduler,
    replace_bn_with_gn,
)


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler,
                    scaler, ema, device,
                    accumulation_steps=4, max_grad_norm=0.5,
                    epoch=0, total_epochs=200):
    """单个 epoch 的训练循环（AMP + 梯度累积 + EMA）。"""
    model.train()
    running_loss = 0.0
    running_ssim = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for i, (input_img, target_img, gt_specular, gt_gradient) in enumerate(dataloader):
        input_img = input_img.to(device)
        target_img = target_img.to(device)
        gt_specular = gt_specular.to(device)
        gt_gradient = gt_gradient.to(device)

        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            output = model(input_img)
            pred_diffuse = output['r']
            pred_grad = output['g']
            est_b = output['b']
            pred_s_lik = output.get('s')

            with torch.amp.autocast(device_type='cuda', enabled=False):
                loss, loss_dict = criterion(
                    pred_diffuse.float(), target_img.float(),
                    est_b.float(), gt_specular.float(),
                    pred_grad.float(), gt_gradient.float(),
                    output.get('cs'), input_img=input_img.float(),
                    pred_s_lik=pred_s_lik.float() if pred_s_lik is not None else None)

            loss = loss / accumulation_steps

        # [FIX] NaN / Inf 保护
        if not torch.isfinite(loss):
            print(f'[WARNING] Epoch {epoch+1} Batch {i+1}: loss NaN/Inf, 跳过')
            optimizer.zero_grad()
            continue

        if device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            if device.type == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update()

        running_loss += loss_dict['total']
        running_ssim += (1.0 - loss_dict['ssim'])
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    avg_ssim = running_ssim / max(n_batches, 1)
    return avg_loss, avg_ssim


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """验证循环。"""
    model.eval()
    running_loss = 0.0
    running_ssim = 0.0
    n_batches = 0

    for input_img, target_img, gt_specular, gt_gradient in dataloader:
        input_img = input_img.to(device)
        target_img = target_img.to(device)
        gt_specular = gt_specular.to(device)
        gt_gradient = gt_gradient.to(device)

        output = model(input_img)
        loss, loss_dict = criterion(
            output['r'], target_img, output['b'], gt_specular,
            output['g'], gt_gradient, output.get('cs'), input_img=input_img,
            pred_s_lik=output.get('s'))

        running_loss += loss_dict['total']
        running_ssim += (1.0 - loss_dict['ssim'])
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    avg_ssim = running_ssim / max(n_batches, 1)
    return avg_loss, avg_ssim


def train(args):
    """完整训练入口。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备: {device}")

    model = XFFC()
    print("使用 XFFC 模型")

    if args.use_gn:
        print("将 BatchNorm 替换为 GroupNorm...")
        replace_bn_with_gn(model, num_groups=8)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 数据集
    if args.use_augmentation:
        img_size = args.image_size
        if isinstance(img_size, (tuple, list)):
            min_side = min(int(img_size[0]), int(img_size[1]))
        else:
            min_side = int(img_size)
        hl_aug = HighlightAugmentation(
            p=0.5, num_range=(1, 6),
            radius_range=(3, max(5, min_side // 20)),
            intensity_range=(0.2, 0.7))
        paired_aug = EnhancedPairedTransform(highlight_aug=hl_aug)
        print("[Aug] 已启用 EnhancedPairedTransform + HighlightAugmentation(p=0.5)")
    else:
        paired_aug = None
    full_train_dataset = FringeDataset(args.input_dir, args.target_dir,
                                       resize=args.image_size, paired_transform=paired_aug)
    if args.val_input_dir and args.val_target_dir:
        train_dataset = full_train_dataset
        val_dataset = FringeDataset(args.val_input_dir, args.val_target_dir,
                                    resize=args.image_size, paired_transform=None)
    else:
        full_val_dataset = FringeDataset(args.input_dir, args.target_dir,
                                         resize=args.image_size, paired_transform=None)
        total_size = len(full_train_dataset)
        if total_size < 2:
            raise ValueError("自动划分训练/验证至少需要 2 张图像；请增加数据或提供独立验证集。")
        train_size = int(args.train_split_ratio * total_size)
        train_size = max(1, min(train_size, total_size - 1))
        val_size = total_size - train_size
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_size, generator=generator).tolist()
        train_dataset = torch.utils.data.Subset(full_train_dataset, indices[:train_size])
        val_dataset = torch.utils.data.Subset(full_val_dataset, indices[train_size:])
        print(f"[Dataset] 自动划分训练/验证: train={train_size}, val={val_size}, seed=42")

    if len(train_dataset) < args.batch_size:
        raise ValueError(
            f"训练样本数({len(train_dataset)}) 小于 batch_size({args.batch_size})，"
            "drop_last=True 会导致没有可训练 batch；请调小 batch_size 或增加数据。")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == 'cuda'), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type == 'cuda'))

    # 损失函数
    criterion = EnhancedCompositeLoss(
        use_perceptual=True, use_ohem=args.use_ohem).to(device)
    loss_weight_scheduler = DynamicLossWeightScheduler(total_epochs=args.epochs)
    print(f"[Loss] 已启用 EnhancedCompositeLoss(use_ohem={args.use_ohem})")

    # 优化器 + 调度器
    optimizer = build_optimizer(model, base_lr=args.base_lr, weight_decay=args.weight_decay)
    iters_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    if iters_per_epoch == 0:
        raise ValueError("训练 DataLoader 为空，请检查数据量、batch_size 和 drop_last 设置。")
    scheduler = build_scheduler(optimizer, warmup_iters=min(500, iters_per_epoch * 3),
                                total_epochs=args.epochs, iters_per_epoch=iters_per_epoch)

    scaler = torch.amp.GradScaler(device='cuda', enabled=(device.type == 'cuda'))
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    best_val_ssim = 0.0
    print(f"\n开始训练: {args.epochs} epochs, batch_size={args.batch_size}")
    print("=" * 70)

    for epoch in range(args.epochs):
        t0 = time.time()
        weights = loss_weight_scheduler.step(criterion, epoch)

        train_loss, train_ssim = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, ema, device, accumulation_steps=args.accumulation_steps,
            epoch=epoch, total_epochs=args.epochs)

        elapsed = time.time() - t0

        val_loss, val_ssim = 0.0, 0.0
        if val_loader is not None:
            if ema is not None:
                ema.apply_shadow()
            val_loss, val_ssim = validate(model, val_loader, criterion, device)
            if ema is not None:
                ema.restore()

        current_lr = optimizer.param_groups[0]['lr']
        log_msg = f"Epoch [{epoch + 1:03d}/{args.epochs}] train_loss={train_loss:.4f} train_SSIM={train_ssim:.4f} "
        if val_loader is not None:
            log_msg += f"val_loss={val_loss:.4f} val_SSIM={val_ssim:.4f} "
        log_msg += f"lr={current_lr:.2e} time={elapsed:.1f}s"
        print(log_msg)

        save_ssim = val_ssim if val_loader is not None else train_ssim
        if save_ssim > best_val_ssim:
            best_val_ssim = save_ssim
            raw_state_dict = model.state_dict()
            save_state_dict = dict(raw_state_dict)
            if ema is not None:
                for name, tensor in ema.shadow.items():
                    if name in save_state_dict:
                        save_state_dict[name] = tensor
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': save_state_dict,
                'raw_model_state_dict': raw_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ssim': best_val_ssim,
                'image_size': args.image_size,
                'uses_ema_weights': bool(ema),
            }
            if ema is not None:
                save_dict['ema'] = ema.shadow
                save_dict['ema_shadow'] = ema.shadow
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(save_dict, save_path)
            print(f"  -> 最佳模型已保存: SSIM={best_val_ssim:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch + 1}.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)

    print("=" * 70)
    print(f"训练完成！最佳 SSIM: {best_val_ssim:.4f}")
