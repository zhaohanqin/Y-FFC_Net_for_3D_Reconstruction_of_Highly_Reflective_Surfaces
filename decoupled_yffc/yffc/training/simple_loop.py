"""Original lightweight train/test loop used by the legacy unified CLI."""

import os
import time

import torch
from torch.utils.data import DataLoader

from ..data import FringeDataset, PairedTransform
from ..losses import CompositeLoss
from ..models import YFFC
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

            with torch.amp.autocast(device_type='cuda', enabled=False):
                loss, loss_dict = criterion(
                    pred_diffuse.float(), target_img.float(),
                    est_b.float(), gt_specular.float(),
                    pred_grad.float(), gt_gradient.float())

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
            if ema is not None:
                ema.update()

        if scheduler is not None:
            scheduler.step()

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
            output['g'], gt_gradient)

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

    model = YFFC()
    print("使用 YFFC 模型")

    if args.use_gn:
        print("将 BatchNorm 替换为 GroupNorm...")
        replace_bn_with_gn(model, num_groups=8)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 数据集
    paired_aug = PairedTransform() if args.use_augmentation else None
    train_dataset = FringeDataset(args.input_dir, args.target_dir,
                                  resize=args.image_size, paired_transform=paired_aug)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == 'cuda'), drop_last=True)

    val_loader = None
    if args.val_input_dir and args.val_target_dir:
        val_dataset = FringeDataset(args.val_input_dir, args.val_target_dir,
                                    resize=args.image_size, paired_transform=None)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=(device.type == 'cuda'))

    # 损失函数
    criterion = CompositeLoss(use_perceptual=True).to(device)
    loss_weight_scheduler = DynamicLossWeightScheduler(total_epochs=args.epochs)

    # 优化器 + 调度器
    optimizer = build_optimizer(model, base_lr=args.base_lr, weight_decay=1e-4)
    iters_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, warmup_iters=min(500, iters_per_epoch * 3),
                                total_epochs=args.epochs, iters_per_epoch=iters_per_epoch)

    scaler = torch.amp.GradScaler(device='cuda', enabled=(device.type == 'cuda'))
    ema = EMA(model, decay=0.999) if args.use_ema else None

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
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ssim': best_val_ssim,
            }
            if ema is not None:
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
