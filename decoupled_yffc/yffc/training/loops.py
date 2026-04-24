"""Enhanced train and validation loops."""

import torch
import torch.nn as nn
from tqdm import tqdm

from .metrics import MetricTracker


def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    scaler, ema, device, accum=4, max_norm=1.0, epoch=0, total_epochs=0):
    """
    执行一个 epoch 的训练。

    返回:
        avg_loss  : float，平均总损失
        avg_ssim  : float，平均 SSIM（漫反射分支，越高越好）
        avg_mae   : float，平均 MAE（漫反射分支）
        avg_mse   : float，平均 MSE（漫反射分支）
        avg_psnr  : float，平均 PSNR（漫反射分支，dB）
        avg_subs  : dict，所有子损失项的 epoch 平均值
    """
    model.train()
    tracker = MetricTracker()
    optimizer.zero_grad()

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f'Train {epoch+1}/{total_epochs}', leave=False)
    for i, (inp, tgt, gt_s, gt_g) in pbar:
        inp, tgt = inp.to(device), tgt.to(device)
        gt_s, gt_g = gt_s.to(device), gt_g.to(device)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out = model(inp)
            with torch.amp.autocast('cuda', enabled=False):
                loss, ld = criterion(
                    out['r'].float(), tgt.float(),
                    out['b'].float(), gt_s.float(),
                    out['g'].float(), gt_g.float())
            loss = loss / accum

        # NaN / Inf 保护
        if not torch.isfinite(loss):
            tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: loss NaN/Inf, 跳过')
            optimizer.zero_grad()
            continue

        # [FIX] 损失尖峰保护：如果单批损失超过历史均值的 10 倍，跳过该批次
        # 防止异常样本或数值不稳定导致的梯度爆炸
        _current_loss_val = ld['total']
        if tracker.n > 5:  # 至少积累 5 批数据后才判断
            _avg_so_far = tracker.tot_loss / tracker.n
            if _avg_so_far > 0 and _current_loss_val > _avg_so_far * 10.0:
                tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: '
                           f'loss spike {_current_loss_val:.4f} >> avg {_avg_so_far:.4f}, 跳过')
                optimizer.zero_grad()
                continue

        if device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            if device.type == 'cuda':
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            if ema:
                ema.update()

        # 使用 MetricTracker 统一累积全部指标（含 MSE / PSNR）
        with torch.no_grad():
            tracker.update(ld, out['r'].float(), tgt.float())

        # 更新进度条
        pbar.set_postfix({
            'loss': f"{ld['total']:.4f}",
            'ssim': f"{1 - ld['ssim']:.4f}",
            'l1':   f"{ld.get('l1', ld.get('mae', 0.0)):.4f}",
        })

    avg_loss, avg_ssim, avg_mae, avg_mse, avg_psnr, avg_subs = tracker.compute()
    return avg_loss, avg_ssim, avg_mae, avg_mse, avg_psnr, avg_subs


def _tta_forward(model, inp):
    """[P15] Test-Time Augmentation：4 次翻转推理取平均。"""
    preds_b = []
    for flip_h in [False, True]:
        for flip_v in [False, True]:
            x = inp
            if flip_h: x = torch.flip(x, [-1])
            if flip_v: x = torch.flip(x, [-2])
            out = model(x)
            b = out['b']
            if flip_v: b = torch.flip(b, [-2])
            if flip_h: b = torch.flip(b, [-1])
            preds_b.append(b)
    avg_b = torch.stack(preds_b, dim=0).mean(0)
    avg_r = torch.clamp(inp - avg_b, 0.0, 1.0)
    if hasattr(model, 'refinement_head'):
        avg_r = model.refinement_head(avg_r, inp)
    return {'r': avg_r, 'b': avg_b, 'g': out['g']}


@torch.no_grad()
def validate(model, loader, criterion, device, epoch=0, total_epochs=0,
             use_tta=False):
    """
    执行验证。

    返回:
        avg_loss  : float，平均总损失
        avg_ssim  : float，平均 SSIM（漫反射分支，越高越好）
        avg_mae   : float，平均 MAE（漫反射分支）
        avg_mse   : float，平均 MSE（漫反射分支）
        avg_psnr  : float，平均 PSNR（漫反射分支，dB）
        avg_subs  : dict，所有子损失项的 epoch 平均值
    """
    model.eval()
    tracker = MetricTracker()

    pbar = tqdm(loader, total=len(loader),
                desc=f'Val   {epoch+1}/{total_epochs}', leave=False)
    for inp, tgt, gt_s, gt_g in pbar:
        inp, tgt = inp.to(device), tgt.to(device)
        gt_s, gt_g = gt_s.to(device), gt_g.to(device)

        out = _tta_forward(model, inp) if use_tta else model(inp)  # [P15]
        loss, ld = criterion(out['r'], tgt, out['b'], gt_s, out['g'], gt_g)

        if not torch.isfinite(loss):
            continue

        tracker.update(ld, out['r'].float(), tgt.float())

    return tracker.compute()
