"""Enhanced train and validation loops."""

import math

import torch
import torch.nn as nn
from tqdm import tqdm

from .metrics import MetricTracker


def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    scaler, ema, device, accum=4, max_norm=1.0, epoch=0, total_epochs=0,
                    use_amp=True):
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
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f'Train {epoch+1}/{total_epochs}', leave=False)
    for i, (inp, tgt, gt_s, gt_g) in pbar:
        inp, tgt = inp.to(device), tgt.to(device)
        gt_s, gt_g = gt_s.to(device), gt_g.to(device)

        if not _tensors_are_finite(inp, tgt, gt_s, gt_g):
            tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: input/target NaN/Inf, 跳过')
            optimizer.zero_grad(set_to_none=True)
            continue

        amp_enabled = (device.type == 'cuda' and bool(use_amp))
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            out = model(inp)

        if not _tensors_are_finite(out):
            repaired = _repair_nonfinite_model(model, ema)
            repaired += _sanitize_optimizer_state(optimizer)
            tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: output NaN/Inf, 跳过; repaired={repaired}')
            optimizer.zero_grad(set_to_none=True)
            continue

        with torch.amp.autocast('cuda', enabled=False):
            # [NEW] 取出 spec_likelihood 头的输出 pred_s_lik (可能不存在, 用 .get 兼容)
            pred_s_lik = out.get('s')
            # ★[V3.2 新增] 取出 MainDecoder 直接漫反射头输出 pred_r_direct
            pred_r_direct = out.get('r_direct')
            loss, ld = criterion(
                out['r'].float(), tgt.float(),
                out['b'].float(), gt_s.float(),
                out['g'].float(), gt_g.float(),
                out.get('cs'), input_img=inp.float(),
                pred_s_lik=pred_s_lik.float() if pred_s_lik is not None else None,
                pred_r_direct=pred_r_direct.float() if pred_r_direct is not None else None)
            loss = loss / accum

        # NaN / Inf 保护
        if not torch.isfinite(loss):
            repaired = _repair_nonfinite_model(model, ema)
            repaired += _sanitize_optimizer_state(optimizer)
            tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: loss NaN/Inf, 跳过; repaired={repaired}')
            optimizer.zero_grad(set_to_none=True)
            continue

        # [FIX] 损失尖峰保护：如果单批损失超过历史均值的 10 倍，跳过该批次
        # 防止异常样本或数值不稳定导致的梯度爆炸
        _current_loss_val = ld['total']
        if tracker.n > 5:  # 至少积累 5 批数据后才判断
            _avg_so_far = tracker.tot_loss / tracker.n
            if _avg_so_far > 0 and _current_loss_val > _avg_so_far * 10.0:
                tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: '
                           f'loss spike {_current_loss_val:.4f} >> avg {_avg_so_far:.4f}, 跳过')
                optimizer.zero_grad(set_to_none=True)
                continue

        if device.type == 'cuda' and bool(use_amp):
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            if device.type == 'cuda' and bool(use_amp):
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=False)
                if not torch.isfinite(grad_norm):
                    repaired = _repair_nonfinite_model(model, ema)
                    repaired += _sanitize_optimizer_state(optimizer)
                    # [DIAG] 打印各子损失值,辅助定位 NaN 源
                    _diag = _format_loss_diag(ld)
                    tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: grad NaN/Inf, '
                               f'跳过 step; repaired={repaired}; sub_losses={_diag}')
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=False)
                if not torch.isfinite(grad_norm):
                    repaired = _repair_nonfinite_model(model, ema)
                    repaired += _sanitize_optimizer_state(optimizer)
                    _diag = _format_loss_diag(ld)
                    tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: grad NaN/Inf, '
                               f'跳过 step; repaired={repaired}; sub_losses={_diag}')
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
            repaired = _repair_nonfinite_model(model, ema)
            repaired += _sanitize_optimizer_state(optimizer)
            if repaired:
                tqdm.write(f'[WARNING] Epoch {epoch+1} Batch {i+1}: optimizer 后发现非有限参数/状态，已修复 {repaired} 项')
            if scheduler:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
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
    """[P15] Test-Time Augmentation：4 次翻转推理取平均。

    ★[V3.2] 兼容 dual-head：除 'b' 外，'r_direct'、's' 也参与翻转累积，
    再走相同的"双路融合 + 精修"路径得到最终 r。
    若模型未输出 'r_direct'（旧 V3 ckpt），退回旧版 r = clamp(x - avg_b, 0, 1) → refinement。
    """
    preds_b = []
    preds_r_direct = []
    preds_g = []
    preds_s = []
    last_alpha = None
    for flip_h in [False, True]:
        for flip_v in [False, True]:
            x = inp
            if flip_h:
                x = torch.flip(x, [-1])
            if flip_v:
                x = torch.flip(x, [-2])
            out = model(x)
            # 把 b/r_direct/g/s 各通道按翻转回原方向，再累加
            for key, bucket in (('b', preds_b),
                                ('r_direct', preds_r_direct),
                                ('g', preds_g),
                                ('s', preds_s)):
                v = out.get(key)
                if v is None:
                    continue
                if flip_v:
                    v = torch.flip(v, [-2])
                if flip_h:
                    v = torch.flip(v, [-1])
                bucket.append(v)
            last_alpha = out.get('alpha')
    avg_b = torch.stack(preds_b, dim=0).mean(0) if preds_b else None
    avg_r_direct = torch.stack(preds_r_direct, dim=0).mean(0) if preds_r_direct else None
    avg_g = torch.stack(preds_g, dim=0).mean(0) if preds_g else out.get('g')
    avg_s = torch.stack(preds_s, dim=0).mean(0) if preds_s else None

    # 复刻 XFFC.forward 末尾的双路融合 + 精修，保证 TTA 输出 r 与正常 forward 路径一致
    r_sub = torch.clamp(inp - avg_b, 0.0, 1.0)
    if avg_r_direct is not None:
        # 用最后一次 forward 的 alpha（与训练保持一致；推理时 alpha 已稳定）
        if last_alpha is not None and last_alpha.dim() == 0:
            alpha_val = last_alpha
        else:
            alpha_val = torch.tensor(0.5, device=inp.device)
        r_fused = alpha_val * avg_r_direct + (1.0 - alpha_val) * r_sub
    else:
        r_fused = r_sub
    if hasattr(model, 'refinement_head'):
        r_final = model.refinement_head(r_fused, inp, avg_r_direct if avg_r_direct is not None else r_fused)
    else:
        r_final = r_fused
    result = {'r': r_final, 'b': avg_b, 'g': avg_g}
    if avg_s is not None:
        result['s'] = avg_s
    if avg_r_direct is not None:
        result['r_direct'] = avg_r_direct
    return result


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
        if not _tensors_are_finite(out):
            continue
        # [NEW] spec_likelihood 头输出
        pred_s_lik = out.get('s')
        # ★[V3.2 新增] MainDecoder 直接漫反射头输出
        pred_r_direct = out.get('r_direct')
        loss, ld = criterion(
            out['r'], tgt, out['b'], gt_s, out['g'], gt_g,
            out.get('cs'), input_img=inp,
            pred_s_lik=pred_s_lik,
            pred_r_direct=pred_r_direct)

        if not torch.isfinite(loss):
            continue

        tracker.update(ld, out['r'].float(), tgt.float())

    return tracker.compute()

def _tensors_are_finite(*items):
    for item in items:
        if isinstance(item, torch.Tensor) and not torch.isfinite(item).all():
            return False
        if isinstance(item, dict):
            for value in item.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    return False
    return True

def _repair_nonfinite_model(model, ema=None):
    """Restore non-finite parameters from EMA when possible, else sanitize.

    If one optimizer step corrupts parameters, simply skipping later NaN losses
    is not enough.  This function makes the next batch recoverable.
    """
    repaired = 0
    ema_shadow = getattr(ema, 'shadow', None) if ema is not None else None
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                if ema_shadow is not None and name in ema_shadow and torch.isfinite(ema_shadow[name]).all():
                    param.data.copy_(ema_shadow[name].to(device=param.device, dtype=param.dtype))
                else:
                    param.data.copy_(torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0))
                repaired += 1
        for buffer in model.buffers():
            if torch.is_floating_point(buffer) and not torch.isfinite(buffer).all():
                buffer.copy_(torch.nan_to_num(buffer, nan=0.0, posinf=1.0, neginf=-1.0))
                repaired += 1
    return repaired

def _sanitize_optimizer_state(optimizer):
    """Clear non-finite Adam/AdamW state tensors after a repaired step."""
    repaired = 0
    with torch.no_grad():
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value) and not torch.isfinite(value).all():
                    value.copy_(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
                    repaired += 1
    return repaired

def _format_loss_diag(loss_dict, top_k=8):
    """[DIAG] 把 loss_dict 格式化为紧凑诊断字符串，按值降序排前 top_k 项。

    grad NaN/Inf 时调用，帮助定位是哪个具体子损失项数值异常。
    若某个项是 nan/inf，会用 'NaN' / 'Inf' 标记。
    """
    if not isinstance(loss_dict, dict) or not loss_dict:
        return '<empty>'
    items = []
    for k, v in loss_dict.items():
        try:
            fv = float(v)
            if math.isnan(fv):
                items.append((k, float('inf'), 'NaN'))
            elif math.isinf(fv):
                items.append((k, float('inf'), 'Inf'))
            else:
                items.append((k, abs(fv), f'{fv:.4g}'))
        except (TypeError, ValueError):
            items.append((k, 0.0, str(v)))
    # 按绝对值降序，并把 NaN/Inf 排在最前
    items.sort(key=lambda t: t[1], reverse=True)
    parts = [f'{k}={s}' for k, _, s in items[:top_k]]
    return '{' + ', '.join(parts) + '}'
