"""Checkpoint loading for X-FFC inference."""

import torch

from ..models import XFFC
from ..training.utils import replace_bn_with_gn


def _uses_group_norm_state_dict(state_dict):
    """检测 checkpoint 是否来自 BN->GN 后的模型。"""
    return not any(
        key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))
        for key in state_dict
    )


def _merge_ema_weights(base_state_dict, ema_state_dict):
    """将 EMA 参数覆盖到完整 state_dict 上，保留非 EMA 项。"""
    merged = dict(base_state_dict)
    for name, tensor in ema_state_dict.items():
        if name in merged:
            merged[name] = tensor
    return merged


def _load_state_dict_flexible(model, state_dict):
    """优先严格加载；结构升级后形状不匹配的旧权重会被跳过。"""
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as exc:
        model_state = model.state_dict()
        compatible = {
            key: value for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        skipped = sorted(set(state_dict.keys()) - set(compatible.keys()))
        result = model.load_state_dict(compatible, strict=False)
        print(
            f"[WARNING] checkpoint 与当前结构不完全匹配，已兼容加载 "
            f"{len(compatible)}/{len(model_state)} 个张量；跳过 {len(skipped)} 个旧权重")
        if result.missing_keys:
            print(f"[WARNING] 新结构随机初始化参数数量: {len(result.missing_keys)}")
        if result.unexpected_keys:
            print(f"[WARNING] checkpoint 额外参数数量: {len(result.unexpected_keys)}")
        print(f"[WARNING] strict 加载失败摘要: {str(exc).splitlines()[0]}")


def load_model(model_path, device, prefer_ema=True):
    """
    智能加载模型权重，自动识别 checkpoint 格式。

    支持的格式：
        1. 训练保存的完整 checkpoint 字典（含 model_state_dict / ema_shadow 等）
        2. 纯 state_dict（旧版兼容）

    参数:
        model_path: 模型文件路径
        device: 计算设备
        prefer_ema: 若 checkpoint 中包含 EMA 权重，是否优先使用（推荐 True）

    返回:
        加载好权重的 XFFC 模型（已 .eval()）
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        base_state_dict = checkpoint['model_state_dict']
        # 推理 checkpoint 已包含训练后的 Swin 权重，禁用 ImageNet 预训练加载以避免离线推理时下载权重。
        model = XFFC(swin_pretrained=False)
        if _uses_group_norm_state_dict(base_state_dict):
            replace_bn_with_gn(model, num_groups=8)

        # ── 训练保存的标准 checkpoint ──
        # 优先使用 EMA 权重（推理更稳定）
        ema_state_dict = checkpoint.get('ema_shadow') or checkpoint.get('ema')
        if prefer_ema and ema_state_dict:
            # EMA 只保存 model.named_parameters()，需要用完整 state_dict 补齐缓冲/非 EMA 项。
            _load_state_dict_flexible(model, _merge_ema_weights(base_state_dict, ema_state_dict))
            epoch = checkpoint.get('epoch', '?')
            best_ssim = checkpoint.get('best_ssim', '?')
            print(f"  [OK] Loaded EMA weights (epoch={epoch}, best_ssim={best_ssim})")
        else:
            _load_state_dict_flexible(model, base_state_dict)
            epoch = checkpoint.get('epoch', '?')
            best_ssim = checkpoint.get('best_ssim', '?')
            print(f"  [OK] Loaded model weights (epoch={epoch}, best_ssim={best_ssim})")
        model.training_image_size = checkpoint.get('image_size')
        model.checkpoint_uses_ema_weights = checkpoint.get('uses_ema_weights', False)
    elif isinstance(checkpoint, dict):
        # ── 可能是纯 state_dict（key 为 layer 名称） ──
        # 推理 checkpoint 已包含训练后的 Swin 权重，禁用 ImageNet 预训练加载以避免离线推理时下载权重。
        model = XFFC(swin_pretrained=False)
        if _uses_group_norm_state_dict(checkpoint):
            replace_bn_with_gn(model, num_groups=8)
        _load_state_dict_flexible(model, checkpoint)
        print("  [OK] Loaded raw state_dict weights")
        model.training_image_size = None
        model.checkpoint_uses_ema_weights = False
    else:
        raise ValueError(f"无法识别的 checkpoint 格式: {type(checkpoint)}")

    model.to(device)
    model.eval()
    return model


def resolve_patch_size_from_checkpoint(model, configured_patch_size):
    """优先使用 checkpoint 记录的训练尺寸，避免推理 patch 与训练口径不一致。"""
    ckpt_image_size = getattr(model, 'training_image_size', None)
    if ckpt_image_size is None:
        return configured_patch_size

    ckpt_image_size = tuple(int(v) for v in ckpt_image_size)
    configured_patch_size = tuple(int(v) for v in configured_patch_size)
    if ckpt_image_size != configured_patch_size:
        print(
            f"[WARNING] checkpoint 训练尺寸为 {ckpt_image_size}，"
            f"当前配置 TRAIN_IMAGE_SIZE={configured_patch_size}；"
            f"将使用 checkpoint 尺寸进行滑窗推理")
    return ckpt_image_size
