"""Sliding-window inference and Tukey blending."""

import numpy as np
import torch
import torch.nn.functional as F

from ..models.common import _tukey_window
from .postprocess import _sanitize_output_dict


def _normalize_hw(size):
    """将训练尺寸规范化为 (H, W)，注意不是 PIL 使用的 (W, H)。"""
    if isinstance(size, int):
        return int(size), int(size)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return int(size[0]), int(size[1])
    raise ValueError(f"patch_size 格式错误: {size}，应为 int 或 (H, W)")


def _sliding_positions(length, patch_length, stride):
    """生成一维滑窗起点；调用前 length 已按 stride 补齐。"""
    if length <= patch_length:
        return [0]
    return list(range(0, length - patch_length + 1, stride))


def _padded_length(length, patch_length, stride):
    """计算 padding 后的长度，确保所有滑窗都按固定 stride 落在完整 patch 上。"""
    if length <= patch_length:
        return patch_length
    n_steps = int(np.ceil((length - patch_length) / stride))
    return patch_length + n_steps * stride


def _build_blend_window(patch_h, patch_w, device, dtype):
    """
    构造 patch 融合权重。

    ★[D11 V3.2] 由 Hann 窗 → Tukey 窗 (α=0.25)。
    Hann 窗在边界处衰减到 0（仅靠 clamp_min(1e-3) 维持非零），
    导致边界像素几乎只由单个 patch 决定，patch 边界会出现细小亮度跳变。
    Tukey 窗中央 75% 保持全权重 1.0，两端 12.5% 余弦平滑过渡，
    既避免边界突变，又给重叠区充分混合空间。
    """
    if patch_h <= 1:
        wy = torch.ones(patch_h, device=device, dtype=dtype)
    else:
        wy = _tukey_window(patch_h, alpha=0.25, device=device, dtype=dtype)
    if patch_w <= 1:
        wx = torch.ones(patch_w, device=device, dtype=dtype)
    else:
        wx = _tukey_window(patch_w, alpha=0.25, device=device, dtype=dtype)

    weight = (wy[:, None] * wx[None, :]).clamp_min(1e-3)
    return weight.view(1, 1, patch_h, patch_w)


def sliding_window_inference(model, input_tensor, device, patch_size, overlap=32,
                              use_tta=False):
    """
    按训练尺寸进行滑窗 patch 推理，并融合回原图大小。

    参数:
        model: 已加载并 eval 的模型
        input_tensor: [1, 1, image_h, image_w]，灰度且已归一化到 [0, 1]
        device: torch.device
        patch_size: 训练输入尺寸，格式为 (H, W)
        overlap: patch 之间的重叠像素
        use_tta: ★[V3.2] 是否对每个 patch 启用 4-flip TTA（推理质量↑ 4×推理时间↑）

    返回:
        {'r': full_diffuse, 'b': full_specular, 'g': full_gradient,
         's': full_spec_likelihood (若模型输出),
         'r_direct': full_r_direct (若模型输出，V3.2 dual-head)}
        输出尺寸均为 [1, 1, image_h, image_w]
    """
    patch_h, patch_w = _normalize_hw(patch_size)
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"patch_size 必须为正数，当前为: {patch_size}")

    max_overlap = min(patch_h, patch_w) - 1
    overlap = int(max(0, min(overlap, max_overlap)))
    stride_h = max(1, patch_h - overlap)
    stride_w = max(1, patch_w - overlap)

    model.eval()
    input_tensor = input_tensor.to(device)
    _, _, image_h, image_w = input_tensor.shape

    # 先补齐到滑窗网格尺寸：不 resize 原图，只在右/下边界 padding，推理后再裁回原图。
    padded_h_target = _padded_length(image_h, patch_h, stride_h)
    padded_w_target = _padded_length(image_w, patch_w, stride_w)
    pad_bottom = padded_h_target - image_h
    pad_right = padded_w_target - image_w
    if pad_bottom > 0 or pad_right > 0:
        can_reflect = pad_bottom < image_h and pad_right < image_w and image_h > 1 and image_w > 1
        pad_mode = 'reflect' if can_reflect else 'replicate'
        input_padded = F.pad(input_tensor, (0, pad_right, 0, pad_bottom), mode=pad_mode)
    else:
        input_padded = input_tensor

    _, _, padded_h, padded_w = input_padded.shape
    ys = _sliding_positions(padded_h, patch_h, stride_h)
    xs = _sliding_positions(padded_w, patch_w, stride_w)

    accum = {
        'r': torch.zeros(1, 1, padded_h, padded_w, device=device, dtype=input_padded.dtype),
        'b': torch.zeros(1, 1, padded_h, padded_w, device=device, dtype=input_padded.dtype),
        'g': torch.zeros(1, 1, padded_h, padded_w, device=device, dtype=input_padded.dtype),
    }
    has_spec_likelihood = False
    has_r_direct = False
    weight_sum = torch.zeros(1, 1, padded_h, padded_w, device=device, dtype=input_padded.dtype)
    blend = _build_blend_window(patch_h, patch_w, device=device, dtype=input_padded.dtype)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = input_padded[:, :, y:y + patch_h, x:x + patch_w]
                # ★[D10 V3.2] 可选 TTA 推理
                if use_tta:
                    from .pipeline import _tta_forward
                    out = _sanitize_output_dict(_tta_forward(model, patch))
                else:
                    out = _sanitize_output_dict(model(patch))

                # 懒初始化 's' 累加器
                if 's' in out and out['s'] is not None and not has_spec_likelihood:
                    accum['s'] = torch.zeros(1, 1, padded_h, padded_w,
                                             device=device, dtype=input_padded.dtype)
                    has_spec_likelihood = True
                # ★[V3.2] 懒初始化 'r_direct' 累加器
                if 'r_direct' in out and out['r_direct'] is not None and not has_r_direct:
                    accum['r_direct'] = torch.zeros(1, 1, padded_h, padded_w,
                                                    device=device, dtype=input_padded.dtype)
                    has_r_direct = True

                channels = ['r', 'b', 'g']
                if has_spec_likelihood:
                    channels.append('s')
                if has_r_direct:
                    channels.append('r_direct')
                for key in channels:
                    if key in out and out[key] is not None:
                        accum[key][:, :, y:y + patch_h, x:x + patch_w] += out[key] * blend
                weight_sum[:, :, y:y + patch_h, x:x + patch_w] += blend

    weight_sum = weight_sum.clamp_min(1e-6)
    return {
        key: (value / weight_sum)[:, :, :image_h, :image_w].clamp(0, 1)
        for key, value in accum.items()
    }
