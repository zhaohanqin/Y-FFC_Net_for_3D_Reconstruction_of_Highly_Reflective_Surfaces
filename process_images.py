"""
Y-FFC Net 图像处理脚本 - 去除高反光

功能特点：
    - 支持单张图像和批量处理
    - 保持原始图像尺寸，输出与输入尺寸完全一致
    - 无需命令行参数，直接在 main() 函数配置区修改路径
    - 自动生成漫反射、镜面反射、梯度和可视化对比图

使用方法：
    1. 打开本文件，找到 main() 函数的配置区
    2. 修改 INPUT_PATH 为你的图像路径或文件夹路径
    3. 修改 MODEL_PATH 为训练好的模型路径
    4. 运行: python process_images.py

输出说明：
    - *_diffuse.png   : 去除高反光后的漫反射图像（主要结果）
    - *_specular.png  : 提取的镜面反射分量（高反光部分）
    - *_gradient.png  : 梯度图像
    - *_visualization.png : 完整可视化对比图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm
# ★[V3.2] 同时导入 _tukey_window 用于滑窗融合
from X_FFC_unified import XFFC, replace_bn_with_gn, _tukey_window


def _sanitize_output_tensor(tensor):
    """Keep inference/export robust if an old or partially compatible checkpoint emits non-finite values."""
    return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _sanitize_output_dict(output):
    return {k: _sanitize_output_tensor(v) if isinstance(v, torch.Tensor) else v
            for k, v in output.items()}


# ════════════════════════════════════════════════════════════════════════
# ★[D10 V3.2] Test-Time Augmentation 推理
# ════════════════════════════════════════════════════════════════════════
def _tta_forward(model, inp):
    """4 次翻转推理取平均，融合 dual-head 输出。

    与 train.py 的 _tta_forward 一致，但只在推理时调用。
    """
    preds_b, preds_r_direct, preds_g, preds_s = [], [], [], []
    last_alpha = None
    with torch.no_grad():
        for flip_h in [False, True]:
            for flip_v in [False, True]:
                x = inp
                if flip_h:
                    x = torch.flip(x, [-1])
                if flip_v:
                    x = torch.flip(x, [-2])
                out = model(x)
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
    avg_g = torch.stack(preds_g, dim=0).mean(0) if preds_g else None
    avg_s = torch.stack(preds_s, dim=0).mean(0) if preds_s else None

    # 双路融合 + 精修（与 XFFC.forward 末尾一致）
    r_sub = torch.clamp(inp - avg_b, 0.0, 1.0)
    if avg_r_direct is not None:
        if last_alpha is not None and last_alpha.dim() == 0:
            alpha_val = last_alpha
        else:
            alpha_val = torch.tensor(0.5, device=inp.device)
        r_fused = alpha_val * avg_r_direct + (1.0 - alpha_val) * r_sub
    else:
        r_fused = r_sub
    if hasattr(model, 'refinement_head'):
        r_final = model.refinement_head(
            r_fused, inp,
            avg_r_direct if avg_r_direct is not None else r_fused)
    else:
        r_final = r_fused
    result = {'r': r_final, 'b': avg_b, 'g': avg_g}
    if avg_s is not None:
        result['s'] = avg_s
    if avg_r_direct is not None:
        result['r_direct'] = avg_r_direct
    return result


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


def load_image(image_path, target_size=None):
    """
    加载并预处理图像（保持原始尺寸）
    
    参数:
        image_path: 图像路径
        target_size: 目标尺寸（可选）
            - None: 保持原始尺寸（推荐用于推理）
            - int: resize到正方形，如512表示512x512
            - tuple: resize到指定尺寸，如(768, 1024)表示高768宽1024
    
    返回:
        image_tensor: 预处理后的tensor [1, 1, H, W]
        original_size: 原始图像尺寸 (width, height)
    
    注意:
        推理时建议使用 target_size=None 保持原始尺寸，以获得最佳效果
    """
    # 读取图像
    image = Image.open(image_path)
    
    # 记录原始尺寸
    original_size = image.size  # (width, height)
    
    # 转换为灰度图
    if image.mode != 'L':
        image = image.convert('L')
    
    # 可选：调整大小
    if target_size is not None:
        if isinstance(target_size, int):
            # 正方形resize
            image = image.resize((target_size, target_size), Image.LANCZOS)
        elif isinstance(target_size, tuple) and len(target_size) == 2:
            # 指定尺寸resize (width, height)
            image = image.resize(target_size, Image.LANCZOS)
    
    # 转换为numpy数组并归一化到[0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # 转换为tensor，添加batch和channel维度: [1, 1, H, W]
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    
    return image_tensor, original_size


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


# ════════════════════════════════════════════════════════════════════════
# ★[FIX-A/C 问题2] SpecLik 自适应二值化阈值
# ════════════════════════════════════════════════════════════════════════
def adaptive_threshold(arr, mode='percentile', value=0.85):
    """计算 SpecLik 二值化阈值。

    旧版固定 0.5 阈值与当前输出分布脱节（SpecLik 峰值常 <0.5），导致二值图全黑。
    本函数支持三种模式：
        mode='fixed'      : 固定阈值 value（兼容旧行为，传 value=0.5）
        mode='percentile' : 取分布的 value 分位数（如 0.85 → 响应最高 15% 判为高光，
                            与 GT specular 的稀疏性匹配，推荐默认）
        mode='otsu'       : Otsu 自动阈值（类间方差最大化）
    """
    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if mode == 'fixed':
        return float(value)
    if mode == 'percentile':
        return float(np.quantile(arr, value))
    # otsu（256 bin 简易实现）
    hist, edges = np.histogram(arr, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    w = hist.cumsum()
    centers = (edges[:-1] + edges[1:]) / 2.0
    mu = (hist * centers).cumsum()
    total_w = w[-1] if w[-1] > 0 else 1.0
    total_mu = mu[-1]
    best_t, best_var = 0.5, -1.0
    for i in range(1, 256):
        w0 = w[i]
        w1 = total_w - w0
        if w0 <= 0 or w1 <= 0:
            continue
        m0 = mu[i] / w0
        m1 = (total_mu - mu[i]) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, float((edges[i] + edges[i + 1]) / 2.0)
    return best_t


def save_result(tensor, save_path, bit_depth=8):
    """
    保存tensor为图像文件
    
    参数:
        tensor: 图像tensor [1, 1, H, W]
        save_path: 保存路径
        bit_depth: 保存位深，支持 8 或 16
    
    注意:
        输出图像尺寸与输入tensor尺寸一致，不进行resize
    """
    # 转换为numpy数组
    tensor = _sanitize_output_tensor(tensor.detach())
    image_array = tensor.squeeze().cpu().numpy()
    
    # 裁剪到[0, 1]范围
    image_array = np.nan_to_num(image_array, nan=0.0, posinf=1.0, neginf=0.0)
    image_array = np.clip(image_array, 0, 1)
    
    if bit_depth == 8:
        image_array = (image_array * 255).round().astype(np.uint8)
        image = Image.fromarray(image_array)
    elif bit_depth == 16:
        image_array = (image_array * 65535).round().astype(np.uint16)
        image = Image.fromarray(image_array, mode='I;16')
    else:
        raise ValueError(f"bit_depth 仅支持 8 或 16，当前为: {bit_depth}")
    image.save(save_path)


def visualize_results(input_img, output_dict, save_path):
    """
    可视化输入和所有输出结果。

    布局：2×3 子图
        第一行：输入图像 | 漫反射输出 | 镜面反射分量
        第二行：梯度先验 | Pred SpecLikelihood (新辅助头) | |SpecLik - Specular| 差异图
        若模型未输出 's' 字段, 自动退回到 2×2 布局以保持向后兼容。

    参数:
        input_img: 输入图像tensor [1, 1, H, W]
        output_dict: 模型输出字典，包含 'r', 'g', 'b', 可选 's'
        save_path: 保存路径
    """
    # 转换为numpy数组
    input_array = input_img.squeeze().cpu().numpy()
    diffuse_array = output_dict['r'].squeeze().cpu().numpy()
    gradient_array = output_dict['g'].squeeze().cpu().numpy()
    specular_array = output_dict['b'].squeeze().cpu().numpy()
    # [NEW] spec_likelihood 输出 (可能不存在, 兼容旧 checkpoint)
    spec_lik_tensor = output_dict.get('s')
    spec_lik_array = (spec_lik_tensor.squeeze().cpu().numpy()
                      if spec_lik_tensor is not None else None)

    if spec_lik_array is None:
        # 旧版兼容: 退回到 2×2 布局
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0, 0].imshow(input_array, cmap='gray')
        axes[0, 0].set_title('Input (with Specular Highlights)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(diffuse_array, cmap='gray')
        axes[0, 1].set_title('Output (Diffuse - Highlights Removed)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(specular_array, cmap='hot')
        axes[1, 0].set_title('Specular Component (Highlights)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(gradient_array, cmap='gray')
        axes[1, 1].set_title('Predicted Input-Gradient Prior', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
    else:
        # 新版布局: 2×3, 加入 spec_likelihood 相关
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 第一行
        axes[0, 0].imshow(input_array, cmap='gray')
        axes[0, 0].set_title('Input (with Specular Highlights)', fontsize=13, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(diffuse_array, cmap='gray')
        axes[0, 1].set_title('Diffuse (Highlights Removed)', fontsize=13, fontweight='bold')
        axes[0, 1].axis('off')

        im_spec = axes[0, 2].imshow(specular_array, cmap='hot')
        axes[0, 2].set_title('Specular Component', fontsize=13, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im_spec, ax=axes[0, 2], fraction=0.046)

        # 第二行
        axes[1, 0].imshow(gradient_array, cmap='gray')
        axes[1, 0].set_title('Input-Gradient Prior', fontsize=13, fontweight='bold')
        axes[1, 0].axis('off')

        # 关键新增: spec_likelihood 热力图
        im_slik = axes[1, 1].imshow(spec_lik_array, cmap='hot', vmin=0, vmax=1)
        axes[1, 1].set_title('Spec-Likelihood (Aux Head)', fontsize=13, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im_slik, ax=axes[1, 1], fraction=0.046)

        # 与主解码 specular 的差异, 直观看一致性
        diff_slik_spec = np.abs(spec_lik_array - specular_array)
        im_diff = axes[1, 2].imshow(diff_slik_spec, cmap='hot', vmin=0, vmax=0.3)
        axes[1, 2].set_title('|SpecLik - Specular|', fontsize=13, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im_diff, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_single_image(model, image_path, output_dir, device, show_visualization=True,
                         collect_diffuse_dir=None, patch_size=None, overlap=32,
                         save_16bit=False, use_tta=False,
                         spec_binary_mode='percentile', spec_binary_value=0.85):
    """
    处理单张图像
    
    参数:
        model: YFFC模型
        image_path: 输入图像路径
        output_dir: 输出目录
        device: 计算设备
        show_visualization: 是否生成可视化结果
        use_tta: ★[V3.2] 是否启用 4-flip TTA（推理质量↑约 0.005~0.015 SSIM，时间×4）

    返回:
        True 如果成功，False 如果失败
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载和预处理输入图像
        input_tensor, original_size = load_image(image_path)
        input_tensor = input_tensor.to(device)
        
        # 执行推理
        model.eval()
        if patch_size is None:
            with torch.no_grad():
                if use_tta:
                    output = _sanitize_output_dict(_tta_forward(model, input_tensor))
                else:
                    output = _sanitize_output_dict(model(input_tensor))
        else:
            patch_h, patch_w = _normalize_hw(patch_size)
            image_h, image_w = input_tensor.shape[-2:]
            if (image_h, image_w) == (patch_h, patch_w):
                print(f"  整图推理: 输入尺寸 {image_h}x{image_w} 与训练尺寸一致"
                      + (" (TTA)" if use_tta else ""))
                with torch.no_grad():
                    if use_tta:
                        output = _sanitize_output_dict(_tta_forward(model, input_tensor))
                    else:
                        output = _sanitize_output_dict(model(input_tensor))
            else:
                print(
                    f"  滑窗推理: 原图 {image_h}x{image_w}, "
                    f"patch {patch_h}x{patch_w}, overlap={overlap}"
                    + (" (TTA)" if use_tta else ""))
                output = _sanitize_output_dict(sliding_window_inference(
                    model, input_tensor, device,
                    patch_size=(patch_h, patch_w),
                    overlap=overlap, use_tta=use_tta))
        
        # 提取输出
        diffuse_output = output['r']      # 漫反射（去除高反光后的结果）
        specular_output = output['b']     # 镜面反射（提取的高反光）
        gradient_output = output['g']     # 梯度
        # [NEW] 梯度分支辅助头: 镜面高光区域软概率 (0~1), 可能不存在 (旧 checkpoint)
        spec_lik_output = output.get('s')

        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 保存结果（输出尺寸与输入一致）
        diffuse_filename = f'{base_name}_diffuse.png'
        save_result(diffuse_output, os.path.join(output_dir, diffuse_filename))
        save_result(specular_output, 
                    os.path.join(output_dir, f'{base_name}_specular.png'))
        save_result(gradient_output, 
                    os.path.join(output_dir, f'{base_name}_gradient.png'))
        # [NEW] 保存 spec_likelihood 输出 (若存在)
        if spec_lik_output is not None:
            save_result(spec_lik_output,
                        os.path.join(output_dir, f'{base_name}_spec_likelihood.png'))
            # ★[FIX-A/C 问题2] 自适应阈值二值高光图 + 分布打印
            slik_arr = _sanitize_output_tensor(spec_lik_output).squeeze().cpu().numpy()
            thr = adaptive_threshold(slik_arr, mode=spec_binary_mode, value=spec_binary_value)
            print(f"  [SpecLik] min={slik_arr.min():.3f} max={slik_arr.max():.3f} "
                  f"mean={slik_arr.mean():.3f} p95={np.percentile(slik_arr, 95):.3f} "
                  f"thr({spec_binary_mode})={thr:.3f} "
                  f">thr_ratio={np.mean(slik_arr > thr):.4f}")
            binary = (slik_arr > thr).astype(np.float32)
            save_result(torch.from_numpy(binary)[None, None],
                        os.path.join(output_dir, f'{base_name}_spec_binary.png'))
        if save_16bit:
            save_result(diffuse_output, os.path.join(output_dir, f'{base_name}_diffuse_16bit.png'),
                        bit_depth=16)
            save_result(specular_output, os.path.join(output_dir, f'{base_name}_specular_16bit.png'),
                        bit_depth=16)
            save_result(gradient_output, os.path.join(output_dir, f'{base_name}_gradient_16bit.png'),
                        bit_depth=16)
            # [NEW] 16-bit 版本同步保存
            if spec_lik_output is not None:
                save_result(spec_lik_output,
                            os.path.join(output_dir, f'{base_name}_spec_likelihood_16bit.png'),
                            bit_depth=16)
        
        # 可选：额外保存一份漫反射结果到集中目录，方便统一处理
        if collect_diffuse_dir:
            os.makedirs(collect_diffuse_dir, exist_ok=True)
            original_filename = os.path.basename(image_path)
            save_result(diffuse_output, os.path.join(collect_diffuse_dir, original_filename))
            if save_16bit:
                stem, _ = os.path.splitext(original_filename)
                save_result(diffuse_output, os.path.join(collect_diffuse_dir, f'{stem}_16bit.png'),
                            bit_depth=16)
        
        # 可视化结果
        if show_visualization:
            visualize_results(input_tensor, output, 
                             save_path=os.path.join(output_dir, f'{base_name}_visualization.png'))
        
        return True
    except Exception as e:
        print(f"  [ERROR] Processing failed: {e}")
        return False


def process_folder(model, input_folder, output_folder, device, show_visualization=True,
                   supported_formats=None, collect_diffuse_dir=None,
                   patch_size=None, overlap=32, save_16bit=False, use_tta=False,
                   spec_binary_mode='percentile', spec_binary_value=0.85):
    """
    批量处理文件夹中的所有图像
    
    参数:
        model: YFFC模型
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        device: 计算设备
        show_visualization: 是否生成可视化结果
        supported_formats: 支持的图像格式列表
        use_tta: ★[V3.2] 是否启用 4-flip TTA
    """
    if supported_formats is None:
        supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    # 获取所有支持格式的图像文件
    # Windows 的 glob 对大小写不敏感，分别搜 .png/.PNG 会重复命中同一文件。
    supported_suffixes = {ext.lower() for ext in supported_formats}
    image_files = sorted(
        str(p) for p in Path(input_folder).iterdir()
        if p.is_file() and p.suffix.lower() in supported_suffixes
    )
    
    if not image_files:
        print(f"[ERROR] No supported image files found in {input_folder}")
        print(f"支持的格式: {', '.join(supported_formats)}")
        return
    
    print(f"\n找到 {len(image_files)} 张图像，开始批量处理"
          + (" (TTA 启用)" if use_tta else "") + "...")
    print("=" * 70)
    
    # 处理每张图像
    success_count = 0
    fail_count = 0
    
    for image_path in tqdm(image_files, desc="处理进度", unit="张"):
        # 为每张图像创建子目录
        image_name = Path(image_path).stem
        image_output_dir = os.path.join(output_folder, image_name)
        
        if process_single_image(model, image_path, image_output_dir, device, show_visualization,
                                collect_diffuse_dir, patch_size=patch_size, overlap=overlap,
                                save_16bit=save_16bit, use_tta=use_tta,
                                spec_binary_mode=spec_binary_mode,
                                spec_binary_value=spec_binary_value):
            success_count += 1
        else:
            fail_count += 1
    
    # 输出统计信息
    print("\n" + "=" * 70)
    print("批量处理完成！")
    print("=" * 70)
    print(f"[OK] Success: {success_count}")
    if fail_count > 0:
        print(f"[ERROR] Failed: {fail_count}")
    print(f"输出目录: {output_folder}")


def main():
    """
    主函数 - 在这里直接配置所有参数
    """
    
    # ==================== ⭐ 配置区 - 在这里修改参数 ⭐ ====================
    
    # 1. 输入配置（修改为你的图像路径或文件夹路径）
    INPUT_PATH = r'E:\images\PSM_FSM\O1'  # 可以是单张图像路径或文件夹路径
    
    # 2. 模型配置
    MODEL_PATH = r'experiments\yffc_v2_20260426_225811\models\best_model_best_ssim.pth'  # 训练好的模型路径
    PREFER_EMA = True  # 若 checkpoint 含 EMA 权重则优先使用（推理更稳定）
    
    # 3. 输出配置
    OUTPUT_DIR = 'processed_results'  # 输出目录
    DIFFUSE_COLLECTION_DIR = os.path.join(OUTPUT_DIR, 'all_diffuse')  # 统一存放漫反射
    
    # 4. 设备配置
    DEVICE = 'cuda'  # 'cuda' 使用GPU，'cpu' 使用CPU
    
    # 5. 可视化配置
    SHOW_VISUALIZATION = True  # 是否生成可视化对比图
    SAVE_16BIT_OUTPUT = False  # 额外保存 16-bit PNG，减少弱纹理量化损失
    # ★[V3.2 新增] TTA (Test-Time Augmentation) 配置
    # 启用后每个 patch 做 4 次翻转推理取平均，预期 SSIM 再提升 0.005~0.015
    # 代价：推理时间 ×4（CPU 上可能较慢，GPU 上影响较小）
    USE_TTA = True

    # ★[FIX-A/C 问题2] SpecLik 二值高光图阈值配置
    #   'percentile' + 0.85 : 响应最高 15% 判为高光（推荐，与高光稀疏性匹配）
    #   'otsu'              : 自动阈值（value 被忽略）
    #   'fixed'  + 0.5      : 兼容旧固定阈值
    SPEC_BINARY_MODE = 'percentile'
    SPEC_BINARY_VALUE = 0.85
    
    # 6. 支持的图像格式
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']

    # 滑窗推理配置：必须与训练阶段 IMAGE_SIZE 一致，格式为 (H, W)。
    TRAIN_IMAGE_SIZE = (128, 160)
    # Swin 版建议 sliding-window 推理，并让 patch 尺寸与训练 IMAGE_SIZE 保持一致；推荐 overlap=64。
    PATCH_OVERLAP = 64
    
    # ====================================================================
    
    print("=" * 70)
    print("Y-FFC Net 图像处理程序 - 去除高反光")
    print("=" * 70)
    
    # 检查输入路径是否存在
    if not os.path.exists(INPUT_PATH):
        print(f"\n[ERROR] Input path does not exist: {INPUT_PATH}")
        print(f"\n请修改配置:")
        print(f"  打开文件: process_images.py")
        print(f"  找到 main() 函数的配置区")
        print(f"  修改 INPUT_PATH = '你的图像路径或文件夹路径'")
        return
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model file does not exist: {MODEL_PATH}")
        print(f"请确保已训练模型并保存为: {MODEL_PATH}")
        return
    
    # 设置设备
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA is unavailable; falling back to CPU")
        device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH, device, prefer_ema=PREFER_EMA)
        print("[OK] Model loaded")
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return
    TRAIN_IMAGE_SIZE = resolve_patch_size_from_checkpoint(model, TRAIN_IMAGE_SIZE)
    
    # 判断输入类型并处理
    if os.path.isfile(INPUT_PATH):
        # 单张图像处理
        print(f"\n检测到输入为单张图像: {INPUT_PATH}")
        print(f"TTA: {'启用 (4 翻转推理)' if USE_TTA else '禁用'}")
        print("-" * 70)
        
        image_name = Path(INPUT_PATH).stem
        image_output_dir = os.path.join(OUTPUT_DIR, image_name)
        
        print(f"开始处理...")
        if process_single_image(model, INPUT_PATH, image_output_dir, device, SHOW_VISUALIZATION,
                                DIFFUSE_COLLECTION_DIR,
                                patch_size=TRAIN_IMAGE_SIZE, overlap=PATCH_OVERLAP,
                                save_16bit=SAVE_16BIT_OUTPUT, use_tta=USE_TTA,
                                spec_binary_mode=SPEC_BINARY_MODE,
                                spec_binary_value=SPEC_BINARY_VALUE):
            print("[OK] Processing complete")
            print(f"\n输出文件位置:")
            print(f"  - {image_output_dir}/{image_name}_diffuse.png       (⭐ 去除高反光后的结果)")
            print(f"  - {image_output_dir}/{image_name}_specular.png     (镜面反射分量)")
            print(f"  - {image_output_dir}/{image_name}_gradient.png     (梯度图)")
            if SHOW_VISUALIZATION:
                print(f"  - {image_output_dir}/{image_name}_visualization.png (可视化对比图)")
        
    elif os.path.isdir(INPUT_PATH):
        # 文件夹批量处理
        print(f"\n检测到输入为文件夹: {INPUT_PATH}")
        print(f"支持的图像格式: {', '.join(SUPPORTED_FORMATS)}")
        print(f"TTA: {'启用 (4 翻转推理)' if USE_TTA else '禁用'}")
        print("-" * 70)
        
        process_folder(model, INPUT_PATH, OUTPUT_DIR, device, SHOW_VISUALIZATION, SUPPORTED_FORMATS,
                       DIFFUSE_COLLECTION_DIR,
                       patch_size=TRAIN_IMAGE_SIZE, overlap=PATCH_OVERLAP,
                       save_16bit=SAVE_16BIT_OUTPUT, use_tta=USE_TTA,
                       spec_binary_mode=SPEC_BINARY_MODE,
                       spec_binary_value=SPEC_BINARY_VALUE)
        
        print(f"\n所有结果已保存到: {OUTPUT_DIR}")
        print(f"\n输出文件说明:")
        print(f"  - *_diffuse.png       : ⭐ 去除高反光后的漫反射图像（主要结果）")
        print(f"  - *_specular.png      : 提取的镜面反射（高反光部分）")
        print(f"  - *_gradient.png      : 梯度图像")
        if SHOW_VISUALIZATION:
            print(f"  - *_visualization.png : 完整可视化对比图")
        print(f"\n所有漫反射结果的副本（保持原始文件名）已集中存放于: {DIFFUSE_COLLECTION_DIR}")
    else:
        print(f"[ERROR] Invalid input path: {INPUT_PATH}")
        return
    
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
