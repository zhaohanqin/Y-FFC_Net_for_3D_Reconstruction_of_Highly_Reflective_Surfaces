"""Single-image and folder inference pipeline."""

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .postprocess import (
    _sanitize_output_dict,
    _sanitize_output_tensor,
    adaptive_threshold,
    load_image,
    save_result,
    visualize_results,
)
from .sliding_window import _normalize_hw, sliding_window_inference


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
