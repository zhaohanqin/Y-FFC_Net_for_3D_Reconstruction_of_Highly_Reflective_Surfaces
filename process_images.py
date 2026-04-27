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
from Y_FFC_unified import YFFC, replace_bn_with_gn


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
        加载好权重的 YFFC 模型（已 .eval()）
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        base_state_dict = checkpoint['model_state_dict']
        model = YFFC()
        if _uses_group_norm_state_dict(base_state_dict):
            replace_bn_with_gn(model, num_groups=8)

        # ── 训练保存的标准 checkpoint ──
        # 优先使用 EMA 权重（推理更稳定）
        ema_state_dict = checkpoint.get('ema_shadow') or checkpoint.get('ema')
        if prefer_ema and ema_state_dict:
            # EMA 只保存 model.named_parameters()，需要用完整 state_dict 补齐缓冲/非 EMA 项。
            model.load_state_dict(_merge_ema_weights(base_state_dict, ema_state_dict))
            epoch = checkpoint.get('epoch', '?')
            best_ssim = checkpoint.get('best_ssim', '?')
            print(f"  ✓ 已加载 EMA 权重 (epoch={epoch}, best_ssim={best_ssim})")
        else:
            model.load_state_dict(base_state_dict)
            epoch = checkpoint.get('epoch', '?')
            best_ssim = checkpoint.get('best_ssim', '?')
            print(f"  ✓ 已加载模型权重 (epoch={epoch}, best_ssim={best_ssim})")
    elif isinstance(checkpoint, dict):
        # ── 可能是纯 state_dict（key 为 layer 名称） ──
        model = YFFC()
        if _uses_group_norm_state_dict(checkpoint):
            replace_bn_with_gn(model, num_groups=8)
        model.load_state_dict(checkpoint)
        print("  ✓ 已加载纯 state_dict 权重")
    else:
        raise ValueError(f"无法识别的 checkpoint 格式: {type(checkpoint)}")

    model.to(device)
    model.eval()
    return model


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
    使用 Hann 窗让重叠区域平滑过渡；最小值钳到正数，避免边缘权重为 0。
    """
    if patch_h <= 1:
        wy = torch.ones(patch_h, device=device, dtype=dtype)
    else:
        wy = torch.hann_window(patch_h, periodic=False, device=device, dtype=dtype)
    if patch_w <= 1:
        wx = torch.ones(patch_w, device=device, dtype=dtype)
    else:
        wx = torch.hann_window(patch_w, periodic=False, device=device, dtype=dtype)

    weight = (wy[:, None] * wx[None, :]).clamp_min(1e-3)
    return weight.view(1, 1, patch_h, patch_w)


def sliding_window_inference(model, input_tensor, device, patch_size, overlap=32):
    """
    按训练尺寸进行滑窗 patch 推理，并融合回原图大小。

    参数:
        model: 已加载并 eval 的模型
        input_tensor: [1, 1, image_h, image_w]，灰度且已归一化到 [0, 1]
        device: torch.device
        patch_size: 训练输入尺寸，格式为 (H, W)
        overlap: patch 之间的重叠像素

    返回:
        {'r': full_diffuse, 'b': full_specular, 'g': full_gradient}
        三个输出尺寸均为 [1, 1, image_h, image_w]
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
        # reflect padding 对极小图像有限制；不满足时退回 replicate padding。
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
    weight_sum = torch.zeros(1, 1, padded_h, padded_w, device=device, dtype=input_padded.dtype)
    blend = _build_blend_window(patch_h, patch_w, device=device, dtype=input_padded.dtype)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = input_padded[:, :, y:y + patch_h, x:x + patch_w]
                out = model(patch)
                for key in ('r', 'b', 'g'):
                    accum[key][:, :, y:y + patch_h, x:x + patch_w] += out[key].clamp(0, 1) * blend
                weight_sum[:, :, y:y + patch_h, x:x + patch_w] += blend

    weight_sum = weight_sum.clamp_min(1e-6)
    return {
        key: (value / weight_sum)[:, :, :image_h, :image_w].clamp(0, 1)
        for key, value in accum.items()
    }


def save_result(tensor, save_path):
    """
    保存tensor为图像文件
    
    参数:
        tensor: 图像tensor [1, 1, H, W]
        save_path: 保存路径
    
    注意:
        输出图像尺寸与输入tensor尺寸一致，不进行resize
    """
    # 转换为numpy数组
    image_array = tensor.squeeze().cpu().numpy()
    
    # 裁剪到[0, 1]范围
    image_array = np.clip(image_array, 0, 1)
    
    # 转换为0-255
    image_array = (image_array * 255).astype(np.uint8)
    
    # 转换为PIL图像并保存
    image = Image.fromarray(image_array)
    image.save(save_path)


def visualize_results(input_img, output_dict, save_path):
    """
    可视化输入和所有输出结果
    
    参数:
        input_img: 输入图像tensor [1, 1, H, W]
        output_dict: 模型输出字典，包含 'r', 'g', 'b'
        save_path: 保存路径
    """
    # 转换为numpy数组
    input_array = input_img.squeeze().cpu().numpy()
    diffuse_array = output_dict['r'].squeeze().cpu().numpy()
    gradient_array = output_dict['g'].squeeze().cpu().numpy()
    specular_array = output_dict['b'].squeeze().cpu().numpy()
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 输入图像（含高反光）
    axes[0, 0].imshow(input_array, cmap='gray')
    axes[0, 0].set_title('Input (with Specular Highlights)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 漫反射输出（去除高反光后的结果）
    axes[0, 1].imshow(diffuse_array, cmap='gray')
    axes[0, 1].set_title('Output (Diffuse - Highlights Removed)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 镜面反射分量（提取的高反光部分）
    axes[1, 0].imshow(specular_array, cmap='hot')
    axes[1, 0].set_title('Specular Component (Highlights)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 梯度输出
    axes[1, 1].imshow(gradient_array, cmap='gray')
    axes[1, 1].set_title('Gradient Output', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_single_image(model, image_path, output_dir, device, show_visualization=True,
                         collect_diffuse_dir=None, patch_size=None, overlap=32):
    """
    处理单张图像
    
    参数:
        model: YFFC模型
        image_path: 输入图像路径
        output_dir: 输出目录
        device: 计算设备
        show_visualization: 是否生成可视化结果
    
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
                output = model(input_tensor)
        else:
            patch_h, patch_w = _normalize_hw(patch_size)
            image_h, image_w = input_tensor.shape[-2:]
            if (image_h, image_w) == (patch_h, patch_w):
                print(f"  整图推理: 输入尺寸 {image_h}x{image_w} 与训练尺寸一致")
                with torch.no_grad():
                    output = model(input_tensor)
            else:
                print(
                    f"  滑窗推理: 原图 {image_h}x{image_w}, "
                    f"patch {patch_h}x{patch_w}, overlap={overlap}")
                output = sliding_window_inference(
                    model, input_tensor, device,
                    patch_size=(patch_h, patch_w),
                    overlap=overlap)
        
        # 提取输出
        diffuse_output = output['r']      # 漫反射（去除高反光后的结果）
        specular_output = output['b']     # 镜面反射（提取的高反光）
        gradient_output = output['g']     # 梯度
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 保存结果（输出尺寸与输入一致）
        diffuse_filename = f'{base_name}_diffuse.png'
        save_result(diffuse_output, os.path.join(output_dir, diffuse_filename))
        save_result(specular_output, 
                    os.path.join(output_dir, f'{base_name}_specular.png'))
        save_result(gradient_output, 
                    os.path.join(output_dir, f'{base_name}_gradient.png'))
        
        # 可选：额外保存一份漫反射结果到集中目录，方便统一处理
        if collect_diffuse_dir:
            os.makedirs(collect_diffuse_dir, exist_ok=True)
            original_filename = os.path.basename(image_path)
            save_result(diffuse_output, os.path.join(collect_diffuse_dir, original_filename))
        
        # 可视化结果
        if show_visualization:
            visualize_results(input_tensor, output, 
                             save_path=os.path.join(output_dir, f'{base_name}_visualization.png'))
        
        return True
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return False


def process_folder(model, input_folder, output_folder, device, show_visualization=True,
                   supported_formats=None, collect_diffuse_dir=None,
                   patch_size=None, overlap=32):
    """
    批量处理文件夹中的所有图像
    
    参数:
        model: YFFC模型
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        device: 计算设备
        show_visualization: 是否生成可视化结果
        supported_formats: 支持的图像格式列表
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
        print(f"❌ 错误: 在目录 {input_folder} 中未找到支持的图像文件")
        print(f"支持的格式: {', '.join(supported_formats)}")
        return
    
    print(f"\n找到 {len(image_files)} 张图像，开始批量处理...")
    print("=" * 70)
    
    # 处理每张图像
    success_count = 0
    fail_count = 0
    
    for image_path in tqdm(image_files, desc="处理进度", unit="张"):
        # 为每张图像创建子目录
        image_name = Path(image_path).stem
        image_output_dir = os.path.join(output_folder, image_name)
        
        if process_single_image(model, image_path, image_output_dir, device, show_visualization,
                                collect_diffuse_dir, patch_size=patch_size, overlap=overlap):
            success_count += 1
        else:
            fail_count += 1
    
    # 输出统计信息
    print("\n" + "=" * 70)
    print("批量处理完成！")
    print("=" * 70)
    print(f"✓ 成功: {success_count} 张")
    if fail_count > 0:
        print(f"❌ 失败: {fail_count} 张")
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
    
    # 6. 支持的图像格式
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']

    # 滑窗推理配置：必须与训练阶段 IMAGE_SIZE 一致，格式为 (H, W)。
    TRAIN_IMAGE_SIZE = (128, 160)
    PATCH_OVERLAP = 32
    
    # ====================================================================
    
    print("=" * 70)
    print("Y-FFC Net 图像处理程序 - 去除高反光")
    print("=" * 70)
    
    # 检查输入路径是否存在
    if not os.path.exists(INPUT_PATH):
        print(f"\n❌ 错误: 输入路径不存在: {INPUT_PATH}")
        print(f"\n请修改配置:")
        print(f"  打开文件: process_images.py")
        print(f"  找到 main() 函数的配置区")
        print(f"  修改 INPUT_PATH = '你的图像路径或文件夹路径'")
        return
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ 错误: 模型文件不存在: {MODEL_PATH}")
        print(f"请确保已训练模型并保存为: {MODEL_PATH}")
        return
    
    # 设置设备
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        print("⚠ 警告: CUDA不可用，自动切换到CPU")
        device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH, device, prefer_ema=PREFER_EMA)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 判断输入类型并处理
    if os.path.isfile(INPUT_PATH):
        # 单张图像处理
        print(f"\n检测到输入为单张图像: {INPUT_PATH}")
        print("-" * 70)
        
        image_name = Path(INPUT_PATH).stem
        image_output_dir = os.path.join(OUTPUT_DIR, image_name)
        
        print(f"开始处理...")
        if process_single_image(model, INPUT_PATH, image_output_dir, device, SHOW_VISUALIZATION,
                                DIFFUSE_COLLECTION_DIR,
                                patch_size=TRAIN_IMAGE_SIZE, overlap=PATCH_OVERLAP):
            print(f"✓ 处理完成!")
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
        print("-" * 70)
        
        process_folder(model, INPUT_PATH, OUTPUT_DIR, device, SHOW_VISUALIZATION, SUPPORTED_FORMATS,
                       DIFFUSE_COLLECTION_DIR,
                       patch_size=TRAIN_IMAGE_SIZE, overlap=PATCH_OVERLAP)
        
        print(f"\n所有结果已保存到: {OUTPUT_DIR}")
        print(f"\n输出文件说明:")
        print(f"  - *_diffuse.png       : ⭐ 去除高反光后的漫反射图像（主要结果）")
        print(f"  - *_specular.png      : 提取的镜面反射（高反光部分）")
        print(f"  - *_gradient.png      : 梯度图像")
        if SHOW_VISUALIZATION:
            print(f"  - *_visualization.png : 完整可视化对比图")
        print(f"\n所有漫反射结果的副本（保持原始文件名）已集中存放于: {DIFFUSE_COLLECTION_DIR}")
    else:
        print(f"❌ 错误: 无效的输入路径: {INPUT_PATH}")
        return
    
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
