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
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
from tqdm import tqdm
from Y_FFC import YFFC


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


def process_single_image(model, image_path, output_dir, device, show_visualization=True):
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
        with torch.no_grad():
            output = model(input_tensor)
        
        # 提取输出
        diffuse_output = output['r']      # 漫反射（去除高反光后的结果）
        specular_output = output['b']     # 镜面反射（提取的高反光）
        gradient_output = output['g']     # 梯度
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 保存结果（输出尺寸与输入一致）
        save_result(diffuse_output, 
                    os.path.join(output_dir, f'{base_name}_diffuse.png'))
        save_result(specular_output, 
                    os.path.join(output_dir, f'{base_name}_specular.png'))
        save_result(gradient_output, 
                    os.path.join(output_dir, f'{base_name}_gradient.png'))
        
        # 可视化结果
        if show_visualization:
            visualize_results(input_tensor, output, 
                             save_path=os.path.join(output_dir, f'{base_name}_visualization.png'))
        
        return True
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return False


def process_folder(model, input_folder, output_folder, device, show_visualization=True, 
                   supported_formats=None):
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
    image_files = []
    for ext in supported_formats:
        image_files.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(input_folder, f'*{ext.upper()}')))
    
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
        
        if process_single_image(model, image_path, image_output_dir, device, show_visualization):
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
    INPUT_PATH = 'h1.bmp'  # 可以是单张图像路径或文件夹路径
    
    # 2. 模型配置
    MODEL_PATH = 'best_model.pth'  # 训练好的模型路径
    
    # 3. 输出配置
    OUTPUT_DIR = 'processed_results'  # 输出目录
    
    # 4. 设备配置
    DEVICE = 'cuda'  # 'cuda' 使用GPU，'cpu' 使用CPU
    
    # 5. 可视化配置
    SHOW_VISUALIZATION = True  # 是否生成可视化对比图
    
    # 6. 支持的图像格式
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
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
        model = YFFC()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
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
        if process_single_image(model, INPUT_PATH, image_output_dir, device, SHOW_VISUALIZATION):
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
        
        process_folder(model, INPUT_PATH, OUTPUT_DIR, device, SHOW_VISUALIZATION, SUPPORTED_FORMATS)
        
        print(f"\n所有结果已保存到: {OUTPUT_DIR}")
        print(f"\n输出文件说明:")
        print(f"  - *_diffuse.png       : ⭐ 去除高反光后的漫反射图像（主要结果）")
        print(f"  - *_specular.png      : 提取的镜面反射（高反光部分）")
        print(f"  - *_gradient.png      : 梯度图像")
        if SHOW_VISUALIZATION:
            print(f"  - *_visualization.png : 完整可视化对比图")
    else:
        print(f"❌ 错误: 无效的输入路径: {INPUT_PATH}")
        return
    
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()

