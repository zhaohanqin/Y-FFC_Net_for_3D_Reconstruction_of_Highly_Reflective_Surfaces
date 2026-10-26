"""Decoupled copy of the original process_images.py entrypoint.

The configuration block matches the repository-root script. Inference
implementation lives in XFFC.inference.
"""

import os
from pathlib import Path

import torch

if __package__:
    from .XFFC.inference import (
        load_model,
        process_folder,
        process_single_image,
        resolve_patch_size_from_checkpoint,
    )
else:
    from XFFC.inference import (
        load_model,
        process_folder,
        process_single_image,
        resolve_patch_size_from_checkpoint,
    )


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
