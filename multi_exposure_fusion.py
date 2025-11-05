"""
多曝光融合脚本
用于将三组不同曝光时间的图像融合成更接近漫反射的图像
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


def find_best_pixel(pixel1, pixel2, pixel3, strategy='avoid_saturation'):
    """
    从三个像素值中选择最接近漫反射的像素值
    
    参数:
        pixel1, pixel2, pixel3: 三个不同曝光的像素值 (可以是单通道或多通道)
        strategy: 选择策略
            'avoid_saturation': 避免饱和，选择不过曝的最大值
            'middle_value': 选择中间值
            'closest_to_middle': 选择最接近中间灰度(128)的值
    
    返回:
        最佳像素值
    """
    
    if strategy == 'avoid_saturation':
        # 策略：避免过曝和欠曝
        # 过曝阈值（接近255表示过曝）
        saturation_threshold = 250
        # 欠曝阈值（太暗的像素）
        underexposure_threshold = 10
        
        # 计算每个像素的平均亮度（如果是彩色图像）
        if len(pixel1.shape) == 0:  # 单通道
            pixels = [pixel1, pixel2, pixel3]
        else:  # 多通道
            pixels = [pixel1, pixel2, pixel3]
            # 使用亮度来判断是否过曝
            brightness1 = np.mean(pixel1)
            brightness2 = np.mean(pixel2)
            brightness3 = np.mean(pixel3)
            brightnesses = [brightness1, brightness2, brightness3]
            
            # 找出不过曝的像素
            valid_indices = [i for i, b in enumerate(brightnesses) 
                           if b < saturation_threshold and b > underexposure_threshold]
            
            if len(valid_indices) > 0:
                # 从不过曝的像素中选择最亮的
                max_brightness_idx = max(valid_indices, key=lambda i: brightnesses[i])
                return pixels[max_brightness_idx]
            else:
                # 如果都过曝或欠曝，选择最接近中间值的
                middle = 128
                closest_idx = min(range(3), key=lambda i: abs(brightnesses[i] - middle))
                return pixels[closest_idx]
    
    elif strategy == 'middle_value':
        # 策略：选择中间值
        if len(pixel1.shape) == 0:  # 单通道
            return np.median([pixel1, pixel2, pixel3])
        else:  # 多通道
            return np.median([pixel1, pixel2, pixel3], axis=0).astype(np.uint8)
    
    elif strategy == 'closest_to_middle':
        # 策略：选择最接近中间灰度的像素
        middle = 128
        if len(pixel1.shape) == 0:  # 单通道
            pixels = [pixel1, pixel2, pixel3]
            distances = [abs(p - middle) for p in pixels]
        else:  # 多通道
            pixels = [pixel1, pixel2, pixel3]
            distances = [abs(np.mean(p) - middle) for p in pixels]
        
        min_dist_idx = np.argmin(distances)
        return pixels[min_dist_idx]
    
    return pixel1  # 默认返回第一个


def fuse_images(img1, img2, img3, strategy='avoid_saturation'):
    """
    融合三张不同曝光的图像
    
    参数:
        img1, img2, img3: 三张不同曝光的图像（按曝光时间从短到长）
        strategy: 像素选择策略
    
    返回:
        融合后的图像
    """
    # 确保三张图像尺寸相同
    assert img1.shape == img2.shape == img3.shape, "三张图像的尺寸必须相同"
    
    # 创建输出图像
    fused_img = np.zeros_like(img1)
    
    # 逐像素处理
    if len(img1.shape) == 2:  # 灰度图像
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                fused_img[i, j] = find_best_pixel(
                    img1[i, j], img2[i, j], img3[i, j], strategy
                )
    else:  # 彩色图像
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                fused_img[i, j] = find_best_pixel(
                    img1[i, j], img2[i, j], img3[i, j], strategy
                )
    
    return fused_img


def process_multi_exposure_fusion(
    exposure1_dir, 
    exposure2_dir, 
    exposure3_dir, 
    output_dir,
    strategy='avoid_saturation'
):
    """
    处理整个文件夹的多曝光融合
    
    参数:
        exposure1_dir: 短曝光图像文件夹路径
        exposure2_dir: 中曝光图像文件夹路径
        exposure3_dir: 长曝光图像文件夹路径
        output_dir: 输出文件夹路径
        strategy: 像素选择策略
    """
    
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取第一个文件夹中的所有图像文件
    exposure1_path = Path(exposure1_dir)
    image_files = sorted(list(exposure1_path.glob('*.png')) + 
                        list(exposure1_path.glob('*.jpg')) + 
                        list(exposure1_path.glob('*.bmp')))
    
    if len(image_files) == 0:
        print(f"错误：在 {exposure1_dir} 中没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像，开始处理...")
    print(f"使用策略: {strategy}")
    
    successful_count = 0
    failed_files = []
    
    # 处理每张图像
    for img_file in tqdm(image_files, desc="融合图像"):
        img_name = img_file.name
        
        # 构建三个曝光图像的路径
        img1_path = Path(exposure1_dir) / img_name
        img2_path = Path(exposure2_dir) / img_name
        img3_path = Path(exposure3_dir) / img_name
        
        # 检查三个文件是否都存在
        if not (img1_path.exists() and img2_path.exists() and img3_path.exists()):
            print(f"\n警告：图像 {img_name} 在某些文件夹中不存在，跳过")
            failed_files.append(img_name)
            continue
        
        try:
            # 读取三张图像
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            img3 = cv2.imread(str(img3_path))
            
            # 检查是否成功读取
            if img1 is None or img2 is None or img3 is None:
                print(f"\n警告：无法读取图像 {img_name}，跳过")
                failed_files.append(img_name)
                continue
            
            # 融合图像
            fused_img = fuse_images(img1, img2, img3, strategy)
            
            # 保存融合后的图像
            output_path = Path(output_dir) / img_name
            cv2.imwrite(str(output_path), fused_img)
            
            successful_count += 1
            
        except Exception as e:
            print(f"\n错误：处理图像 {img_name} 时出错: {str(e)}")
            failed_files.append(img_name)
    
    # 输出统计信息
    print("\n" + "="*50)
    print(f"处理完成！")
    print(f"成功处理: {successful_count} 张图像")
    print(f"失败: {len(failed_files)} 张图像")
    if failed_files:
        print(f"失败的文件: {', '.join(failed_files[:10])}")
        if len(failed_files) > 10:
            print(f"... 还有 {len(failed_files) - 10} 个文件")
    print(f"融合后的图像已保存到: {output_dir}")
    print("="*50)


def main():
    """
    主函数 - 配置路径并运行融合
    """
    
    print("="*50)
    print("多曝光融合脚本")
    print("="*50)
    
    # ==================== 配置参数 ====================
    # 请修改以下路径为你的实际路径
    
    # 三个不同曝光时间的图像文件夹（按曝光时间从短到长）
    exposure1_dir = r"E:\code\images\short_luminance100_17000us"      # 短曝光
    exposure2_dir = r"E:\code\images\mid_luminance200_17000us"     # 中曝光
    exposure3_dir = r"E:\code\images\long_luminance300_17000us"       # 长曝光
    
    # 输出文件夹
    output_dir = "data/fused_diffuse"
    
    # 像素选择策略：
    # 'avoid_saturation': 避免饱和，选择不过曝的最大值（推荐）
    # 'middle_value': 选择三个像素的中间值
    # 'closest_to_middle': 选择最接近中间灰度(128)的值
    strategy = 'avoid_saturation'
    
    # ==================================================
    
    # 检查输入文件夹是否存在
    if not os.path.exists(exposure1_dir):
        print(f"错误：文件夹 {exposure1_dir} 不存在")
        print("请修改脚本中的路径配置")
        return
    if not os.path.exists(exposure2_dir):
        print(f"错误：文件夹 {exposure2_dir} 不存在")
        print("请修改脚本中的路径配置")
        return
    if not os.path.exists(exposure3_dir):
        print(f"错误：文件夹 {exposure3_dir} 不存在")
        print("请修改脚本中的路径配置")
        return
    
    # 运行融合
    process_multi_exposure_fusion(
        exposure1_dir,
        exposure2_dir,
        exposure3_dir,
        output_dir,
        strategy
    )


if __name__ == "__main__":
    main()

