"""Image IO, thresholding, and visualization for X-FFC inference."""

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


def _sanitize_output_tensor(tensor):
    """Keep inference/export robust if an old or partially compatible checkpoint emits non-finite values."""
    return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _sanitize_output_dict(output):
    return {k: _sanitize_output_tensor(v) if isinstance(v, torch.Tensor) else v
            for k, v in output.items()}


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
