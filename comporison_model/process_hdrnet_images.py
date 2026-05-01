"""
HDRNet 图像处理脚本 - 去除高反光

功能：
    - 支持单张图像和文件夹批量处理
    - 支持整图推理和按训练尺寸滑窗推理
    - 输出去反光后的漫反射图像，以及输入/输出/差异图可视化

使用方法：
    1. 修改 main() 配置区中的 INPUT_PATH、MODEL_PATH、OUTPUT_DIR
    2. 运行: python comporison_model/process_hdrnet_images.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from HDRNet import unet


def _normalize_hw(size):
    """将训练尺寸规范化为 (H, W)。"""
    if isinstance(size, int):
        return int(size), int(size)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return int(size[0]), int(size[1])
    raise ValueError(f"patch_size 格式错误: {size}，应为 int 或 (H, W)")


def _load_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict'], checkpoint
    if isinstance(checkpoint, dict):
        return checkpoint, {}
    raise ValueError(f"无法识别的 checkpoint 格式: {type(checkpoint)}")


def load_model(model_path, device):
    """加载 HDRNet，兼容新 checkpoint 字典和旧版裸 state_dict。"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict, metadata = _load_state_dict(checkpoint)
    model = unet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    epoch = metadata.get('epoch', '?')
    val_loss = metadata.get('val_loss', '?')
    print(f"  已加载 HDRNet 权重 (epoch={epoch}, val_loss={val_loss})")
    return model


def load_image(image_path):
    image = Image.open(image_path).convert('L')
    original_size = image.size
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor, original_size


def _sliding_positions(length, patch_length, stride):
    if length <= patch_length:
        return [0]
    return list(range(0, length - patch_length + 1, stride))


def _padded_length(length, patch_length, stride):
    if length <= patch_length:
        return patch_length
    n_steps = int(np.ceil((length - patch_length) / stride))
    return patch_length + n_steps * stride


def _build_blend_window(patch_h, patch_w, device, dtype):
    wy = torch.ones(patch_h, device=device, dtype=dtype) if patch_h <= 1 else \
        torch.hann_window(patch_h, periodic=False, device=device, dtype=dtype)
    wx = torch.ones(patch_w, device=device, dtype=dtype) if patch_w <= 1 else \
        torch.hann_window(patch_w, periodic=False, device=device, dtype=dtype)
    return (wy[:, None] * wx[None, :]).clamp_min(1e-3).view(1, 1, patch_h, patch_w)


def postprocess_output(raw_output):
    """HDRNet 输出 raw logits，使用 sigmoid 映射到 [0, 1]。"""
    return torch.sigmoid(raw_output).clamp(0.0, 1.0)


@torch.no_grad()
def run_model(model, input_tensor):
    return postprocess_output(model(input_tensor))


def sliding_window_inference(model, input_tensor, device, patch_size, overlap=32):
    patch_h, patch_w = _normalize_hw(patch_size)
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"patch_size 必须为正数，当前为: {patch_size}")

    overlap = int(max(0, min(overlap, min(patch_h, patch_w) - 1)))
    stride_h = max(1, patch_h - overlap)
    stride_w = max(1, patch_w - overlap)

    input_tensor = input_tensor.to(device)
    _, _, image_h, image_w = input_tensor.shape
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
    accum = torch.zeros(1, 1, padded_h, padded_w, device=device, dtype=input_padded.dtype)
    weight_sum = torch.zeros_like(accum)
    blend = _build_blend_window(patch_h, patch_w, device, input_padded.dtype)

    for y in ys:
        for x in xs:
            patch = input_padded[:, :, y:y + patch_h, x:x + patch_w]
            pred = run_model(model, patch)
            accum[:, :, y:y + patch_h, x:x + patch_w] += pred * blend
            weight_sum[:, :, y:y + patch_h, x:x + patch_w] += blend

    return (accum / weight_sum.clamp_min(1e-6))[:, :, :image_h, :image_w].clamp(0.0, 1.0)


def save_result(tensor, save_path):
    arr = tensor.squeeze().detach().cpu().numpy()
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(save_path)


def visualize_result(input_tensor, diffuse_tensor, save_path):
    inp = input_tensor.squeeze().detach().cpu().numpy()
    out = diffuse_tensor.squeeze().detach().cpu().numpy()
    diff = np.abs(inp - out)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(inp, cmap='gray')
    axes[0].set_title('Input')
    axes[0].axis('off')
    axes[1].imshow(out, cmap='gray')
    axes[1].set_title('HDRNet Diffuse')
    axes[1].axis('off')
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def process_single_image(model, image_path, output_dir, device, show_visualization=True,
                         collect_diffuse_dir=None, patch_size=None, overlap=32):
    try:
        os.makedirs(output_dir, exist_ok=True)
        input_tensor, _ = load_image(image_path)
        input_tensor = input_tensor.to(device)

        if patch_size is None:
            diffuse = run_model(model, input_tensor)
        else:
            patch_h, patch_w = _normalize_hw(patch_size)
            image_h, image_w = input_tensor.shape[-2:]
            if (image_h, image_w) == (patch_h, patch_w):
                diffuse = run_model(model, input_tensor)
            else:
                print(f"  滑窗推理: 原图 {image_h}x{image_w}, patch {patch_h}x{patch_w}, overlap={overlap}")
                diffuse = sliding_window_inference(model, input_tensor, device, patch_size, overlap)

        base_name = Path(image_path).stem
        diffuse_path = os.path.join(output_dir, f'{base_name}_diffuse.png')
        save_result(diffuse, diffuse_path)

        if collect_diffuse_dir:
            os.makedirs(collect_diffuse_dir, exist_ok=True)
            save_result(diffuse, os.path.join(collect_diffuse_dir, os.path.basename(image_path)))

        if show_visualization:
            visualize_result(input_tensor, diffuse,
                             os.path.join(output_dir, f'{base_name}_visualization.png'))
        return True
    except Exception as exc:
        print(f"  处理失败: {exc}")
        return False


def process_folder(model, input_folder, output_folder, device, show_visualization=True,
                   supported_formats=None, collect_diffuse_dir=None, patch_size=None, overlap=32):
    supported_formats = supported_formats or ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_paths = [
        os.path.join(input_folder, name)
        for name in sorted(os.listdir(input_folder))
        if Path(name).suffix.lower() in supported_formats
    ]
    print(f"找到 {len(image_paths)} 张图像")
    success_count = 0
    for image_path in tqdm(image_paths, desc='Processing'):
        image_output_dir = os.path.join(output_folder, Path(image_path).stem)
        if process_single_image(model, image_path, image_output_dir, device, show_visualization,
                                collect_diffuse_dir, patch_size, overlap):
            success_count += 1
    print(f"批处理完成: 成功 {success_count}, 失败 {len(image_paths) - success_count}")


def main():
    # ==================== 配置区 ====================
    INPUT_PATH = r'E:\images\PSM_FSM\O1'
    MODEL_PATH = r'experiments\hdrnet_YYYYMMDD_HHMMSS\models\best_model.pth'
    OUTPUT_DIR = 'processed_results_hdrnet'
    DIFFUSE_COLLECTION_DIR = os.path.join(OUTPUT_DIR, 'all_diffuse')
    DEVICE = 'cuda'
    SHOW_VISUALIZATION = True
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    TRAIN_IMAGE_SIZE = (128, 160)
    PATCH_OVERLAP = 64
    # =================================================

    if not os.path.exists(INPUT_PATH):
        print(f"输入路径不存在: {INPUT_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"模型文件不存在: {MODEL_PATH}")
        return

    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"加载模型: {MODEL_PATH}")
    model = load_model(MODEL_PATH, device)

    if os.path.isfile(INPUT_PATH):
        image_output_dir = os.path.join(OUTPUT_DIR, Path(INPUT_PATH).stem)
        ok = process_single_image(model, INPUT_PATH, image_output_dir, device, SHOW_VISUALIZATION,
                                  DIFFUSE_COLLECTION_DIR, TRAIN_IMAGE_SIZE, PATCH_OVERLAP)
        if ok:
            print(f"处理完成，结果保存到: {image_output_dir}")
    elif os.path.isdir(INPUT_PATH):
        process_folder(model, INPUT_PATH, OUTPUT_DIR, device, SHOW_VISUALIZATION, SUPPORTED_FORMATS,
                       DIFFUSE_COLLECTION_DIR, TRAIN_IMAGE_SIZE, PATCH_OVERLAP)
        print(f"所有结果已保存到: {OUTPUT_DIR}")
    else:
        print(f"无效输入路径: {INPUT_PATH}")


if __name__ == '__main__':
    main()
