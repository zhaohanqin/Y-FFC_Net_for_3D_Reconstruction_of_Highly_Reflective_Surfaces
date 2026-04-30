"""
Y-FFC original-size inference script for highly reflective images.

This script is designed for checkpoints produced by the current train.py:
    - model: Y_FFC_unified.YFFC
    - optional GN structure: replace_bn_with_gn(model, 8)
    - optional EMA weights: checkpoint["ema_shadow"] or checkpoint["ema"]

Usage:
    1. Edit the configuration block in main().
    2. Run: python predict_yffc.py

Main result:
    *_diffuse.png is the highlight-removed diffuse image.
"""

import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from Y_FFC_unified import YFFC, replace_bn_with_gn


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def extract_state_dict(checkpoint):
    """Return model weights from a full checkpoint or a plain state_dict."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], True
    if isinstance(checkpoint, dict):
        return checkpoint, False
    raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")


def looks_like_gn_state_dict(state_dict):
    """GN checkpoints do not contain BatchNorm running-stat buffers."""
    running_tokens = ("running_mean", "running_var", "num_batches_tracked")
    has_running_stats = any(
        any(token in key for token in running_tokens)
        for key in state_dict.keys()
    )
    has_norm_weights = any(
        key.endswith((".bn.weight", ".bn.bias"))
        for key in state_dict.keys()
    )
    return has_norm_weights and not has_running_stats


def adapt_model_for_checkpoint(model, state_dict):
    """
    Keep compatibility with older checkpoints that predate main_decoder.xg_proj.
    """
    has_xg_proj = any(key.startswith("main_decoder.xg_proj.") for key in state_dict)
    if not has_xg_proj and hasattr(model.main_decoder, "xg_proj"):
        model.main_decoder.xg_proj = nn.Identity()
        print("  [OK] checkpoint has no xg_proj; using legacy gradient path")


def load_ema_shadow(model, ema_shadow):
    """Overlay EMA parameters onto an already-loaded model."""
    model_sd = model.state_dict()
    loaded = 0
    skipped = 0

    for name, value in ema_shadow.items():
        if name not in model_sd or model_sd[name].shape != value.shape:
            skipped += 1
            continue
        model_sd[name] = value.to(
            device=model_sd[name].device,
            dtype=model_sd[name].dtype,
        )
        loaded += 1

    model.load_state_dict(model_sd, strict=True)
    return loaded, skipped


def load_model(model_path, device, prefer_ema=True):
    """Load a YFFC model in the same structure used by train.py."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict, is_full_checkpoint = extract_state_dict(checkpoint)

    model = YFFC()
    if looks_like_gn_state_dict(state_dict):
        replace_bn_with_gn(model, 8)
        print("  [OK] detected GN checkpoint; applied BN -> GN")

    adapt_model_for_checkpoint(model, state_dict)
    model.to(device)
    model.load_state_dict(state_dict, strict=True)

    if is_full_checkpoint:
        epoch = checkpoint.get("epoch", "?")
        best_ssim = checkpoint.get("best_ssim", checkpoint.get("best_val_ssim", "?"))
        val_loss = checkpoint.get("val_loss", checkpoint.get("best_val_loss", "?"))
        print(f"  [OK] loaded checkpoint weights (epoch={epoch}, best_ssim={best_ssim}, val_loss={val_loss})")
    else:
        print("  [OK] loaded plain state_dict weights")

    ema_key = None
    if is_full_checkpoint and prefer_ema:
        if "ema_shadow" in checkpoint:
            ema_key = "ema_shadow"
        elif "ema" in checkpoint:
            ema_key = "ema"

    if ema_key:
        loaded, skipped = load_ema_shadow(model, checkpoint[ema_key])
        print(f"  [OK] applied EMA weights ({ema_key}): loaded={loaded}, skipped={skipped}")
    elif is_full_checkpoint and prefer_ema:
        print("  [WARN] checkpoint has no EMA weights; using normal model weights")

    model.eval()
    return model


def normalize_inference_size(inference_size):
    """
    None means original-size inference.
    int means square resize.
    (H, W) means explicit resize before inference.
    """
    if inference_size is None:
        return None
    if isinstance(inference_size, int):
        return int(inference_size), int(inference_size)
    if isinstance(inference_size, (tuple, list)) and len(inference_size) == 2:
        return int(inference_size[0]), int(inference_size[1])
    raise ValueError("INFERENCE_SIZE must be None, int, or (H, W)")


def load_image(image_path, inference_size=None):
    """
    Load an image as [1, 1, H, W].

    By default this preserves the original image size. If inference_size is set,
    resize only for inference and keep original_size for output restoration.
    """
    image = Image.open(image_path).convert("L")
    original_size = image.size  # PIL: (W, H)

    size_hw = normalize_inference_size(inference_size)
    if size_hw is not None:
        target_h, target_w = size_hw
        image = image.resize((target_w, target_h), Image.LANCZOS)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return tensor, original_size


def resize_tensor_to_pil_size(tensor, pil_size):
    """Resize [B, C, H, W] to a PIL size tuple (W, H)."""
    target_w, target_h = pil_size
    if tensor.shape[-2:] == (target_h, target_w):
        return tensor
    return F.interpolate(
        tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )


def tensor_to_uint8_image(tensor):
    array = tensor.detach().squeeze().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).round().astype(np.uint8)


def save_tensor_image(tensor, save_path, original_size):
    tensor = resize_tensor_to_pil_size(tensor, original_size)
    image = Image.fromarray(tensor_to_uint8_image(tensor))
    image.save(save_path)


def save_result_visualization(input_tensor, output, save_path, original_size):
    input_vis = resize_tensor_to_pil_size(input_tensor, original_size)
    diffuse = resize_tensor_to_pil_size(output["r"], original_size)
    specular = resize_tensor_to_pil_size(output["b"], original_size)
    gradient = resize_tensor_to_pil_size(output["g"], original_size)

    input_array = np.clip(input_vis.squeeze().detach().cpu().numpy(), 0.0, 1.0)
    diffuse_array = np.clip(diffuse.squeeze().detach().cpu().numpy(), 0.0, 1.0)
    specular_array = np.clip(specular.squeeze().detach().cpu().numpy(), 0.0, 1.0)
    gradient_array = np.clip(gradient.squeeze().detach().cpu().numpy(), 0.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(input_array, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Input", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(diffuse_array, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Diffuse Output", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(specular_array, cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title("Specular Component", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gradient_array, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Gradient Output", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_inference(model, input_tensor, device):
    input_tensor = input_tensor.to(device)
    with torch.inference_mode():
        output = model(input_tensor)
    return input_tensor, output


def process_single_image(
    model,
    image_path,
    output_dir,
    device,
    inference_size=None,
    save_visualization=True,
    save_specular=True,
    save_gradient=True,
    diffuse_collection_dir=None,
):
    try:
        os.makedirs(output_dir, exist_ok=True)
        input_tensor, original_size = load_image(image_path, inference_size)
        input_tensor, output = run_inference(model, input_tensor, device)

        base_name = Path(image_path).stem
        diffuse_path = os.path.join(output_dir, f"{base_name}_diffuse.png")
        save_tensor_image(output["r"], diffuse_path, original_size)

        if save_specular:
            save_tensor_image(
                output["b"],
                os.path.join(output_dir, f"{base_name}_specular.png"),
                original_size,
            )

        if save_gradient:
            save_tensor_image(
                output["g"],
                os.path.join(output_dir, f"{base_name}_gradient.png"),
                original_size,
            )

        if diffuse_collection_dir:
            os.makedirs(diffuse_collection_dir, exist_ok=True)
            save_tensor_image(
                output["r"],
                os.path.join(diffuse_collection_dir, Path(image_path).name),
                original_size,
            )

        if save_visualization:
            save_visualization_path = os.path.join(output_dir, f"{base_name}_visualization.png")
            save_result_visualization(input_tensor, output, save_visualization_path, original_size)

        return True, diffuse_path
    except Exception as exc:
        print(f"  [ERROR] failed to process {image_path}: {exc}")
        return False, None


def collect_image_files(input_folder, recursive=False):
    pattern_root = "**/*" if recursive else "*"
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(input_folder, f"{pattern_root}{ext}"), recursive=recursive))
        image_files.extend(glob.glob(os.path.join(input_folder, f"{pattern_root}{ext.upper()}"), recursive=recursive))
    return sorted(set(image_files))


def process_folder(
    model,
    input_folder,
    output_folder,
    device,
    inference_size=None,
    save_visualization=True,
    save_specular=True,
    save_gradient=True,
    diffuse_collection_dir=None,
    recursive=False,
):
    image_files = collect_image_files(input_folder, recursive=recursive)
    if not image_files:
        print(f"[ERROR] no supported image files found in: {input_folder}")
        print(f"Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
        return 0, 0

    print(f"\nFound {len(image_files)} images. Starting batch inference...")
    print("=" * 70)

    success_count = 0
    fail_count = 0
    for image_path in tqdm(image_files, desc="Inference", unit="image"):
        relative_stem = Path(image_path).stem
        image_output_dir = os.path.join(output_folder, relative_stem)
        ok, _ = process_single_image(
            model=model,
            image_path=image_path,
            output_dir=image_output_dir,
            device=device,
            inference_size=inference_size,
            save_visualization=save_visualization,
            save_specular=save_specular,
            save_gradient=save_gradient,
            diffuse_collection_dir=diffuse_collection_dir,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def main():
    # ======================== Configuration ========================

    INPUT_PATH = r"H:\images\before_2025_12_27\images_PSM_81_72_64\place2\11_luminance300_81"
    MODEL_PATH = r"experiments\yffc_v2_20260425_154101\models\best_model_best_ssim.pth"
    OUTPUT_DIR = "processed_results"

    DEVICE = "cuda"
    INFERENCE_SIZE = None  # None means original-size inference.
    PREFER_EMA = True

    SAVE_VISUALIZATION = True
    SAVE_SPECULAR = True
    SAVE_GRADIENT = True
    RECURSIVE = False

    DIFFUSE_COLLECTION_DIR = os.path.join(OUTPUT_DIR, "all_diffuse")

    # ===============================================================

    print("=" * 70)
    print("Y-FFC original-size inference")
    print("=" * 70)
    print(f"Input path      : {INPUT_PATH}")
    print(f"Model path      : {MODEL_PATH}")
    print(f"Output dir      : {OUTPUT_DIR}")
    print(f"Inference size  : {'original image size' if INFERENCE_SIZE is None else INFERENCE_SIZE}")
    print(f"Prefer EMA      : {PREFER_EMA}")

    if not os.path.exists(INPUT_PATH):
        print(f"\n[ERROR] input path does not exist: {INPUT_PATH}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] model file does not exist: {MODEL_PATH}")
        return

    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(DEVICE)
    print(f"Device          : {device}")

    print(f"\nLoading model: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH, device, prefer_ema=PREFER_EMA)
    except Exception as exc:
        print(f"[ERROR] model loading failed: {exc}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.isfile(INPUT_PATH):
        image_name = Path(INPUT_PATH).stem
        image_output_dir = os.path.join(OUTPUT_DIR, image_name)
        print(f"\nProcessing single image: {INPUT_PATH}")
        ok, diffuse_path = process_single_image(
            model=model,
            image_path=INPUT_PATH,
            output_dir=image_output_dir,
            device=device,
            inference_size=INFERENCE_SIZE,
            save_visualization=SAVE_VISUALIZATION,
            save_specular=SAVE_SPECULAR,
            save_gradient=SAVE_GRADIENT,
            diffuse_collection_dir=DIFFUSE_COLLECTION_DIR,
        )
        if ok:
            print("\n[OK] inference complete")
            print(f"Main result: {diffuse_path}")
            print(f"Output dir : {image_output_dir}")
        return

    if os.path.isdir(INPUT_PATH):
        print(f"\nProcessing folder: {INPUT_PATH}")
        success_count, fail_count = process_folder(
            model=model,
            input_folder=INPUT_PATH,
            output_folder=OUTPUT_DIR,
            device=device,
            inference_size=INFERENCE_SIZE,
            save_visualization=SAVE_VISUALIZATION,
            save_specular=SAVE_SPECULAR,
            save_gradient=SAVE_GRADIENT,
            diffuse_collection_dir=DIFFUSE_COLLECTION_DIR,
            recursive=RECURSIVE,
        )
        print("\n" + "=" * 70)
        print("Batch inference complete")
        print(f"Success: {success_count}")
        print(f"Failed : {fail_count}")
        print(f"Output : {OUTPUT_DIR}")
        print(f"Diffuse collection: {DIFFUSE_COLLECTION_DIR}")
        return

    print(f"[ERROR] invalid input path: {INPUT_PATH}")


if __name__ == "__main__":
    main()
