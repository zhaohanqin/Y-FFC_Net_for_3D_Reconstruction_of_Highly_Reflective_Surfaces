"""
对 data2 验证集（seed=42）批量推理 DC-UNet / HDRNet，并生成预览拼图。

用法（仓库根目录）:
  python comporison_model/infer_data2_val.py
  python comporison_model/infer_data2_val.py --max-samples 8   # 仅试跑前 N 张
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from process_dcunet_images import (  # noqa: E402
    load_model as load_dcunet,
    load_image,
    run_model as run_dcunet,
    sliding_window_inference as slide_dcunet,
)
from process_hdrnet_images import (  # noqa: E402
    load_model as load_hdrnet,
    run_model as run_hdrnet,
    sliding_window_inference as slide_hdrnet,
)

VAL_LIST = os.path.join(_SCRIPT_DIR, "experiments", "data2_split_seed42", "val_files.txt")
INPUT_DIR = os.path.join(_REPO_ROOT, "data2", "inputs")
TARGET_DIR = os.path.join(_REPO_ROOT, "data2", "targets")
DCUNET_CKPT = os.path.join(
    _SCRIPT_DIR, "experiments", "dcunet_20261103_113944", "models", "best_model.pth")
HDRNET_CKPT = os.path.join(
    _SCRIPT_DIR, "experiments", "hdrnet_20261103_163735", "models", "best_model.pth")
OUT_ROOT = os.path.join(_SCRIPT_DIR, "experiments", "inference_val_seed42")
PREVIEW_DIR = os.path.join(_REPO_ROOT, "phase_analysis", "output", "val_inference_preview")

PATCH_SIZE = (128, 160)
PATCH_OVERLAP = 64
PREVIEW_COUNT = 8
PREVIEW_STEMS = ("0",)  # 若在 val 中则优先加入预览


def _read_val_list(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _tensor_to_uint8(t) -> np.ndarray:
    arr = t.squeeze().detach().cpu().numpy()
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _save_tensor(t, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    Image.fromarray(_tensor_to_uint8(t)).save(save_path)


@torch.no_grad()
def _infer_one(model, run_fn, slide_fn, input_path, device):
    inp, _ = load_image(input_path)
    inp = inp.to(device)
    h, w = inp.shape[-2:]
    ph, pw = PATCH_SIZE
    if (h, w) == (ph, pw):
        return run_fn(model, inp)
    return slide_fn(model, inp, device, PATCH_SIZE, PATCH_OVERLAP)


def _pick_preview_names(val_files: list[str], n: int) -> list[str]:
    by_stem = {os.path.splitext(f)[0]: f for f in val_files}
    chosen: list[str] = []
    for stem in PREVIEW_STEMS:
        if stem in by_stem and by_stem[stem] not in chosen:
            chosen.append(by_stem[stem])
    for f in val_files:
        if len(chosen) >= n:
            break
        if f not in chosen:
            chosen.append(f)
    return chosen[:n]


def _build_preview_grid(preview_files: list[str], dc_dir: str, hdr_dir: str, out_path: str) -> None:
    n = len(preview_files)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.2 * n))
    if n == 1:
        axes = np.array([axes])
    cols = ["Input", "DC-UNet", "HDRNet", "GT"]
    for row, name in enumerate(preview_files):
        inp_p = os.path.join(INPUT_DIR, name)
        gt_p = os.path.join(TARGET_DIR, name)
        dc_p = os.path.join(dc_dir, name)
        hdr_p = os.path.join(hdr_dir, name)
        paths = [inp_p, dc_p, hdr_p, gt_p]
        for col, (ax, title, p) in enumerate(zip(axes[row], cols, paths)):
            img = np.array(Image.open(p).convert("L"))
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{title}\n{name}" if col == 0 else title, fontsize=9)
            ax.axis("off")
    plt.suptitle("Validation preview: Input | DC-UNet | HDRNet | GT", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] preview grid -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=0, help="0=全部验证集")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not os.path.isfile(VAL_LIST):
        raise FileNotFoundError(f"请先运行 export_data2_val_split.py，缺少: {VAL_LIST}")

    val_files = _read_val_list(VAL_LIST)
    if args.max_samples > 0:
        val_files = val_files[: args.max_samples]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  验证集样本: {len(val_files)}")

    dc_dir = os.path.join(OUT_ROOT, "dcunet")
    hdr_dir = os.path.join(OUT_ROOT, "hdrnet")
    os.makedirs(dc_dir, exist_ok=True)
    os.makedirs(hdr_dir, exist_ok=True)

    print("加载 DC-UNet...")
    dc_model = load_dcunet(DCUNET_CKPT, device)
    print("加载 HDRNet...")
    hdr_model = load_hdrnet(HDRNET_CKPT, device)

    for name in tqdm(val_files, desc="infer"):
        inp_path = os.path.join(INPUT_DIR, name)
        if not os.path.isfile(inp_path):
            tqdm.write(f"[SKIP] 缺少输入: {inp_path}")
            continue
        dc_out = _infer_one(dc_model, run_dcunet, slide_dcunet, inp_path, device)
        hdr_out = _infer_one(hdr_model, run_hdrnet, slide_hdrnet, inp_path, device)
        _save_tensor(dc_out, os.path.join(dc_dir, name))
        _save_tensor(hdr_out, os.path.join(hdr_dir, name))

    full_val = _read_val_list(VAL_LIST)
    preview_names = _pick_preview_names(full_val, PREVIEW_COUNT)
    grid_path = os.path.join(PREVIEW_DIR, "preview_grid.png")
    _build_preview_grid(preview_names, dc_dir, hdr_dir, grid_path)

    manifest = os.path.join(PREVIEW_DIR, "preview_samples.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        for n in preview_names:
            f.write(n + "\n")
    print(f"[SAVE] preview list -> {manifest}")


if __name__ == "__main__":
    main()
