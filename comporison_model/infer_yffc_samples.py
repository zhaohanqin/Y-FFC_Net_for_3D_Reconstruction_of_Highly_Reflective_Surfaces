"""
对指定 data2 样本做 Y-FFC 滑窗推理（与 DC/HDR 对比实验一致：无 TTA）。

用法（仓库根目录）:
  python comporison_model/infer_yffc_samples.py
  python comporison_model/infer_yffc_samples.py --list path/to/names.txt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from process_images import (  # noqa: E402
    _normalize_hw,
    _sanitize_output_dict,
    load_image,
    load_model,
    resolve_patch_size_from_checkpoint,
    sliding_window_inference,
)

DEFAULT_LIST = os.path.join(
    _REPO_ROOT,
    "phase_analysis",
    "output",
    "val_inference_preview",
    "preview_samples.txt",
)
INPUT_DIR = os.path.join(_REPO_ROOT, "data2", "inputs")
YFFC_CKPT = os.path.join(
    _REPO_ROOT,
    "experiments",
    "yffc_v2_20261102_192433",
    "models",
    "best_model.pth",
)
OUT_ROOT = os.path.join(
    _REPO_ROOT,
    "phase_analysis",
    "output",
    "model_compare_yffc_dc_hdr",
    "inference",
    "yffc",
)

PATCH_SIZE = (128, 160)
PATCH_OVERLAP = 64
MAX_SAMPLES = 5


def _read_lines(path: str, limit: int) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[:limit] if limit > 0 else lines


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze().detach().cpu().numpy()
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _save_diffuse(output_dict: dict, save_path: str) -> None:
    diffuse = output_dict["r"]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    Image.fromarray(_tensor_to_uint8(diffuse)).save(save_path)


@torch.no_grad()
def _infer_one(model, input_path: str, device: torch.device, patch_size: tuple[int, int]):
    inp, _ = load_image(input_path)
    inp = inp.to(device)
    ph, pw = _normalize_hw(patch_size)
    h, w = inp.shape[-2:]
    if (h, w) == (ph, pw):
        return _sanitize_output_dict(model(inp))
    return _sanitize_output_dict(
        sliding_window_inference(
            model, inp, device, patch_size=patch_size, overlap=PATCH_OVERLAP, use_tta=False
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default=DEFAULT_LIST, help="每行一个 bmp 文件名")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--out-dir", default=OUT_ROOT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not os.path.isfile(YFFC_CKPT):
        raise FileNotFoundError(f"缺少 Y-FFC 权重，请先迁移实验目录: {YFFC_CKPT}")
    if not os.path.isfile(args.list):
        raise FileNotFoundError(f"缺少样本列表: {args.list}")

    names = _read_lines(args.list, args.max_samples)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  样本数: {len(names)}")
    print(f"权重: {YFFC_CKPT}")

    model = load_model(YFFC_CKPT, device, prefer_ema=True)
    patch_size = resolve_patch_size_from_checkpoint(model, PATCH_SIZE)
    os.makedirs(args.out_dir, exist_ok=True)

    for name in tqdm(names, desc="yffc-infer"):
        inp_path = os.path.join(INPUT_DIR, name)
        if not os.path.isfile(inp_path):
            tqdm.write(f"[SKIP] 缺少输入: {inp_path}")
            continue
        out = _infer_one(model, inp_path, device, patch_size)
        _save_diffuse(out, os.path.join(args.out_dir, name))

    manifest = os.path.join(os.path.dirname(args.out_dir), "..", "selected_samples.txt")
    manifest = os.path.normpath(manifest)
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")
    print(f"[SAVE] selected_samples -> {manifest}")


if __name__ == "__main__":
    main()
