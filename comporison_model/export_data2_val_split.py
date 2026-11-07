"""
复现 train_dcunet / train_hdrnet 的 data2 验证集划分（80/20, seed=42）。

输出:
  comporison_model/experiments/data2_split_seed42/val_files.txt
  comporison_model/experiments/data2_split_seed42/train_files.txt
"""

import os
import sys

import torch
from torch.utils.data import random_split

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from Y_FFC_unified import FringeDataset  # noqa: E402

INPUT_DIR = os.path.join(_REPO_ROOT, "data2", "inputs")
TARGET_DIR = os.path.join(_REPO_ROOT, "data2", "targets")
TRAIN_SPLIT = 0.8
SEED = 42
OUT_DIR = os.path.join(_SCRIPT_DIR, "experiments", "data2_split_seed42")


def main():
    dataset = FringeDataset(INPUT_DIR, TARGET_DIR, resize=(128, 160))
    n = len(dataset)
    train_size = int(TRAIN_SPLIT * n)
    val_size = n - train_size
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

    os.makedirs(OUT_DIR, exist_ok=True)

    def write_subset(subset, path):
        with open(path, "w", encoding="utf-8") as f:
            for idx in subset.indices:
                f.write(dataset.files[idx] + "\n")

    train_path = os.path.join(OUT_DIR, "train_files.txt")
    val_path = os.path.join(OUT_DIR, "val_files.txt")
    write_subset(train_ds, train_path)
    write_subset(val_ds, val_path)

    print(f"数据集总数: {n}")
    print(f"训练: {train_size} -> {train_path}")
    print(f"验证: {val_size} -> {val_path}")


if __name__ == "__main__":
    main()
