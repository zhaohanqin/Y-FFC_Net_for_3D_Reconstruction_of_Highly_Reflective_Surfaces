"""
Y-FFC vs DC-UNet vs HDRNet：5 张验证样本剖面对比（row72，与 run_val_triple_compare 一致）。

用法（仓库根目录）:
  python phase_analysis/run_model_compare_yffc.py
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from fringe_gray_analysis import ImageLoader, ProfileExtractor, ROISelector  # noqa: E402
from fringe_gray_analysis_triple import TripleImageComparator  # noqa: E402

OUT_ROOT = os.path.join(_REPO_ROOT, "phase_analysis", "output", "model_compare_yffc_dc_hdr")
SAMPLE_LIST = os.path.join(
    _REPO_ROOT,
    "phase_analysis",
    "output",
    "val_inference_preview",
    "preview_samples.txt",
)
INPUT_DIR = os.path.join(_REPO_ROOT, "data2", "inputs")
TARGET_DIR = os.path.join(_REPO_ROOT, "data2", "targets")
YFFC_DIR = os.path.join(OUT_ROOT, "inference", "yffc")
DC_SRC = os.path.join(
    _REPO_ROOT, "comporison_model", "experiments", "inference_val_seed42", "dcunet"
)
HDR_SRC = os.path.join(
    _REPO_ROOT, "comporison_model", "experiments", "inference_val_seed42", "hdrnet"
)
DC_DIR = os.path.join(OUT_ROOT, "inference", "dcunet")
HDR_DIR = os.path.join(OUT_ROOT, "inference", "hdrnet")

EXTRACT_MODE = "row"
LINE_INDEX = 72
OUTPUT_BASENAME = "triple_row72"
MAX_SAMPLES = 5

COLORS = {
    "gt": "#DC2626",
    "yffc": "#059669",
    "dc": "#2563EB",
    "hdr": "#9333EA",
}


def _read_lines(path: str, limit: int) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[:limit] if limit > 0 else lines


def _metrics(err: np.ndarray) -> dict:
    err = np.asarray(err, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
    }


def _ensure_inference_copies(basename: str) -> None:
    os.makedirs(DC_DIR, exist_ok=True)
    os.makedirs(HDR_DIR, exist_ok=True)
    for src_dir, dst_dir in ((DC_SRC, DC_DIR), (HDR_SRC, HDR_DIR)):
        src = os.path.join(src_dir, basename)
        dst = os.path.join(dst_dir, basename)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"缺少对比模型推理结果: {src}")
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)


def _extract_profile(path: str, mode: str, line_index: int):
    loader = ImageLoader(path)
    selector = ROISelector(loader.gray_img)
    selector.select_manual(mode, line_index, None)
    rb = selector.roi_bounds
    roi = loader.gray_img[rb[0] : rb[1], rb[2] : rb[3]]
    ext = ProfileExtractor(roi, selector.mode, selector.line_index, (rb[0], rb[2]))
    return loader.gray_img, ext, rb


def _plot_quad_profiles(
    gt_ext,
    profiles: dict[str, np.ndarray],
    labels: dict[str, str],
    save_path: str,
    metrics: dict[str, dict],
):
    pos = gt_ext.pixel_positions
    p_gt = gt_ext.profile
    mode = gt_ext.mode
    ai = gt_ext.abs_index

    fig, axes = plt.subplots(2, 1, figsize=(19.2, 14), gridspec_kw={"height_ratios": [2.2, 1]})
    ax0, ax1 = axes
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.07, hspace=0.22)

    ax0.plot(pos, p_gt, color=COLORS["gt"], lw=3.5, alpha=0.95, label=labels["gt"], zorder=3)
    for key in ("yffc", "dc", "hdr"):
        ax0.plot(
            pos,
            profiles[key],
            color=COLORS[key],
            lw=1.8,
            ls="--" if key == "hdr" else "-",
            label=labels[key],
            zorder=4,
        )
    xlabel = "Pixel Coordinates (Column)" if mode == "row" else "Pixel Coordinates (Row)"
    ax0.set_xlabel(xlabel, fontsize=12)
    ax0.set_ylabel("Pixel Gray Value", fontsize=13)
    ax0.set_title(
        f"Four-Model Profile — {'Row' if mode == 'row' else 'Column'} {ai}",
        fontsize=15,
        fontweight="bold",
    )
    ax0.set_xlim(pos[0], pos[-1])
    ax0.set_ylim(-60, 270)
    ax0.grid(True, alpha=0.2, ls="--")
    ax0.legend(loc="upper right", fontsize=11, framealpha=0.9)

    for key, color in (("yffc", COLORS["yffc"]), ("dc", COLORS["dc"]), ("hdr", COLORS["hdr"])):
        err = profiles[key] - p_gt
        ax1.plot(pos, err, color=color, lw=1.5, label=f"{labels[key]} − GT")
    ax1.axhline(0, color="black", lw=0.8)
    emax = max(
        np.abs(profiles[k] - p_gt).max() for k in ("yffc", "dc", "hdr")
    )
    ax1.set_ylim(-emax * 1.15 - 1e-6, emax * 1.15 + 1e-6)
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("Gray Difference (vs GT)", fontsize=12)
    ax1.set_title("Error vs GT (漫反射参考)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.legend(loc="upper right", fontsize=10)

    txt = "\n".join(
        f"{labels[k]:10s} RMSE={metrics[k]['rmse']:.2f}  MAE={metrics[k]['mae']:.2f}"
        for k in ("yffc", "dc", "hdr")
    )
    ax1.text(
        0.02,
        0.95,
        txt,
        transform=ax1.transAxes,
        fontsize=10,
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow", ec="gray", alpha=0.85),
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] quad profile -> {save_path}")


def _plot_preview_row(basename: str, out_path: str):
    paths = [
        ("Input", os.path.join(INPUT_DIR, basename)),
        ("Y-FFC", os.path.join(YFFC_DIR, basename)),
        ("DC-UNet", os.path.join(DC_DIR, basename)),
        ("HDRNet", os.path.join(HDR_DIR, basename)),
        ("GT", os.path.join(TARGET_DIR, basename)),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    for ax, (title, p) in zip(axes, paths):
        img = np.array(Image.open(p).convert("L"))
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle(basename, fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] preview_row -> {out_path}")


def _run_triple(gt: str, net: str, hdr: str, out_dir: str, ln_net: str, ln_hdr: str):
    os.makedirs(out_dir, exist_ok=True)
    comp = TripleImageComparator(
        path_diffuse=gt,
        path_network=net,
        path_hdr=hdr,
        output_dir=out_dir,
        output_basename=OUTPUT_BASENAME,
        label_diffuse="GT漫反射",
        label_network=ln_net,
        label_hdr=ln_hdr,
    )
    comp.run_manual(mode=EXTRACT_MODE, line_index=LINE_INDEX, roi=None)


def _process_one(basename: str) -> dict | None:
    gt = os.path.join(TARGET_DIR, basename)
    yffc = os.path.join(YFFC_DIR, basename)
    dc = os.path.join(DC_DIR, basename)
    hdr = os.path.join(HDR_DIR, basename)
    for p in (gt, yffc, dc, hdr):
        if not os.path.isfile(p):
            print(f"[SKIP] 缺少: {p}")
            return None

    stem = os.path.splitext(basename)[0]
    sample_dir = os.path.join(OUT_ROOT, stem)

    _run_triple(gt, yffc, dc, os.path.join(sample_dir, "triple_vs_dc"), "Y-FFC", "DC-UNet")
    _run_triple(gt, yffc, hdr, os.path.join(sample_dir, "triple_vs_hdr"), "Y-FFC", "HDRNet")

    _, gt_ext, _ = _extract_profile(gt, EXTRACT_MODE, LINE_INDEX)
    profiles = {}
    for key, path in (("yffc", yffc), ("dc", dc), ("hdr", hdr)):
        _, ext, _ = _extract_profile(path, EXTRACT_MODE, LINE_INDEX)
        profiles[key] = ext.profile

    m = {k: _metrics(profiles[k] - gt_ext.profile) for k in ("yffc", "dc", "hdr")}
    _plot_quad_profiles(
        gt_ext,
        profiles,
        {"gt": "GT", "yffc": "Y-FFC", "dc": "DC-UNet", "hdr": "HDRNet"},
        os.path.join(sample_dir, "quad_profile_compare.png"),
        m,
    )
    _plot_preview_row(basename, os.path.join(sample_dir, "preview_row.png"))

    row = {"basename": basename}
    for prefix, key in (("yffc", "yffc"), ("dc", "dc"), ("hdr", "hdr")):
        row[f"{prefix}_rmse"] = m[key]["rmse"]
        row[f"{prefix}_mae"] = m[key]["mae"]
        row[f"{prefix}_max_abs"] = m[key]["max_abs"]
    return row


def _write_summary(rows: list[dict], csv_path: str) -> None:
    fields = [
        "basename",
        "yffc_rmse", "yffc_mae", "yffc_max_abs",
        "dc_rmse", "dc_mae", "dc_max_abs",
        "hdr_rmse", "hdr_mae", "hdr_max_abs",
    ]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def mean(key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    print(f"[SAVE] {csv_path} ({len(rows)} rows)")
    for name, key in (("Y-FFC", "yffc_rmse"), ("DC-UNet", "dc_rmse"), ("HDRNet", "hdr_rmse")):
        print(f"  平均 {name} RMSE={mean(key):.3f} MAE={mean(key.replace('rmse', 'mae')):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default=SAMPLE_LIST)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    args = parser.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    names = _read_lines(args.list, args.max_samples)
    manifest = os.path.join(OUT_ROOT, "selected_samples.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")

    rows = []
    for i, name in enumerate(names, 1):
        print(f"\n[{i}/{len(names)}] {name}")
        _ensure_inference_copies(name)
        row = _process_one(name)
        if row:
            rows.append(row)

    _write_summary(rows, os.path.join(OUT_ROOT, "summary.csv"))


if __name__ == "__main__":
    main()
