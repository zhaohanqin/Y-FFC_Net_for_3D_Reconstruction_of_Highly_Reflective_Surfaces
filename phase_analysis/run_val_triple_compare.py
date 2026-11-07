"""
验证集批量三图对比：GT 漫反射 vs DC-UNet vs HDRNet（fringe_gray_analysis_triple）。

用法（仓库根目录）:
  python phase_analysis/run_val_triple_compare.py --preview-only
  python phase_analysis/run_val_triple_compare.py --full
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from fringe_gray_analysis_triple import TripleImageComparator  # noqa: E402

VAL_LIST = os.path.join(_REPO_ROOT, "comporison_model", "experiments", "data2_split_seed42", "val_files.txt")
PREVIEW_LIST = os.path.join(
    _REPO_ROOT, "phase_analysis", "output", "val_inference_preview", "preview_samples.txt")
TARGET_DIR = os.path.join(_REPO_ROOT, "data2", "targets")
DC_DIR = os.path.join(_REPO_ROOT, "comporison_model", "experiments", "inference_val_seed42", "dcunet")
HDR_DIR = os.path.join(_REPO_ROOT, "comporison_model", "experiments", "inference_val_seed42", "hdrnet")
OUT_ROOT = os.path.join(_REPO_ROOT, "phase_analysis", "output", "val_triple_seed42")
SUMMARY_CSV = os.path.join(OUT_ROOT, "summary.csv")

EXTRACT_MODE = "row"
LINE_INDEX = 72


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _metrics(err: np.ndarray) -> dict:
    err = np.asarray(err, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
    }


def _process_one(basename: str) -> dict | None:
    gt = os.path.join(TARGET_DIR, basename)
    dc = os.path.join(DC_DIR, basename)
    hdr = os.path.join(HDR_DIR, basename)
    for p in (gt, dc, hdr):
        if not os.path.isfile(p):
            print(f"[SKIP] 缺少文件: {p}")
            return None

    stem = os.path.splitext(basename)[0]
    out_dir = os.path.join(OUT_ROOT, stem)
    comp = TripleImageComparator(
        path_diffuse=gt,
        path_network=dc,
        path_hdr=hdr,
        output_dir=out_dir,
        output_basename=f"triple_{EXTRACT_MODE}{LINE_INDEX}",
        label_diffuse="GT漫反射",
        label_network="DC-UNet",
        label_hdr="HDRNet",
    )
    comp.run_manual(mode=EXTRACT_MODE, line_index=LINE_INDEX, roi=None)

    err_dc = comp.ext_network.profile - comp.ext_diffuse.profile
    err_hdr = comp.ext_hdr.profile - comp.ext_diffuse.profile
    m_dc = _metrics(err_dc)
    m_hdr = _metrics(err_hdr)
    return {
        "basename": basename,
        "dc_rmse": m_dc["rmse"],
        "dc_mae": m_dc["mae"],
        "dc_max_abs": m_dc["max_abs"],
        "hdr_rmse": m_hdr["rmse"],
        "hdr_mae": m_hdr["mae"],
        "hdr_max_abs": m_hdr["max_abs"],
    }


def _write_summary(rows: list[dict]) -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    fields = [
        "basename", "dc_rmse", "dc_mae", "dc_max_abs",
        "hdr_rmse", "hdr_mae", "hdr_max_abs",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def mean(key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    print(f"[SAVE] {SUMMARY_CSV} ({len(rows)} rows)")
    print(
        f"  均值 DC-UNet vs GT: RMSE={mean('dc_rmse'):.3f} MAE={mean('dc_mae'):.3f}"
    )
    print(
        f"  均值 HDRNet vs GT:  RMSE={mean('hdr_rmse'):.3f} MAE={mean('hdr_mae'):.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--preview-only", action="store_true", help="仅 preview_samples.txt")
    g.add_argument("--full", action="store_true", help="全部 val_files.txt")
    args = parser.parse_args()

    if args.preview_only:
        if not os.path.isfile(PREVIEW_LIST):
            raise FileNotFoundError(f"缺少预览列表，请先运行 infer_data2_val.py: {PREVIEW_LIST}")
        files = _read_lines(PREVIEW_LIST)
    else:
        files = _read_lines(VAL_LIST)

    rows = []
    for i, name in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {name}")
        row = _process_one(name)
        if row:
            rows.append(row)

    _write_summary(rows)


if __name__ == "__main__":
    main()
