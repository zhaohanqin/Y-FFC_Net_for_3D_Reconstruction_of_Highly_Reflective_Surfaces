#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三图灰度剖面对比工具（基于 fringe_gray_analysis.py 扩展）

用途：在同一 ROI、同一行/列上，对比三条灰度剖面，例如：
  - 漫反射原图（参考）
  - 网络处理结果
  - 高动态（HDR）成像结果

相对参考图（默认漫反射原图）计算两条误差曲线并导出 PNG / NPY / MAT。

依赖：与 fringe_gray_analysis.py 相同（numpy, opencv, matplotlib, scipy）
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

# 同目录模块
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from fringe_gray_analysis import (  # noqa: E402
    ImageLoader,
    ProfileExtractor,
    ResultPlotter,
    ROISelector,
    _INTERACTIVE,
)


# ╔════════════════════════════════════════════════════════════════════╗
# ║              TripleComparisonPlotter 三图对比可视化                 ║
# ╚════════════════════════════════════════════════════════════════════╝

class TripleComparisonPlotter:
    """三图并排标注 + 三线剖面 + 双误差（相对漫反射参考）。"""

    COLORS = {
        "diffuse": "#DC2626",
        "network": "#2563EB",
        "hdr": "#9333EA",
    }

    def __init__(
        self,
        gray_diffuse,
        gray_network,
        gray_hdr,
        ext_diffuse,
        ext_network,
        ext_hdr,
        roi_bounds,
        label_diffuse="Diffuse",
        label_network="Network",
        label_hdr="HDR",
    ):
        self.grays = (gray_diffuse, gray_network, gray_hdr)
        self.exts = (ext_diffuse, ext_network, ext_hdr)
        self.labels = (label_diffuse, label_network, label_hdr)
        self.roi_bounds = roi_bounds
        self.ext_ref = ext_diffuse
        self.err_network = ext_network.profile - ext_diffuse.profile
        self.err_hdr = ext_hdr.profile - ext_diffuse.profile

    def plot_triple_images(self, save_path=None, dpi=200):
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        fig.subplots_adjust(wspace=0.12, left=0.03, right=0.98, top=0.90, bottom=0.08)
        keys = ("diffuse", "network", "hdr")
        for ax, gray, label, key in zip(axes, self.grays, self.labels, keys):
            ResultPlotter._annotate(
                ax,
                gray,
                self.roi_bounds,
                self.ext_ref.mode,
                self.ext_ref.abs_index,
                title=label,
            )
            ax.set_title(label, fontsize=13, fontweight="bold", color=self.COLORS[key])
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"[SAVE] 三原图标注: {save_path}")
        return fig

    def plot_single_annotated(self, which: str, save_path=None):
        idx_map = {"diffuse": 0, "network": 1, "hdr": 2}
        if which not in idx_map:
            raise ValueError(f"which 应为 diffuse/network/hdr，收到: {which}")
        i = idx_map[which]
        target_dpi = 100
        fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
        fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.06)
        ResultPlotter._annotate(
            ax,
            self.grays[i],
            self.roi_bounds,
            self.ext_ref.mode,
            self.ext_ref.abs_index,
            title=self.labels[i],
        )
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=target_dpi, facecolor="white")
            print(f"[SAVE] 单图标注({self.labels[i]}): {save_path} (1920×1080)")
        return fig

    def plot_comparison_curve(self, save_path=None):
        pos = self.ext_ref.pixel_positions
        p0, p1, p2 = (e.profile for e in self.exts)
        mode = self.ext_ref.mode
        ai = self.ext_ref.abs_index

        target_dpi = 100
        fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
        fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.09)

        l0, = ax.plot(
            pos, p0, color=self.COLORS["diffuse"], lw=3.5, alpha=0.9,
            label=self.labels[0], zorder=3,
        )
        l1, = ax.plot(
            pos, p1, color=self.COLORS["network"], lw=1.8,
            label=self.labels[1], zorder=4,
        )
        l2, = ax.plot(
            pos, p2, color=self.COLORS["hdr"], lw=1.8, ls="--",
            label=self.labels[2], zorder=4,
        )

        xlabel = "Pixel Coordinates (Column)" if mode == "row" else "Pixel Coordinates (Row)"
        title = f"Triple-Image Comparison — {'Row' if mode == 'row' else 'Column'} {ai}"
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Pixel Gray Value", fontsize=13)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax.set_xlim(pos[0], pos[-1])
        ax.set_ylim(-60, 270)
        ax.grid(True, alpha=0.2, ls="--")
        ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
        ax.tick_params(labelsize=11)

        stats = []
        for prof, lab in zip((p0, p1, p2), self.labels):
            stats.append(f"{lab[:12]:12s} Mean={np.mean(prof):.1f} Std={np.std(prof):.1f}")
        ax.text(
            0.02, 0.95, "\n".join(stats),
            transform=ax.transAxes, fontsize=10, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.85),
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=target_dpi, facecolor="white")
            print(f"[SAVE] 三图对比曲线: {save_path} (1920×1080)")
        return fig

    def plot_error_curves(self, save_path=None, ref_label_short="Diffuse"):
        pos = self.ext_ref.pixel_positions
        e1, e2 = self.err_network, self.err_hdr
        mode = self.ext_ref.mode
        ai = self.ext_ref.abs_index

        target_dpi = 100
        fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
        fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.09)

        ax.plot(pos, e1, color=self.COLORS["network"], lw=1.5,
                label=f"{self.labels[1]} − {ref_label_short}")
        ax.plot(pos, e2, color=self.COLORS["hdr"], lw=1.5, ls="--",
                label=f"{self.labels[2]} − {ref_label_short}")
        ax.axhline(y=0, color="black", lw=0.8)
        emax = max(np.abs(e1).max(), np.abs(e2).max(), 1.0) * 1.15
        ax.set_ylim(-emax, emax)

        xlabel = "Pixel Coordinates (Column)" if mode == "row" else "Pixel Coordinates (Row)"
        title = f"Error vs {ref_label_short} — {'Row' if mode == 'row' else 'Column'} {ai}"
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(f"Gray Difference (vs {ref_label_short})", fontsize=12)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax.set_xlim(pos[0], pos[-1])
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3, ls="--")
        ax.tick_params(labelsize=11)

        def _rmse_mae(err):
            return np.sqrt(np.mean(err ** 2)), np.mean(np.abs(err))

        r1, m1 = _rmse_mae(e1)
        r2, m2 = _rmse_mae(e2)
        txt = (
            f"{self.labels[1]}: RMSE={r1:.2f}, MAE={m1:.2f}\n"
            f"{self.labels[2]}: RMSE={r2:.2f}, MAE={m2:.2f}"
        )
        ax.text(
            0.02, 0.95, txt, transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=target_dpi, facecolor="white")
            print(f"[SAVE] 双误差曲线: {save_path} (1920×1080)")
        return fig


class TripleDataExporter:
    @staticmethod
    def export_npy(data: dict, save_path: str):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.save(save_path, data)
        print(f"[SAVE] NPY: {save_path}")

    @staticmethod
    def export_mat(
        ext_diffuse,
        ext_network,
        ext_hdr,
        err_network,
        err_hdr,
        roi_bounds,
        img_shape,
        labels,
        save_path,
    ):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        savemat(
            save_path,
            {
                "diffuse_profile": ext_diffuse.profile,
                "network_profile": ext_network.profile,
                "hdr_profile": ext_hdr.profile,
                "error_network_vs_diffuse": err_network,
                "error_hdr_vs_diffuse": err_hdr,
                "pixel_positions": ext_diffuse.pixel_positions,
                "extract_mode": ext_diffuse.mode,
                "abs_line_index": ext_diffuse.abs_index,
                "roi_row_start": roi_bounds[0],
                "roi_row_end": roi_bounds[1],
                "roi_col_start": roi_bounds[2],
                "roi_col_end": roi_bounds[3],
                "image_height": img_shape[0],
                "image_width": img_shape[1],
                "label_diffuse": labels[0],
                "label_network": labels[1],
                "label_hdr": labels[2],
                "network_rmse": float(np.sqrt(np.mean(err_network ** 2))),
                "network_mae": float(np.mean(np.abs(err_network))),
                "hdr_rmse": float(np.sqrt(np.mean(err_hdr ** 2))),
                "hdr_mae": float(np.mean(np.abs(err_hdr))),
            },
        )
        print(f"[SAVE] MAT: {save_path}")


class TripleImageComparator:
    """
    三图对比主控器：共享 ROI 与行/列，输出剖面、误差与数据文件。

    输出（basename 由 output_basename 指定）：
      {basename}_images.png       — 三原图并排
      {basename}_compare.png      — 三线剖面
      {basename}_error.png        — 相对漫反射的两条误差
      {basename}_diffuse.png 等   — 单图标注
      {basename}.npy / .mat
    """

    def __init__(
        self,
        path_diffuse: str,
        path_network: str,
        path_hdr: str,
        output_dir: str = "./output_triple",
        output_basename: str = "triple_compare",
        label_diffuse: str = "漫反射原图",
        label_network: str = "网络处理",
        label_hdr: str = "高动态",
    ):
        self.output_dir = output_dir
        self.output_basename = output_basename
        self.labels = (label_diffuse, label_network, label_hdr)

        self.loader_diffuse = ImageLoader(path_diffuse)
        self.loader_network = ImageLoader(path_network)
        self.loader_hdr = ImageLoader(path_hdr)

        shapes = (
            self.loader_diffuse.shape,
            self.loader_network.shape,
            self.loader_hdr.shape,
        )
        if len(set(shapes)) != 1:
            raise ValueError(f"三图尺寸必须一致，当前: {shapes}")

        self.selector = ROISelector(self.loader_diffuse.gray_img)
        self.ext_diffuse = None
        self.ext_network = None
        self.ext_hdr = None

    def run_interactive(self):
        self._header()
        self.selector.select_interactive()
        self._do()

    def run_manual(self, mode="row", line_index=0, roi=None):
        self._header()
        self.selector.select_manual(mode, line_index, roi)
        self._do()

    def _header(self):
        print("=" * 62)
        print("  条纹灰度分析 — 三图对比模式")
        print("  Triple-Image Comparison")
        print("=" * 62)
        print(f"  [1] 漫反射 : {self.loader_diffuse.info()}")
        print(f"  [2] 网络   : {self.loader_network.info()}")
        print(f"  [3] 高动态 : {self.loader_hdr.info()}")
        print()

    def _do(self):
        rb = self.selector.roi_bounds
        m = self.selector.mode
        li = self.selector.line_index
        off = (rb[0], rb[2])

        roi_d = self.loader_diffuse.gray_img[rb[0]:rb[1], rb[2]:rb[3]]
        roi_n = self.loader_network.gray_img[rb[0]:rb[1], rb[2]:rb[3]]
        roi_h = self.loader_hdr.gray_img[rb[0]:rb[1], rb[2]:rb[3]]

        self.ext_diffuse = ProfileExtractor(roi_d, m, li, off)
        self.ext_network = ProfileExtractor(roi_n, m, li, off)
        self.ext_hdr = ProfileExtractor(roi_h, m, li, off)

        print(f"\n【{self.labels[0]}】 {self.ext_diffuse.summary()}")
        print(f"【{self.labels[1]}】 {self.ext_network.summary()}")
        print(f"【{self.labels[2]}】 {self.ext_hdr.summary()}")

        plotter = TripleComparisonPlotter(
            self.loader_diffuse.gray_img,
            self.loader_network.gray_img,
            self.loader_hdr.gray_img,
            self.ext_diffuse,
            self.ext_network,
            self.ext_hdr,
            rb,
            *self.labels,
        )

        od = self.output_dir
        bn = self.output_basename
        os.makedirs(od, exist_ok=True)

        paths = {
            "images": os.path.join(od, f"{bn}_images.png"),
            "compare": os.path.join(od, f"{bn}_compare.png"),
            "error": os.path.join(od, f"{bn}_error.png"),
            "diffuse": os.path.join(od, f"{bn}_diffuse.png"),
            "network": os.path.join(od, f"{bn}_network.png"),
            "hdr": os.path.join(od, f"{bn}_hdr.png"),
        }

        for key, path in [
            ("images", paths["images"]),
            ("compare", paths["compare"]),
            ("error", paths["error"]),
        ]:
            if key == "images":
                plt.close(plotter.plot_triple_images(path))
            elif key == "compare":
                plt.close(plotter.plot_comparison_curve(path))
            else:
                plt.close(plotter.plot_error_curves(path, ref_label_short=self.labels[0]))

        for role, path in [("diffuse", paths["diffuse"]), ("network", paths["network"]), ("hdr", paths["hdr"])]:
            plt.close(plotter.plot_single_annotated(role, path))

        err_n = plotter.err_network
        err_h = plotter.err_hdr
        npy_path = os.path.join(od, f"{bn}.npy")
        TripleDataExporter.export_npy(
            {
                "diffuse_profile": self.ext_diffuse.profile,
                "network_profile": self.ext_network.profile,
                "hdr_profile": self.ext_hdr.profile,
                "error_network_vs_diffuse": err_n,
                "error_hdr_vs_diffuse": err_h,
                "pixel_positions": self.ext_diffuse.pixel_positions,
                "labels": self.labels,
            },
            npy_path,
        )
        mat_path = os.path.join(od, f"{bn}.mat")
        TripleDataExporter.export_mat(
            self.ext_diffuse,
            self.ext_network,
            self.ext_hdr,
            err_n,
            err_h,
            rb,
            self.loader_diffuse.shape,
            self.labels,
            mat_path,
        )

        print()
        print("-" * 55)
        print("  三图对比完成！主要输出：")
        for p in paths.values():
            print(f"    {p}")
        print(f"    {npy_path}")
        print(f"    {mat_path}")
        print("-" * 55)
        print("=" * 62)


# ╔════════════════════════════════════════════════════════════════════╗
# ║                          程序入口                                  ║
# ╚════════════════════════════════════════════════════════════════════╝

# ============================================================
# ★★★ 用户可修改区域 ★★★
# ============================================================

# 三张图像路径（请按实际数据修改）
PATH_DIFFUSE = "./data2/targets/0.bmp"   # 漫反射原图 / 参考
PATH_NETWORK = "./data2/targets/0.bmp"   # 网络处理结果
PATH_HDR     = "./data2/inputs/0.bmp"    # 高动态（或含高光）结果

LABEL_DIFFUSE = "漫反射原图"
LABEL_NETWORK = "网络处理"
LABEL_HDR     = "高动态"

OUTPUT_DIR = "./phase_analysis/output/triple_demo"
OUTPUT_BASENAME = "triple_row72"

EXTRACT_MODE = "row"      # "row" 或 "col"
LINE_INDEX = 72           # ROI 内行/列索引；全图时即绝对行号
ROI = None                # (r0,r1,c0,c1) 或 None=全图

# True：鼠标框选 ROI + 滑块选行/列（需图形界面）；False：使用上面 EXTRACT_MODE / LINE_INDEX / ROI
USE_INTERACTIVE = False

# ============================================================


def main():
    comp = TripleImageComparator(
        path_diffuse=PATH_DIFFUSE,
        path_network=PATH_NETWORK,
        path_hdr=PATH_HDR,
        output_dir=OUTPUT_DIR,
        output_basename=OUTPUT_BASENAME,
        label_diffuse=LABEL_DIFFUSE,
        label_network=LABEL_NETWORK,
        label_hdr=LABEL_HDR,
    )
    if USE_INTERACTIVE and _INTERACTIVE:
        comp.run_interactive()
    else:
        if USE_INTERACTIVE and not _INTERACTIVE:
            print("[WARN] 无图形界面，已回退为手动参数模式。")
        comp.run_manual(mode=EXTRACT_MODE, line_index=LINE_INDEX, roi=ROI)


if __name__ == "__main__":
    main()
