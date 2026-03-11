#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合版：绝对相位生成与可视化工具
Integrated Absolute Phase Generation & Visualization Tool

整合思路：
  ① 参考 fringe_gray_analysis.py 的方式 —— 确定掩码（ROI）、选择要分析的行或列
  ② 参考 单频相位处理工具.py 的读图方式 —— 读取多频率相移条纹图像
  ③ 复用 单频相位处理工具.py → test_multiple_frequencies2 的核心逻辑 —— 生成绝对相位
  ④ 参考 绝对相位可视化工具.py 的方法 —— 对指定行/列的绝对相位进行可视化

借鉴来源详细说明：
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 步骤                          借鉴来源                                   │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ 1. 掩码(ROI)生成 & 行列选择    fringe_gray_analysis.py                    │
  │    - ROISelector 类            （交互式鼠标框选 / 手动指定 ROI）            │
  │    - 指定 mode="row"/"col"   （EXTRACT_MODE / LINE_INDEX 参数风格）      │
  │    - ROI → bool 掩码          （ROI 矩形区域内为 True，外部为 False）      │
  │                                                                        │
  │ 2. 图像读取                   单频相位处理工具.py                          │
  │    - ImageLoader 类           (load_single_frequency_images 函数)        │
  │    - 自动检测扩展名 / 频率数  (detect_num_frequencies 函数)                │
  │                                                                        │
  │ 3. 绝对相位计算               单频相位处理工具.py                          │
  │    - test_multiple_frequencies2 函数的核心流程：                           │
  │      a) 三频率包裹相位计算     (calculate_single_frequency_wrapped_phase)  │
  │      b) 两两相位差 → 外差     (calculate_phase_difference)               │
  │      c) 逐级参考展开          (unwrap_phase_with_reference)              │
  │                                                                        │
  │ 4. 可视化                     绝对相位可视化工具.py                        │
  │    - 相位归一化到 [0,2π]      (PhaseNormalizer.normalize_phase_to_2pi)    │
  │    - 单行/列相位曲线图        (PhaseVisualizer.visualize_phase_line_profile│
  │      的绘图逻辑)                                                         │
  │    - 2D 相位伪彩色图          (同上方法中的 imshow 部分)                    │
  │    - 3D 表面图                (PhaseVisualizer.visualize_phase_3d_surface) │
  └──────────────────────────────────────────────────────────────────────────┘

作者: Claude (Anthropic)
日期: 2026-03-10
"""

import os
import re
import sys
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────
# matplotlib 后端自动选择（借鉴 fringe_gray_analysis.py）
# ────────────────────────────────────────────────────────
_INTERACTIVE = False
try:
    matplotlib.use("TkAgg")
    _fig_test = plt.figure()
    plt.close(_fig_test)
    _INTERACTIVE = True
except Exception:
    matplotlib.use("Agg")
    _INTERACTIVE = False

if _INTERACTIVE:
    from matplotlib.widgets import RectangleSelector, RadioButtons, Slider, Button

# ────────────────────────────────────────────────────────
# 中文字体配置（综合三个脚本的方案）
# ────────────────────────────────────────────────────────
_CJK_FONTS = [
    "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
    "Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans",
]
for _f in _CJK_FONTS:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  步骤一：图像读取（借鉴 单频相位处理工具.py → ImageLoader 类）         ║
# ╚════════════════════════════════════════════════════════════════════════╝

def detect_image_extension(folder_path, orientation='h', index=1):
    """
    自动检测文件夹中图像文件的扩展名。
    来源: 单频相位处理工具.py → ImageLoader.detect_image_extension
    """
    for ext in ['.bmp', '.png', '.jpg', '.jpeg']:
        path = os.path.join(folder_path, f"{orientation}{index}{ext}")
        if os.path.exists(path):
            return ext
    return None


def detect_num_frequencies(folder_path, phase_shift_step=4,
                           orientation='h', image_ext=None, verbose=True):
    """
    自动检测给定方向下包含多少个频率段。
    来源: 单频相位处理工具.py → ImageLoader.detect_num_frequencies
    """
    N = phase_shift_step
    if image_ext is None:
        image_ext = detect_image_extension(folder_path, orientation, index=1)
        if image_ext is None:
            raise FileNotFoundError(
                f"在 {folder_path} 中未找到以 {orientation}1 开头的图像文件")
    if not image_ext.startswith('.'):
        image_ext = '.' + image_ext

    idx, count = 1, 0
    while True:
        path = os.path.join(folder_path, f"{orientation}{idx}{image_ext}")
        if not os.path.exists(path):
            break
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            break
        count += 1
        idx += 1

    if count == 0:
        raise FileNotFoundError(f"在 {folder_path} 中未找到 {orientation} 开头的 {image_ext} 图像")
    if count % N != 0:
        raise ValueError(f"检测到 {count} 张图像，不能被 N={N} 整除")

    num_freqs = count // N
    if verbose:
        print(f"检测到 {orientation.upper()} 方向共有 {num_freqs} 个频率段，每个频率 {N} 步。")
    return num_freqs


def load_single_frequency_images(folder_path, phase_shift_step=4,
                                  orientation='h', frequency_index=0,
                                  num_frequencies=None, image_ext=None):
    """
    读取单一频率的一组N步相移条纹图像。
    来源: 单频相位处理工具.py → ImageLoader.load_single_frequency_images
    返回: ndarray (H, W, N)
    """
    if orientation not in ('h', 'v'):
        raise ValueError("orientation 只能为 'h' 或 'v'")
    N = phase_shift_step

    if image_ext is None:
        image_ext = detect_image_extension(folder_path, orientation, index=1)
        if image_ext is None:
            raise FileNotFoundError(
                f"在 {folder_path} 中未找到以 {orientation}1 开头的图像文件")
    if not image_ext.startswith('.'):
        image_ext = '.' + image_ext

    if num_frequencies is None:
        num_frequencies = detect_num_frequencies(
            folder_path, phase_shift_step, orientation, image_ext)
    if not (0 <= frequency_index < num_frequencies):
        raise ValueError(f"frequency_index={frequency_index} 超出范围 0~{num_frequencies - 1}")

    start_idx = frequency_index * N + 1
    first_path = os.path.join(folder_path, f"{orientation}{start_idx}{image_ext}")
    first_img = cv2.imread(first_path, cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        raise FileNotFoundError(f"无法读取: {first_path}")

    h, w = first_img.shape
    images = np.zeros((h, w, N), dtype=np.uint8)
    for i, img_idx in enumerate(range(start_idx, start_idx + N)):
        path = os.path.join(folder_path, f"{orientation}{img_idx}{image_ext}")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取: {path}")
        images[:, :, i] = img
    return images


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  步骤二：ROI 选择 & 掩码生成                                          ║
# ║  （完整移植自 fringe_gray_analysis.py → ROISelector 类）               ║
# ║  支持交互式鼠标框选（需要显示器）和手动指定 ROI 两种方式                ║
# ╚════════════════════════════════════════════════════════════════════════╝

class ROISelector:
    """
    交互式 ROI 选择器（鼠标框选 + 滑块选行列），也支持手动指定。
    来源: fringe_gray_analysis.py → ROISelector 类（完整移植）

    用法:
        1) 交互模式: selector.select_interactive()
           弹出窗口让用户鼠标框选 ROI，再用滑块选择行/列
        2) 手动模式: selector.select_manual(mode, line_index, roi)
           直接指定 ROI 和行/列参数
    """

    def __init__(self, gray_img: np.ndarray):
        self.gray_img = gray_img
        self.h, self.w = gray_img.shape
        self.roi_r0 = 0; self.roi_r1 = self.h
        self.roi_c0 = 0; self.roi_c1 = self.w
        self.mode = "row"; self.line_index = 0
        self._roi_confirmed = False

    # ─── 交互模式（需要显示器）───
    def select_interactive(self):
        """交互式选择：先鼠标框选 ROI，再滑块选行/列。"""
        if not _INTERACTIVE:
            print("[WARN] 无可用显示器，请使用 select_manual()。")
            return
        self._phase1_select_roi()
        if not self._roi_confirmed:
            print("[INFO] 未选择 ROI，使用全图。")
        self._phase2_select_line()

    def _phase1_select_roi(self):
        """阶段1：鼠标框选 ROI。来源: fringe_gray_analysis.py → ROISelector._phase1_select_roi"""
        self._roi_confirmed = False
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.canvas.manager.set_window_title("阶段 1/2 — 框选 ROI")
        ax.imshow(self.gray_img, cmap="gray", aspect="equal")
        ax.set_title("鼠标拖拽框选 ROI，然后关闭窗口（不框选=全图）",
                      fontsize=12, fontweight="bold")
        ax.set_xlabel("Column"); ax.set_ylabel("Row")
        self._rect_coords = None

        def on_select(ec, er):
            x0, y0 = int(ec.xdata), int(ec.ydata)
            x1, y1 = int(er.xdata), int(er.ydata)
            self._rect_coords = (max(0, min(y0, y1)), min(self.h, max(y0, y1)),
                                 max(0, min(x0, x1)), min(self.w, max(x0, x1)))

        self._rect_selector = RectangleSelector(
            ax, on_select, useblit=True, button=[1],
            minspanx=5, minspany=5, spancoords="pixels", interactive=True,
            props=dict(facecolor="cyan", edgecolor="red", alpha=0.3, linewidth=2))
        plt.tight_layout(); plt.show()

        if self._rect_coords is not None:
            r0, r1, c0, c1 = self._rect_coords
            if (r1 - r0) >= 2 and (c1 - c0) >= 2:
                self.roi_r0, self.roi_r1 = r0, r1
                self.roi_c0, self.roi_c1 = c0, c1
                self._roi_confirmed = True
                print(f"[INFO] ROI: 行[{r0}:{r1}], 列[{c0}:{c1}]")

    def _phase2_select_line(self):
        """阶段2：滑块选行/列。来源: fringe_gray_analysis.py → ROISelector._phase2_select_line"""
        roi = self.gray_img[self.roi_r0:self.roi_r1, self.roi_c0:self.roi_c1]
        rh, rw = roi.shape
        fig, (axI, axC) = plt.subplots(1, 2, figsize=(16, 7),
                                         gridspec_kw={"width_ratios": [1, 1.2]})
        fig.canvas.manager.set_window_title("阶段 2/2 — 选择行/列")
        fig.subplots_adjust(bottom=0.28, left=0.06, right=0.96, wspace=0.25)
        axI.imshow(roi, cmap="gray", aspect="equal",
                   extent=[self.roi_c0, self.roi_c1, self.roi_r1, self.roi_r0])
        axI.set_title("ROI", fontsize=12, fontweight="bold")
        self.mode = "row"; self.line_index = rh // 2

        hline = axI.axhline(y=self.roi_r0 + self.line_index, color="#FF4444", linewidth=2)
        prof = roi[self.line_index, :]
        cL, = axC.plot(np.arange(len(prof)), prof, color="#2563EB", linewidth=1.2)
        mL = axC.axhline(y=np.mean(prof), color="#DC2626", linewidth=1, linestyle="--", alpha=0.7)
        axC.set_ylim(-5, 265); axC.set_xlim(0, max(rw - 1, 1))
        axC.grid(True, alpha=0.3, linestyle="--")

        axR = fig.add_axes([0.06, 0.06, 0.08, 0.12])
        self._radio = RadioButtons(axR, ("row", "col"), active=0)
        axR.set_title("方向", fontsize=10)
        axS = fig.add_axes([0.22, 0.10, 0.55, 0.04])
        self._slider = Slider(axS, "Index", 0, max(rh - 1, 1),
                               valinit=self.line_index, valstep=1)
        axB = fig.add_axes([0.84, 0.06, 0.10, 0.06])
        self._btn = Button(axB, "确认", color="lightgreen", hovercolor="lime")

        radio = self._radio; sl = self._slider; btn = self._btn

        def upd(_=None):
            idx = int(sl.val); m = radio.value_selected
            self.mode = m; self.line_index = idx
            if m == "row":
                ic = min(idx, rh - 1); p = roi[ic, :]
                hline.set_visible(True); hline.set_ydata([self.roi_r0 + ic])
                for a in list(axI.lines):
                    if a is not hline:
                        a.remove()
                axC.set_xlabel("Column Index"); axC.set_xlim(0, max(rw - 1, 1))
            else:
                ic = min(idx, rw - 1); p = roi[:, ic]
                hline.set_visible(False)
                for a in list(axI.lines):
                    if a is not hline:
                        a.remove()
                axI.axvline(x=self.roi_c0 + ic, color="#FF4444", linewidth=2)
                axC.set_xlabel("Row Index"); axC.set_xlim(0, max(rh - 1, 1))
            cL.set_xdata(np.arange(len(p))); cL.set_ydata(p)
            mL.set_ydata([np.mean(p)]); fig.canvas.draw_idle()

        def mc(label):
            if label == "row":
                sl.valmax = max(rh - 1, 1)
            else:
                sl.valmax = max(rw - 1, 1)
            sl.ax.set_xlim(0, sl.valmax)
            sl.set_val(min(int(sl.val), int(sl.valmax))); upd()

        sl.on_changed(upd); radio.on_clicked(mc)
        btn.on_clicked(lambda _: plt.close(fig)); plt.show()
        print(f"[INFO] 选择: mode={self.mode}, line_index={self.line_index}")

    # ─── 手动模式 ───
    def select_manual(self, mode="row", line_index=0, roi=None):
        """
        手动指定 ROI 和行/列参数。
        来源: fringe_gray_analysis.py → ROISelector.select_manual

        Args:
            mode: 'row' 或 'col'
            line_index: ROI 内部的行/列号
            roi: (r0, r1, c0, c1) 或 None（全图）
        """
        if roi is not None:
            self.roi_r0 = max(0, roi[0]); self.roi_r1 = min(self.h, roi[1])
            self.roi_c0 = max(0, roi[2]); self.roi_c1 = min(self.w, roi[3])
        self.mode = mode; self.line_index = line_index

    # ─── 属性 ───
    @property
    def roi_image(self):
        return self.gray_img[self.roi_r0:self.roi_r1, self.roi_c0:self.roi_c1]

    @property
    def roi_bounds(self):
        return (self.roi_r0, self.roi_r1, self.roi_c0, self.roi_c1)


def roi_to_mask(shape, roi_bounds):
    """
    将 ROI 矩形区域转换为 bool 掩码。
    来源: fringe_gray_analysis.py 中 ROI 矩形区域的概念；
          本函数将其转换为与 绝对相位可视化工具.py 中掩码格式一致的 bool 数组。

    Args:
        shape: (H, W) 相位数据尺寸
        roi_bounds: (r0, r1, c0, c1) ROI 边界

    Returns:
        mask: bool ndarray (H, W)，ROI 内部为 True，外部为 False
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    r0, r1, c0, c1 = roi_bounds
    r0 = max(0, r0); r1 = min(H, r1)
    c0 = max(0, c0); c1 = min(W, c1)
    mask[r0:r1, c0:c1] = True
    return mask


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  步骤三：绝对相位计算                                                  ║
# ║  （借鉴 单频相位处理工具.py → PhaseCalculator 类的三个核心方法          ║
# ║    以及 test_multiple_frequencies2 函数的完整流程）                     ║
# ╚════════════════════════════════════════════════════════════════════════╝

def calculate_wrapped_phase(stripe_images, phase_shifts=None):
    """
    计算单频 N 步相移数据的包裹相位 / 调制系数 B / 平均强度。
    来源: 单频相位处理工具.py → PhaseCalculator.calculate_single_frequency_wrapped_phase

    公式:
        φ(x,y) = -arctan2(∑I_n sin Δφ_n, ∑I_n cos Δφ_n)  → mod 2π 映射到 [0, 2π)
        B(x,y) = (2/N)·√[(∑I_n cos Δφ_n)² + (∑I_n sin Δφ_n)²]
        A(x,y) = (1/N)·∑I_n

    Returns: (wrapped_phase, modulation_B, mean_intensity)
    """
    H, W, N = stripe_images.shape
    imgs = stripe_images.astype(np.float64)

    if phase_shifts is None:
        phase_shifts = 2 * np.pi * np.arange(N) / N
    else:
        phase_shifts = np.asarray(phase_shifts, dtype=np.float64)

    cos_sum = np.zeros((H, W), dtype=np.float64)
    sin_sum = np.zeros((H, W), dtype=np.float64)
    for i in range(N):
        cos_sum += imgs[:, :, i] * np.cos(phase_shifts[i])
        sin_sum += imgs[:, :, i] * np.sin(phase_shifts[i])

    wrapped = -np.arctan2(sin_sum, cos_sum)
    wrapped = np.mod(wrapped, 2 * np.pi)

    mod_B = (2.0 / N) * np.sqrt(cos_sum ** 2 + sin_sum ** 2)
    mean_I = np.mean(imgs, axis=2)
    return wrapped, mod_B, mean_I


def calculate_phase_difference(phase1, phase2):
    """
    计算两个包裹相位的相位差（[0,2π) 范围）。
    来源: 单频相位处理工具.py → PhaseCalculator.calculate_phase_difference
    """
    p1 = phase1 / (2 * np.pi)
    p2 = phase2 / (2 * np.pi)
    result = p1 - p2
    result[result < 0] += 1
    return result * (2 * np.pi)


def unwrap_phase_with_reference(reference_phase, target_phase,
                                 reference_freq, target_freq):
    """
    使用参考相位展开高频相位（绝对相位，弧度制）。
    来源: 单频相位处理工具.py → PhaseCalculator.unwrap_phase_with_reference

    返回: 绝对相位（弧度），可大于 2π
    """
    ref = reference_phase / (2 * np.pi)
    tar = target_phase / (2 * np.pi)
    temp = target_freq / reference_freq * ref
    k = np.round(temp - tar)
    unwrap = tar + k

    gauss_size = (3, 3)
    unwrap_noise = unwrap - cv2.GaussianBlur(unwrap, gauss_size, 0)
    ref_noise = temp - cv2.GaussianBlur(temp, gauss_size, 0)

    noise_ratio = np.abs(unwrap_noise) / (np.abs(ref_noise) + 0.001)
    flag = (np.abs(unwrap_noise) - np.abs(ref_noise) > 0.15) & (noise_ratio > 1.5)

    if np.sum(flag) > 0:
        err = unwrap[flag]
        d = unwrap_noise[flag]
        err[d > 0] -= 1
        err[d < 0] += 1
        unwrap[flag] = err

        unwrap_noise2 = unwrap - cv2.GaussianBlur(unwrap, gauss_size, 0)
        flag2 = np.abs(unwrap_noise2) > 0.2
        if np.sum(flag2) > 0:
            err2 = unwrap[flag2]
            d2 = unwrap_noise2[flag2]
            err2[d2 > 0] -= 1
            err2[d2 < 0] += 1
            unwrap[flag2] = err2

    return unwrap * (2 * np.pi)


def compute_absolute_phase_3freq(folder, phase_shift_step=4, orientation='v',
                                  freq_combination=None, verbose=True):
    """
    三频改进塔形法计算绝对相位 —— 核心流程。
    来源: 单频相位处理工具.py → test_multiple_frequencies2 函数

    流程:
      1) 读取三个频率的 N 步相移图像
      2) 分别计算包裹相位
      3) 两两做相位差（外差）→ 得到等效低频
      4) 逐级参考展开，最终得到最高频的绝对相位

    Args:
        folder: 图像文件夹路径
        phase_shift_step: 相移步数
        orientation: 'h' 或 'v'
        freq_combination: 频率组合字符串，如 "199-201-204"
                          若为 None，则尝试从文件夹路径自动提取
        verbose: 是否打印中间信息

    Returns:
        absolute_phase: ndarray (H, W) 绝对相位（弧度）
        freq_combination: 使用的频率组合字符串
    """
    # 自动提取频率组合
    if freq_combination is None:
        m = re.search(r'(\d+-\d+-\d+)', folder)
        freq_combination = m.group(1) if m else "unknown"
    freqs = [int(f) for f in freq_combination.split('-')]
    if len(freqs) != 3:
        raise ValueError(f"频率组合 '{freq_combination}' 不包含3个频率")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  三频改进塔形法 —— 绝对相位计算")
        print(f"  文件夹:  {folder}")
        print(f"  步数:    {phase_shift_step}")
        print(f"  方向:    {orientation}")
        print(f"  频率组合: {freq_combination}  ({freqs})")
        print(f"{'=' * 60}")

    # ---- 1) 读取三个频率图像（来源: test_multiple_frequencies2 行2118~2120）----
    imgs_highest = load_single_frequency_images(folder, phase_shift_step, orientation, frequency_index=0)
    imgs_middle = load_single_frequency_images(folder, phase_shift_step, orientation, frequency_index=1)
    imgs_lowest = load_single_frequency_images(folder, phase_shift_step, orientation, frequency_index=2)

    # ---- 2) 计算包裹相位（来源: test_multiple_frequencies2 行2123~2125）----
    w_highest, _, _ = calculate_wrapped_phase(imgs_highest)
    w_middle, _, _ = calculate_wrapped_phase(imgs_middle)
    w_lowest, _, _ = calculate_wrapped_phase(imgs_lowest)
    if verbose:
        print("✓ 三个频率的包裹相位计算完成")

    # ---- 3) 外差 —— 两两做相位差（来源: test_multiple_frequencies2 行2133~2139）----
    diff1 = calculate_phase_difference(w_highest, w_middle)   # 等效频率 f1-f2
    diff2 = calculate_phase_difference(w_middle, w_lowest)    # 等效频率 f2-f3
    diff_lowest = calculate_phase_difference(diff1, diff2)    # 最低等效频率
    if verbose:
        print("✓ 外差相位差计算完成 (diff1, diff2, diff_lowest)")

    # ---- 4) 逐级参考展开（来源: test_multiple_frequencies2 行2141~2149）----
    # 根据频率组合计算等效频率比
    f1, f2, f3 = freqs
    eq_f1 = f1 - f2     # diff1 的等效频率
    eq_f2 = f2 - f3     # diff2 的等效频率
    eq_lowest = eq_f1 - eq_f2   # 最低等效频率

    # 展开 diff1（等效频率 eq_f1）
    unwrap_diff1 = unwrap_phase_with_reference(diff_lowest, diff1, eq_lowest, eq_f1)

    # 展开 diff2（等效频率 eq_f2）—— 计算但此处不一定用到
    # unwrap_diff2 = unwrap_phase_with_reference(diff_lowest, diff2, eq_lowest, eq_f2)

    # 使用展开后的 diff1 展开最高频率（f1）
    absolute_phase = unwrap_phase_with_reference(unwrap_diff1, w_highest, eq_f1, f1)

    if verbose:
        print(f"✓ 绝对相位计算完成")
        print(f"  - 绝对相位范围: [{np.nanmin(absolute_phase):.2f}, {np.nanmax(absolute_phase):.2f}] rad")

    return absolute_phase, freq_combination


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  步骤四：可视化（借鉴 绝对相位可视化工具.py → PhaseVisualizer 类）      ║
# ║  以及 fringe_gray_analysis.py 的行/列选择 + 曲线绘制风格              ║
# ╚════════════════════════════════════════════════════════════════════════╝

def determine_line_index(shape, mode='row', position='center',
                          line_index=None):
    """
    确定要分析的行/列索引。
    来源: fringe_gray_analysis.py 的 LINE_INDEX / EXTRACT_MODE 参数 +
          绝对相位可视化工具.py → visualize_phase_line_profile 的 position 逻辑

    Args:
        shape: (H, W)
        mode: 'row' 或 'col'
        position: 'top'/'bottom'/'center'/'quarter'/'three_quarter'
        line_index: 手动指定索引，优先级最高

    Returns:
        int: 实际索引
    """
    H, W = shape
    dim = H if mode == 'row' else W
    if line_index is not None:
        return max(0, min(int(line_index), dim - 1))

    mapping = {
        'top': dim // 10,
        'bottom': dim * 9 // 10,
        'center': dim // 2,
        'quarter': dim // 4,
        'three_quarter': dim * 3 // 4,
    }
    return mapping.get(position, dim // 2)


def normalize_phase_to_2pi(phase, periods):
    """
    将绝对相位归一化到 [0, 2π] 范围。
    来源: 绝对相位可视化工具.py → PhaseNormalizer.normalize_phase_to_2pi
    """
    max_p = np.nanmax(phase)
    min_p = np.nanmin(phase)
    if max_p <= 2 * np.pi and min_p >= 0:
        return phase.copy()
    return phase / periods


def visualize_line_profile(phase, mask, mode, idx, save_folder,
                            freq_combination, periods, position='center'):
    """
    可视化指定行/列的绝对相位剖面（含掩码支持）。
    来源:
      - 绝对相位可视化工具.py → PhaseVisualizer.visualize_phase_line_profile（曲线绘制逻辑）
      - fringe_gray_analysis.py → ResultPlotter.plot_curve（曲线样式 / 统计信息框）

    Args:
        phase: 绝对相位 (H, W)，弧度
        mask: bool 掩码 (H, W) 或 None
        mode: 'row' 或 'col'
        idx: 行/列索引
        save_folder: 保存目录
        freq_combination: 频率组合字符串（标题用）
        periods: 周期数（归一化用）
        position: 位置描述
    """
    os.makedirs(save_folder, exist_ok=True)

    # ---- 归一化到 [0, 2π]（来源: 绝对相位可视化工具.py）----
    phase_norm = normalize_phase_to_2pi(phase, periods)

    # ---- 应用掩码（来源: 绝对相位可视化工具.py → visualize_phase_line_profile 行408~416）----
    display_phase = phase_norm.copy()
    if mask is not None and mask.shape == phase_norm.shape:
        display_phase[~mask] = np.nan

    # ---- 提取行/列数据 ----
    H, W = phase_norm.shape
    if mode == 'row':
        line_data = display_phase[idx, :]
        pixel_pos = np.arange(W)
        label_x = '列索引 (pixels)'
        label_y = '相位值 (rad, 0-2π)'
        direction_text = f"行 {idx}"
    else:
        line_data = display_phase[:, idx]
        pixel_pos = np.arange(H)
        label_x = '行索引 (pixels)'
        label_y = '相位值 (rad, 0-2π)'
        direction_text = f"列 {idx}"

    valid = line_data[~np.isnan(line_data)]
    mean_val = np.mean(valid) if len(valid) > 0 else 0.0
    std_val = np.std(valid) if len(valid) > 0 else 0.0

    mask_suffix = "（使用掩码后）" if mask is not None else ""

    # ================================================================
    # 图 1：相位剖面曲线（来源: 绝对相位可视化工具.py 行472~510）
    # ================================================================
    fig_c, ax_c = plt.subplots(1, 1, figsize=(16, 6))
    ax_c.plot(pixel_pos, line_data, linewidth=1.5, color='blue', alpha=0.8)
    ax_c.set_title(
        f'绝对相位变化曲线 — {direction_text}（{position}位置）'
        f'{mask_suffix}\n频率组合: {freq_combination}',
        fontsize=14)
    ax_c.set_xlabel(label_x, fontsize=12)
    ax_c.set_ylabel(label_y, fontsize=12)
    ax_c.set_ylim(0, 2 * np.pi)
    ax_c.grid(True, alpha=0.3, linestyle='--')
    if len(valid) > 0:
        ax_c.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5,
                      alpha=0.7, label=f'Mean={mean_val:.4f}')
    # 统计信息框（借鉴 fringe_gray_analysis.py → ResultPlotter.plot_curve 行372）
    ax_c.text(0.02, 0.95,
              f"Mean={mean_val:.4f}  Std={std_val:.4f}  Points={len(valid)}",
              transform=ax_c.transAxes, fontsize=10, va='top',
              family='monospace',
              bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='gray', alpha=0.85))
    ax_c.legend(fontsize=10)
    plt.tight_layout()

    curve_path = os.path.join(save_folder, f"phase_line_{mode}{idx}_curve.png")
    fig_c.savefig(curve_path, dpi=300, bbox_inches='tight')
    print(f"[SAVE] 相位剖面曲线: {curve_path}")
    if _INTERACTIVE:
        plt.show()
    plt.close(fig_c)

    # ================================================================
    # 图 2：2D 相位伪彩色图 + 标记线
    # （来源: 绝对相位可视化工具.py 行514~543 + fringe_gray_analysis.py 标注风格）
    # ================================================================
    fig_i, ax_i = plt.subplots(1, 1, figsize=(12, 10))
    im = ax_i.imshow(display_phase, cmap='viridis', aspect='auto',
                      vmin=0, vmax=2 * np.pi)
    ax_i.set_title(
        f'绝对相位图像（归一化到0-2π）{mask_suffix}\n频率组合: {freq_combination}',
        fontsize=14)
    ax_i.set_xlabel('列索引 (pixels)', fontsize=12)
    ax_i.set_ylabel('行索引 (pixels)', fontsize=12)
    if mode == 'row':
        ax_i.axhline(y=idx, color='red', linestyle='--', linewidth=2, alpha=0.9,
                      label=f'选择的行: {idx}')
    else:
        ax_i.axvline(x=idx, color='red', linestyle='--', linewidth=2, alpha=0.9,
                      label=f'选择的列: {idx}')
    ax_i.legend(fontsize=11)
    plt.colorbar(im, ax=ax_i, label='相位值 (rad, 0-2π)', fraction=0.046, pad=0.04)
    plt.tight_layout()

    image_path = os.path.join(save_folder, f"phase_line_{mode}{idx}_image.png")
    fig_i.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"[SAVE] 相位伪彩色图: {image_path}")
    if _INTERACTIVE:
        plt.show()
    plt.close(fig_i)


def visualize_3d_surface(phase, mask, save_folder, freq_combination,
                          periods, downsample_factor=None):
    """
    生成绝对相位 3D 表面图。
    来源: 绝对相位可视化工具.py → PhaseVisualizer.visualize_phase_3d_surface
    """
    os.makedirs(save_folder, exist_ok=True)

    phase_norm = normalize_phase_to_2pi(phase, periods)
    display = phase_norm.copy()
    if mask is not None and mask.shape == phase_norm.shape:
        display[~mask] = np.nan

    H, W = display.shape
    if downsample_factor is None:
        max_dim = max(H, W)
        downsample_factor = max(1, max_dim // 500) if max_dim > 1000 else 1

    if downsample_factor > 1:
        phase_3d = display[::downsample_factor, ::downsample_factor]
        x_3d = np.arange(0, W, downsample_factor)[:phase_3d.shape[1]]
        y_3d = np.arange(0, H, downsample_factor)[:phase_3d.shape[0]]
    else:
        phase_3d = display
        x_3d = np.arange(W)
        y_3d = np.arange(H)

    X, Y = np.meshgrid(x_3d, y_3d)
    h3, w3 = phase_3d.shape

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    rstride = max(1, h3 // 50)
    cstride = max(1, w3 // 50)
    ax.plot_surface(X, Y, phase_3d, cmap='viridis', linewidth=0,
                    edgecolor='none', antialiased=True, alpha=0.9,
                    shade=True, rstride=rstride, cstride=cstride)
    ax.set_title(f'绝对相位 3D 表面图（{freq_combination}）', fontsize=14, pad=20)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('相位值 (rad, 0-2π)')
    ax.set_zlim(0, 2 * np.pi)
    plt.tight_layout()

    path_3d = os.path.join(save_folder, f"phase_3d_surface_{freq_combination}.png")
    fig.savefig(path_3d, dpi=300, bbox_inches='tight')
    print(f"[SAVE] 3D 表面图: {path_3d}")
    if _INTERACTIVE:
        plt.show()
    plt.close(fig)


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  主流程：整合所有步骤                                                  ║
# ╚════════════════════════════════════════════════════════════════════════╝

def run_integrated_pipeline(
    # ---- 条纹图像参数（来源: 单频相位处理工具.py → test_multiple_frequencies2）----
    projection_folder: str,
    phase_shift_step: int = 4,
    orientation: str = 'v',
    freq_combination: str = None,

    # ---- ROI 掩码参数（来源: fringe_gray_analysis.py → ROISelector）----
    # 第一张条纹图像路径，用于交互式 ROI 选择时显示背景图
    # 设为 None 时使用第一张相移图像作为背景
    roi_background_image: str = None,
    # 手动 ROI: (r0, r1, c0, c1)，设为 None 表示全图（即不使用掩码）
    roi: tuple = None,

    # ---- 行/列选择参数（来源: fringe_gray_analysis.py → EXTRACT_MODE / LINE_INDEX
    #                    + 绝对相位可视化工具.py → position 参数）----
    extract_mode: str = 'row',
    line_index: int = None,
    position: str = 'center',

    # ---- 输出参数 ----
    output_dir: str = './integrated_results',
    generate_3d: bool = True,
):
    """
    整合版主流程。

    Args:
        projection_folder: 包含多频率 N 步相移图像的文件夹
        phase_shift_step: 相移步数（默认 4）
        orientation: 条纹方向 'h' 或 'v'
        freq_combination: 频率组合字符串，如 "199-201-204"
        roi_background_image: 用于 ROI 选择时显示的背景图路径（None=用首张条纹图）
        roi: 手动指定 ROI (r0, r1, c0, c1)，None=全图（不使用掩码）
             若在交互模式下运行，此参数被忽略，由鼠标框选确定
        extract_mode: 'row' 或 'col'
        line_index: 指定行/列号（None 则根据 position 自动确定）
        position: 自动位置 'top'/'bottom'/'center'/'quarter'/'three_quarter'
        output_dir: 输出目录
        generate_3d: 是否生成 3D 表面图
    """
    print("=" * 70)
    print("  整合版：绝对相位生成与可视化流程")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════
    # STEP 1：ROI 选择 → 掩码
    #   来源: fringe_gray_analysis.py → ROISelector 类
    #   交互模式：鼠标框选 ROI + 滑块选行/列
    #   手动模式：通过 roi=(r0, r1, c0, c1) 参数指定
    # ═══════════════════════════════════════════════════════════
    mask = None
    roi_bounds = None
    selector = None

    # 确定背景图（用于交互式 ROI 选择时显示）
    if roi_background_image and os.path.isfile(roi_background_image):
        bg_img = cv2.imread(roi_background_image, cv2.IMREAD_GRAYSCALE)
    else:
        # 用第一张条纹图像作为背景
        img_ext = detect_image_extension(projection_folder, orientation, index=1)
        if img_ext is None:
            raise FileNotFoundError(f"在 {projection_folder} 中未找到条纹图像")
        bg_path = os.path.join(projection_folder, f"{orientation}1{img_ext}")
        bg_img = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
    if bg_img is None:
        raise FileNotFoundError("无法读取背景图像用于 ROI 选择")

    if _INTERACTIVE and roi is None:
        # ---- 交互模式：弹窗让用户鼠标框选 ROI + 选行/列 ----
        print("\n[STEP 1] 交互式 ROI 选择（来源: fringe_gray_analysis.py → ROISelector）...")
        selector = ROISelector(bg_img)
        selector.select_interactive()
        roi_bounds = selector.roi_bounds
        # 交互模式下，行/列选择也由滑块确定，覆盖外部参数
        extract_mode = selector.mode
        line_index = selector.line_index
        print(f"  交互结果: ROI={roi_bounds}, mode={extract_mode}, line_index={line_index}")
    elif roi is not None:
        # ---- 手动模式：用户通过参数指定 ROI ----
        print(f"\n[STEP 1] 手动指定 ROI: {roi}（来源: fringe_gray_analysis.py → select_manual 风格）")
        selector = ROISelector(bg_img)
        selector.select_manual(mode=extract_mode, line_index=line_index or 0, roi=roi)
        roi_bounds = selector.roi_bounds
        print(f"  ROI 边界: 行[{roi_bounds[0]}:{roi_bounds[1]}], 列[{roi_bounds[2]}:{roi_bounds[3]}]")
    else:
        print("\n[STEP 1] 未指定 ROI，使用全图（不使用掩码）。")

    # 如果选择了非全图的 ROI，生成 bool 掩码
    if roi_bounds is not None:
        r0, r1, c0, c1 = roi_bounds
        is_full_image = (r0 == 0 and r1 == bg_img.shape[0] and
                         c0 == 0 and c1 == bg_img.shape[1])
        if not is_full_image:
            # 此处先记录 ROI，在绝对相位计算完成后再生成与相位数据同尺寸的掩码
            print(f"  ROI 区域: 行[{r0}:{r1}], 列[{c0}:{c1}]")
            ratio = (r1 - r0) * (c1 - c0) / (bg_img.shape[0] * bg_img.shape[1]) * 100
            print(f"  ROI 占比: {ratio:.2f}%")
        else:
            roi_bounds = None  # 全图时不需要掩码
            print("  ROI 为全图，等同于不使用掩码。")

    # ═══════════════════════════════════════════════════════════
    # STEP 2：读取图像 & 计算绝对相位
    #   读图来源: 单频相位处理工具.py → ImageLoader
    #   计算来源: 单频相位处理工具.py → test_multiple_frequencies2 核心逻辑
    # ═══════════════════════════════════════════════════════════
    print("\n[STEP 2] 读取图像并计算绝对相位...")
    absolute_phase, freq_comb = compute_absolute_phase_3freq(
        projection_folder, phase_shift_step, orientation,
        freq_combination, verbose=True)

    # 在绝对相位计算完成后，根据 ROI 生成与相位数据同尺寸的 bool 掩码
    if roi_bounds is not None:
        mask = roi_to_mask(absolute_phase.shape, roi_bounds)
        # 保存掩码
        mask_dir = os.path.join(output_dir, "mask")
        os.makedirs(mask_dir, exist_ok=True)
        mask_uint8 = mask.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(mask_dir, "mask.bmp"), mask_uint8)
        np.save(os.path.join(mask_dir, "mask.npy"), mask)
        print(f"  ROI 掩码已保存到: {mask_dir}")

    # 保存绝对相位
    phase_dir = os.path.join(output_dir, "phase_data")
    os.makedirs(phase_dir, exist_ok=True)
    npy_path = os.path.join(phase_dir, f"absolute_phase_{freq_comb}.npy")
    np.save(npy_path, absolute_phase)
    print(f"  绝对相位已保存: {npy_path}")

    # ═══════════════════════════════════════════════════════════
    # STEP 3：确定分析行/列
    #   来源: fringe_gray_analysis.py 的 EXTRACT_MODE / LINE_INDEX
    #         + 绝对相位可视化工具.py 的 position 参数
    #   注意: 若 STEP 1 使用了交互式 ROI 选择，extract_mode 和 line_index
    #         已由滑块确定，此处直接使用；否则按 position 自动确定。
    # ═══════════════════════════════════════════════════════════
    print(f"\n[STEP 3] 确定分析方向和索引...")
    # 交互模式下 line_index 已确定为 ROI 内部索引，需加上 ROI 偏移量转为绝对索引
    if selector is not None and selector._roi_confirmed:
        # line_index 是 ROI 内部索引，转换为全图绝对索引
        if extract_mode == 'row':
            idx = roi_bounds[0] + line_index  # r0 + 内部行号
        else:
            idx = roi_bounds[2] + line_index  # c0 + 内部列号
        idx = max(0, min(idx, absolute_phase.shape[0 if extract_mode == 'row' else 1] - 1))
    else:
        idx = determine_line_index(absolute_phase.shape, extract_mode, position, line_index)

    dim_name = "行" if extract_mode == 'row' else "列"
    print(f"  分析方向: {dim_name} ({extract_mode})")
    print(f"  索引: {idx}")
    if selector is not None and selector._roi_confirmed:
        print(f"  (来自交互式 ROI 选择)")
    else:
        print(f"  位置策略: {position}")

    # ═══════════════════════════════════════════════════════════
    # STEP 4：可视化
    #   来源: 绝对相位可视化工具.py → PhaseVisualizer
    # ═══════════════════════════════════════════════════════════
    # 从频率组合中提取最高频率（用于归一化）
    freqs = [int(f) for f in freq_comb.split('-')] if '-' in freq_comb else [1]
    highest_freq = max(freqs) if freqs else 1
    periods = float(highest_freq)

    vis_dir = os.path.join(output_dir, "visualization")

    print(f"\n[STEP 4] 可视化（归一化周期数: {periods}）...")
    visualize_line_profile(
        absolute_phase, mask, extract_mode, idx,
        vis_dir, freq_comb, periods, position)

    if generate_3d:
        print(f"\n[STEP 4-3D] 生成 3D 表面图...")
        visualize_3d_surface(
            absolute_phase, mask, vis_dir, freq_comb, periods)

    # ═══════════════════════════════════════════════════════════
    # 完成
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  ✓ 整合流程全部完成！")
    print(f"  输出目录: {output_dir}")
    if mask is not None:
        print(f"  - ROI 掩码:   {os.path.join(output_dir, 'mask')}")
    print(f"  - 相位数据:   {os.path.join(output_dir, 'phase_data')}")
    print(f"  - 可视化:     {os.path.join(output_dir, 'visualization')}")
    print("=" * 70)

    return {
        'absolute_phase': absolute_phase,
        'mask': mask,
        'roi_bounds': roi_bounds,
        'freq_combination': freq_comb,
        'line_index': idx,
        'extract_mode': extract_mode,
        'output_dir': output_dir,
    }


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                       ★★★ 用户参数区 ★★★                            ║
# ║  （参数化风格来源: fringe_gray_analysis.py 主入口的用户可修改区域）     ║
# ╚════════════════════════════════════════════════════════════════════════╝

# ---- 条纹图像参数 ----
PROJECTION_FOLDER   = r"E:\images\PSM_FSM\O2"   # 相移图像文件夹
PHASE_SHIFT_STEP    = 4                                 # 相移步数
ORIENTATION         = "v"                               # 条纹方向
FREQ_COMBINATION    = "81-72-64"                             # None=自动从路径提取

# ---- ROI 掩码参数（来源: fringe_gray_analysis.py → ROISelector 的参数风格）----
# 背景图路径：交互式 ROI 选择时显示的图像（None=自动使用首张条纹图）
ROI_BACKGROUND_IMAGE = None
# ROI 区域：(r0, r1, c0, c1) 手动指定，或 None（全图 / 交互选择）
# 示例:  ROI = (100, 900, 200, 1600)   → 手动指定矩形区域
#        ROI = None                     → 有显示器时交互选择，否则全图
ROI                 = None

# ---- 行列分析参数（来源: fringe_gray_analysis.py → EXTRACT_MODE / LINE_INDEX）----
EXTRACT_MODE        = "row"        # "row" 或 "col"
LINE_INDEX          = None         # None = 自动根据 POSITION 选择
POSITION            = "center"     # 'top'/'bottom'/'center'/'quarter'/'three_quarter'

# ---- 输出参数 ----
OUTPUT_DIR          = "./integrated_results"
GENERATE_3D         = True


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                           程序入口                                    ║
# ╚════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    run_integrated_pipeline(
        projection_folder=PROJECTION_FOLDER,
        phase_shift_step=PHASE_SHIFT_STEP,
        orientation=ORIENTATION,
        freq_combination=FREQ_COMBINATION,
        roi_background_image=ROI_BACKGROUND_IMAGE,
        roi=ROI,
        extract_mode=EXTRACT_MODE,
        line_index=LINE_INDEX,
        position=POSITION,
        output_dir=OUTPUT_DIR,
        generate_3d=GENERATE_3D,
    )
