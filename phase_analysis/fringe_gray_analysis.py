#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
条纹投影图像灰度变化分析工具（模块化交互版 + 双图对比）
Fringe Projection Image Gray Value Analysis Tool
(Modular & Interactive, with Dual-Image Comparison)

功能：
  【单图模式】
  1. 读取条纹投影图像（支持彩色自动转灰度）
  2. 交互式 ROI 区域选择（鼠标框选感兴趣区域）
  3. 交互式选择提取方向（行/列）及具体行列号
  4. 在选定 ROI 内提取灰度剖面
  5. 原始图像标注图与灰度曲线图分别独立保存
  6. 保存结果为 PNG、NPY、MAT 格式

  【双图对比模式】
  7. 指定基准图 + 待比较图，使用相同 ROI 和行/列
  8. 三线对比曲线（基准=红, 待比较=蓝, 误差=绿[右侧独立 Y 轴]）
  9. 双原图并排标注 + 对比曲线图 + 误差曲线图，各自独立保存
  10. 多格式数据导出（PNG / NPY / MAT）

模块结构：
  ImageLoader          —— 图像读取与预处理
  SampleGenerator      —— 模拟条纹图像生成（含双图模式）
  ROISelector          —— 交互式 ROI 选择
  ProfileExtractor     —— 灰度剖面提取
  ResultPlotter        —— 单图结果可视化
  ComparisonPlotter    —— 双图对比可视化
  DataExporter         —— 多格式数据导出（单图 + 双图）
  FringeAnalyzer       —— 单图分析主控器
  DualImageComparator  —— 双图对比主控器

作者：Claude (Anthropic)
日期：2026-03-10
"""

import os
import sys
import numpy as np
import cv2

# ────────────────────────────────────────────────────────
# matplotlib 后端自动选择
# ────────────────────────────────────────────────────────
import matplotlib
_INTERACTIVE = False
try:
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    fig_test = plt.figure()
    plt.close(fig_test)
    _INTERACTIVE = True
except Exception:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _INTERACTIVE = False

if _INTERACTIVE:
    from matplotlib.widgets import RectangleSelector, RadioButtons, Slider, Button

from scipy.io import savemat

# ────────────────────────────────────────────────────────
# 字体配置
# ────────────────────────────────────────────────────────
_CJK_FONTS = [
    "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
    "Noto Sans CJK SC", "AR PL UMing CN",
]
_USE_CJK = False
for _f in _CJK_FONTS:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        _USE_CJK = True
        break
    except Exception:
        continue


# ╔════════════════════════════════════════════════════════════════════╗
# ║                      ImageLoader 图像读取器                       ║
# ╚════════════════════════════════════════════════════════════════════╝

class ImageLoader:
    """图像读取与预处理：读取图像文件并自动转为灰度图。"""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.raw_img = None
        self.gray_img = None
        self.is_color = False
        self._load()

    def _load(self):
        self.raw_img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if self.raw_img is None:
            raise FileNotFoundError(f"无法读取图像: {self.image_path}")
        if len(self.raw_img.shape) == 3 and self.raw_img.shape[2] >= 3:
            self.is_color = True
            self.gray_img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2GRAY)
        else:
            self.is_color = False
            self.gray_img = self.raw_img.copy()

    @property
    def height(self) -> int:
        return self.gray_img.shape[0]

    @property
    def width(self) -> int:
        return self.gray_img.shape[1]

    @property
    def shape(self) -> tuple:
        return self.gray_img.shape

    def info(self) -> str:
        color_str = "彩色→灰度" if self.is_color else "灰度"
        return (f"{self.image_path} | {color_str} | "
                f"{self.width}×{self.height}")


# ╔════════════════════════════════════════════════════════════════════╗
# ║                   SampleGenerator 样例图像生成器                   ║
# ╚════════════════════════════════════════════════════════════════════╝

class SampleGenerator:
    """模拟条纹投影图像生成器（含普通版和高反光版）。"""

    @staticmethod
    def generate(save_path: str, height: int = 600, width: int = 800,
                 noise_std: float = 5.0) -> np.ndarray:
        """生成标准模拟条纹图像。"""
        x = np.arange(width); y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        period = 40.0; freq = 2.0 * np.pi / period
        phase_mod = 0.3 * np.sin(2.0 * np.pi * yy / height)
        fringe = 128.0 + 100.0 * np.cos(freq * xx + phase_mod)
        fringe += np.random.normal(0, noise_std, (height, width))
        fringe = np.clip(fringe, 0, 255).astype(np.uint8)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, fringe)
        print(f"[INFO] 已生成模拟条纹图像: {save_path}")
        return fringe

    @staticmethod
    def generate_highlight(save_path: str, height: int = 600, width: int = 800,
                           noise_std: float = 5.0) -> np.ndarray:
        """生成带高反光干扰的模拟条纹图像（在标准条纹上叠加高斯 blob）。"""
        x = np.arange(width); y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        period = 40.0; freq = 2.0 * np.pi / period
        phase_mod = 0.3 * np.sin(2.0 * np.pi * yy / height)
        fringe = 128.0 + 100.0 * np.cos(freq * xx + phase_mod)
        fringe += np.random.normal(0, noise_std, (height, width))
        rng = np.random.RandomState(42)
        for _ in range(5):
            cy = rng.randint(height // 4, 3 * height // 4)
            cx = rng.randint(width // 4, 3 * width // 4)
            r = rng.uniform(15, 50); intensity = rng.uniform(80, 150)
            fringe += intensity * np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*r**2))
        fringe = np.clip(fringe, 0, 255).astype(np.uint8)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, fringe)
        print(f"[INFO] 已生成高反光模拟条纹图像: {save_path}")
        return fringe


# ╔════════════════════════════════════════════════════════════════════╗
# ║                      ROISelector ROI 交互选择器                   ║
# ╚════════════════════════════════════════════════════════════════════╝

class ROISelector:
    """交互式 ROI 选择器（鼠标框选 + 滑块选行列），也支持手动指定。"""

    def __init__(self, gray_img: np.ndarray):
        self.gray_img = gray_img
        self.h, self.w = gray_img.shape
        self.roi_r0 = 0; self.roi_r1 = self.h
        self.roi_c0 = 0; self.roi_c1 = self.w
        self.mode = "row"; self.line_index = 0
        self._roi_confirmed = False

    def select_interactive(self):
        if not _INTERACTIVE:
            print("[WARN] 无可用显示器，请使用 select_manual()。"); return
        self._phase1_select_roi()
        if not self._roi_confirmed:
            print("[INFO] 未选择 ROI，使用全图。")
        self._phase2_select_line()

    def _phase1_select_roi(self):
        self._roi_confirmed = False
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.canvas.manager.set_window_title("阶段 1/2 — 框选 ROI")
        ax.imshow(self.gray_img, cmap="gray", aspect="equal")
        ax.set_title("鼠标拖拽框选 ROI，然后关闭窗口（不框选=全图）",
                      fontsize=12, fontweight="bold")
        ax.set_xlabel("Column"); ax.set_ylabel("Row")
        self._rect_coords = None
        def on_select(ec, er):
            x0,y0=int(ec.xdata),int(ec.ydata); x1,y1=int(er.xdata),int(er.ydata)
            self._rect_coords = (max(0,min(y0,y1)),min(self.h,max(y0,y1)),
                                 max(0,min(x0,x1)),min(self.w,max(x0,x1)))
        # 必须赋值给变量，否则 Python 会立刻回收 selector 导致框选失效
        self._rect_selector = RectangleSelector(ax, on_select, useblit=True, button=[1],
            minspanx=5, minspany=5, spancoords="pixels", interactive=True,
            props=dict(facecolor="cyan",edgecolor="red",alpha=0.3,linewidth=2))
        plt.tight_layout(); plt.show()
        if self._rect_coords is not None:
            r0,r1,c0,c1 = self._rect_coords
            if (r1-r0)>=2 and (c1-c0)>=2:
                self.roi_r0,self.roi_r1,self.roi_c0,self.roi_c1 = r0,r1,c0,c1
                self._roi_confirmed = True
                print(f"[INFO] ROI: 行[{r0}:{r1}], 列[{c0}:{c1}]")

    def _phase2_select_line(self):
        roi = self.gray_img[self.roi_r0:self.roi_r1, self.roi_c0:self.roi_c1]
        rh, rw = roi.shape
        fig,(axI,axC)=plt.subplots(1,2,figsize=(16,7),gridspec_kw={"width_ratios":[1,1.2]})
        fig.canvas.manager.set_window_title("阶段 2/2 — 选择行/列")
        fig.subplots_adjust(bottom=0.28,left=0.06,right=0.96,wspace=0.25)
        axI.imshow(roi,cmap="gray",aspect="equal",
                   extent=[self.roi_c0,self.roi_c1,self.roi_r1,self.roi_r0])
        axI.set_title("ROI",fontsize=12,fontweight="bold")
        self.mode="row"; self.line_index=rh//2
        hline=axI.axhline(y=self.roi_r0+self.line_index,color="#FF4444",linewidth=2)
        prof=roi[self.line_index,:]
        cL,=axC.plot(np.arange(len(prof)),prof,color="#2563EB",linewidth=1.2)
        mL=axC.axhline(y=np.mean(prof),color="#DC2626",linewidth=1,linestyle="--",alpha=0.7)
        axC.set_ylim(-5,265); axC.set_xlim(0,max(rw-1,1))
        axC.grid(True,alpha=0.3,linestyle="--")
        axR=fig.add_axes([0.06,0.06,0.08,0.12])
        self._radio=RadioButtons(axR,("row","col"),active=0); axR.set_title("方向",fontsize=10)
        axS=fig.add_axes([0.22,0.10,0.55,0.04])
        self._slider=Slider(axS,"Index",0,max(rh-1,1),valinit=self.line_index,valstep=1)
        axB=fig.add_axes([0.84,0.06,0.10,0.06])
        self._btn=Button(axB,"确认",color="lightgreen",hovercolor="lime")
        # 使用短别名方便内部回调引用
        radio=self._radio; sl=self._slider; btn=self._btn
        def upd(_=None):
            idx=int(sl.val); m=radio.value_selected; self.mode=m; self.line_index=idx
            if m=="row":
                ic=min(idx,rh-1); p=roi[ic,:]
                hline.set_visible(True); hline.set_ydata([self.roi_r0+ic])
                for a in list(axI.lines):
                    if a is not hline: a.remove()
                axC.set_xlabel("Column Index"); axC.set_xlim(0,max(rw-1,1))
            else:
                ic=min(idx,rw-1); p=roi[:,ic]
                hline.set_visible(False)
                for a in list(axI.lines):
                    if a is not hline: a.remove()
                axI.axvline(x=self.roi_c0+ic,color="#FF4444",linewidth=2)
                axC.set_xlabel("Row Index"); axC.set_xlim(0,max(rh-1,1))
            cL.set_xdata(np.arange(len(p))); cL.set_ydata(p)
            mL.set_ydata([np.mean(p)]); fig.canvas.draw_idle()
        def mc(label):
            if label=="row": sl.valmax=max(rh-1,1)
            else: sl.valmax=max(rw-1,1)
            sl.ax.set_xlim(0,sl.valmax)
            sl.set_val(min(int(sl.val),int(sl.valmax))); upd()
        sl.on_changed(upd); radio.on_clicked(mc)
        btn.on_clicked(lambda _: plt.close(fig)); plt.show()
        print(f"[INFO] 选择: mode={self.mode}, line_index={self.line_index}")

    def select_manual(self, mode="row", line_index=0, roi=None):
        if roi is not None:
            self.roi_r0=max(0,roi[0]); self.roi_r1=min(self.h,roi[1])
            self.roi_c0=max(0,roi[2]); self.roi_c1=min(self.w,roi[3])
        self.mode=mode; self.line_index=line_index

    @property
    def roi_image(self): return self.gray_img[self.roi_r0:self.roi_r1,self.roi_c0:self.roi_c1]
    @property
    def roi_bounds(self): return (self.roi_r0,self.roi_r1,self.roi_c0,self.roi_c1)


# ╔════════════════════════════════════════════════════════════════════╗
# ║                 ProfileExtractor 灰度剖面提取器                    ║
# ╚════════════════════════════════════════════════════════════════════╝

class ProfileExtractor:
    """从 ROI 中提取一行或一列的灰度剖面及统计信息。"""

    def __init__(self, roi_img, mode, line_index, roi_offset=(0,0)):
        self.roi_img=roi_img; self.mode=mode
        self.roi_h,self.roi_w=roi_img.shape
        self.roi_row_offset,self.roi_col_offset=roi_offset
        if mode=="row": self.line_index=int(np.clip(line_index,0,self.roi_h-1))
        else:           self.line_index=int(np.clip(line_index,0,self.roi_w-1))
        self.profile=None; self.pixel_positions=None; self.abs_index=0; self.stats={}
        self._extract()

    def _extract(self):
        if self.mode=="row":
            self.profile=self.roi_img[self.line_index,:].astype(np.float64)
            self.abs_index=self.roi_row_offset+self.line_index
            self.pixel_positions=np.arange(self.roi_col_offset,self.roi_col_offset+self.roi_w)
        elif self.mode=="col":
            self.profile=self.roi_img[:,self.line_index].astype(np.float64)
            self.abs_index=self.roi_col_offset+self.line_index
            self.pixel_positions=np.arange(self.roi_row_offset,self.roi_row_offset+self.roi_h)
        self.stats={"min":float(self.profile.min()),"max":float(self.profile.max()),
                    "mean":float(np.mean(self.profile)),"std":float(np.std(self.profile))}

    def summary(self):
        d="行" if self.mode=="row" else "列"
        return (f"方向: {d}({self.mode}) | 绝对索引: {self.abs_index} | "
                f"采样: {len(self.profile)}点\n"
                f"  Min={self.stats['min']:.1f}, Max={self.stats['max']:.1f}, "
                f"Mean={self.stats['mean']:.1f}, Std={self.stats['std']:.1f}")


# ╔════════════════════════════════════════════════════════════════════╗
# ║                   ResultPlotter 单图结果可视化器                    ║
# ╚════════════════════════════════════════════════════════════════════╝

class ResultPlotter:
    """单图模式：生成原始图像标注图 + 灰度曲线图。"""

    def __init__(self, gray_img, extractor, roi_bounds):
        self.gray_img=gray_img; self.ext=extractor; self.roi_bounds=roi_bounds

    @staticmethod
    def _annotate(ax, gray, bounds, mode, abs_idx, title="Original Image"):
        """在 ax 上绘制灰度图 + ROI 框 + 行/列标记。"""
        r0,r1,c0,c1=bounds; h,w=gray.shape
        ax.imshow(gray,cmap="gray",aspect="equal")
        if not (r0==0 and r1==h and c0==0 and c1==w):
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((c0,r0),c1-c0,r1-r0,lw=2,ec="#00FF00",fc="none",ls="--"))
            ax.text(c0,r0-8,"ROI",color="#00FF00",fontsize=9,fontweight="bold")
        if mode=="row":
            ax.axhline(y=abs_idx,color="#FF4444",lw=2,alpha=0.9)
            ty=abs_idx+h*0.02 if abs_idx<h*0.85 else abs_idx-h*0.04
            ax.text(w*0.01,ty,f"Row {abs_idx}",color="#FF4444",fontsize=9,fontweight="bold")
        else:
            ax.axvline(x=abs_idx,color="#FF4444",lw=2,alpha=0.9)
            tx=abs_idx+w*0.01 if abs_idx<w*0.85 else abs_idx-w*0.08
            ax.text(tx,h*0.03,f"Col {abs_idx}",color="#FF4444",fontsize=9,fontweight="bold")
        ax.set_title(title,fontsize=13,fontweight="bold")
        ax.set_xlabel("Column",fontsize=10); ax.set_ylabel("Row",fontsize=10)

    def plot_image(self, save_path=None, dpi=200):
        fig,ax=plt.subplots(1,1,figsize=(10,8))
        self._annotate(ax,self.gray_img,self.roi_bounds,self.ext.mode,self.ext.abs_index)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".",exist_ok=True)
            fig.savefig(save_path,dpi=dpi,bbox_inches="tight",facecolor="white")
            print(f"[SAVE] 原始图像: {save_path}")
        return fig

    def plot_curve(self, save_path=None, dpi=200):
        p=self.ext.profile; pos=self.ext.pixel_positions; s=self.ext.stats; m=s["mean"]
        fig,ax=plt.subplots(1,1,figsize=(12,6))
        ax.plot(pos,p,color="#2563EB",lw=1.2,label="Gray Value")
        ax.axhline(y=m,color="#DC2626",lw=1,ls="--",alpha=0.7,label=f"Mean={m:.1f}")
        if self.ext.mode=="row":
            ax.set_title(f"Gray Value along Row {self.ext.abs_index}",fontsize=13,fontweight="bold")
            ax.set_xlabel("Column Index",fontsize=11)
        else:
            ax.set_title(f"Gray Value along Column {self.ext.abs_index}",fontsize=13,fontweight="bold")
            ax.set_xlabel("Row Index",fontsize=11)
        ax.set_ylabel("Gray Value",fontsize=11); ax.set_ylim(-5,265)
        ax.set_xlim(pos[0],pos[-1]); ax.legend(loc="upper right",fontsize=10)
        ax.grid(True,alpha=0.3,ls="--")
        ax.text(0.02,0.95,f"Min={s['min']:.0f} Max={s['max']:.0f} Mean={m:.1f} Std={s['std']:.1f}",
                transform=ax.transAxes,fontsize=9,va="top",
                bbox=dict(boxstyle="round,pad=0.3",fc="lightyellow",ec="gray",alpha=0.8))
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".",exist_ok=True)
            fig.savefig(save_path,dpi=dpi,bbox_inches="tight",facecolor="white")
            print(f"[SAVE] 灰度曲线: {save_path}")
        return fig


# ╔════════════════════════════════════════════════════════════════════╗
# ║               ComparisonPlotter 双图对比可视化器                    ║
# ╚════════════════════════════════════════════════════════════════════╝

class ComparisonPlotter:
    """
    双图对比可视化器，生成三张独立图：
      图 1 — 双原图并排标注（相同 ROI + 行/列标记）
      图 2 — 三线对比曲线（基准=红, 待比较=蓝, 误差=绿[右Y轴]）
      图 3 — 单独误差分布图
    """

    def __init__(self, ref_gray, cmp_gray, ref_ext, cmp_ext, roi_bounds,
                 ref_label="Reference", cmp_label="Comparison"):
        self.ref_gray=ref_gray; self.cmp_gray=cmp_gray
        self.ref_ext=ref_ext; self.cmp_ext=cmp_ext
        self.roi_bounds=roi_bounds
        self.ref_label=ref_label; self.cmp_label=cmp_label
        self.error = cmp_ext.profile - ref_ext.profile  # 逐点差值

    def plot_dual_images(self, save_path=None, dpi=200):
        """图 1：两张原图并排，标注相同 ROI 和行/列。"""
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,7))
        fig.subplots_adjust(wspace=0.15,left=0.04,right=0.97,top=0.90,bottom=0.08)
        for ax,gray,label in [(ax1,self.ref_gray,self.ref_label),
                               (ax2,self.cmp_gray,self.cmp_label)]:
            ResultPlotter._annotate(ax,gray,self.roi_bounds,
                                     self.ref_ext.mode,self.ref_ext.abs_index,title=label)
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".",exist_ok=True)
            fig.savefig(save_path,dpi=dpi,bbox_inches="tight",facecolor="white")
            print(f"[SAVE] 双原图标注: {save_path}")
        return fig

    def plot_single_annotated(self, which="ref", save_path=None, dpi=200):
        """
        生成单张原图标注图（带 ROI 框 + 行/列标记）。

        参数：
          which  — "ref" 输出基准图，"cmp" 输出待比较图
        输出分辨率固定为 1920×1080。
        """
        if which == "ref":
            gray, label = self.ref_gray, self.ref_label
        else:
            gray, label = self.cmp_gray, self.cmp_label

        target_dpi = 100
        fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
        fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.06)
        ResultPlotter._annotate(ax, gray, self.roi_bounds,
                                self.ref_ext.mode, self.ref_ext.abs_index, title=label)
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=target_dpi, facecolor="white")
            print(f"[SAVE] 单图标注({label}): {save_path} (1920×1080)")
        return fig

    def plot_comparison_curve(self, save_path=None, dpi=200):
        """
        图 2：三线对比曲线（双 Y 轴）。
        左 Y 轴 — 基准(红) + 待比较(蓝) 的灰度值
        右 Y 轴 — 误差(绿)

        输出分辨率固定为 1920×1080。
        基准曲线（Non Highly Reflective）使用更粗的线宽，确保与待比较曲线
        在重合区域也能清晰区分。
        """
        pos=self.ref_ext.pixel_positions
        rp=self.ref_ext.profile; cp=self.cmp_ext.profile; err=self.error
        mode=self.ref_ext.mode; ai=self.ref_ext.abs_index

        # ── 固定输出分辨率 1920×1080 ──
        # figsize(英寸) × dpi = 像素，19.2×10.8 @ 100dpi = 1920×1080
        target_dpi = 100
        fig,axL=plt.subplots(1,1,figsize=(19.2, 10.8))
        fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.09)

        # 左 Y 轴：灰度
        # ★ 基准曲线：加粗(lw=3.5) + 低 zorder，作为底层背景线
        l1,=axL.plot(pos,rp,color="#DC2626",lw=3.5,alpha=0.85,
                     label=self.ref_label,zorder=3)
        # ★ 待比较曲线：标准线宽实线，高 zorder，始终显示在基准线上方
        l2,=axL.plot(pos,cp,color="#2563EB",lw=1.6,
                     label=self.cmp_label,zorder=4)
        axL.set_ylabel("Pixel Gray Value",fontsize=13,color="red")
        axL.tick_params(axis="y",labelcolor="red",labelsize=11)
        axL.set_ylim(-60,270)
        xlabel = "Pixel Coordinates (Column)" if mode=="row" else "Pixel Coordinates (Row)"
        title  = f"Dual-Image Comparison — {'Row' if mode=='row' else 'Column'} {ai}"
        axL.set_xlabel(xlabel,fontsize=12)
        axL.set_title(title,fontsize=15,fontweight="bold",pad=12)
        axL.set_xlim(pos[0],pos[-1])
        axL.tick_params(axis="x",labelsize=11)
        axL.grid(True,alpha=0.2,ls="--")

        # 右 Y 轴：误差
        axR=axL.twinx()
        l3,=axR.plot(pos,err,color="#16A34A",lw=1.3,label="Error (Diff)",zorder=2)
        emax=max(np.abs(err).max()*1.3,1.0)
        axR.set_ylabel("Error Value (Diff)",fontsize=13,color="green")
        axR.tick_params(axis="y",labelcolor="green",labelsize=11)
        axR.set_ylim(-emax,emax)

        # 统一图例 — 字号适当增大以匹配 1920×1080 画幅
        axL.legend([l1,l2,l3],[l.get_label() for l in [l1,l2,l3]],
                   loc="upper right",fontsize=12,framealpha=0.9)

        # 统计信息
        rmse=np.sqrt(np.mean(err**2)); mae=np.mean(np.abs(err))
        txt=(f"Ref:  Mean={np.mean(rp):.1f}, Std={np.std(rp):.1f}\n"
             f"Cmp:  Mean={np.mean(cp):.1f}, Std={np.std(cp):.1f}\n"
             f"Err:  RMSE={rmse:.1f}, MAE={mae:.1f}, |Max|={np.abs(err).max():.1f}")
        axL.text(0.02,0.95,txt,transform=axL.transAxes,fontsize=10,va="top",
                 family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4",fc="lightyellow",ec="gray",alpha=0.85))

        # ── 保存时不使用 bbox_inches="tight"，以保证精确 1920×1080 ──
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".",exist_ok=True)
            fig.savefig(save_path, dpi=target_dpi, facecolor="white")
            print(f"[SAVE] 对比曲线: {save_path} (1920×1080)")
        return fig

    def plot_error_curve(self, save_path=None, dpi=200):
        """图 3：单独误差分布图。输出分辨率固定为 1920×1080。"""
        pos=self.ref_ext.pixel_positions; err=self.error
        mode=self.ref_ext.mode; ai=self.ref_ext.abs_index

        # ── 固定输出分辨率 1920×1080 ──
        target_dpi = 100
        fig,ax=plt.subplots(1,1,figsize=(19.2, 10.8))
        fig.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.09)

        ax.fill_between(pos,0,err,where=(err>=0),color="#16A34A",alpha=0.3,interpolate=True)
        ax.fill_between(pos,0,err,where=(err<0), color="#DC2626",alpha=0.3,interpolate=True)
        ax.plot(pos,err,color="#16A34A",lw=1.4,label="Error")
        ax.axhline(y=0,color="black",lw=0.8)

        xlabel="Pixel Coordinates (Column)" if mode=="row" else "Pixel Coordinates (Row)"
        title =f"Error Distribution — {'Row' if mode=='row' else 'Column'} {ai}"
        ax.set_xlabel(xlabel,fontsize=12); ax.set_title(title,fontsize=15,fontweight="bold",pad=12)
        ax.set_ylabel("Error (Comparison − Reference)",fontsize=12)
        ax.set_xlim(pos[0],pos[-1]); ax.legend(loc="upper right",fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(True,alpha=0.3,ls="--")
        rmse=np.sqrt(np.mean(err**2)); mae=np.mean(np.abs(err))
        ax.text(0.02,0.95,f"RMSE={rmse:.2f}  MAE={mae:.2f}  Max={err.max():.1f}  Min={err.min():.1f}",
                transform=ax.transAxes,fontsize=10,va="top",
                bbox=dict(boxstyle="round,pad=0.3",fc="lightyellow",ec="gray",alpha=0.8))

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".",exist_ok=True)
            fig.savefig(save_path, dpi=target_dpi, facecolor="white")
            print(f"[SAVE] 误差曲线: {save_path} (1920×1080)")
        return fig


# ╔════════════════════════════════════════════════════════════════════╗
# ║                    DataExporter 数据导出器                         ║
# ╚════════════════════════════════════════════════════════════════════╝

class DataExporter:
    """多格式数据导出（支持单图和双图对比）。"""

    @staticmethod
    def export_npy(data, save_path):
        os.makedirs(os.path.dirname(save_path) or ".",exist_ok=True)
        np.save(save_path,data); print(f"[SAVE] NPY: {save_path}")

    @staticmethod
    def export_mat_single(ext, roi_bounds, img_shape, save_path):
        os.makedirs(os.path.dirname(save_path) or ".",exist_ok=True)
        savemat(save_path,{
            "gray_profile":ext.profile,"pixel_positions":ext.pixel_positions,
            "extract_mode":ext.mode,"abs_line_index":ext.abs_index,
            "roi_row_start":roi_bounds[0],"roi_row_end":roi_bounds[1],
            "roi_col_start":roi_bounds[2],"roi_col_end":roi_bounds[3],
            "image_height":img_shape[0],"image_width":img_shape[1],
            "mean_value":ext.stats["mean"],"std_value":ext.stats["std"],
        }); print(f"[SAVE] MAT: {save_path}")

    @staticmethod
    def export_mat_comparison(ref_ext, cmp_ext, error, roi_bounds, ref_shape, save_path):
        os.makedirs(os.path.dirname(save_path) or ".",exist_ok=True)
        rmse=float(np.sqrt(np.mean(error**2))); mae=float(np.mean(np.abs(error)))
        savemat(save_path,{
            "ref_profile":ref_ext.profile,"cmp_profile":cmp_ext.profile,
            "error":error,"pixel_positions":ref_ext.pixel_positions,
            "extract_mode":ref_ext.mode,"abs_line_index":ref_ext.abs_index,
            "roi_row_start":roi_bounds[0],"roi_row_end":roi_bounds[1],
            "roi_col_start":roi_bounds[2],"roi_col_end":roi_bounds[3],
            "image_height":ref_shape[0],"image_width":ref_shape[1],
            "ref_mean":ref_ext.stats["mean"],"ref_std":ref_ext.stats["std"],
            "cmp_mean":cmp_ext.stats["mean"],"cmp_std":cmp_ext.stats["std"],
            "error_rmse":rmse,"error_mae":mae,
        }); print(f"[SAVE] MAT: {save_path}")


# ╔════════════════════════════════════════════════════════════════════╗
# ║                   FringeAnalyzer 单图分析主控器                    ║
# ╚════════════════════════════════════════════════════════════════════╝

class FringeAnalyzer:
    """单图灰度分析主控器。"""
    def __init__(self, image_path, output_dir="./output", output_basename="gray_profile"):
        self.output_dir=output_dir; self.output_basename=output_basename
        if not os.path.isfile(image_path):
            SampleGenerator.generate(image_path)
        self.loader=ImageLoader(image_path)
        self.selector=ROISelector(self.loader.gray_img)
        self.extractor=None

    def run_interactive(self):
        print("="*62); print("  单图灰度分析 — 交互模式"); print("="*62)
        print(self.loader.info()); print()
        self.selector.select_interactive(); self._do()

    def run_manual(self, mode="row", line_index=0, roi=None):
        print("="*62); print("  单图灰度分析 — 命令行模式"); print("="*62)
        print(self.loader.info()); print()
        self.selector.select_manual(mode,line_index,roi); self._do()

    def _do(self):
        rb=self.selector.roi_bounds
        self.extractor=ProfileExtractor(self.selector.roi_image,self.selector.mode,
                                         self.selector.line_index,(rb[0],rb[2]))
        print(); print(self.extractor.summary())
        rp=ResultPlotter(self.loader.gray_img,self.extractor,rb)
        bn=self.output_basename; od=self.output_dir
        f1=rp.plot_image(os.path.join(od,f"{bn}_image.png")); plt.close(f1)
        f2=rp.plot_curve(os.path.join(od,f"{bn}_curve.png")); plt.close(f2)
        DataExporter.export_npy(self.extractor.profile,os.path.join(od,f"{bn}.npy"))
        DataExporter.export_mat_single(self.extractor,rb,self.loader.shape,
                                        os.path.join(od,f"{bn}.mat"))


# ╔════════════════════════════════════════════════════════════════════╗
# ║              DualImageComparator 双图对比主控器                     ║
# ╚════════════════════════════════════════════════════════════════════╝

class DualImageComparator:
    """
    双图对比主控器。

    接收基准图和待比较图，使用完全相同的 ROI 和行/列参数，
    提取两条灰度剖面并生成对比可视化与数据导出。

    输出文件：
      {basename}_images.png    — 双原图并排标注
      {basename}_compare.png   — 三线对比曲线（红/蓝/绿，双Y轴）
      {basename}_error.png     — 单独误差分布图
      {basename}.npy           — 字典: ref_profile / cmp_profile / error
      {basename}.mat           — 完整数据（MATLAB 可读）
    """

    def __init__(self, ref_path, cmp_path,
                 output_dir="./output_compare", output_basename="comparison",
                 ref_label="Reference", cmp_label="Comparison"):
        self.output_dir=output_dir; self.output_basename=output_basename
        self.ref_label=ref_label; self.cmp_label=cmp_label

        if not os.path.isfile(ref_path):
            print(f"[WARN] 基准图不存在，自动生成: {ref_path}")
            SampleGenerator.generate(ref_path)
        if not os.path.isfile(cmp_path):
            print(f"[WARN] 待比较图不存在，自动生成高反光版: {cmp_path}")
            SampleGenerator.generate_highlight(cmp_path)

        self.ref_loader=ImageLoader(ref_path)
        self.cmp_loader=ImageLoader(cmp_path)

        if self.ref_loader.shape != self.cmp_loader.shape:
            raise ValueError(f"尺寸不一致! {self.ref_loader.shape} vs {self.cmp_loader.shape}")

        self.selector=ROISelector(self.ref_loader.gray_img)
        self.ref_ext=None; self.cmp_ext=None

    def run_interactive(self):
        self._header()
        self.selector.select_interactive()
        self._do()

    def run_manual(self, mode="row", line_index=0, roi=None):
        self._header()
        self.selector.select_manual(mode,line_index,roi)
        self._do()

    def _header(self):
        print("="*62)
        print("  条纹投影图像灰度分析 — 双图对比模式")
        print("  Dual-Image Comparison")
        print("="*62)
        print(f"  基准图  : {self.ref_loader.info()}")
        print(f"  待比较  : {self.cmp_loader.info()}")
        print()

    def _do(self):
        rb=self.selector.roi_bounds; m=self.selector.mode; li=self.selector.line_index
        off=(rb[0],rb[2])

        ref_roi=self.ref_loader.gray_img[rb[0]:rb[1],rb[2]:rb[3]]
        cmp_roi=self.cmp_loader.gray_img[rb[0]:rb[1],rb[2]:rb[3]]

        self.ref_ext=ProfileExtractor(ref_roi,m,li,off)
        self.cmp_ext=ProfileExtractor(cmp_roi,m,li,off)
        error=self.cmp_ext.profile - self.ref_ext.profile

        print(f"\n【基准图】 {self.ref_ext.summary()}")
        print(f"【待比较】 {self.cmp_ext.summary()}")
        rmse=np.sqrt(np.mean(error**2)); mae=np.mean(np.abs(error))
        print(f"【误  差】 RMSE={rmse:.2f}, MAE={mae:.2f}, |Max|={np.abs(error).max():.1f}")

        cp=ComparisonPlotter(self.ref_loader.gray_img,self.cmp_loader.gray_img,
                             self.ref_ext,self.cmp_ext,rb,self.ref_label,self.cmp_label)
        od=self.output_dir; bn=self.output_basename; os.makedirs(od,exist_ok=True)

        p1=os.path.join(od,f"{bn}_images.png")
        p2=os.path.join(od,f"{bn}_compare.png")
        p3=os.path.join(od,f"{bn}_error.png")
        p4=os.path.join(od,f"{bn}_ref.png")
        p5=os.path.join(od,f"{bn}_cmp.png")
        for fig,path in [(cp.plot_dual_images(p1),p1),
                          (cp.plot_comparison_curve(p2),p2),
                          (cp.plot_error_curve(p3),p3),
                          (cp.plot_single_annotated("ref",p4),p4),
                          (cp.plot_single_annotated("cmp",p5),p5)]:
            plt.close(fig)

        npy_p=os.path.join(od,f"{bn}.npy")
        DataExporter.export_npy({"ref_profile":self.ref_ext.profile,
                                  "cmp_profile":self.cmp_ext.profile,
                                  "error":error,
                                  "pixel_positions":self.ref_ext.pixel_positions},npy_p)
        mat_p=os.path.join(od,f"{bn}.mat")
        DataExporter.export_mat_comparison(self.ref_ext,self.cmp_ext,error,rb,
                                            self.ref_loader.shape,mat_p)

        print()
        print("-"*55)
        print("  对比分析完成！输出文件：")
        print(f"    双原图标注 : {p1}")
        print(f"    基准图单独 : {p4}")
        print(f"    比较图单独 : {p5}")
        print(f"    对比曲线   : {p2}")
        print(f"    误差分布   : {p3}")
        print(f"    数据 NPY   : {npy_p}")
        print(f"    数据 MAT   : {mat_p}")
        print("-"*55)
        print(f"  误差: RMSE={rmse:.4f}, MAE={mae:.4f}")
        print(f"  ★ MATLAB: d=load('{bn}.mat'); plot(d.pixel_positions,d.ref_profile,'r',...)")
        print(f"  ★ Python: d=np.load('{bn}.npy',allow_pickle=True).item()")
        print("="*62)


# ╔════════════════════════════════════════════════════════════════════╗
# ║                          程序入口                                  ║
# ╚════════════════════════════════════════════════════════════════════╝

# ============================================================
# ★★★ 用户可修改区域 — 在此处设置所有输入参数 ★★★
# ============================================================

# 【运行模式】 "single" = 单图分析,  "compare" = 双图对比
RUN_MODE = "compare"

# ──── 双图对比模式参数 ────
REF_IMAGE_PATH = "./v10.bmp"       # 基准正弦条纹图像路径
CMP_IMAGE_PATH = "./v10.bmp"       # 待比较条纹图像路径
REF_LABEL      = "Non Highly Reflective"  # 基准图标签
CMP_LABEL      = "Highly Reflective"      # 待比较图标签

# ──── 单图模式参数 ────
SINGLE_IMAGE_PATH = "./v10.bmp"

# ──── 共享参数 ────
OUTPUT_DIR      = "./output"
OUTPUT_BASENAME = "gray_profile"
EXTRACT_MODE    = "row"         # "row" 或 "col"
LINE_INDEX      = 300           # ROI 内的行/列号
ROI             = None          # (r0, r1, c0, c1) 或 None=全图

# ============================================================
# ★★★ 用户可修改区域结束 ★★★
# ============================================================


def main():
    if RUN_MODE == "compare":
        comp = DualImageComparator(
            ref_path=REF_IMAGE_PATH, cmp_path=CMP_IMAGE_PATH,
            output_dir=OUTPUT_DIR, output_basename=OUTPUT_BASENAME,
            ref_label=REF_LABEL, cmp_label=CMP_LABEL)
        if _INTERACTIVE:
            comp.run_interactive()
        else:
            comp.run_manual(mode=EXTRACT_MODE, line_index=LINE_INDEX, roi=ROI)

    elif RUN_MODE == "single":
        analyzer = FringeAnalyzer(
            image_path=SINGLE_IMAGE_PATH,
            output_dir=OUTPUT_DIR, output_basename=OUTPUT_BASENAME)
        if _INTERACTIVE:
            analyzer.run_interactive()
        else:
            analyzer.run_manual(mode=EXTRACT_MODE, line_index=LINE_INDEX, roi=ROI)
    else:
        print(f"[ERROR] RUN_MODE='{RUN_MODE}' 无效，应为 'single' 或 'compare'")


if __name__ == "__main__":
    main()
