"""
文件名：three_frequency_phase_visualization.py
功能：
1. 读取3个频率的N步相移条纹图像
2. 计算三个频率的包裹相位
3. 利用多频外差/塔形展开逻辑生成绝对相位
4. 读取掩码文件（.npy 或 .png），提取掩码区域内的绝对相位
5. 保存绝对相位与掩码相位为 .npy 和 .mat 格式
6. 在掩码区域上进行可视化显示，并保存图片
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from typing import Optional
from scipy.io import savemat

# =============================================================================
# 部分 A：来自《单频相位处理工具.py》的核心算法与逻辑
# =============================================================================

class ImageLoader:
    """图像加载器：负责读取条纹图像"""
    @staticmethod
    def load_single_frequency_images(folder_path, phase_shift_step=4, orientation='h', frequency_index=0, image_ext=None):
        if image_ext is None:
            # 自动检测扩展名
            common_exts = ['.bmp', '.png', '.jpg', '.jpeg']
            for ext in common_exts:
                if os.path.exists(os.path.join(folder_path, f"{orientation}1{ext}")):
                    image_ext = ext
                    break
        
        if image_ext is None:
            raise FileNotFoundError(f"未找到图像文件，路径: {folder_path}")

        images = []
        N = phase_shift_step
        # 计算该频率段起始的索引（假设所有频率都在同一文件夹下或遵循特定索引）
        start_idx = frequency_index * N + 1
        for i in range(start_idx, start_idx + N):
            path = os.path.join(folder_path, f"{orientation}{i}{image_ext}")
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"无法读取图像: {path}")
            images.append(img)
        return np.stack(images, axis=2)


class MaskLoader:
    """掩码加载器：支持 .npy 和 .png 格式"""
    @staticmethod
    def load_mask(mask_path):
        """
        读取掩码文件，返回布尔类型的二维数组。
        支持 .npy 和 .png（以及其他常见图像格式）。
        掩码约定：非零区域为有效区域（True），零区域为无效区域（False）。
        """
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"掩码文件不存在: {mask_path}")
        
        ext = os.path.splitext(mask_path)[1].lower()
        
        if ext == '.npy':
            mask = np.load(mask_path)
        elif ext in ['.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff']:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"无法读取掩码图像: {mask_path}")
        else:
            raise ValueError(f"不支持的掩码文件格式: {ext}，请使用 .npy 或 .png")
        
        # 转换为布尔掩码（非零为True）
        mask_bool = mask.astype(bool)
        print(f"掩码已加载: {mask_path}，形状={mask_bool.shape}，有效像素数={np.sum(mask_bool)}")
        return mask_bool


class PhaseCalculator:
    """相位计算器：包裹相位计算、相位差（外差）、相位展开"""
    @staticmethod
    def calculate_single_frequency_wrapped_phase(stripe_images):
        H, W, N = stripe_images.shape
        imgs = stripe_images.astype(np.float64)
        phase_shifts = 2 * np.pi * np.arange(N) / N
        
        cos_sum = np.zeros((H, W), dtype=np.float64)
        sin_sum = np.zeros((H, W), dtype=np.float64)
        for i in range(N):
            cos_sum += imgs[:, :, i] * np.cos(phase_shifts[i])
            sin_sum += imgs[:, :, i] * np.sin(phase_shifts[i])
        
        wrapped_phase = -np.arctan2(sin_sum, cos_sum)
        wrapped_phase = np.mod(wrapped_phase, 2 * np.pi)
        return wrapped_phase

    @staticmethod
    def calculate_phase_difference(phase1, phase2):
        """计算相位差（外差过程）"""
        p1 = phase1 / (2 * np.pi)
        p2 = phase2 / (2 * np.pi)
        result = p1 - p2
        result[result < 0] += 1
        return result * (2 * np.pi)

    @staticmethod
    def unwrap_phase_with_reference(reference_phase, target_phase, reference_freq, target_freq):
        """使用参考相位展开高频相位"""
        ref = reference_phase / (2 * np.pi)
        tar = target_phase / (2 * np.pi)
        temp = target_freq / reference_freq * ref
        k = np.round(temp - tar)
        unwrap_phase = (tar + k) * (2 * np.pi)
        return unwrap_phase

# =============================================================================
# 部分 B：来自《相位数据处理工具.py》的可视化与配置
# =============================================================================

# 配置中文字体显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


def save_phase_data(absolute_phase, masked_phase, mask, save_dir,
                    wrapped_phases=None, intermediate_phases=None,
                    freq_list=None):
    """
    保存相位数据，格式与 绝对相位可视化工具.py 完全一致，
    可直接被 phase_visualization.m 识别并显示。

    保存规则（与 绝对相位可视化工具.py 保持一致）：
      - 主输出 .mat：单变量 {'phase_data': 2D_float64}
          * 不使用掩码版：全场绝对相位（无 NaN）
          * 使用掩码版：掩码外像素置为 NaN
      - 主输出 .npy：与对应 .mat 内容相同的 2D 数组
      - phase_visualization.m 加载 phase_data 变量即可直接使用

    Parameters
    ----------
    absolute_phase : ndarray (H, W), float64
        最终绝对相位（弧度）
    masked_phase : ndarray (H, W), float64
        掩码区域内的绝对相位，掩码外已置为 NaN
    mask : ndarray (H, W), bool / uint8
        布尔掩码（True / 1 = 有效区域）
    save_dir : str
        保存目录
    wrapped_phases : list of ndarray, optional
        三个频率的包裹相位 [高频, 中频, 低频]，额外保存为独立文件
    intermediate_phases : list of ndarray, optional
        外差中间相位列表，额外保存为独立文件
    freq_list : list of int, optional
        三个频率值 [高, 中, 低]
    """
    os.makedirs(save_dir, exist_ok=True)
    H, W = absolute_phase.shape

    # 确保 mask 为 uint8（0/1），与 绝对相位可视化工具.py 一致
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)

    # =========================================================================
    # ① 使用掩码前——全场绝对相位
    #    格式：{'phase_data': 2D float64}，与 绝对相位可视化工具.py 完全一致
    # =========================================================================
    folder_no_mask = os.path.join(save_dir, "使用掩码前")
    os.makedirs(folder_no_mask, exist_ok=True)

    phase_no_mask = absolute_phase.astype(np.float64)   # 全场，无 NaN

    # .npy
    npy_no_mask = os.path.join(folder_no_mask, "absolute_phase_original.npy")
    np.save(npy_no_mask, phase_no_mask)
    print(f"  [使用掩码前] npy 已保存: {npy_no_mask}")

    # .mat — 变量名 phase_data，可直接被 phase_visualization.m 读取
    mat_no_mask = os.path.join(folder_no_mask, "absolute_phase_original.mat")
    savemat(mat_no_mask, {"phase_data": phase_no_mask})
    print(f"  [使用掩码前] mat 已保存: {mat_no_mask}")

    # =========================================================================
    # ② 使用掩码后——掩码区域绝对相位（掩码外为 NaN）
    #    格式：{'phase_data': 2D float64}，与 绝对相位可视化工具.py 完全一致
    # =========================================================================
    folder_with_mask = os.path.join(save_dir, "使用掩码后")
    os.makedirs(folder_with_mask, exist_ok=True)

    # 应用掩码：mask==0 的位置置为 NaN（与 绝对相位可视化工具.py 第1394~1396行一致）
    phase_with_mask = absolute_phase.astype(np.float64).copy()
    phase_with_mask[mask_u8 == 0] = np.nan

    # .npy
    npy_with_mask = os.path.join(folder_with_mask, "absolute_phase_masked.npy")
    np.save(npy_with_mask, phase_with_mask)
    print(f"  [使用掩码后] npy 已保存: {npy_with_mask}")

    # .mat — 变量名 phase_data，可直接被 phase_visualization.m 读取
    mat_with_mask = os.path.join(folder_with_mask, "absolute_phase_masked.mat")
    savemat(mat_with_mask, {"phase_data": phase_with_mask})
    print(f"  [使用掩码后] mat 已保存: {mat_with_mask}")

    # =========================================================================
    # ③ 中间相位（包裹相位 + 外差相位）独立保存
    #    每个文件同样使用 phase_data 作为变量名，保持格式统一
    # =========================================================================
    inter_keys = [
        "diff_phase_high_mid",
        "diff_phase_mid_low",
        "diff_phase_final_low",
        "unwrapped_diff_phase",
    ]

    if wrapped_phases is not None and len(wrapped_phases) == 3:
        wrap_labels = ["wrapped_phase_high", "wrapped_phase_mid", "wrapped_phase_low"]
        for lbl, arr in zip(wrap_labels, wrapped_phases):
            arr_f64 = arr.astype(np.float64)
            np.save(os.path.join(folder_no_mask, f"{lbl}.npy"), arr_f64)
            savemat(os.path.join(folder_no_mask, f"{lbl}.mat"), {"phase_data": arr_f64})
        print(f"  [中间相位] 三频包裹相位已保存至: {folder_no_mask}")

    if intermediate_phases is not None:
        for k, arr in zip(inter_keys, intermediate_phases):
            arr_f64 = arr.astype(np.float64)
            np.save(os.path.join(folder_no_mask, f"{k}.npy"), arr_f64)
            savemat(os.path.join(folder_no_mask, f"{k}.mat"), {"phase_data": arr_f64})
        print(f"  [中间相位] 外差相位已保存至: {folder_no_mask}")

    # =========================================================================
    # ④ 掩码文件
    # =========================================================================
    np.save(os.path.join(save_dir, "mask.npy"), mask_u8)
    savemat(os.path.join(save_dir, "mask.mat"), {"mask": mask_u8})
    print(f"  [掩码] mask.npy / mask.mat 已保存至: {save_dir}")

    # =========================================================================
    # 统计信息
    # =========================================================================
    valid_vals = phase_with_mask[~np.isnan(phase_with_mask)]
    print(f"\n  绝对相位统计（掩码内有效区域）:")
    print(f"    范围: [{np.nanmin(phase_with_mask):.4f}, {np.nanmax(phase_with_mask):.4f}] rad")
    print(f"    均值: {np.mean(valid_vals):.4f} rad  标准差: {np.std(valid_vals):.4f} rad")
    print(f"    有效像素: {len(valid_vals):,} / {H*W:,}  "
          f"({len(valid_vals)/H/W*100:.1f}%)")
    print(f"\n  MATLAB 读取示例（phase_visualization.m）:")
    print(f"    data = load('absolute_phase_masked.mat');")
    print(f"    phase = data.phase_data;   %% shape: ({H}, {W}), 掩码外为 NaN")
    print(f"\n所有相位数据已保存至: {save_dir}")


def show_phase_line_profile(phase: np.ndarray, 
                            row_index: Optional[int] = None,
                            position: str = 'center',
                            title: str = "绝对相位走势图",
                            save_path: Optional[str] = None):
    """
    可视化核心：显示指定行（纵坐标）的相位曲线
    """
    height, width = phase.shape
    if row_index is None:
        positions = {'top': height // 10, 'bottom': height * 9 // 10, 'center': height // 2}
        row_index = positions.get(position, height // 2)
    
    row_index = max(0, min(row_index, height - 1))
    line_data = phase[row_index, :]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(np.arange(width), line_data, color='blue', linewidth=1.5)
    ax.set_title(f'{title} - 第 {row_index} 行', fontsize=14)
    ax.set_xlabel('列索引 (pixels)', fontsize=12)
    ax.set_ylabel('绝对相位 (rad)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  走势图已保存: {save_path}")
    
    plt.show()


# =============================================================================
# 部分 C：整合处理流程（核心逻辑重组）
# =============================================================================

def process_3_frequency_pipeline(folder_path, phase_step, freq_list, orientation='v',
                                  mask_path=None, save_dir=None):
    """
    核心流程：输入条纹 -> 绝对相位计算 -> 掩码处理 -> 保存与可视化

    Parameters
    ----------
    folder_path : str
        条纹图像所在文件夹路径
    phase_step : int
        相移步数 N
    freq_list : list
        三个频率 [高, 中, 低]
    orientation : str
        条纹方向，'h' 水平或 'v' 垂直
    mask_path : str or None
        掩码文件路径（.npy 或 .png），为 None 则不使用掩码
    save_dir : str or None
        结果保存目录，为 None 则默认保存在 folder_path 下的 "results" 子文件夹
    """
    print(f"--- 开始处理: {folder_path} ---")
    
    # 默认保存目录
    if save_dir is None:
        save_dir = os.path.join(folder_path, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    # =========================================================================
    # 1. 读取条纹图并计算包裹相位
    # =========================================================================
    print("Step 1: 正在读取条纹图像并计算包裹相位...")
    imgs_h = ImageLoader.load_single_frequency_images(folder_path, phase_step, orientation, frequency_index=0)
    imgs_m = ImageLoader.load_single_frequency_images(folder_path, phase_step, orientation, frequency_index=1)
    imgs_l = ImageLoader.load_single_frequency_images(folder_path, phase_step, orientation, frequency_index=2)
    
    w_phase_h = PhaseCalculator.calculate_single_frequency_wrapped_phase(imgs_h)
    w_phase_m = PhaseCalculator.calculate_single_frequency_wrapped_phase(imgs_m)
    w_phase_l = PhaseCalculator.calculate_single_frequency_wrapped_phase(imgs_l)

    # =========================================================================
    # 2. 多频外差/展开逻辑
    # =========================================================================
    print("Step 2: 正在进行多频外差与相位展开...")
    phi12 = PhaseCalculator.calculate_phase_difference(w_phase_h, w_phase_m)
    phi23 = PhaseCalculator.calculate_phase_difference(w_phase_m, w_phase_l)
    phi_final_low = PhaseCalculator.calculate_phase_difference(phi12, phi23)
    
    f_diff12 = freq_list[0] - freq_list[1]
    f_diff23 = freq_list[1] - freq_list[2]
    f_final = abs(f_diff12 - f_diff23)
    
    unwrap_phi12 = PhaseCalculator.unwrap_phase_with_reference(phi_final_low, phi12, f_final, f_diff12)
    absolute_phase = PhaseCalculator.unwrap_phase_with_reference(unwrap_phi12, w_phase_h, f_diff12, freq_list[0])

    # 汇总各阶段相位，供后续完整保存
    _wrapped_phases      = [w_phase_h, w_phase_m, w_phase_l]
    _intermediate_phases = [phi12, phi23, phi_final_low, unwrap_phi12]

    # =========================================================================
    # 3. 掩码处理
    # =========================================================================
    if mask_path is not None:
        print("Step 3: 正在加载掩码并提取掩码区域相位...")
        mask = MaskLoader.load_mask(mask_path)
        
        # 检查掩码尺寸与相位图是否一致
        if mask.shape != absolute_phase.shape:
            raise ValueError(
                f"掩码尺寸 {mask.shape} 与相位图尺寸 {absolute_phase.shape} 不匹配！"
            )
        
        # 掩码区域内保留相位值，掩码外置为 NaN
        masked_phase = np.full_like(absolute_phase, np.nan)
        masked_phase[mask] = absolute_phase[mask]
    else:
        print("Step 3: 未指定掩码文件，将使用全场数据...")
        mask = np.ones(absolute_phase.shape, dtype=bool)
        masked_phase = absolute_phase.copy()

    # =========================================================================
    # 4. 保存数据（.npy 和 .mat）
    # =========================================================================
    print("Step 4: 正在保存相位数据...")
    save_phase_data(
        absolute_phase, masked_phase, mask, save_dir,
        wrapped_phases=_wrapped_phases,
        intermediate_phases=_intermediate_phases,
        freq_list=freq_list,
    )

    # =========================================================================
    # 5. 可视化并保存图片
    # =========================================================================
    print("Step 5: 生成可视化图像并保存...")
    
    # --- 5a. 全场绝对相位图 ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    im1 = ax1.imshow(absolute_phase, cmap='jet')
    ax1.set_title(f"全场绝对相位图 (频率: {freq_list[0]})", fontsize=14)
    fig1.colorbar(im1, ax=ax1, label="弧度 (rad)")
    plt.tight_layout()
    path_full = os.path.join(save_dir, "absolute_phase_full.png")
    fig1.savefig(path_full, dpi=200, bbox_inches='tight')
    print(f"  已保存: {path_full}")
    plt.show()
    
    # --- 5b. 掩码区域绝对相位图 ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    # 使用 masked array 以便掩码外区域显示为背景色
    display_phase = np.ma.masked_where(~mask, absolute_phase)
    im2 = ax2.imshow(display_phase, cmap='jet')
    ax2.set_title(f"掩码区域绝对相位图 (频率: {freq_list[0]})", fontsize=14)
    ax2.set_facecolor('white')
    fig2.colorbar(im2, ax=ax2, label="弧度 (rad)")
    plt.tight_layout()
    path_masked = os.path.join(save_dir, "absolute_phase_masked.png")
    fig2.savefig(path_masked, dpi=200, bbox_inches='tight')
    print(f"  已保存: {path_masked}")
    plt.show()
    
    # --- 5c. 掩码叠加在相位图上的对比图 ---
    fig3, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].imshow(absolute_phase, cmap='jet')
    axes[0].set_title("全场绝对相位", fontsize=13)
    
    axes[1].imshow(display_phase, cmap='jet')
    axes[1].set_title("掩码区域绝对相位", fontsize=13)
    axes[1].set_facecolor('white')
    
    for ax in axes:
        ax.set_xlabel('列索引 (pixels)')
        ax.set_ylabel('行索引 (pixels)')
    
    plt.suptitle(f"绝对相位对比 (频率: {freq_list[0]})", fontsize=15)
    plt.tight_layout()
    path_compare = os.path.join(save_dir, "phase_comparison.png")
    fig3.savefig(path_compare, dpi=200, bbox_inches='tight')
    print(f"  已保存: {path_compare}")
    plt.show()
    
    # --- 5d. 掩码区域中心行相位走势 ---
    show_phase_line_profile(
        masked_phase, 
        position='center', 
        title="掩码区域绝对相位中心行走势",
        save_path=os.path.join(save_dir, "phase_line_profile_masked.png")
    )
    
    print(f"\n所有结果已保存至: {save_dir}")
    return absolute_phase, masked_phase, mask

# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == "__main__":
    # ========================= 配置区 =========================
    IMAGE_FOLDER = r"E:\images\PSM_FSM\O1"     # 请修改为您的实际条纹图像路径
    N_STEP = 4                                   # 相移步数
    FREQUENCIES = [81, 72, 64]                   # 三个频率 [高, 中, 低]
    STRIPE_DIR = 'v'                             # 'h' 水平或 'v' 垂直
    
    # 掩码文件路径（支持 .npy 或 .png，设为 None 则不使用掩码）
    MASK_PATH = r"integrated_results\mask\mask.bmp"  # 请修改为您的实际掩码路径
    
    # 结果保存目录（设为 None 则默认保存在图像文件夹下的 results 子目录）
    SAVE_DIR = None
    # ==========================================================
    
    try:
        abs_phase, masked_phase, mask = process_3_frequency_pipeline(
            folder_path=IMAGE_FOLDER,
            phase_step=N_STEP,
            freq_list=FREQUENCIES,
            orientation=STRIPE_DIR,
            mask_path=MASK_PATH,
            save_dir=SAVE_DIR
        )
        print("\n--- 处理完成 ---")
    except Exception as e:
        print(f"\n❌ 处理出错: {e}")
