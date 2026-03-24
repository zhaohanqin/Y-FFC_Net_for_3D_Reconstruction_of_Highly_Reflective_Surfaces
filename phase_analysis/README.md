# phase_analysis 可视化脚本说明

本目录下的脚本主要用于 **条纹图像的灰度剖面可视化**、以及 **三频相位（包裹相位/外差相位/绝对相位）的生成、保存与可视化**。

## 运行环境与依赖

- **Python**: 建议 3.8+
- **依赖**: `numpy`, `opencv-python (cv2)`, `matplotlib`, `scipy`
- **交互模式说明**:
  - 脚本内部会尝试启用 `matplotlib` 的 `TkAgg` 后端来支持 **鼠标框选 ROI / 滑块** 等交互控件。
  - 若环境不支持图形界面（例如远程/无显示器），脚本会自动切换到 `Agg`，此时请使用脚本中提供的 **手动参数**（ROI、行/列索引等）。

## 数据/文件命名约定（条纹相移图像）

多个脚本默认使用同一种相移条纹命名方式（以 `orientation` 决定前缀）：

- **垂直条纹**: `v1.*`, `v2.*`, ...（`orientation='v'`）
- **水平条纹**: `h1.*`, `h2.*`, ...（`orientation='h'`）
- **相移步数**: `N = phase_shift_step`（常见为 4 步）
- **三频数据组织**: 3 个频率段连续排列，每个频率段有 N 张图。以 `N=4` 为例：
  - 频率段 0：`v1..v4`
  - 频率段 1：`v5..v8`
  - 频率段 2：`v9..v12`

支持常见扩展名：`.bmp` / `.png` / `.jpg` / `.jpeg`。

## 脚本一览

### 1) `fringe_gray_analysis.py`

**用途**

- 对单张条纹图像进行灰度剖面分析：**ROI 框选** → 选择 **行/列** → 绘制灰度曲线并导出数据。
- 支持双图对比：同 ROI、同行/列下比较两张图的灰度剖面，并输出 **误差曲线**（Comparison − Reference）。

**怎么用**

- 直接运行脚本；在文件底部“用户可修改区域”修改以下参数：
  - **`RUN_MODE`**: `"single"`（单图）或 `"compare"`（双图对比）
  - **单图**: `SINGLE_IMAGE_PATH`
  - **双图**: `REF_IMAGE_PATH`, `CMP_IMAGE_PATH`, `REF_LABEL`, `CMP_LABEL`
  - **输出**: `OUTPUT_DIR`, `OUTPUT_BASENAME`
  - **手动提取参数**: `EXTRACT_MODE`（`"row"`/`"col"`）, `LINE_INDEX`, `ROI=(r0,r1,c0,c1)` 或 `None`

**输出**

- 单图模式（默认 basename 为 `gray_profile`）：
  - `..._image.png`: 原图 + ROI 框 + 行/列标记
  - `..._curve.png`: 灰度剖面曲线
  - `... .npy`: 灰度剖面数组
  - `... .mat`: MATLAB 可读结构（剖面、ROI、统计信息等）
- 双图对比模式：
  - `..._images.png`: 双原图并排标注
  - `..._compare.png`: 对比曲线（红/蓝 + 右轴误差绿线）
  - `..._error.png`: 误差分布图
  - `..._ref.png`, `..._cmp.png`: 单图标注（固定输出 1920×1080）
  - `... .npy` / `... .mat`: 对比数据与误差

---

### 2) `three_frequency_phase_visualization.py`

**用途**

- 读取三频 N 步相移条纹图像，计算：
  - 三个频率的 **包裹相位**
  - 两两外差相位 + 最低等效相位
  - 逐级参考展开得到 **绝对相位**
- 可选读取掩码（`.npy` 或图像），对掩码区域做可视化并保存结果。

**怎么用**

- 在脚本底部“主程序入口”修改参数后运行：
  - **`IMAGE_FOLDER`**: 条纹图像所在文件夹
  - **`N_STEP`**: 相移步数（如 4）
  - **`FREQUENCIES`**: 三频列表 `[高, 中, 低]`（例如 `[81, 72, 64]`）
  - **`STRIPE_DIR`**: `'v'` 或 `'h'`
  - **`MASK_PATH`**: 掩码路径（支持 `.npy` / `.png` / `.bmp` 等），`None` 则全场
  - **`SAVE_DIR`**: 输出目录（`None` 则输出到 `IMAGE_FOLDER/results`）

**输出**

- 数据保存（与脚本内 `save_phase_data()` 约定一致）：
  - `使用掩码前/absolute_phase_original.(npy|mat)`（`mat` 变量名为 `phase_data`）
  - `使用掩码后/absolute_phase_masked.(npy|mat)`（掩码外为 `NaN`，`mat` 变量名 `phase_data`）
  - `mask.(npy|mat)`
  - 以及中间相位（包裹相位、外差相位）单独保存为 `*.npy/*.mat`
- 可视化图片：
  - `absolute_phase_full.png`
  - `absolute_phase_masked.png`
  - `phase_comparison.png`
  - `phase_line_profile_masked.png`（掩码区域中心行走势）

---

### 3) `integrated_absolute_phase_tool.py`

**用途（推荐作为“整合版流水线”）**

把“ROI 选择/掩码生成 + 三频绝对相位计算 + 数据保存 + 多种可视化”整合到一个脚本里，主流程为：

- **STEP 1**: ROI 选择（交互/手动）→ 生成 `mask`
- **STEP 2**: 三频绝对相位计算（包裹相位 → 外差 → 逐级参考展开）
- **STEP 2b**: 保存完整相位链（`absolute_phase` + 中间相位）为 `.npy/.mat`
- **STEP 4**: 可视化输出：
  - 单行/列相位剖面曲线（归一化到 \(0 \sim 2\pi\)）
  - 2D 伪彩色相位图 + 标记线
  - 可选 3D 表面图

**怎么用**

- 在脚本底部“用户参数区”修改后直接运行：
  - **`PROJECTION_FOLDER`**: 条纹相移图像文件夹
  - **`PHASE_SHIFT_STEP`**: 相移步数（默认 4）
  - **`ORIENTATION`**: `'v'` 或 `'h'`
  - **`FREQ_COMBINATION`**: 形如 `"81-72-64"`（也可设为 `None` 尝试从路径提取）
  - **ROI 参数**:
    - `ROI = None`：若环境支持 GUI，则进入交互式框选；否则使用全图
    - `ROI = (r0, r1, c0, c1)`：手动 ROI（无 GUI 时推荐）
  - **剖面参数**:
    - `EXTRACT_MODE = "row" | "col"`
    - `LINE_INDEX = None` 时会按 `POSITION` 自动选取（`top/bottom/center/quarter/three_quarter`）
  - **输出**: `OUTPUT_DIR`
  - **`GENERATE_3D`**: 是否生成 3D 图

**输出目录结构（关键）**

- `OUTPUT_DIR/mask/`
  - `mask.bmp`, `mask.npy`
- `OUTPUT_DIR/phase_data/`
  - `使用掩码前/absolute_phase_*_original.(npy|mat)`
  - `使用掩码后/absolute_phase_*_masked.(npy|mat)`
  - `mask.(npy|mat)`
  - 以及中间相位 `wrapped_phase_*`, `diff_phase_*`, `unwrapped_diff_phase` 等
- `OUTPUT_DIR/visualization/`
  - `phase_line_row*/col*_curve.png`
  - `phase_line_row*/col*_image.png`
  - `phase_3d_surface_*.png`（可选）

**MATLAB 读取提示**

这些脚本保存的 `.mat` 主要遵循同一约定：相位数据变量名为 **`phase_data`**。
例如：

```matlab
d = load('absolute_phase_81-72-64_masked.mat');
phase = d.phase_data;   % 2D，相位；掩码外为 NaN
```

## 选型建议

- **只看灰度剖面 / 想做高反光前后对比**：用 `fringe_gray_analysis.py`
- **已有掩码，想快速算三频绝对相位并出图**：用 `three_frequency_phase_visualization.py`
- **想要一站式（ROI→绝对相位→保存→曲线+伪彩+3D）**：用 `integrated_absolute_phase_tool.py`

