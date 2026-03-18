"""
点云比较工具 - Python 版本
=====================================
功能：
  1. 读取两个 .ply 点云文件（基准点云 + 待比较点云）
  2. 计算待比较点云相对于基准点云的逐点最近邻欧氏距离误差
  3. 统计误差指标（均值、最大值、最小值、RMSE、标准差、中位数）
  4. 按阈值判定差异点，差异点标注为红色
  5. 生成带颜色标注的新 .ply 文件

依赖库：
  pip install numpy scipy plyfile
"""

import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
import os
import sys


# ============================================================
# 用户配置区域（根据实际情况修改）
# ============================================================
REFERENCE_PLY   = "pointCloud_hight2.ply"      # 基准点云文件路径
COMPARE_PLY     = "pointCloud_hight.ply"        # 待比较点云文件路径
OUTPUT_PLY      = "output_colored.ply" # 输出带颜色标注的点云文件路径
THRESHOLD       = 1.5                 # 差异点判定阈值（单位与点云坐标单位一致）
DEFAULT_COLOR   = (180, 180, 180)      # 非差异点的默认颜色 (R, G, B)
DIFF_COLOR      = (255, 0, 0)          # 差异点颜色：红色 (R, G, B)
# ============================================================


def read_ply(filepath: str) -> tuple[np.ndarray, np.ndarray | None]:
    """
    读取 .ply 文件，返回点坐标数组和颜色数组。

    参数：
        filepath : str  — .ply 文件路径

    返回：
        points : np.ndarray, shape (N, 3)  — XYZ 坐标
        colors : np.ndarray or None, shape (N, 3)  — RGB 颜色（uint8），若无颜色信息则为 None
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"文件未找到：{filepath}")

    plydata = PlyData.read(filepath)
    vertex = plydata["vertex"]

    # 读取坐标
    x = np.asarray(vertex["x"], dtype=np.float64)
    y = np.asarray(vertex["y"], dtype=np.float64)
    z = np.asarray(vertex["z"], dtype=np.float64)
    points = np.column_stack([x, y, z])

    # 尝试读取颜色（兼容 red/green/blue 或 r/g/b 字段名）
    colors = None
    prop_names = {p.name for p in vertex.properties}
    if {"red", "green", "blue"}.issubset(prop_names):
        r = np.asarray(vertex["red"],   dtype=np.uint8)
        g = np.asarray(vertex["green"], dtype=np.uint8)
        b = np.asarray(vertex["blue"],  dtype=np.uint8)
        colors = np.column_stack([r, g, b])
    elif {"r", "g", "b"}.issubset(prop_names):
        r = np.asarray(vertex["r"], dtype=np.uint8)
        g = np.asarray(vertex["g"], dtype=np.uint8)
        b = np.asarray(vertex["b"], dtype=np.uint8)
        colors = np.column_stack([r, g, b])

    print(f"  已读取：{filepath}  |  点数：{len(points)}"
          f"  |  含颜色：{'是' if colors is not None else '否'}")
    return points, colors


def compute_nn_distances(ref_points: np.ndarray, cmp_points: np.ndarray) -> np.ndarray:
    """
    计算待比较点云中每个点到基准点云最近邻点的欧氏距离。

    方法：对基准点云构建 cKDTree（c 版本速度更快），
          然后逐点查询待比较点云中每点的最近邻距离。

    参数：
        ref_points : np.ndarray (M, 3)  — 基准点云
        cmp_points : np.ndarray (N, 3)  — 待比较点云

    返回：
        distances : np.ndarray (N,)  — 每个待比较点对应的最近邻距离
    """
    print("  正在构建 KD-Tree（基准点云）...")
    tree = cKDTree(ref_points)
    print("  正在查询最近邻距离...")
    distances, _ = tree.query(cmp_points, k=1, workers=-1)  # workers=-1 使用全部 CPU 核心
    return distances


def print_statistics(distances: np.ndarray, threshold: float) -> dict:
    """
    统计并打印误差指标，返回统计结果字典。

    统计项目：
        mean    — 平均误差
        max     — 最大误差
        min     — 最小误差
        rmse    — 均方根误差
        std     — 标准差
        median  — 中位数
        p95     — 95 百分位数
        n_diff  — 差异点数量（误差 > 阈值）
        ratio   — 差异点占比
    """
    mean_e   = np.mean(distances)
    max_e    = np.max(distances)
    min_e    = np.min(distances)
    rmse     = np.sqrt(np.mean(distances ** 2))
    std_e    = np.std(distances)
    median_e = np.median(distances)
    p95_e    = np.percentile(distances, 95)
    n_diff   = int(np.sum(distances > threshold))
    ratio    = n_diff / len(distances) * 100

    print("\n" + "=" * 50)
    print("  点云误差统计结果")
    print("=" * 50)
    print(f"  待比较点云总点数     : {len(distances)}")
    print(f"  平均误差  (Mean)     : {mean_e:.6f}")
    print(f"  最大误差  (Max)      : {max_e:.6f}")
    print(f"  最小误差  (Min)      : {min_e:.6f}")
    print(f"  均方根误差 (RMSE)    : {rmse:.6f}")
    print(f"  标准差   (Std)       : {std_e:.6f}")
    print(f"  中位数   (Median)    : {median_e:.6f}")
    print(f"  95 百分位 (P95)      : {p95_e:.6f}")
    print(f"  差异点数（>{threshold:.4f}）: {n_diff} / {len(distances)}"
          f"  ({ratio:.2f}%)")
    print("=" * 50 + "\n")

    return {
        "mean": mean_e, "max": max_e, "min": min_e,
        "rmse": rmse, "std": std_e, "median": median_e,
        "p95": p95_e, "n_diff": n_diff, "ratio": ratio,
    }


def colorize_points(
    distances:     np.ndarray,
    orig_colors:   np.ndarray | None,
    threshold:     float,
    default_color: tuple,
    diff_color:    tuple,
) -> np.ndarray:
    """
    根据误差距离为每个点赋颜色：
      - 差异点（距离 > threshold）：diff_color（默认红色）
      - 正常点：保留原始颜色；若无原始颜色则使用 default_color

    返回：
        new_colors : np.ndarray (N, 3) uint8
    """
    N = len(distances)

    # 初始化颜色矩阵
    if orig_colors is not None:
        new_colors = orig_colors.copy().astype(np.uint8)
    else:
        new_colors = np.full((N, 3), default_color, dtype=np.uint8)

    # 标注差异点为红色
    diff_mask = distances > threshold
    new_colors[diff_mask] = diff_color

    return new_colors


def write_ply(filepath: str, points: np.ndarray, colors: np.ndarray) -> None:
    """
    将点坐标和颜色写入 .ply 文件（ASCII 格式，兼容性最佳）。

    格式说明：
      - 坐标字段：x, y, z（float32）
      - 颜色字段：red, green, blue（uint8）
      - 使用标准 PLY ASCII 格式，可被 MeshLab、CloudCompare、
        Python plyfile、MATLAB pcread 等工具正常读取。

    参数：
        filepath : str          — 输出文件路径
        points   : (N, 3) float — XYZ 坐标
        colors   : (N, 3) uint8 — RGB 颜色
    """
    N = len(points)

    # 构造结构化数组（plyfile 要求）
    vertex_data = np.zeros(
        N,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex_data["x"]     = points[:, 0].astype(np.float32)
    vertex_data["y"]     = points[:, 1].astype(np.float32)
    vertex_data["z"]     = points[:, 2].astype(np.float32)
    vertex_data["red"]   = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"]  = colors[:, 2]

    el = PlyElement.describe(vertex_data, "vertex")
    PlyData([el], text=True).write(filepath)  # text=True → ASCII 格式

    print(f"  已保存彩色点云：{filepath}  |  点数：{N}")


def main():
    print("\n========================================")
    print("  点云比较工具  (Python 版)")
    print("========================================\n")

    # ----------------------------------------------------------
    # Step 1：读取点云
    # ----------------------------------------------------------
    print("【Step 1】读取点云文件...")
    ref_points, _          = read_ply(REFERENCE_PLY)
    cmp_points, cmp_colors = read_ply(COMPARE_PLY)

    # ----------------------------------------------------------
    # Step 2：计算最近邻误差
    # ----------------------------------------------------------
    print("\n【Step 2】计算最近邻欧氏距离误差...")
    distances = compute_nn_distances(ref_points, cmp_points)

    # ----------------------------------------------------------
    # Step 3：统计误差
    # ----------------------------------------------------------
    print("\n【Step 3】统计误差指标...")
    stats = print_statistics(distances, THRESHOLD)

    # ----------------------------------------------------------
    # Step 4：差异点标注
    # ----------------------------------------------------------
    print("【Step 4】标注差异点（红色）...")
    new_colors = colorize_points(
        distances, cmp_colors, THRESHOLD, DEFAULT_COLOR, DIFF_COLOR
    )
    print(f"  差异点数量：{stats['n_diff']}  |  阈值：{THRESHOLD}")

    # ----------------------------------------------------------
    # Step 5：写出新的彩色 .ply 文件
    # ----------------------------------------------------------
    print("\n【Step 5】写出带颜色标注的点云文件...")
    write_ply(OUTPUT_PLY, cmp_points, new_colors)

    print("\n✅ 全部完成！输出文件：", OUTPUT_PLY)


if __name__ == "__main__":
    main()

