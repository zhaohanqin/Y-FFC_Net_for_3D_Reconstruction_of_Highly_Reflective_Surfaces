"""Decoupled copy of the original enhanced training entrypoint.

The root train.py is untouched. This copy keeps the same configuration block
and delegates implementation to yffc.training.engine.
"""

import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from yffc.training.engine import train_model
except ModuleNotFoundError:
    from .yffc.training.engine import train_model


def main():
    """
    主函数：所有训练参数在此集中配置。
    修改实验设置时只需修改这一个区域。
    """
    print("=" * 60)
    print("Y-FFC V2 Net 训练程序启动（增强可观测性版）")
    print("=" * 60)

    # ╔══════════════════════════════════════════════════════════════╗
    # ║              ★ 训练参数集中配置区 ★                          ║
    # ║  修改实验配置时，只需修改此区域内的变量                         ║
    # ╚══════════════════════════════════════════════════════════════╝

    # ========== 基础训练参数 ==========
    EPOCHS                = 400       # 训练总轮数
    BATCH_SIZE            = 4         # 384 分辨率推荐 2~4
    BASE_LR               = 2e-4     # 基础学习率（分层 LR 以此为基准）
    WEIGHT_DECAY          = 1e-4     # AdamW 权重衰减
    ACCUMULATION_STEPS    = 4        # 梯度累积步数（等效 batch = BATCH_SIZE × 此值）
    GRAD_CLIP_VALUE       = 0.5      # [FIX] 梯度裁剪阈值（从 1.0 降低到 0.5，更稳定）

    # ========== 图像与数据参数 ==========
    IMAGE_SIZE            = (128, 160)  # ⭐ 输入分辨率 (H, W)，支持任意矩形
    TRAIN_SPLIT_RATIO     = 0.8      # 自动划分时训练集比例（有独立验证集时忽略）
    NUM_WORKERS           = 4        # DataLoader 工作进程数

    # ========== V2 特有开关 ==========
    USE_GN                = True     # BN → GN（小 batch 必备）
    USE_EMA               = True     # 指数移动平均
    EMA_DECAY             = 0.9995   # [P16] EMA 衰减系数（调高增加平滑力度）
    USE_OHEM              = True     # 在线难样本挖掘

    # ========== 数据目录 ==========
    INPUT_DIR             = 'data/inputs'
    TARGET_DIR            = 'data/targets'
    VAL_INPUT_DIR         = None     # 独立验证集输入目录（None 则自动划分）
    VAL_TARGET_DIR        = None     # 独立验证集目标目录

    # ========== 模型保存与检查点 ==========
    SAVE_EVERY_N_EPOCHS   = 20       # 每 N 轮保存检查点（0=不保存）

    # ========== 可视化间隔 ==========
    VIS_EVERY_N_EPOCHS    = 10       # 每 N 轮保存验证样本可视化（0=不保存）

    # ========== 早停策略 ==========
    USE_EARLY_STOPPING    = False
    EARLY_STOPPING_PATIENCE = 30

    # ========== 输出目录（自动带时间戳） ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FOLDER = f'experiments/yffc_v2_{timestamp}'

    # ╔══════════════════════════════════════════════════════════════╗
    # ║              ★ 配置区结束 ★                                  ║
    # ╚══════════════════════════════════════════════════════════════╝

    # 将所有配置打包为字典，统一传递
    cfg = {
        'EPOCHS':                 EPOCHS,
        'BATCH_SIZE':             BATCH_SIZE,
        'BASE_LR':                BASE_LR,
        'WEIGHT_DECAY':           WEIGHT_DECAY,
        'ACCUMULATION_STEPS':     ACCUMULATION_STEPS,
        'GRAD_CLIP_VALUE':        GRAD_CLIP_VALUE,
        'IMAGE_SIZE':             IMAGE_SIZE,
        'TRAIN_SPLIT_RATIO':      TRAIN_SPLIT_RATIO,
        'NUM_WORKERS':            NUM_WORKERS,
        'USE_GN':                 USE_GN,
        'USE_EMA':                USE_EMA,
        'EMA_DECAY':              EMA_DECAY,
        'USE_OHEM':               USE_OHEM,
        'INPUT_DIR':              INPUT_DIR,
        'TARGET_DIR':             TARGET_DIR,
        'VAL_INPUT_DIR':          VAL_INPUT_DIR,
        'VAL_TARGET_DIR':         VAL_TARGET_DIR,
        'SAVE_EVERY_N_EPOCHS':    SAVE_EVERY_N_EPOCHS,
        'VIS_EVERY_N_EPOCHS':     VIS_EVERY_N_EPOCHS,
        'USE_EARLY_STOPPING':     USE_EARLY_STOPPING,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'OUTPUT_FOLDER':          OUTPUT_FOLDER,
    }

    # 打印配置总览
    print(f"\n实验输出目录: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"\n当前训练配置:")
    for k, v in cfg.items():
        print(f"  {k:30s}: {v}")
    print("=" * 60)

    # 执行训练
    history = train_model(cfg)

    print("\n✅ 所有流程完成！")


if __name__ == '__main__':
    # Matplotlib ??????????
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    main()
