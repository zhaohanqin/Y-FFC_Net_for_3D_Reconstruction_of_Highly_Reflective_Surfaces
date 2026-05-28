"""Decoupled copy of the original enhanced training entrypoint.

The root train.py is untouched. This copy keeps the same configuration block
and delegates implementation to yffc.training.engine.
"""

import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __package__:
    from .yffc.training.engine import train_model
else:
    from yffc.training.engine import train_model


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
    EPOCHS                = 50       # 训练总轮数
    BATCH_SIZE            = 3         # 384 分辨率推荐 2~4
    # ★[V3.2 调整] BASE_LR 1e-4 → 2e-4
    # 原因：V3.2 新增 dual-head 与 r 系列损失，新结构从头训练，1e-4 收敛过慢。
    # 配合 V3.1 反向稳定性修复，2e-4 在 dual-head + 三路调制下更平衡。
    BASE_LR               = 2e-4
    WEIGHT_DECAY          = 1e-4     # AdamW 权重衰减
    ACCUMULATION_STEPS    = 3        # 梯度累积步数（等效 batch = BATCH_SIZE × 此值）
    # ★[V3.2 调整] GRAD_CLIP 0.5 → 1.0
    # 原因：V3.1 频域损失反向稳定性已修复（无除法相位项 + sqrt eps 提升），
    # 旧版 0.5 过紧会压制 dual-head 早期梯度信号。1.0 是更标准的设定。
    GRAD_CLIP_VALUE       = 1.0

    # ========== 图像与数据参数 ==========
    IMAGE_SIZE            = (128, 160)  # ⭐ 输入分辨率 (H, W)，支持任意矩形
    TRAIN_SPLIT_RATIO     = 0.8      # 自动划分时训练集比例（有独立验证集时忽略）
    NUM_WORKERS           = 5        # DataLoader 工作进程数

    # ========== V2 特有开关 ==========
    USE_GN                = True     # BN → GN（小 batch 必备）
    USE_EMA               = True     # 指数移动平均
    EMA_DECAY             = 0.9995   # [P16] EMA 衰减系数（调高增加平滑力度）
    USE_OHEM              = True     # 在线难样本挖掘
    USE_AMP               = False    # 稳定版默认关闭 AMP；确认稳定后可改 True
    USE_HIGHLIGHT_AUG     = True     # 默认开启训练集高反光合成增强
    HIGHLIGHT_AUG_P       = 0.5      # 降低合成高反光占比，避免训练分布过度偏向合成模式

    # ========== 数据目录 ==========
    INPUT_DIR             = 'data2/inputs'
    TARGET_DIR            = 'data2/targets'
    VAL_INPUT_DIR         = None     # 独立验证集输入目录（None 则自动划分）
    VAL_TARGET_DIR        = None     # 独立验证集目标目录

    # ========== 模型保存与检查点 ==========
    SAVE_EVERY_N_EPOCHS   = 20       # 每 N 轮保存检查点（0=不保存）

    # ========== 可视化间隔 ==========
    VIS_EVERY_N_EPOCHS    = 10       # 每 N 轮保存验证样本可视化（0=不保存）
    VIS_NUM_SAMPLES       = 2        # 每次保存几张固定验证样本
    VIS_SAMPLE_SEED       = None     # None=每次训练随机抽样；填整数可复现同一批图像

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
        'USE_AMP':                USE_AMP,
        'USE_HIGHLIGHT_AUG':      USE_HIGHLIGHT_AUG,
        'HIGHLIGHT_AUG_P':        HIGHLIGHT_AUG_P,
        'INPUT_DIR':              INPUT_DIR,
        'TARGET_DIR':             TARGET_DIR,
        'VAL_INPUT_DIR':          VAL_INPUT_DIR,
        'VAL_TARGET_DIR':         VAL_TARGET_DIR,
        'SAVE_EVERY_N_EPOCHS':    SAVE_EVERY_N_EPOCHS,
        'VIS_EVERY_N_EPOCHS':     VIS_EVERY_N_EPOCHS,
        'VIS_NUM_SAMPLES':        VIS_NUM_SAMPLES,
        'VIS_SAMPLE_SEED':        VIS_SAMPLE_SEED,
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
