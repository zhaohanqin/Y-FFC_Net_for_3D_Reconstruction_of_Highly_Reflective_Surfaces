# -*- coding: utf-8 -*-
"""
train_no_frequency.py — 消融实验 A 训练入口（去除频域模块）
==========================================================

设计原则：不复制 train.py 的任何训练逻辑，采用"薄包装"方式：
  1. `import train as base_train` 复用其全部训练 / 验证 / 日志 / 保存流程；
  2. 把 base_train.XFFC 替换为 YFFC_NoFrequency
     （train_model 内部 `model = XFFC()` 按模块全局名解析，替换即生效）；
  3. 配置字典与 baseline train.py 的 main() 逐项相同，
     仅 OUTPUT_FOLDER 与 VIS_SAMPLE_SEED 不同；
  4. 设置全局随机种子 42。
     ★ 公平性提示：请给 baseline train.py 也加同样的种子后重跑一次作为
       对照组，否则 baseline 与消融组之间会多出随机性这一混杂变量。

相对 train.py 的修改清单：
  - [新增] set_global_seed()：random / numpy / torch / cuda 全局种子
  - [修改] 模型类：XFFC → X_FFC_no_frequency.YFFC_NoFrequency
  - [修改] OUTPUT_FOLDER → experiments/ablation_no_frequency_<时间戳>
  - [修改] VIS_SAMPLE_SEED：None → 42（固定可视化抽样，跨实验对比同一批图）
  - 其余（数据、增强、损失、权重调度、优化器、LR 调度、指标、保存）零修改

数据输入格式：与 baseline 完全一致（inp, tgt, gt_spec, gt_grad），无需调整。
"""

import os
import random
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import train as base_train
from X_FFC_no_frequency import YFFC_NoFrequency

# ════════════════════════════════════════════════════════════════════
# ★ 核心修改：替换 train.train_model 内部实例化的模型类
# ════════════════════════════════════════════════════════════════════
base_train.XFFC = YFFC_NoFrequency

GLOBAL_SEED = 42


def set_global_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("=" * 60)
    print("消融实验 A：去除频域模块 (SpectralTransform → 纯空间卷积)")
    print("模型类:", base_train.XFFC.__name__)
    print("=" * 60)
    set_global_seed()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FOLDER = f'experiments/ablation_no_frequency_{timestamp}'

    # ★ 除 OUTPUT_FOLDER / VIS_SAMPLE_SEED 外，与 baseline train.py 逐项相同
    cfg = {
        'EPOCHS':                 300,
        'BATCH_SIZE':             3,
        'BASE_LR':                2e-4,
        'WEIGHT_DECAY':           1e-4,
        'ACCUMULATION_STEPS':     3,
        'GRAD_CLIP_VALUE':        1.0,
        'IMAGE_SIZE':             (128, 160),
        'TRAIN_SPLIT_RATIO':      0.8,
        'NUM_WORKERS':            5,
        'USE_GN':                 True,
        'USE_EMA':                True,
        'EMA_DECAY':              0.9995,
        'USE_OHEM':               True,
        'USE_AMP':                False,
        'USE_HIGHLIGHT_AUG':      True,
        'HIGHLIGHT_AUG_P':        0.5,
        'INPUT_DIR':              'data2/inputs',
        'TARGET_DIR':             'data2/targets',
        'VAL_INPUT_DIR':          None,
        'VAL_TARGET_DIR':         None,
        'SAVE_EVERY_N_EPOCHS':    10,
        'VIS_EVERY_N_EPOCHS':     10,
        'VIS_NUM_SAMPLES':        2,
        'VIS_SAMPLE_SEED':        42,   # 固定抽样，便于跨实验逐图对比
        'USE_EARLY_STOPPING':     False,
        'EARLY_STOPPING_PATIENCE': 30,
        'OUTPUT_FOLDER':          OUTPUT_FOLDER,
    }

    print(f"\n实验输出目录: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"全局随机种子: {GLOBAL_SEED}")
    print("\n当前训练配置:")
    for k, v in cfg.items():
        print(f"  {k:30s}: {v}")
    print("=" * 60)

    base_train.train_model(cfg)
    print("\n✅ 消融实验 A 训练完成！")


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                       'WenQuanYi Micro Hei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    main()
