# -*- coding: utf-8 -*-
"""
train_no_dualhead_refine.py — 消融实验 D 训练入口（去除双头输出 + 残差精修头）
=============================================================================

薄包装方式复用 train.py 全部流程，相对 train.py 的修改清单：
  - [新增] set_global_seed()：全局随机种子 42
  - [修改] 模型类：XFFC → X_FFC_no_dualhead_refine.YFFC_NoDualRefine
  - [修改] OUTPUT_FOLDER → experiments/ablation_no_dualhead_refine_<时间戳>
  - [修改] VIS_SAMPLE_SEED：None → 42
  - 损失函数、权重调度、优化器、LR 调度、数据管线全部零修改

损失侧说明（为何不需要改权重调度器）：
  - 模型不返回 "r_direct" → train.py 中 pred_r_direct = out.get('r_direct')
    为 None → criterion 内 l_r_dual_cons 自动为 0。这是结构前提消失带来的
    必然结果，属于消融本身，不是混杂变量；
  - l_r_focal_hl / l_r_smooth 直接监督最终 r 本身，与 r 的形成方式无关，
    原样保留 → 四组实验损失口径一致。

兼容性说明（自动成立，无需手动处理）：
  - train.py 的 _tta_forward 用 hasattr(model, 'refinement_head') 与
    r_direct 是否存在做了双重保护（且训练中 TTA 本就关闭）；
  - 可视化函数用 estimations.get('s') 兼容，本模型仍输出 s，正常显示；
  - 优化器分组中 refinement_head 参数自然不存在，分组规则不变。

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
from X_FFC_no_dualhead_refine import YFFC_NoDualRefine

# ════════════════════════════════════════════════════════════════════
# ★ 核心修改：替换 train.train_model 内部实例化的模型类
# ════════════════════════════════════════════════════════════════════
base_train.XFFC = YFFC_NoDualRefine

GLOBAL_SEED = 42


def set_global_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("=" * 60)
    print("消融实验 D：去除双头输出 + 残差精修头 (输出端 → 纯减法链)")
    print("模型类:", base_train.XFFC.__name__)
    print("=" * 60)
    set_global_seed()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FOLDER = f'experiments/ablation_no_dualhead_refine_{timestamp}'

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
        'VIS_SAMPLE_SEED':        42,
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
    print("\n✅ 消融实验 D 训练完成！")


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                       'WenQuanYi Micro Hei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    main()
