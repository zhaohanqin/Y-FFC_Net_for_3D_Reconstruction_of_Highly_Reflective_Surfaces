# -*- coding: utf-8 -*-
"""
train_no_gradient.py — 消融实验 B 训练入口（去除梯度辅助分支）
=============================================================

薄包装方式复用 train.py 全部流程，相对 train.py 的修改清单：
  - [新增] set_global_seed()：全局随机种子 42
  - [修改] 模型类：XFFC → X_FFC_no_gradient.YFFC_NoGradient
  - [修改] 权重调度器：ImprovedWeightScheduler → NoGradientWeightScheduler
      把以下 5 个权重置 0（分支不存在则其监督必然移除，属于消融的一部分）：
        lambda_grad / lambda_grad_edge / lambda_grad_spec /
        lambda_s_lik / lambda_s_cons
      其余权重数值与退火逻辑（grad_spec / r_smooth 的 60% 后退火）原样继承。
  - [修改] OUTPUT_FOLDER → experiments/ablation_no_gradient_<时间戳>
  - [修改] VIS_SAMPLE_SEED：None → 42

数据输入格式：与 baseline 完全一致（inp, tgt, gt_spec, gt_grad）。
  dataloader 仍照常生成 gt_grad / gt_spec —— 模型只是不再使用其辅助监督
  （对应权重为 0），保持数据管线零差异以保证公平。
  gt_spec 仍被 spec/focal/dice/gc/spec_peak 等镜面侧损失使用（这些属于
  主任务监督，与梯度分支无关，保持不变）。

日志说明：模型返回的 "g" 为全零占位图，因此日志中的 grad / grad_edge
子损失项会显示非零数值，但其权重为 0，不参与优化，可忽略。
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
from X_FFC_no_gradient import YFFC_NoGradient

# ════════════════════════════════════════════════════════════════════
# ★ 核心修改 1：替换模型类
# ════════════════════════════════════════════════════════════════════
base_train.XFFC = YFFC_NoGradient


# ════════════════════════════════════════════════════════════════════
# ★ 核心修改 2：梯度分支相关损失权重置 0，其余继承 baseline 调度
# ════════════════════════════════════════════════════════════════════
_BaseScheduler = base_train.ImprovedWeightScheduler

_ZEROED_KEYS = (
    'lambda_grad',        # 梯度图 SSIM 监督（g 头已移除）
    'lambda_grad_edge',   # 梯度边缘加权监督（g 头已移除）
    'lambda_grad_spec',   # 梯度-镜面边界约束（依赖 g 头）
    'lambda_s_lik',       # 高反光概率头监督（s 头已移除）
    'lambda_s_cons',      # s 与 b 一致性（baseline 中本已为 0，显式保留）
)


class NoGradientWeightScheduler(_BaseScheduler):
    """继承 baseline 调度逻辑，仅将梯度分支相关权重置 0。"""

    def get_weights(self, ep):
        w = super().get_weights(ep)
        for k in _ZEROED_KEYS:
            w[k] = 0.0
        return w


base_train.ImprovedWeightScheduler = NoGradientWeightScheduler

GLOBAL_SEED = 42


def set_global_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("=" * 60)
    print("消融实验 B：去除梯度辅助分支 (GradientDecoder + EnhancementLayers)")
    print("模型类:", base_train.XFFC.__name__)
    print("置零权重:", ", ".join(_ZEROED_KEYS))
    print("=" * 60)
    set_global_seed()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FOLDER = f'experiments/ablation_no_gradient_{timestamp}'

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
    print("\n✅ 消融实验 B 训练完成！")


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                       'WenQuanYi Micro Hei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    main()
