# X-FFC / YFFC 消融实验套件总览（README_ablation）

三组消融均采用经典星型对照：**完整模型 − 单一模块**，每组只改一个自变量。

| 实验 | 消融对象 | 检验假设 | 模型文件 | 训练文件 |
|---|---|---|---|---|
| A | FFC 频域核心 SpectralTransform → 等规模纯空间卷积 | 频域全局建模优于局部卷积 | `Y_FFC_no_frequency.py` | `train_no_frequency.py` |
| B | 梯度辅助臂（GradientDecoder + EnhancementLayers + g/s 先验门控） | 显式结构/高光先验优于隐式学习（X 型解码） | `Y_FFC_no_gradient.py` | `train_no_gradient.py` |
| C | Swin 分支 + AdaptiveSkipFusion + CrossModalFusion | 双编码跨模态融合优于单 CNN 编码器（X 型编码） | `Y_FFC_single_encoder.py` | `train_single_encoder.py` |
| D | 双头输出 (r_direct + α 融合) + ResidualRefinementHead → 纯减法链 r=clamp(x−b) | 双头融合与残差精修补救减法链脆弱性 | `Y_FFC_no_dualhead_refine.py` | `train_no_dualhead_refine.py` |

## 使用方法

1. 把 7 个 `.py` 文件与 `Y_FFC_unified.py`、`train.py` 放在同一目录；
2. 各实验单独起进程运行：`python train_no_frequency.py` 等；
3. **重要**：给 baseline `train.py` 的 `main()` 开头也加上同样的种子设置
   （random/numpy/torch/cuda 全部 seed=42，且 `VIS_SAMPLE_SEED=42`）后重跑一次，
   作为公平对照组；
4. 每个实验输出到独立目录 `experiments/ablation_*_<时间戳>/`，
   日志、曲线、子损失、可视化、模型保存格式与 baseline 完全一致，可直接对比。

## 公平性保证清单

- 数据与划分：同一 `data2/`，划分种子 42（train.py 内置），全局种子 42；
- 增强：同一 EnhancedPairedTransform + HighlightAugmentation(p=0.5)；
- 训练：300 epochs，batch 3 × 累积 3，AdamW(2e-4, wd 1e-4)，同一分层 lr 规则与 warmup+余弦调度，clip 1.0，GN + EMA(0.9995)，AMP 关闭；
- 损失：同一 EnhancedCompositeLoss 与权重调度；仅实验 B 把被移除分支自身的 5 个监督权重置 0（这属于消融本身）；
- 指标与选模：同一 SSIM/MAE/MSE/PSNR 与综合评分 0.7·SSIM + 0.3·1/(1+Loss)，验证均不开 TTA。

## 结果汇报建议

对比四组（baseline + A/B/C）的：验证 SSIM/PSNR/MAE、`spec`/`focal`/`dice` 子损失（高光质量）、`freq` 子损失（频谱一致性）、固定可视化样本（VIS_SAMPLE_SEED=42 保证四组看同一批图）的高光区残差图、以及参数量/显存/单 epoch 时间。条件允许时每组跑 2~3 个种子取均值±方差，结论更稳。

## 注意事项

- 实验 A 的模型文件导入时会 monkey-patch `Y_FFC_unified.SpectralTransform`，请勿在同一进程内混用 baseline 与该消融模型；
- 实验 B 日志中 grad/grad_edge 子损失显示非零数值属正常（占位零图所致），其权重为 0；
- 实验 D 输出字典不含 `r_direct`，criterion 中 `l_r_dual_cons` 自动为 0（结构前提消失，属于消融本身），训练文件无需改权重调度；
- 推理脚本 `process_images.py` 如需评测消融模型，把其中 `YFFC` 的导入替换为对应消融类即可（A 与 C 的输出字典与 baseline 完全一致；B 不含 `s` 且 `g` 为占位零图）。
