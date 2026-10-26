# decoupled_XFFC 结构说明

`decoupled_XFFC` 是根目录 `X_FFC_unified.py`、`train.py` 和 `process_images.py` 的拆分版副本。目标是把单文件里的模型、损失、数据集、训练和推理逻辑拆成更清晰的包结构，方便阅读、维护和后续让 AI 精确定位代码。内部 Python 包名和主网络类名都是 `XFFC`。

当前拆分版已按根目录主实现对齐：

- 模型结构与 `X_FFC_unified.py` 对齐：`XFFC`、ResNet50 + Swin-Tiny 双分支编码器、AdaptiveSkipFusion、FFC 模块、高频增强、梯度分支、resize-conv 主解码器、残差精修头等。
- 训练主链路与 `train.py` 对齐：`EnhancedCompositeLoss`、跨尺度一致性辅助损失、OHEM、合成高光增强、BN->GN、EMA、分层学习率、warmup+cosine scheduler、自动训练/验证划分、固定验证可视化样本。
- 图像尺寸语义统一为 `(H, W)`；传给 PIL resize 时内部转换为 `(W, H)`。

## 顶层入口

- `train.py`：推荐的增强训练入口，配置块风格与根目录 `train.py` 保持一致，并转发到 `XFFC.training.engine.train_model`。
- `X_FFC_unified.py`：兼容入口，转发到拆分后的 `XFFC.cli`，用于保留类似旧单文件的 `test/train` 子命令。`train` 子命令会转成增强训练配置并调用 `train_model`。
- `process_images.py`：推理入口，配置区与根目录 `process_images.py` 一致，实现放在 `XFFC.inference`。
- `README.md`：当前文件，说明拆分结构和使用方式。

## 包结构

- `XFFC/models/`：模型结构。
- `XFFC/models/architecture/`：`Encoder`、`GradientDecoder`、`MainDecoder`、`EnhancementLayers`、`XFFC` 主网络。
- `XFFC/models/attention/`：注意力模块，包括 cross-modal fusion 和 specular-aware attention。
- `XFFC/models/branches/`：Swin-Tiny 单通道多尺度分支与 `AdaptiveSkipFusion`。
- `XFFC/models/blocks/`：卷积、上采样、多尺度卷积等通用块。
- `XFFC/models/ffc/`：Fast Fourier Convolution 相关模块。
- `XFFC/models/heads/`：高频增强和残差精修头。
- `XFFC/losses/`：SSIM/MS-SSIM、Charbonnier、感知损失、频域损失、specular focal/dice、梯度一致性，以及包含 `s_lik`、`spec_peak`、`r_focal_hl`、`r_dual_cons`、`r_smooth` 的增强复合损失。
- `XFFC/data/`：数据集和配对增强。
- `XFFC/training/`：训练循环、优化器、EMA、scheduler、指标记录、可视化、报告和增强训练编排。
- `XFFC/inference/`：权重加载、滑窗推理、TTA、结果保存和批量处理。
- `XFFC/cli.py`：兼容 `X_FFC_unified.py train/test` 的命令行入口。

兼容导出：

- `XFFC/data/` 和 `XFFC/losses/` 是包，由各自的 `__init__.py` 导出数据集、增强和损失函数。不要再在同级放置 `data.py` 或 `losses.py`，否则会挡住这两个包。
- `XFFC/models/network.py` 保留模型组件的旧导入路径。
- `XFFC/__init__.py` 汇总常用类和函数，方便 `from XFFC import XFFC` 这类导入。

## 推荐训练方式

在仓库根目录运行：

```powershell
cd decoupled_XFFC
python train.py
```

`train.py` 内部集中配置训练参数。当前默认值已与当前拆分实现对齐，包括：

- `EPOCHS = 300`
- `BATCH_SIZE = 3`
- `BASE_LR = 2e-4`
- `ACCUMULATION_STEPS = 3`
- `GRAD_CLIP_VALUE = 1.0`
- `IMAGE_SIZE = (128, 160)`
- `INPUT_DIR = 'data2/inputs'`
- `TARGET_DIR = 'data2/targets'`
- `USE_GN = True`
- `USE_EMA = True`（`EMA_DECAY = 0.9995`）
- `USE_OHEM = True`
- `USE_AMP = False`
- `USE_HIGHLIGHT_AUG = True`，并使用 `HighlightAugmentation(p=0.5)`
- `SAVE_EVERY_N_EPOCHS = 10`

## 推理

在本目录修改 `process_images.py` 的配置区后运行：

```powershell
cd decoupled_XFFC
python process_images.py
```

推理实现位于 `XFFC/inference/`：`checkpoint.py` 加载权重，`sliding_window.py` 做滑窗融合，`postprocess.py` 保存和可视化，`pipeline.py` 负责单张、批处理和推理版 TTA。

## X-FFC-Swin 结构要点

当前拆分版使用 X-FFC-Swin 三阶段结构：

1. ResNet50 分支提取 `x_bb_1~x_bb_4`，作为局部纹理和 decoder skip connection。
2. Swin-Tiny 分支直接处理单通道 patch，不强制 resize 到 `224x224`，输出 `swin_4/swin_8/swin_16`。
3. `swin_4` 和 `swin_8` 通过 `AdaptiveSkipFusion` 分别增强 `x_bb_2` 和 `x_bb_3`。
4. `swin_16` 投影到 512 通道后，与 `x_bb_4` 通过 `CrossModalFusion` 做 H/16 瓶颈融合。
5. 瓶颈融合结果依次经过 `cba_after_backbone -> FFC bottleneck -> HighFrequencyEnhancement`，再送入梯度解码器和主解码器。
6. 主解码器的 `TransposedConvBlock` 保留类名和外部接口，但内部已改为 `bilinear upsample + 3x3/5x5/7x7 conv` 的 resize-conv，降低棋盘格伪影风险。
7. 编码器在 H/16 cross-modal 融合后通过 `CrossScaleTopDownRefinement` 做 H/16 -> H/8 -> H/4 的受限残差回流，把深层上下文反馈到浅层 skip。
8. `XFFC.forward()` 会额外返回训练辅助标量 `cs`，用于 H/4、H/8、H/16 融合特征的跨尺度一致性约束；推理主输出仍使用 `r/g/b`。

训练与推理建议：

- 推荐训练 patch：`320x256`、`384x384`、`512x512`，尽量让 H/W 能被 32 整除。
- 不建议直接使用 `2448x2048` 整图训练。
- 大图推理建议 sliding-window patch 推理，patch 尺寸与训练保持一致，overlap 可从 `64` 开始。
- 如果使用训练好的 Swin checkpoint 推理，加载模型时应使用 `swin_pretrained=False`，避免离线环境重复下载 ImageNet 权重。

## 兼容 CLI

测试模型前向：

```powershell
cd decoupled_XFFC
python X_FFC_unified.py test
```

使用 CLI 训练：

```powershell
cd decoupled_XFFC
python X_FFC_unified.py train --input_dir data2/inputs --target_dir data2/targets --use_gn --use_ema
```

CLI `train` 子命令默认会启用合成高光增强和 OHEM；如果需要关闭：

```powershell
python X_FFC_unified.py train --input_dir data2/inputs --target_dir data2/targets --no_augmentation --no_ohem
```

## 训练链路说明

增强训练入口 `XFFC/training/engine.py` 的流程：

1. 构建 `XFFC` 模型。
2. 按配置执行 BN->GN。
3. 使用 `ImprovedFringeDataset` 加载输入/目标图像，训练集使用 `EnhancedPairedTransform`。
4. `HighlightAugmentation` 会随机叠加高斯 blob 高反光，并同步更新 `gt_specular`。
5. 没有独立验证集时，使用固定 seed 自动划分训练/验证，验证集不做增强。
6. 使用 `EnhancedCompositeLoss(use_ohem=USE_OHEM)`，训练循环会把模型输出里的 `cs` 传入复合损失。
7. 使用 `ImprovedWeightScheduler` 动态更新损失权重：当前策略以常量权重为主，仅 `grad_spec` 与 `r_smooth` 在后期退火，`lambda_s_cons` 固定为 `0.0`。
8. 使用 `build_optimizer_v2` 分层学习率：`encoder.swin_branch.swin` 使用低学习率（`0.1x base_lr`），`fuse_h4/fuse_h8/swin16_proj/refinement_head/top_down_refine` 等新模块使用 `base_lr`。
9. scheduler 只在真实 `optimizer.step()` 后推进，避免梯度累积时学习率走快。
10. 训练日志会打印 `fuse_h4.beta`、`fuse_h8.beta` 和 gate mean，用于观察 Swin skip 是否被模型使用。
11. 验证和可视化时可临时切换 EMA 权重。
12. best checkpoint 在启用 EMA 验证时保存 EMA 合并权重为 `model_state_dict`，同时保留 `raw_model_state_dict`、`ema_shadow` 和 `image_size`，便于恢复训练和推理尺寸对齐。

## 数据命名匹配

数据集会按以下顺序寻找目标图：

- 输入文件同名目标图。
- `input_xxx` -> `target_xxx`。
- `inputxxx` -> `targetxxx`。
- 从文件名提取数字后尝试 `target_{digits}{ext}`。

如果无法匹配，会抛出 `FileNotFoundError`。

## 当前损失设计要点

当前 `EnhancedCompositeLoss` 相比旧版 README 中的描述又补充了几项关键监督：

- `l_s_lik`：对高光概率图 `pred_s_lik` 做二值高光监督，使用带 `pos_weight` 的 BCE + Dice，提升高反光区域的可见性。
- `l_spec_peak`：对高光区域的 `est_b` 欠预测进行单侧惩罚。
- `l_r_focal_hl`：在高光区域直接约束 `pred_r` 与目标漫反射的一致性。
- `l_r_dual_cons`：约束 `pred_r_direct` 与最终 `pred_r` 的一致性。
- `l_r_smooth`：对高光区域的漫反射结果加入局部平滑先验。
- `l_grad_spec`：当前使用 `gt_specular` 的 Sobel 结果作为稳定监督目标，避免动态目标带来的中期回升。

## 注意

- `comporison_model/` 是对比实验目录，不属于此拆分包。
- 训练输出、权重、曲线和中间结果仍应放在 `experiments/`、`checkpoints/`、`outputs/` 等产物目录，不应作为源码提交。
- 如果修改根目录 `X_FFC_unified.py` 或 `train.py` 的核心逻辑，需要同步检查本目录对应模块。
