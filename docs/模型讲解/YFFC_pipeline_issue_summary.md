# Y-FFC 当前流程问题总结与改动标注

## 背景与目标
本文档用于跟踪 `X_FFC_unified.py`、`train.py`、`process_images.py` 中影响 SSIM 与漫反射输出质量的问题、已落地改动，以及后续版本仍需迭代的位置。

状态说明：
- `已实现`：已经在当前工作区完成代码改动。
- `已缓解`：已经做低风险修复，但问题本身还可能需要训练结果验证。
- `部分实现`：已做兼容性较好的小改动，未做高风险结构重构。
- `暂缓/未改动`：本轮刻意隔离，后续单独做 ablation 或版本迭代。

## 本轮已修改文件
- `X_FFC_unified.py`
- `train.py`
- `process_images.py`
- `X-FFC_V3_模型完整技术文档.md`
- `decoupled_yffc/README.md`
- `decoupled_yffc/train.py`
- `decoupled_yffc/yffc/cli.py`
- `decoupled_yffc/yffc/losses/basic.py`
- `decoupled_yffc/yffc/losses/composite.py`
- `decoupled_yffc/yffc/models/architecture/encoder.py`
- `decoupled_yffc/yffc/models/architecture/model.py`
- `decoupled_yffc/yffc/models/attention/modules.py`
- `decoupled_yffc/yffc/models/blocks/convolution.py`
- `decoupled_yffc/yffc/models/branches/swin.py`
- `decoupled_yffc/yffc/training/engine.py`
- `decoupled_yffc/yffc/training/loops.py`
- `decoupled_yffc/yffc/training/metrics.py`
- `decoupled_yffc/yffc/training/optimization.py`
- `decoupled_yffc/yffc/training/reporting.py`
- `decoupled_yffc/yffc/training/simple_loop.py`

## 已处理与已缓解问题

### 高优先级

1. `[已缓解]` 训练/推理分辨率与上下文机制存在不一致
- 原问题：训练固定 resize 到 `IMAGE_SIZE`，推理通常保留原图并滑窗切 patch。
- 已改动：训练 checkpoint 保存 `image_size`；推理加载 checkpoint 后优先使用 checkpoint 中的训练尺寸作为滑窗 patch，并在配置尺寸不一致时提示。
- 改动位置：`train.py`、`X_FFC_unified.py` checkpoint 保存；`process_images.py` 的 `resolve_patch_size_from_checkpoint()`。
- 仍需关注：这不是“全局上下文训练”的结构性解决，只是避免推理 patch 尺寸配置错误。若后续要彻底解决，需要多尺度/大上下文训练或全图上下文分支。

2. `[已实现]` EMA 验证口径与 best 模型保存口径不一致
- 原问题：验证使用 EMA 权重，但 best checkpoint 保存非 EMA `model.state_dict()`。
- 已改动：best checkpoint 的 `model_state_dict` 保存 EMA 合并后的权重；同时保留 `raw_model_state_dict`、`ema` / `ema_shadow` 和 `uses_ema_weights`。
- 改动位置：`train.py`、`X_FFC_unified.py`。

3. `[已实现]` 梯度累积与 scheduler 步进频率不匹配
- 原问题：scheduler 按 `len(train_loader)` 构建，但实际在 optimizer step 后才 `scheduler.step()`。
- 已改动：`iters_per_epoch` 和 warmup step 改为基于 `ceil(len(loader) / accumulation_steps)`。
- 改动位置：`train.py`、`X_FFC_unified.py`。

4. `[已实现]` `gt_specular` 在不同数据路径下定义口径不一致
- 原问题：增强路径 clamp，默认路径直接 `input-target`。
- 已改动：默认路径统一为 `(input - target).clamp(0, 1)`。
- 改动位置：`X_FFC_unified.py` 的 `FringeDataset`，`train.py` 的 `ImprovedFringeDataset` 已保持 clamp。

### 中优先级

5. `[已缓解]` 漫反射主恢复依赖硬减法，易放大 `est_b` 误差
- 原问题：`coarse = clamp(x - est_b)` 后精修空间较小。
- 已改动：保留主链路不变，将 `ResidualRefinementHead` 默认残差尺度从 `0.05` 提到 `0.08`。
- 改动位置：`X_FFC_unified.py`。
- 仍需关注：硬减法链路没有被替换。若训练后仍存在过截断或细节漂移，需要单独设计 soft subtraction / confidence residual 版本。

6. `[已缓解]` 小 batch 下 BN 统计稳定性风险
- 原状态：训练入口已有 `USE_GN=True` 并调用 `replace_bn_with_gn(model, 8)`。
- 本轮处理：保留该默认策略，没有额外改 BN 结构。
- 仍需关注：若用户关闭 `USE_GN`，小 batch 仍有 BN 风险。

7. `[已实现]` 高光合成增强默认始终开启 `p=1.0`
- 原问题：训练集每个样本都叠加合成高光，可能偏向合成分布。
- 已改动：`HIGHLIGHT_AUG_P` 默认从 `1.0` 降为 `0.5`；统一版 CLI 训练路径中的高光增强默认也改为 `p=0.5`。
- 改动位置：`train.py`、`X_FFC_unified.py`。

8. `[已实现]` 推理输出 8-bit 量化导致细节损失
- 原问题：只保存 8-bit PNG。
- 已改动：`save_result()` 支持 `bit_depth=8/16`；`process_images.py` 增加 `SAVE_16BIT_OUTPUT` 开关，可额外保存 16-bit PNG。
- 改动位置：`process_images.py`。

### 低优先级中已处理/部分处理项

9. `[已实现]` 转置卷积上采样存在棋盘纹理风险
- 原问题：主解码器多处 `TransposedConvBlock` 使用转置卷积上采样，存在棋盘纹理风险。
- 已改动：保留 `TransposedConvBlock` 类名和外部调用接口，但内部改为三路 `Upsample(bilinear) + Conv2d` 的 resize-conv 上采样。
- 改动位置：`X_FFC_unified.py`、`decoupled_yffc/yffc/models/blocks/convolution.py`。
- 兼容策略：旧 checkpoint 中转置卷积权重与新 resize-conv 权重形状不一致；`process_images.py` 已增加 flexible load，strict 失败时会跳过不兼容权重并加载其余可兼容权重。新训练不受影响。

10. `[部分实现]` 镜面注意力包含亮度先验，可能误判亮漫反射区域
- 原问题：`SpecularAwareAttention` 主要使用原图亮度作为阈值先验。
- 已改动：加入局部对比度先验 `spec_prior = 0.75 * brightness + 0.25 * local_contrast`，降低纯亮度误判风险。
- 改动位置：`X_FFC_unified.py` 的 `SpecularAwareAttention.forward()`。
- 仍需关注：这仍是轻量启发式。若后续有真实 specular mask 或相位残差特征，可以替换成更可靠的 confidence-aware attention。

11. `[已实现]` Swin padding 与推理边界行为存在域偏差
- 原问题：Swin 分支 pad 到 32 倍数时使用常数 0，推理滑窗边界用 reflect/replicate。
- 已改动：Swin padding 改为优先 `reflect`，不满足 reflect 条件时退回 `replicate`。
- 改动位置：`X_FFC_unified.py` 的 `SwinEncoderBranch._pad_to_multiple()`。

12. `[部分实现]` Swin 与 ResNet 多阶段融合还可以更统一
- 原问题：H/4、H/8 单向注入，H/16 双向交叉注意力，范式不完全统一。
- 已改动：保留现有结构，降低破坏性；对浅层融合使用小正数 beta 初始化，并在 H/16 cross-attn 前增加 modality norm。
- 改动位置：`X_FFC_unified.py` 的 `AdaptiveSkipFusion`、`Encoder`、`CrossModalFusion`。
- 仍需关注：没有把 H/4、H/8 改成双向交互，也没有统一成同一种融合模块。后续应作为结构实验单独对比。

13. `[部分实现]` H/4、H/8 融合偏保守，可能低估 Swin 上下文
- 原问题：`beta` 初始为 0，Swin 信息注入可能过慢。
- 已改动：`AdaptiveSkipFusion` 默认 `init_beta=0.03`；Encoder 中 H/4 使用 `0.03`，H/8 使用 `0.05`。
- 改动位置：`X_FFC_unified.py`。
- 仍需关注：没有加入分阶段 schedule 或 confidence-aware gating。

14. `[已实现]` H/16 双向交叉注意力前置对齐与归一化不足
- 原问题：CNN/Transformer 特征统计差异较大。
- 已改动：在 `CrossModalFusion.forward()` 中对 `cnn_feat` 和 `context_feat` 先做无参数 `group_norm` 风格 modality norm，再进入 cross-attn。
- 改动位置：`X_FFC_unified.py`。

15. `[已实现]` 当前没有显式跨尺度一致性约束
- 原问题：H/4、H/8、H/16 各自独立融合，没有显式约束它们对同一结构区域给出一致上下文。
- 已改动：`YFFC.forward()` 计算 H/4、H/8、H/16 融合特征的归一化响应图一致性标量 `cs`；`EnhancedCompositeLoss` 增加 `lambda_cs` 和 `cs` 子损失；训练循环将 `out.get('cs')` 传入损失函数。
- 已改动：新增 `CrossScaleTopDownRefinement`，在 H/16 cross-modal 融合后执行 H/16 -> H/8 -> H/4 的受限残差回流，直接把深层上下文反馈到浅层 skip，同时用 `gamma.clamp(0, 0.2)` 限制结构实验的初始扰动。
- 改动位置：`X_FFC_unified.py`、`train.py`、`decoupled_yffc/yffc/models/architecture/encoder.py`、`decoupled_yffc/yffc/models/architecture/model.py`、`decoupled_yffc/yffc/losses/composite.py`、`decoupled_yffc/yffc/training/loops.py`、`decoupled_yffc/yffc/training/simple_loop.py`。
- 推理说明：推理输出字典仍兼容 `r/g/b` 主输出，`cs` 仅作为训练辅助标量存在，不需要保存为图像。

16. `[已实现]` 训练监控的 SSIM 口径与实际优化目标不一致
- 原问题：优化 mixed SSIM，但日志只用 single SSIM。
- 已改动：`EnhancedCompositeLoss` 返回 `ssim_single_loss`、`msssim_loss`、`ssim_mixed_loss`；训练日志额外显示 `SSIM mixed`。
- 改动位置：`X_FFC_unified.py`、`train.py`。

17. `[已实现]` `refinement_head` 学习率分组偏保守
- 原问题：`refinement_head` 可能落入 0.5x LR 的 `other_p`。
- 已改动：`refinement_head` 纳入 1.0x base LR 的新模块组。
- 改动位置：`train.py`、`X_FFC_unified.py`。

18. `[已实现]` 感知损失输入域与 VGG 预训练域未严格对齐
- 原问题：灰度 repeat 后直接送 VGG，未做 ImageNet mean/std；预训练加载失败时会退化随机 VGG。
- 已改动：加入 ImageNet mean/std 标准化；预训练 VGG 加载失败时打印提示并自动关闭 perceptual 项。
- 改动位置：`X_FFC_unified.py`。

19. `[已实现]` `gt_specular` 默认路径仍为未截断差分
- 说明：与问题 4 重复。
- 已改动：默认路径统一 clamp。
- 改动位置：`X_FFC_unified.py`。

20. `[已实现]` best 权重保存为非 EMA 快照
- 说明：与问题 2 重复。
- 已改动：best checkpoint 保存 EMA 合并权重，同时保留 raw 权重。
- 改动位置：`train.py`、`X_FFC_unified.py`。

21. `[已实现]` 学习率日志仅显示首个参数组
- 原问题：日志只打印 `optimizer.param_groups[0]['lr']`。
- 已改动：新增 `format_lr_groups()`，epoch 日志输出各命名参数组 LR。
- 改动位置：`train.py`。

22. `[已实现]` Warmup/Cosine 步进与梯度累积步频存在偏差
- 说明：与问题 3 重复。
- 已改动：scheduler 总 step 与真实 optimizer step 对齐。
- 改动位置：`train.py`、`X_FFC_unified.py`。

## 暂缓/未改动隔离区

原先隔离的 9、15 已在本轮落实。当前仍建议暂缓的是更进一步的结构实验，例如 H/4/H/8 双向 cross-attention、beta schedule 和 confidence-aware gating，这些应在当前新版本重新训练后再做 ablation。

### 12 的剩余结构实验
- 当前只做了小正数 beta 和 H/16 modality norm，没有把 H/4、H/8 改成双向交互。
- 后续若要继续改，应与问题 15 一起做结构实验，并保留当前版本作为稳定 baseline。

### 13 的剩余训练策略实验
- 当前只做了分尺度 beta 初始化，没有实现 beta schedule / confidence-aware gating。
- 后续可以尝试：
  - 按 epoch 对 beta 上限做 warmup。
  - 用 specular confidence 或 edge confidence 调制 Swin 注入强度。

## 当前验证记录
- `python -m py_compile train.py X_FFC_unified.py process_images.py decoupled_yffc\train.py decoupled_yffc\X_FFC_unified.py ...` 已通过，覆盖根目录主链路与 decoupled 训练/模型/损失文件。
- `git diff --check` 已通过；仅有 Windows 换行符提示。
- 轻量 smoke test 已覆盖：
  - 根目录与 decoupled `YFFC(swin_pretrained=False)` forward。
  - 根目录与 decoupled `TransposedConvBlock` 不再包含 `nn.ConvTranspose2d`，输出尺寸正常。
  - 根目录与 decoupled `encoder.top_down_refine` 已存在，且其参数位于 `new_modules` 的 1.0x base LR 参数组。
  - `YFFC.forward()` 输出 `r/g/b/cs`，`r/g/b` 尺寸和值域正常，`cs` 为有限标量。
  - `EnhancedCompositeLoss` 新增 SSIM key 与 `cs` key。
  - optimizer 中 `refinement_head` 的 1.0x LR 分组。
  - scheduler 与 accumulation 的 step 对齐。
  - checkpoint 训练尺寸接管推理 patch。
  - `process_images.py` flexible load 可跳过旧 checkpoint 中形状不兼容的权重。
  - Swin reflect/replicate padding。
  - 8-bit / 16-bit PNG 保存。
  - `CrossModalFusion` 与 `SpecularAwareAttention` 结构 smoke。

## 后续版本建议
1. 先用当前版本重新训练或做小样本训练，确认 SSIM、PSNR、视觉细节是否稳定提升。
2. 若输出仍有局部周期伪影，基于当前 resize-conv 版本继续检查是否来自 FFC/频域边界或滑窗融合。
3. 若跨 patch 或高光边界仍有结构漂移，再在当前 `cs` loss 与受限 top-down 回流基础上尝试更强的双向 cross-attention 或 confidence-aware 融合实验。
4. 每次只引入一个结构实验，并保存训练配置、checkpoint `image_size`、LR 分组日志和可视化样本，便于归因。
