# Y-FFC 当前流程问题总结（仅分析，不改代码）

## 背景与目标
本次仅针对当前实现进行问题发现与归纳，聚焦 [Y_FFC_unified.py](Y_FFC_unified.py)、[train.py](train.py)、[process_images.py](process_images.py) 的模型-训练-推理闭环，定位影响 SSIM 与漫反射输出精度的关键薄弱点。

## 当前流程概览
- 模型主链路：`输入x -> Encoder -> GradientDecoder -> EnhancementLayers -> MainDecoder(est_b) -> coarse=x-est_b -> refinement_head -> pred_diffuse`，输出 `{r,g,b}`，见 [Y_FFC_unified.py:2144-2169](Y_FFC_unified.py#L2144-L2169)。
- 训练主链路：灰度读取+resize，`EnhancedCompositeLoss` 监督，Warmup+Cosine 调度，AMP+EMA，见 [train.py:128-135](train.py#L128-L135)、[train.py:1149-1169](train.py#L1149-L1169)。
- 推理主链路：按训练 patch 进行滑窗 + Hann 融合，最终保存 8-bit PNG，见 [process_images.py:195-261](process_images.py#L195-L261)、[process_images.py:264-286](process_images.py#L264-L286)。

## 问题清单（按优先级）

### 高优先级
1. 训练/推理分辨率与上下文机制存在不一致
- 现象：训练固定 resize 到 `IMAGE_SIZE`，推理通常保留原图并滑窗切 patch。
- 定位：训练 resize [train.py:131-135](train.py#L131-L135)；推理 patch 常量 [process_images.py:494-497](process_images.py#L494-L497)、滑窗 [process_images.py:195-261](process_images.py#L195-L261)。
- 影响：分布偏移 + 全局上下文不足，易导致跨 patch 结构一致性下降，SSIM 受损。

2. EMA 验证口径与 best 模型保存口径不一致
- 现象：验证阶段启用 EMA 权重评估，但 best checkpoint 保存使用 `model.state_dict()`（非 EMA 权重快照）。
- 定位：验证前后 EMA 切换 [train.py:1249-1258](train.py#L1249-L1258)；best 保存 [train.py:1330-1354](train.py#L1330-L1354)。
- 影响：日志中的“最优指标”与最终落盘模型性能可能不一致。

3. 梯度累积与 scheduler 步进频率可能不匹配
- 现象：scheduler 配置按 `len(train_loader)` 估算 epoch 内迭代，但真实 `scheduler.step()` 在累积后才调用。
- 定位：累积逻辑 [train.py:409-420](train.py#L409-L420)；scheduler 构建 [train.py:1161-1165](train.py#L1161-L1165)。
- 影响：有效学习率曲线与预期 warmup/cosine 节奏偏离，可能影响收敛后期 SSIM 上限。

4. `gt_specular` 在不同数据路径下定义口径不一致
- 现象：一处使用 clamp 后 specular，另一处直接 `input-target` 未 clamp。
- 定位：增强路径 [Y_FFC_unified.py:1589-1596](Y_FFC_unified.py#L1589-L1596)；默认路径 [Y_FFC_unified.py:1685](Y_FFC_unified.py#L1685)。
- 影响：监督目标分布不统一，增加训练噪声与不稳定性。

### 中优先级
5. 漫反射主恢复依赖硬减法，易放大 `est_b` 误差
- 现象：`coarse = clamp(x - est_b)` 后再小尺度残差精修。
- 定位：硬减法与精修 [Y_FFC_unified.py:2167-2168](Y_FFC_unified.py#L2167-L2168)；精修头尺度注释 [Y_FFC_unified.py:2129-2141](Y_FFC_unified.py#L2129-L2141)。
- 影响：一旦 `est_b` 偏差较大，coarse 信息被截断，后续修正空间有限。

6. 小 batch 下 BN 统计稳定性风险
- 现象：工程默认 batch 较小，网络大量 BN。
- 定位：batch 配置 [train.py:1516-1518](train.py#L1516-L1518)；网络各模块广泛使用 BN（见模型定义）。
- 影响：统计抖动可能引入纹理噪声或细节不稳定，影响 SSIM。

7. 高光合成增强默认始终开启（`p=1.0`）
- 现象：训练集每样本均叠加合成高光增强。
- 定位：增强配置 [train.py:1083-1089](train.py#L1083-L1089)。
- 影响：若合成分布与真实高光差异较大，模型可能偏向“合成模式”.

8. 推理输出 8-bit 量化与重采样导致细节损失
- 现象：输出保存前 clamp 后转换 `uint8`。
- 定位：保存逻辑 [process_images.py:264-286](process_images.py#L264-L286)。
- 影响：弱纹理/平滑过渡细节丢失，肉眼质量与评估指标都可能受影响。

### 低优先级
9. 转置卷积上采样存在棋盘纹理风险
- 现象：主干与分支存在多处 transpose conv 上采样。
- 定位：`TransposedConvBlock` [Y_FFC_unified.py:611-621](Y_FFC_unified.py#L611-L621)。
- 影响：局部周期伪影可能拉低结构一致性。

10. 镜面注意力包含亮度先验，可能误判亮漫反射区域
- 现象：注意力使用原图亮度作为先验输入之一。
- 定位：`SpecularAwareAttention.forward` [Y_FFC_unified.py:1391-1403](Y_FFC_unified.py#L1391-L1403)。
- 影响：高亮但非镜面区域可能被过度抑制或错误重分配。

11. Swin padding 机制与边界行为存在潜在域偏差
- 现象：Swin 分支 pad 到 32 倍数时使用常数 0；推理滑窗边界 pad 用 reflect/replicate。
- 定位：Swin pad [Y_FFC_unified.py:716-723](Y_FFC_unified.py#L716-L723)；推理 pad [process_images.py:229-233](process_images.py#L229-L233)。
- 影响：边界区域处理口径不一致，可能出现边缘细节偏差。

12. Swin 与 ResNet 的多阶段融合“能用，但还可以更统一”
- 现象：H/4、H/8 使用 `AdaptiveSkipFusion` 做单向残差注入；H/16 使用 `CrossModalFusion` 做双向交叉注意力。
- 定位：[Y_FFC_unified.py:747-795](Y_FFC_unified.py#L747-L795)、[Y_FFC_unified.py:797-903](Y_FFC_unified.py#L797-L903)、[Y_FFC_unified.py:1770-1850](Y_FFC_unified.py#L1770-L1850)
- 影响：不同尺度的融合范式不一致，浅层更偏“注入”，瓶颈层更偏“对话”，可能导致各尺度语义对齐不稳定，decoder 看到的特征分布也不完全一致。
- 改进方向：考虑让 H/4、H/8 使用更明确的尺度感知门控（按尺度分别设定融合强度/结构），或在浅层加入轻量双向交互，让 Swin 上下文不只是单向补充 ResNet。

13. H/4、H/8 的融合偏保守，可能低估 Swin 的有效上下文
- 现象：`beta` 初始为 0，且融合只做 `res_feat + beta * gate * swin_proj`，没有 reverse path。
- 定位：[Y_FFC_unified.py:756-794](Y_FFC_unified.py#L756-L794)
- 影响：训练早期稳定性更好，但 Swin 信息注入可能过慢；在需要结构上下文的高光区域，SSIM 提升可能不够快。
- 改进方向：让 `beta` / gate 具备分阶段或分尺度的自适应上升；对 H/4 与 H/8 使用不同融合强度，必要时加入 confidence-aware gating。

14. H/16 的双向交叉注意力方向合理，但前置对齐与归一化还可以更强
- 现象：`CrossModalFusion` 直接对 ResNet x_bb_4 与 Swin H/16 做双向 cross-attn，再用 gate 混合。
- 定位：[Y_FFC_unified.py:797-903](Y_FFC_unified.py#L797-L903)、[Y_FFC_unified.py:1845-1850](Y_FFC_unified.py#L1845-L1850)
- 影响：瓶颈层做双向交互是合理的，但 CNN/Transformer 特征统计差异较大，若缺少更显式的归一化/对齐，注意力可能偏向某一模态。
- 改进方向：在跨模态注意力前加入更明确的 modality norm / projection；或采用“先对齐再交互”的轻量块，降低模态偏置。

15. 当前没有显式的跨尺度一致性约束
- 现象：H/4、H/8、H/16 各自独立融合，没有显式约束它们对同一结构区域给出一致的上下文。
- 定位：[Y_FFC_unified.py:1828-1850](Y_FFC_unified.py#L1828-L1850)
- 影响：局部边缘、细条纹、高光边界在不同尺度可能语义不一致，最后会通过 decoder 放大成细节漂移。
- 改进方向：考虑引入渐进式 top-down 融合、跨尺度对齐约束，或让更高层上下文反向校正 H/8、H/4。

16. 训练监控的 SSIM 口径与实际优化目标不一致
- 现象：`EnhancedCompositeLoss` 内部优化的是 `l_ssim = 0.8*l_msssim + 0.2*l_ssim_single`，但 `loss_dict['ssim']` 返回的是 `l_ssim_single`，训练统计又用 `1-loss_dict['ssim']` 作为 SSIM。
- 定位：[Y_FFC_unified.py:1295-1298](Y_FFC_unified.py#L1295-L1298)、[Y_FFC_unified.py:1340-1342](Y_FFC_unified.py#L1340-L1342)、[train.py:310-312](train.py#L310-L312)
- 影响：监控指标和真实优化目标存在偏差，容易误判“是否真的在提升结构一致性”。
- 改进措施：统一日志口径，显式同时记录 `ssim_single` 与 `ssim_mixed`（或直接把 `loss_dict['ssim']` 改为 mixed 对应项）。

17. `refinement_head` 学习率分组可能偏保守
- 现象：`build_optimizer_v2` 未将 `refinement_head` 单独归为解码/新模块组，通常会落入 `other_p`（0.5×lr）。
- 定位：[train.py:198-226](train.py#L198-L226)、[Y_FFC_unified.py:2154](Y_FFC_unified.py#L2154)
- 影响：漫反射最后精修头学习速度可能不足，限制高频细节和 SSIM 上限。
- 改进措施：将 `refinement_head` 纳入 1.0×lr 组（或单独设略高 lr）以提高末端校正能力。

18. 感知损失输入域与 VGG 预训练域未严格对齐
- 现象：灰度图直接 repeat 到 3 通道后送 VGG，未做 ImageNet mean/std 标准化；若权重加载失败会退化到随机 VGG。
- 定位：[Y_FFC_unified.py:1027-1030](Y_FFC_unified.py#L1027-L1030)、[Y_FFC_unified.py:1044-1047](Y_FFC_unified.py#L1044-L1047)
- 影响：感知项监督信号稳定性不足，可能对纹理恢复产生噪声甚至负迁移。
- 改进措施：对感知分支输入显式做 ImageNet 标准化，并在日志中标记是否成功加载预训练 VGG；加载失败时默认关闭 perceptual 项。

19. `gt_specular` 在数据集默认路径仍为未截断差分
- 现象：增强路径使用 clamp 后 specular，而默认路径仍是 `input-target` 直接差分。
- 定位：[Y_FFC_unified.py:1589-1596](Y_FFC_unified.py#L1589-L1596)、[Y_FFC_unified.py:1685](Y_FFC_unified.py#L1685)
- 影响：同一任务出现两套监督分布，影响 specular 分支与 diffuse 分支协同学习。
- 改进措施：统一 `gt_specular` 构造口径（建议同域截断/归一化策略一致），避免训练信号分裂。

20. 验证期用了 EMA 权重评估，但 best 权重保存为非 EMA 快照
- 现象：验证前 `ema.apply_shadow()`，验证后 `ema.restore()`，但 best 保存调用 `model.state_dict()`。
- 定位：[train.py:1249-1258](train.py#L1249-L1258)、[train.py:1330-1354](train.py#L1330-L1354)
- 影响：记录的最优分数与落盘 best 权重不完全一致，复现实验时会有落差。
- 改进措施：保存 best 时同步保存 EMA 版本权重（或直接将 best 定义为 EMA 权重）。

21. 学习率日志仅显示首个参数组，不利于分层优化诊断
- 现象：训练日志只打印 `optimizer.param_groups[0]['lr']`。
- 定位：[train.py:1285](train.py#L1285)
- 影响：难以观察 `adaptive_skip_fusion/decoder/ffc` 等关键组是否按预期学习。
- 改进措施：按组输出 lr（至少输出各组 min/max 或命名组 lr），让调参可解释。

22. Warmup/Cosine 步进与梯度累积步频存在潜在节奏偏差
- 现象：scheduler 以 `iters_per_epoch=len(train_loader)` 构建，但 `scheduler.step()` 在 accumulation 后才执行。
- 定位：[train.py:409-420](train.py#L409-L420)、[train.py:1161-1165](train.py#L1161-L1165)
- 影响：有效学习率轨迹可能偏离设计值，影响中后期收敛与最终 SSIM。
- 改进措施：让 scheduler 的总 step 与真实 optimizer step 严格一致（基于 `ceil(len(loader)/accum)` 计算）。

## 对 SSIM 与漫反射精度的总体影响判断
- 当前主要瓶颈更偏“流程一致性与训练口径一致性”，其次是“结构先验导致的误差放大”。
- 若先解决高优先级问题，再处理中优先级项，通常比先做结构性大改更容易带来稳定 SSIM 提升。

## 后续改进方向（仅方向，不落地改动）
1. 先统一训练-验证-推理口径（尺寸、patch、EMA 保存/评估一致性、specular GT 定义一致性）。
2. 再校准优化过程（scheduler 与 accumulation 的节奏对齐、损失统计口径更一致）。
3. 最后再评估模型结构项（硬减法链路敏感性、上采样伪影、亮度先验注意力）。