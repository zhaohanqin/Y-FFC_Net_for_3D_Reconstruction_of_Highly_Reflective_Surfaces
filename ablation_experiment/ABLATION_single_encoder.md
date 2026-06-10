# 消融实验 C：去除 Swin 编码分支（ABLATION_single_encoder）

**文件**：`Y_FFC_single_encoder.py` / `train_single_encoder.py`
**对照**：完整 YFFC（baseline `train.py`，建议加同种子 42 重跑）

## 为什么这么做

完整模型是编码侧的 X 型结构：ResNet50 局部纹理分支与 Swin-Tiny 窗口上下文分支并行处理同一灰度输入，在 H/4、H/8（AdaptiveSkipFusion 门控注入）和 H/16（CrossModalFusion 双向交叉注意力）三个尺度汇聚。其假设是：CNN 的局部纹理（条纹边、高光边界形态）与 Transformer 的窗口级长程上下文（大面积反射区域、区域间关系）互补，融合后优于任一单分支。

本实验删除 `SwinEncoderBranch`、`fuse_h4/fuse_h8`、`swin16_proj`、`CrossModalFusion`，编码器退化为单 ResNet50；`CrossScaleTopDownRefinement` 只作用于 ResNet 特征、与 Swin 无关，故予以保留以隔离变量。单分支编码器输出 5 级特征的通道与分辨率与原版完全一致，因此**两条解码臂一行代码都不需要修改**——这正是该消融能做到严格单变量的原因。

## 这样有什么效果（如何解读）

- 大尺度高光区域恢复变差、长程结构一致性下降 → 窗口上下文互补有效，双编码 X 结构是性能来源；
- 基本不变 → 在当前 128×160 小 patch 训练分辨率下 Swin 的长程优势可能发挥有限（这本身是有价值的结论，提示在 320×256 / 384×384 等更大 patch 上复验）；
- 精度损失很小但显存/时间显著下降 → 给出明确的精度-效率权衡数据，支撑部署场景的裁剪决策。

## 为什么能与完整模型严格区分（体现结构优越性）

1. 解码侧（梯度臂、增强层、主解码器、双头、精修头）与全部损失、权重调度逐项不变，唯一变量是"Swin 上下文分支及其三处融合模块"；
2. 优化器分组规则不变：Swin 相关参数组自然为空并被 `build_optimizer_v2` 自动过滤，ResNet 仍在 0.1× lr 预训练组，分组语义与 baseline 一致；
3. 若完整模型显著占优，可证明跨模态双编码融合（而非单纯加深 CNN）是结构优越性的来源；与实验 A 联合分析还能回答"全局信息到底由 FFC 还是 Swin 提供、二者是否冗余"。

## 训练方式

```bash
python train_single_encoder.py
```
