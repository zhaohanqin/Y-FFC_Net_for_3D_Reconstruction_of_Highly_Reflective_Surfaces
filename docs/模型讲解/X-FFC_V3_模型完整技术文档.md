# X-FFC V3 模型完整技术文档

---

## 1. 文档说明

本文档是对 **X-FFC（X-shaped Fast Fourier Convolutional Network）** 模型的完整技术说明，基于 `X_FFC_unified.py`（模型定义）与 `train.py`（训练脚本）的源代码逐行分析而成。相比先前版本的总结文档，本文档：

- 完全忠于当前代码实现，所有模块描述均可溯源至具体代码行
- 增加了当前 X-FFC-Swin 版本引入的全部改进点（Swin-Tiny 编码器、AdaptiveSkipFusion、H/16 CrossModalFusion、残差精修头、梯度门控注入等）
- 详细标注了每个改进点的编号标签（如 \[P1\]、\[S5\]、\[F6\] 等），便于与代码注释对照
- 新增了训练系统的完整说明（损失调度、优化器配置、数据增强流水线等）

### 1.1 当前模型整体模块流

当前代码实现可以按模块划分为以下数据流：

1. **输入层**：输入为单通道灰度条纹图像 `x`，形状为 `[B, 1, H, W]`，归一化到 `[0, 1]`。
2. **编码层 EncoderSwin**：
   - **ResNet50 局部纹理分支**输出 `x_bb_1/x_bb_2/x_bb_3/x_bb_4`，分别对应 H/2、H/4、H/8、H/16，用作解码器 skip connection。
   - **Swin-Tiny 窗口上下文分支**直接处理原始 patch，不强制 resize 到 224×224，输出 `swin_4/swin_8/swin_16`。这会保留结构光条纹的空间采样、局部相位关系和条纹周期。
   - **多尺度 skip 融合**：`swin_4` 与 `x_bb_2`、`swin_8` 与 `x_bb_3` 通过 `AdaptiveSkipFusion` 做内容自适应残差注入。
3. **瓶颈层数据流**：`swin_16` 先经 `swin16_proj` 投影到 512 通道，再与 ResNet 深层特征 `x_bb_4` 通过 `CrossModalFusion` 双向交叉注意力融合；融合结果依次经过 `cba_after_backbone -> FFC bottleneck -> HighFrequencyEnhancement`，得到瓶颈特征 `x_c`。
4. **高反光/结构识别辅助分支（V3.1 双头）**：`GradientDecoder` 接收 `x_c` 与编码器 skip，逐级上采样并并联输出两个头：
   - **梯度头**：输出梯度图 \(g\)（监督目标 \(\nabla I\)）；
   - **spec_likelihood 头（V3.1 新增）**：输出高光软概率图 \(s \in [0,1]\)（监督目标 \(gt\_specular\)）；
   - 两头共享 `block_5_conv_1` 输出特征；
   - 同时输出 4 级中间特征供 `EnhancementLayers` 使用。
5. **增强桥接模块**：`EnhancementLayers` 对梯度分支的中间特征做频域分解和增强，**V3.1 中 `specular_gate` 由 2 通道升级为 3 通道**，同时接收 `g`、原始 `input` 和 `s`，为主解码器提供结构化上下文。
6. **主解码层（V3.1 三路调制）**：`MainDecoder` 结合 `x_c`、编码器 skip、增强特征和梯度图。在 level 2/3/4 三个尺度上做**三路联合调制**：`grad_gate(g) + joint_gate(cat(g, input, s)) + spec_lik_gate(s)`，每路独立 scale 参数控制。最终预测镜面反射分量 `b`。
7. **漫反射恢复与精修**：先执行物理减法 `coarse = clamp(x - b, 0, 1)`，再由 `ResidualRefinementHead` 对粗漫反射结果做小幅像素级修正，得到最终漫反射输出 `r`。
8. **跨尺度一致性辅助项与 top-down 回流**：训练时额外计算 H/4、H/8、H/16 融合特征的归一化响应图一致性标量 `cs`，用于约束不同尺度对同一结构区域的响应保持一致。编码器还在 H/16 cross-modal 融合后加入轻量 top-down refinement，将 H/16 上下文逐级反馈到 H/8、H/4 skip。`cs` 只参与训练损失，不作为推理图像结果保存。

### 1.2 当前实现更新（resize-conv 与跨尺度一致性）

当前版本相对原 V3 文档新增了两项结构级改动：

- **主解码器上采样从转置卷积切换为 resize-conv**：`TransposedConvBlock` 保留类名和外部接口，但内部改为三路 `Upsample(mode='bilinear') + Conv2d`。这降低了转置卷积带来的棋盘纹理风险。由于权重形状发生变化，旧 checkpoint 的部分 decoder 权重无法 strict 复用；推理加载器会过滤形状不匹配的旧权重并加载其余可兼容权重。
- **新增 cross-scale consistency loss 与 top-down refinement**：`YFFC.forward()` 返回 `cs` 标量，`EnhancedCompositeLoss` 使用 `lambda_cs` 将其纳入总损失。`CrossScaleTopDownRefinement` 在 H/16 -> H/8 -> H/4 上执行受限残差回流，配合 `cs` loss 约束多尺度结构响应一致性，目的是减少高光边界、细条纹和局部结构在不同尺度间的语义漂移。

### 1.3 V3.1 改造：梯度分支任务化与高光自信度增强

V3.1 在 V3 基础上做了三轮密切关联的改造，目标是让模型对**镜面反射的预测更合理、更自信**。

**第一轮：梯度分支监督目标重定义（已落地）**

将梯度分支的监督目标由"漫反射梯度 \(\nabla I_d\)"重定向为"输入图像梯度 \(\nabla I\)"。理由：在结构光去高光任务中，输入图的边缘已经天然携带了高光与材质的全部边界信号，远比漫反射梯度更直接、更稳定。`compute_input_gradient_target` 由 `EnhancedPairedTransform` 在数据增强后即刻执行，作为 `gt_gradient` 写入数据集字段。

**第二轮：梯度分支任务化辅助头（GradientDecoder.spec_lik_head）**

为 `GradientDecoder` 新增并联辅助头 `spec_lik_head`，与梯度头 `block_5` 共享 `block_5_conv_1` 输出特征，独立预测全分辨率 **高光软概率图 \(s \in [0,1]\)**。监督信号为 `gt_specular`。这一辅助头同时承担三个角色：

1. 作为辅助任务，给梯度分支注入"任务化"的高光语义；
2. 作为 `EnhancementLayers._SpectralDecompositionUnit` 的第三通道输入（与 \(\nabla I\) 和原始 input 一起生成 specular gate）；
3. 作为 `MainDecoder` 的独立调制通道（`spec_lik_gate_2/3/4`），与现有 `grad_gate` / `joint_gate` 并列。

`spec_lik_head` 的最后一层卷积采用 **零权重 + bias = −2.0** 的初始化，使训练初期 sigmoid 输出 ≈ 0.12（接近高光像素的统计先验比例），避免对下游主任务造成冲击。

**第三轮：损失加固、高光自信度增强与数值稳定性修复**

通过对训练监控的诊断，发现 `pred_s_lik` 与 `pred_b` 在高光区域**位置已对、但幅度均偏低**（pred_s_lik 大量像素停在 0.3~0.5，无法跨过 0.5 阈值）。同时早期版本在 Epoch 3 后出现大量 `grad NaN/Inf` 跳过。两类问题一并修复：

- **\[NaN-Fix\]** 损失中所有 `1/(p_abs*g_abs).clamp_min(1e-6)` 类操作改为能量加权的无除法形式；所有 `sum / mask.sum().clamp_min(1e-6)` 改为 `mean` 或 `clamp_min(1.0)`；`normalize_per_sample` 的 scale 下界由 \(10^{-8}\) 提升至 \(10^{-3}\)；所有 `sqrt(... + 1e-12)` 改为 `sqrt(... + 1e-6)`；`charbonnier_loss` 的 \(\epsilon\) 由 \(10^{-6}\) 提升至 \(10^{-3}\)。
- **\[P1-UPGRADE\]** `l_s_lik` 由 Charbonnier+SSIM 改为 **BCE+SSIM**。Charbonnier 对 sigmoid 输出在中段反向梯度被 \(\sigma'(x)=\sigma(x)(1-\sigma(x))\) 压扁，BCE 损失对 sigmoid logits 的梯度恒为 \(p - t\)、不饱和，是 sigmoid 输出的理论最优损失，直接驱动 `pred_s_lik` 在高光区域跨过 0.5 阈值。
- **\[P2-UPGRADE\]** `lambda_s_lik` / `lambda_s_cons` / `lambda_spec_peak` 调度由 60% 截断 改为 **40% 截断**，加速 spec_likelihood 系列的收敛。
- **\[P3-UPGRADE\]** `l_s_cons` 由单向（`pred_s_lik` 学 `est_b`）改为**双向一致性**：`est_b` 也反过来从 `pred_s_lik.detach()` 学习，让主解码器获得已被强监督的辅助头的"应该多高"信号。
- **\[NEW S3\]** 新增 `l_spec_peak`：在 `gt_specular > 0.3` 区域单侧惩罚 `pred_b` 欠预测（\(F.relu(gt - pred) \cdot \mathbb{1}_{gt>0.3}\)）。仅惩罚欠预测，不惩罚过预测，明确鼓励主解码器在强高光区敢于拉满。

V3.1 的所有改造**完全向后兼容旧推理 checkpoint**：`process_images.py._load_state_dict_flexible` 会过滤形状不匹配的旧权重，输出字典中的 `'s'` 字段在旧权重缺失时使用 `output.get('s')` 静默降级。

---

## 2. 模型命名与版本演进

### 2.1 命名来源

模型命名为 **X-FFC**，其中 **X** 代表双编码器分支在瓶颈层交汇的拓扑形态。具体而言：

- **上方两条编码路径**：ResNet50（空洞卷积版）与 Swin-Tiny 并行提取特征；H/4、H/8 通过 AdaptiveSkipFusion 增强 skip connection，H/16 在瓶颈层通过 CrossModalFusion 交叉注意力模块汇聚
- **下方两条解码路径**：GradientDecoder（辅助梯度预测）与 MainDecoder（主任务镜面反射重建）分叉执行各自任务，通过 EnhancementLayers 桥梁进行信息交互

这一拓扑结构整体呈"X"形，故以此命名。

### 2.2 版本标签体系

代码中通过方括号标签标记每个改进点，本文档沿用此体系：

| 标签前缀 | 含义 | 示例 |
|---------|------|------|
| \[P\*\] | 流水线/训练/输出头改进 | \[P1\] 残差精修头、\[P2\] 扩展输出头 |
| \[S\*\] | Swin 编码器分支改进 | \[S1\] 单通道 Swin、\[S5\] 自适应 skip 融合 |
| \[F\*\] | FFC 模块优化 | \[F1\] GELU 激活、\[F6\] Tukey 窗 |

---

## 3. 模型总体概述

### 3.1 任务定义

X-FFC 是一个专为**图像高光分解任务**设计的深度学习模型。给定一张包含高反射条纹的单通道灰度输入图像 \(x\)，模型的目标是将其精确分解为：

- **镜面反射分量（specular reflection）** \(b\)：由网络直接预测
- **漫反射分量（diffuse reflection）** \(r\)：通过减法获得，\(r = \text{clamp}(x - b, 0, 1)\)

最终输出经 \[P1\] 残差精修头进一步像素级校正后生成最终漫反射预测。

### 3.2 网络输出格式

网络返回一个字典，包含四个分量（V3.1）：

```python
{'r': pred_diffuse,    # [B, 1, H, W]  漫反射分量（主输出）
 'g': gradient_map,    # [B, 1, H, W]  输入图像 Sobel 梯度图（辅助输出）
 'b': est_specular,    # [B, 1, H, W]  镜面反射分量
 's': spec_likelihood, # [B, 1, H, W]  高光软概率图（V3.1 新增辅助输出）
 'cs': cs_loss}        # 标量，仅训练使用的跨尺度一致性辅助项
```

`'s'` 字段由 `GradientDecoder.spec_lik_head` 输出，取值范围 \([0,1]\)。推理时该字段同时作为附加可视化结果输出为 `*_spec_likelihood.png`。

### 3.3 整体架构概览

X-FFC V3 / X-FFC-Swin 的整体架构可概括为 **"双编码 → 多尺度自适应融合 → 频域瓶颈增强 → 双解码 → 残差精修"** 的设计范式：

1. **双编码器主干（EncoderSwin）**：ResNet50（空洞卷积版）负责局部纹理与条纹边界，Swin-Tiny 负责窗口上下文和层级特征。Swin 分支输出 H/4、H/8、H/16；H/4/H/8 通过 AdaptiveSkipFusion 增强中层 skip，H/16 通过 CrossModalFusion 与 `x_bb_4` 做瓶颈层双向交互。融合特征经 FFC 瓶颈层和 HighFrequencyEnhancement 高频增强后输出 5 级编码特征。

2. **梯度解码分支（GradientDecoder + EnhancementLayers）**：执行辅助任务——由网络预测输入图像梯度图，并输出 4 级中间特征经 EnhancementLayers 频域精炼后供主解码器使用。训练监督目标由原始高反光输入的 Sobel 梯度生成，但推理时的 `g` 由 GradientDecoder 输出，不直接调用 Sobel 后处理。

3. **主解码分支（MainDecoder）**：执行核心任务——重建镜面反射分量。融合编码器跳跃特征（经 Attention\_block 门控加权）、增强特征、梯度门控注入 \[P10\]，并在每一级应用 SpecularAwareAttention 高光感知注意力。输出头采用扩展设计 \[P2\]（4 层卷积 + 膨胀卷积 + skip connection）。

4. **残差精修头（ResidualRefinementHead）** \[P1\]：对硬减法 \(x - b\) 得到的粗漫反射分量做像素级修正，使用 Tanh 与 `scale=0.08` 限制修正幅度在 \(\pm 0.08\) 范围内。

---

## 4. 各模块详解

### 4.1 FFC 核心模块（快速傅里叶卷积）

FFC（Fast Fourier Convolution）是 X-FFC 的频域处理基础，由三层嵌套结构组成：FourierUnit → SpectralTransform → FFC\_BN\_ACT。它被广泛嵌入到编码器瓶颈层、两个解码器的多个尺度级别以及 EnhancementLayers 中。

#### 4.1.1 FourierUnit（傅里叶单元）

- **模块名称**：`FourierUnit`
- **模块作用**：FFC 的最底层核心组件，执行空间域 → 频域 → 频域卷积 → 空间域的完整变换。
- **在模型中的位置**：被 SpectralTransform 调用，间接被所有 FFC\_BN\_ACT 实例使用。

**工作流程**：

```
空间特征 x → [F6] Tukey 窗（α=0.25）→ rFFT → 实虚部拼接为通道
→ 1×1 Conv → [F1] BN + GELU → [F2] SE 通道注意力
→ 复数合并 → iRFFT → 空间域输出
```

**X-FFC 中的优化改进**：

- **\[F1\] GELU 替代 ReLU**：ReLU 会将负值归零，但频域中负值（尤其是虚部）携带重要的相位信息。GELU 的平滑特性避免了对负的虚部执行 hard-zero，保护了频域相位信息的完整性。

- **\[F2\] 频率 SE 通道注意力**：在频域卷积之后插入一个 SE（Squeeze-and-Excitation）模块，结构为 `AdaptiveAvgPool2d → Conv2d(2C, C//4) → GELU → Conv2d(C//4, 2C) → Sigmoid`。该模块对频谱的不同频段进行自适应加权，使网络能够自动选择对当前任务有用的频段（例如：增强条纹对应的特定频率分量），抑制无关频段的噪声。

- **\[F6\]\[P11\] Tukey 窗**（α=0.25）：FFT 假设输入信号是周期性的，但实际图像边界存在不连续性，会导致频谱泄漏（能量从真实频率扩散到邻近频率）。Tukey 窗在保留中间 75% 信号不衰减的前提下，对两端各 12.5% 进行余弦过渡衰减，有效减少了频谱泄漏，同时避免了 Hann 窗过度衰减边缘信息的问题。代码中通过 `_tukey_window()` 函数生成一维窗函数，再通过外积构造二维窗。

- **AMP 兼容**：当输入为 float16 时自动转换为 float32 进行 FFT 运算（PyTorch 的 FFT 不支持 float16），运算完成后再转回 float16，确保混合精度训练的兼容性。

#### 4.1.2 SpectralTransform（谱变换模块）

- **模块名称**：`SpectralTransform`
- **模块作用**：在 FourierUnit 之上封装通道降维、全局/局部傅里叶处理和门控融合。
- **在模型中的位置**：被 FFC\_BN\_ACT 的全局分支（convg2l、convg2g）调用。

**工作流程**：

```
输入 → 1×1 Conv 降维（out_channels // 2）→ BN + ReLU
→ 全局 FU（FourierUnit）处理全图
→ [F3] 独立 lfu_top / lfu_bot 分别处理上/下半区
→ [F4] 门控融合（Sigmoid 门控）
→ 1×1 Conv 升维 → 输出
```

**X-FFC 中的优化改进**：

- **\[F3\] LFU 拆分**：原始 FFC 论文使用单个 LFU（局部傅里叶单元）处理整张特征图。X-FFC 将其拆分为两个独立的 FourierUnit（`lfu_top` 和 `lfu_bot`），分别处理特征图的上半部分和下半部分。每个 LFU 拥有独立的参数，避免了同一组参数需要同时适配不同空间区域的参数竞争问题。

- **\[F4\] FU / LFU 门控融合**：全局 FU 和局部 LFU 的输出不再简单相加，而是通过一个可学习的 Sigmoid 门控（`fusion_gate`：1×1 Conv → Sigmoid）自适应决定融合比例：`gate * fu_out + (1 - gate) * lfu_out`。这使网络能够根据不同位置的内容特性动态调整全局信息和局部信息的贡献比例。

#### 4.1.3 FFC\_BN\_ACT（FFC 统一入口模块）

- **模块名称**：`FFC_BN_ACT`
- **模块作用**：将输入特征分为局部（L）和全局（G）两个分支，通过四路交互实现空间-频域信息融合。
- **在模型中的位置**：是 FFC 在网络中的统一调用入口，被 Encoder 瓶颈层、GradientDecoder 各级（ffc\_grad\_h16/h8/h4）、MainDecoder 各级（FFC1/FFC2、ffc\_h8/ffc\_h4）和 EnhancementLayers 的频谱分解单元广泛使用。

**四路交互机制**：

给定局部/全局通道比例 `ratio_gin` 和 `ratio_gout`，输入被按通道分为局部分支 \(x_l\) 和全局分支 \(x_g\)，输出同样分为两部分：

- **L2L（convl2l）**：局部→局部，标准空间卷积，保持局部纹理细节
- **G2L（convg2l）**：全局→局部，SpectralTransform，将频域全局信息注入局部分支
- **L2G（convl2g）**：局部→全局，标准空间卷积，将局部细节送入频域处理
- **G2G（convg2g）**：全局→全局，SpectralTransform，在频域内进行全局信息更新

局部输出 = L2L + \[F5\] 门控(G2L)；全局输出 = L2G + \[F5\] 门控(G2G)。

- **\[F5\] 跨域门控融合**：当频域分支（G2L 或 G2G）的输出注入到另一个分支时，不再直接相加，而是通过 `gate_l` / `gate_g`（1×1 Conv → Sigmoid）进行门控加权。这防止了频域噪声直接污染空间分支，使网络能够自适应调节频域信息的注入强度。

#### 4.1.4 FFCResnetBlock（FFC 残差块）

- **模块名称**：`FFCResnetBlock`
- **模块作用**：将两个 FFC\_BN\_ACT 层串联并加上残差连接，用于堆叠多层 FFC 处理。
- **结构**：`输入 → FFC1 → FFC2 → + 输入（残差）→ 输出`

---

### 4.2 Encoder（编码器）

- **模块名称**：`Encoder`
- **模块作用**：X-FFC 的特征提取主干，从单通道输入图像中提取多尺度、多语义层级的特征，为下游两个解码器提供信息源。
- **在模型中的位置**：网络最前端，接收原始输入图像，输出 5 组特征图。
- **与其他模块的关系**：输出 \((x\_bb\_1, x\_bb\_2, x\_bb\_3, x\_bb\_4)\) 被同时送入 GradientDecoder 和 MainDecoder 作为跳跃连接；瓶颈特征 \(x\_c\) 被送入两个解码器。

#### 4.2.1 ResNet50 分支（空洞卷积版）

采用 ImageNet 预训练的 ResNet50，关键修改如下：

- **单通道输入适配**：原始 ResNet50 的 7×7 stem 卷积接受 3 通道 RGB 输入，X-FFC 将其修改为单通道输入。初始化方式为将原始 3 通道权重沿通道维度取均值，即 `new_weight = old_weight.mean(dim=1, keepdim=True)`，这样保留了预训练的低级滤波器特性。

- **空洞卷积**（replace\_stride\_with\_dilation）：在 layer4 中使用空洞卷积（dilation=2）替代 stride=2 的下采样。效果是 layer4 的输出保持与 layer3 相同的空间分辨率（H/16 × W/16），但感受野等效于 stride=2 时的 H/32 覆盖范围。这对于捕捉条纹图案的细微变化尤为重要——既需要大感受野理解全局条纹结构，又不能过度下采样丢失空间细节。

- **通道降维**：4 个 stage 的输出分别为 256/512/1024/2048 通道，通过 1×1 卷积（`reduce_l1` 至 `reduce_l4`）降维至统一的 128/256/512/512 通道。layer3（1024→512）和 layer4（2048→512）的降维输出沿通道拼接后经 `fuse_l34`（3×3 卷积，1024→512）融合为 \(x\_bb\_4\)。

**ResNet50 各级输出**：

| Stage | 原始通道数 | 降维后 | 输出名称 | 空间尺寸 |
|-------|-----------|-------|---------|---------|
| stem (conv1 + pool) | 64 | 64 | \(x\_bb\_1\) | H/2 × W/2 |
| layer1 | 256 | 128 | \(x\_bb\_2\) | H/4 × W/4 |
| layer2 | 512 | 256 | \(x\_bb\_3\) | H/8 × W/8 |
| layer3 + layer4 | 1024+2048 → 融合 | 512 | \(x\_bb\_4\) | H/16 × W/16 |

#### 4.2.2 SwinEncoderBranch（Swin-Tiny 多尺度上下文分支）

- **模块名称**：`SwinEncoderBranch`
- **模块作用**：并行于 ResNet50 提取窗口上下文特征，并提供 H/4、H/8、H/16 三个层级输出。

**\[S1\] 单通道 Swin patch embedding**：

当前实现使用 `torchvision.models.swin_t` 作为默认骨干。原始 Swin patch embedding 是 `Conv2d(3, 96, kernel_size=4, stride=4)`，代码中将其改为 `Conv2d(1, 96, kernel_size=4, stride=4)`。若加载 ImageNet 预训练权重，则用 RGB 卷积核均值初始化灰度卷积核：

```python
new_weight = old_weight.mean(dim=1, keepdim=True)
```

这样既适配单通道灰度输入，又保留预训练低层滤波器能力。

**\[S2\] 不再强制 224×224 resize，保护结构光相位信息**：

Swin 分支直接处理训练/推理 patch。若输入 H/W 不能被 32 整除，先在右侧和下侧 padding，保证 Swin 分层下采样可以正常前向；后续在 Encoder 中插值对齐到 ResNet 对应尺度。推荐 patch 为 `320×256`、`384×384`、`512×512`，不建议直接整图训练 `2448×2048`。

这一点是 X-FFC-Swin 相比旧 ViT 路径的重要优势。结构光条纹图像的相位、周期和条纹边界都与空间采样强相关；强制 resize 到 `224×224` 会压缩或拉伸条纹，改变局部频率与相位关系，也会改变高光边界形态。当前 Swin 分支只做必要 padding，不做固定尺寸缩放，因此可以在原始 patch 尺度上保留相位恢复所需的条纹几何关系。

**\[S3\] 多尺度层级输出**：

torchvision Swin 的中间特征为 NHWC 格式，返回前统一转换为 NCHW：

```python
feat = feat.permute(0, 3, 1, 2).contiguous()
```

输出对应关系如下：

| Swin 输出 | 通道数 | 空间尺度 | 用途 |
|----------|--------|----------|------|
| `swin_4` | 96 | H/4 × W/4 | 增强 `x_bb_2` |
| `swin_8` | 192 | H/8 × W/8 | 增强 `x_bb_3` |
| `swin_16` | 384 | H/16 × W/16 | 瓶颈层 CrossModalFusion |

**\[S4\] AdaptiveSkipFusion 自适应 skip 融合**：

H/4、H/8 分辨率较高，不直接使用重型 CrossModalFusion，而是使用轻量内容自适应融合：

```python
swin_proj = Conv3x3_BN_GELU(swin_feat)
gate = sigmoid(gate_net(concat(res_feat, swin_proj)))
fused = res_feat + beta * gate * swin_proj
```

其中 `gate` 是逐空间位置、逐通道的融合权重，`beta` 是可学习标量，初始为 `0.0`。这种初始化让模型初始时接近纯 ResNet skip，避免突然破坏 Decoder 已适配的特征分布；训练过程中模型再自动学习 Swin skip 的注入强度。

**\[S5\] H/16 瓶颈上下文融合**：

`swin_16` 先经过 `swin16_proj` 投影到 512 通道，再与 ResNet 深层特征 `x_bb_4` 进入 `CrossModalFusion`。H/16 空间分辨率较低，适合做更强的双向交叉注意力交互。

#### 4.2.3 CrossModalFusion（交叉注意力融合模块）\[S6\]

- **模块名称**：`CrossModalFusion`
- **模块作用**：实现 ResNet 深层局部纹理特征与 Swin H/16 窗口上下文特征之间的双向信息交互。
- **在模型中的位置**：Encoder 中，位于 ResNet50 输出 \(x\_bb\_4\) 与 Swin 输出 `swin_16` 投影对齐之后。

**双向交叉注意力机制**：

**方向一：CNN → Swin（获取窗口上下文）**
CNN 特征作为 Query，Swin 特征作为 Key/Value。语义解读：对于 CNN 检测到的每个局部高亮区域，去查询 Swin 的窗口上下文——判断该高亮与周围条纹结构、反射区域形态之间的关系。

**方向二：Swin → CNN（获取局部细节）**
Swin 特征作为 Query，CNN 特征作为 Key/Value。语义解读：对于 Swin 建模到的中大范围高光上下文，去查询 CNN 的局部特征——获取高光边界的精确空间位置和细条纹纹理。

**计算效率优化**：使用 `sr_ratio=2` 对 Key/Value 进行空间降采样（AvgPool2d），将注意力计算的复杂度从 \(O(H^2W^2)\) 降低到 \(O(H^2W^2/4)\)。

**门控融合**：两个方向的输出通过 `gate = Sigmoid(Conv2d(2×out_dim → out_dim))` 自适应融合：`fused = gate * c2ctx_out + (1 - gate) * ctx2c_out`，再经 `3×3 Conv + BN + GELU` 最终投影。

#### 4.2.4 FFC 瓶颈层

融合后的 512 通道特征经 `cba_after_backbone`（3×3 Conv + BN + ReLU）降维至 256 通道后，进入 FFC 瓶颈层。具体操作为：

1. 将 256 通道特征按 `ratio_gin=0.5` 分为局部分支（128 通道）和全局分支（128 通道）
2. 经 FFC\_BN\_ACT 处理（包含 L2L、G2L、L2G、G2G 四路交互）
3. 输出的局部和全局分支拼接回 256 通道
4. 与输入做残差连接：`x_c = ffc_output + x_c`

该操作赋予编码特征**全局感受野**：频域中的 1×1 卷积等价于空间域中的全图范围卷积，使网络能够捕捉条纹的全局周期性模式。

#### 4.2.5 HighFrequencyEnhancement（高频增强模块）

- **模块名称**：`HighFrequencyEnhancement`
- **模块作用**：显式提取编码特征中的高频分量（边缘、跃变），为高反光检测提供清晰的边界信号。
- **在模型中的位置**：Encoder 的 FFC 瓶颈层之后，作用于 \(x\_c\)。

**内部结构**（三路融合）：

1. **固定 Laplacian 核**：使用不可学习的 Laplacian 卷积核 \(\begin{bmatrix}0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0\end{bmatrix}\) 对每个通道独立卷积（groups=C），直接提取二阶导数（边缘/跃变）。该核固定不参与训练，保证始终提取物理意义明确的边缘响应。

2. **可学习多尺度分支**（hf\_branch）：`3×3 Conv → BN → GELU → 3×3 空洞卷积(dilation=2) → BN → GELU`，通道数为输入的一半。空洞卷积扩大了感受野（等效 5×5），使该分支能够捕捉更大尺度的高频变化。

3. **融合与通道注意力**：三者（原始特征 + Laplacian 高频 + 可学习高频）沿通道拼接后经 `1×1 Conv → BN → GELU → 3×3 Conv → BN` 融合。随后经 SE 通道注意力（`AdaptiveAvgPool2d → Conv(C, C//8) → ReLU → Conv(C//8, C) → Sigmoid`）对融合后的高频特征进行通道级加权，自适应选择对当前图像最有用的高频通道。最终通过残差连接与输入相加：`output = x + se(fusion(cat[x, lap, learn]))`。

**设计意义**：高反光区域通常伴随着强烈的亮度跃变。通过显式的高频增强，编码特征中的高光边界信息被放大，有助于后续 SpecularAwareAttention 更精确地定位高光区域，并帮助 GradientDecoder 更好地预测梯度。

#### 4.2.6 编码器完整输出

| 输出名称 | 张量尺寸 | 来源 | 用途 |
|---------|---------|------|------|
| \(x\_bb\_1\) | \([B, 64, H/2, W/2]\) | ResNet50 stem | 跳跃连接至两个解码器 Level 4 |
| \(x\_bb\_2\) | \([B, 128, H/4, W/4]\) | ResNet50 layer1 降维 | 跳跃连接至两个解码器 Level 3 |
| \(x\_bb\_3\) | \([B, 256, H/8, W/8]\) | ResNet50 layer2 降维 | 跳跃连接至两个解码器 Level 2 |
| \(x\_bb\_4\) | \([B, 512, H/16, W/16]\) | ResNet50 layer3+4 融合 | 跳跃连接至两个解码器 Level 1 |
| \(x\_c\) | \([B, 256, H/16, W/16]\) | CNN-Swin 交叉融合 + FFC + HFE | MainDecoder 起始输入 + GradientDecoder 全局上下文 |

---

### 4.3 GradientDecoder（梯度解码器）

- **模块名称**：`GradientDecoder`
- **模块作用**：执行辅助任务——预测图像梯度图 \(x\_g\)，并在每一级解码过程中输出中间特征供 EnhancementLayers 处理。
- **在模型中的位置**：X-FFC 的左分支（辅助任务分支），接收编码器的四级跳跃特征和瓶颈特征。
- **与其他模块的关系**：
  - **输入**：\((x\_bb\_1, ..., x\_bb\_4)\) 作为跳跃连接，\(x\_c\) 作为全局上下文
  - **输出**：4 级中间特征送入 EnhancementLayers；最终梯度图 \(x\_g\) 送入 MainDecoder
- **在任务中的功能**：梯度信息能有效表征图像的结构和边缘。通过梯度预测辅助任务，网络获得额外的结构化约束，促使主解码器在重建镜面反射时更好地保持边缘清晰度。

**内部结构详解**：

**Level 1（H/16 × W/16）— 全局上下文融合**：
1. `fuse_global_context`：将 \(x\_bb\_4\)（512ch）和 \(x\_c\)（256ch）拼接后经 1×1 卷积（768→512）融合，确保梯度分支在最深层就能获取全局语义信息
2. `block_1_conv_1/2/3`：7×7 Conv(512→1024) → 1×1 Conv(1024→512) → 3×3 Conv(512→256)
3. `ffc_grad_h16`：FFC\_BN\_ACT（ratio=0.25），在频域处理梯度特征
4. 残差连接 + 5×5 同尺寸卷积输出 `x_to_be_enhanced_1`（256ch, H/16）
5. 与 \(x\_bb\_4\) 拼接后送入 Level 2

**Level 2（H/8 × W/8）**：
1. 3×3 Conv 降维（256+512→128）
2. `ffc_grad_h8`：FFC\_BN\_ACT（ratio=0.25）+ 残差
3. 4×4 转置卷积上采样至 H/8
4. 输出 `x_to_be_enhanced_2`（128ch, H/8），与 \(x\_bb\_3\) 拼接送入 Level 3

**Level 3（H/4 × W/4）**：
1. 3×3 Conv 降维（128+256→64）
2. `ffc_grad_h4`：FFC\_BN\_ACT（ratio=0.25）+ 残差
3. 4×4 转置卷积上采样至 H/4
4. 输出 `x_to_be_enhanced_3`（64ch, H/4），与 \(x\_bb\_2\) 拼接送入 Level 4

**Level 4（H/2 × W/2）**：
1. 3×3 Conv 降维（64+128→32）
2. 4×4 转置卷积上采样至 H/2
3. 输出 `x_to_be_enhanced_4`（32ch, H/2），与 \(x\_bb\_1\) 拼接送入 Level 5

**Level 5（H × W）— 梯度图输出**：
1. 4×4 转置卷积上采样至 H × W
2. 5×5 卷积（64→1），无 BN 和激活
3. Sigmoid 输出值域 \([0, 1]\) 的单通道梯度图 \(x\_g\)

**设计特点**：GradientDecoder 在 Level 1-3 中嵌入了 FFC 模块（ratio\_gin=ratio\_gout=0.25），使梯度分支也具备频域全局信息捕获能力。这对于捕捉条纹的整体周期性梯度模式至关重要。

---

### 4.4 EnhancementLayers（增强层）

- **模块名称**：`EnhancementLayers`
- **模块作用**：对 GradientDecoder 输出的 4 级中间特征进行频域精炼，生成更高质量的增强特征供 MainDecoder 使用。
- **在模型中的位置**：位于 GradientDecoder 和 MainDecoder 之间，是两个解码分支的"桥梁"。

**内部结构**：

由 4 个 `_SpectralDecompositionUnit`（频谱分解单元）组成，分别处理 4 个尺度的特征：

| 单元 | 输入通道 | 输出通道 | 输入来源 |
|------|---------|---------|---------|
| decomp\_unit\_1 | 256 | 128 | x\_to\_be\_enhanced\_1 |
| decomp\_unit\_2 | 128 | 64 | x\_to\_be\_enhanced\_2 |
| decomp\_unit\_3 | 64 | 32 | x\_to\_be\_enhanced\_3 |
| decomp\_unit\_4 | 32 | 16 | x\_to\_be\_enhanced\_4 |

**\_SpectralDecompositionUnit 内部结构**：

每个频谱分解单元包含三路处理和门控融合：

1. **投影分支**（projection\_branch）：7×7 大卷积核，扩大空间感受野，提取较大范围的结构信息
2. **镜面分支**（specular\_branch）：3×3 卷积后接 FFC\_BN\_ACT（ratio=0.25），在频域中提取与高光相关的频率分量
3. **门控融合**（gate\_conv）：将投影分支和镜面分支的输出拼接后经 `1×1 Conv → Sigmoid` 生成门控权重，最终输出为 `specular + specular * gate`（即 specular 分支的自增强）

**设计意义**：梯度分支生成的中间特征包含有用的结构信息，但未必是主解码器所需的最优表示。EnhancementLayers 通过频域分解和门控融合机制，对这些特征进行任务导向的提炼，使其更好地服务于镜面反射分量的重建。

---

### 4.5 MainDecoder（主解码器）

- **模块名称**：`MainDecoder`
- **模块作用**：X-FFC 的核心组件，负责重建镜面反射分量 \(est\_b\)。
- **在模型中的位置**：X-FFC 的右分支（主任务分支），汇聚了网络中所有其他模块的输出。
- **输入来源**：编码器全部 5 级特征、EnhancementLayers 的 4 级增强特征、GradientDecoder 的梯度图以及原始输入图像。

#### 4.5.1 瓶颈处理

```
x_c (256ch) → chunk(2) → FFC1 → FFC2 → cat + 残差 → MultiscaleConvBlock → (576ch)
```

1. **FFC 深层处理**：\(x\_c\) 被分为两个 128 通道的局部/全局分支，经两级级联的 FFC\_BN\_ACT（ratio=0.5）处理后拼接回 256 通道并与输入残差连接。这在编码器的频域处理基础上进一步增强条纹的全局周期性模式。

2. **MultiscaleConvBlock**：三路并行卷积——7×7 路径（96→192ch，大感受野）、3×3 路径（256→192ch，标准感受野）、3×3+3×3 路径（256→256→192ch，两步感受野）。三路输出拼接得到 576 通道特征。多尺度设计使网络能够同时感知大范围的光斑和小范围的点状高光。

#### 4.5.2 逐级上采样（Level 1-5）

MainDecoder 的核心是 5 级逐步上采样过程，每一级都执行精心设计的多源融合：

**Level 1（H/16 → H/16，同尺寸）**：
1. TransposedConvBlock\_same（576→768ch）：三路 3×3 卷积（不改变空间尺寸）
2. SpecularAwareAttention（sa1）：利用原始输入亮度增强高光区域响应
3. Attention\_block（att1）：用深层特征（768ch）作为 gate 对 \(x\_bb\_4\)（512ch）加权
4. 三源融合：`cat([自身 768ch, bridge 512ch, xe1 128ch]) → 1408ch`

**Level 2（H/16 → H/8）**：
1. TransposedConvBlock（1408→384ch）：三路 resize-conv（bilinear upsample + 3×3/5×5/7×7 卷积）上采样
2. ffc\_h8：FFC\_BN\_ACT（ratio=0.25）+ 残差连接
3. **\[P10\] 梯度门控注入**：`x = x * (1 + grad_gate_2(x_g))`，梯度图经 `3×3 Conv → Sigmoid` 生成调制门控，在边缘区域增强特征响应
4. SpecularAwareAttention（sa2）
5. Attention\_block（att2）对 \(x\_bb\_3\) 加权
6. 三源融合：`cat([自身 384ch, bridge 256ch, xe2 64ch]) → 704ch`

**Level 3（H/8 → H/4）**：
1. FeatureExtrationLayersB（704→640ch）：四路并行卷积（7×7/3×3/3×3/3×3+3×3），输出 192+192+128+128=640ch
2. TransposedConvBlock（640→192ch）resize-conv 上采样
3. ffc\_h4：FFC\_BN\_ACT（ratio=0.25）+ 残差连接
4. **\[P10\] 梯度门控注入**：`x = x * (1 + grad_gate_3(x_g))`
5. SpecularAwareAttention（sa3）
6. Attention\_block（att3）对 \(x\_bb\_2\) 加权
7. 三源融合：`cat([自身 192ch, bridge 128ch, xe3 32ch]) → 352ch`

**Level 4（H/4 → H/2）**：
1. TransposedConvBlock（352→96ch）resize-conv 上采样
2. **\[P10\] 梯度门控注入**：`x = x * (1 + grad_gate_4(x_g))`
3. SpecularAwareAttention（sa4）
4. Attention\_block（att4）对 \(x\_bb\_1\) 加权
5. 三源融合：`cat([自身 96ch, bridge 64ch, xe4 16ch]) → 176ch`

**Level 5（H/2 → H）— 输出**：
1. TransposedConvBlock（176→48ch）resize-conv 上采样至原始分辨率
2. 拼接梯度图：`cat([自身 48ch, x_g 1ch]) → 49ch`
3. **\[P2\] 扩展输出头**：
   ```
   49ch → output_reduce(1×1 Conv→32ch)
        → skip connection (1×1 Conv→32ch)
        → output_dilated(3×3 空洞Conv, dilation=2→32ch)
        → + skip
        → output_conv3(3×3 Conv→16ch)
        → output_conv4(3×3 Conv→1ch, 无BN无激活)
        → Sigmoid → est_b
   ```
   扩展输出头使用 4 层卷积（包含膨胀卷积和 skip connection），相比简单的单层输出卷积，能更好地融合最终层的多尺度信息，生成更精细的镜面反射估计。

#### 4.5.3 梯度门控注入机制 \[P10\]

MainDecoder 在 Level 2-4 中引入了梯度图调制门控：

```python
x_g_resized = interpolate(x_g, size=x.shape[-2:])
x = x * (1.0 + grad_gate(x_g_resized))
```

每个 `grad_gate` 是独立的 `3×3 Conv → Sigmoid`，将梯度图（值域 \([0,1]\)）转换为调制系数（值域 \([0,1]\)）。乘法调制的效果是：边缘区域（梯度值高）的特征被放大（最多 2×），平坦区域（梯度值低）保持不变。这使得梯度信息不仅作为辅助任务的约束，还直接参与主解码器的特征调制，显式强化边缘区域的重建精度。

---

### 4.6 SpecularAwareAttention（镜面反射感知注意力）

- **模块名称**：`SpecularAwareAttention`
- **模块作用**：利用原始输入图像的亮度信息作为先验，动态增强高光区域对应的特征响应。
- **在模型中的位置**：MainDecoder 的每一级 TransposedConvBlock 输出之后（共 4 处：sa1 至 sa4）。
- **在任务中的功能**：在解码器的每个尺度级别，根据输入图像的亮度分布生成高光先验掩码，使网络在所有分辨率层级上都对高光区域保持高灵敏度。

**\[P9\] 多级阈值设计**：

原始设计使用单一阈值，X-FFC V3 引入三级阈值 \([0.5, 0.7, 0.9]\)，分别对应弱/中/强高光：

1. **自适应阈值**：每个基础阈值通过轻量级 SE 模块从当前特征中动态调整：`eff_threshold = base_t * (0.5 + 0.5 * adaptive_offset)`。自适应偏移量由 `AdaptiveAvgPool2d → Conv(C, mid) → ReLU → Conv(mid, 3) → Sigmoid` 从特征全局统计中估计。

2. **软掩码生成**：`mask_i = Sigmoid(k * (input - eff_threshold_i))`，其中陡度参数 \(k=6\)（\[P9\] 从 15 降低到 6）。降低陡度使掩码更平滑，避免了硬阈值导致的梯度不稳定。三级掩码沿通道拼接得到 3 通道的 `multi_mask`。

3. **调制网络**：`cat([features, multi_mask]) → 3×3 Conv + BN + GELU → 3×3 Conv + BN + Sigmoid → att`

4. **特征增强**：`output = features * (1 + att)`。乘法形式 `(1 + att)` 保证高光区域被放大（att 接近 1 时特征翻倍），非高光区域不受损害（att 接近 0 时特征不变）。

---

### 4.7 Attention\_block（注意力门控块）

- **模块名称**：`Attention_block`
- **模块作用**：在跳跃连接融合时，对编码器浅层特征进行注意力加权。
- **在模型中的位置**：MainDecoder 的每个融合点（att1 至 att4）。

**工作流程**：

```
g (深层特征, gate signal) → W_g(1×1 Conv + BN) → g1
x (浅层特征, 待加权)     → W_x(1×1 Conv + BN) → x1
→ ReLU(g1 + x1) → psi(1×1 Conv + BN + Sigmoid) → 注意力权重
→ output = x * psi
```

**X-FFC 改进**：psi 输出 \(F\_l\) 通道（与浅层特征通道数相同），而非原版 Attention U-Net 的单通道。这意味着每个通道都有独立的注意力权重，提供了比单通道空间注意力更精细的通道级+空间级联合注意力控制。中间维度 \(F_{int}\) 被缩小以降低参数量并减轻过拟合。

---

### 4.8 ResidualRefinementHead（残差精修头）\[P1\]

- **模块名称**：`ResidualRefinementHead`
- **模块作用**：对硬减法 \(r_{coarse} = x - est\_b\) 得到的粗漫反射分量做像素级微调。
- **在模型中的位置**：YFFC 的最终输出阶段。

**内部结构**：

```python
refine = Conv2d(2→16) + GELU + Conv2d(16→16) + GELU + Conv2d(16→1) + Tanh
delta = refine(cat[coarse, original])  # 值域 [-1, 1]
output = clamp(coarse + 0.08 * delta, 0, 1)
```

**设计意义**：硬减法 \(x - b\) 虽然物理上精确，但 \(b\) 的预测误差会直接传递到漫反射分量。残差精修头接受粗预测和原始输入两路输入，输出一个微小的修正量（通过 Tanh 限制在 \([-1, 1]\) 范围，再乘以 scale=0.08 限制实际修正幅度在 \(\pm 0.08\) 内）。这种"先粗后精"的策略保证了物理一致性的同时给末端细节保留更大的可学习修正空间。

---

### 4.9 多尺度卷积块

#### MultiscaleConvBlock

- **位置**：MainDecoder 瓶颈处（FFC 之后、Level 1 之前）
- **结构**：三路并行——`1×1→7×7`（96→192ch）/ `3×3`（256→192ch）/ `1×1→3×3→3×3`（256→256→192ch）
- **输出**：三路拼接得到 576ch

#### FeatureExtrationLayersB

- **位置**：MainDecoder Level 2 与 Level 3 之间
- **结构**：四路并行——`1×1→7×7`（128→192ch）/ `1×1→3×3`（128→192ch）/ `1×1→3×3`（128→128ch）/ `1×1→3×3→3×3`（128→128→128ch）
- **输出**：四路拼接得到 640ch

#### TransposedConvBlock（上采样块）

- **结构**：三路 resize-conv，上采样统一使用 `Upsample(scale_factor=2, mode='bilinear')`，随后接 3×3/5×5/7×7 卷积，拼接后输出 3×out\_ch
- **作用**：用确定性插值先完成空间放大，再用多尺度卷积补充局部重建能力，降低转置卷积常见的棋盘格伪影风险

#### TransposedConvBlock\_same（同尺寸块）

- **结构**：三路 3×3 卷积（stride=1），拼接后输出 3×out\_ch
- **作用**：用于 Level 1（不需要上采样时）的特征扩展

---

### 4.10 网络基础构件

#### Conv2D\_BN\_activa

通用的 Conv/TransposedConv + BN + Activation 组合块。当前主解码器的 `TransposedConvBlock` 已改为 resize-conv，不再依赖该模块的转置卷积分支；梯度分支等历史结构仍可使用 `transpose=True` 配置：

- `transpose=True/False`：切换标准卷积和转置卷积
- `if_bn=True/False`：是否使用 BatchNorm
- `activation="relu"/None`：激活函数选择
- `dilation`：支持空洞卷积
- `initializer="truncated_norm"`：可选截断正态初始化

该模块被网络各处大量使用，是所有卷积层的统一封装。

---

## 5. YFFC 主网络前向传播

`YFFC` 类是模型的最顶层容器，其 `forward` 方法展示了完整的数据流动：

```python
class YFFC(nn.Module):
    def __init__(self):
        self.encoder = Encoder()
        self.gradient_decoder = GradientDecoder()
        self.enhancement_layers = EnhancementLayers()
        self.main_decoder = MainDecoder(in_channels=256)
        self.refinement_head = ResidualRefinementHead(scale=0.08)  # [P1]

    def forward(self, x):
        # 第一步：双编码器特征提取
        c_feat = self.encoder(x)
        # c_feat = (x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c)

        # 第二步：梯度解码 + 中间特征提取
        g_feat = self.gradient_decoder(c_feat[:-1], c_feat[-1])
        # g_feat = (x_tbe_1, x_tbe_2, x_tbe_3, x_tbe_4, x_g)

        # 第三步：增强层频域精炼
        enh = self.enhancement_layers(g_feat[:-1])
        # enh = (x_enhanced_1, x_enhanced_2, x_enhanced_3, x_enhanced_4)

        # 第四步：主解码器重建镜面反射分量
        # 注意：Encoder 内部已经执行 CrossScaleTopDownRefinement。
        est_b = self.main_decoder(c_feat, enh, g_feat[-1], original_input=x)

        # 第五步：硬减法 + [P1] 残差精修
        coarse = clamp(x - est_b, 0, 1)
        pred_diffuse = self.refinement_head(coarse, x)

        # 第六步：跨尺度一致性辅助标量，仅训练损失使用
        cs_loss = cross_scale_consistency_loss([c_feat[1], c_feat[2], c_feat[3]])

        return {"r": pred_diffuse, "g": g_feat[-1], "b": est_b, "cs": cs_loss}
```

---

## 6. 数据在网络中的完整流动路径

以 128×160 单通道输入（即 H=128, W=160）为例：

### 第一阶段：双编码器特征提取（EncoderSwin）

```
➡️ 初始输入: x [B, 1, 128, 160]

⬇️ Encoder
    │
    ├─> ResNet50 分支:
    │   ├─ stem_conv (1→64, stride=2) + BN + ReLU
    │   │  └─ x_bb_1: [B, 64, 64, 80]              ==> (保存)
    │   ├─ stem_pool (MaxPool stride=2)
    │   ├─ layer1 (64→256) → reduce_l1 (256→128)
    │   │  └─ x_bb_2: [B, 128, 32, 40]              ==> (保存)
    │   ├─ layer2 (256→512) → reduce_l2 (512→256)
    │   │  └─ x_bb_3: [B, 256, 16, 20]              ==> (保存)
    │   ├─ layer3 (512→1024) → reduce_l3 (1024→512)
    │   ├─ layer4 (dilation=2, 1024→2048) → reduce_l4 (2048→512)
    │   └─ fuse_l34: cat([l3_512, l4_512]) → Conv(1024→512)
    │      └─ x_bb_4: [B, 512, 8, 10]               ==> (保存)
    │
    ├─> Swin-Tiny 分支 (并行):
    │   ├─ 输入保持 [B, 1, 128, 160]，不 resize 到 224×224，保留条纹相位与周期关系
    │   ├─ 单通道 patch embedding: Conv2d(1→96, kernel=4, stride=4)
    │   ├─ torchvision Swin 内部为 NHWC，取出后统一转 NCHW
    │   ├─ stage H/4  → swin_4:  [B, 96,  32, 40]
    │   ├─ stage H/8  → swin_8:  [B, 192, 16, 20]
    │   └─ stage H/16 → swin_16: [B, 384, 8, 10]
    │
    ├─> H/4 AdaptiveSkipFusion:
    │   ├─ swin_4 经 Conv3×3+BN+GELU 投影到 128 通道
    │   ├─ gate = sigmoid(gate_net(cat[x_bb_2, swin_4_proj]))
    │   └─ x_bb_2 = x_bb_2 + beta_h4 * gate * swin_4_proj
    │
    ├─> H/8 AdaptiveSkipFusion:
    │   ├─ swin_8 经 Conv3×3+BN+GELU 投影到 256 通道
    │   ├─ gate = sigmoid(gate_net(cat[x_bb_3, swin_8_proj]))
    │   └─ x_bb_3 = x_bb_3 + beta_h8 * gate * swin_8_proj
    │
    ├─> H/16 CrossModalFusion（双向交叉注意力）:
    │   ├─ swin_16_proj: Conv1×1+BN+GELU+Conv3×3+BN+GELU → [B, 512, 8, 10]
    │   ├─ CNN→Swin: x_bb_4 查询 Swin 窗口上下文 (sr_ratio=2)
    │   ├─ Swin→CNN: Swin 查询 ResNet 局部细节
    │   ├─ 门控融合: gate * c2ctx + (1-gate) * ctx2c
    │   └─ 最终投影: 3×3 Conv + BN + GELU → [B, 512, 8, 10]
    │
    ├─> cba_after_backbone: Conv(512→256) + BN + ReLU
    │   └─ [B, 256, 8, 10]
    │
    ├─> FFC 瓶颈层:
    │   ├─ 按 ratio=0.5 分为 L(128ch) / G(128ch)
    │   ├─ FFC_BN_ACT (L2L + G2L + L2G + G2G 四路交互)
    │   └─ cat + 残差连接 → [B, 256, 8, 10]
    │
    └─> HighFrequencyEnhancement:
        ├─ Laplacian 高频提取
        ├─ 可学习多尺度分支 (3×3 + 3×3 dilation=2)
        ├─ 融合 (1×1 + 3×3) + SE 通道注意力
        └─ 残差连接 → x_c: [B, 256, 8, 10]         ==> (保存)
```

### 第二阶段：X 形分叉

```
左分支: (x_bb_1..4, x_c) → GradientDecoder → EnhancementLayers
右分支: (x_bb_1..4, x_c) + 增强特征 + 梯度图 + 原始输入 → MainDecoder
```

### 第三阶段：左分支处理

```
⬇️ GradientDecoder
    │
    ├─ Level 1 (8×10):
    │   ├─ fuse_global_context: cat(x_bb_4, x_c) → Conv(768→512)
    │   ├─ 7×7 Conv → 1×1 Conv → 3×3 Conv → 256ch
    │   ├─ ffc_grad_h16 (ratio=0.25) + 残差
    │   └─ x_to_be_enhanced_1: [B, 256, 8, 10]      ==> (→ EnhancementLayers)
    │
    ├─ Level 2 (16×20):
    │   ├─ cat(x_tbe_1, x_bb_4) → Conv(256+512→128)
    │   ├─ ffc_grad_h8 (ratio=0.25) + 残差
    │   └─ x_to_be_enhanced_2: [B, 128, 16, 20]     ==> (→ EnhancementLayers)
    │
    ├─ Level 3 (32×40):
    │   ├─ cat(x_tbe_2, x_bb_3) → Conv(128+256→64)
    │   ├─ ffc_grad_h4 (ratio=0.25) + 残差
    │   └─ x_to_be_enhanced_3: [B, 64, 32, 40]      ==> (→ EnhancementLayers)
    │
    ├─ Level 4 (64×80):
    │   ├─ cat(x_tbe_3, x_bb_2) → Conv(64+128→32)
    │   └─ x_to_be_enhanced_4: [B, 32, 64, 80]      ==> (→ EnhancementLayers)
    │
    └─ Level 5 (128×160):
        ├─ cat(x_tbe_4, x_bb_1) → TransposedConv(32+64→64)
        ├─ 梯度头分支:
        │   ├─ Conv(64→1, 无BN无激活) → Sigmoid
        │   └─ x_g: [B, 1, 128, 160]                ==> (→ MainDecoder / Loss)
        └─ [V3.1][S-LIK] 高光概率头分支（spec_lik_head）:
            ├─ spec_lik_conv_a: Conv(64→32) + GN + GELU
            ├─ spec_lik_conv_b: Conv(32→16) + GN + GELU
            ├─ spec_lik_conv_c: Conv(16→1)            // weight=0, bias=−2.0
            └─ Sigmoid → x_s: [B, 1, 128, 160]        ==> (→ EnhancementLayers / MainDecoder / Loss)

⬇️ EnhancementLayers (频谱分解精炼)
    ├─ x_tbe_1(256ch) → _SpectralDecompositionUnit(g_map=x_g↓, s_map=x_s↓, input↓) → x_enh_1: [B, 128, 8, 10]
    ├─ x_tbe_2(128ch) → _SpectralDecompositionUnit(g_map=x_g↓, s_map=x_s↓, input↓) → x_enh_2: [B, 64, 16, 20]
    ├─ x_tbe_3(64ch)  → _SpectralDecompositionUnit(g_map=x_g↓, s_map=x_s↓, input↓) → x_enh_3: [B, 32, 32, 40]
    └─ x_tbe_4(32ch)  → _SpectralDecompositionUnit(g_map=x_g↓, s_map=x_s↓, input↓) → x_enh_4: [B, 16, 64, 80]
```

`_SpectralDecompositionUnit` 的 `specular_gate` 由 V3 的 2 通道（`g + input`）升级为 V3.1 的 **3 通道**（`g + input + s`），spec_likelihood 作为显式监督下的高光语义信号注入高光分支门控。

### 第四阶段：右分支处理（MainDecoder — 核心重建）

```
⬇️ MainDecoder
    │
    ├─ 瓶颈处理:
    │   ├─ x_c chunk(2) → FFC1 → FFC2 → cat + 残差 → [B, 256, 8, 10]
    │   └─ MultiscaleConvBlock(7×7/3×3/3×3+3×3) → [B, 576, 8, 10]
    │
    ├─ Level 1 (8×10, 同尺寸):
    │   ├─ TransposedConvBlock_same(576→768ch)
    │   ├─ sa1: SpecularAwareAttention(768ch)         ← 高光感知
    │   ├─ att1: Attention_block(768, 512) → 加权 x_bb_4
    │   └─ cat([768, 512, 128]) → [B, 1408, 8, 10]
    │
    ├─ Level 2 (8×10 → 16×20):
    │   ├─ TransposedConvBlock(1408→384ch) resize-conv 上采样
    │   ├─ ffc_h8(ratio=0.25) + 残差
    │   ├─ [P10] 三路联合调制：
    │   │     x = x * (1 + scale_g·grad_gate_2(x_g)        ← 梯度门控
    │   │              + scale_j·joint_gate_2(cat(x_g, input, x_s))   ← [V3.1] 三通道联合门控
    │   │              + scale_s·spec_lik_gate_2(x_s))     ← [V3.1] 高光概率门控
    │   ├─ sa2: SpecularAwareAttention(384ch)
    │   ├─ att2: Attention_block(384, 256) → 加权 x_bb_3
    │   └─ cat([384, 256, 64]) → [B, 704, 16, 20]
    │
    ├─ Level 3 (16×20 → 32×40):
    │   ├─ FeatureExtrationLayersB(704→640ch)
    │   ├─ TransposedConvBlock(640→192ch) resize-conv 上采样
    │   ├─ ffc_h4(ratio=0.25) + 残差
    │   ├─ [P10] 三路联合调制：grad_gate_3 + joint_gate_3 + spec_lik_gate_3
    │   ├─ sa3: SpecularAwareAttention(192ch)
    │   ├─ att3: Attention_block(192, 128) → 加权 x_bb_2
    │   └─ cat([192, 128, 32]) → [B, 352, 32, 40]
    │
    ├─ Level 4 (32×40 → 64×80):
    │   ├─ TransposedConvBlock(352→96ch) resize-conv 上采样
    │   ├─ [P10] 三路联合调制：grad_gate_4 + joint_gate_4 + spec_lik_gate_4
    │   ├─ sa4: SpecularAwareAttention(96ch)
    │   ├─ att4: Attention_block(96, 64) → 加权 x_bb_1
    │   └─ cat([96, 64, 16]) → [B, 176, 64, 80]
    │
    └─ Level 5 (64×80 → 128×160):
        ├─ TransposedConvBlock(176→48ch) resize-conv 上采样
        ├─ cat([48, x_g 1ch]) → [B, 49, 128, 160]
        ├─ [P2] 扩展输出头:
        │   ├─ output_reduce: 1×1 Conv(49→32) → skip
        │   ├─ output_dilated: 3×3 Conv(dilation=2, 32→32) + skip
        │   ├─ output_conv3: 3×3 Conv(32→16)
        │   └─ output_conv4: 3×3 Conv(16→1) → Sigmoid
        └─ est_b: [B, 1, 128, 160]
```

### 第五阶段：残差精修与最终输出

```
coarse_diffuse = clamp(x - est_b, 0, 1)        # 硬减法

[P1] ResidualRefinementHead:
    cat([coarse, x]) → Conv(2→16) + GELU → Conv(16→16) + GELU → Conv(16→1) + Tanh
    delta ∈ [-1, 1]
    pred_diffuse = clamp(coarse + 0.08 * delta, 0, 1)

cs_loss = cross_scale_consistency_loss([H4_feat, H8_feat, H16_feat])

输出字典: {'r': pred_diffuse, 'g': x_g, 'b': est_b, 's': x_s, 'cs': cs_loss}    # 'V3.1' 'x_s' 来自 spec_lik_head
```

---

## 7. 损失函数体系

### 7.1 EnhancedCompositeLoss（增强复合损失 V3.1）

X-FFC V3.1 使用 `EnhancedCompositeLoss` 作为训练损失，它整合了 **14 个子损失项**（V3.1 在 V3 的 11 项基础上新增 `l_s_lik`、`l_s_cons`、`l_spec_peak`），从像素级、结构级、频域、感知、分割、跨尺度特征一致性、以及**高光概率辅助任务**等多个维度约束网络学习：

#### 子损失项详解

**1. SSIM 混合损失** \[P3\]

\[l_{ssim} = 0.8 \times (1 - \text{MS-SSIM}) + 0.2 \times (1 - \text{SSIM})\]

MS-SSIM 在 3 个尺度上评估结构相似性，对多尺度结构变化更敏感。SSIM 使用 \[P12\] 改进：`as_loss=True` 时 soft clamp 到 \([-0.5, 1.0]\)，保留梯度流但阻止极端值。

**2. 边缘加权 Charbonnier 损失** \[P13\]

\[l_{l1} = \text{mean}\left(\sqrt{(pred - target)^2 + \epsilon^2} \times w_{edge}\right)\]

用 Charbonnier 损失（V3.1 中 \(\epsilon = 10^{-3}\)，原 V3 为 \(10^{-6}\)）替代标准 L1。**V3.1 数值稳定性改进**：原 \(\epsilon = 10^{-6}\) 在 fp16 下接近下溢边缘，反向梯度可被 \(1/\epsilon\) 放大约 \(10^6\) 倍；提升至 \(10^{-3}\) 在精度损失可忽略的同时显著提高反向数值稳定性。`use_edge_weight=True` 路径下的 `sqrt eps` 也由 \(10^{-12}\) 提升至 \(10^{-6}\)。\(w_{edge}\) 为根据镜面反射分量梯度生成的边缘权重图（边缘区域权重为 3.0，平坦区域为 1.0），使网络在高光边缘区域承受更大的重建误差惩罚。

**3. 频域损失** \[P4\]\[V3.1 数值稳定性修复\]

\[l_{freq} = \text{L1}(|\text{FFT}(pred)|, |\text{FFT}(gt)|) + 0.25 \times \frac{\sum |\:|p|\cdot|g| - \text{Re}(p \cdot \bar{g})\:| \cdot M_E}{\sum M_E}\]

**V3.1 关键稳定性升级**：原 V3 相位项 \((1 - \cos(\theta_p - \theta_g)) = 1 - \text{Re}(p\bar{g}) / (|p||g|)\) 在反向时会出现 \(1 / (|p|\,|g|)^2\) 量级的爆炸性梯度（频谱低能量频点处尤甚），这是早期版本 Epoch 3 后出现大量 `grad NaN/Inf` 的核心根因之一。V3.1 改用 "能量加权的准相位差" \(|\,|p|\,|g| - \text{Re}(p \bar g)\,|\)（其中 \(\text{Re}(p\bar g) \leq |p|\,|g|\) 恒成立，等号当且仅当两者同相），完全避开除法，反向稳定；同时用 GT 频谱能量构造 `detach` 的能量掩码 \(M_E = \mathbb{1}_{|g| > 0.01 \cdot \overline{|g|}}\)，屏蔽几乎为零的高频。

**4. VGG16 感知损失**

使用冻结的 VGG16（ImageNet 预训练）的 relu1\_2、relu2\_2、relu3\_3 三层特征计算 L1 距离，权重分别为 1.0/0.5/0.25。感知损失从人类视觉感知的角度约束重建质量，弥补像素级损失对结构感知的不足。

**5. 镜面分量直接监督损失**

\[l_{spec} = \text{Charbonnier}(est\_b, gt\_spec) + 0.5 \times (1 - \text{SSIM}(est\_b, gt\_spec))\]

对镜面反射分量 \(est\_b\) 直接施加 Charbonnier + SSIM 联合监督，确保网络不仅在漫反射输出端表现好，在镜面反射的预测精度上也有保障。

**6. 输入梯度分支监督损失**

\[l_{grad} = 1 - \text{SSIM}(pred\_grad, gt\_grad)\]

对梯度预测辅助任务的直接监督。**V3.1 中梯度真值定义**：由输入图像 \(I\)（而非漫反射 \(I_d\)）的 Sobel 梯度幅值计算，并按样本归一化到 \([0,1]\)。因此 `g` 的学习目标是原始高反光输入中的边缘、条纹突变和高光边界。Sobel 只用于生成训练标签；推理输出 `pred_grad` 由 GradientDecoder 预测。

**6b. 输入梯度边缘加权 L1**（`l_grad_edge`）

\[l_{grad\_edge} = \text{mean}\left(\sqrt{(pred\_grad - gt\_grad)^2 + 10^{-6}} \times w_{grad} \times (1 + \gamma_{hp} \cdot p_{hp})\right)\]

在 `gt_gradient` 强响应区加权，并叠加高光软先验 \(p_{hp}\)（由输入亮度通过 sigmoid 计算）做额外增强，使 `pred_grad` 在高光边缘处更准确。

**6c. 梯度-镜面一致性损失**（`l_grad_spec`）\[V3.1 数值稳定性修复\]

\[l_{grad\_spec} = \text{mean}\big(\,|pred\_grad - \text{Sobel}_\text{norm}(\text{detach}(est\_b))| \cdot p_{hp}\,\big)\]

约束 `pred_grad` 与"基于 `est_b` 计算的 Sobel 边缘"在高光区域一致。**V3.1 关键修复**：(1) `est_b` 端必须 `.detach()`，否则反向梯度经过 `sobel + normalize_per_sample` 的 quantile 归一化分母会产生 \(1 / \text{scale}\) 量级的爆炸性梯度；(2) 原版 `sum / hp.sum().clamp_min(1e-6)` 在 batch 内高光极少时会把 loss 放大到 \(10^6\) 量级，改为直接 `mean`，量纲与其他损失一致。

**7. SpecularFocalLoss（高光焦点损失）**

\[l_{focal} = \text{mean}\left(|pred - gt| \times w_{spec} \times (1 + |pred - gt|)^\gamma\right)\]

其中 \(w_{spec} = 1 + (5-1) \times \sigma(30 \times (gt - 0.05))\)。对高亮像素（gt > threshold）给予更大权重（最高 5×），同时 focal 机制对误差大的像素额外加权（\(\gamma=2\)），聚焦网络在难预测的高光区域。

**8. SpecularDiceLoss（高光 Dice 损失）**

\[l_{dice} = 1 - \frac{2 \times |pred\_mask \cap gt\_mask| + \epsilon}{|pred\_mask| + |gt\_mask| + \epsilon}\]

从图像分割的视角约束高光区域的重叠度，pred\_mask 和 gt\_mask 通过软阈值化（Sigmoid）获得。Dice 损失对前景/背景不平衡具有天然的鲁棒性，适合高光区域通常只占图像小部分的场景。

**9. GradientConsistencyLoss（梯度一致性损失）**\[V3.1 数值稳定性修复\]

\[l_{gc} = \text{L1}(|\nabla pred|, |\nabla gt|) + 0.3 \times \frac{\sum (1 - \cos\theta) \times mask}{\max(\sum mask,\,1)}\]

使用 Sobel 算子计算预测和真值的空间梯度，同时约束梯度幅值（L1 距离）和梯度方向（余弦距离，仅在显著梯度区域计算）的一致性。**V3.1 数值稳定性**：`sqrt(... + eps)` 中 \(\epsilon\) 由 \(10^{-8}\) 提升至 \(10^{-4}\)；`cos = ... / (pm·gm + eps)` 的 \(\epsilon\) 同步提升；`mask.sum() + 1e-8` 改为 `mask.sum().clamp_min(1.0)`。

**10. OHEM 在线难样本挖掘**

当 `use_ohem=True` 时，SpecularFocalLoss 被替换为 OHEM 版本：先计算所有像素的误差，选出误差最大的 top-30% 像素，对这些"难样本"像素额外加权。最终损失为 `0.5 * base_loss + 0.5 * hard_loss`。归一化分母由 \(\sum \text{mask} + 10^{-8}\) 改为 \(\max(\sum \text{mask}, 1)\)。

**11. CrossScaleConsistencyLoss（跨尺度一致性辅助损失）**

\[l_{cs} = \text{mean}\left(\text{L1}(R_{h4}, up(R_{h8})) + \text{L1}(R_{h4}, up(R_{h16}))\right)\]

`YFFC.forward()` 会从 H/4、H/8、H/16 融合特征中构造归一化响应图，并统一上采样到 H/4 尺度后计算一致性约束。该项不改变推理输出的主图像路径，只作为训练期辅助标量 `cs` 传入 `EnhancedCompositeLoss`，用于抑制不同尺度对同一结构区域给出相互冲突的响应。

**12. 高光概率监督损失（`l_s_lik`，V3.1 新增）**\[NEW S1\]\[P1-UPGRADE\]

\[l_{s\_lik} = \text{BCE}(\text{clamp}(pred\_s\_lik, 10^{-4}, 1{-}10^{-4}),\;gt\_spec) + 0.3 \times (1 - \text{SSIM}(pred\_s\_lik, gt\_spec))\]

对 `spec_lik_head` 的输出 `pred_s_lik` 直接监督，目标为 `gt_specular`。**为何使用 BCE 而非 Charbonnier**：`pred_s_lik` 是 sigmoid 输出（已落在 \([0,1]\)），Charbonnier 的 \(\sqrt{(p-t)^2 + \epsilon^2}\) 形式对 sigmoid 输出在中段 \(0.3 \sim 0.7\) 的反向梯度被 \(\sigma'(x) = \sigma(x)(1-\sigma(x))\) 进一步压扁，导致预测停滞在中等概率（训练监控中观察到的"位置对但幅度不足"现象的数学根因）；BCE 损失 \(-t\log p - (1-t)\log(1-p)\) 对 sigmoid 中段反向梯度恒为 \(p - t\)，不被 \(\sigma'\) 压扁，是 sigmoid 输出的理论最优损失，直接驱动 `pred_s_lik` 在高光区域跨过 0.5 阈值。`F.binary_cross_entropy` 对 `pred=0` 或 `pred=1` 时 \(\log(0) = -\infty\)，前后加一道 `clamp(1e-4, 1-1e-4)` 防御。

**13. 高光概率双向一致性损失（`l_s_cons`，V3.1 新增）**\[NEW S2\]\[P3-UPGRADE\]

\[l_{s\_cons} = \text{mean}\big(|pred\_s\_lik - \text{detach}(est\_b)| \cdot M_s\big) + \text{mean}\big(|est\_b - \text{detach}(pred\_s\_lik)| \cdot M_s\big)\]

其中软掩码 \(M_s = \text{detach}(gt\_spec)\)。**双向一致性**让 `pred_s_lik` 与 `est_b` 在高光区结构对齐：
- 方向 1（`pred_s_lik` 学 `est_b`）：保留原 V3.1 单向设计；
- 方向 2（`est_b` 学 `pred_s_lik.detach()`）：因为 `pred_s_lik` 已被 `gt_specular` 直接、强力监督（BCE），让 `est_b` 也从这个"软标签"学习相当于给主解码器额外的高光语义信号。

两个方向都用 `.detach()` 防止互相拉扯死循环。`.mean()` 替代原 `sum / mask.sum().clamp_min(1e-6)`，避免低高光 batch 把 loss 放大到 \(10^6\) 量级。

**14. 高光峰值欠预测惩罚（`l_spec_peak`，V3.1 新增）**\[NEW S3\]

\[l_{spec\_peak} = \text{mean}\big(\text{ReLU}(gt\_spec - est\_b) \cdot \mathbb{1}_{gt\_spec > 0.3}\big)\]

**单侧损失**，仅在 `gt_spec > 0.3` 区域、且 `est_b < gt_spec` 时计入。设计动机：训练监控发现现有 `l_focal / l_dice / l_spec` 虽然在强高光区有较大权重，但都是双侧 L1/MSE 形式，既惩罚欠预测也惩罚过预测，平均下来"保守预测中间值"是局部最优。`l_spec_peak` 给主解码器一个明确的"在强高光区敢于拉满"的方向性信号，且不会反向惩罚"在弱光区敢于压零"。

#### 总损失计算（V3.1）

\[\begin{aligned}
l_{total} = \;& \lambda_{ssim} l_{ssim} + \lambda_{l1} l_{l1} + \lambda_{freq} l_{freq} + \lambda_{percep} l_{percep} \\
       & + \lambda_{spec} l_{spec} + \lambda_{grad} l_{grad} + \lambda_{grad\_edge} l_{grad\_edge} + \lambda_{grad\_spec} l_{grad\_spec} \\
       & + \lambda_{focal} l_{focal} + \lambda_{dice} l_{dice} + \lambda_{gc} l_{gc} + \lambda_{cs} l_{cs} \\
       & + \lambda_{s\_lik} l_{s\_lik} + \lambda_{s\_cons} l_{s\_cons} + \lambda_{spec\_peak} l_{spec\_peak}
\end{aligned}\]

所有项 \(\geq 0\)，\(\lambda_{stat}\) 默认为 0（\[P5\] 移除冗余统计损失）。不使用可学习权重（这是导致早期版本出现负数损失的根源），所有权重完全由 \(\lambda\_*\) 和动态调度器控制。

### 7.2 ImprovedWeightScheduler（动态损失权重调度，V3.1 双速率）

训练过程中各损失项的权重并非固定，而是根据训练进度动态调整。V3.1 引入**双速率调度策略**：

**双速率核心机制**：
- **主权重系列**（\(p_{clamped} = \min(p / 0.6, 1.0)\)）：60% 进度处截断，覆盖 SSIM/L1/freq/perceptual/grad 等主任务损失，避免后期优化目标漂移与学习率衰减冲突。
- **spec_likelihood 系列**（\(p_{fast} = \min(p / 0.4, 1.0)\)）：40% 进度处截断，覆盖 `lambda_s_lik` / `lambda_s_cons` / `lambda_spec_peak`。**加速理由**：BCE 对 sigmoid 输出梯度不饱和（不同于旧版 Charbonnier 在中段被 \(\sigma'\) 压扁），加大权重可直接驱动 `pred_s_lik` 在高光区域跨过 0.5 阈值，无需缓慢爬升。

**核心策略**（修复版）：
- **早期（0\~30%）**：以 L1/Charbonnier 为主快速收敛，频域和感知损失权重较低
- **中期（30\~60%）**：逐步加大 focal/dice/freq 的权重，强化细节和频域约束
- **后期（60\~100%）**：主权重冻结在平台值，不再变化
- **spec_lik 系列**：0\~40% 快速爬升到最终值，40\~100% 保持高位

主要权重的变化范围：

| 子损失 | 初始值 | 最终值 | 速率截断点 |
|--------|-------|-------|----------|
| \(\lambda_{ssim}\) | 1.0 | 1.0（不变） | — |
| \(\lambda_{l1}\) | 1.0 | 0.5（下降） | 60% |
| \(\lambda_{freq}\) | 0.05 | 0.15（上升） | 60% |
| \(\lambda_{focal}\) | 0.3 | 0.8（上升） | 60% |
| \(\lambda_{dice}\) | 0.1 | 0.3（上升） | 60% |
| \(\lambda_{gc}\) | 0.05 | 0.2（上升） | 60% |
| \(\lambda_{cs}\) | 0.02 | 0.05（上升） | 60% |
| \(\lambda_{grad}\) | 0.5 | 0.3（下降） | 60% |
| \(\lambda_{grad\_edge}\) | 0.3 | 0.8（上升） | 60% |
| **\(\lambda_{s\_lik}\)** *(V3.1)* | **0.3** | **0.8（上升）** | **40%** |
| **\(\lambda_{s\_cons}\)** *(V3.1)* | **0.02** | **0.08（上升）** | **40%** |
| **\(\lambda_{spec\_peak}\)** *(V3.1)* | **0.1** | **0.5（上升）** | **40%** |

---

## 8. 训练系统详解

### 8.1 分层学习率优化器

`build_optimizer_v2` 将模型参数分为 5 组，各组使用不同的学习率：

| 参数组 | 包含模块 | 学习率倍率 | 设计理由 |
|--------|---------|-----------|---------|
| encoder\_pretrained | ResNet50 stem/layers, Swin backbone | 0.1× | 预训练权重已包含良好特征，微调即可 |
| fusion | 编码器融合层等其他参数 | 0.5× | 融合层需要适度调整 |
| decoders | GradientDecoder, MainDecoder, EnhancementLayers | 1.0× | 解码器从头训练，需要最大学习率 |
| ffc | 所有 FFC 模块 | 0.5× | FFC 含频域操作，过大 LR 易不稳定 |
| new\_modules | SpecularAwareAttention (sa1-4), HighFrequencyEnhancement | 1.0× | 新增模块从头训练 |

优化器使用 AdamW（weight\_decay=1e-4）。

### 8.2 学习率调度器 \[P6-FIX\]

采用 Warmup + 余弦退火组合调度（SequentialLR），分两个阶段：

**阶段一：线性 Warmup**（前 500 迭代或 3 个 epoch 的迭代数，取较小值）
从 `0.01 × base_lr` 线性增加到 `base_lr`，防止训练初期大学习率导致的梯度爆炸。

**阶段二：余弦退火（无重启）**
关键修复：原版使用 `CosineAnnealingLR`，当训练超过 `T_max` 迭代后会周期性回升。修复后使用 `LambdaLR` + 自定义余弦函数，超过总迭代数后 LR 钳位在 `eta_min=1e-6`，不再回弹。

代码中还包含 LR 回弹检测：如果当前 epoch 的 LR 比上一 epoch 增加超过 50% 且绝对值大于 1e-5，会输出警告。

### 8.3 EMA（指数移动平均）

```python
shadow[name] = decay * shadow[name] + (1 - decay) * param.data
```

验证时使用 EMA 权重（`ema.apply_shadow()`），训练时恢复原始权重（`ema.restore()`）。decay=0.9995（\[P16\] 调高以增加平滑力度），使推理时的模型参数更加稳定。

### 8.4 BN → GN 替换

通过 `replace_bn_with_gn` 递归将所有 `BatchNorm2d` 替换为 `GroupNorm`（默认 8 组）。这对于小 batch size（如 2-4）至关重要——BatchNorm 在小 batch 下的统计量估计不准确，会导致训练不稳定，而 GroupNorm 不依赖 batch 维度的统计。

### 8.5 训练稳定性保障措施

1. **梯度累积**（accumulation\_steps=4）：等效 batch size = 4×4 = 16
2. **梯度裁剪**（max\_norm=0.5）：防止梯度爆炸
3. **混合精度训练**（AMP）：使用 `torch.amp.autocast('cuda')` + `GradScaler`
4. **NaN/Inf 保护**：检测到非有限损失时跳过该 batch
5. **损失尖峰保护**：单批损失超过历史均值的 10 倍时跳过该 batch
6. **\[V3.1 数值加固\] 损失函数反向稳定性专项修复**：
   - `frequency_loss` 相位项改用能量加权的"准相位差"，无除法（V3 中 \(1 / (|p|\,|g|)^2\) 的反向爆炸已根除）
   - `l_grad_spec` / `l_s_cons` 等所有 `sum / mask.sum().clamp_min(1e-6)` 模式改为 `.mean()` 或 `clamp_min(1.0)`
   - `est_b_grad` 在 `l_grad_spec` 中必须 `.detach()`（避免反向经过 sobel + quantile 归一化产生 \(1 / \text{scale}\) 爆炸）
   - `normalize_per_sample.scale.clamp_min` 由 \(10^{-8}\) 提升至 \(10^{-3}\)
   - 所有 `sqrt(... + 1e-12)` 改为 `sqrt(... + 1e-6)`，`charbonnier_loss` 的 \(\epsilon\) 由 \(10^{-6}\) 提升至 \(10^{-3}\)
   - `GradientConsistencyLoss` / `compute_edge_weight_map` / `sobel_magnitude` 的 sqrt eps 由 \(10^{-8}\) / \(0\) 提升至 \(10^{-4}\) / \(10^{-6}\)
7. **\[V3.1 诊断增强\] NaN/Inf 跳过时打印子损失明细**：`_format_loss_diag()` 在跳过 step 时按绝对值降序打印各子损失项的当前值（NaN/Inf 标记为字符串），可直接定位是哪个具体损失项导致爆炸，无需事后猜测。
8. **\[V3.1 训练监控\] spec_likelihood 三联面板**：可视化函数 `visualize_sample_during_training` 输出 3×4 布局，第三行专门展示 `pred_s_lik` / `|SpecLik - GT_Spec|` / `|SpecLik - Pred_Spec|` / `SpecLik > 0.5 Binary` 四张诊断图，直接观察辅助头训练状态。

### 8.6 数据增强流水线

**EnhancedPairedTransform**：

1. **随机水平翻转**（p=0.5）
2. **随机垂直翻转**（p=0.5）
3. **随机旋转**（p=0.3）：正方形图像支持 90°/180°/270°，非正方形仅支持 180°
4. **随机裁剪+缩放**（p=0.5）：裁剪比例 0.7\~1.0，裁剪后插值回原始尺寸
5. **亮度增强**（0.9\~1.1 倍）：同步应用于 input、target 和 gt\_specular
6. **HighlightAugmentation（合成高光增强）**（p=0.5）：随机叠加 1-6 个高斯 blob 高光（半径 3\~min\_side/20，强度 0.2\~0.7），同步更新 input 和 gt\_specular
7. **梯度真值计算**：对增强后的 input 计算 Sobel 梯度幅值，并按样本归一化到 \([0,1]\)，得到输入图像梯度图。

### 8.7 多指标最优模型保存

训练过程中同时追踪三种最优模型：

1. **综合最优**（best\_model.pth）：`composite = 0.7 × val_ssim + 0.3 × 1/(1+val_loss)`
2. **SSIM 最优**（best\_model\_best\_ssim.pth）：验证集 SSIM 最高
3. **Loss 最优**（best\_model\_best\_loss.pth）：验证集总损失最低

早停基于综合评分（默认关闭），patience=30。

### 8.8 TTA（测试时增强）\[P15\]

`_tta_forward` 实现 4 次翻转推理（原图、水平翻转、垂直翻转、双翻转）取平均，提升推理精度。训练过程中的验证阶段禁用 TTA（\[P15-FIX\]），仅在最终评估/推理时使用，避免验证指标突变影响模型选择。

### 8.9 推理流水线（process_images.py）\[V3.1 扩展\]

推理脚本 `process_images.py` 在 V3.1 中同步扩展，以支持 spec_likelihood 输出。核心流程：

1. **`load_model`**：加载训练好的 checkpoint，通过 `_load_state_dict_flexible` 容忍结构差异（V3.1 模型多了 `spec_lik_head`、`spec_lik_gate_2/3/4`，从 V3 旧 checkpoint 启动时这些层使用 `__init__` 默认初始化）。

2. **`sliding_window_inference`**：滑窗推理（patch_size=`(128,160)`，overlap=64）。返回字典：
   ```python
   {'r': full_diffuse,         # [B, 1, H, W]
    'b': full_specular,        # [B, 1, H, W]
    'g': full_gradient,        # [B, 1, H, W]
    's': full_spec_likelihood} # [B, 1, H, W]  (V3.1 新增，若模型不输出 's' 则该键不存在)
   ```
   `'s'` 字段使用与 `'r'/'b'/'g'` 同样的高斯权重累加 blend，确保滑窗拼接边缘无可见接缝。`has_spec_likelihood` 标志位用于兼容旧 V3 checkpoint。

3. **`process_single_image`**：将每张图的输出保存为单独文件：
   - `*_diffuse.png` / `*_diffuse_16bit.png`：漫反射主输出
   - `*_specular.png` / `*_specular_16bit.png`：镜面反射
   - `*_gradient.png` / `*_gradient_16bit.png`：梯度图
   - `*_spec_likelihood.png` / `*_spec_likelihood_16bit.png`：**V3.1 新增**，高光软概率热力图

4. **`visualize_results`**：可视化输出由 V3 的 2×2 布局扩展为 V3.1 的 2×3 布局，新增一个 `Pred SpecLikelihood`（hot colormap）面板，用户可直接看到模型对高光区域的"置信度图"。

5. **数值后处理**：所有 4 个输出通道在保存前 `clamp(0, 1)`，16-bit 版本通过 `(x * 65535).round().astype(uint16)` 写入，便于下游做精细处理。

---

## 9. 代码结构总览

```
YFFC (X-FFC V3)
  │
  ├─ (encoder): Encoder                               // ResNet50 + Swin-Tiny 双编码器主干
  │   ├─ (stem_conv / stem_bn / stem_relu / stem_pool)    // ResNet50 stem (1→64ch)
  │   ├─ (layer1 ... layer4): ResNet50 Stages              // 含空洞卷积
  │   ├─ (reduce_l1 ... reduce_l4, fuse_l34)               // 通道降维
  │   ├─ (swin_branch): SwinEncoderBranch                  // ★ Swin-Tiny H/4,H/8,H/16
  │   │   └─ (swin): torchvision swin_t                    //   单通道 patch embedding
  │   ├─ (fuse_h4): AdaptiveSkipFusion                     // ★ H/4 skip 自适应融合
  │   │   ├─ (swin_proj): Conv3x3+BN+GELU                  //   96→128
  │   │   ├─ (gate_net): 1×1 gate                          //   逐位置/逐通道权重
  │   │   └─ (beta): learnable scalar                      //   初始 0
  │   ├─ (fuse_h8): AdaptiveSkipFusion                     // ★ H/8 skip 自适应融合
  │   │   ├─ (swin_proj): Conv3x3+BN+GELU                  //   192→256
  │   │   ├─ (gate_net): 1×1 gate
  │   │   └─ (beta): learnable scalar
  │   ├─ (swin16_proj): Conv1x1+BN+GELU+Conv3x3+BN+GELU    // 384→512
  │   ├─ (cross_modal_fusion): CrossModalFusion            // ★ H/16 交叉注意力融合
  │   │   ├─ (q_c2ctx / k_c2ctx / v_c2ctx / proj_c2ctx)    //   CNN→Swin 方向
  │   │   ├─ (q_ctx2c / k_ctx2c / v_ctx2c / proj_ctx2c)    //   Swin→CNN 方向
  │   │   ├─ (gate): 门控融合                               //   自适应平衡
  │   │   └─ (final_proj): 最终投影                         //   3×3 Conv
  │   ├─ (cba_after_backbone): Conv(512→256)               // 瓶颈降维
  │   ├─ (ffc_bottleneck): FFC_BN_ACT                      // FFC 频域处理
  │   └─ (hf_enhance): HighFrequencyEnhancement            // ★ 高频增强
  │       ├─ (lap_kernel): 固定 Laplacian 核
  │       ├─ (hf_branch): 可学习多尺度分支
  │       ├─ (fusion): 三路融合
  │       └─ (se): SE 通道注意力
  │
  ├─ (gradient_decoder): GradientDecoder                 // 梯度解码分支
  │   ├─ (fuse_global_context): Conv(768→512)
  │   ├─ (block_1_conv_1/2/3): 7×7 / 1×1 / 3×3
  │   ├─ (ffc_grad_h16/h8/h4): 3 × FFC_BN_ACT
  │   ├─ (block_2/3/4_conv): 降维卷积
  │   ├─ (block_2/3/4_conv_transpose): 转置卷积上采样
  │   ├─ (block_5_conv_1/2): 梯度头输出卷积
  │   ├─ (sigmoid_layer): 梯度头 Sigmoid
  │   └─ ★[V3.1] spec_lik_head 并联辅助头:
  │       ├─ (spec_lik_conv_a): Conv(64→32)+GN+GELU
  │       ├─ (spec_lik_conv_b): Conv(32→16)+GN+GELU
  │       ├─ (spec_lik_conv_c): Conv(16→1)        // weight=0, bias=−2.0
  │       └─ (spec_lik_sigmoid): Sigmoid          // → x_s ∈ [0,1]
  │
  ├─ (enhancement_layers): EnhancementLayers              // 频谱分解增强桥梁
  │   └─ (decomp_unit_1 ... _4): _SpectralDecompositionUnit
  │       ├─ (projection_branch): 7×7 卷积
  │       ├─ (specular_branch): 3×3 卷积
  │       ├─ (ffc_specular): FFC_BN_ACT
  │       ├─ (edge_gate): Conv+Sigmoid                    // 由 g_map 生成
  │       ├─ (specular_gate): Conv(★3ch→1)+Sigmoid        // ★[V3.1] 3 通道: g+input+s
  │       └─ (gate_conv): 联合门控融合
  │
  ├─ (main_decoder): MainDecoder                          // 主解码分支
  │   ├─ (FFC1, FFC2): FFC_BN_ACT                         // 瓶颈 FFC
  │   ├─ (multiscale_conv): MultiscaleConvBlock            // 多尺度卷积
  │   ├─ (transposed_block_1): TransposedConvBlock_same    // 同尺寸
  │   ├─ (transposed_block_2 ... _5): TransposedConvBlock  // resize-conv 上采样
  │   ├─ (feature_extraction_B): FeatureExtrationLayersB   // 四路特征提取
  │   ├─ (att1 ... att4): Attention_block                  // 注意力门控
  │   ├─ (sa1 ... sa4): SpecularAwareAttention             // ★ 高光感知注意力
  │   ├─ (grad_gate_2/3/4): Conv+Sigmoid                   // ★ [P10] 梯度门控
  │   ├─ (joint_gate_2/3/4): Conv(★3ch→1)+Sigmoid          // ★[V3.1] 三通道联合门控 (g+input+s)
  │   ├─ (spec_lik_gate_2/3/4): Conv(1→1)+Sigmoid          // ★[V3.1] 独立的高光概率门控
  │   ├─ (grad_gate_scale, joint_gate_scale,
  │   │   spec_lik_gate_scale): Parameter                 // 三个独立 scale 参数
  │   ├─ (ffc_h8, ffc_h4): FFC_BN_ACT                     // 中间层 FFC
  │   ├─ (output_reduce): 1×1 Conv                         // [P2] 扩展输出头
  │   ├─ (output_dilated): 3×3 dilation=2 Conv
  │   ├─ (output_skip): 1×1 Conv skip
  │   ├─ (output_conv3): 3×3 Conv
  │   ├─ (output_conv4): 3×3 Conv (→1ch)
  │   └─ (output_sigmoid): Sigmoid
  │
  └─ (refinement_head): ResidualRefinementHead             // ★ [P1] 残差精修
      └─ (refine): Conv(2→16)+GELU → Conv(16→16)+GELU → Conv(16→1)+Tanh
```

---

## 10. 全部改进点索引

### FFC 模块优化

| 编号 | 改进 | 位置 | 说明 |
|------|------|------|------|
| \[F1\] | GELU 替代 ReLU | FourierUnit | 保护频域相位信息 |
| \[F2\] | 频率 SE 注意力 | FourierUnit | 自适应频段选择 |
| \[F3\] | LFU 拆分 | SpectralTransform | 避免参数竞争 |
| \[F4\] | FU/LFU 门控融合 | SpectralTransform | 替代硬相加 |
| \[F5\] | 跨域门控融合 | FFC\_BN\_ACT | 防止频域噪声污染 |
| \[F6\]\[P11\] | Tukey 窗 | FourierUnit | 减少频谱泄漏 |

### Swin 编码器增强

| 编号 | 改进 | 位置 | 说明 |
|------|------|------|------|
| \[S1\] | 单通道 Swin-Tiny | SwinEncoderBranch | patch embedding 由 3 通道改为 1 通道 |
| \[S2\] | 取消 224×224 强制 resize | SwinEncoderBranch | 直接处理训练/推理 patch，保护结构光条纹相位关系 |
| \[S3\] | H/4、H/8、H/16 层级输出 | SwinEncoderBranch | NCHW 多尺度特征 |
| \[S4\] | 自适应 skip 融合 | AdaptiveSkipFusion | gate + beta 控制 Swin 注入 |
| \[S5\] | H/16 交叉注意力融合 | CrossModalFusion | ResNet 与 Swin 双向信息交互 |
| \[S6\] | Top-down 跨尺度回流 | CrossScaleTopDownRefinement | H/16 上下文受限反馈到 H/8/H/4 skip |

### 流水线与训练改进

| 编号 | 改进 | 位置 | 说明 |
|------|------|------|------|
| \[P1\] | 残差精修头 | ResidualRefinementHead | ±0.08 像素级修正 |
| \[P2\] | 扩展输出头 | MainDecoder | 4 层 + 膨胀 + skip |
| \[P3\] | MS-SSIM 混合 | EnhancedCompositeLoss | 多尺度结构评估 |
| \[P4\] | 频域相位损失 | EnhancedCompositeLoss | 相位一致性约束（V3.1 改为无除法能量加权版） |
| \[P5\] | 移除 stat 损失 | EnhancedCompositeLoss | 去除冗余项 |
| \[P6\] | LR 调度修复 | build\_scheduler | 防止余弦回弹 |
| \[P7\] | 亮度增强顺序 | EnhancedPairedTransform | 先算 specular 再增强 |
| \[P9\] | 多级阈值 | SpecularAwareAttention | 弱/中/强高光分级 |
| \[P10\] | 梯度门控注入 | MainDecoder | 边缘区域特征调制（V3.1 扩展为三路：grad + joint + spec_lik） |
| \[P12\] | SSIM soft clamp | SSIM | 保留梯度流 |
| \[P13\] | Charbonnier 损失 | EnhancedCompositeLoss | 零点光滑可导 |
| \[P14\] | 跨尺度一致性 | EnhancedCompositeLoss / YFFC.forward | H/4、H/8、H/16 响应图一致性约束 |
| \[P15\] | TTA（测试时增强） | \_tta\_forward | 4 翻转推理平均 |
| \[P16\] | EMA decay 调高 | train\_model | 0.9995 增加平滑 |

### V3.1 改进点（梯度分支任务化、高光自信度增强、数值稳定性）

| 编号 | 改进 | 位置 | 说明 |
|------|------|------|------|
| \[V3.1\] 输入梯度监督 | 监督目标改为 \(\nabla I\) | `compute_input_gradient_target`、`EnhancedPairedTransform` | 梯度分支学习输入图边缘（含高光+材质），非漫反射梯度 |
| \[S-LIK\] spec_lik_head 辅助头 | 高光软概率预测头 | `GradientDecoder` | 与梯度头并联，监督 `gt_specular`，初始化 bias=−2.0 |
| \[S-LIK\] 3 通道 specular_gate | spec_likelihood 注入 enhancement | `_SpectralDecompositionUnit` | gate 输入由 2 通道（g+input）升级为 3 通道（g+input+s）|
| \[S-LIK\] 3 通道 joint_gate | spec_likelihood 注入主解码 | `MainDecoder` level 2/3/4 | joint_gate 由 2 通道升级为 3 通道；额外新增独立的 spec_lik_gate |
| \[NEW S1\]\[P1-UPGRADE\] | BCE+SSIM 替代 Charbonnier+SSIM | `l_s_lik` | sigmoid 输出的理论最优损失，根治"位置对但幅度不足" |
| \[NEW S2\]\[P3-UPGRADE\] | 双向一致性损失 | `l_s_cons` | `est_b` 反过来从 `pred_s_lik.detach()` 学习，强化高光信号 |
| \[NEW S3\] | 高光峰值欠预测惩罚 | `l_spec_peak` | 单侧 `F.relu(gt - pred) * 1[gt > 0.3]`，鼓励主解码器拉满高光 |
| \[P2-UPGRADE\] | 双速率调度 | `ImprovedWeightScheduler` / `DynamicLossWeightScheduler` | spec_lik 系列在 40% 进度处截断；主权重 60% 截断 |
| \[NaN-Fix\] 频域损失反向稳定 | 能量加权准相位 | `frequency_loss` | 消除 \(1/(|p||g|)^2\) 反向爆炸 |
| \[NaN-Fix\] sum/sum 归一化加固 | `mean` 替代 `sum.clamp_min(1e-6)` | 多处损失 | 防低高光 batch 把 loss 放大 \(10^6\) 倍 |
| \[NaN-Fix\] `est_b_grad` detach | `l_grad_spec` | 避免反向经 sobel + quantile 归一化产生 \(1/\text{scale}\) 爆炸 |
| \[NaN-Fix\] 各 sqrt eps 提升 | 1e-12 → 1e-6；1e-8 → 1e-4；scale 1e-8 → 1e-3 | `charbonnier_loss` / `sobel_magnitude` / `normalize_per_sample` / `compute_edge_weight_map` / `GradientConsistencyLoss` | fp16/fp32 下都稳定 |
| \[NaN-Diag\] 子损失明细诊断 | `_format_loss_diag` | `train.py` | grad NaN 跳过时按值降序打印子损失，精确定位 |
| \[V3.1-INFER\] 推理输出 spec_likelihood | `*_spec_likelihood.png` | `process_images.py` | 滑窗推理累加 `'s'` 字段，保存附加可视化 |

---

## 11. 设计优势总结

1. **双编码器 + 多尺度 Swin 融合**：CNN 保留局部纹理细节，Swin-Tiny 提供窗口上下文和层级特征；H/4、H/8 通过 AdaptiveSkipFusion 温和增强 skip，H/16 通过 CrossModalFusion 实现强上下文交互，突破单一骨干网络的信息瓶颈。

2. **全链路频域处理**：优化的 FFC 模块（6 项改进）贯穿编码器瓶颈、两个解码器和增强层，在多个尺度上捕获全局频率信息，尤其适合处理具有周期性的条纹图案。

3. **高光专用感知通路**：HighFrequencyEnhancement（编码端）和 SpecularAwareAttention（解码端每一级）形成从编码到解码的完整高光感知通路，配合梯度门控注入 \[P10\]，三重机制确保网络在所有尺度上对高光区域保持高灵敏度。

4. **多任务协同学习**（V3.1 强化）：梯度预测辅助任务不仅提供结构化约束（梯度 SSIM 损失），其中间特征经 EnhancementLayers 精炼后直接参与主解码器的多源融合，梯度图还通过门控机制在 3 个尺度上调制主解码器特征。**V3.1 新增 spec_likelihood 任务化辅助头**，与梯度头共享 `block_5_conv_1` 特征但输出独立的高光软概率，作为 EnhancementLayers / MainDecoder 的第三调制通道。辅助任务与主任务深度耦合。

5. **精细化融合策略**（V3.1 升级）：注意力门控跳跃连接（每通道独立权重）、三源融合（自身 + 门控编码器特征 + 增强特征）、**梯度+联合+spec_likelihood 三路调制**——多层次、多来源的融合策略确保信息传递的高效性。

6. **残差精修范式** \[P1\]：先通过物理减法获得粗预测，再通过轻量级网络做微量修正（\(\pm 0.08\)），兼顾物理一致性和像素级精度。

7. **训练稳定性工程**（V3.1 加固）：分层学习率、EMA、GroupNorm、梯度累积/裁剪、混合精度、NaN/损失尖峰保护、无重启余弦退火、双速率（60%/40%）权重调度，**外加 V3.1 损失反向稳定性专项修复**（无除法相位损失、所有 sum/sum 归一化加固、`detach` 防爆点、sqrt eps 提升）和 **NaN/Inf 子损失明细诊断**——系统性的工程手段保障了大型网络的训练稳定性和收敛效率。

8. **高光自信度增强（V3.1 新增）**：通过 BCE 替代 Charbonnier 监督 sigmoid 输出（中段梯度不饱和）、双向一致性（主辅头互相学习）、单侧欠预测惩罚（鼓励敢于拉满），三重机制系统性地解决"位置对但幅度不足"问题，让主解码器在强高光区敢于自信预测。

---

## 12. 附录：主要超参数默认配置

| 参数 | 默认值 | 说明 |
|------|-------|------|
| EPOCHS | 400 | 训练总轮数 |
| BATCH\_SIZE | 4 | 批次大小 |
| BASE\_LR | 2e-4 | 基础学习率 |
| WEIGHT\_DECAY | 1e-4 | AdamW 权重衰减 |
| ACCUMULATION\_STEPS | 4 | 梯度累积（等效 batch=16） |
| GRAD\_CLIP\_VALUE | 0.5 | 梯度裁剪阈值 |
| IMAGE\_SIZE | (128, 160) | 输入分辨率 (H, W) |
| EMA\_DECAY | 0.9995 | EMA 衰减系数 |
| USE\_GN | True | BN → GN 替换 |
| USE\_OHEM | True | 在线难样本挖掘 |
| VIS\_EVERY\_N\_EPOCHS | 10 | 可视化间隔 |
| SAVE\_EVERY\_N\_EPOCHS | 20 | 检查点保存间隔 |
