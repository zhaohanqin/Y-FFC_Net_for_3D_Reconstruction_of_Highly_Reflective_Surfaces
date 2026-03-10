# X-FFC 模型深度解析与总结文档

---

## 1. 模型命名说明

本文档所描述的模型统一命名为 **X-FFC（X-shaped Fast Fourier Convolutional Network）**。

命名依据如下：该模型的编码器（EncoderV2）内部采用了 **ResNet50** 和 **ViT-B/16** 两条并行的特征提取路径，二者在瓶颈层处交汇融合。这一"双编码器"结构使得整体网络在拓扑形态上更接近字母"X"——上方为两条编码分支汇聚，下方为两条解码分支分叉。相较于原始 Y-FFC 仅使用单一 VGG16 主干的"Y"形结构，X-FFC 的双编码器设计赋予了网络更强大的特征提取能力，因此以"X"命名更为贴切。

---

## 2. 模型总体概述

**X-FFC** 是一个专为图像分解任务设计的深度学习模型，其核心目标是将一张包含高反射条纹的输入图像精确分解为 **漫反射分量（diffuse reflection）** 和 **镜面反射分量（specular reflection）**。

X-FFC 模型的整体架构可以概括为"**双编码、双解码、多层次融合**"的设计范式：

- **双编码器主干（EncoderV2）**：由 ResNet50（空洞卷积版）和 ViT-B/16 两条路径并行提取特征，兼顾局部纹理细节和全局语义信息，并在瓶颈层经 FFC 频域处理和高频增强后输出多尺度编码特征。
- **梯度解码分支（GradientDecoder + EnhancementLayers）**：负责辅助任务——预测图像梯度图，并为主解码器生成经过频域精炼的增强特征。
- **主解码分支（MainDecoderV2）**：负责核心任务——重建镜面反射分量，在逐级上采样过程中融合来自编码器的跳跃特征、来自梯度分支的增强特征，并引入镜面反射感知注意力（SpecularAwareAttention）进一步提升高光区域的重建精度。

最终，漫反射分量通过简单相减得到：\( r = x - b \)，其中 \( x \) 为输入图像，\( b \) 为网络预测的镜面反射分量。网络输出格式为字典：`{'r': pred_diffuse, 'g': gradient_map, 'b': est_specular}`。

---

## 3. 各模块详解

### 3.1 EncoderV2（改进版双编码器）

- **模块名称**：`EncoderV2`
- **模块作用**：X-FFC 模型的特征提取主干，负责从单通道输入图像中提取多尺度、多语义层级的特征，为下游两个解码器提供信息源。
- **在模型中的位置**：网络的最前端，接收原始输入图像，输出 5 组特征图。
- **与其他模块的关系**：其输出 \( (x\_bb\_1, x\_bb\_2, x\_bb\_3, x\_bb\_4) \) 被同时送入 GradientDecoder 和 MainDecoderV2 作为跳跃连接；瓶颈特征 \( x\_c \) 被送入 MainDecoderV2 作为起始输入，也被送入 GradientDecoder 作为全局上下文。
- **在任务中的功能**：通过 ResNet50 + ViT 双路径编码，既保留了 CNN 对局部纹理、边缘的敏感性，又获得了 Transformer 对全局结构的建模能力。末端的 FFC 瓶颈层在频域中处理特征，进一步捕获条纹的周期性模式；HighFrequencyEnhancement 模块则显式增强高频边缘信号，为后续的高光检测提供更清晰的边界线索。

**内部结构详解：**

EncoderV2 内部包含以下关键子组件：

1. **ResNet50 分支（空洞卷积版）**：采用预训练的 ResNet50，在最后一个 stage 使用空洞卷积（dilation）替代下采样步长，以在不损失空间分辨率的前提下扩大感受野。输入层的 7×7 卷积被修改为单通道输入（原始 3 通道权重取均值初始化）。经过 4 个 ResNet stage 后，通过 1×1 卷积进行通道降维，分别输出 \( x\_bb\_1 \) 至 \( x\_bb\_4 \) 四级特征；其中 layer3 和 layer4 的输出经融合后生成 \( x\_bb\_4 \)。

2. **ViT-B/16 分支**：并行的 Vision Transformer 编码器，将输入图像插值至 224×224 后送入 ViT-B/16 提取全局 token 特征。ViT 输出的 patch embedding 被 reshape 为 2D 特征图，与 ResNet 的 \( x\_bb\_4 \) 进行空间对齐后沿通道维度拼接，再经 1×1 卷积融合。

3. **FFC 瓶颈层（ffc_bottleneck）**：对融合后的 256 通道特征执行一次 FFC_BN_ACT 操作。FFC 将特征分为局部分支和全局分支，全局分支在傅里叶域中进行 1×1 卷积，从而赋予该层全局感受野。该操作通过残差连接与输入相加。

4. **高频增强模块（HighFrequencyEnhancement）**：这是 X-FFC 相对于原始 Y-FFC 的关键改进之一。该模块位于 FFC 瓶颈层之后，使用固定的 Laplacian 核显式提取高频分量（边缘/跃变），同时配合可学习的多尺度卷积分支和 SE 通道注意力，将高频信息有选择地注入编码特征中。这为后续的高光边界检测提供了更强的信号。

**输出规格：**

| 输出名称 | 张量尺寸 | 用途 |
|---------|---------|------|
| \( x\_bb\_1 \) | \([B, 64, H/2, W/2]\) | 跳跃连接至两个解码器 Level 4 |
| \( x\_bb\_2 \) | \([B, 128, H/4, W/4]\) | 跳跃连接至两个解码器 Level 3 |
| \( x\_bb\_3 \) | \([B, 256, H/8, W/8]\) | 跳跃连接至两个解码器 Level 2 |
| \( x\_bb\_4 \) | \([B, 512, H/16, W/16]\) | 跳跃连接至两个解码器 Level 1 |
| \( x\_c \) | \([B, 256, H/16, W/16]\) | MainDecoderV2 起始输入 + GradientDecoder 全局上下文 |

---

### 3.2 GradientDecoder（梯度解码器）

- **模块名称**：`GradientDecoder`
- **模块作用**：执行辅助任务——预测图像的梯度图 \( x\_g \)，并在每一级解码过程中生成中间特征供 EnhancementLayers 处理。
- **在模型中的位置**：位于 X-FFC 的左分支（辅助任务分支），接收编码器的四级跳跃特征 \( (x\_bb\_1, ..., x\_bb\_4) \) 和瓶颈特征 \( x\_c \)。
- **与其他模块的关系**：其 4 级中间特征 \( (x\_to\_be\_enhanced\_1, ..., x\_to\_be\_enhanced\_4) \) 被送入 EnhancementLayers 进行频域精炼；最终输出的梯度图 \( x\_g \) 被送入 MainDecoderV2 的最后一级融合。
- **在任务中的功能**：梯度信息能够有效表征图像的结构和边缘。通过增加梯度预测这一辅助任务，网络获得了额外的结构化约束，促使主解码器在重建镜面反射时更好地保持边缘清晰度，防止细节模糊。

**内部结构特点：**

GradientDecoder 采用类似 U-Net 的逐级上采样结构，但在多个尺度级别中穿插了 FFC 模块（ffc_grad_h16、ffc_grad_h8、ffc_grad_h4），使梯度分支也具备频域全局信息捕获能力。每一级的处理流程为：卷积降维 → FFC 频域处理（含残差连接）→ 转置卷积上采样 → 与编码器对应尺度特征拼接。最终经 Sigmoid 激活输出值域 \([0, 1]\) 的单通道梯度图。

此外，GradientDecoder 在第一级还引入了全局上下文融合机制：将编码器瓶颈特征 \( x\_c \) 与 \( x\_bb\_4 \) 拼接后经 1×1 卷积融合，确保梯度分支在最深层就能获取全局语义信息。

---

### 3.3 EnhancementLayers（增强层）

- **模块名称**：`EnhancementLayers`
- **模块作用**：对 GradientDecoder 输出的 4 级中间特征进行频域精炼与二次提炼，生成更高质量的增强特征供 MainDecoderV2 使用。
- **在模型中的位置**：位于 GradientDecoder 和 MainDecoderV2 之间，是两个解码分支的"桥梁"。
- **与其他模块的关系**：输入来自 GradientDecoder 的 4 级桥梁特征；输出的 \( (x\_enhanced\_1, ..., x\_enhanced\_4) \) 被送入 MainDecoderV2 的对应层级进行特征融合。
- **在任务中的功能**：梯度分支生成的中间特征虽然包含有用的结构信息，但未必是主解码器所需的最优表示。EnhancementLayers 通过频域分解和门控融合机制，对这些特征进行针对性的提炼，使其更好地服务于镜面反射分量的重建任务。

**内部结构特点：**

EnhancementLayers 由 4 个 `_SpectralDecompositionUnit`（频谱分解单元）组成，分别处理 4 个尺度的特征。每个单元内部包含：

- **投影分支**：使用 7×7 大卷积核进行空间域特征投影，扩大感受野。
- **镜面分支**：使用 3×3 卷积后接 FFC_BN_ACT，在频域中提取与高光相关的频率分量。
- **门控融合**：通过 Sigmoid 门控机制将两个分支的输出自适应融合，避免简单相加导致的信息冲突。

---

### 3.4 MainDecoderV2（改进版主解码器）

- **模块名称**：`MainDecoderV2`
- **模块作用**：X-FFC 模型的核心组件，负责重建镜面反射分量 \( est\_b \)。
- **在模型中的位置**：位于 X-FFC 的右分支（主任务分支），接收编码器全部 5 级特征、EnhancementLayers 的 4 级增强特征、GradientDecoder 的梯度图以及原始输入图像。
- **与其他模块的关系**：汇聚了网络中所有其他模块的输出，是信息流的最终汇合点。
- **在任务中的功能**：在逐级上采样恢复空间分辨率的过程中，通过注意力门控的跳跃连接、增强特征融合以及镜面反射感知注意力，精确重建高光分量的空间分布和强度。

**内部结构特点：**

MainDecoderV2 融合了多种先进技术，是整个网络中最为复杂的组件：

1. **FFC 深层处理**：在瓶颈部分使用两级级联的 FFC_BN_ACT（FFC1、FFC2），对编码器瓶颈特征进行频域增强，捕获条纹的全局周期性模式。

2. **多尺度特征提取（MultiscaleConvBlock）**：通过三路并行卷积（7×7 / 3×3 / 3×3+3×3）从不同感受野提取特征，增强模型对不同尺度目标的适应性。

3. **注意力门控跳跃连接（Attention_block）**：在每一级上采样的融合点，使用注意力门对编码器的浅层特征进行加权。注意力门利用深层特征（语义信息）生成权重掩码，作用于浅层特征（细节信息），使得只有与任务相关的区域被重点保留。X-FFC 中的注意力门采用了改进设计——psi 输出 \( F\_l \) 通道（每通道独立权重），而非原版的单通道，提供了更精细的注意力控制。

4. **镜面反射感知注意力（SpecularAwareAttention）**：这是 X-FFC 相对于原始 Y-FFC 的另一关键改进。在每一级 transposed_block 输出之后，该模块利用原始输入图像的亮度信息生成高反光空间先验掩码，与深层语义特征融合后产生自适应注意力权重。高亮区域的特征被选择性放大，使网络天然对高光区域具有更高的响应灵敏度。该模块还引入了自适应阈值机制——阈值不是固定的，而是由特征本身通过 SE 模块动态调整。

5. **中间层 FFC 增强**：在 Level 2 和 Level 3 分别插入了 ffc_h8 和 ffc_h4 模块（含残差连接），在中间分辨率下继续进行频域信息提取。

6. **多路转置卷积上采样（TransposedConvBlock）**：使用三路不同 kernel 尺寸（2×2 / 4×4 / 6×6）的转置卷积并行上采样后拼接，缓解单一上采样方式的棋盘格伪影。

**每级融合点的三源融合模式：**

MainDecoderV2 在每一级上采样中都执行一次"三源融合"：
- 来源一：自身上一级处理后的特征（经 SpecularAwareAttention 增强）
- 来源二：编码器对应尺度的跳跃特征（经 Attention_block 加权）
- 来源三：EnhancementLayers 对应尺度的增强特征

三者沿通道维度拼接后送入下一级处理。在最终的 Level 5，梯度图 \( x\_g \) 也被融合进来以提供额外的结构约束。

---

### 3.5 FFC 核心模块（快速傅里叶卷积）

- **模块名称**：`FourierUnit` / `SpectralTransform` / `FFC_BN_ACT`
- **模块作用**：X-FFC 的频域处理基础，通过将空间特征经 FFT 转入频域，在频域执行 1×1 卷积获取全局感受野，再经 IFFT 回到空间域。
- **在模型中的位置**：被广泛嵌入到 EncoderV2 的瓶颈层、GradientDecoder 的各级、MainDecoderV2 的各级以及 EnhancementLayers 的频谱分解单元中。
- **与其他模块的关系**：FFC_BN_ACT 是 FFC 在网络中的统一入口，它内部调用 SpectralTransform，SpectralTransform 再调用 FourierUnit 完成核心的 FFT/IFFT 操作。
- **在任务中的功能**：传统 CNN 在捕捉长距离依赖和全局周期性模式方面能力有限。FFC 赋予了网络全局感受野，这对于识别和分离具有全局性的条纹图案至关重要。

**X-FFC 中 FFC 模块的优化改进（相对于原始 FFC 论文）：**

- **F1**：激活函数由 ReLU 替换为 GELU，避免对负的虚部执行 hard-zero，保护频域相位信息。
- **F2**：引入频率 SE 通道注意力，自适应选择有用的频段。
- **F3**：LFU（局部傅里叶单元）拆分为独立的 lfu_top 和 lfu_bot，对特征图的上半部分和下半部分分别处理，避免参数竞争。
- **F4**：FU 与 LFU 之间采用门控融合（替代硬相加），由可学习的 Sigmoid 门控决定融合比例。
- **F5**：跨域分支（局部/全局）之间采用门控融合，防止频域噪声污染空间分支。
- **F6**：FFT 前施加 Hann 窗函数，减少频谱泄漏。

---

### 3.6 HighFrequencyEnhancement（高频特征增强模块）

- **模块名称**：`HighFrequencyEnhancement`
- **模块作用**：显式提取编码特征中的高频分量（边缘、跃变），为高反光检测提供清晰的边界信号。
- **在模型中的位置**：位于 EncoderV2 的 FFC 瓶颈层之后，作用于编码特征 \( x\_c \)。
- **与其他模块的关系**：接收 FFC 瓶颈层的输出，增强后的 \( x\_c \) 被传递给 GradientDecoder 和 MainDecoderV2。
- **在任务中的功能**：高反光区域通常伴随着强烈的亮度跃变。通过显式的高频增强，编码特征中的高光边界信息被放大，有助于后续解码器更精确地定位和重建高光区域。

**内部结构：** 该模块包含三部分——固定的 Laplacian 核提取边缘响应、可学习的多尺度卷积分支（含空洞卷积以扩大感受野）、SE 通道注意力对融合后的高频特征进行通道级加权。最终通过残差连接与输入相加，确保原始信息不丢失。

---

### 3.7 SpecularAwareAttention（镜面反射感知注意力模块）

- **模块名称**：`SpecularAwareAttention`
- **模块作用**：利用原始输入图像的亮度信息作为先验，动态增强高光区域对应的特征响应。
- **在模型中的位置**：位于 MainDecoderV2 的每一级 transposed_block 输出之后（共 4 处：sa1 至 sa4）。
- **与其他模块的关系**：接收 MainDecoderV2 当前级的特征和原始输入图像；输出增强后的特征继续参与后续的三源融合。
- **在任务中的功能**：在解码器的每个尺度级别，该模块根据输入图像的亮度分布生成高光先验掩码。高于阈值的亮区域获得更高的注意力权重，使网络在所有分辨率层级上都对高光区域保持高灵敏度。

**内部结构特点：**

- **自适应阈值**：阈值并非固定值，而是通过一个轻量级 SE 模块从当前特征中动态估计，使阈值随图像内容自适应调整。
- **软掩码生成**：使用 \( \text{Sigmoid}(k \cdot (I - \theta)) \) 生成平滑的高光掩码，其中 \( k \) 为陡度参数，\( I \) 为输入亮度，\( \theta \) 为自适应阈值。
- **调制网络**：将特征与掩码拼接后经两层卷积+GELU+Sigmoid 生成调制权重，最终通过 \( \text{features} \times (1 + \text{att}) \) 的方式增强特征，保证高光区域被放大而非高光区域不受损害。

---

### 3.8 Attention_block（注意力门控块）

- **模块名称**：`Attention_block`
- **模块作用**：在跳跃连接融合时，对编码器浅层特征进行注意力加权，抑制无关噪声。
- **在模型中的位置**：位于 MainDecoderV2 的每个融合点（att1 至 att4），在三源融合之前对编码器特征进行预处理。
- **与其他模块的关系**：接收 MainDecoderV2 当前级的深层特征（作为 gate signal）和编码器对应尺度的浅层特征（作为待加权目标）。
- **在任务中的功能**：深层特征包含丰富的语义信息，浅层特征包含精细的空间细节。注意力门利用语义信息自动判断哪些空间位置的细节信息对当前任务有用，从而实现更高效的特征融合。

---

### 3.9 MultiscaleConvBlock 与 FeatureExtrationLayersB（多尺度卷积块）

- **模块名称**：`MultiscaleConvBlock` / `FeatureExtrationLayersB`
- **模块作用**：通过多路并行卷积从不同感受野提取特征，增强模型对不同尺度目标的适应性。
- **在模型中的位置**：MultiscaleConvBlock 位于 MainDecoderV2 的瓶颈处（FFC 之后、第一级上采样之前）；FeatureExtrationLayersB 位于 Level 2 和 Level 3 之间。
- **在任务中的功能**：高光区域在不同图像中可能呈现不同的尺度和形态。多尺度卷积确保网络能够同时感知大范围的光斑和小范围的点状高光。

---

### 3.10 损失函数体系

X-FFC 模型配备了完善的复合损失函数体系，用于从多个维度约束网络的学习：

**基础复合损失（CompositeLoss）** 包含：SSIM 结构相似性损失、边缘加权 L1 损失、频域 L1 损失、VGG16 感知损失、镜面分量直接监督损失、统计损失（均值/标准差一致性）、梯度分支 SSIM 损失。

**增强复合损失（EnhancedCompositeLoss）** 在此基础上新增：SpecularFocalLoss（对难预测的高亮像素给予更大权重）、SpecularDiceLoss（从分割视角约束高光区域重叠度）、GradientConsistencyLoss（约束梯度幅值和方向一致性）、OHEM 在线难样本挖掘。

此外，DynamicLossWeightScheduler 根据训练进度动态调整各损失项的权重，使网络在训练初期关注全局重建质量，后期逐步强化细节和频域约束。

---

## 4. 数据在网络中的完整流动路径

以下是 X-FFC 模型中数据的完整流动路径（以 512×512 单通道输入为例）：

### 第一阶段：双编码器特征提取（EncoderV2）

```
➡️ 初始输入: x [B, 1, 512, 512]

⬇️ (encoder): EncoderV2
    │
    ├─> ResNet50 分支:
    │   ├─ stem_conv (1→64 通道, stride=2)
    │   │  └─ x_bb_1: [B, 64, 256, 256]         ==> (保存)
    │   ├─ layer1 → reduce_l1
    │   │  └─ x_bb_2: [B, 128, 128, 128]         ==> (保存)
    │   ├─ layer2 → reduce_l2
    │   │  └─ x_bb_3: [B, 256, 64, 64]           ==> (保存)
    │   ├─ layer3 + layer4 → reduce + fuse_l34
    │   │  └─ x_bb_4: [B, 512, 32, 32]           ==> (保存)
    │
    ├─> ViT-B/16 分支 (并行):
    │   ├─ 输入插值至 224×224, 扩展为 3 通道
    │   ├─ ViT 编码 → patch embedding reshape
    │   └─ vit_feat: [B, 768, 14, 14] → 插值至 [B, 768, 32, 32]
    │
    ├─> 双分支融合:
    │   ├─ torch.cat([x_bb_4, vit_feat]) → [B, 1280, 32, 32]
    │   ├─ vit_resnet_fuse_1x1 → [B, 512, 32, 32]
    │   └─ cba_after_backbone → [B, 256, 32, 32]
    │
    ├─> FFC 瓶颈层:
    │   └─ ffc_bottleneck + 残差连接 → [B, 256, 32, 32]
    │
    └─> 高频增强 (V2 新增):
        └─ hf_enhance → x_c: [B, 256, 32, 32]   ==> (送往两个解码器)
```

### 第二阶段：X 形分叉

- **左分支**：\( (x\_bb\_1, ..., x\_bb\_4, x\_c) \) → GradientDecoder
- **右分支**：\( (x\_bb\_1, ..., x\_bb\_4, x\_c) \) → MainDecoderV2（后续还将接收左分支的增强特征）

### 第三阶段：左分支处理（GradientDecoder + EnhancementLayers）

```
➡️ 输入: x_bb_4 [B, 512, 32, 32] + x_c [B, 256, 32, 32]

⬇️ GradientDecoder
    │
    ├─ Level 1 (32×32): 全局上下文融合 → 卷积提炼 → FFC_grad_h16
    │   └─ x_to_be_enhanced_1: [B, 256, 32, 32]  ==> (送往 EnhancementLayers)
    │
    ├─ Level 2 (64×64): 降维 → FFC_grad_h8 → 转置卷积上采样
    │   └─ x_to_be_enhanced_2: [B, 128, 64, 64]  ==> (送往 EnhancementLayers)
    │
    ├─ Level 3 (128×128): 降维 → FFC_grad_h4 → 转置卷积上采样
    │   └─ x_to_be_enhanced_3: [B, 64, 128, 128] ==> (送往 EnhancementLayers)
    │
    ├─ Level 4 (256×256): 降维 → 转置卷积上采样
    │   └─ x_to_be_enhanced_4: [B, 32, 256, 256]  ==> (送往 EnhancementLayers)
    │
    └─ Level 5 (512×512): 上采样 → 输出层 → Sigmoid
        └─ x_g: [B, 1, 512, 512]                  ==> (送往 MainDecoderV2 最终层)

⬇️ EnhancementLayers (频谱分解精炼)
    ├─ x_to_be_enhanced_1 → _SpectralDecompositionUnit → x_enhanced_1: [B, 128, 32, 32]
    ├─ x_to_be_enhanced_2 → _SpectralDecompositionUnit → x_enhanced_2: [B, 64, 64, 64]
    ├─ x_to_be_enhanced_3 → _SpectralDecompositionUnit → x_enhanced_3: [B, 32, 128, 128]
    └─ x_to_be_enhanced_4 → _SpectralDecompositionUnit → x_enhanced_4: [B, 16, 256, 256]
```

### 第四阶段：右分支处理（MainDecoderV2 — 核心重建）

```
➡️ 输入: x_c [B, 256, 32, 32] + 原始输入 x [B, 1, 512, 512]

⬇️ MainDecoderV2
    │
    ├─ 瓶颈处理: FFC1 → FFC2 → 残差 → MultiscaleConvBlock
    │   └─ [B, 576, 32, 32]
    │
    ├─ Level 1 (32×32):
    │   ├─ TransposedConvBlock_same → [B, 768, 32, 32]
    │   ├─ SpecularAwareAttention (sa1) → [B, 768, 32, 32]     ← V2 新增
    │   ├─ Attention_block(att1) 加权 x_bb_4 → bridge
    │   └─ cat([自身, bridge, x_enhanced_1]) → [B, 1408, 32, 32]
    │
    ├─ Level 2 (64×64):
    │   ├─ TransposedConvBlock → [B, 384, 64, 64]
    │   ├─ FFC_h8 + 残差
    │   ├─ SpecularAwareAttention (sa2) → [B, 384, 64, 64]     ← V2 新增
    │   ├─ Attention_block(att2) 加权 x_bb_3 → bridge
    │   └─ cat([自身, bridge, x_enhanced_2]) → [B, 704, 64, 64]
    │
    ├─ Level 3 (128×128):
    │   ├─ FeatureExtrationLayersB → TransposedConvBlock → [B, 192, 128, 128]
    │   ├─ FFC_h4 + 残差
    │   ├─ SpecularAwareAttention (sa3) → [B, 192, 128, 128]   ← V2 新增
    │   ├─ Attention_block(att3) 加权 x_bb_2 → bridge
    │   └─ cat([自身, bridge, x_enhanced_3]) → [B, 352, 128, 128]
    │
    ├─ Level 4 (256×256):
    │   ├─ TransposedConvBlock → [B, 96, 256, 256]
    │   ├─ SpecularAwareAttention (sa4) → [B, 96, 256, 256]    ← V2 新增
    │   ├─ Attention_block(att4) 加权 x_bb_1 → bridge
    │   └─ cat([自身, bridge, x_enhanced_4]) → [B, 176, 256, 256]
    │
    ├─ Level 5 (512×512):
    │   ├─ TransposedConvBlock → [B, 48, 512, 512]
    │   └─ cat([自身, x_g]) → [B, 49, 512, 512]
    │
    └─ 输出层: output_conv1 → output_conv2 → Sigmoid
        └─ est_b: [B, 1, 512, 512]
```

### 第五阶段：最终输出

```
pred_diffuse = clamp(x - est_b, 0, 1)

输出字典: {'r': pred_diffuse, 'g': x_g, 'b': est_b}
```

---

## 5. 整体结构总结

### 5.1 整体架构设计

X-FFC 模型采用"**双编码 → 双解码 → 多层次融合**"的整体架构。网络的拓扑结构呈"X"形：上方两条编码路径（ResNet50 + ViT）在瓶颈层汇聚，下方两条解码路径（GradientDecoder + MainDecoderV2）分叉后通过 EnhancementLayers 桥梁进行信息交互，最终汇聚于主解码器的输出。

### 5.2 双编码器特点

X-FFC 的双编码器设计是其区别于原始 Y-FFC 的核心特征之一：

- **ResNet50（空洞卷积版）**：作为局部特征提取的主力，在 4 个 stage 中逐层构建从低级纹理到高级语义的特征金字塔。最后一个 stage 使用空洞卷积替代下采样，在保持空间分辨率的同时扩大感受野，这对于捕捉条纹图案的细微变化尤为重要。
- **ViT-B/16**：作为全局语义提取的补充，通过 self-attention 机制建模长距离像素间的依赖关系。ViT 的全局视野使其能够理解图像的整体照明条件和高光的空间分布规律，为 ResNet50 的局部特征提供宏观语境。

两条路径在瓶颈层通过通道拼接 + 1×1 卷积进行融合，随后经 FFC 频域处理和高频增强进一步精炼。这种"CNN + Transformer"的互补设计使编码器能够同时捕获局部纹理细节和全局结构语义。

### 5.3 特征提取与融合流程

X-FFC 的特征融合是多层次、多来源的，体现了精细化的信息传递策略：

1. **编码阶段**：ResNet50 和 ViT 的特征在瓶颈层融合（跨架构融合）。
2. **频域增强**：FFC 模块在编码器瓶颈层和两个解码器的多个尺度级别中均有部署（空间-频域融合）。
3. **辅助→主任务传递**：GradientDecoder 的中间特征经 EnhancementLayers 精炼后传递给 MainDecoderV2（跨分支融合）。
4. **多尺度跳跃连接**：编码器的 4 级特征通过注意力门控传递给 MainDecoderV2（跨尺度融合）。
5. **高光先验注入**：SpecularAwareAttention 在每一级解码中引入原始输入的亮度先验（任务先验融合）。
6. **梯度约束**：梯度图在最终层被融合进主解码器（结构约束融合）。

---

## 6. 设计特点与优势

### 6.1 设计特点

1. **"X"形双编码器架构**：CNN + Transformer 双路径编码，兼顾局部精度和全局理解。
2. **多任务学习范式**：梯度预测辅助任务为主任务提供结构化约束和增强特征。
3. **全链路频域处理**：FFC 模块贯穿编码器和两个解码器，在多个尺度上捕获全局频率信息。
4. **高光感知机制**：SpecularAwareAttention 和 HighFrequencyEnhancement 两个专用模块，从特征响应和频率分量两个角度强化网络对高光区域的敏感性。
5. **注意力门控的智能融合**：在跳跃连接中使用注意力门，避免浅层噪声无差别地注入深层特征。
6. **频谱分解增强层**：EnhancementLayers 中的 _SpectralDecompositionUnit 通过投影 + 频域 + 门控三路机制精炼跨分支特征。

### 6.2 整体优势

- **高光检测精度高**：双编码器提供丰富的多尺度特征，SpecularAwareAttention 在每一级解码中显式引导网络关注高光区域，配合 HighFrequencyEnhancement 增强边界信号，使高光区域的检测和重建精度显著提升。
- **全局-局部平衡**：ViT 捕获全局照明模式，ResNet50 保留局部纹理细节，FFC 在频域中补充周期性信息，三者互补，使网络在处理全局性条纹和局部性高光时都表现出色。
- **结构保持能力强**：梯度辅助任务提供边缘和结构约束，注意力门控过滤无关噪声，多尺度卷积适应不同尺度的目标，共同确保重建结果中的结构信息不丢失。
- **训练稳定性好**：EMA、梯度累积、混合精度训练、动态损失权重调度、分层学习率优化器等工程手段保障了大型网络的训练稳定性和收敛效率。
- **鲁棒性强**：FFC 中的 Hann 窗减少频谱泄漏，门控融合避免频域噪声污染，GELU 保护相位信息，SE 通道注意力自适应频段选择——这些细节优化共同提升了网络的鲁棒性。

---

## 7. 代码结构总览

以下是 X-FFC 模型的模块化层级结构：

```
YFFC_V2 (X-FFC)
  │
  ├─ (encoder): EncoderV2                          // 双编码器主干
  │   ├─ (stem_conv / stem_bn / stem_relu / stem_pool)   // ResNet50 stem
  │   ├─ (layer1 ... layer4): ResNet50 Stages       // ResNet50 特征提取
  │   ├─ (reduce_l1 ... reduce_l4, fuse_l34)        // 通道降维与融合
  │   ├─ (vit_branch): ViTEncoderBranch             // ViT-B/16 全局编码
  │   ├─ (vit_resnet_fuse_1x1): Conv2D_BN_activa    // 双分支融合
  │   ├─ (cba_after_backbone): Conv2D_BN_activa      // 瓶颈降维
  │   ├─ (ffc_bottleneck): FFC_BN_ACT               // FFC 频域处理
  │   └─ (hf_enhance): HighFrequencyEnhancement     // ★ V2 新增：高频增强
  │
  ├─ (gradient_decoder): GradientDecoder            // 梯度解码分支
  │   ├─ (fuse_global_context): Conv2D_BN_activa
  │   ├─ (block_1 ... block_5): 含卷积/转置卷积/FFC
  │   └─ (sigmoid_layer): Sigmoid
  │
  ├─ (enhancement_layers): EnhancementLayers        // 频谱分解增强桥梁
  │   └─ (decomp_unit_1 ... _4): _SpectralDecompositionUnit
  │       ├─ projection_branch (7×7 卷积)
  │       ├─ specular_branch (3×3 卷积 + FFC)
  │       └─ gate_conv (门控融合)
  │
  └─ (main_decoder): MainDecoderV2                  // 主解码分支
      ├─ (FFC1, FFC2): FFC_BN_ACT                   // 瓶颈 FFC
      ├─ (multiscale_conv): MultiscaleConvBlock      // 多尺度卷积
      ├─ (transposed_block_1 ... _5): TransposedConvBlock // 多路上采样
      ├─ (att1 ... att4): Attention_block            // 注意力门控
      ├─ (sa1 ... sa4): SpecularAwareAttention       // ★ V2 新增：高光感知注意力
      ├─ (ffc_h8, ffc_h4): FFC_BN_ACT               // 中间层 FFC
      └─ (output_conv1, output_conv2): Conv2D_BN_activa // 输出层
```

---

## 8. 总结

X-FFC（X-shaped Fast Fourier Convolutional Network）是在原始 Y-FFC 基础上进行系统性改进的高级图像分解模型。其命名源自 ResNet50 + ViT-B/16 双编码器的"X"形拓扑结构，这一设计赋予了网络同时捕获局部纹理细节和全局语义结构的能力。

X-FFC 模型的核心创新可归纳为以下三点：

1. **双编码器架构**：CNN + Transformer 互补编码，突破了单一骨干网络的信息瓶颈。
2. **高光专用模块**：HighFrequencyEnhancement 和 SpecularAwareAttention 两个针对性设计的模块，从编码端和解码端双向强化网络对高反光区域的感知和重建能力。
3. **全链路频域增强**：优化的 FFC 模块贯穿网络的编码器、两个解码器和增强层，在多个尺度上提供全局频率信息，尤其适合处理具有周期性的条纹图案。

这些改进与原有的多任务学习框架、注意力门控跳跃连接、多尺度特征提取、频谱分解增强层等技术有机结合，使 X-FFC 成为一个在图像高光分解任务上功能强大、设计精巧的深度学习模型。
