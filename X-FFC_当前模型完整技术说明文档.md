# X-FFC / YFFC 当前模型完整技术说明文档

## 1. 文档说明

本文档基于以下 4 个文件整理：

- `Y_FFC_unified.py`：模型结构、核心模块、损失函数、数据增强、部分训练工具。
- `train.py`：工程化训练入口、训练数据集、动态损失权重、优化器参数组、训练/验证/保存流程。
- `process_images.py`：单图与批量推理、模型加载、滑窗推理、TTA、结果保存。
- `X-FFC_V3_模型完整技术文档.md`：原始说明文档中的结构性信息来源。

本文只描述当前代码中实际存在的模型结构、模块位置、模块功能、训练数据流和推理流程，不再描述模型相对于历史实现的变化过程，也不再保留“改进点索引”“版本演进说明”等内容。

---

## 2. 模型任务定义

X-FFC / YFFC 面向单通道灰度结构光条纹图像的高反光分解任务。给定输入图像：

\[
x \in [0,1]^{B\times 1\times H\times W}
\]

模型学习输出以下内容：

| 输出键 | 张量形状 | 含义 | 主要用途 |
|---|---:|---|---|
| `r` | `[B, 1, H, W]` | 最终漫反射 / 去高反光图像 | 主结果 |
| `b` | `[B, 1, H, W]` | 镜面反射 / 高反光分量 | 主解码器输出之一 |
| `g` | `[B, 1, H, W]` | 输入图像梯度预测图 | 辅助监督与结构先验 |
| `s` | `[B, 1, H, W]` | 高反光软概率图 | 辅助监督、增强层与主解码调制 |
| `r_direct` | `[B, 1, H, W]` | 主解码器直接预测的漫反射图 | 训练辅助与最终融合 |
| `alpha` | 标量或可广播张量 | 双路漫反射融合系数 | 控制 `r_direct` 与减法路径比例 |
| `cs` | 标量 | 跨尺度一致性辅助项 | 只参与训练损失 |

模型核心目标是使 `r` 接近干净的目标漫反射图，使 `b` 接近由输入与目标差值得到的高反光分量，使 `g` 接近输入图像的 Sobel 梯度幅值，使 `s` 接近高反光软标签。

---

## 3. 源文件职责

### 3.1 `Y_FFC_unified.py`

该文件是模型主文件，包含：

1. 数值安全工具、窗口函数、Sobel 梯度函数。
2. FFC 频域卷积核心模块。
3. 基础网络构件。
4. Swin-Tiny 编码分支、ResNet50 编码分支、多模态融合模块。
5. 高反光感知注意力、高频增强模块。
6. 数据集和数据增强类。
7. `Encoder`、`GradientDecoder`、`EnhancementLayers`、`MainDecoder`、`ResidualRefinementHead`、`YFFC`。
8. 复合损失函数 `EnhancedCompositeLoss`。
9. EMA、优化器、调度器、训练与验证辅助函数。

### 3.2 `train.py`

该文件是当前训练入口，主要负责：

1. 定义训练集读取类 `ImprovedFringeDataset`。
2. 创建训练增强流水线。
3. 构建模型、损失函数、优化器和学习率调度器。
4. 执行多 epoch 训练与验证。
5. 记录 Loss、SSIM、MAE、MSE、PSNR 和全部子损失。
6. 保存综合最优、SSIM 最优、Loss 最优、最后一轮模型和定期 checkpoint。
7. 保存训练曲线、验证样本可视化和训练总结文件。

### 3.3 `process_images.py`

该文件是当前推理入口，主要负责：

1. 加载 checkpoint 并构建 `YFFC` 模型。
2. 自动识别 EMA 权重、GroupNorm 状态和 checkpoint 中记录的训练尺寸。
3. 将输入图像转换为灰度 `[1,1,H,W]` 张量。
4. 对大图执行滑窗推理，并使用 Tukey 窗融合 patch 输出。
5. 可选执行 4 次翻转 TTA。
6. 保存漫反射、高反光、梯度、高反光概率图、16-bit 图和可视化结果。

---

## 4. 模型总体结构

当前 `YFFC.forward(x)` 的主数据流如下：

```text
输入 x [B,1,H,W]
        │
        ▼
Encoder
        │
        ├─ x_bb_1 [B,  64, H/2,  W/2]
        ├─ x_bb_2 [B, 128, H/4,  W/4]
        ├─ x_bb_3 [B, 256, H/8,  W/8]
        ├─ x_bb_4 [B, 512, H/16, W/16]
        └─ x_c    [B, 256, H/16, W/16]
        │
        ├───────────────► GradientDecoder
        │                  ├─ xe1 [B,256,H/16,W/16]
        │                  ├─ xe2 [B,128,H/8, W/8]
        │                  ├─ xe3 [B, 64,H/4, W/4]
        │                  ├─ xe4 [B, 32,H/2, W/2]
        │                  ├─ g   [B,  1,H,   W]
        │                  └─ s   [B,  1,H,   W]
        │
        ├───────────────► EnhancementLayers(xe1..xe4, g, x, s)
        │                  ├─ enh1 [B,128,H/16,W/16]
        │                  ├─ enh2 [B, 64,H/8, W/8]
        │                  ├─ enh3 [B, 32,H/4, W/4]
        │                  └─ enh4 [B, 16,H/2, W/2]
        │
        ▼
MainDecoder(encoder features, enhanced features, g, s, x)
        ├─ b        [B,1,H,W]
        ├─ r_direct [B,1,H,W]
        └─ alpha_logit
        │
        ▼
双路漫反射融合
        ├─ r_sub   = clamp(x - b, 0, 1)
        ├─ alpha   = sigmoid(alpha_logit)
        └─ r_fused = alpha * r_direct + (1-alpha) * r_sub
        │
        ▼
ResidualRefinementHead(r_fused, x, r_direct)
        │
        ▼
最终输出 r
```

同时，模型根据编码器中的 H/4、H/8、H/16 融合特征计算跨尺度一致性标量 `cs`，该标量进入训练损失。

---

## 5. 基础数值与图像处理工具

### 5.1 `_safe_tensor`

`_safe_tensor` 用于在 FFT、注意力、融合、输出等高风险位置执行数值保护。它会将 NaN、正无穷、负无穷替换为有限数值，并可选执行 clamp。该函数在频域模块、注意力模块、主输出和跨尺度一致性计算中都会被调用。

### 5.2 `_tukey_window`

`_tukey_window(size, alpha=0.25)` 生成 Tukey 一维窗。它在两个位置使用：

1. `FourierUnit.forward`：FFT 之前对空间特征施加二维 Tukey 窗，降低频谱边界不连续。
2. `process_images.py` 的滑窗推理：生成 patch 融合权重，使重叠区域平滑融合。

### 5.3 Sobel 梯度目标

模型使用 `compute_input_gradient_target(input_img)` 生成梯度监督目标。底层为 `normalized_sobel_magnitude`：

```text
input_img -> sobel_magnitude -> per-sample normalize -> gt_gradient
```

训练集中 `gt_gradient` 对应增强后的输入图像，而不是目标漫反射图像。

---

## 6. FFC 频域卷积核心模块

FFC 是模型中频域特征建模的基础。它出现在编码器瓶颈、梯度解码器、主解码器和增强层中。

### 6.1 `FourierUnit`

`FourierUnit` 是最底层的频域处理单元。

输入输出：

```text
输入:  x [B,C,H,W]
输出:  y [B,C_out,H,W]
```

内部流程：

```text
x
 ├─ 可选空间缩放
 ├─ Tukey 窗
 ├─ rFFT: 空间域 -> 频域
 ├─ real/imag 拆分并拼接到通道维
 ├─ 1×1 Conv
 ├─ BatchNorm
 ├─ GELU
 ├─ 频域 SE 通道注意力
 ├─ real/imag 重新组合为复数
 ├─ iRFFT: 频域 -> 空间域
 └─ 数值安全处理
```

该模块的作用是利用 FFT 在频域内进行全局特征混合，使网络在较低计算成本下获得全局感受野，并保留频域幅度与相位相关信息。

### 6.2 `SpectralTransform`

`SpectralTransform` 是对 `FourierUnit` 的封装，用于 FFC 的全局分支。

内部流程：

```text
输入特征
  -> 1×1 Conv 降维
  -> FourierUnit 全局频域处理
  -> LFU 局部频域处理
  -> 门控融合全局频域输出与局部频域输出
  -> 1×1 Conv 升维
  -> 输出
```

其中 LFU 将特征在高度方向分成上半部分和下半部分，分别通过独立 FourierUnit 处理，然后重新拼接。这样可以在全局频域建模之外保留局部空间区域内的频域结构。

### 6.3 `FFC_BN_ACT`

`FFC_BN_ACT` 是网络中使用 FFC 的统一入口。它将输入通道分为局部分支 L 与全局分支 G，并执行四路交互：

```text
L -> L: 普通空间卷积
L -> G: 普通空间卷积
G -> L: SpectralTransform
G -> G: SpectralTransform
```

输出为 `(out_xl, out_xg)`，分别对应局部输出和全局输出。下游模块通常会将二者在通道维拼接：

```python
out = self.ffc_module((x_local, x_global))
x = torch.cat([out[0], out[1]], dim=1)
```

`ratio_gin` 和 `ratio_gout` 控制输入和输出中参与全局频域分支的通道比例。

### 6.4 `FFCResnetBlock`

`FFCResnetBlock` 是由两个 `FFC_BN_ACT` 组成的残差块。当前模型主干中主要直接使用 `FFC_BN_ACT`，该残差块作为可复用结构保留在模型文件中。

---

## 7. 基础网络构件

### 7.1 `Conv2D_BN_activa`

这是模型中最常用的基础卷积块，支持普通卷积和转置卷积：

```text
Conv2d / ConvTranspose2d -> BatchNorm2d(可选) -> ReLU(可选)
```

它用于编码器通道压缩、解码器卷积、注意力前投影、增强层投影等位置。

### 7.2 `Attention_block`

该模块用于主解码器中的 skip connection 门控。

输入：

- `g`：来自解码器的当前尺度特征。
- `x`：来自编码器的 skip 特征。

流程：

```text
g -> 1×1 Conv -> BN
x -> 1×1 Conv -> BN
两者相加 -> ReLU -> 1×1 Conv -> BN -> Sigmoid
输出 = x * psi
```

与单通道注意力不同，`psi` 输出为 `F_l` 通道，使每个 skip 通道拥有独立权重。

### 7.3 `MultiscaleConvBlock`

用于主解码器瓶颈后的多尺度卷积提取。它包含三条路径：

1. `1×1 -> 7×7`
2. `3×3`
3. `1×1 -> 3×3 -> 3×3`

三条路径输出拼接，形成 576 通道特征。

### 7.4 `FeatureExtrationLayersB`

用于主解码器中间层。它包含四条卷积路径，输出通道数分别为 192、192、128、128，拼接后得到 640 通道特征。

### 7.5 `TransposedConvBlock`

该模块虽然保留了类名，但实际由三条 resize-conv 分支组成：

```text
Upsample ×2 + 3×3 Conv
Upsample ×2 + 5×5 Conv
Upsample ×2 + 7×7 Conv
```

三路输出按通道拼接。它负责主解码器中的空间上采样。

### 7.6 `TransposedConvBlock_same`

该模块不改变空间分辨率，包含三条同尺寸 `3×3` 卷积分支，三路输出按通道拼接。它用于主解码器的第一个 level。

---

## 8. 编码器 `Encoder`

`Encoder` 是模型的双分支编码模块，由 ResNet50 局部纹理分支、Swin-Tiny 窗口上下文分支、多尺度融合、FFC 瓶颈和高频增强组成。

### 8.1 编码器输入输出

输入：

```text
x [B,1,H,W]
```

输出 5 个特征：

| 名称 | 形状 | 来源 | 用途 |
|---|---:|---|---|
| `x_bb_1` | `[B, 64, H/2, W/2]` | ResNet stem | 主/梯度解码器浅层 skip |
| `x_bb_2` | `[B, 128, H/4, W/4]` | ResNet layer1 + Swin H/4 融合 | 中浅层 skip |
| `x_bb_3` | `[B, 256, H/8, W/8]` | ResNet layer2 + Swin H/8 融合 | 中层 skip |
| `x_bb_4` | `[B, 512, H/16, W/16]` | ResNet layer3/layer4 + Swin H/16 融合 + top-down 回流 | 深层 skip |
| `x_c` | `[B, 256, H/16, W/16]` | FFC 瓶颈 + 高频增强 | 主瓶颈特征 |

### 8.2 ResNet50 局部纹理分支

ResNet50 被修改为单通道输入。第一层卷积权重通过 RGB 权重均值初始化为灰度卷积核。ResNet 的 layer4 使用空洞卷积配置，使深层输出保持 H/16 分辨率。

数据流：

```text
x
 -> conv1/bn/relu
 -> x_bb_1 [64, H/2, W/2]
 -> maxpool
 -> layer1 -> reduce_l1 -> x_bb_2 [128, H/4, W/4]
 -> layer2 -> reduce_l2 -> x_bb_3 [256, H/8, W/8]
 -> layer3 -> reduce_l3 -> x_l3_512 [512, H/16, W/16]
 -> layer4 -> reduce_l4 -> x_l4_512 [512, H/16, W/16]
 -> concat(x_l3_512, x_l4_512)
 -> fuse_l34 -> x_bb_4 [512, H/16, W/16]
```

该分支提供局部纹理、条纹边缘、亮暗跃迁和浅层空间定位信息。

### 8.3 Swin-Tiny 窗口上下文分支 `SwinEncoderBranch`

Swin 分支同样接收原始灰度输入。Patch embedding 被定义为单通道卷积，并使用 RGB 预训练权重均值初始化。输入尺寸若不能被 32 整除，会只在右侧和下侧进行 padding。

输出：

| Swin 特征 | 形状 | 对应阶段 |
|---|---:|---|
| `swin_4` | `[B, 96, H/4, W/4]` | Swin features 0/1 |
| `swin_8` | `[B, 192, H/8, W/8]` | Swin features 2/3 |
| `swin_16` | `[B, 384, H/16, W/16]` | Swin features 4/5 |

Swin 分支用于提供窗口级上下文、区域关系和较长距离的结构信息。

### 8.4 H/4 与 H/8 自适应 skip 融合 `AdaptiveSkipFusion`

该模块用于把 Swin 特征注入 ResNet skip：

```text
swin_feat -> 3×3 Conv + BN + GELU -> swin_proj
cat(res_feat, swin_proj) -> 1×1 Conv + GELU + 1×1 Conv + Sigmoid -> gate
输出 = res_feat + beta * gate * swin_proj
```

位置：

- `fuse_h4(x_bb_2, swin_4)`：输出更新后的 H/4 skip。
- `fuse_h8(x_bb_3, swin_8)`：输出更新后的 H/8 skip。

`beta` 是可学习强度系数，`gate` 是逐位置逐通道权重。

### 8.5 H/16 双向交叉注意力融合 `CrossModalFusion`

Swin H/16 特征先经过 `swin16_proj` 投影到 512 通道，然后与 ResNet 的 `x_bb_4` 在 `CrossModalFusion` 中融合。

输入：

```text
cnn_feat     = x_bb_4       [B,512,H/16,W/16]
context_feat = swin_feat_16 [B,512,H/16,W/16]
```

内部包含两个方向：

1. CNN 查询 Swin：ResNet 特征作为 Query，Swin 特征作为 Key/Value。
2. Swin 查询 CNN：Swin 特征作为 Query，ResNet 特征作为 Key/Value。

两路结果通过门控融合：

```text
gate = sigmoid(Conv(cat(c2ctx_out, ctx2c_out)))
fused = gate * c2ctx_out + (1-gate) * ctx2c_out
final = 3×3 Conv + BN + GELU
```

输出作为 H/16 深层融合特征。

### 8.6 跨尺度 top-down 回流 `CrossScaleTopDownRefinement`

该模块把 H/16 上下文逐级回流到 H/8 和 H/4：

```text
h16 -> h16_refine -> h16_out
h16_ctx -> 1×1 投影到 H/8 通道 -> 上采样 -> 与 h8 拼接生成 gate -> h8_out
h8_out -> h8_refine -> 1×1 投影到 H/4 通道 -> 上采样 -> 与 h4 拼接生成 gate -> h4_out
```

输出为更新后的：

```text
x_bb_2, x_bb_3, x_bb_4
```

该模块的强度由可学习 `gamma` 控制，并限制在较小范围内。

### 8.7 FFC 瓶颈与高频增强

H/16 融合特征经过：

```text
x_bb_4 -> cba_after_backbone -> x_c [B,256,H/16,W/16]
x_c -> FFC_BN_ACT(ratio=0.5) -> residual add
x_c -> HighFrequencyEnhancement
```

`HighFrequencyEnhancement` 使用固定 Laplacian 分支、可学习高频分支、SE 通道注意力和残差缩放。它负责突出深层特征中的高频边界和突变结构。

---

## 9. 梯度解码器 `GradientDecoder`

`GradientDecoder` 从编码器特征中解码两类辅助输出：

1. `g`：输入图像梯度图。
2. `s`：高反光软概率图。

同时，它输出 4 个中间特征，供 `EnhancementLayers` 进一步处理。

### 9.1 输入

```text
backbone_features = (x_bb_1, x_bb_2, x_bb_3, x_bb_4)
x_c               = 编码器瓶颈特征
original_input    = 原始输入 x
```

### 9.2 内部数据流

| 阶段 | 输入特征 | 操作 | 中间输出 |
|---|---|---|---|
| H/16 全局融合 | `cat(x_bb_4, x_c)` | `fuse_global_context` | `x_bb_4_fused` |
| Block 1 | `x_bb_4_fused` | `7×7 -> 1×1 -> 3×3 -> FFC` | `xe1 [B,256,H/16,W/16]` |
| Block 2 | `cat(xe1, x_bb_4)` | `3×3 -> FFC -> ConvTranspose stride2` | `xe2 [B,128,H/8,W/8]` |
| Block 3 | `cat(xe2, x_bb_3)` | `3×3 -> FFC -> ConvTranspose stride2` | `xe3 [B,64,H/4,W/4]` |
| Block 4 | `cat(xe3, x_bb_2)` | `3×3 -> ConvTranspose stride2` | `xe4 [B,32,H/2,W/2]` |
| Block 5 | `cat(xe4, x_bb_1)` | `ConvTranspose stride2` | 全分辨率共享特征 |

全分辨率共享特征会与原始输入拼接：

```text
x_concat = cat(shared_feature, original_input)
```

然后进入两个并列输出头：

```text
g = sigmoid(block_5_conv_2(x_concat))
s = sigmoid(spec_lik_head(x_concat))
```

### 9.3 梯度头

梯度头输出 `g`，监督目标为增强后输入图像的 Sobel 梯度幅值 `gt_gradient`。它提供结构突变、边界和条纹局部变化的辅助信息。

### 9.4 高反光概率头

高反光概率头输出 `s`，监督目标为：

```text
gt_specular = clamp(input - target, 0, 1)
```

`s` 不只作为辅助输出，也会进入：

1. `EnhancementLayers` 的 `specular_gate`。
2. `MainDecoder` 的联合门控和独立高反光概率门控。
3. 训练损失中的 `s_lik` 与 `s_cons` 项。

---

## 10. 增强桥接模块 `EnhancementLayers`

`EnhancementLayers` 接收梯度解码器的 4 个中间特征，并结合 `g`、原始输入 `x`、高反光概率 `s` 生成 4 个增强特征，传给主解码器。

输入：

```text
to_be_enhanced_features = (xe1, xe2, xe3, xe4)
g_map = g
original_input = x
s_map = s
```

输出：

| 输出 | 形状 | 对应主解码器位置 |
|---|---:|---|
| `enh1` | `[B,128,H/16,W/16]` | MainDecoder Level 1 |
| `enh2` | `[B,64,H/8,W/8]` | MainDecoder Level 2 |
| `enh3` | `[B,32,H/4,W/4]` | MainDecoder Level 3 |
| `enh4` | `[B,16,H/2,W/2]` | MainDecoder Level 4 |

### 10.1 `_SpectralDecompositionUnit`

每一级增强单元包含：

```text
projection_branch: 大核空间投影
specular_branch:  小核高反光分支
ffc_specular:     对高反光分支做频域建模
gate_conv:        融合 projection 与 specular
edge_gate:        由 g_map 生成边缘门控
specular_gate:    由 cat(g_map, original_input, s_map) 生成高反光门控
```

核心计算：

```text
projection = projection_branch(x)
specular   = specular_branch(x)
specular   = specular + FFC(specular)
gate       = sigmoid(Conv(cat(projection, specular)))
specular_refined = specular + specular * gate

edge_gate = sigmoid(Conv(g_down))
spec_gate = sigmoid(Conv(cat(g_down, input_down, s_down)))
final_gate = edge_gate * spec_gate

output = specular_refined + beta * specular_refined * final_gate
```

该模块将梯度分支中的通用结构特征转换为与高反光重建更相关的增强特征。

---

## 11. 主解码器 `MainDecoder`

`MainDecoder` 是模型的主任务分支，负责输出：

1. 镜面反射分量 `b`。
2. 直接漫反射预测 `r_direct`。
3. 双路融合系数 `alpha_logit`。

### 11.1 输入

```text
c_features = (x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c)
enhanced_features = (enh1, enh2, enh3, enh4)
x_g = g
x_s = s
original_input = x
```

### 11.2 主解码器整体流程

```text
x_c
 -> FFC1 + FFC2 + residual
 -> MultiscaleConvBlock
 -> Level 1: H/16
 -> Level 2: H/8
 -> Level 3: H/4
 -> Level 4: H/2
 -> Level 5: H
 -> 输出头: b 与 r_direct
```

### 11.3 Level 1：H/16 深层融合

```text
x_c -> FFC1 -> FFC2 -> 拼接残差 -> MultiscaleConvBlock
    -> TransposedConvBlock_same
    -> SpecularAwareAttention(sa1)
    -> Attention_block(att1, x_bb_4)
    -> cat(decoder_feature, gated_x_bb_4, enh1)
```

输出通道组合：

```text
768 + 512 + 128 = 1408
```

### 11.4 Level 2：H/8 解码

```text
1408 -> TransposedConvBlock -> 384通道
     -> FFC_H8
     -> 三路先验调制
     -> SpecularAwareAttention(sa2)
     -> Attention_block(att2, x_bb_3)
     -> cat(decoder_feature, gated_x_bb_3, enh2)
```

三路先验调制为：

```text
x = x * (
    1
    + grad_gate_scale * grad_gate_2(g)
    + joint_gate_scale * joint_gate_2(cat(g, input, s))
    + spec_lik_gate_scale * spec_lik_gate_2(s)
)
```

### 11.5 Level 3：H/4 解码

```text
cat特征 -> FeatureExtrationLayersB -> 640通道
        -> TransposedConvBlock -> 192通道
        -> FFC_H4
        -> 三路先验调制
        -> SpecularAwareAttention(sa3)
        -> Attention_block(att3, x_bb_2)
        -> cat(decoder_feature, gated_x_bb_2, enh3)
```

### 11.6 Level 4：H/2 解码

```text
cat特征 -> TransposedConvBlock -> 96通道
        -> 三路先验调制
        -> SpecularAwareAttention(sa4)
        -> Attention_block(att4, x_bb_1)
        -> cat(decoder_feature, gated_x_bb_1, enh4)
```

### 11.7 Level 5：全分辨率输出

```text
cat特征 -> TransposedConvBlock -> 48通道
        -> cat(gradient g) -> 49通道
        -> output_reduce -> 32通道
        -> output_dilated
        -> residual skip
        -> output_conv3 -> 16通道
        -> b_head
        -> r_direct_head
```

输出：

```text
b        = sigmoid(output_conv4(x))
r_direct = sigmoid(output_r_direct_conv(x))
alpha_logit = dual_head_alpha_logit
```

---

## 12. `SpecularAwareAttention`

`SpecularAwareAttention` 位于主解码器的 4 个 level 中，分别为 `sa1`、`sa2`、`sa3`、`sa4`。

输入：

```text
features
original_input
```

流程：

1. 将原始输入插值到当前特征分辨率。
2. 计算局部均值与局部对比度。
3. 构造高反光先验：

\[
spec\_prior = clamp(0.75 \cdot input + 0.25 \cdot local\_contrast, 0, 1)
\]

4. 使用多个阈值生成多级软 mask。
5. 将特征与多级 mask 拼接，通过卷积生成调制注意力。
6. 输出残差调制后的特征：

\[
out = features \cdot (1 + scale \cdot attention)
\]

该模块使主解码器在不同尺度上都能够参考输入图像亮度与局部对比度。

---

## 13. 最终漫反射融合与残差精修

### 13.1 双路漫反射融合

主解码器输出 `b` 和 `r_direct` 后，`YFFC.forward` 首先构造减法路径：

\[
r_{sub} = clamp(x - b, 0, 1)
\]

然后用可学习系数融合直接预测路径与减法路径：

\[
\alpha = sigmoid(\alpha_{logit})
\]

\[
r_{fused} = \alpha \cdot r_{direct} + (1-\alpha) \cdot r_{sub}
\]

### 13.2 `ResidualRefinementHead`

`ResidualRefinementHead` 接收 3 个输入：

```text
coarse = r_fused
original = x
r_direct = r_direct
```

内部结构：

```text
cat(coarse, original, r_direct)
 -> 3×3 Conv 3->24 + GELU
 -> 轻量残差块 24->24->24
 -> GELU
 -> 3×3 Conv 24->16 + GELU
 -> 3×3 Conv 16->1 + Tanh
 -> delta
```

最终输出：

\[
r = clamp(r_{fused} + 0.15 \cdot \delta, 0, 1)
\]

残差精修头的作用是在最终输出阶段进行小幅像素级修正，并利用 `r_direct` 给出直接漫反射路径的参考。

---

## 14. 跨尺度一致性项 `cross_scale_consistency_loss`

该函数作用于编码器中 H/4、H/8、H/16 的融合特征：

```text
features = (x_bb_2, x_bb_3, x_bb_4)
```

每个特征先计算通道绝对均值响应图：

```text
response = mean(abs(feature), dim=channel)
```

然后做 per-sample 标准化，将 H/8、H/16 响应图插值到 H/4 分辨率，计算相邻尺度的 L1 差异：

\[
cs = 0.5 \cdot (L1(H4,H8) + L1(H8,H16))
\]

`cs` 不保存为图像结果，只作为训练损失的一部分。

---

## 15. `YFFC.forward` 完整前向流程

伪代码如下：

```python
def forward(x):
    c_feat = encoder(x)
    # c_feat = (x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c)

    g_feat = gradient_decoder(c_feat[:-1], c_feat[-1], original_input=x)
    # g_feat = (xe1, xe2, xe3, xe4, g_raw, s_raw)

    xe_feats = g_feat[:-2]
    g_raw = g_feat[-2]
    s_raw = g_feat[-1]

    enh = enhancement_layers(xe_feats, g_raw, x, s_raw)

    est_b, r_direct, alpha_logit = main_decoder(
        c_feat, enh, g_raw, s_raw, original_input=x
    )

    est_b = resize_and_safe(est_b)
    r_direct = resize_and_safe(r_direct)
    g_map = resize_and_safe(g_raw)
    s_map = resize_and_safe(s_raw)

    r_sub = clamp(x - est_b, 0, 1)
    alpha = sigmoid(alpha_logit)
    r_fused = alpha * r_direct + (1-alpha) * r_sub

    pred_diffuse = refinement_head(r_fused, x, r_direct)

    cs_loss = cross_scale_consistency_loss((x_bb_2, x_bb_3, x_bb_4))

    return {
        "r": pred_diffuse,
        "g": g_map,
        "b": est_b,
        "s": s_map,
        "r_direct": r_direct,
        "alpha": alpha,
        "cs": cs_loss,
    }
```

---

## 16. 训练数据流

### 16.1 数据集读取

`train.py` 使用 `ImprovedFringeDataset`。每个样本返回：

```text
input      [1,H,W]
target     [1,H,W]
gt_spec    [1,H,W]
gt_grad    [1,H,W]
```

文件匹配规则：

1. 输入文件名与目标文件名完全一致。
2. `input_` 前缀替换为 `target_`。
3. `input` 前缀替换为 `target`。
4. 从文件名中提取数字，尝试匹配 `target_数字.扩展名`。

图像会被转为灰度，并归一化到 `[0,1]`。

### 16.2 监督目标构造

若使用增强流水线，监督目标在增强后计算：

```text
gt_specular = clamp(input_img - target_img, 0, 1)
gt_gradient = compute_input_gradient_target(input_img)
```

其中 `gt_specular` 是高反光软标签，`gt_gradient` 是输入图像 Sobel 梯度幅值。

### 16.3 数据增强 `EnhancedPairedTransform`

训练集增强包括：

1. 水平翻转。
2. 垂直翻转。
3. 旋转；非正方形输入采用 180° 旋转。
4. 随机裁剪并 resize 回原尺寸。
5. 亮度缩放，同步作用于 input、target 和 gt_specular。
6. 合成高反光 `HighlightAugmentation`。
7. 计算增强后输入图像的 Sobel 梯度目标。

验证集默认不使用随机增强。

### 16.4 合成高反光 `HighlightAugmentation`

该模块在输入图像上随机叠加若干高斯 blob，并同步更新 `gt_specular`：

```text
input_img  = clamp(input_img + synth, 0, 1)
gt_specular = clamp(gt_specular + synth, 0, 1)
```

训练配置中默认：

```text
p = 0.5
num_range = (1, 6)
radius_range = (3, max(5, min_side//20))
intensity_range = (0.2, 0.7)
```

---

## 17. 损失函数 `EnhancedCompositeLoss`

训练损失由多个子损失加权求和。输入参数为：

```text
pred_diffuse  = out["r"]
target        = target diffuse
est_b         = out["b"]
gt_specular   = gt_spec
pred_grad     = out["g"]
gt_grad       = gt_grad
cross_scale_consistency = out["cs"]
input_img     = input
pred_s_lik    = out["s"]
pred_r_direct = out["r_direct"]
```

### 17.1 漫反射主损失

| 子损失 | 含义 |
|---|---|
| `ssim_mixed_loss` | `0.8 * MS-SSIM loss + 0.2 * SSIM loss` |
| `l1` | 边缘加权 Charbonnier 损失 |
| `freq` | 幅度谱差异 + 能量加权相位一致性 |
| `perceptual` | VGG 感知损失；若预训练权重不可用则关闭 |

### 17.2 镜面反射与高反光区域损失

| 子损失 | 含义 |
|---|---|
| `spec` | `est_b` 与 `gt_specular` 的 Charbonnier + SSIM 损失 |
| `focal` | 高反光区域 focal/OHEM 损失 |
| `dice` | 高反光区域 Dice 损失 |
| `gc` | 镜面分量梯度一致性损失 |
| `spec_peak` | 在强高反光区域惩罚 `est_b` 欠预测 |

### 17.3 梯度分支损失

| 子损失 | 含义 |
|---|---|
| `grad` | `pred_grad` 与 `gt_grad` 的 SSIM 损失 |
| `grad_edge` | 输入梯度权重图加权的 Charbonnier 损失 |
| `grad_spec` | 利用输入高亮先验约束梯度与镜面分量边界关系 |

### 17.4 高反光概率头损失

| 子损失 | 含义 |
|---|---|
| `s_lik` | `pred_s_lik` 与 `gt_specular` 的 BCE + SSIM 损失 |
| `s_cons` | `pred_s_lik` 与 `est_b` 在高反光区域的双向一致性 |

### 17.5 直接漫反射路径损失

| 子损失 | 含义 |
|---|---|
| `r_focal_hl` | 在高反光区域重点约束最终 `r` 与 target 的差异 |
| `r_dual_cons` | 约束 `r_direct` 与最终 `r` 的一致性 |
| `r_smooth` | 在高反光区域对最终 `r` 施加局部平滑先验 |

### 17.6 跨尺度一致性损失

| 子损失 | 含义 |
|---|---|
| `cs` | H/4、H/8、H/16 融合特征响应图一致性 |

### 17.7 总损失

总损失为：

```text
total =
  lambda_ssim       * l_ssim
+ lambda_l1         * l_l1
+ lambda_freq       * l_freq
+ lambda_perceptual * l_percep
+ lambda_spec       * l_spec
+ lambda_stat       * l_stat
+ lambda_grad       * l_grad
+ lambda_grad_edge  * l_grad_edge
+ lambda_grad_spec  * l_grad_spec
+ lambda_focal      * l_focal
+ lambda_dice       * l_dice
+ lambda_gc         * l_gc
+ lambda_cs         * l_cs
+ lambda_s_lik      * l_s_lik
+ lambda_s_cons     * l_s_cons
+ lambda_spec_peak  * l_spec_peak
+ lambda_r_focal_hl * l_r_focal_hl
+ lambda_r_dual_cons* l_r_dual_cons
+ lambda_r_smooth   * l_r_smooth
```

---

## 18. 动态损失权重 `ImprovedWeightScheduler`

`train.py` 使用 `ImprovedWeightScheduler` 在训练过程中更新 `EnhancedCompositeLoss` 中的权重。调度逻辑基于当前 epoch 在总 epoch 中的比例。

主要权重分组：

| 权重 | 训练作用 |
|---|---|
| `lambda_ssim` | 主结构相似度约束，固定为 1.0 |
| `lambda_l1` | 早期较大，后续保持下限 |
| `lambda_freq` | 逐步增加频域约束 |
| `lambda_perceptual` | 逐步增加感知约束 |
| `lambda_spec` | 镜面分量主监督 |
| `lambda_grad` | 梯度图 SSIM 监督 |
| `lambda_grad_edge` | 梯度边缘加权监督 |
| `lambda_focal`、`lambda_dice`、`lambda_gc` | 高反光区域结构监督 |
| `lambda_cs` | 跨尺度一致性 |
| `lambda_s_lik`、`lambda_s_cons` | 高反光概率头监督 |
| `lambda_spec_peak` | 强高反光区域峰值约束 |
| `lambda_r_focal_hl`、`lambda_r_dual_cons`、`lambda_r_smooth` | 直接保护最终漫反射输出 |

`criterion.update_weights(**w)` 会在每个 epoch 开始时更新损失权重。

---

## 19. 优化器与学习率分组

`train.py` 使用 `build_optimizer_v2` 创建 AdamW 优化器，按模块名称将参数划分为不同学习率组。

| 参数组 | 匹配模块 | 学习率 |
|---|---|---:|
| `encoder_pretrained` | ResNet stem/layer、Swin 主体 | `base_lr * 0.1` |
| `adaptive_skip_fusion` | `fuse_h4`、`fuse_h8`、`swin16_proj` | `base_lr` |
| `fusion` | 其余融合相关参数 | `base_lr * 0.5` |
| `decoders` | `gradient_decoder`、`main_decoder`、`enhancement_layers` | `base_lr` |
| `ffc` | 名称中包含 `ffc` 的参数 | `base_lr * 0.5` |
| `new_modules` | `sa1`-`sa4`、`hf_enhance`、`refinement_head`、`top_down_refine` | `base_lr` |

默认优化器：

```text
AdamW(weight_decay=1e-4)
```

学习率调度器由 `Y_FFC_unified.py` 中的 `build_scheduler` 构建，包含 warmup 与后续调度。训练脚本按 optimizer step 调用 `scheduler.step()`。

---

## 20. 训练循环

### 20.1 训练前初始化

`train_model(cfg)` 执行以下步骤：

1. 选择 CUDA 或 CPU。
2. 规范化 `IMAGE_SIZE` 为 `(H, W)`。
3. 创建输出目录。
4. 构建 `YFFC()`。
5. 根据配置执行 `replace_bn_with_gn(model, 8)`。
6. 统计总参数和可训练参数。
7. 构建训练集、验证集、DataLoader。
8. 构建 `EnhancedCompositeLoss`。
9. 构建分层 AdamW 优化器。
10. 构建学习率调度器。
11. 创建 AMP scaler 和 EMA。
12. 初始化历史记录与最优模型记录。

### 20.2 单个 epoch 的训练数据流

`train_one_epoch` 中每个 batch 的数据流为：

```text
(inp, tgt, gt_s, gt_g)
      │
      ▼
model(inp)
      │
      ├─ out["r"]
      ├─ out["b"]
      ├─ out["g"]
      ├─ out["s"]
      ├─ out["r_direct"]
      └─ out["cs"]
      │
      ▼
criterion(...)
      │
      ▼
loss / accumulation_steps
      │
      ▼
backward
      │
      ▼
梯度裁剪
      │
      ▼
optimizer.step
      │
      ▼
scheduler.step
      │
      ▼
ema.update
```

训练循环还包含：

- 输入、输出和 loss 的 NaN/Inf 检查。
- 异常 batch 跳过。
- 梯度范数检查。
- optimizer state 数值修复。
- 损失尖峰保护。
- tqdm 进度条显示当前 loss、SSIM 和 L1。

### 20.3 验证流程

`validate` 使用 `model.eval()` 和 `torch.no_grad()`。验证阶段可选择 TTA，但当前训练中的模型选择默认不启用 TTA，使不同 epoch 之间指标口径一致。

若启用 EMA，验证前执行：

```text
ema.apply_shadow()
```

验证后执行：

```text
ema.restore()
```

### 20.4 指标记录

每个 epoch 记录：

| 指标 | 含义 |
|---|---|
| `train_loss` / `val_loss` | 平均总损失 |
| `train_ssim` / `val_ssim` | 漫反射输出 SSIM |
| `train_mae` / `val_mae` | 漫反射输出 MAE |
| `train_mse` / `val_mse` | 漫反射输出 MSE |
| `train_psnr` / `val_psnr` | 漫反射输出 PSNR |
| `train_sub_losses` / `val_sub_losses` | 所有子损失的 epoch 平均值 |

### 20.5 模型保存

训练脚本保存以下模型：

| 文件名 | 选择标准 |
|---|---|
| `best_model.pth` | 综合评分最优 |
| `best_model_best_ssim.pth` | 验证 SSIM 最优 |
| `best_model_best_loss.pth` | 验证 Loss 最低 |
| `last_model.pth` | 训练结束时的最后一轮 |
| `checkpoint_epoch_*.pth` | 定期保存的 checkpoint |

综合评分为：

\[
score = 0.7 \cdot SSIM + 0.3 \cdot \frac{1}{1+Loss}
\]

保存内容包括模型权重、optimizer 状态、epoch、best_ssim、val_loss、image_size、EMA 权重和 AMP scaler 状态等。

---

## 21. 默认训练配置

`train.py` 的默认配置如下：

| 参数 | 默认值 |
|---|---:|
| `EPOCHS` | `300` |
| `BATCH_SIZE` | `3` |
| `BASE_LR` | `2e-4` |
| `WEIGHT_DECAY` | `1e-4` |
| `ACCUMULATION_STEPS` | `3` |
| `GRAD_CLIP_VALUE` | `1.0` |
| `IMAGE_SIZE` | `(128, 160)` |
| `TRAIN_SPLIT_RATIO` | `0.8` |
| `NUM_WORKERS` | `5` |
| `USE_GN` | `True` |
| `USE_EMA` | `True` |
| `EMA_DECAY` | `0.9995` |
| `USE_OHEM` | `True` |
| `USE_AMP` | `False` |
| `USE_HIGHLIGHT_AUG` | `True` |
| `HIGHLIGHT_AUG_P` | `0.5` |
| `SAVE_EVERY_N_EPOCHS` | `20` |
| `VIS_EVERY_N_EPOCHS` | `10` |
| `VIS_NUM_SAMPLES` | `2` |
| `USE_EARLY_STOPPING` | `False` |
| `EARLY_STOPPING_PATIENCE` | `30` |

---

## 22. 推理流程 `process_images.py`

### 22.1 模型加载

`load_model(model_path, device, prefer_ema=True)` 支持两种 checkpoint 格式：

1. 完整训练 checkpoint，包含 `model_state_dict`。
2. 纯 `state_dict`。

加载流程：

```text
torch.load checkpoint
 -> 构建 YFFC(swin_pretrained=False)
 -> 判断 checkpoint 是否使用 GroupNorm
 -> 如需要则 replace_bn_with_gn
 -> 若存在 EMA 且 prefer_ema=True，则合并 EMA 权重
 -> flexible load state_dict
 -> model.to(device)
 -> model.eval()
```

`_load_state_dict_flexible` 会优先尝试严格加载。如果结构或权重形状不完全匹配，会筛选出名称和形状都兼容的张量并以 `strict=False` 加载。

### 22.2 输入图像加载

`load_image` 会执行：

```text
PIL open
 -> 转灰度 L
 -> 可选 resize
 -> numpy float32 / 255.0
 -> torch tensor [1,1,H,W]
```

推理默认保持原始尺寸，不直接 resize 原图。

### 22.3 训练 patch 尺寸解析

`resolve_patch_size_from_checkpoint` 优先读取 checkpoint 中保存的 `image_size`。如果 checkpoint 记录的训练尺寸与脚本配置的 `TRAIN_IMAGE_SIZE` 不一致，推理会使用 checkpoint 中的尺寸进行滑窗。

### 22.4 滑窗推理

`sliding_window_inference` 用于大图推理。流程：

1. 将 `patch_size` 标准化为 `(patch_h, patch_w)`。
2. 根据 `overlap` 计算 stride。
3. 只在右侧和下侧 padding，使图像尺寸落在滑窗网格上。
4. 生成所有 patch 起点。
5. 每个 patch 单独送入模型。
6. 对输出 `r`、`b`、`g`、`s`、`r_direct` 使用 Tukey 窗加权累加。
7. 除以 `weight_sum` 得到融合结果。
8. 裁回原始图像尺寸。

输出字典中至少包含：

```text
r, b, g
```

若模型输出存在，也包含：

```text
s, r_direct
```

### 22.5 TTA 推理

`_tta_forward` 对同一输入执行 4 次翻转组合：

```text
原图
水平翻转
垂直翻转
水平+垂直翻转
```

每次输出翻转回原方向后，对 `b`、`r_direct`、`g`、`s` 取平均。随后使用与 `YFFC.forward` 相同的双路融合和残差精修逻辑重新计算最终 `r`：

```text
avg_b, avg_r_direct -> r_sub -> r_fused -> refinement_head -> r
```

### 22.6 输出保存

单图推理会保存：

| 文件 | 含义 |
|---|---|
| `*_diffuse.png` | 最终去高反光结果 `r` |
| `*_specular.png` | 镜面反射分量 `b` |
| `*_gradient.png` | 梯度图 `g` |
| `*_spec_likelihood.png` | 高反光概率图 `s` |
| `*_visualization.png` | 输入、输出和中间结果可视化 |
| `*_diffuse_16bit.png` | 可选 16-bit 漫反射结果 |
| `*_specular_16bit.png` | 可选 16-bit 镜面分量 |
| `*_gradient_16bit.png` | 可选 16-bit 梯度图 |
| `*_spec_likelihood_16bit.png` | 可选 16-bit 高反光概率图 |

批量推理时，每张图像会创建独立子目录，同时可将所有漫反射结果复制到统一的 `all_diffuse` 文件夹。

---

## 23. 训练阶段完整数据流总结

从单个训练 batch 的角度，完整数据流为：

```text
1. ImprovedFringeDataset
   input image, target image
      -> 灰度化
      -> resize 到 IMAGE_SIZE
      -> 归一化到 [0,1]
      -> 配对增强
      -> gt_specular = clamp(input - target, 0, 1)
      -> gt_gradient = Sobel(input)

2. DataLoader
   batch:
      inp [B,1,H,W]
      tgt [B,1,H,W]
      gt_s [B,1,H,W]
      gt_g [B,1,H,W]

3. YFFC.forward(inp)
   Encoder:
      inp -> x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c
   GradientDecoder:
      encoder features + x_c + inp -> xe1..xe4, pred_g, pred_s
   EnhancementLayers:
      xe1..xe4 + pred_g + inp + pred_s -> enh1..enh4
   MainDecoder:
      encoder features + enhanced features + pred_g + pred_s + inp
      -> pred_b, pred_r_direct, alpha_logit
   Fusion:
      pred_b + pred_r_direct + inp -> pred_r
   Consistency:
      x_bb_2, x_bb_3, x_bb_4 -> cs

4. EnhancedCompositeLoss
   pred_r vs tgt
   pred_b vs gt_s
   pred_g vs gt_g
   pred_s vs gt_s
   pred_r_direct vs pred_r
   cs as auxiliary scalar
      -> total loss + loss_dict

5. Optimizer step
   loss / accumulation_steps
      -> backward
      -> clip_grad_norm
      -> optimizer.step
      -> scheduler.step
      -> ema.update
```

---

## 24. 推理阶段完整数据流总结

```text
1. load_model
   checkpoint -> YFFC -> weights -> eval mode

2. load_image
   image -> grayscale -> [1,1,H,W] -> [0,1]

3. patch decision
   if image size == training patch size:
       whole-image inference
   else:
       sliding-window inference

4. per patch forward
   patch -> YFFC -> r,b,g,s,r_direct

5. optional TTA
   4 flip variants -> average outputs -> recompute r

6. patch fusion
   Tukey window weighted accumulation
   crop back to original size

7. save
   diffuse/specular/gradient/spec_likelihood/visualization
```

---

## 25. 主要模块位置索引

| 模块 | 文件 | 位置/角色 |
|---|---|---|
| `YFFC` | `Y_FFC_unified.py` | 主网络入口 |
| `Encoder` | `Y_FFC_unified.py` | 双分支编码器 |
| `SwinEncoderBranch` | `Y_FFC_unified.py` | Swin-Tiny 上下文分支 |
| `AdaptiveSkipFusion` | `Y_FFC_unified.py` | H/4、H/8 skip 融合 |
| `CrossModalFusion` | `Y_FFC_unified.py` | H/16 CNN-Swin 双向交叉注意力 |
| `CrossScaleTopDownRefinement` | `Y_FFC_unified.py` | H/16 到 H/8/H/4 的 top-down 回流 |
| `HighFrequencyEnhancement` | `Y_FFC_unified.py` | 编码器瓶颈后的高频增强 |
| `GradientDecoder` | `Y_FFC_unified.py` | 梯度图和高反光概率图分支 |
| `EnhancementLayers` | `Y_FFC_unified.py` | 将梯度分支中间特征转为主解码增强特征 |
| `_SpectralDecompositionUnit` | `Y_FFC_unified.py` | 增强层内部单元 |
| `MainDecoder` | `Y_FFC_unified.py` | 镜面分量和直接漫反射解码器 |
| `SpecularAwareAttention` | `Y_FFC_unified.py` | 主解码器多尺度亮度/对比度注意力 |
| `ResidualRefinementHead` | `Y_FFC_unified.py` | 最终漫反射残差精修 |
| `EnhancedCompositeLoss` | `Y_FFC_unified.py` | 训练复合损失 |
| `ImprovedFringeDataset` | `train.py` | 工程训练数据集 |
| `ImprovedWeightScheduler` | `train.py` | 动态损失权重 |
| `build_optimizer_v2` | `train.py` | 分层学习率 AdamW |
| `train_one_epoch` | `train.py` | 单轮训练 |
| `validate` | `train.py` | 验证 |
| `train_model` | `train.py` | 完整训练流程 |
| `load_model` | `process_images.py` | 推理模型加载 |
| `sliding_window_inference` | `process_images.py` | 大图滑窗推理 |
| `_tta_forward` | `process_images.py` | 4 翻转 TTA 推理 |
| `process_single_image` | `process_images.py` | 单图处理 |
| `process_folder` | `process_images.py` | 文件夹批处理 |

---

## 26. 模型结构简表

| 阶段 | 输入 | 输出 | 核心模块 |
|---|---|---|---|
| 输入 | `[B,1,H,W]` | `[B,1,H,W]` | 归一化灰度图 |
| 编码浅层 | `x` | `x_bb_1` | ResNet stem |
| 编码 H/4 | ResNet layer1 + Swin H/4 | `x_bb_2` | `AdaptiveSkipFusion` |
| 编码 H/8 | ResNet layer2 + Swin H/8 | `x_bb_3` | `AdaptiveSkipFusion` |
| 编码 H/16 | ResNet layer3/4 + Swin H/16 | `x_bb_4` | `CrossModalFusion` + `CrossScaleTopDownRefinement` |
| 瓶颈 | `x_bb_4` | `x_c` | `FFC_BN_ACT` + `HighFrequencyEnhancement` |
| 辅助解码 | `x_bb_1..4, x_c` | `xe1..4, g, s` | `GradientDecoder` |
| 增强桥接 | `xe1..4, g, x, s` | `enh1..4` | `EnhancementLayers` |
| 主解码 | `c_features, enh, g, s, x` | `b, r_direct, alpha_logit` | `MainDecoder` |
| 融合 | `x, b, r_direct, alpha` | `r_fused` | 双路漫反射融合 |
| 精修 | `r_fused, x, r_direct` | `r` | `ResidualRefinementHead` |
| 训练辅助 | `x_bb_2,x_bb_3,x_bb_4` | `cs` | `cross_scale_consistency_loss` |

---

## 27. 输出含义与使用建议

训练和推理中最重要的输出是：

- `r`：最终去高反光图像，应作为主要结果保存和评价。
- `b`：网络估计出的高反光分量，可用于分析高光位置和强度。
- `g`：网络预测的结构梯度图，可用于观察模型对边界和条纹变化的关注。
- `s`：高反光软概率图，可用于观察模型对高反光区域的定位。
- `r_direct`：主解码器直接预测的漫反射路径，训练中参与损失，推理中用于最终融合。
- `cs`：训练辅助标量，不作为图像结果使用。

在 `process_images.py` 中，最终交付图像应优先使用 `*_diffuse.png`；`*_specular.png`、`*_gradient.png`、`*_spec_likelihood.png` 更适合作为调试、分析和可视化结果。

---

## 28. 结论

当前 X-FFC / YFFC 模型采用双编码器结构，将 ResNet50 的局部纹理特征与 Swin-Tiny 的窗口上下文特征在多尺度上融合。编码器输出同时服务于梯度辅助分支与主解码分支。梯度辅助分支输出输入梯度图和高反光概率图，并通过增强层向主解码器提供多尺度结构化特征。主解码器输出镜面分量和直接漫反射分量，最终通过可学习双路融合与残差精修得到漫反射结果。

训练系统围绕漫反射质量、高反光分量、高反光概率、梯度结构、跨尺度一致性和高反光区域局部行为构建复合损失。推理系统支持原尺寸输入、按训练 patch 尺寸滑窗、Tukey 窗融合、EMA 权重加载和 4 翻转 TTA，能够在单图和批量图像处理场景中输出完整的结果文件。
