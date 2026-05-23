# Y-FFC / X-FFC 当前模型关键模块问答总结

> 本文档基于你当前上传的模型说明文档、`Y_FFC_unified.py`、`train.py` 和 `process_images.py` 中的实际实现整理。  
> 重点总结你前面提出的几个疑问：`CrossScaleTopDownRefinement`、`_SpectralDecompositionUnit`、`s` 高光软概率图、`cs` 跨尺度一致性、`MainDecoder` 双头输出、`SpecularAwareAttention`、`Attention_block`，以及 FFC 模块的层级结构。

---

## 0. 当前模型总体前向流程

当前 `YFFC.forward(x)` 可以理解为下面这条主流程：

```text
输入 x [B,1,H,W]
        │
        ▼
Encoder
        │
        ├─ x_bb_1 [B, 64,H/2, W/2]
        ├─ x_bb_2 [B,128,H/4, W/4]   ← 已经过 Swin 融合 + top-down 回流
        ├─ x_bb_3 [B,256,H/8, W/8]   ← 已经过 Swin 融合 + top-down 回流
        ├─ x_bb_4 [B,512,H/16,W/16]  ← H/16 CrossModalFusion + top-down 更新
        └─ x_c    [B,256,H/16,W/16]  ← 由更新后的 x_bb_4 继续经过 CBA + FFC + 高频增强得到
        │
        ▼
GradientDecoder
        │
        ├─ xe1 [B,256,H/16,W/16]
        ├─ xe2 [B,128,H/8, W/8]
        ├─ xe3 [B, 64,H/4, W/4]
        ├─ xe4 [B, 32,H/2, W/2]
        ├─ g   [B,  1,H,   W]        ← 梯度预测图
        └─ s   [B,  1,H,   W]        ← 高光软概率图 / 高反光 likelihood
        │
        ▼
EnhancementLayers
        │
        ├─ decomp_unit_1(xe1, g, x, s) → enh1 [B,128,H/16,W/16]
        ├─ decomp_unit_2(xe2, g, x, s) → enh2 [B, 64,H/8, W/8]
        ├─ decomp_unit_3(xe3, g, x, s) → enh3 [B, 32,H/4, W/4]
        └─ decomp_unit_4(xe4, g, x, s) → enh4 [B, 16,H/2, W/2]
        │
        ▼
MainDecoder
        │
        ├─ 使用 Encoder 特征：x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c
        ├─ 使用增强特征：enh1, enh2, enh3, enh4
        ├─ 使用辅助图：g, s, original_input
        │
        ├─ 输出 b        ：镜面反射 / 高反光分量
        ├─ 输出 r_direct ：直接预测的漫反射图
        └─ 输出 alpha_logit
        │
        ▼
双路漫反射融合
        │
        ├─ r_sub   = clamp(x - b, 0, 1)
        ├─ alpha   = sigmoid(alpha_logit)
        ├─ r_fused = alpha * r_direct + (1 - alpha) * r_sub
        └─ r       = ResidualRefinementHead(r_fused, x, r_direct)
```

最终模型输出字典大致为：

```text
{
  "r":        最终漫反射 / 去高反光图像,
  "b":        镜面反射 / 高反光分量,
  "g":        梯度预测图,
  "s":        高光软概率图,
  "r_direct": 主解码器直接预测的漫反射图,
  "alpha":    双路融合系数,
  "cs":       跨尺度一致性训练辅助项
}
```

---

# 1. CrossScaleTopDownRefinement

## 1.1 这个模块在哪里？

`CrossScaleTopDownRefinement` 定义在 `Y_FFC_unified.py` 中，属于 `Encoder` 内部使用的跨尺度回流模块。

在 `Encoder.__init__()` 中初始化：

```python
self.top_down_refine = CrossScaleTopDownRefinement(
    h4_channels=128,
    h8_channels=256,
    h16_channels=512,
    init_gamma=0.05
)
```

在 `Encoder.forward()` 中调用：

```python
fused = self.cross_modal_fusion(x_bb_4, swin_feat_16)
x_bb_2, x_bb_3, x_bb_4 = self.top_down_refine(x_bb_2, x_bb_3, fused)
```

---

## 1.2 它的输入是什么？

`CrossScaleTopDownRefinement.forward()` 的输入是：

```python
def forward(self, h4, h8, h16):
```

在当前模型里对应：

```text
h4  = x_bb_2  [B,128,H/4, W/4]
h8  = x_bb_3  [B,256,H/8, W/8]
h16 = fused   [B,512,H/16,W/16]
```

但这里要注意：  
它的输入不是最原始的 ResNet 特征，而是已经完成 Swin 融合后的特征。

更完整地说：

```text
输入图像 x
   │
   ├──────────────────────────────┐
   │                              │
   ▼                              ▼
ResNet50 分支                  Swin-Tiny 分支
   │                              │
   ├─ x_bb_1 [B, 64,H/2, W/2]     │
   ├─ x_bb_2 [B,128,H/4, W/4] ◄── swin_4  [B, 96,H/4, W/4]
   ├─ x_bb_3 [B,256,H/8, W/8] ◄── swin_8  [B,192,H/8, W/8]
   └─ x_bb_4 [B,512,H/16,W/16]◄── swin_16 [B,384,H/16,W/16]
```

然后：

```text
H/4:
x_bb_2 + swin_4
        ↓
AdaptiveSkipFusion
        ↓
融合后的 x_bb_2

H/8:
x_bb_3 + swin_8
        ↓
AdaptiveSkipFusion
        ↓
融合后的 x_bb_3

H/16:
swin_16
        ↓
swin16_proj: 384 → 512
        ↓
CrossModalFusion(x_bb_4, swin_feat_16)
        ↓
fused [B,512,H/16,W/16]
```

所以 `CrossScaleTopDownRefinement` 的真实输入是：

```text
CrossScaleTopDownRefinement(
    h4  = AdaptiveSkipFusion 后的 x_bb_2,
    h8  = AdaptiveSkipFusion 后的 x_bb_3,
    h16 = CrossModalFusion 后的 fused
)
```

---

## 1.3 它的输出是什么？

它返回 3 个尺度的更新特征：

```python
return h4_out, h8_out, h16_out
```

在 `Encoder.forward()` 中重新赋值为：

```python
x_bb_2, x_bb_3, x_bb_4 = self.top_down_refine(x_bb_2, x_bb_3, fused)
```

所以输出是：

```text
x_bb_2 [B,128,H/4, W/4]   ← 更新后的 H/4 skip 特征
x_bb_3 [B,256,H/8, W/8]   ← 更新后的 H/8 skip 特征
x_bb_4 [B,512,H/16,W/16]  ← 更新后的 H/16 深层特征
```

注意：

```text
CrossScaleTopDownRefinement 本身不直接输出 x_c。
```

`x_c` 是在它之后由更新后的 `x_bb_4` 继续生成的：

```python
x_c = self.cba_after_backbone(x_bb_4)

nl = 256 - int(256 * 0.5)
out = self.ffc_bottleneck((x_c[:, :nl], x_c[:, nl:]))
x_c = torch.cat([out[0], out[1]], 1) + x_c
x_c = self.hf_enhance(x_c)
```

因此：

```text
x_c 是基于 top-down 更新后的 x_bb_4 生成的瓶颈特征。
```

---

## 1.4 CrossScaleTopDownRefinement 的内部流程

该模块的核心是：

```text
H/16 深层上下文
        ↓
回流到 H/8
        ↓
再回流到 H/4
```

具体结构可以写成：

```text
CrossScaleTopDownRefinement
│
├── 输入:
│   ├── h4  [B,128,H/4, W/4]
│   ├── h8  [B,256,H/8, W/8]
│   └── h16 [B,512,H/16,W/16]
│
├── Step 1: H/16 自身精炼
│   ├── h16_ctx = Conv3×3_BN_ReLU(h16)
│   └── h16_out = h16 + gamma * h16_ctx
│
├── Step 2: H/16 → H/8 回流
│   ├── h8_from_h16 = Conv1×1(h16_ctx)       # 512 → 256
│   ├── h8_from_h16 = Upsample 到 H/8
│   ├── h8_gate = Sigmoid(Conv1×1(cat(h8, h8_from_h16)))
│   └── h8_out = h8 + gamma * h8_gate * h8_from_h16
│
├── Step 3: H/8 → H/4 回流
│   ├── h8_ctx = Conv3×3_BN_ReLU(h8_out)
│   ├── h4_from_h8 = Conv1×1(h8_ctx)         # 256 → 128
│   ├── h4_from_h8 = Upsample 到 H/4
│   ├── h4_gate = Sigmoid(Conv1×1(cat(h4, h4_from_h8)))
│   └── h4_out = h4 + gamma * h4_gate * h4_from_h8
│
└── 输出:
    ├── h4_out  → x_bb_2
    ├── h8_out  → x_bb_3
    └── h16_out → x_bb_4
```

其中 `gamma` 是一个可学习残差强度：

```text
gamma 初始值为 0.05，并被 clamp 到 [0, 0.2]
```

这说明该模块是“轻量回流”，不是强行改写浅层 skip。

---

## 1.5 它的作用是什么？

`CrossScaleTopDownRefinement` 的作用可以概括为：

```text
把 H/16 深层融合特征中的全局上下文和高光区域判断，
逐级回流到 H/8 和 H/4 的浅层细节特征中。
```

这样做的原因是：

```text
H/4 特征：更擅长条纹边缘、局部纹理、细节
H/8 特征：更擅长局部区域结构
H/16 特征：更擅长全局上下文、高光区域整体判断
```

如果没有 top-down 回流，浅层 skip 可能只知道局部哪里亮、哪里有边缘，但不知道这些边缘是不是高光相关。  
加入该模块后，浅层 skip 会被深层语义“提醒”：

```text
这个局部亮边更可能是正常条纹？
还是高反光边界？
```

最终效果：

```text
1. 增强浅层 skip 的语义判断能力
2. 减少多尺度特征冲突
3. 让主解码器和梯度解码器拿到更一致的 skip
4. 有助于降低高光残留和正常条纹误去除
```

---

## 1.6 CrossScaleTopDownRefinement 的输出是不是后面 skip 到解码器？

是的。

`Encoder` 最后返回：

```python
return x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c
```

其中：

```text
x_bb_2、x_bb_3、x_bb_4 是 CrossScaleTopDownRefinement 更新后的特征。
x_c 是基于更新后的 x_bb_4 继续生成的瓶颈特征。
```

这些特征会进入：

```text
GradientDecoder
MainDecoder
```

更具体地说：

```text
Encoder 输出:
x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c
        │
        ├── GradientDecoder
        │       ├─ 使用 x_c 作为深层输入
        │       └─ 使用 x_bb_4/x_bb_3/x_bb_2/x_bb_1 作为逐级 skip
        │
        └── MainDecoder
                ├─ 使用 x_c 作为主解码起点
                └─ 使用 x_bb_4/x_bb_3/x_bb_2/x_bb_1 作为 skip
```

---

# 2. _SpectralDecompositionUnit 与 EnhancementLayers

## 2.1 这个模块在哪里？

`_SpectralDecompositionUnit` 定义在 `Y_FFC_unified.py` 中，是 `EnhancementLayers` 内部使用的子模块。

初始化位置：

```python
class EnhancementLayers(nn.Module):
    def __init__(self):
        self.decomp_unit_1 = _SpectralDecompositionUnit(in_c=256, out_c=128)
        self.decomp_unit_2 = _SpectralDecompositionUnit(in_c=128, out_c=64)
        self.decomp_unit_3 = _SpectralDecompositionUnit(in_c=64,  out_c=32)
        self.decomp_unit_4 = _SpectralDecompositionUnit(in_c=32,  out_c=16)
```

调用位置：

```python
def forward(self, to_be_enhanced_features, g_map, original_input, s_map):
    x1 = self.decomp_unit_1(to_be_enhanced_features[0], g_map, original_input, s_map)
    x2 = self.decomp_unit_2(to_be_enhanced_features[1], g_map, original_input, s_map)
    x3 = self.decomp_unit_3(to_be_enhanced_features[2], g_map, original_input, s_map)
    x4 = self.decomp_unit_4(to_be_enhanced_features[3], g_map, original_input, s_map)
    return x1, x2, x3, x4
```

---

## 2.2 它处在整个模型的什么位置？

它位于：

```text
GradientDecoder 和 MainDecoder 之间
```

完整关系是：

```text
GradientDecoder
   ├─ xe1 [B,256,H/16,W/16]
   ├─ xe2 [B,128,H/8, W/8]
   ├─ xe3 [B, 64,H/4, W/4]
   ├─ xe4 [B, 32,H/2, W/2]
   ├─ g   [B,  1,H,   W]
   └─ s   [B,  1,H,   W]
        │
        ▼
EnhancementLayers
   ├─ _SpectralDecompositionUnit 1
   ├─ _SpectralDecompositionUnit 2
   ├─ _SpectralDecompositionUnit 3
   └─ _SpectralDecompositionUnit 4
        │
        ▼
enh1, enh2, enh3, enh4
        │
        ▼
MainDecoder
```

---

## 2.3 你问的“xe + g + s → enh”是不是对的？

大方向是对的，但要更精确。

你可以写成：

```text
decomp_unit_1(xe1, g, input, s) → enh1
decomp_unit_2(xe2, g, input, s) → enh2
decomp_unit_3(xe3, g, input, s) → enh3
decomp_unit_4(xe4, g, input, s) → enh4
```

也就是说：

```text
g 和 s 参与了每一级 _SpectralDecompositionUnit。
```

但它不是简单地把 `xe_i + g + s` 做相加，也不是简单 concat 后卷积。  
更准确地说：

```text
xe_i 是主特征输入；
g、s、original_input 是先验图，用来生成门控；
门控决定 xe_i 中哪些位置应该被增强成高反光相关特征。
```

---

## 2.4 四个尺度的输入输出

```text
EnhancementLayers
│
├── decomp_unit_1
│   ├── 输入: xe1 [B,256,H/16,W/16], g [B,1,H,W], input [B,1,H,W], s [B,1,H,W]
│   └── 输出: enh1 [B,128,H/16,W/16]
│
├── decomp_unit_2
│   ├── 输入: xe2 [B,128,H/8,W/8], g [B,1,H,W], input [B,1,H,W], s [B,1,H,W]
│   └── 输出: enh2 [B,64,H/8,W/8]
│
├── decomp_unit_3
│   ├── 输入: xe3 [B,64,H/4,W/4], g [B,1,H,W], input [B,1,H,W], s [B,1,H,W]
│   └── 输出: enh3 [B,32,H/4,W/4]
│
└── decomp_unit_4
    ├── 输入: xe4 [B,32,H/2,W/2], g [B,1,H,W], input [B,1,H,W], s [B,1,H,W]
    └── 输出: enh4 [B,16,H/2,W/2]
```

每一级都会把 `g`、`s`、`input` resize 到当前 `xe_i` 的空间尺寸。

---

## 2.5 _SpectralDecompositionUnit 内部结构

单个 `_SpectralDecompositionUnit` 可以写成：

```text
_SpectralDecompositionUnit(x, g_map, original_input, s_map)
│
├── 输入:
│   ├── x              : 当前尺度的梯度分支特征 xe_i
│   ├── g_map          : 梯度预测图
│   ├── original_input : 原始输入灰度图
│   └── s_map          : 高光软概率图
│
├── 1. 主特征分解
│   │
│   ├── projection_branch
│   │       └── Conv7×7_BN_ReLU(x)
│   │           → projection
│   │
│   └── specular_branch
│           └── Conv3×3_BN_ReLU(x)
│               → specular
│
├── 2. specular 分支进入 FFC 频域增强
│   │
│   ├── specular 分为局部 L 和全局 G
│   ├── FFC_BN_ACT(specular_L, specular_G)
│   └── specular = FFC_out + specular
│
├── 3. projection 与 specular 生成内部门控
│   │
│   ├── gate = Sigmoid(Conv1×1(cat(projection, specular)))
│   └── specular_refined = specular + specular * gate
│
├── 4. 准备先验图
│   │
│   ├── g_down     = resize(g_map)          到当前特征尺寸
│   ├── input_down = resize(original_input) 到当前特征尺寸
│   └── s_down     = resize(s_map)          到当前特征尺寸
│
├── 5. 生成高光相关门控
│   │
│   ├── edge_gate = Sigmoid(Conv3×3(g_down))
│   │       └── 表示哪里有结构突变 / 边缘
│   │
│   ├── spec_gate = Sigmoid(Conv3×3(cat(g_down, input_down, s_down)))
│   │       └── 表示哪些结构突变更像高反光
│   │
│   └── final_gate = edge_gate * spec_gate
│
└── 6. 输出增强特征
    │
    └── output = specular_refined
                 + beta * specular_refined * final_gate
```

其中 `beta` 是可学习强度，限制在 `[0,1]`。

---

## 2.6 它的核心作用

`_SpectralDecompositionUnit` 的核心作用是：

```text
把 GradientDecoder 产生的通用梯度/结构特征 xe_i，
转换成对高反光分解更有用的增强特征 enh_i。
```

它不是直接预测最终图像，而是作为桥接层：

```text
GradientDecoder 的结构特征
        ↓
结合 g、s、input 进行高光语义门控
        ↓
得到 enh1~enh4
        ↓
送入 MainDecoder 辅助预测 b 和 r_direct
```

没有 `s` 时，它只能依赖：

```text
亮度 + 梯度
```

去猜高光。  
加入 `s` 后，它可以利用：

```text
亮度 + 梯度 + 有监督高光概率
```

来判断高光相关区域，因此更可靠。

---

# 3. s：高光软概率图

## 3.1 s 是什么？

`s` 是 `GradientDecoder` 输出的一张高光软概率图，也可以叫：

```text
高反光 likelihood map
高光软 mask
高反光语义先验图
```

形状为：

```text
s [B,1,H,W]
```

取值范围为：

```text
0 ~ 1
```

含义是：

```text
越接近 0：越不像高反光区域
越接近 1：越像高反光区域
```

---

## 3.2 s 和 b 的区别

这是非常重要的一点：

```text
b：镜面反射 / 高反光分量强度图
s：高反光概率 / 高光区域软掩码
```

也就是说：

```text
b 问的是：这个位置应该减掉多少高光？
s 问的是：这个位置是不是高光区域？
```

举例：

```text
input = 0.9
target = 0.4
gt_specular = input - target = 0.5
```

理想情况下：

```text
b 应该接近 0.5
s 应该较高，例如 0.8 ~ 1.0
```

---

## 3.3 s 从哪里来？

`GradientDecoder` 尾部有两个并列头：

```text
GradientDecoder
│
├── 梯度头:
│   └── 输出 g，监督目标 = Sobel(input)
│
└── 高光概率头:
    └── 输出 s，监督目标 = gt_specular = clamp(input - target, 0, 1)
```

所以 `s` 不是无监督注意力图，而是有明确监督目标的辅助输出。

---

## 3.4 为什么要加入 s？

因为只有 `g` 不够。

`g` 只能告诉模型：

```text
哪里有边缘
哪里有梯度
哪里有结构突变
```

但它不能告诉模型：

```text
这个边缘是不是高光边缘？
这个亮边是不是正常条纹？
这个强梯度是不是物体轮廓？
```

结构光条纹图像中常见情况：

```text
正常条纹边缘：g 高，但不应该去掉
物体轮廓边缘：g 高，但不一定是高光
高光边界：g 高，应该重点处理
正常亮条纹：亮，但不一定是高光
```

`s` 的加入就是为了补充高光语义：

```text
g：哪里有结构变化？
s：哪里更可能是高反光？
input：哪里亮？
```

三者结合后，模型更容易区分：

```text
正常亮条纹 vs 镜面高光
普通边缘 vs 高光边界
```

---

## 3.5 s 在模型里的作用位置

`s` 主要有两个作用位置。

### 位置 1：进入 EnhancementLayers

```text
decomp_unit_1(xe1, g, input, s) → enh1
decomp_unit_2(xe2, g, input, s) → enh2
decomp_unit_3(xe3, g, input, s) → enh3
decomp_unit_4(xe4, g, input, s) → enh4
```

在 `_SpectralDecompositionUnit` 内部：

```text
edge_gate     ← g
specular_gate ← concat(g, input, s)
final_gate    = edge_gate * specular_gate
```

因此 `s` 会参与每一级高反光增强特征的生成。

### 位置 2：进入 MainDecoder

`MainDecoder.forward()` 接收：

```python
def forward(self, c_features, enhanced_features, x_g, x_s, original_input):
```

其中：

```text
x_g = g
x_s = s
```

在主解码过程中，`s` 用于调制解码特征，让主解码器知道哪些区域更可能需要高光分解处理。

---

## 3.6 s 的最终作用总结

```text
s 的本质:
    有监督高光概率图

s 的作用:
    1. 显式定位高反光区域
    2. 弥补 g 只能表示边缘、不能区分高光边缘和普通边缘的问题
    3. 在 EnhancementLayers 中参与生成 enh1~enh4
    4. 在 MainDecoder 中调制解码特征
    5. 帮助 b 预测高光强度
    6. 帮助 r_direct 和最终 r 判断哪里需要恢复干净漫反射
```

---

# 4. MainDecoder 为什么可以同时输出 r_direct 和 b？

## 4.1 当前 MainDecoder 的最后输出

`MainDecoder` 最后从同一个共享特征 `x` 输出两个头：

```text
共享主解码特征 x
        │
        ├── output_conv4
        │       └── b：镜面反射 / 高反光分量
        │
        └── output_r_direct_conv
                └── r_direct：直接预测的漫反射
```

代码形式是：

```python
est_b = self.output_sigmoid(self.output_conv4(x))
r_direct = self.output_r_direct_sigmoid(self.output_r_direct_conv(x))
return est_b, r_direct, self.dual_head_alpha_logit
```

---

## 4.2 r_direct 和 b 是不是完全不相干？

不是。

它们虽然物理含义不同：

```text
b        ：要去掉的镜面反射 / 高光残差
r_direct ：去掉高光后的干净漫反射预测
```

但它们来自同一个输入分解关系：

```text
input ≈ diffuse + specular
x     ≈ r       + b
```

训练中：

```text
target       = 干净漫反射图
gt_specular  = clamp(input - target, 0, 1)
```

所以：

```text
r 的目标 = target
b 的目标 = input - target
```

它们不是无关任务，而是同一张输入图像的两个互补解释。

---

## 4.3 主解码器学习的不是单一 r 或 b，而是分解特征

`MainDecoder` 的共享特征 `x` 可以理解为：

```text
decomposition feature
高反光分解特征
```

它里面同时编码：

```text
1. 哪些区域是正常条纹结构
2. 哪些区域是高反光区域
3. 高光大致强度是多少
4. 高光边界在哪里
5. 高光覆盖区域下的漫反射应如何恢复
6. 哪些地方应该相信 input - b
7. 哪些地方应该相信 r_direct
```

所以两个输出头只是从同一个分解特征中读取不同信息：

```text
b_head:
    关注输入中多出来的高光残差是多少

r_direct_head:
    关注去高光后的干净图像应该长什么样
```

---

## 4.4 为什么不只预测 b，然后用 input - b 得到 r？

只用 `b` 的方式是：

```text
r = clamp(input - b, 0, 1)
```

优点是物理直观，但问题是：

```text
b 高估 → r 被压暗，甚至 clamp 到 0
b 低估 → r 中残留高光
```

例如：

```text
input = 0.9
target = 0.4
真实 b = 0.5

如果预测 b = 0.6:
r_sub = input - b = 0.3  → 偏暗

如果预测 b = 0.3:
r_sub = input - b = 0.6  → 高光残留
```

因此单纯依赖 `input - b` 风险较大。

加入 `r_direct` 的目的就是：

```text
提供一条不完全依赖 b 的直接漫反射恢复路径。
```

---

## 4.5 最终 r 是如何形成的？

当前模型不是直接把 `r_direct` 当最终结果，而是：

```text
b
│
├── r_sub = clamp(input - b, 0, 1)
│
r_direct
│
├── 与 r_sub 通过 alpha 融合
│
▼
r_fused = alpha * r_direct + (1 - alpha) * r_sub
│
▼
ResidualRefinementHead(r_fused, input, r_direct)
│
▼
最终 r
```

用结构写成：

```text
MainDecoder
│
├── b
│   └── r_sub = clamp(input - b, 0, 1)
│
├── r_direct
│
└── alpha_logit
    └── alpha = sigmoid(alpha_logit)

r_fused = alpha * r_direct + (1 - alpha) * r_sub

r_final = refinement_head(
    coarse   = r_fused,
    original = input,
    r_direct = r_direct
)
```

最终真正的主输出是：

```text
r_final，也就是输出字典里的 "r"
```

---

## 4.6 当前双头设计的意义

```text
b:
    提供物理减法路径 input - b

r_direct:
    提供直接恢复路径，避免 b 误差导致 r 严重偏暗或残留

alpha:
    学习两条路径的权重

refinement_head:
    对融合后的 r_fused 做像素级小幅修正
```

所以当前双头设计不是让两个头各学各的，而是形成：

```text
物理一致性路径 + 直接恢复路径 + 残差精修
```

---

## 4.7 当前是否有硬约束 r_direct + b ≈ input？

当前模型已经有多种软耦合：

```text
1. b 有 gt_specular 监督
2. r 有 target 监督
3. b 通过 r_sub = input - b 影响最终 r
4. r_direct 通过 r_fused 和 refinement_head 影响最终 r
5. r_direct 与最终 r 有一致性约束
6. s 与 b 有一致性约束
```

但它没有很强的硬约束：

```text
r_direct + b ≈ input
```

因此如果后续想进一步强化物理分解关系，可以考虑加入重建一致性损失：

```text
L_recon = |r_direct + b - input|
```

或者更谨慎地只在可信区域加入：

```text
L_recon = mask * |r_direct + b - input|
```

但需要注意，真实图像可能有曝光截断、噪声、非线性响应，过强的重建约束可能反而影响恢复质量。

---


# 5. SpecularAwareAttention 与 Attention_block

这一部分补充你后面提出的几个关键疑问：

```text
1. Attention_block 里的 g 是不是高光软概率图？
2. 当前 decoder 特征 x 到底是什么？
3. SpecularAwareAttention 和 Attention_block 的输出有没有参与 MainDecoder 后续过程？
4. 当前 decoder 特征是不是解码过程中的上一级 / 上一步中间输出？
```

先给出最重要的结论：

```text
Attention_block.forward(g, x) 里的 g 不是高光软概率图 s，也不是梯度图 x_g。
这里的 g 是 gating signal，表示当前 decoder 特征。

当前 decoder 特征 x 是 MainDecoder 在逐级解码过程中的中间特征。
它会在每个 level 内被不断更新、拼接、上采样，最后用于输出 b 和 r_direct。
```

---

## 5.1 Attention_block 是什么？

`Attention_block` 是主解码器中的 skip connection 注意力门控模块。

它的作用是：

```text
在 encoder skip 特征进入 MainDecoder 之前，
根据当前 decoder 特征判断哪些 skip 信息应该保留，哪些应该抑制。
```

它的代码形式可以理解为：

```python
def forward(self, g, x):
    g1 = self.W_g(g)
    x1 = self.W_x(x)
    psi = self.relu(g1 + x1)
    psi = self.psi(psi)
    return x * psi
```

这里容易混淆的是变量名：

```text
g:
    gating signal
    当前 decoder 特征
    不是梯度图 g
    不是高光软概率图 s

x:
    encoder skip 特征
    不是 MainDecoder 里正在不断更新的变量 x
```

因此更清晰的写法应该理解为：

```text
Attention_block.forward(
    gate_feature = 当前 decoder 特征,
    skip_feature = encoder skip 特征
)
```

---

## 5.2 Attention_block 的内部结构

```text
Attention_block
│
├── 输入:
│   ├── gate_feature：当前 decoder 特征
│   └── skip_feature：encoder skip 特征
│
├── W_g(gate_feature)
│   ├── 1×1 Conv
│   └── BatchNorm
│
├── W_x(skip_feature)
│   ├── 1×1 Conv
│   └── BatchNorm
│
├── 融合:
│   ├── g1 + x1
│   └── ReLU
│
├── psi
│   ├── 1×1 Conv
│   ├── BatchNorm
│   └── Sigmoid
│
└── 输出:
    └── skip_feature * psi
        = bridge
```

其中 `bridge` 表示：

```text
被当前 decoder 状态筛选后的 encoder skip 特征。
```

---

## 5.3 Attention_block 在 MainDecoder 哪些位置使用？

`Attention_block` 在 `MainDecoder` 中有 4 个：

```text
att1: 作用于 H/16 的 x_bb_4
att2: 作用于 H/8  的 x_bb_3
att3: 作用于 H/4  的 x_bb_2
att4: 作用于 H/2  的 x_bb_1
```

对应流程为：

```text
MainDecoder
│
├── Level 1, H/16
│   ├── 当前 decoder 特征 x: [B,768,H/16,W/16]
│   ├── encoder skip x_bb_4: [B,512,H/16,W/16]
│   └── bridge = att1(x, x_bb_4)
│
├── Level 2, H/8
│   ├── 当前 decoder 特征 x: [B,384,H/8,W/8]
│   ├── encoder skip x_bb_3: [B,256,H/8,W/8]
│   └── bridge = att2(x, x_bb_3)
│
├── Level 3, H/4
│   ├── 当前 decoder 特征 x: [B,192,H/4,W/4]
│   ├── encoder skip x_bb_2: [B,128,H/4,W/4]
│   └── bridge = att3(x, x_bb_2)
│
└── Level 4, H/2
    ├── 当前 decoder 特征 x: [B,96,H/2,W/2]
    ├── encoder skip x_bb_1: [B,64,H/2,W/2]
    └── bridge = att4(x, x_bb_1)
```

---

## 5.4 SpecularAwareAttention 是什么？

`SpecularAwareAttention` 是主解码器中的高反光感知注意力模块。

它不是用于筛选 encoder skip 的，而是直接作用在当前 decoder 特征 `x` 上。

它的输入是：

```text
features:
    当前 decoder 特征 x

original_input:
    原始输入灰度图
```

输出是：

```text
高反光先验增强后的当前 decoder 特征 x
```

也就是说：

```python
x = self.sa1(x, original_input)
```

这句代码表示：

```text
用原图亮度和局部对比度构造高反光先验，
然后增强当前 decoder 特征中疑似高反光区域的响应。
```

---

## 5.5 SpecularAwareAttention 的内部逻辑

`SpecularAwareAttention` 可以理解为：

```text
SpecularAwareAttention
│
├── 输入:
│   ├── features       : 当前 decoder 特征 x
│   └── original_input : 原始输入图像
│
├── Step 1: 将 original_input resize 到 features 的空间尺寸
│
├── Step 2: 构造高反光先验
│   │
│   ├── local_mean = avg_pool(original_input)
│   ├── local_contrast = abs(original_input - local_mean)
│   └── spec_prior = 0.75 * original_input + 0.25 * local_contrast
│
├── Step 3: 根据 features 生成自适应阈值偏移
│   │
│   └── adaptive_thresh(features)
│
├── Step 4: 多阈值高光 mask
│   │
│   ├── threshold 0.5 → 弱高光 mask
│   ├── threshold 0.7 → 中等高光 mask
│   └── threshold 0.9 → 强高光 mask
│
├── Step 5: 根据 features + multi_mask 生成注意力图
│   │
│   ├── concat(features, multi_mask)
│   ├── 3×3 Conv + BN + GELU
│   ├── 3×3 Conv + BN
│   └── Sigmoid → att
│
└── 输出:
    └── features * (1 + scale * att)
```

其中 `scale` 是可学习强度，并且会被限制在较小范围内。  
所以它不是完全重写 decoder 特征，而是做残差式高光增强。

---

## 5.6 SpecularAwareAttention 在 MainDecoder 哪些位置使用？

`SpecularAwareAttention` 在 `MainDecoder` 中也有 4 个：

```text
sa1: 作用于 Level 1 的 H/16 decoder 特征
sa2: 作用于 Level 2 的 H/8  decoder 特征
sa3: 作用于 Level 3 的 H/4  decoder 特征
sa4: 作用于 Level 4 的 H/2  decoder 特征
```

它们的位置是：

```text
MainDecoder
│
├── Level 1
│   ├── transposed_block_1
│   ├── sa1(x, original_input)
│   ├── att1(x, x_bb_4)
│   └── concat(x, bridge, enh1)
│
├── Level 2
│   ├── transposed_block_2
│   ├── ffc_h8
│   ├── g/s/input 三路门控调制
│   ├── sa2(x, original_input)
│   ├── att2(x, x_bb_3)
│   └── concat(x, bridge, enh2)
│
├── Level 3
│   ├── feature_extraction_B
│   ├── transposed_block_3
│   ├── ffc_h4
│   ├── g/s/input 三路门控调制
│   ├── sa3(x, original_input)
│   ├── att3(x, x_bb_2)
│   └── concat(x, bridge, enh3)
│
└── Level 4
    ├── transposed_block_4
    ├── g/s/input 三路门控调制
    ├── sa4(x, original_input)
    ├── att4(x, x_bb_1)
    └── concat(x, bridge, enh4)
```

---

## 5.7 当前 decoder 特征 x 到底是什么？

你可以这样理解：

```text
当前 decoder 特征 x
= MainDecoder 在当前解码阶段正在处理的中间特征
= 上一个解码步骤输出后，传到当前步骤继续加工的特征
= 还不是最终 b，也不是最终 r_direct
```

它不是固定的某一层输出，而是一个会不断被覆盖更新的中间变量。

例如：

```text
x_c
 ↓
FFC1 / FFC2
 ↓
multiscale_conv
 ↓
x = Level 1 开始时的当前 decoder 特征
 ↓
transposed_block_1
 ↓
x = Level 1 更新后的当前 decoder 特征
 ↓
sa1(x, input)
 ↓
x = 高光增强后的当前 decoder 特征
 ↓
concat(x, bridge, enh1)
 ↓
x = Level 1 输出，同时作为 Level 2 输入
```

因此你可以这样总结：

```text
当前 decoder 特征 x 就是解码过程中的中间输出特征。
它通常来自上一级 / 上一步解码操作的输出，
然后在当前 level 内继续被上采样、注意力增强、skip 融合和增强特征拼接。
```

---

## 5.8 Attention_block 里的 g 是高光软概率图吗？

不是。

这个问题需要特别强调：

```text
Attention_block.forward(g, x) 里的 g
    = gating signal
    = 当前 decoder 特征

MainDecoder 里的 x_g
    = GradientDecoder 输出的梯度图

MainDecoder 里的 x_s
    = GradientDecoder 输出的高光软概率图 s
```

三者不能混淆。

对应关系如下：

```text
Attention_block.forward(g, x)
│
├── g:
│   └── 当前 decoder 特征 / gate feature
│
└── x:
    └── encoder skip 特征
```

而主解码器函数里的：

```text
x_g:
    梯度预测图 g

x_s:
    高光软概率图 s

x:
    当前 decoder 中间特征
```

---

## 5.9 SpecularAwareAttention 和 Attention_block 的输出是否参与后续？

参与，而且是直接参与。

以 Level 1 为例，真实流程是：

```python
x = self.transposed_block_1(x)

x = self.sa1(x, original_input)

bridge = self.att1(x, x_bb_4)

x = torch.cat([x, bridge, xe1], 1)
```

这里：

```text
sa1 的输出:
    直接覆盖原来的 x，成为新的当前 decoder 特征。

att1 的输出:
    得到 bridge，即被筛选后的 x_bb_4 skip 特征。

concat:
    将 x、bridge、enh1 拼接，生成新的 x。
```

也就是说：

```text
sa1 输出 → 进入 concat，也作为 att1 的 gate_feature
att1 输出 → 作为 bridge 进入 concat
concat 后的新 x → 进入 Level 2
```

所以这两个模块的输出不是旁路结果，而是主解码路径的一部分。

---

## 5.10 Level 1 的详细数据流

```text
Level 1
│
├── 输入:
│   └── x，来自前面 bottleneck / multiscale_conv 的解码中间特征
│
├── x = transposed_block_1(x)
│   └── 得到当前 decoder 特征
│       x [B,768,H/16,W/16]
│
├── x = sa1(x, original_input)
│   │
│   ├── 根据 original_input 的亮度和局部对比度生成高反光先验
│   ├── 增强 x 中疑似高反光区域的响应
│   └── 输出仍然是 x [B,768,H/16,W/16]
│
├── bridge = att1(x, x_bb_4)
│   │
│   ├── x 是当前 decoder 特征 / gate_feature
│   ├── x_bb_4 是 H/16 encoder skip 特征
│   └── bridge 是被筛选后的 x_bb_4
│       bridge [B,512,H/16,W/16]
│
├── enh1
│   └── 来自 EnhancementLayers
│       enh1 [B,128,H/16,W/16]
│
└── x = concat(x, bridge, enh1)
    │
    ├── x      [B, 768,H/16,W/16]
    ├── bridge [B, 512,H/16,W/16]
    ├── enh1   [B, 128,H/16,W/16]
    └── 新 x   [B,1408,H/16,W/16]
         ↓
       作为 Level 2 的输入
```

这说明：

```text
sa1 的输出和 att1 的输出都会进入 MainDecoder 后续过程。
```

---

## 5.11 Level 1 到 Level 2 的衔接

```text
Level 1
│
├── x = transposed_block_1(x)
├── x = sa1(x, original_input)
├── bridge = att1(x, x_bb_4)
└── x = concat(x, bridge, enh1)
    │
    ▼
Level 2
│
├── x = transposed_block_2(x)
├── x = ffc_h8(x) + x
├── x = g/s/input 三路门控调制
├── x = sa2(x, original_input)
├── bridge = att2(x, x_bb_3)
└── x = concat(x, bridge, enh2)
```

所以：

```text
Level 1 concat 后的新 x 会直接作为 Level 2 的输入。
```

---

## 5.12 MainDecoder 四级解码完整流程

```text
MainDecoder
│
├── 初始输入:
│   ├── Encoder features:
│   │   x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c
│   │
│   ├── Enhanced features:
│   │   enh1, enh2, enh3, enh4
│   │
│   ├── Gradient map:
│   │   x_g = g
│   │
│   ├── Specular likelihood map:
│   │   x_s = s
│   │
│   └── original_input
│
├── Bottleneck decoding start
│   ├── x = x_c
│   ├── FFC1 + FFC2
│   └── multiscale_conv
│
├── Level 1, H/16
│   ├── x = transposed_block_1(x)
│   ├── x = sa1(x, original_input)
│   ├── bridge = att1(x, x_bb_4)
│   └── x = concat(x, bridge, enh1)
│       └── 新 x 进入 Level 2
│
├── Level 2, H/8
│   ├── x = transposed_block_2(x)
│   ├── x = ffc_h8(x) + x
│   ├── x = gate(g, input, s) 调制
│   ├── x = sa2(x, original_input)
│   ├── bridge = att2(x, x_bb_3)
│   └── x = concat(x, bridge, enh2)
│       └── 新 x 进入 Level 3
│
├── Level 3, H/4
│   ├── x = feature_extraction_B(x)
│   ├── x = transposed_block_3(x)
│   ├── x = ffc_h4(x) + x
│   ├── x = gate(g, input, s) 调制
│   ├── x = sa3(x, original_input)
│   ├── bridge = att3(x, x_bb_2)
│   └── x = concat(x, bridge, enh3)
│       └── 新 x 进入 Level 4
│
├── Level 4, H/2
│   ├── x = transposed_block_4(x)
│   ├── x = gate(g, input, s) 调制
│   ├── x = sa4(x, original_input)
│   ├── bridge = att4(x, x_bb_1)
│   └── x = concat(x, bridge, enh4)
│       └── 新 x 进入输出阶段
│
└── Output stage, H
    ├── x = transposed_block_5(x)
    ├── x = concat(x, x_g)
    ├── x = output_reduce(x)
    ├── x = output_dilated(x) + output_skip(x)
    ├── x = output_conv3(x)
    ├── b = output_conv4(x)
    └── r_direct = output_r_direct_conv(x)
```

---

## 5.13 为什么要设计这两个模块？

二者解决的问题不同。

### Attention_block 解决的问题

如果 encoder skip 无条件进入 MainDecoder，可能会把以下干扰带回来：

```text
1. 高光边界残留
2. 局部噪声
3. 过亮条纹
4. 与当前解码目标不一致的浅层细节
```

所以 `Attention_block` 的作用是：

```text
让当前 decoder 状态决定 encoder skip 中哪些信息应该被传入。
```

它本质上是：

```text
skip connection filter
```

### SpecularAwareAttention 解决的问题

高反光区域往往具有：

```text
1. 亮度高
2. 局部对比度强
3. 可能伴随过曝或强反射边界
```

`SpecularAwareAttention` 直接利用原始输入图像构造高反光先验，增强当前 decoder 特征中的疑似高光区域。

它本质上是：

```text
decoder feature specular enhancer
```

---

## 5.14 两个模块的区别和配合

```text
Attention_block
│
├── 输入:
│   ├── 当前 decoder 特征
│   └── encoder skip 特征
│
├── 作用对象:
│   └── encoder skip
│
└── 目的:
    └── 筛选 skip 信息
```

```text
SpecularAwareAttention
│
├── 输入:
│   ├── 当前 decoder 特征
│   └── original_input
│
├── 作用对象:
│   └── 当前 decoder 特征
│
└── 目的:
    └── 增强高反光相关响应
```

它们在每个 level 中的配合是：

```text
当前 decoder 特征 x
        ↓
SpecularAwareAttention
        ↓
高光增强后的 x
        ↓
Attention_block
        ↓
用增强后的 x 筛选 encoder skip
        ↓
得到 bridge
        ↓
concat(x, bridge, enh_i)
        ↓
进入下一解码阶段
```

一句话概括：

```text
SpecularAwareAttention 负责让当前 decoder 特征“更关注高光”；
Attention_block 负责让 encoder skip “有选择地传进来”。
```

---

## 5.15 当前 decoder 特征 x 的最终理解

你最后问：

```text
当前 decoder 特征 x 也就是解码过程中的上一级解码层的输出，
即解码过程中的中间输出过程，是这样的吗？
```

答案是：可以这么理解。

更准确地说：

```text
当前 decoder 特征 x
= 解码器逐层解码过程中的中间特征
= 上一个解码步骤输出后，传到当前步骤继续加工的特征
= 当前 level 内又会继续经过上采样、FFC、门控、注意力和拼接
= 最后不断更新，直到输出 b 和 r_direct
```

它的生命周期是：

```text
上一阶段输出 x
   ↓
当前阶段继续加工 x
   ↓
融合 skip / enh 后得到新的 x
   ↓
传给下一阶段
   ↓
最后由 x 输出 b 和 r_direct
```

所以：

```text
x 不是固定含义的某一层输出；
x 是 MainDecoder 中不断被覆盖更新的中间特征变量。
```

---


# 6. cs：跨尺度一致性

## 6.1 cs 是什么？

`cs` 是 `cross_scale_consistency_loss` 的输出，是一个训练用的辅助标量。

它不是图像结果，也不是推理阶段主要使用的输出。

可以理解为：

```text
cs = 跨尺度特征响应一致性损失
```

---

## 6.2 cs 在哪里计算？

在 `YFFC.forward()` 中：

```python
cs_loss = cross_scale_consistency_loss((c_feat[1], c_feat[2], c_feat[3]))
```

其中：

```text
c_feat[1] = x_bb_2  # H/4
c_feat[2] = x_bb_3  # H/8
c_feat[3] = x_bb_4  # H/16
```

也就是说，`cs` 使用的是：

```text
已经经过 Swin 融合 + CrossScaleTopDownRefinement 更新后的
x_bb_2、x_bb_3、x_bb_4
```

---

## 6.3 cs 如何计算？

结构可以写成：

```text
cross_scale_consistency_loss(x_bb_2, x_bb_3, x_bb_4)
│
├── 对每个尺度 feat:
│   │
│   ├── response = mean(abs(feat), channel)
│   │       └── 得到 [B,1,h,w] 的空间响应图
│   │
│   ├── mean = response 的空间均值
│   ├── std  = response 的空间标准差
│   │
│   └── norm_response = (response - mean) / std
│
├── 对齐尺度:
│   ├── H/4 response 保持不变
│   ├── H/8 response 上采样到 H/4
│   └── H/16 response 上采样到 H/4
│
└── 计算一致性:
    cs = 0.5 * (
          L1(H/4_response, H/8_response)
        + L1(H/8_response, H/16_response)
    )
```

这里比较的不是原始特征值，而是：

```text
不同尺度的空间响应分布
```

也就是要求：

```text
哪里重要，各尺度都应该相对重要；
哪里不重要，各尺度都应该相对不重要。
```

---

## 6.4 cs 在哪里起作用？

`cs` 会作为模型输出的一部分进入损失函数：

```text
YFFC.forward()
    └── 输出 "cs"
            ↓
EnhancedCompositeLoss
            ↓
lambda_cs * cs
            ↓
total loss
            ↓
反向传播
```

因此它只在训练阶段起作用。

推理时即使计算出了 `cs`，也不会参与反向传播，也不是最终要保存的图像。

---

## 6.5 为什么要设计 cs？

因为当前模型有多尺度 skip：

```text
x_bb_2: H/4  局部细节强
x_bb_3: H/8  中层结构强
x_bb_4: H/16 全局语义强
```

如果这些尺度判断不一致，后面的解码器会收到冲突信息。

例如：

```text
H/4 认为某个亮边是正常条纹，需要保留；
H/16 认为这个区域是高光，需要去除。
```

或者：

```text
H/16 认为某个区域是高光区域；
H/4 在该区域没有对应细节响应。
```

这会导致主解码器难以判断：

```text
这个亮结构到底要保留，还是要减掉？
```

`cs` 的目的就是：

```text
约束 H/4、H/8、H/16 对重要结构 / 高光相关区域的响应保持一致。
```

---

## 6.6 cs 和 CrossScaleTopDownRefinement 的关系

这两个模块是配合关系：

```text
CrossScaleTopDownRefinement:
    结构上把 H/16 深层上下文回流到 H/8 和 H/4

cs:
    损失上要求 H/4、H/8、H/16 的响应分布保持一致
```

也可以理解为：

```text
CrossScaleTopDownRefinement 负责“传递信息”
cs 负责“约束传递后的多尺度响应不要互相打架”
```

---

# 7. FFC 模块层级结构

## 7.1 你的理解是否正确？

你问：

```text
当前模型中的 FFC 模块是不是包含 FourierUnit、SpectralTransform 和 FFC_BN_ACT？
模型中每一次使用 FFC 都是由这三个子模块组成？
```

更准确的回答是：

```text
模型中真正被调用的 FFC 统一入口是 FFC_BN_ACT；
FFC_BN_ACT 内部的全局分支会调用 SpectralTransform；
SpectralTransform 内部会调用 FourierUnit。
```

因此它们不是三个并列模块，而是嵌套关系：

```text
FFC_BN_ACT
    └── SpectralTransform
            └── FourierUnit
```

---

## 7.2 FFC_BN_ACT 的完整结构

你可以将一次 FFC 使用理解成：

```text
FFC_BN_ACT
│
├── 输入分裂:
│   ├── x_l：局部分支 Local channels
│   └── x_g：全局分支 Global channels
│
├── L → L：Conv2d
│       └── 局部特征到局部特征，普通空间卷积
│
├── L → G：Conv2d
│       └── 局部特征到全局特征，普通空间卷积
│
├── G → L：SpectralTransform
│       ├── 1×1 Conv 降维
│       ├── FourierUnit，全局频域处理
│       │       ├── Tukey window
│       │       ├── rFFT
│       │       ├── real/imag 通道拼接
│       │       ├── 1×1 Conv
│       │       ├── BatchNorm
│       │       ├── GELU
│       │       ├── 频域 SE 注意力
│       │       ├── real/imag 复数重组
│       │       └── iRFFT
│       │
│       ├── FourierUnit，LFU top
│       │       └── 处理上半部分局部频域区域
│       │
│       ├── FourierUnit，LFU bottom
│       │       └── 处理下半部分局部频域区域
│       │
│       ├── FU/LFU 门控融合
│       └── 1×1 Conv 升维
│
└── G → G：SpectralTransform
        ├── 1×1 Conv 降维
        ├── FourierUnit，全局频域处理
        │       ├── Tukey window
        │       ├── rFFT
        │       ├── real/imag 通道拼接
        │       ├── 1×1 Conv
        │       ├── BatchNorm
        │       ├── GELU
        │       ├── 频域 SE 注意力
        │       ├── real/imag 复数重组
        │       └── iRFFT
        │
        ├── FourierUnit，LFU top
        │       └── 处理上半部分局部频域区域
        │
        ├── FourierUnit，LFU bottom
        │       └── 处理下半部分局部频域区域
        │
        ├── FU/LFU 门控融合
        └── 1×1 Conv 升维
```

最后，`FFC_BN_ACT` 会对局部输出和全局输出分别做 BN + 激活：

```text
out_xl = activation(BN(out_xl))
out_xg = activation(BN(out_xg))
return out_xl, out_xg
```

---

## 7.3 FourierUnit 的作用

`FourierUnit` 是最底层的频域计算单元。

它的作用是：

```text
把空间特征转到频域，
在频域中通过 1×1 卷积混合全局信息，
再转回空间域。
```

结构如下：

```text
FourierUnit
│
├── 输入 x [B,C,H,W]
│
├── 可选空间缩放
│
├── Tukey window
│       └── 降低 FFT 边界不连续导致的频谱泄漏
│
├── rFFT
│       └── 空间域 → 频域
│
├── real / imag 拆分并拼接到通道维
│
├── 1×1 Conv
│       └── 在频域通道上进行混合
│
├── BatchNorm
│
├── GELU
│       └── 相比 ReLU 更温和，减少对频域相位信息的破坏
│
├── 频域 SE 注意力
│       └── 自适应选择有用频段
│
├── real / imag 重新组合为复数
│
├── iRFFT
│       └── 频域 → 空间域
│
└── 数值安全处理
```

---

## 7.4 SpectralTransform 的作用

`SpectralTransform` 是 FFC 的全局分支模块。

它封装了：

```text
全局 FourierUnit
+
局部频域 LFU top / LFU bottom
+
门控融合
```

结构为：

```text
SpectralTransform
│
├── 输入全局分支特征 x_g
│
├── 1×1 Conv 降维
│
├── fu = FourierUnit(x)
│       └── 全局频域建模
│
├── LFU 局部频域建模
│   │
│   ├── top_out = FourierUnit(x 的上半部分)
│   ├── bot_out = FourierUnit(x 的下半部分)
│   └── lfu_out = concat(top_out, bot_out)
│
├── gate = Sigmoid(Conv1×1(fu_out))
│
├── combined = gate * fu_out + (1 - gate) * lfu_out
│
└── 1×1 Conv 升维
```

它的作用是：

```text
既保留全局频域感受野，
又保留局部区域内的频域结构。
```

---

## 7.5 FFC_BN_ACT 的作用

`FFC_BN_ACT` 是模型中使用 FFC 的统一入口。

它的作用是：

```text
把特征分为局部分支 L 和全局分支 G，
通过空间卷积和频域 SpectralTransform 完成局部-全局交互。
```

四路交互为：

```text
L → L：局部到局部，普通卷积
L → G：局部到全局，普通卷积
G → L：全局到局部，SpectralTransform
G → G：全局到全局，SpectralTransform
```

所以更推荐的论文表述是：

```text
当前模型中的 FFC 层以 FFC_BN_ACT 作为统一调用模块。
FFC_BN_ACT 将特征划分为局部/全局分支，并通过普通卷积和 SpectralTransform 完成四路交互。
其中 SpectralTransform 是全局频域分支，内部调用 FourierUnit 进行 FFT 频域建模。
```

---

# 8. 各模块之间的关键关系总图

下面是把前面几个疑问放到同一张流程图中的版本：

```text
输入 x [B,1,H,W]
│
▼
Encoder
│
├── ResNet50 分支
│   ├── x_bb_1 [B, 64,H/2, W/2]
│   ├── x_bb_2 [B,128,H/4, W/4]
│   ├── x_bb_3 [B,256,H/8, W/8]
│   └── x_bb_4 [B,512,H/16,W/16]
│
├── Swin-Tiny 分支
│   ├── swin_4  [B, 96,H/4, W/4]
│   ├── swin_8  [B,192,H/8, W/8]
│   └── swin_16 [B,384,H/16,W/16]
│
├── H/4 融合
│   └── AdaptiveSkipFusion(x_bb_2, swin_4)
│       → x_bb_2
│
├── H/8 融合
│   └── AdaptiveSkipFusion(x_bb_3, swin_8)
│       → x_bb_3
│
├── H/16 融合
│   ├── swin16_proj(swin_16): 384 → 512
│   └── CrossModalFusion(x_bb_4, swin_feat_16)
│       → fused
│
├── CrossScaleTopDownRefinement
│   ├── 输入: x_bb_2, x_bb_3, fused
│   └── 输出: 更新后的 x_bb_2, x_bb_3, x_bb_4
│
└── Bottleneck
    ├── cba_after_backbone(x_bb_4)
    ├── FFC_BN_ACT
    └── HighFrequencyEnhancement
        → x_c
│
▼
Encoder 输出:
x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c
│
├─────────────────────────────────────┐
│                                     │
▼                                     ▼
GradientDecoder                       MainDecoder 后续会使用这些 Encoder 特征
│
├── xe1, xe2, xe3, xe4
├── g：梯度预测图
└── s：高光软概率图
│
▼
EnhancementLayers
│
├── _SpectralDecompositionUnit(xe1, g, x, s) → enh1
├── _SpectralDecompositionUnit(xe2, g, x, s) → enh2
├── _SpectralDecompositionUnit(xe3, g, x, s) → enh3
└── _SpectralDecompositionUnit(xe4, g, x, s) → enh4
│
▼
MainDecoder
│
├── 输入:
│   ├── Encoder 特征: x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c
│   ├── Enhancement 特征: enh1, enh2, enh3, enh4
│   ├── 梯度图 g
│   ├── 高光概率图 s
│   └── 原始输入 x
│
├── 学习共享 decomposition feature
│
├── b_head
│   └── 输出 b：镜面反射 / 高反光分量
│
└── r_direct_head
    └── 输出 r_direct：直接漫反射预测
│
▼
最终融合
│
├── r_sub = clamp(x - b, 0, 1)
├── r_fused = alpha * r_direct + (1 - alpha) * r_sub
└── r = ResidualRefinementHead(r_fused, x, r_direct)
```

---

# 9. 最终核心理解总结

## 9.1 CrossScaleTopDownRefinement

```text
输入:
    已融合 Swin 的 H/4 特征 x_bb_2
    已融合 Swin 的 H/8 特征 x_bb_3
    H/16 CrossModalFusion 后的 fused 特征

输出:
    更新后的 x_bb_2, x_bb_3, x_bb_4

作用:
    把 H/16 深层上下文逐级回流到 H/8 和 H/4，
    让浅层 skip 不只是局部边缘，而是带有全局高光语义。
```

---

## 9.2 _SpectralDecompositionUnit

```text
输入:
    xe_i, g, original_input, s

输出:
    enh_i

作用:
    把 GradientDecoder 的通用结构特征 xe_i，
    结合梯度图 g、原图亮度 input 和高光概率图 s，
    转换成高反光相关增强特征 enh_i。
```

可以简写为：

```text
xe1 + g + input + s → enh1
xe2 + g + input + s → enh2
xe3 + g + input + s → enh3
xe4 + g + input + s → enh4
```

但更准确是：

```text
xe_i 是主特征；
g、input、s 生成门控；
门控增强 xe_i 中高反光相关部分。
```

---

## 9.3 s 高光软概率图

```text
s 是 GradientDecoder 输出的高光 likelihood map。

它不是 b。
b 是高光强度；
s 是高光概率 / 高光区域先验。

加入 s 的原因:
    仅靠梯度 g 无法区分正常条纹边缘和高光边界。
    s 提供显式高光语义，让增强层和主解码器知道哪里更可能是高光。
```

---

## 9.4 MainDecoder 双头输出

```text
MainDecoder 最后从共享分解特征中输出:
    b        ：要去掉的高光分量
    r_direct ：直接恢复的漫反射图

它们不是无关输出，而是同一输入分解关系的两个互补解释。
```

最终输出不是单独的 `r_direct`：

```text
r_sub   = input - b
r_fused = alpha * r_direct + (1 - alpha) * r_sub
r       = refinement_head(r_fused, input, r_direct)
```

---


## 9.5 SpecularAwareAttention 与 Attention_block

```text
SpecularAwareAttention:
    作用对象是当前 decoder 特征 x。
    它利用 original_input 的亮度和局部对比度生成高反光先验，
    对当前 decoder 特征进行残差式高光增强。

Attention_block:
    作用对象是 encoder skip 特征。
    它用当前 decoder 特征作为 gating signal，
    筛选对应尺度的 encoder skip，输出 bridge。

二者关系:
    x = sa_i(x, original_input)
    bridge = att_i(x, x_bb_i)
    x = concat(x, bridge, enh_i)

因此:
    sa_i 的输出直接更新当前 decoder 特征 x；
    att_i 的输出 bridge 会拼接进新的 x；
    concat 后的新 x 会进入下一解码阶段。
```


## 9.6 cs 跨尺度一致性

```text
cs 作用在 Encoder 的 x_bb_2、x_bb_3、x_bb_4 上。

它不是图像输出，而是训练辅助损失。
它要求 H/4、H/8、H/16 的空间响应分布保持一致，
减少多尺度 skip 互相冲突。
```

---

## 9.7 FFC 模块

```text
当前模型中的 FFC 不是三个并列模块，
而是嵌套结构:

FFC_BN_ACT
    └── SpectralTransform
            └── FourierUnit
```

最准确的理解是：

```text
模型中真正调用的 FFC 层是 FFC_BN_ACT；
FFC_BN_ACT 内部的全局分支调用 SpectralTransform；
SpectralTransform 内部调用 FourierUnit 完成频域建模。
```

---

# 10. 一句话总括

当前模型的核心设计逻辑是：

```text
Encoder 通过 ResNet + Swin 获得局部纹理和全局上下文；
CrossScaleTopDownRefinement 让深层高光语义回流到浅层 skip；
GradientDecoder 显式预测梯度 g 和高光概率 s；
EnhancementLayers 用 xe + g + input + s 生成高反光相关增强特征；
MainDecoder 中 SpecularAwareAttention 增强当前 decoder 特征的高光响应，Attention_block 筛选 encoder skip；
MainDecoder 从共享分解特征中同时预测 b 和 r_direct；
最终通过 input - b 的物理路径与 r_direct 的直接恢复路径融合，得到最终漫反射 r；
cs 损失则约束多尺度编码特征在空间响应上保持一致；
FFC 模块贯穿编码器、解码器和增强层，用频域建模增强全局结构和条纹/高光频率信息。
```
