# Y-FFC V3 SSIM偏低与训练后期损失回升问题诊断及改进方向

## 1. 文档目的

本文档不直接修改模型，而是基于当前仓库中的以下文件，对现有 Y-FFC / X-FFC V3 体系进行问题诊断，并整理后续最值得优先尝试的改进方向：

- `X-FFC_V3_模型完整技术文档.md`
- `Y_FFC_unified.py`
- `train.py`
- `process_images.py`（用于补充核查 EMA 推理链路）

本次重点回答两个问题：

1. 为什么当前模型的最佳 SSIM 仍然只有约 0.9765，提升幅度不够明显？
2. 为什么训练后期仍然会出现损失重新升高的问题？

本文档的目标是把问题拆开、定位到代码，并给出后续改造的优先级，而不是立即动手实现。

---

## 2. 当前系统与现象概览

### 2.1 当前模型的核心任务定义

根据 `X-FFC_V3_模型完整技术文档.md:43-58`，当前模型的任务定义是：

- 输入：一张包含高光条纹的单通道灰度图像 `x`
- 输出：
  - 镜面分量 `b`
  - 梯度图 `g`
  - 漫反射分量 `r`

其中，文档中明确写出 diffuse 的主路径为：

```python
r = clamp(x - b, 0, 1)
```

文档还说明最终输出会经过残差精修头继续做像素级修正。

### 2.2 当前训练配置的几个关键点

根据 `train.py:1491-1508`，当前训练的一些关键设置如下：

- `EPOCHS = 400`
- `BATCH_SIZE = 4`
- `ACCUMULATION_STEPS = 4`
- 等效 batch size = 16
- `BASE_LR = 2e-4`
- `WEIGHT_DECAY = 1e-4`
- `USE_EMA = True`
- `EMA_DECAY = 0.9995`
- `USE_OHEM = True`
- 输入尺寸 `IMAGE_SIZE = (128, 160)`

这说明当前训练并不是一个极简训练设置，而是一个带有 EMA、梯度累积、动态权重、多分量复合损失、OHEM 的相对复杂训练流程。

### 2.3 从训练曲线可以直接观察到的现象

根据用户提供的训练历史图：

1. **Train / Val SSIM 在较早阶段就已经快速上升到 0.97 以上，之后提升幅度很小。**
   这更像是模型较早进入了结构性平台期，而不是单纯训练轮数不够。

2. **Train / Val Loss 在早期快速下降后，中后期存在缓慢回升。**
   这意味着当前优化目标在后期并没有保持稳定收敛，至少总损失的定义与优化过程之间仍存在冲突。

3. **MAE / MSE 已经很低，但 SSIM 仍然没有进一步明显上涨。**
   这通常说明问题不再主要是像素级误差，而是结构保持、边界重建、局部细节一致性上存在瓶颈。

换句话说，当前模型更像是进入了“数值上已经不差，但结构质量上还不够高”的阶段。

---

## 3. SSIM 仍然偏低的主要原因诊断

下面按照“问题 -> 代码证据 -> 为什么会压制 SSIM”的方式逐项分析。

### 3.1 硬减法 `x - b` 再 `clamp`，本身就可能形成 SSIM 上限

#### 代码证据

技术文档中：

- `X-FFC_V3_模型完整技术文档.md:43-48`

模型实现中：

- `Y_FFC_unified.py:2112-2115`

核心逻辑为：

```python
est_b = self.main_decoder(...)
coarse = torch.clamp(x - est_b, 0.0, 1.0)
pred_diffuse = self.refinement_head(coarse, x)
```

#### 问题本质

当前模型并不是直接预测高质量的 diffuse，而是先预测 specular `b`，再通过 `x - b` 得到一个硬约束的 coarse diffuse。

这条链路有一个很强的隐含前提：

> 只要 `b` 预测得足够准，`x - b` 就会自然得到高质量 diffuse。

但真实情况没有这么理想，尤其是在以下区域：

- 高光强饱和区域
- 高光边界区域
- 细条纹、窄高亮带
- 亮 diffuse 与 specular 混杂区域

一旦 `est_b` 在这些区域出现过减或欠减，`coarse = clamp(x - est_b, 0, 1)` 会立即把错误“硬写进结果里”：

- 过减时可能把局部结构压到 0
- 欠减时高亮残留会保留下来
- clamp 会把超过范围的细微信息直接截断

这种截断对 MSE / MAE 的影响未必最大，但对 SSIM 尤其致命，因为 SSIM更关心局部结构、对比关系、纹理连续性。

#### 为什么这会形成 SSIM 平台

在训练早期，模型靠主干网络已经能把大部分低频误差压下去，因此 MAE / MSE 很容易下降。

但到后期，剩下的主要误差往往集中在结构最敏感的区域。此时如果主输出路径本身带有不可逆的 `clamp(x - b)` 截断，模型就会很容易进入下面这种状态：

- 像素值整体已经接近正确
- 但局部结构细节一旦在 coarse 阶段被截断，后面很难补回来
- 因而 SSIM 提升会明显变慢

这很像是一个“先天上限瓶颈”。

#### 结论

这是当前最值得优先怀疑的 SSIM 天花板来源之一。

---

### 3.2 残差精修头修正能力偏弱，只能做很小幅度补丁

#### 代码证据

- `Y_FFC_unified.py:2082-2094`

核心逻辑为：

```python
class ResidualRefinementHead(nn.Module):
    def __init__(self, scale=0.05):
        ...
    def forward(self, coarse, original):
        delta = self.refine(torch.cat([coarse, original], dim=1))
        return torch.clamp(coarse + self.scale * delta, 0.0, 1.0)
```

#### 问题本质

当前 refinement head 的设计思想是“对 `x - b` 做小幅像素级微调”，这在思路上是合理的。

问题在于，它当前的表达能力和修正幅度都比较保守：

1. 网络很浅，仅由少量卷积组成；
2. 最终输出经过 `tanh`；
3. 再乘上固定 `scale = 0.05`；
4. 最后再次 `clamp`。

这意味着 refinement head 默认只能做“非常小的修修补补”。

#### 为什么这会限制 SSIM

如果 coarse diffuse 的误差只是轻微亮度偏差，±0.05 的修正也许够用。

但如果 coarse 在高光边缘区域已经出现：

- 结构断裂
- 边界锯齿
- 亮斑残留
- 局部纹理被压平

那么这种误差通常不是一个很小的标量偏移，而是局部结构重建问题。

在这种场景下：

- 小幅 residual 只能微调亮度
- 很难真正重塑结构
- 模型会表现为“后期还能细抠一点点，但抠不出明显提升”

这和当前“SSIM 卡在 0.9765 左右”的现象是吻合的。

#### 结论

当前 refinement head 更像是一个“微调器”，还不是一个有足够纠错能力的“结构修复器”。

---

### 3.3 多个关键门控路径都是“只增强，不抑制”

#### 代码证据 1：梯度注入

- `Y_FFC_unified.py:1981-2013`

例如：

```python
x = x * (1.0 + self.grad_gate_2(_xg2))
x = x * (1.0 + self.grad_gate_3(_xg3))
x = x * (1.0 + self.grad_gate_4(_xg4))
```

#### 代码证据 2：增强分支

- `Y_FFC_unified.py:2051-2061`

```python
gate = self.gate_conv(torch.cat([projection, specular], dim=1))
specular_refined = specular + specular * gate
```

#### 代码证据 3：高光感知注意力

- `Y_FFC_unified.py:1424-1436`

```python
att = self.modulation(torch.cat([features, multi_mask], dim=1))
return features * (1.0 + att)
```

#### 问题本质

这三个地方有一个共同特征：

> gate / attention 只会让特征变强，不会让特征变弱。

因为它们都采用了 `x * (1 + gate)` 或 `x + x * gate` 这种形式，而 `gate` 又来自 `Sigmoid()`，取值基本在 `[0, 1]`。

于是输出范围就天然落在：

- `x` 到 `2x`

也就是说：

- 可以增强
- 不能抑制
- 更不能做负向校正

#### 为什么这会压制 SSIM

在高光去除任务里，增强型 gate 的确有好处：

- 可以突出边界
- 可以强调高频
- 可以放大疑似高光区域响应

但它有一个明显副作用：

- 如果某一层已经对某些局部结构“增强过头”，后续模块没有同等级别的抑制自由度
- 很容易出现 halo、局部过锐、伪边缘、相位型增强误差

这些误差对 MAE / MSE 的伤害不一定极大，但对 SSIM 的伤害会更明显。

因为 SSIM 不只是看亮度差异，还看局部结构关系是否自然。

#### 结论

当前多个关键模块都偏向“把高光相关线索放大”，但缺少“把错误增强压回去”的机制。这很可能是模型后期 SSIM 上不去的重要原因之一。

---

### 3.4 `SpecularAwareAttention` 过于依赖原图亮度阈值，可能把亮 diffuse 误当成 specular

#### 代码证据

- `Y_FFC_unified.py:1424-1436`

核心逻辑中，注意力 mask 的生成直接依赖原图输入亮度：

```python
masks.append(torch.sigmoid(self.steepness * (inp - eff_t)))
```

#### 问题本质

这意味着该模块在判断“哪里需要高光注意力”时，使用了非常强的亮度先验：

- 原图亮 -> 更容易激活 mask
- 原图不亮 -> 更不容易激活 mask

这个假设在很多场景下是有用的，但也可能带来混淆：

- 亮 diffuse 区域未必就是 specular
- 反射边缘周围亮度变化复杂，单纯靠阈值可能不够可靠
- 当输入亮度本身已经包含复杂成像效应时，亮度阈值更容易误导注意力

#### 为什么这会影响 SSIM

一旦亮 diffuse 区域被误导成需要更强 specular 处理，就有可能：

- 过度削弱有效结构
- 在边界处引入错误注意力增强
- 让局部对比关系发生不自然改变

这类误差本质上是结构归因错误，同样会优先伤害 SSIM。

#### 结论

`SpecularAwareAttention` 很可能在“快速定位高光”上有效，但在“细粒度区分亮 diffuse 和真实 specular”方面仍然过于粗糙。

---

### 3.5 主解码器多级 transposed convolution 可能带来局部结构伪影

#### 代码证据

在主解码器中，多处重复使用反卷积上采样：

- `Y_FFC_unified.py:1977`
- `Y_FFC_unified.py:1994`
- `Y_FFC_unified.py:2010`
- `Y_FFC_unified.py:2023`

即：

- `transposed_block_2`
- `transposed_block_3`
- `transposed_block_4`
- `transposed_block_5`

#### 问题本质

反卷积在图像重建任务里很常见，但它的已知风险也很明确：

- 棋盘格伪影
- 局部相位不连续
- 上采样纹理不均匀

在某些任务里，这些伪影可能只表现为轻微视觉问题；但在追求高 SSIM 的任务里，它们通常会先成为天花板。

#### 为什么这会压制 SSIM

因为 SSIM 对“结构是否自然连续”很敏感，而 transposed conv 产生的误差通常恰好是局部结构型误差。

即使：

- 全局亮度对了
- 轮廓大致对了
- MSE 已经很小

只要局部存在微弱棋盘格、振铃、边界相位错位，SSIM 都会被拖住。

#### 结论

如果后续只从损失权重上继续调，而不处理上采样伪影风险，SSIM 很可能仍然只能小幅改进。

---

### 3.6 ViT 分支固定缩放到 224x224，可能损失关键空间细节

#### 代码证据

- `Y_FFC_unified.py:699-760`

关键逻辑：

```python
x = F.interpolate(x, size=(self.vit_image_size, self.vit_image_size),
                  mode="bilinear", align_corners=False)
```

其中 `vit_image_size=224`。

#### 问题本质

当前训练输入尺寸在 `train.py:1500` 中被设为 `(128, 160)`，但 ViT 分支仍固定走 224x224 的重采样路径。

从绝对尺寸上看，这未必总是缩小，也可能是放大；真正的问题不是“变大还是变小”，而是：

- 它把所有输入强制投到一个统一分辨率上
- 再经过 patch embedding 与后续融合
- 最终再与 CNN 分支对齐

这样做会让一部分细粒度空间细节在 ViT 路径中经历较强的重采样和重构。

#### 为什么这会影响 SSIM

高反光条纹、细边界、小尺度高亮结构，对空间位置非常敏感。

一旦某一条分支在这些区域的几何细节被模糊、平滑或重映射，最终融合时就可能出现：

- 边界偏软
- 条纹位置不够准
- 局部结构一致性不足

这类误差再次是 MAE / MSE 未必很严重，但 SSIM 很敏感。

#### 结论

ViT 路径的固定重采样可能不是当前最大瓶颈，但大概率是限制极限精度的次级瓶颈之一。

---

### 3.7 当前训练目标与日志/最佳模型跟踪指标并不完全一致

#### 代码证据

在损失函数中：

- `Y_FFC_unified.py:1328-1332`

```python
l_ssim_single = 1.0 - self.ssim_fn(pred_diffuse.float(), target.float())
l_msssim = 1.0 - self.msssim_fn(pred_diffuse.float(), target.float())
l_ssim = 0.8 * l_msssim + 0.2 * l_ssim_single
```

说明训练时真正参与 total loss 的 SSIM 项是：

- `0.8 * MS-SSIM + 0.2 * SSIM`

但是在损失字典中被记录出来的 `ssim` 却是：

- `l_ssim_single`

见：

- `Y_FFC_unified.py:1373-1377`

```python
('ssim', l_ssim_single)
```

在 `MetricTracker` 中：

- `train.py:304-307`

```python
self.tot_ssim += (1.0 - loss_dict['ssim'])
```

在最佳 SSIM 模型保存逻辑中：

- `train.py:1322-1329`

也是直接用 `vs` 作为 best SSIM。

#### 问题本质

当前模型训练时在优化一种“混合结构相似性目标”，但日志与最佳模型选择却主要以 single-scale SSIM 为准。

这会导致一个很实际的问题：

- 模型可能在训练目标上确实继续改善
- 但你看到的 best SSIM 却提升有限
- 于是看起来像“模型没变好”

这不一定意味着模型结构没问题，但它会严重干扰判断。

#### 为什么这很重要

如果度量体系本身不完全对齐，就会出现以下风险：

1. 错把指标错配当作模型瓶颈；
2. 错把某次实验判定为“无效”；
3. best checkpoint 的筛选不一定和真正最优视觉结构结果一致。

#### 结论

这个问题未必是 SSIM 提升受限的根因，但它会显著干扰后续实验判断，因此优先级非常高。

---

### 3.8 当前 `MSSSIM` 的实现并不是标准 MS-SSIM，可能直接误导训练目标

#### 代码证据

- `Y_FFC_unified.py:997-1030`
- `Y_FFC_unified.py:1328-1332`

在 `MSSSIM.forward()` 中，`_ssim_components()` 返回了 `lum` 和 `cs`，但最终前向里只累乘了 `cs_list`：

```python
lum, cs = self._ssim_components(img1, img2)
cs_list.append(cs)
...
msssim = 1.0
for i in range(n_actual):
    msssim = msssim * (torch.clamp(cs_list[i], min=1e-8) ** w[i])
```

而标准 MS-SSIM 应该在最粗尺度保留 luminance 项，只在其余尺度累乘 contrast-structure 项。

#### 问题本质

当前训练里 `l_msssim` 并不是严格意义上的标准 MS-SSIM，而是一个更偏向多尺度结构/对比度乘积的近似量。

这样会带来两个问题：

1. 它和日志里统计的 single-scale SSIM 已经不一致；
2. 它和真正的 MS-SSIM 也不完全一致。

于是当前的主损失项之一，实际上在优化一个“看起来像 MS-SSIM，但并不完全等价”的目标。

#### 为什么这会影响 SSIM

这会导致训练出现一种很危险的情况：

- 模型在当前 `l_msssim` 上持续下降；
- 但真实 SSIM / 标准 MS-SSIM 提升没有同步发生；
- 最终表现为 total loss 在优化，单看 SSIM 却提升有限。

这不是普通的“权重没调好”，而是主目标定义本身可能存在偏差。

#### 对应解决办法

1. 将 `MSSSIM` 改为标准实现：保留最终尺度的 luminance 项，只对前面各尺度累乘 `cs`；
2. 在真正修正前，可先暂时把混合项退回到 single-scale SSIM 或接入成熟库实现，验证是否立刻改善“指标对不上”的现象；
3. 后续训练日志中同时记录：single-scale SSIM、标准 MS-SSIM、当前混合损失值，避免再次把实现偏差误认为模型瓶颈。

---

### 3.9 梯度监督按“每张图独立归一化”，会让梯度分支的强度语义不稳定

#### 代码证据

在训练数据集里：

- `train.py:144-148`

在增强版配对变换里：

- `Y_FFC_unified.py:1631-1635`

核心逻辑是：

```python
gt_gradient = (mag / (p995 + 1e-8)).clamp(0, 1)
```

而这个 `gt_gradient` 随后不仅用于辅助损失，还被直接送进主解码器调制：

- `Y_FFC_unified.py:1982-1983`
- `Y_FFC_unified.py:1999-2000`
- `Y_FFC_unified.py:2012-2013`
- `Y_FFC_unified.py:2024-2026`

#### 问题本质

当前梯度真值不是在全数据集统一尺度下定义的，而是每张图各自按自己的 99.5 分位做归一化。

这意味着：

- 同样强度的真实边缘，在不同图里目标值可能不一样；
- 梯度分支学到的更像“相对排序”，而不是稳定的绝对强度语义；
- 但主解码器又把 `x_g` 当成真实强度提示去调制特征。

#### 为什么这会影响 SSIM

如果梯度图本身的强度语义在样本间漂移，那么主解码器在不同图像上收到的调制强度就不稳定：

- 某些图会被过度强调边缘；
- 某些图会被低估边缘；
- 最终容易出现局部过锐、边界起伏、细结构不一致。

这些问题同样是 SSIM 很敏感、但 MAE / MSE 不一定非常敏感的类型。

#### 对应解决办法

1. 把 `gt_gradient` 改为全局一致的归一化方式，而不是每张图各自除以 `p995`；
2. 或者把梯度分支改成更接近“边缘概率 / 边缘掩码”的辅助任务，避免把不稳定的幅值直接拿去调主干；
3. 即使保留当前梯度分支，也建议降低它对主解码器的直接乘性影响，优先改成更温和的残差注入或置信度加权融合。

---

### 3.10 当前 skip attention 可能把高分辨率细节直接“门掉”

#### 代码证据

- `Y_FFC_unified.py:524-556`
- `Y_FFC_unified.py:1973`
- `Y_FFC_unified.py:1989`
- `Y_FFC_unified.py:2006`
- `Y_FFC_unified.py:2019`

`Attention_block` 的输出是：

```python
psi = self.psi(psi)
return x * psi
```

#### 问题本质

skip connection 在高保真重建任务里，最重要的价值就是保留高分辨率细节和局部几何结构。

但当前 attention block 的形式是纯门控乘法：

- `psi` 小时，skip 特征会被大幅压制；
- 没有 identity 保底项；
- 一旦门控估计偏小，细节会被直接丢掉，而不是“少给一点权重”。

#### 为什么这会影响 SSIM

SSIM 对高分辨率局部结构非常敏感，而 skip 恰好是这些结构的主要来源之一。

如果 skip 被过度门控：

- 边缘可能发软；
- 细条纹可能不连续；
- 微结构可能被主干低频特征覆盖掉。

这类损失在视觉上会表现成“整体干净，但细节不够真”，非常符合当前高分段 SSIM 平台期的特征。

#### 对应解决办法

1. 将 skip 门控改为保留 identity 的形式，例如 `x * (1 + alpha * psi)` 或 `x * (0.5 + 0.5 * psi)`；
2. 或者在 attention 后追加残差旁路，避免任何一层把高分辨率细节完全压没；
3. 后续消融时，应单独比较“当前纯门控”和“保底残差门控”对 SSIM 的影响。

---

### 3.11 主解码器把 `x_g` 直接拼到最终输出头，可能把辅助分支误差直接带入主输出

#### 代码证据

- `Y_FFC_unified.py:2023-2026`

```python
x = self.transposed_block_5(x)
if x_g.shape[-2:] != x.shape[-2:]:
    x_g = F.interpolate(x_g, x.shape[-2:], mode='bilinear', align_corners=False)
x = torch.cat([x, x_g], 1)
```

#### 问题本质

当前 gradient branch 不只是中间层辅助，它还在最后一层直接作为一个显式输入通道送入 specular 输出头。

这意味着只要 `x_g` 有噪声、偏移或强度不稳定，输出头就会把这类误差直接当作“有用特征”混入最终预测。

#### 为什么这会影响 SSIM

梯度图本身更接近边缘提示，不是完整的结构表示。把它直接拼到最后输出头，会有两个风险：

1. 辅助分支误差直接污染主输出；
2. 主输出头会倾向于学习“借梯度图强行勾边”，而不是稳定还原结构。

这样容易带来：

- 边缘发硬；
- 条纹附近局部伪影；
- diffuse / specular 分界过度依赖梯度图质量。

#### 对应解决办法

1. 将 `x_g` 从最终输出头的直接输入中移除，或仅保留中间层调制；
2. 若必须保留，可先通过一个置信度模块或小型投影层，避免裸拼接；
3. 更稳妥的做法是让梯度分支在中层提供提示，而不是在最后一层直接参与生成。

---

### 3.12 当前随机裁剪后再双线性回采样，会系统性平滑高频边界

#### 代码证据

- `Y_FFC_unified.py:1608-1620`
- `train.py:1076-1080`

增强里存在大量：

```python
input_img = F.interpolate(..., mode='bilinear', align_corners=False)
target_img = F.interpolate(..., mode='bilinear', align_corners=False)
```

并且训练集中默认启用了 `EnhancedPairedTransform`。

#### 问题本质

随机裁剪本身没有问题，问题在于：

- 裁完后又被强制插值回原尺寸；
- 插值使用双线性，会明显平滑边界；
- 而任务恰恰非常依赖高光边界、窄条纹、细结构的精确位置。

如果这种增强占比不低，模型会持续看到“被平滑过”的 target / specular 对应关系。

#### 为什么这会影响 SSIM

模型会逐渐学到一种更容易优化、但结构更软的映射：

- 边界更圆滑；
- 细节更平；
- 高频误差更少，但结构锐度也被一起抹掉。

这有利于降低一部分像素损失，却可能直接限制高分段 SSIM。

#### 对应解决办法

1. 降低 `p_crop` 或把 `crop_scale` 的下限从 `0.7` 提高到更保守的范围；
2. 对极依赖边界精度的实验，可先完全关掉这类“裁剪后再放大”的增强，单独看 SSIM 是否上升；
3. 如果保留该增强，应把它视为泛化增强，而不是默认最优增强。

---

### 3.13 合成高光增强的形状先验过于偏向高斯 blob，可能与真实条纹高光分布不匹配

#### 代码证据

- `Y_FFC_unified.py:1542-1572`
- `train.py:1076-1080`

当前 `HighlightAugmentation` 是通过随机高斯 blob 在输入图上叠加合成高光：

```python
synth[0] += torch.exp(-((gy - cy) ** 2 + (gx - cx) ** 2) / (2 * r * r)) * I
```

#### 问题本质

这种增强适合制造“光斑型”反光，但当前任务图像从描述上更偏向：

- 条纹状高光
- 细长高亮带
- 边界明显、方向性更强的反射结构

如果训练里频繁加入圆形/团块状 blob，高光分支和注意力模块就会更容易学到“圆斑式 specular”先验，而不是真实条纹高光的几何特征。

#### 为什么这会影响 SSIM

这种分布错配不一定会立刻把 loss 拉坏，但会让模型在真实高光条纹上：

- 高光形状估计偏平滑；
- 边界几何不够准；
- 高频细结构恢复不足。

最终仍然会落到 SSIM 上限偏低的问题上。

#### 对应解决办法

1. 把高光增强从“圆 blob”扩展到“方向性条带 / 拉伸椭圆 / 局部条纹”分布；
2. 把增强形状设计得更接近真实反光模式，而不是只模拟亮斑；
3. 进行增强消融：对比“无合成高光增强”“高斯 blob 增强”“方向性条纹增强”对最终 SSIM 的影响。

---

## 4. 训练后期损失回升的主要原因诊断

下面讨论第二个核心问题：为什么后期 loss 还会重新变大。

### 4.1 调度器步数与梯度累积步数之间很可能存在时间尺度错配

#### 代码证据

构造调度器时：

- `train.py:1150-1154`

```python
scheduler = build_scheduler(
    optimizer,
    warmup_iters=min(500, len(train_loader) * 3),
    total_epochs=cfg['EPOCHS'],
    iters_per_epoch=len(train_loader))
```

训练更新时：

- `train.py:404-415`

```python
if (i + 1) % accum == 0 or (i + 1) == len(loader):
    ...
    optimizer.step()
    if scheduler:
        scheduler.step()
```

其中：

- `ACCUMULATION_STEPS = 4`，见 `train.py:1496`

#### 问题本质

调度器是按 `iters_per_epoch=len(train_loader)` 去估算总迭代数的，看起来默认一个 mini-batch 对应一次调度 step。

但实际上，`scheduler.step()` 并不是每个 mini-batch 都调用，而是每 `accum` 个 mini-batch 才调用一次，也就是每次真实 `optimizer.step()` 之后才调用。

这就会造成一个潜在的时间尺度错配：

- 调度器以为一轮里有 `len(train_loader)` 次 step
- 实际上每轮只有大约 `len(train_loader) / accum` 次 step

于是 warmup、cosine 衰减、最小学习率到达位置，都可能和预期不一致。

#### 为什么这会导致后期 loss 回升

如果学习率在训练后期并没有如预期那样衰减到足够低，模型就可能出现：

- 在已经接近收敛的多目标损失空间里来回摆动
- 对极少数难样本持续过度响应
- 总损失出现回升或震荡

这尤其容易出现在当前这种：

- 复合损失很多
- OHEM 开启
- 目标中既有结构项也有频域项
- 训练已经进入高精度微调区间

#### 结论

这是当前“后期 loss 回升”最值得优先核查的工程性问题之一，而且可验证性很强。

---

### 4.2 当前总损失过于复杂，后期容易出现多目标冲突

#### 代码证据

- `Y_FFC_unified.py:1276-1377`

当前总损失由以下部分加权组成：

- diffuse SSIM / MS-SSIM 混合项
- diffuse Charbonnier
- frequency loss
- perceptual loss
- specular loss
- gradient loss
- focal loss
- dice loss
- gradient consistency loss
- 另有 stat loss 逻辑虽然默认关闭，但保留了接口

总损失计算：

```python
total = (self.lambda_ssim * l_ssim + self.lambda_l1 * l_l1 +
         self.lambda_freq * l_freq + self.lambda_perceptual * l_percep +
         self.lambda_spec * l_spec + self.lambda_stat * l_stat +
         self.lambda_grad * l_grad + self.lambda_focal * l_focal +
         self.lambda_dice * l_dice + self.lambda_gc * l_gc)
```

#### 问题本质

在训练前中期，多目标损失的好处是明显的：

- 主输出更稳定
- 高频更清晰
- specular 分支更可控
- gradient 辅助任务能提供边界信息

但训练后期，问题会反过来出现：

- 主任务已经接近收敛
- 不同辅助目标的最优点未必一致
- 某些子损失继续下降，可能会牺牲 diffuse 的结构自然性

例如：

- 频域项强调频谱一致性
- 感知项强调语义或感知层面一致性
- specular 辅助项强调高光定位
- diffuse SSIM 更关心最终结构相似性

这些目标在后期不一定完全同向。

#### 为什么这会表现为后期 total loss 回升

后期 total loss 变大，并不一定表示主输出全面变差，更可能表示：

- 某几个子损失开始互相拉扯
- 为了继续优化某个辅助项，模型对另一个项变差
- 最终总和上升

换句话说：

> 后期 loss 回升可能不是“模型坏了”，而是“当前总目标对后期微调不够友好”。

#### 结论

后续分析一定不能只看 total loss，必须拆开看每个子损失项在后期的变化趋势。

---

### 4.3 OHEM 会在后期持续放大难样本波动

#### 代码证据

- `Y_FFC_unified.py:1279-1280`
- `Y_FFC_unified.py:1358-1363`
- `train.py:1508`

当前配置：

- `USE_OHEM = True`
- `ohem_ratio = 0.3`

逻辑：

```python
if self.use_ohem:
    l_focal = specular_ohem_loss(est_b, gt_sc, self.focal, self.ohem_ratio)
```

#### 问题本质

OHEM 的作用是把训练重心更集中到难样本或难区域上。

在前中期，它的确能帮助模型更快处理难点。

但在后期，OHEM 也会带来一个副作用：

- 大部分容易区域已经学会了
- 剩下的 loss 越来越由少数极难区域主导
- 这些极难区域往往噪声更大、统计更不稳定

于是整体表现就会变成：

- 指标看起来基本不差
- 但某一类损失始终被难样本牵着走
- total loss 容易反弹或抖动

#### 结论

在当前复杂损失体系下，OHEM 很可能是后期 loss 回升的重要放大器。

---

### 4.4 动态损失权重虽已冻结，但不代表后期目标漂移已彻底解决

#### 代码证据

- `train.py:157-191`

`ImprovedWeightScheduler` 当前的策略是：

- 0~30%：偏向快速收敛
- 30%~60%：逐步加大 focal / dice / freq 等权重
- 60% 之后：冻结在平台值

代码中也明确说明，之前版本的问题是：

- 权重持续增大到最后
- 与 LR 衰减冲突
- 导致后期损失突增

#### 当前判断

这个修复方向本身是对的，说明作者已经意识到“目标漂移”是问题来源之一。

但从当前训练曲线来看，后期 loss 仍然存在回升，说明至少有两种可能：

1. 动态权重只是部分修复，真正的问题不止这一处；
2. 即使权重冻结，冻结后的目标组合本身仍然不够适合后期微调。

#### 结论

当前不能把“已经加了 60% 冻结”视为问题已经解决。它更像是把一个显性的持续漂移问题，缩小成了一个仍待验证的后期冲突问题。

---

### 4.5 当前记录的是 total loss，但真正恶化的可能只是某一两项子损失

#### 代码证据

- `train.py:235-241`
- `train.py:319-342`
- `train.py:1214-1218`

训练脚本已经能够记录多个子损失项：

- `total`
- `ssim`
- `l1`
- `freq`
- `perceptual`
- `spec`
- `grad`
- `focal`
- `dice`
- `gc`

这说明后续完全有条件做更细粒度分析。

#### 问题本质

当前用户从训练历史图上看到的是 overall loss 曲线回升，但要真正定位原因，必须知道：

- 是 `freq` 回升？
- 是 `focal` / `dice` 回升？
- 还是 `grad` 分支波动？
- 还是单纯 `ssim` 与其他项开始冲突？

如果不拆开看，就很容易把所有问题都归结为“学习率不稳定”，但实际不一定如此。

#### 结论

后续实验必须把 total loss 拆解成子损失轨迹，不然会一直停留在表面判断。

---

### 4.6 EMA 验证链路是正确方向，但训练保存与推理加载之间存在命名不一致风险

#### 代码证据

训练保存时：

- `train.py:1296-1308`

```python
if ema:
    sd['ema'] = ema.shadow
```

推理加载时：

- `process_images.py:55-70`

```python
if prefer_ema and 'ema_shadow' in checkpoint:
    ema_shadow = checkpoint['ema_shadow']
    ...
```

#### 问题本质

这里存在一个明显的不一致：

- `train.py` 保存的是 `ema`
- `process_images.py` 优先读取的是 `ema_shadow`

这意味着：

- 训练过程中验证是基于 EMA 权重的，这是正确的
- 但推理阶段如果按 `process_images.py` 读取，可能根本读不到 `train.py` 保存的 EMA 键
- 于是实际推理用到的是普通 `model_state_dict`

#### 为什么这和当前问题相关

这不是“训练后期 loss 回升”的直接根因，但它会影响后续对模型性能的判断：

- 训练中你以为最优的是 EMA 版本
- 推理时却可能没真正用上 EMA
- 最终得到的 SSIM 可能比训练中看到的更低或更不稳定

这会让“模型已经改好了没有”这件事变得更难判断。

#### 结论

EMA 键名不一致不是当前两大问题的根因，但属于后续必须尽快清理掉的评估链路风险点。

---

### 4.7 训练中对“loss spike”直接跳过 batch，可能在后期引入选择性偏置

#### 代码证据

- `train.py:388-397`

```python
if tracker.n > 5:
    _avg_so_far = tracker.tot_loss / tracker.n
    if _avg_so_far > 0 and _current_loss_val > _avg_so_far * 10.0:
        ...
        optimizer.zero_grad()
        continue
```

#### 问题本质

这个保护机制的初衷是避免异常 batch 造成梯度爆炸，这是合理的。

但它有一个副作用：当训练进入后期以后，平均 loss 已经很低，只要某个 batch 恰好包含更多困难区域、高光边界或极端样本，它就更容易被判定为 spike 而直接跳过。

于是训练分布会被悄悄改变：

- 容易样本继续训练；
- 困难样本更容易被跳过；
- 模型在真正难点上的收敛会被打断。

#### 为什么这会影响后期 loss 和 SSIM

这会造成一种表面矛盾：

- 模型对普通样本越来越稳定；
- 但剩余困难区域一直学不好；
- 一旦某轮验证刚好更依赖这些区域，loss / SSIM 就会表现出不稳定。

对高光边界这类稀疏难点任务来说，这种选择性跳过尤其危险。

#### 对应解决办法

1. 将“跳过整个 batch”改为“记录告警 + 观察是否为真异常”，不要默认直接丢弃；
2. 若确实需要保护，更稳妥的做法是对该 batch 执行更强裁剪或缩小 scaler，而不是完全不学；
3. 训练日志中额外记录每个 epoch 被跳过的 batch 数量，避免这个机制在后台持续影响训练却没人注意到。

---

### 4.8 验证集抽样不是分层抽样，困难样本占比可能随划分随机波动

#### 代码证据

- `train.py:1093-1102`

当前做法是固定随机种子后，对样本做一次随机划分：

```python
g = torch.Generator().manual_seed(42)
rand_idx = torch.randperm(len(full_ds), generator=g).tolist()
train_ds = torch.utils.data.Subset(full_ds, rand_idx[:train_size])
val_ds = torch.utils.data.Subset(full_ds_no_aug, rand_idx[train_size:])
```

#### 问题本质

当前验证集划分并没有依据高光强度、条纹复杂度、边界数量等难度特征做分层。

如果数据集本身规模不大，或者难样本占比较少，那么一次普通随机划分就可能带来：

- 验证集偏容易；
- 或验证集偏难；
- 或困难样本集中在少数 batch。

#### 为什么这会影响后期 loss 回升判断

当前用户观察到的是后期 loss 回升。如果验证集本身在困难区域的分布上不够稳定，那么这类回升有一部分可能来自“验证集难度结构本来就不均匀”。

也就是说，当前曲线里的回升，不一定全是优化器或损失函数问题，可能混入了验证集统计噪声。

#### 对应解决办法

1. 对训练/验证划分做一次样本难度统计，至少看高光面积、亮度分布、边界密度等指标；
2. 尽量采用分层划分，而不是纯随机划分；
3. 对关键实验固定验证集名单，避免每次实验的难度分布变化干扰结论。

---

### 4.9 `drop_last=True` 会让训练始终丢掉一小部分样本，困难样本可能长期缺席

#### 代码证据

- `train.py:1104-1107`

```python
train_loader = DataLoader(
    train_ds, batch_size=cfg['BATCH_SIZE'], shuffle=True,
    num_workers=cfg['NUM_WORKERS'],
    pin_memory=(device.type == 'cuda'), drop_last=True)
```

#### 问题本质

`drop_last=True` 在使用 BN 时常用于稳定 batch 统计，但当前训练脚本默认还会执行 BN -> GN 替换：

- `train.py:1063-1067`

在 GN 条件下，小尾 batch 的统计问题已经大幅减弱，此时继续 `drop_last=True` 的收益会降低，但代价仍在：

- 每个 epoch 总有一部分样本看不到；
- 如果数据集不大，这种信息损失会更明显；
- 若恰好被丢掉的是困难样本，对后期收敛更不利。

#### 为什么这会影响后期表现

后期训练本来就在追少量难样本和边界细节，这些样本若长期被低频地遗漏，就会造成：

- 主体样本已收敛；
- 少数关键难点没学透；
- 训练或验证曲线在后段表现为“指标不再涨，loss 偶尔回弹”。

#### 对应解决办法

1. 在启用 GN 的训练设置下，优先测试 `drop_last=False`；
2. 若担心最后一个 batch 太小，可单独记录其 batch size，而不是直接丢弃；
3. 对少样本数据集，尾 batch 往往比统计整齐更重要。

---

### 4.10 训练增强与验证分布差异较大，可能放大后期 train/val 曲线错位

#### 代码证据

训练集默认使用：

- `train.py:1076-1080`

验证集默认使用：

- `train.py:1093-1102`

增强内容包括：

- 随机翻转 / 旋转 / 裁剪回采样
- 亮度缩放
- 合成高光 blob

见：

- `Y_FFC_unified.py:1575-1637`

#### 问题本质

训练样本经过较强增强，而验证样本保持原始分布，这本来是合理的。

但当前增强不只是轻微扰动，而是会显式改变：

- 亮度分布
- 高光形状分布
- 高频边界平滑程度

这意味着训练看到的是一个“更宽、更人工”的分布，验证看到的是原始分布。

#### 为什么这会影响后期 loss 回升

在训练前期，这种增强有助于泛化；但在训练后期，它也可能造成：

- 训练目标仍然在适应大量人工扰动；
- 验证指标却只看原始样本；
- 某些子损失在训练集继续下降，却不再对应验证集收益。

最终表现就是：

- train loss 和 val loss 的关系越来越复杂；
- 后期曲线容易分叉或回升；
- 难以判断是优化问题还是分布问题。

#### 对应解决办法

1. 对关键高精度实验，准备一个“弱增强版训练配置”作为对照；
2. 将强增强更明确地用于泛化实验，而不是默认配置；
3. 分别比较“弱增强”和“强增强”条件下后期 SSIM 上限与 loss 稳定性。

---

## 5. 补充诊断：原文尚未覆盖但值得纳入的问题

在重新核查 `Y_FFC_unified.py` 与 `train.py` 后，除了前面已经覆盖的问题，还发现以下风险点。这些问题不一定都是当前 SSIM 平台或后期 loss 回升的主因，但会影响实验判断、训练稳定性或模型上限。

### 5.1 预训练链路可能静默降级，且输入归一化与 ImageNet 预训练分布不一致

#### 代码证据

- `Y_FFC_unified.py:705-717`：ViT 权重加载失败时退回 `weights=None`
- `Y_FFC_unified.py:1059-1062`：VGG16 感知损失权重加载失败时退回 `weights=None`
- `Y_FFC_unified.py:1756-1768`：ResNet50 权重加载失败时退回 `weights=None`
- `Y_FFC_unified.py:2263-2274`：BN 替换为 GN 时只新建 `GroupNorm`，没有继承 BN 的 affine 参数

#### 问题本质

当前代码里，ResNet / ViT / VGG 的预训练权重加载失败不会中断训练，也没有显式告警或记录。这会导致一个实验可解释性问题：

- 表面上看是“预训练骨干 + 感知损失”；
- 实际上可能是随机初始化骨干或随机 VGG 特征；
- 训练曲线变差时，很难判断是结构问题、数据问题，还是预训练根本没有生效。

另一个问题是，输入图像直接是 `[0, 1]` 灰度值重复到 3 通道后送入 ViT / VGG，并没有按 ImageNet mean / std 做归一化。对预训练特征来说，这会造成分布偏移，尤其是 VGG 感知损失可能不再表示稳定的“感知相似性”。

BN -> GN 替换也会丢掉预训练 BN 层里已经学到的 `weight` / `bias` affine 信息。虽然 GN 对小 batch 更稳，但直接替换可能削弱预训练初始化的收益。

#### 为什么这会影响 SSIM 和后期训练

如果预训练链路实际没有按预期工作，模型早期可能仍能靠监督信号收敛到不错的 MAE / MSE，但高分段结构细节会更依赖随机初始化后的特征学习。感知损失若来自随机 VGG 或未归一化输入，还可能在后期提供噪声目标，放大多目标冲突。

#### 对应解决办法

1. 显式打印并保存 ResNet / ViT / VGG 权重是否成功加载；
2. 如果权重加载失败，应调整实验标记、学习率和训练轮数，或暂时禁用感知损失；
3. 对 VGG 感知损失和 ViT 输入使用与预训练权重匹配的 ImageNet normalization；
4. BN -> GN 替换时继承 BN 的 affine 参数，减少预训练信息损失。

---

### 5.2 数据增强可能破坏 `input = target + specular` 的物理一致性

#### 代码证据

- `Y_FFC_unified.py:1623-1629`
- `Y_FFC_unified.py:1569-1571`

核心流程是：

```python
gt_specular = (input_img - target_img).clamp(0, 1)
input_img = (input_img * bf).clamp(0, 1)
target_img = (target_img * bf).clamp(0, 1)
gt_specular = (gt_specular * bf).clamp(0, 1)
input_img, target_img, gt_specular = self.hl_aug(...)
```

合成高光增强里又会执行：

```python
input_img = (input_img + synth).clamp(0, 1)
gt_specular = (gt_specular + synth).clamp(0, 1)
```

#### 问题本质

理论上，监督信号里最稳定的关系应该是：

```python
gt_specular ~= clamp(input - target, 0, 1)
```

但当前增强顺序里，亮度缩放和合成高光都可能触发 `clamp`。一旦 `input` 或 `target` 因饱和被截断，`gt_specular` 再按独立路径缩放或叠加，就不一定等于最终的 `input - target`。

这会让模型同时面对两个不完全一致的目标：

- diffuse 主路径希望 `r = input - est_b` 接近 target；
- specular 分支却被监督到一个可能不再严格匹配 `input-target` 的 `gt_specular`。

#### 为什么这会影响 SSIM 和 loss

这种不一致会直接制造目标冲突。模型可能为了降低 specular loss 学到一个更接近 `gt_specular` 的 `b`，但 `input - b` 反而不再最适合恢复 diffuse。后期主任务已经接近收敛时，这类小冲突会更容易表现为 loss 回升或 SSIM 上限受限。

#### 对应解决办法

1. 在所有增强完成后重新计算 `gt_specular = (input_img - target_img).clamp(0, 1)`；
2. 或至少记录增强后 `abs(gt_specular - clamp(input-target))` 的均值/最大值，确认不一致程度；
3. 对高饱和区域单独 mask 或降低合成高光强度，避免 clamp 大量破坏物理关系。

---

### 5.3 分层 LR 的最低学习率和日志只按第一个参数组观察，可能误判真实训练状态

#### 代码证据

- `Y_FFC_unified.py:2243-2255`
- `train.py:1274`

调度器中 `eta_min` 比例按 `base_lrs[0]` 计算：

```python
min_ratio = eta_min / base_lrs[0]
```

训练日志中当前学习率也只读取：

```python
current_lr = optimizer.param_groups[0]['lr']
```

#### 问题本质

当前优化器是分层学习率，不同参数组初始 LR 不同：

- 预训练编码器：`base_lr * 0.1`
- fusion / FFC：`base_lr * 0.5`
- decoder / new modules：`base_lr`

但 scheduler 的最小 LR 比例只按第一个参数组计算。如果第一个参数组是最低 LR 的预训练编码器，那么其他高 LR 参数组在余弦末端可能不会真正降到 `eta_min`，而是停在更高的绝对 LR。

同时日志只打印 `param_groups[0]`，容易让人误以为所有模块的 LR 都已经很低，但真正影响输出的 decoder/new modules 可能仍然高于预期。

#### 为什么这会影响后期 loss 回升判断

后期 loss 回升如果来自 decoder 或新模块 LR 偏高，仅看第一个参数组会漏掉关键证据。这样可能把问题误判为 OHEM 或损失权重冲突，而忽略实际的学习率尾段没有足够低。

#### 对应解决办法

1. scheduler 对每个参数组按自己的 `base_lr` 计算 `eta_min / group_base_lr`；
2. 每个 epoch 日志输出全部参数组 LR，至少包含 encoder、decoder、new_modules；
3. 后续判断 LR 是否回弹或是否衰减充分时，不再只看 `param_groups[0]`。

---

### 5.4 当前 MAE 日志实际不是纯 MAE，容易误导“像素误差已经很低”的判断

#### 代码证据

- `train.py:304-310`
- `Y_FFC_unified.py:1333-1338`

`MetricTracker` 中的 MAE 来源是：

```python
self.tot_mae += loss_dict.get('l1', loss_dict.get('mae', 0.0))
```

而 `EnhancedCompositeLoss` 里的 `l1` 实际是带边缘权重的 Charbonnier 项：

```python
l_l1 = (torch.sqrt((pred_diffuse - target)**2 + 1e-12) * wm).mean()
```

#### 问题本质

日志里标为 `MAE` 的值并不是严格的：

```python
mean(abs(pred - target))
```

它混入了 edge weight 和 Charbonnier 平滑形式，本质上是一个训练子损失，而不是纯评价指标。

#### 为什么这会影响实验判断

文档前面提到“MAE / MSE 已经很低，但 SSIM 不再明显上涨”。其中 MSE 是真实计算的，但 MAE 目前不是纯 MAE。如果后续用这个值判断像素误差是否已经充分收敛，可能会误判：

- 边缘权重变化会影响 MAE 读数；
- `l1` 权重和定义调整会改变曲线含义；
- 不同实验之间的 `MAE` 可能不是固定口径指标。

#### 对应解决办法

1. `MetricTracker` 单独计算 `true_mae = torch.mean(torch.abs(pred - target)).item()`；
2. 当前 `loss_dict['l1']` 保留为 `l1_loss` 或 `weighted_charbonnier` 子损失；
3. 报告和曲线中区分“评价 MAE”和“训练 L1 子损失”。

---

### 5.5 综合最优评分混用了动态 total loss，不同 epoch 间可比性不足

#### 代码证据

- `train.py:1226-1228`
- `train.py:1282-1284`

当前每个 epoch 会更新 loss 权重：

```python
w = ws.step(criterion, ep)
```

综合评分使用：

```python
composite = COMPOSITE_ALPHA * vs + (1.0 - COMPOSITE_ALPHA) * (1.0 / (1.0 + vl))
```

#### 问题本质

`vl` 是动态权重下的 validation total loss。训练前 60% 内，`lambda_l1 / lambda_freq / lambda_perceptual / lambda_focal / lambda_dice / lambda_gc` 都在变化。

这意味着不同 epoch 的 total loss 不完全是同一个目标函数下的数值。把它直接放进跨 epoch 综合评分，可能会产生口径不一致：

- 早期 loss 的权重组合偏向快速收敛；
- 中后期 loss 的权重组合更偏向辅助项；
- 综合评分中的 loss 部分并不完全可比。

#### 为什么这会影响 best checkpoint

如果综合最优模型依赖动态 total loss，那么 best overall checkpoint 可能受权重日程影响，而不一定是真正视觉质量或固定指标最优。尤其在 SSIM 差距很小、loss 权重正在变化时，这个问题会放大。

#### 对应解决办法

1. 综合评分优先使用固定口径指标，例如 SSIM、true MAE、MSE、PSNR；
2. 或者只在动态权重冻结后再启用 loss 参与综合评分；
3. 如果仍使用 total loss，应额外记录“固定权重 validation loss”作为可比参考。

---

### 5.6 loss spike 跳过机制在梯度累积中会清空已累积的正常梯度

#### 代码证据

- `train.py:388-397`
- `train.py:399-415`

当前 spike batch 被跳过时会执行：

```python
optimizer.zero_grad()
continue
```

#### 问题本质

前文 4.7 已经讨论过“跳过困难 batch”会引入选择性偏置。这里还需要补充一个梯度累积层面的副作用：

当 `ACCUMULATION_STEPS = 4` 时，如果第 3 个 micro-batch 被判定为 spike，`optimizer.zero_grad()` 会把前 2 个正常 micro-batch 已经累积的梯度也清掉。

也就是说，实际被跳过的不只是异常 batch，而可能是一整个累积窗口中已经处理过的正常样本。

#### 为什么这会影响后期稳定性

训练后期平均 loss 很低，spike 判定更容易触发。一旦触发，它会同时造成：

- 困难样本不学习；
- 同一累积窗口内的正常样本梯度也被丢弃；
- 实际 optimizer step 数减少且不规则。

这会让学习率调度、EMA 更新、有效 batch size 都出现隐性波动。

#### 对应解决办法

1. 记录每个 epoch 的 spike batch 数量和发生位置；
2. 跳过异常 batch 时避免清空此前正常 micro-batch 的梯度，或显式重启整个累积窗口并记录；
3. 更推荐把 spike 处理改为强梯度裁剪、loss 降权或 scaler 调整，而不是直接丢弃。

---

### 5.7 频域相位损失对低幅值频率同等约束，可能引入无意义相位噪声

#### 代码证据

- `Y_FFC_unified.py:1310-1322`

当前频域损失为：

```python
p_norm = torch.abs(p_fft).clamp(min=1e-8)
g_norm = torch.abs(g_fft).clamp(min=1e-8)
cos_diff = cross.real / (p_norm * g_norm)
phase_loss = (1.0 - cos_diff).mean()
```

#### 问题本质

相位只有在频谱幅值足够大时才有稳定意义。对于接近零幅值的频率分量，相位本身非常不稳定，甚至可以近似看作噪声。

当前 `phase_loss` 对所有频率位置直接平均，意味着低能频率的相位差也会被纳入同等约束。虽然分母做了 `clamp`，但这只解决数值除零，不解决“低幅值相位没有可靠语义”的问题。

此外，`cos_diff` 没有显式 clamp 到 `[-1, 1]`。浮点误差或极低幅值频率可能让局部值超出合理范围，使 phase loss 出现异常贡献。

#### 为什么这会影响后期 loss

后期像素误差和主结构误差已经较低时，低能频率上的相位噪声可能继续驱动模型调整。这类调整未必改善 SSIM，却会让 total loss 不稳定，甚至与 diffuse 主任务发生冲突。

#### 对应解决办法

1. 用幅值权重或阈值 mask 只约束有意义的频率相位；
2. 对 `cos_diff` 做 `clamp(-1, 1)`；
3. 消融对比“只用幅度谱”“幅值加权相位”“当前无权重相位”对后期 loss 的影响。

---

### 5.8 FourierUnit 的 Tukey 窗可能反复削弱图像边界信息

#### 代码证据

- `Y_FFC_unified.py:178-186`

当前 FourierUnit 进入 FFT 前会执行：

```python
x_win = x * tukey_2d
ffted = torch.fft.rfftn(x_win, dim=fft_dim, norm=self.fft_norm)
```

#### 问题本质

Tukey 窗可以降低频谱泄漏，这对频域卷积是有帮助的。但它也会在每次 FourierUnit 中系统性降低特征图边缘区域的响应。

当前模型里 FFC / FourierUnit 在 encoder、gradient decoder、main decoder、enhancement layers 中多次出现。如果每次进入频域前都乘窗，而后续没有显式能量补偿，那么边界区域信息可能被反复压低。

#### 为什么这会影响 SSIM

条纹图和高光边界往往对局部空间位置非常敏感。频域窗函数造成的边缘衰减，可能在全图平均误差中不明显，但会影响：

- 图像边缘附近的条纹连续性；
- 高光边界的局部结构；
- 小尺度细节的相位一致性。

这些都是 SSIM 比 MAE / MSE 更敏感的区域。

#### 对应解决办法

1. 做 `Tukey 窗开启 / 关闭` 消融；
2. 测试反射 padding 后再 FFT，减少直接压低原图边缘；
3. 若保留窗函数，考虑能量补偿或只在特定 FFC 层使用；
4. 单独统计边界区域 SSIM，确认窗函数是否造成边界质量下降。

---

### 5.9 `Y_FFC_unified.py` 自带 CLI 训练链路与主 `train.py` 不一致，可能造成实验不可比

#### 代码证据

- `Y_FFC_unified.py:2392-2419`
- `Y_FFC_unified.py:2523-2552`
- `train.py:1020-1158`

主要差异包括：

1. 主训练脚本使用 `EnhancedCompositeLoss`，备用 CLI 使用 `CompositeLoss`；
2. 主训练脚本的 `ImprovedFringeDataset` 正确把 `(H, W)` 转成 PIL 的 `(W, H)`，备用 `FringeDataset` 的 tuple resize 逻辑容易按 PIL 顺序误用；
3. 主训练脚本的增强对非正方形图只做 180 度旋转，备用 `PairedTransform` 对非正方形图也可能做 90 / 270 度旋转；
4. 主训练集 `gt_specular` 是 `(input-target).clamp(0,1)`，备用 `FringeDataset` 中是 `input_img - target_img`，没有 clamp；
5. 主训练脚本的 scheduler step 跟随 optimizer step，备用 `Y_FFC_unified.py` 的 `train_one_epoch` 中 scheduler step 在每个 mini-batch 后调用。

#### 问题本质

项目中同时存在两个训练入口，但它们的损失函数、数据预处理、增强策略、scheduler 粒度并不一致。如果不同实验混用这两个入口，得到的曲线和 checkpoint 很难直接比较。

#### 为什么这会影响诊断

当用户观察到 SSIM 平台或后期 loss 回升时，必须先确认实验到底来自哪个入口。否则可能把备用 CLI 的数据/调度问题误认为主 `train.py` 的问题，或者反过来。

#### 对应解决办法

1. 文档中明确当前主训练入口是根目录 `train.py`；
2. 后续若保留 `Y_FFC_unified.py train`，应统一数据集、损失函数、scheduler、EMA 保存键名和矩形图像处理逻辑；
3. 实验记录里写清入口脚本和关键配置，避免跨入口比较。

---

### 5.10 随机增强未完整固定 seed 与 worker 初始化，影响复现实验

#### 代码证据

- `train.py:1099-1100`：只固定了 train/val 划分的 generator
- `Y_FFC_unified.py:1595-1628`：增强大量依赖 Python `random`
- `train.py:1104-1111`：DataLoader 未设置 `worker_init_fn` 或 generator

#### 问题本质

当前训练/验证划分是固定的，但训练增强本身并没有完整固定随机源。尤其在 `num_workers=4` 时，Python `random` 在 worker 中的状态会影响：

- 随机翻转；
- 随机旋转；
- 随机裁剪；
- 亮度缩放；
- 合成高光位置、半径、强度。

这会让不同训练之间即使配置相同，也可能看到不同增强序列。对于高分段 SSIM 对比，这种随机性可能掩盖小幅改进。

#### 对应解决办法

1. 在配置中加入全局 seed，并统一设置 `random / numpy / torch`；
2. 给 DataLoader 配置 `generator` 和 `worker_init_fn`；
3. 对关键复现实验可启用 deterministic 设置，并在 summary 中记录 seed；
4. 可视化样本 seed 已有记录，但训练增强 seed 也需要同步记录。

---

## 6. 综合判断：当前最核心的瓶颈是什么

结合上面的分析，当前问题大致可以分成两层。

### 6.1 第一层：SSIM 上限的结构性瓶颈

优先级最高的几个点是：

1. `clamp(x - b)` 主路径本身带来的不可逆信息截断
2. refinement head 纠错能力不足
3. 多个 gate 都只有增强能力，没有抑制能力
4. 当前 `MSSSIM` 实现并非标准 MS-SSIM，主目标本身存在偏差风险
5. 梯度监督按样本独立归一化，导致梯度强度语义不稳定
6. skip attention 纯乘性门控可能直接压掉高分辨率细节
7. 梯度图 `x_g` 在最终输出头中裸拼接，容易把辅助分支误差直接带入主输出
8. 反卷积上采样、随机裁剪后回采样、ViT 固定重采样共同造成局部结构软化或伪影
9. 合成高光增强的形状先验与真实条纹高光可能并不匹配
10. 预训练输入归一化、预训练权重加载状态与 BN->GN 参数继承会影响高分段结构特征质量
11. 增强后 `gt_specular` 与最终 `input-target` 不一致，可能让 diffuse 与 specular 监督目标互相拉扯
12. 频域相位项和 FourierUnit Tukey 窗可能继续引入高频/边界区域的细节误差

这几项共同解释了为什么：

- MAE / MSE 已经很低
- 但 SSIM 迟迟冲不上去
- 而且高分段 SSIM 提升越来越依赖局部结构细节，而不是平均像素误差

### 6.2 第二层：训练后期 loss 回升的优化性瓶颈

优先级最高的几个点是：

1. scheduler step 粒度与 accumulation 不一致
2. 多目标总损失在后期发生冲突
3. OHEM 放大难样本波动
4. “loss spike 直接跳过 batch” 会对困难样本产生选择性偏置
5. 训练集 `drop_last=True` 可能让少量关键难样本长期缺席
6. 训练增强分布与验证分布差异较大，后期更容易放大 train/val 错位
7. 验证集当前只是随机划分，没有难度分层，验证曲线可能混入统计波动
8. 日志和 best-model 指标与训练目标不完全对齐
9. 分层 LR 日志和 `eta_min` 计算只关注第一个参数组，可能漏掉 decoder/new modules 的真实 LR
10. 综合评分混用动态 total loss，不同 epoch 的 loss 可比性不足
11. loss spike 跳过在梯度累积中会清空已累积的正常梯度
12. 随机增强未完整固定 seed，可能放大高分段实验的不可重复性

这几项共同解释了为什么：

- 曲线在后期不够稳
- 即使模型没有明显坏掉，total loss 仍可能重新上升
- 而且这类回升不一定都是“优化器问题”，也可能来自样本选择与评估链路本身

---

## 7. 后续改进方向汇总与优先级

下面给出建议的改造方向，但这里仍然只做方案整理，不动代码。

### 优先级 P0：先修正“评估体系与训练目标不对齐”的问题

这是最先做的，不是因为它一定能直接提升 SSIM，而是因为不修这个，后面的实验判断会失真。

#### 建议方向

1. 对齐训练损失中的 SSIM 项与日志记录指标
2. 同时跟踪：
   - single-scale SSIM
   - MS-SSIM
   - 混合 SSIM 指标
3. 最佳模型选择不要只看单一 SSIM，要同时对比：
   - best SSIM
   - best loss
   - best composite
4. 明确推理阶段是否真正使用了 EMA
5. 单独记录 true MAE，不再把 edge-weighted Charbonnier 当作 MAE
6. 日志输出全部参数组 LR，尤其是 decoder / new_modules
7. 记录 ResNet / ViT / VGG 预训练权重加载状态与输入归一化状态

#### 作用

- 避免误判实验效果
- 避免把“指标定义不一致”误当成“模型不行”
- 避免把“预训练未生效”“LR 观察口径错误”“MAE 口径错误”误判为架构瓶颈

---

### 优先级 P1：解决 diffuse 主路径的结构瓶颈

这部分最可能直接决定 SSIM 能不能明显再上一个台阶。

#### 建议方向

1. 放宽或重构 `x - b` 之后的硬约束链路
2. 提高 refinement head 的纠错能力，而不是只允许 ±0.05 微调
3. 让 diffuse 分支在后期具备更强的局部结构恢复能力

#### 这类改进针对的问题

- 结构信息在 coarse 阶段就被截断
- refinement 只能修小误差，修不了结构错误

#### 预期收益

- 对 SSIM 的提升潜力最大
- 尤其可能改善高光边界与饱和区域

---

### 优先级 P1：把增强型 gate 改成同时具备增强和抑制能力

#### 建议方向

1. 让 gate 不再只能输出 `[0, 1]` 的放大因子
2. 允许对错误增强进行抑制或负向修正
3. 避免每一级都只把高频和亮区往上推

#### 这类改进针对的问题

- 过锐化
- halo
- 局部伪边缘
- 错误高光强化

#### 预期收益

- 更有利于提升结构自然度
- 有利于 SSIM 而不只是视觉上更“硬”

---

### 优先级 P1：修正预训练输入与增强监督的一致性

#### 建议方向

1. 对 ViT / VGG 等预训练分支使用匹配 ImageNet 预训练权重的归一化
2. 确认预训练权重加载失败时不会被静默忽略
3. 增强完成后重新校验或重算 `gt_specular = clamp(input - target, 0, 1)`
4. 对合成高光导致的饱和区域单独统计，避免 specular 监督与 diffuse 主路径冲突

#### 这类改进针对的问题

- 预训练特征分布偏移
- 随机 VGG 感知损失误导训练
- 增强后 `input-target` 与 `gt_specular` 不一致

#### 预期收益

- 让预训练和感知损失真正提供稳定结构先验
- 降低 diffuse / specular 两个监督目标之间的隐性冲突

---

### 优先级 P2：处理主解码器上采样与 ViT 重采样带来的空间误差

#### 建议方向

1. 检查 transposed conv 是否在高光边界处引入结构伪影
2. 对比更平滑或更稳定的上采样方案
3. 降低 ViT 路径固定 224 重采样对空间细节的破坏

#### 这类改进针对的问题

- 棋盘格伪影
- 条纹边界模糊
- 局部结构相位误差

#### 预期收益

- 对极限 SSIM 的提升更敏感
- 特别适合在主输出路径优化后继续往上抠

---

### 优先级 P2：修正频域相位项与 FourierUnit 边界处理

#### 建议方向

1. 给频域相位损失加入幅值权重或低能频率 mask
2. 对 `cos_diff` 显式 clamp 到 `[-1, 1]`
3. 消融 Tukey 窗、反射 padding、能量补偿对边界 SSIM 的影响

#### 这类改进针对的问题

- 低幅值频率相位噪声
- 边界区域被频域窗函数反复削弱
- 后期频域项继续驱动无收益调整

#### 预期收益

- 降低频域子损失在后期的无效波动
- 改善高光边界和图像边缘附近的结构一致性

---

### 优先级 P2：解决后期训练不稳定

#### 建议方向

1. 校正 scheduler 的 step 粒度与真实 optimizer step 的关系
2. 明确 warmup / cosine 衰减的真实生效位置
3. 单独核查 OHEM 对后期曲线的影响
4. 拆开分析子损失，确认到底是谁导致 total loss 回升
5. 修正分层 LR 的 `eta_min` 计算，并观察所有参数组的真实 LR
6. 处理 loss spike 在梯度累积中清空正常梯度的问题
7. 综合评分避免直接混用动态权重下的 total loss

#### 这类改进针对的问题

- 后期 loss 回升
- 训练后段振荡
- 实验可重复性差
- 有效 optimizer step 数和有效 batch size 隐性变化

#### 预期收益

- 更稳定地达到高质量收敛点
- 更容易判断架构改动是否真的有效

---

### 优先级 P3：数据与评估层的进一步增强

这一部分目前还没有做逐样本统计，因此这里只能作为待验证方向。

#### 建议方向

1. 统计训练集与验证集的高光强度分布
2. 单独评估高光边界区域、饱和区域、细条纹区域
3. 引入局部区域 SSIM 或困难区域评估
4. 统一 `Y_FFC_unified.py train` 与主 `train.py` 的训练链路，或明确前者仅作备用入口
5. 补充全局 seed、DataLoader worker seed 和增强随机源记录

#### 这类改进针对的问题

- 当前全图平均指标可能掩盖局部失败区域
- 数据分布不匹配可能被误认为是模型结构问题
- 不同入口、不同随机增强序列导致实验不可比

#### 预期收益

- 让后续实验判断更准确
- 可能发现“SSIM 抬不上去”其实集中在少数困难区域
- 提高关键实验的可复现性

---

## 8. 推荐的实验顺序

为了避免后续一次改太多变量，建议按以下顺序进行。

### 阶段 1：先把“看什么算变好”校准

先做：

1. 对齐训练指标与日志指标
2. 核查 EMA 保存与加载是否一致
3. 把 total loss 拆成子损失轨迹
4. 单独记录 true MAE、全部参数组 LR、预训练权重加载状态
5. 明确 best composite 是否使用固定口径指标

原因：

- 这一步能避免后面做大量实验却无法正确判断结果

### 阶段 2：优先处理最可能卡住 SSIM 上限的结构问题

先做：

1. `x - b` 硬约束链路
2. refinement head 修正能力
3. gate 只能增强不能抑制的问题
4. 预训练输入归一化与增强后 specular 监督一致性

原因：

- 这三项最可能直接决定 SSIM 能否明显提升
- 预训练和监督口径不稳会干扰结构改动的判断

### 阶段 3：处理后期收敛稳定性

再做：

1. scheduler 与 accumulation 步数对齐
2. OHEM 对后期的影响消融
3. 子损失权重后期冲突分析
4. loss spike 与梯度累积的交互修正
5. 分层 LR `eta_min` 和所有参数组 LR 日志修正

原因：

- 在主结构没修好前，单纯调训练很可能只能得到有限收益
- 但结构问题缓解后，训练稳定性会决定能不能稳定收敛到更高点

### 阶段 4：进一步追求高上限

最后再做：

1. 上采样方式优化
2. ViT 路径的空间保持优化
3. 局部区域指标与数据分布增强
4. 频域相位项与 Tukey 窗边界消融
5. 备用 CLI 训练链路统一与随机 seed 控制

原因：

- 这些更像是“逼近极限”的工作
- 适合在主瓶颈已缓解之后继续推进

---

## 9. 后续验证时必须重点观察的指标

后续真正改模型时，不应只看一个 SSIM 值，应至少同时记录：

### 9.1 主指标

- Val SSIM
- Val MS-SSIM
- 混合 SSIM 指标
- Val true MAE
- Val weighted Charbonnier / L1 子损失
- Val MSE
- Val PSNR

### 9.2 训练稳定性指标

- 每 epoch 全部参数组学习率
- 每 epoch total loss
- 每个子损失项的 epoch 曲线
- raw model 与 EMA model 的验证表现差异
- 每 epoch spike batch 数量
- 每 epoch 实际 optimizer step 数
- 预训练权重加载状态和输入归一化状态

### 9.3 困难区域指标

重点关注：

- 高光边界区域
- 饱和区域
- 细条纹区域
- 亮 diffuse 与 specular 混杂区域

如果全图 SSIM 提升不大，但这些区域改善明显，说明模型方向可能是正确的，只是当前平均指标还不够敏感。

---

## 10. 当前结论总结

### 10.1 关于 SSIM 偏低

当前 SSIM 提升受限，最可能不是单一原因，而是多个结构因素叠加：

1. `clamp(x - b)` 主路径过于刚性，容易产生不可逆信息截断；
2. refinement head 修正能力过弱，只适合微调，不适合修复结构错误；
3. 多个关键 gate 都只有增强能力，缺少抑制自由度，容易引入过增强误差；
4. 解码器反卷积与 ViT 重采样可能在局部结构上继续消耗 SSIM；
5. 预训练输入归一化、增强监督一致性、频域相位项和 Tukey 窗也可能继续压制高分段结构质量；
6. 训练目标与 best SSIM 跟踪指标不完全一致，干扰了实验判断。

### 10.2 关于后期损失回升

当前后期 loss 回升，最可能来自以下几类原因叠加：

1. scheduler 的 step 粒度与梯度累积不一致，导致学习率时间轴可能失真；
2. 复合损失过于复杂，后期不同目标之间发生冲突；
3. OHEM 放大了极难区域的波动；
4. 虽然动态权重已经做了 60% 后冻结修复，但后期目标漂移问题未必完全消失；
5. total loss 可能掩盖了“真正波动的只是少数子损失项”这一事实；
6. 分层 LR 观察口径、动态综合评分和 loss spike 跳过机制会进一步放大后期判断误差。

### 10.3 最值得优先做的事

如果后续要继续推进，建议优先顺序是：

1. 先修正指标与评估链路，确保实验结论可靠；
2. 再处理 diffuse 主路径和 refinement 的结构瓶颈；
3. 再处理 gate 的增强/抑制平衡、预训练归一化和增强物理一致性；
4. 再处理 scheduler、OHEM、loss spike 与后期多目标冲突；
5. 最后再进一步优化上采样、ViT 空间保持、频域相位项和 FourierUnit 边界处理。

---

## 11. 附：本次诊断中重点引用的代码位置

### 文档与主结构
- `X-FFC_V3_模型完整技术文档.md:43-48`
- `X-FFC_V3_模型完整技术文档.md:62-70`

### 模型结构
- `Y_FFC_unified.py:1424-1436` `SpecularAwareAttention`
- `Y_FFC_unified.py:1981-2013` 梯度注入
- `Y_FFC_unified.py:2051-2061` `_SpectralDecompositionUnit`
- `Y_FFC_unified.py:2082-2094` `ResidualRefinementHead`
- `Y_FFC_unified.py:2112-2115` `YFFC.forward`
- `Y_FFC_unified.py:699-760` ViT 固定重采样
- `Y_FFC_unified.py:1977, 1994, 2010, 2023` 主解码器反卷积上采样
- `Y_FFC_unified.py:705-717` ViT 预训练权重加载
- `Y_FFC_unified.py:1756-1768` ResNet50 预训练权重加载
- `Y_FFC_unified.py:178-186` FourierUnit Tukey 窗
- `Y_FFC_unified.py:1569-1571, 1623-1629` 高光增强与 specular 真值更新

### 损失与训练
- `Y_FFC_unified.py:1276-1377` `EnhancedCompositeLoss`
- `Y_FFC_unified.py:1059-1062` VGG16 感知损失权重加载
- `Y_FFC_unified.py:1310-1322` 频域相位损失
- `Y_FFC_unified.py:2243-2255` 分层 LR 的 `eta_min` 计算
- `train.py:157-191` `ImprovedWeightScheduler`
- `train.py:304-307` `MetricTracker` 中的 SSIM 记录
- `train.py:404-415` scheduler.step 调用位置
- `train.py:388-397` loss spike 跳过逻辑
- `train.py:1274` 当前 LR 日志读取第一个参数组
- `train.py:1150-1154` build_scheduler 参数
- `train.py:1296-1308` best model 保存与 EMA 键名
- `train.py:1322-1329` best SSIM 模型保存逻辑
- `train.py:1491-1508` 核心训练配置
- `Y_FFC_unified.py:2392-2419` 备用 CLI 训练入口
- `Y_FFC_unified.py:2523-2552` 备用 CLI 参数定义

### 推理加载链路
- `process_images.py:55-70` EMA 优先加载逻辑

---

## 12. 结语

从当前代码和训练现象来看，模型已经具备不错的基础能力，但它现在更像是进入了“高分段的结构瓶颈期”。

如果只继续沿着原有方向轻微调权重，SSIM 继续提升的空间可能有限。真正更有希望带来明显提升的，不是单纯再训更久，而是：

- 先把评估和优化目标对齐；
- 再对 diffuse 主路径、refinement 能力、gate 机制做更有针对性的改造；
- 同时把后期训练的时间尺度与多目标冲突问题理顺。

这样后续的每一步改动，才更有可能把 SSIM 从“已经不低”推进到“明显更高”。
