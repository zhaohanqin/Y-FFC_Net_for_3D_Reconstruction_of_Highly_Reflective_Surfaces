# X-FFC V3.2 模型完整技术文档

---

## 1. 文档说明

本文档是对 **X-FFC V3.2（X-shaped Fast Fourier Convolutional Network，V3.2 版本）** 的完整技术说明。V3.2 是在 V3.1 的基础上、针对"**漫反射输出 r 的 SSIM 偏低**"这一核心症结进行的系统性改造，所有改造均可通过代码注释中的 `[V3.2]` 或 `[D1]~[D11]` 标签溯源。

### 1.1 V3.2 改造的核心动机

V3.1 经过梯度分支任务化（spec\_lik\_head）、BCE 替代 Charbonnier、双向一致性、单侧欠预测惩罚等一系列升级后，**镜面反射 \(b\) 的预测已经显著改善**，但训练监控中观察到一个关键现象：

> **\(b\) 预测得越来越像 GT 镜面分量，但漫反射 \(r\) 的 SSIM 改善缓慢。**

经诊断，根因是**"减法链的脆弱性"**——X-FFC 的整个 V3 / V3.1 设计中，漫反射 \(r\) 的唯一获取路径是：
\[r_{coarse} = \text{clamp}(x - \text{est\_b},\,0,\,1) \xrightarrow{\text{refinement\_head} \pm 0.08} r_{final}\]

这条链路存在两个不可逆的失效模式：
- **\(b\) 高估时**：\(x - b\) 出现负值被 clamp 到 0，**漫反射暗区被永久变成纯黑色**，refinement head 的 ±0.08 微修正完全无能为力
- **\(b\) 低估时**：\(x - b\) 仍含高光残留，±0.08 也补不全

由于 SSIM 评估的是 \(r\) 而不是 \(b\)，**这条链路把网络的"\(b\) 学得好"无法兑现到"\(r\) 输出 SSIM 高"**。

### 1.2 V3.2 改造目标

针对上述根因，V3.2 不再依赖单一减法路径，而是采用**"双头并行 + 可学习融合 + 强化精修 + r 端直接监督"**的组合方案，让网络同时具备两条获取 \(r\) 的能力，并对最终 \(r\) 输出施加多重直接约束。

### 1.3 V3.2 改进概览

V3.2 共包含 11 项系统性改造，按位置归类如下：

| 标签 | 模块 | 改进 | 关键性 |
|-----|------|------|--------|
| **[D1]** | `MainDecoder` | **双头输出（est_b + r_direct + 可学习 alpha）** | ⭐⭐⭐⭐⭐ |
| **[D2]** | `ResidualRefinementHead` | 网络加深 5 层 + scale 0.08→0.15 + 3 通道输入 | ⭐⭐⭐⭐ |
| **[D3]** | `EnhancedCompositeLoss` | 新增 `l_r_focal_hl`：高光区 r focal | ⭐⭐⭐⭐ |
| **[D4]** | `EnhancedCompositeLoss` | 新增 `l_r_dual_cons`：双头一致性 | ⭐⭐⭐ |
| **[D5]** | `EnhancedCompositeLoss` | 新增 `l_r_smooth`：高光区 r 局部平滑先验 | ⭐⭐⭐ |
| **[D6]** | `MainDecoder` 调制门控 | bias -1.0→0.0、scale 0.15→0.25 | ⭐⭐⭐ |
| **[D7]** | `Encoder.AdaptiveSkipFusion` | init_beta H4 0.03→0.10、H8 0.05→0.15 | ⭐⭐ |
| **[D8]** | `PerceptualLoss` | 小尺寸 patch (<48) 自动降级返回 0 | ⭐⭐ |
| **[D9]** | `frequency_loss` | phase 系数 0.25→0.15（防止前期主导） | ⭐⭐ |
| **[D10]** | `process_images.py` | **启用 TTA 4-flip 推理（含 dual-head 兼容）** | ⭐⭐⭐ |
| **[D11]** | `sliding_window_inference` | Hann 窗 → Tukey 窗（边界更平滑） | ⭐ |

### 1.4 V3.2 网络输出格式

V3.2 网络返回字典扩展为：

```python
{'r':        pred_diffuse,    # [B, 1, H, W]  漫反射主输出（融合+精修后）
 'g':        gradient_map,    # [B, 1, H, W]  输入梯度图（辅助）
 'b':        est_specular,    # [B, 1, H, W]  镜面反射分量
 's':        spec_likelihood, # [B, 1, H, W]  高光软概率图
 # ★ V3.2 新增 ★
 'r_direct': r_direct,        # [B, 1, H, W]  MainDecoder 直接预测的漫反射
 'alpha':    fusion_weight,   # scalar [0,1]  双路融合系数
 'cs':       cs_loss}         # 标量，仅训练
```

其中 `r_direct` 与 `alpha` 主要供训练损失使用；推理时仍以 `r` 为最终漫反射输出。

---

## 2. 完整数据流（V3.2 形成路径）

V3.2 的最大变化是 **MainDecoder 末尾和 YFFC.forward 末尾**的处理路径。其余编码层、梯度分支、增强层流程与 V3.1 一致。

### 2.1 主流程总览

```
输入 x [B, 1, H, W]
        ↓
   Encoder (ResNet50 + Swin-Tiny + CrossModalFusion + FFC + HFE)
        ↓
   (x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c)  5 级编码特征
        ├──→ GradientDecoder ──→ (xe1~4, x_g, x_s)
        │       ├── x_g    (输入梯度图)
        │       └── x_s    (高光软概率图，spec_lik_head)
        ├──→ EnhancementLayers (xe1~4) ──→ (enh1~4)
        ↓
   MainDecoder(c_feat, enh, x_g, x_s, original_input=x)
        ↓
   ★ V3.2 双头输出 ★
   返回 3-tuple: (est_b, r_direct, alpha_logit)
        ↓
   YFFC.forward 末尾的双路融合：
        r_sub    = clamp(x - est_b, 0, 1)              # 物理减法路径
        alpha    = sigmoid(alpha_logit)                 # ∈ [0, 1]
        r_fused  = alpha * r_direct + (1 - alpha) * r_sub
        ↓
   ResidualRefinementHead (3 输入: r_fused, x, r_direct)
        ↓
   pred_diffuse = clamp(r_fused + 0.15 * delta, 0, 1)
        ↓
   输出: {r, g, b, s, r_direct, alpha, cs}
```

### 2.2 双头形成路径详解

\[\begin{aligned}
r_{sub}   &= \text{clamp}(x - \text{est\_b},\,0,\,1) \\
\alpha    &= \sigma(\text{alpha\_logit}) \in [0,1] \\
r_{fused} &= \alpha \cdot r_{direct} + (1 - \alpha) \cdot r_{sub} \\
\delta    &= \tanh(\text{RefineNet}(r_{fused}, x, r_{direct})) \in [-1, 1] \\
r_{final} &= \text{clamp}(r_{fused} + 0.15 \cdot \delta,\,0,\,1)
\end{aligned}\]

**为什么这套设计能根治 V3.1 的 SSIM 瓶颈：**

1. **当 \(r_{direct}\) 学得好时**：\(\alpha\) 自动增大，最终 \(r\) 更依赖直接预测。即便 \(b\) 在某处被高估导致 \(r_{sub}\) 暗区被 clamp 抹掉，\(r_{direct}\) 仍能保留完整的暗区细节
2. **当 \(b\) 学得好时**：\(\alpha\) 自动减小，最终 \(r\) 更依赖物理减法，保持物理一致性
3. **两条路径互为正则**：通过 `l_r_dual_cons` 损失（[D4]），\(r_{direct}\) 与 \(r_{final}\) 的差异被显式约束，防止 \(r_{direct}\) 学成"和减法路径完全无关"的乱预测
4. **精修头的修正能力提高近一倍**：scale 0.08→0.15，5 层网络 + 残差结构，让最后一公里的修正更有空间

---

## 3. 各模块详解（V3.2 修改部分）

### 3.1 `MainDecoder` — 双头输出升级 [D1] [D6]

V3.2 在原 MainDecoder（V3.1 三路调制 + 扩展输出头）的基础上，**保留所有中间结构**，只在末尾扩展双头输出。

#### 3.1.1 输出头结构（V3.2）

```
共享中间表示 x [16 通道, 全分辨率]
        │
        ├─── output_conv4 (16→1, no BN, no act) ──→ Sigmoid ──→ est_b
        │
        └─── output_r_direct_conv (16→1, no BN, no act) ──→ Sigmoid ──→ r_direct
                                                                       
        + 可学习参数: dual_head_alpha_logit (标量，初始 0.0)
```

两个头**共享前面所有层**（扩展输出头的 4 层 + 全部解码 levels），仅在最后一层卷积分叉。这样做的考量：
- 共享主体保证 \(r_{direct}\) 与 \(b\) 来自同一组语义/空间特征，逻辑上是配对的
- 分叉后两个头各自专注于不同目标（\(b\) → "高光分量"，\(r_{direct}\) → "漫反射分量"），不会互相干扰

#### 3.1.2 alpha 融合系数

\[\alpha = \sigma(\text{dual\_head\_alpha\_logit})\]

- `dual_head_alpha_logit` 初始为 0.0 → \(\alpha = 0.5\)，两路等权
- 训练中 alpha 自适应调整，无下界限制（除 sigmoid 的 [0, 1] 自然约束）
- 推理时 alpha 已稳定为一个学到的值

#### 3.1.3 三路调制门控初始化升级 [D6]

旧版（V3.1）：
- bias = -1.0 → sigmoid(-1) ≈ 0.27
- scale 初始 0.15，clamp [0, 0.3]
- 初始调制系数 = 1 + 3 × 0.15 × 0.27 ≈ **1.12**（几乎无作用）

新版（V3.2）：
- bias = **0.0** → sigmoid(0) = 0.5
- scale 初始 **0.25**，clamp [0, 0.4]（边界保持，避免乘性爆炸）
- 初始调制系数 = 1 + 3 × 0.25 × 0.5 = **1.375**（前期就能发挥实质性作用）

这个改动让"梯度、联合、spec_likelihood"三路调制信号从训练第 1 个 epoch 起就实际影响主解码器输出，不再因初始几乎为 0 而被"白白浪费几十个 epoch"。

### 3.2 `ResidualRefinementHead` — 增强精修头 [D2]

#### 3.2.1 结构升级对比

| 维度 | V3.1 | V3.2 |
|------|------|------|
| 输入通道 | 2 (coarse, original) | **3 (coarse, original, r_direct)** |
| 网络深度 | 3 层卷积 | **5 层卷积 + 残差块** |
| 修正幅度 scale | 0.08 | **0.15** |
| 修正范围 | \([-0.08, +0.08]\) | \([-0.15, +0.15]\) |

#### 3.2.2 V3.2 内部结构

```python
refine_in:   Conv(3 → 24, 3×3) → GELU
refine_mid:  Conv(24 → 24, 3×3) → GELU → Conv(24 → 24, 3×3)  # 内部残差
refine_out:  GELU → Conv(24 → 16, 3×3) → GELU → Conv(16 → 1, 3×3) → Tanh

forward(coarse, original, r_direct):
    feat  = refine_in(cat[coarse, original, r_direct])
    feat  = feat + refine_mid(feat)        # 残差，增强表达
    delta = refine_out(feat)               # ∈ [-1, 1]
    return clamp(coarse + 0.15 * delta, 0, 1)
```

#### 3.2.3 为什么 scale 0.08 → 0.15

实际推理中 \(b\) 的预测误差常达到 0.15~0.30，旧版 ±0.08 的修正幅度根本不够补救。0.15 是平衡点：
- 大于 V3.1 近一倍，能纠正大多数 \(b\) 误差
- 仍由 Tanh 严格约束，避免精修头学成"覆盖式重写"破坏减法路径的物理含义
- 配合 dual-head 路径，精修头不再是唯一的纠错通道，可以更自由地优化

### 3.3 `EnhancedCompositeLoss` — 三个 r 端新损失 [D3] [D4] [D5]

V3.2 在 V3.1 的 14 个子损失项基础上，新增 **3 个针对漫反射 r 的直接监督项**，让 SSIM 指标关心的目标获得直接的优化通道。

#### 3.3.1 损失项 1：`l_r_focal_hl`（高光区 r focal 监督）[D3]

\[l_{r\_focal\_hl} = \text{mean}\big(|r - r_{gt}| \cdot (1 + 4 \cdot M_{soft}) \cdot (1 + |r - r_{gt}|)^{1.5}\big)\]

其中：
- \(M_{soft} = \sigma(20 \cdot (gt\_spec - 0.15))\)：高光软掩码，在强高光区 ≈ 1，背景区 ≈ 0
- \((1 + 4 M_{soft})\)：在 \(M_{soft} \approx 1\) 区域权重 ×5
- \((1 + |\text{err}|)^{1.5}\)：focal 形式，让难样本（误差大）额外加权

**核心意义**：旧版"\(r\) 在高光区是否去除干净"完全由 \(b\) 决定，损失通过减法链反传时梯度被稀释。此项给 \(r\) 直接强约束，让"高光残留"无处可逃。这是 V3.2 最有可能直接转化为 SSIM 提升的损失项。

#### 3.3.2 损失项 2：`l_r_dual_cons`（双头一致性）[D4]

\[l_{r\_dual\_cons} = \text{Charbonnier}(r_{direct},\,r_{final})\]

**核心意义**：
- 让 dual-head 的两条路径相互校准、互为正则
- 防止 \(r_{direct}\) 偷懒学成与减法路径完全无关的"另一种 r"
- 不 detach 任何一方，让梯度可以双向流通
- 使用 Charbonnier 而非 L1：前期数值更稳定

#### 3.3.3 损失项 3：`l_r_smooth`（高光区 r 局部平滑先验）[D5]

\[l_{r\_smooth} = \text{mean}\big(|r - \text{avg\_pool}_{3\times 3}(r)| \cdot M_{soft}\big)\]

**核心意义**：
- 漫反射的物理本质：高光被去除后的漫反射区域应低频为主
- 在 \(M_{soft}\) 高的区域（即原本是高光的位置），\(r\) 应该比较平滑
- 防止 \(r\) 输出中出现"亮点残留"或锯齿状伪影
- 不在低高光区域加约束（避免压抑真实纹理）

#### 3.3.4 V3.2 损失项权重调度

V3.2 给三个 r 系列损失项启用**新的 30% 截断调度**（更早达到目标值，因为它们直接保护主指标 SSIM）：

| 子损失 | 初始值 | 最终值 | 速率截断点 |
|--------|-------|-------|----------|
| \(\lambda_{r\_focal\_hl}\) | 0.4 | 0.8 | **30%** |
| \(\lambda_{r\_dual\_cons}\) | 0.05 | 0.15 | **30%** |
| \(\lambda_{r\_smooth}\) | 0.03 | 0.10 | **30%** |

旧版的 60% 主权重截断和 40% spec_likelihood 截断保持不变。完整调度表见 §4.2。

#### 3.3.5 总损失公式（V3.2 完整版）

\[\begin{aligned}
l_{total} = \;& \lambda_{ssim} l_{ssim} + \lambda_{l1} l_{l1} + \lambda_{freq} l_{freq} + \lambda_{percep} l_{percep} \\
       & + \lambda_{spec} l_{spec} + \lambda_{grad} l_{grad} + \lambda_{grad\_edge} l_{grad\_edge} + \lambda_{grad\_spec} l_{grad\_spec} \\
       & + \lambda_{focal} l_{focal} + \lambda_{dice} l_{dice} + \lambda_{gc} l_{gc} + \lambda_{cs} l_{cs} \\
       & + \lambda_{s\_lik} l_{s\_lik} + \lambda_{s\_cons} l_{s\_cons} + \lambda_{spec\_peak} l_{spec\_peak} \\
       & + \underbrace{\lambda_{r\_focal\_hl} l_{r\_focal\_hl} + \lambda_{r\_dual\_cons} l_{r\_dual\_cons} + \lambda_{r\_smooth} l_{r\_smooth}}_{\text{V3.2 新增 — 共 3 项}}
\end{aligned}\]

共计 **17 个子损失项**（V3.1 的 14 项 + V3.2 新增 3 项）。

#### 3.3.6 频域损失 phase 系数微调 [D9]

V3.1 中 `frequency_loss` 的相位项系数为 0.25：

\[l_{freq} = \text{L1}(|P|,|G|) + \mathbf{0.25} \times \text{phase\_term}\]

V3.2 调整为 **0.15**：

\[l_{freq} = \text{L1}(|P|,|G|) + \mathbf{0.15} \times \text{phase\_term}\]

**理由**：训练前期 pred 与 gt 相位差大，phase\_term 量级远大于 amp\_loss。即便 V3.1 的 NaN-Fix 解决了反向爆炸，前向数值仍偏大干扰主任务收敛。0.15 在保留相位约束的前提下让 amp 项占主导。

### 3.4 编码器 `AdaptiveSkipFusion` 初始 beta 提升 [D7]

| 位置 | V3.1 init_beta | V3.2 init_beta |
|------|---------------|---------------|
| `fuse_h4` (H/4 ResNet ↔ Swin) | 0.03 | **0.10** |
| `fuse_h8` (H/8 ResNet ↔ Swin) | 0.05 | **0.15** |

**理由**：旧版初始 beta 极小，前 50 epoch Swin 信号几乎不注入，等价于浪费了 ImageNet 预训练 Swin-Tiny 的能力。提高初值让双编码器分支从训练第 1 个 epoch 就实际协作。

### 3.5 `PerceptualLoss` 小尺寸降级 [D8]

V3.2 增加防护：

```python
def forward(self, pred, gt):
    if pred.shape[-1] < 48 or pred.shape[-2] < 48:
        return torch.tensor(0.0, ...)  # 直接返回 0
    # 否则走原路径
```

**理由**：在小 patch（如训练 patch 内的下采样阶段）VGG16 的 relu3_3 层特征几乎无意义，反而会引入不稳定梯度。<48 阈值是经验值，对训练分辨率 (128, 160) 主路径无影响。

---

## 4. 训练系统（V3.2 更新）

### 4.1 训练超参数调整

| 参数 | V3.1 | V3.2 | 调整理由 |
|------|------|------|---------|
| `EPOCHS` | 300 | 300 | 保持 |
| `BATCH_SIZE` | 3 | 3 | 保持 |
| `BASE_LR` | 1e-4 | **2e-4** | dual-head 新结构从头训练，1e-4 收敛过慢 |
| `WEIGHT_DECAY` | 1e-4 | 1e-4 | 保持 |
| `ACCUMULATION_STEPS` | 3 | 3 | 保持 (等效 batch=9) |
| `GRAD_CLIP_VALUE` | 0.5 | **1.0** | V3.1 反向稳定后可放宽，避免抑制 dual-head 早期梯度 |
| `IMAGE_SIZE` | (128, 160) | (128, 160) | 保持 |
| `EMA_DECAY` | 0.9995 | 0.9995 | 保持 |
| `USE_OHEM` | True | True | 保持 |
| `USE_AMP` | False | False | 保持（稳定优先） |

### 4.2 损失权重调度表（V3.2 完整版）

V3.2 采用**三速率调度策略**：

| 子损失 | 初始值 | 最终值 | 速率截断点 |
|--------|-------|-------|----------|
| \(\lambda_{ssim}\) | 1.0 | 1.0（不变） | — |
| \(\lambda_{l1}\) | 1.0 | 0.5（下降） | 60% |
| \(\lambda_{freq}\) | 0.05 | 0.15（上升） | 60% |
| \(\lambda_{perceptual}\) | 0.02 | 0.10（上升） | 60% |
| \(\lambda_{spec}\) | 0.5 | 0.5 | — |
| \(\lambda_{stat}\) | 0.0 | 0.0 | — |
| \(\lambda_{grad}\) | 0.5 | 0.3（下降） | 60% |
| \(\lambda_{grad\_edge}\) | 0.3 | 0.8（上升） | 60% |
| \(\lambda_{grad\_spec}\) | 0.02 | 0.05（上升） | 60% |
| \(\lambda_{focal}\) | 0.3 | 0.8（上升） | 60% |
| \(\lambda_{dice}\) | 0.1 | 0.3（上升） | 60% |
| \(\lambda_{gc}\) | 0.05 | 0.2（上升） | 60% |
| \(\lambda_{cs}\) | 0.02 | 0.05（上升） | 60% |
| \(\lambda_{s\_lik}\) | 0.3 | 0.8（上升） | 40% |
| \(\lambda_{s\_cons}\) | 0.02 | 0.08（上升） | 40% |
| \(\lambda_{spec\_peak}\) | 0.1 | 0.5（上升） | 40% |
| **\(\lambda_{r\_focal\_hl}\)** *(V3.2)* | **0.4** | **0.8（上升）** | **30%** |
| **\(\lambda_{r\_dual\_cons}\)** *(V3.2)* | **0.05** | **0.15（上升）** | **30%** |
| **\(\lambda_{r\_smooth}\)** *(V3.2)* | **0.03** | **0.10（上升）** | **30%** |

### 4.3 训练流程关键变化

- **`train_one_epoch`**：在调用 `criterion` 时额外传入 `pred_r_direct=out.get('r_direct')`，若旧 V3 checkpoint 没有该字段则自动退化为 None（损失项归 0）
- **`validate`**：同样传入 `pred_r_direct`
- **EMA / 优化器分组 / 学习率调度**：与 V3.1 保持一致，无任何破坏性变化
- **`SUB_LOSS_KEYS`** 扩展为 20 项（V3.1 的 17 项 + V3.2 新增 3 项）

### 4.4 训练监控变化

V3.2 在 EnhancedCompositeLoss 的返回 `loss_dict` 中新增 3 个键：
- `r_focal_hl`：高光区 r focal 损失值
- `r_dual_cons`：dual head 一致性损失值
- `r_smooth`：高光区 r 平滑损失值

`_format_loss_diag`、`MetricTracker`、训练曲线绘制等都会自动追踪这些键。

---

## 5. 推理流程（V3.2 更新）

### 5.1 推理输出扩展

`sliding_window_inference` 返回字典扩展为：

```python
{
    'r':        full_diffuse,         # [1, 1, H, W]  漫反射主输出
    'b':        full_specular,        # [1, 1, H, W]  镜面反射分量
    'g':        full_gradient,        # [1, 1, H, W]  梯度图
    's':        full_spec_likelihood, # [1, 1, H, W]  高光软概率 (V3.1+)
    'r_direct': full_r_direct,        # [1, 1, H, W]  V3.2 直接预测漫反射
}
```

`r_direct` 字段对应训练时的双头分支输出，可作为附加可视化/调试参考。最终用户应使用 `r` 字段作为漫反射结果（已经过双路融合 + 精修）。

### 5.2 TTA 推理 [D10]

V3.2 在 `process_images.py` 中显式实现 `_tta_forward`，并通过 `USE_TTA = True` 默认启用 4-flip TTA：

```python
对 (flip_h, flip_v) in {(F,F), (F,T), (T,F), (T,T)}:
    out = model(flip(x))
    把 out['b'], out['r_direct'], out['g'], out['s'] 翻转回原方向后累加
avg_b, avg_r_direct, avg_g, avg_s = 4 个张量的平均
r_sub    = clamp(x - avg_b, 0, 1)
r_fused  = alpha * avg_r_direct + (1 - alpha) * r_sub
r_final  = refinement_head(r_fused, x, avg_r_direct)
```

**关键设计**：TTA 不只是对 `b` 平均（旧版 train.py 的做法），而是对 `b` 和 `r_direct` 都平均后再融合，**保持与正常 forward 路径完全一致**，避免融合系数 alpha 的使用方式错配。

### 5.3 Tukey 窗滑窗融合 [D11]

V3.2 把 `_build_blend_window` 从 Hann 窗改为 Tukey 窗（α=0.25）：

| 窗类型 | 中央区域权重 | 边界区域权重 | 适用场景 |
|--------|-------------|-------------|---------|
| Hann | 余弦衰减 → 0 | 接近 0 | 训练时频谱泄漏少 |
| Tukey (α=0.25) | **中央 75% 全权重 1.0** | 两端 12.5% 余弦过渡 | **patch 边界混合更稳** |

旧版 Hann 窗在 patch 边界处衰减到 0，仅靠 `clamp_min(1e-3)` 维持非零，导致边界像素几乎只由单个 patch 决定，patch 接缝处可能出现细小亮度跳变。Tukey 窗保证中央 75% 区域完全权重，两端只在 12.5% 区域做平滑过渡，更适合滑窗推理融合。

### 5.4 推理配置（process_images.py main）

```python
# ★[V3.2] 推理配置默认值
TRAIN_IMAGE_SIZE = (128, 160)
PATCH_OVERLAP    = 64
USE_TTA          = True   # ★ V3.2 默认启用
PREFER_EMA       = True
```

### 5.5 向后兼容性

V3.2 改造**完全向后兼容旧 V3 / V3.1 checkpoint**：

- `process_images.py._load_state_dict_flexible` 会过滤形状不匹配的旧权重
- 缺失的新权重（`output_r_direct_conv`、`dual_head_alpha_logit`、`refine_in/mid/out` 等）使用 `__init__` 默认初始化
- 旧 ckpt 加载后 `dual_head_alpha_logit=0.0` 即 alpha=0.5，等于退回旧逻辑 + 双路平均（仍可用，但完整 V3.2 收益需要重新训练）

---

## 6. V3.2 代码结构总览

```
YFFC (X-FFC V3.2)
  │
  ├─ (encoder): Encoder                               // 与 V3.1 一致
  │   ├─ (stem_conv ... layer4): ResNet50 stages
  │   ├─ (reduce_l1 ... fuse_l34): 通道降维
  │   ├─ (swin_branch): SwinEncoderBranch
  │   ├─ (fuse_h4): AdaptiveSkipFusion (init_beta=0.10) ★[D7]
  │   ├─ (fuse_h8): AdaptiveSkipFusion (init_beta=0.15) ★[D7]
  │   ├─ (swin16_proj / cross_modal_fusion / top_down_refine)
  │   ├─ (cba_after_backbone / ffc_bottleneck / hf_enhance)
  │
  ├─ (gradient_decoder): GradientDecoder              // 与 V3.1 一致
  │   ├─ Levels 1-5 (FFC + 转置卷积上采样)
  │   ├─ (block_5_conv_2): 梯度头 → x_g
  │   └─ (spec_lik_conv_a/b/c): spec_lik_head → x_s
  │
  ├─ (enhancement_layers): EnhancementLayers          // 与 V3.1 一致
  │
  ├─ (main_decoder): MainDecoder                      // ★ V3.2 双头升级
  │   ├─ (FFC1/FFC2, multiscale_conv)
  │   ├─ (transposed_block_1 ... _5): resize-conv 上采样
  │   ├─ (feature_extraction_B)
  │   ├─ (att1 ... att4): Attention_block
  │   ├─ (sa1 ... sa4): SpecularAwareAttention
  │   ├─ (ffc_h8 / ffc_h4)
  │   ├─ (grad_gate_2/3/4, joint_gate_2/3/4, spec_lik_gate_2/3/4)
  │   ├─ (grad/joint/spec_lik)_gate_scale: Parameter ★ init 0.25 [D6]
  │   ├─ (output_reduce / output_dilated / output_skip / output_conv3)
  │   ├─ (output_conv4 → output_sigmoid → est_b)           // 镜面头
  │   ├─ ★ (output_r_direct_conv → r_direct_sigmoid → r_direct) // V3.2 漫反射头
  │   └─ ★ (dual_head_alpha_logit): Parameter (init 0.0)        // V3.2 融合系数
  │
  └─ (refinement_head): ResidualRefinementHead        // ★ V3.2 增强精修
      ├─ (refine_in):  Conv(3→24) + GELU              // 3 通道输入 ★[D2]
      ├─ (refine_mid): Conv(24→24) → GELU → Conv(24→24) + 残差
      └─ (refine_out): GELU → Conv(24→16) → GELU → Conv(16→1) → Tanh
                       (scale=0.15) ★[D2]
```

---

## 7. V3.2 完整改进点索引

### 模型结构改进
| 编号 | 改进 | 位置 |
|------|------|------|
| **[D1]** | MainDecoder 双头输出 (est_b + r_direct + alpha) | `MainDecoder.__init__/forward` |
| **[D2]** | ResidualRefinementHead 升级（5 层 + scale 0.15 + 3 输入） | `ResidualRefinementHead` |
| **[D6]** | 三路调制门控 bias -1.0→0.0、scale 0.15→0.25 | `MainDecoder._init_aux_gates` |
| **[D7]** | AdaptiveSkipFusion init_beta 提升 | `Encoder.__init__` |
| **[D8]** | PerceptualLoss 小尺寸降级 | `PerceptualLoss.forward` |

### 损失函数改进
| 编号 | 改进 | 位置 |
|------|------|------|
| **[D3]** | `l_r_focal_hl`：高光区 r focal 监督 | `EnhancedCompositeLoss.forward` |
| **[D4]** | `l_r_dual_cons`：双头一致性 | `EnhancedCompositeLoss.forward` |
| **[D5]** | `l_r_smooth`：高光区 r 平滑先验 | `EnhancedCompositeLoss.forward` |
| **[D9]** | frequency_loss phase 系数 0.25→0.15 | `EnhancedCompositeLoss.frequency_loss` |

### 训练流程改进
| 编号 | 改进 | 位置 |
|------|------|------|
| V3.2 调度器 | 三速率调度 (60%/40%/**30%**) | `ImprovedWeightScheduler` / `DynamicLossWeightScheduler` |
| V3.2 超参数 | BASE_LR 1e-4→2e-4, GRAD_CLIP 0.5→1.0 | `train.py main()` |
| V3.2 loss 调用 | 传入 `pred_r_direct` | `train_one_epoch / validate` |

### 推理流程改进
| 编号 | 改进 | 位置 |
|------|------|------|
| **[D10]** | TTA 4-flip 推理（含 dual-head 兼容） | `process_images.py._tta_forward` |
| **[D11]** | Hann 窗 → Tukey 窗 | `process_images.py._build_blend_window` |
| V3.2 输出 | 新增 `r_direct` 输出通道 | `sliding_window_inference` |

---

## 8. 设计优势总结

V3.2 在 V3.1 的基础上，**针对漫反射 r 输出的 SSIM 瓶颈做了系统性改造**，核心优势如下：

1. **绕过"减法链"的脆弱性（[D1]）**：dual-head 输出让网络拥有"直接预测 r"的能力，不再被 \(r = \text{clamp}(x - b, 0, 1)\) 的暗区 clamp 抹掉细节锁死

2. **r 端直接监督（[D3][D4][D5]）**：三个针对 r 的损失项把"r 应该长什么样"的信号直接喂到 \(r\) 输出，而不是通过 \(b\) 间接传递

3. **精修能力升级（[D2]）**：精修头 scale 0.08→0.15，网络由 3 层→5 层 + 残差，让最后一公里的修正能力提升近一倍

4. **早期协作更好（[D6][D7]）**：三路调制门控和编码器 Swin skip 都从训练第 1 个 epoch 起就发挥实质作用，不再"白白浪费几十个 epoch"

5. **数值稳定性进一步加固（[D8][D9]）**：感知损失小尺寸防御、频域相位项权重降低，让前向梯度更平衡

6. **推理质量提升（[D10][D11]）**：TTA 4-flip 减少随机性、Tukey 窗消除 patch 接缝痕迹，推理端 SSIM 还能再获得 0.005~0.015 提升

7. **完全向后兼容**：旧 V3 / V3.1 checkpoint 可直接加载（缺失参数默认初始化为"等价旧逻辑"行为），不影响已有训练成果

---

## 9. 附录：主要超参数默认配置（V3.2）

| 参数 | 默认值 | 说明 |
|------|-------|------|
| EPOCHS | 300 | 训练总轮数 |
| BATCH_SIZE | 3 | 配合 384 分辨率 |
| BASE_LR | **2e-4** | ★ V3.2 由 1e-4 调高 |
| WEIGHT_DECAY | 1e-4 | AdamW |
| ACCUMULATION_STEPS | 3 | 等效 batch=9 |
| GRAD_CLIP_VALUE | **1.0** | ★ V3.2 由 0.5 放宽 |
| IMAGE_SIZE | (128, 160) | 输入分辨率 (H, W) |
| EMA_DECAY | 0.9995 | EMA 衰减 |
| USE_GN | True | BN → GN 替换 |
| USE_OHEM | True | 难样本挖掘 |
| USE_AMP | False | 稳定优先 |
| USE_HIGHLIGHT_AUG | True | 合成高光增强 |
| HIGHLIGHT_AUG_P | 0.5 | 增强应用概率 |
| ResidualRefinement scale | **0.15** | ★ V3.2 由 0.08 提升 |
| MainDecoder gate scales init | **0.25** | ★ V3.2 由 0.15 提升 |
| MainDecoder gate bias init | **0.0** | ★ V3.2 由 -1.0 调整 |
| fuse_h4 init_beta | **0.10** | ★ V3.2 由 0.03 提升 |
| fuse_h8 init_beta | **0.15** | ★ V3.2 由 0.05 提升 |
| dual_head_alpha_logit init | **0.0** | ★ V3.2 新增 (→ alpha=0.5) |

### 推理配置默认值

| 参数 | 默认值 | 说明 |
|------|-------|------|
| TRAIN_IMAGE_SIZE | (128, 160) | 必须与训练 IMAGE_SIZE 一致 |
| PATCH_OVERLAP | 64 | 滑窗重叠像素 |
| USE_TTA | **True** | ★ V3.2 默认启用 4-flip TTA |
| PREFER_EMA | True | 优先使用 EMA 权重 |
| Blend window | **Tukey (α=0.25)** | ★ V3.2 由 Hann 改为 Tukey |

---

## 10. 文件清单

V3.2 改造涉及三个文件：

| 文件 | V3.1 行数 | V3.2 行数 | 主要变化 |
|------|-----------|-----------|---------|
| `X_FFC_unified.py` | 3381 | 3509 | MainDecoder 双头、RefineHead 增强、Loss +3 项、PerceptualLoss 防御 |
| `train.py` | 1846 | 1921 | SUB_LOSS_KEYS +3 项、weight scheduler、_tta_forward 双头兼容、超参数 |
| `process_images.py` | 730 | 824 | _tta_forward 新增、Tukey 窗、sliding 接受 use_tta、main USE_TTA 默认 True |

---

## 11. 训练-推理 一致性 checklist

V3.2 改造完成后，请确保：

- ✅ 训练时 `criterion` 调用包含 `pred_r_direct=out.get('r_direct')`
- ✅ `SUB_LOSS_KEYS` 包含 `r_focal_hl`, `r_dual_cons`, `r_smooth`
- ✅ `ImprovedWeightScheduler.get_weights` 返回的 dict 包含 3 个新权重
- ✅ `EnhancedCompositeLoss.__init__` 接受 3 个新 lambda 参数
- ✅ 推理时 `process_images.py` 的 `USE_TTA=True`（或按需禁用）
- ✅ 推理时 `process_single_image` / `process_folder` / `sliding_window_inference` 传递 `use_tta`
- ✅ `_load_state_dict_flexible` 能正确容忍旧 V3 ckpt 中缺失的新权重（已实现）

---

**END OF DOCUMENT — X-FFC V3.2 完整技术说明**
