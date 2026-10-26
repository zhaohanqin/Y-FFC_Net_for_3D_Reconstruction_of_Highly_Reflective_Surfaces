# 消融实验 B：去除梯度辅助分支（ABLATION_no_gradient）

**文件**：`X_FFC_no_gradient.py` / `train_no_gradient.py`
**对照**：完整 YFFC（baseline `train.py`，建议加同种子 42 重跑）

## 为什么这么做

完整模型是解码侧的 X 型结构：编码器瓶颈分叉成梯度辅助臂（GradientDecoder 输出 `g`、`s` 和 xe1..4）与主解码臂，辅助臂通过三条通路注入主臂——①EnhancementLayers 的 enh1..4 多尺度拼接；②`grad_gate/joint_gate/spec_lik_gate` 三路先验调制；③输出头处与 `g` 的直接拼接。其假设是：显式的"哪里是边界 + 哪里是高光"先验，优于让主解码器隐式学习。

本实验把整条辅助臂及其全部注入通路移除，主解码器各级通道相应缩减（1408→1280、704→640、352→320、176→160、49→48），网络退化为单解码器 U 型结构；训练侧同步把 `lambda_grad / lambda_grad_edge / lambda_grad_spec / lambda_s_lik / lambda_s_cons` 置 0（分支不存在则其监督必然移除）。

## 这样有什么效果（如何解读）

- 高光区域指标（focal/dice/spec 子损失、可视化中的高光残留）明显变差、边缘变模糊 → 辅助臂有效，价值集中在高光定位与结构先验；
- 漫反射整体 SSIM 不变但高光区残留增多 → 辅助臂的贡献集中在 `b`/高光局部而非全局结构；
- 基本不变 → 说明 `SpecularAwareAttention` 的亮度先验已足够，辅助臂（约占模型相当比例的参数与计算）可裁剪。

注意：分支级消融必然同时移除"辅助监督"与"辅助特征注入"两件事；若需进一步拆分，可追加二级实验"保留分支但置零 enh 与门控输入"。

## 为什么能与完整模型严格区分（体现结构优越性）

1. 编码器（含 Swin 融合）、所有 FFC、SpecularAwareAttention、skip 注意力门控、双头输出、精修头、镜面/漫反射侧全部损失**逐项保留**，唯一变量是辅助臂的存在与否——这是对"X 型双解码结构"假设最直接的检验；
2. dataloader 仍照常生成 `gt_grad / gt_spec`（gt_spec 仍服务于镜面侧损失），数据管线与 baseline 零差异；
3. 若完整模型显著占优，即可证明"梯度 + 高光概率先验臂"是结构优越性的来源之一，而非单纯堆参数。

## 训练方式

```bash
python train_no_gradient.py
```

日志中 grad/grad_edge 子损失会显示非零数值（模型返回全零占位 `g`），但权重为 0，不参与优化，可忽略。
