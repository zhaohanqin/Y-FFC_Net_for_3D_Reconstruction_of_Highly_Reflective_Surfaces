# Y-FFC Net 项目总览与训练流程

本项目实现了论文 “Y-FFC Net for 3D Reconstruction of Highly Reflective Surfaces” 的核心思想：通过 Y 形结构与频域 FFC 模块，从高反射条纹图像中分离漫反射与镜面反射，同时输出梯度图以帮助重建。

## 目录结构

```bash
├── Y_FFC.py                          # 主网络与数据集实现（Encoder/Decoders/SSIM/数据集）
├── ffc.py                            # FFC（Fourier Feature Convolution）模块实现
├── train.py                          # 训练/验证/测试与可视化脚本
├── README.md                         # 简要说明
├── PROJECT_OVERVIEW.md               # 本文档（项目详解）
├── Y-FFC Net for ... .md/.pdf        # 论文解读与原论文
└── data/
    ├── inputs/                       # 输入（高反射条纹图）
    └── targets/                      # 目标（漫反射条纹图）
```

## 模块关系与调用顺序

### 顶层脚本

- `train.py`
  - 准备数据集与 DataLoader
  - 创建模型 `YFFC`
  - 调用 `train_model()` 进行训练与验证
  - 训练结束后 `test_model()`、`visualize_results()` 输出指标与可视化

### 网络与数据

- `Y_FFC.py`
  - `class YFFC(nn.Module)`：整体 Y 形网络封装
    - `Encoder`：VGG16-BN 前四段作为 backbone 特征提取
    - `GradientDecoder`：由最深层特征开始逐级上采样，输出梯度图 `g`
    - `EnhancementLayers`：对梯度路径中间特征做增强（辅助主解码器）
    - `MainDecoder`：融合 Encoder 跳连 + 梯度增强分支 + 频域 FFC，输出镜面反射 `b`，漫反射 `r = x - b`
  - `FringeDataset`：自定义数据集，管理输入-目标映射与张量生成
  - `SSIM`、`statistical_loss`：损失函数
- `ffc.py`
  - `SpectralTransform`/`FourierUnit`：频域处理（FFT -> 1×1 Conv -> iFFT）
  - `FFC_BN_ACT`：局部/全局分支的 FFC 核心，含 BN/激活与形状对齐

### 训练流程图

```mermaid
flowchart TD
  A[train.py] --> B[创建 FringeDataset / DataLoader]
  B --> C[构建模型 YFFC]
  C --> D[训练循环]
  D --> D1[前向: YFFC.forward]
  D1 --> E1[Encoder 提取 x_bb_1..x_bb_4, x_c]
  D1 --> E2[GradientDecoder 上采样生成 g 及中间增强特征]
  D1 --> E3[EnhancementLayers 对增强特征变换]
  D1 --> E4[MainDecoder 融合 FFC + 注意力 + 跳连 输出 b]
  D1 --> E5[计算 r = x - b]
  D --> F[计算损失: SSIM/MAE/统计]
  F --> G[反向传播/优化器更新]
  G --> H[验证集评估]
  H --> I[保存最佳模型]
  I --> J[test_model()/visualize_results()]
```

## 关键模块与函数说明

### 1) 数据集 `FringeDataset`（Y_FFC.py）

- 目录：`data/inputs` 与 `data/targets`
- 文件映射：
  - 输入通常命名为 `input_XXXX.png`
  - 目标通常命名为 `target_XXXX.png`
  - 数据集内部自动将 `input_` 前缀映射为 `target_`；若同名存在则优先同名
- 返回四个张量（均为 [B,1,H,W]）：
  - 输入 `input_img`
  - 目标漫反射 `target_img`
  - 镜面反射 GT（模拟）`gt_specular = input - target`
  - 梯度 GT `gt_gradient`（当前用 target 的 x/y 梯度之一，dims=(1,2)）

### 2) 编码器 `Encoder`（Y_FFC.py）

- 基于 `VGG16_BN` 的前四段 `features`，输出多尺度特征：
  - `x_bb_1`: 1/2 尺度, 64通道
  - `x_bb_2`: 1/4 尺度, 128通道
  - `x_bb_3`: 1/8 尺度, 256通道
  - `x_bb_4`: 1/16~1/32 尺度（取决于切片池化），512通道
  - `x_c`: 在 `x_bb_4` 上用 `1×1` conv 压到 256 通道，作为瓶颈特征
- 输入为单通道时，先用 `1→3` 的卷积适配至 VGG 输入

### 3) 梯度解码器 `GradientDecoder`（Y_FFC.py）

- 从 `x_bb_4` 开始，经 `7×7 → 1×1 → 3×3` 提炼
- 逐级上采样并与 `x_bb_3 → x_bb_2 → x_bb_1` 对齐融合
- 最终再上采样一次到输入尺寸（与原图同 H×W），输出梯度图 `x_g`
- 同时在每个阶段导出中间特征 `x_to_be_enhanced_1..4` 提供给增强层

### 4) 增强层 `EnhancementLayers`（Y_FFC.py）

- 对 `x_to_be_enhanced_1..4` 做 7×7 的通道缩减与平滑增强，分别输出 `x_enhanced_1..4`
- 用于主解码器拼接，增强主支路的信息表达

### 5) 主解码器 `MainDecoder`（Y_FFC.py）

- 先对 `x_c` 通过两层 `FFC_BN_ACT`（核 3×3, padding=1）进行频域与时域融合，并残差连接回 `x_c`
- 接多尺度卷积 `MultiscaleConvBlock`（并行 1×1→7×7, 3×3, 1×1→3×3→3×3）
- 之后进入 5 个上采样/同分辨率块，逐级与 `x_bb_4→x_bb_1` 跳连以及 `x_enhanced_1..4` 拼接，用注意力块 `Attention_block` 做门控融合
- 末端再与 `x_g` 拼接，经过两层 3×3 输出镜面反射 `est_b`
- 输出：
  - `b = est_b`（镜面反射）
  - `r = input - b`（漫反射）

### 6) FFC 模块（ffc.py）

- `FourierUnit`：将输入做 rFFT，实虚分量拼接后用 `1×1` 卷积变换，再 iFFT 回时域；可选位置编码
- `SpectralTransform`：将特征先降维，再通过 `FourierUnit`（含局部FU）、最后 `1×1` 恢复通道
- `FFC_BN_ACT`：
  - 将输入通道切成局部/全局两支（通过 `ratio_gin/ratio_gout`）
  - 局部用标准 `Conv2d`，全局路径用 `SpectralTransform`
  - 在相加前做双线性插值对齐，确保空间尺寸严格一致
  - 输出分别经过 BN+激活，返回 `(local, global)` 两支

## 训练细节与损失函数

- 优化器：Adam（可在 `train.py` 中调参）
- 学习率调度：StepLR(step_size=30, gamma=0.5)
- 损失：
  - 漫反射 `L_d = (1-SSIM(r, gt_diffuse)) + MAE(r, gt_diffuse) + Statistical(r, gt_diffuse)`
  - 镜面反射 `L_s = (1-SSIM(b, gt_specular)) + Statistical(b, gt_specular)`
  - 梯度 `L_g = (1-SSIM(g, gt_gradient))`
  - 总损失：`L = L_d + L_s + L_g`
- `SSIM` 实现：
  - 高斯窗注册为 `buffer`，在前向时复制到输入张量的 device/dtype，并 repeat 到通道数，避免 CUDA/CPU 类型不一致错误

## 运行与验证

- 安装依赖：

  ```bash
  pip install -r requirements.txt
  ```

- 训练：

  ```bash
  python train.py
  ```

  初次运行若无数据，脚本会生成 200 组虚拟条纹数据至 `data/`。
- 输出：
  - 最佳权重：`best_model.pth`
  - 可视化：`results_sample_*.png`、`training_history.png`

## 常见问题与解决方案

- TorchVision 预训练权重参数变更：
  - 已将 `vgg16_bn(pretrained=True)` 替换为：

    ```python
    models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
    ```

- Windows 多进程 DataLoader 偶现报错：
  - 统一设置 `num_workers=0` 提升稳定性
- SSIM 报错 "Input type (cuda) and weight type (cpu)"：
  - 通过将窗口注册为 `buffer` 并在前向匹配 device/dtype 解决
- 形状不匹配（例如 30 vs 32）：
  - `FFC_BN_ACT` 中在相加前强制对齐；`MainDecoder` 拼接前也会对齐分辨率
- 输入分辨率建议：
  - 使用 16 的倍数（如 512×512）最稳

## 如何扩展/修改

- 更换 backbone：将 `Encoder` 中的 VGG16 替换为其他模型（注意保持四级跳连与瓶颈通道）
- 调整 FFC 比例：修改 `MainDecoder` 中 `FFC_BN_ACT` 的 `ratio_gin/ratio_gout` 控制局部/全局通道占比
- 改进损失：在 `train.py` 中替换或增添损失项（例如 Sobel 梯度、感知损失）
- 数据格式：若目标不是与输入严格对齐的漫反射，可在 `FringeDataset` 中定制读取与 GT 构造逻辑

## 参考

- 论文：Y-FFC Net for 3D Reconstruction of Highly Reflective Surfaces
- FFC 思想：频域变换 + 1×1 卷积进行全局上下文建模，与局部卷积互补

如需快速定位实现：

- 网络入口：`Y_FFC.YFFC.forward`
- 频域单元：`ffc.FourierUnit`
- 数据映射：`Y_FFC.FringeDataset.__getitem__`
- 训练循环：`train.train_model`
