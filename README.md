# Y-FFC / X-FFC-Swin 高反光条纹图像去高光项目

本仓库面向高反光结构光条纹图像的去高光、漫反射恢复和后续相位分析。当前主模型已经从旧的 ResNet + ViT 结构升级为 X-FFC-Swin：使用 ResNet50 提取局部纹理，使用 Swin-Tiny 提取窗口上下文，并在编码器中做多尺度融合。

当前主流程：

1. `train.py` 训练 X-FFC-Swin。
2. `process_images.py` 加载训练 checkpoint 做单张或批量推理。
3. `phase_analysis/` 对去高光后的条纹图像做相位分析和可视化。
4. `decoupled_yffc/` 是拆分模块版实现，结构与顶层主实现保持同步，便于后续工程化维护。

## 当前模型结构

主模型入口是 `Y_FFC_unified.py` 中的 `YFFC`。

核心结构：

- `Encoder`
  - ResNet50 分支输出 `x_bb_1 ~ x_bb_4`，提供局部边缘、条纹和纹理特征。
  - `SwinEncoderBranch` 使用 `torchvision.models.swin_t`，输入为灰度图 `[B,1,H,W]`。
  - Swin 不再强制 resize 到 `224x224`，只在尺寸不能被 32 整除时做右侧和下侧 padding。
  - Swin 输出 `swin_4 / swin_8 / swin_16`。
  - H/4、H/8 使用 `AdaptiveSkipFusion` 自适应增强 ResNet skip。
  - H/16 使用 `CrossModalFusion` 与 ResNet 深层特征融合。

- `FFC bottleneck`
  - 融合后的深层特征依次经过 `cba_after_backbone -> FFC bottleneck -> HighFrequencyEnhancement`。
  - 主要用于频域增强和高频结构恢复。

- `GradientDecoder`
  - 从瓶颈特征和 encoder skip 中辅助预测梯度图 `g`。

- `MainDecoder`
  - 结合 encoder skip、增强特征和梯度信息预测镜面反射分量 `b`。

- 漫反射恢复
  - 使用 `r = clamp(x - b, 0, 1)` 得到初步漫反射图。
  - 再经过 `ResidualRefinementHead` 精修。

模型输出接口保持不变：

```python
{
    "r": pred_diffuse,
    "g": gradient_map,
    "b": est_specular,
}
```

## 为什么使用 Swin 替代旧 ViT

旧 ViT 分支通常需要把输入 resize 到 `224x224` 以适配固定 token grid。对结构光条纹图像来说，这会带来明显问题：

- 压缩高分辨率条纹细节；
- 改变条纹频率、周期和局部相位关系；
- 模糊高光边界和弱高光区域；
- 破坏后续相位恢复依赖的空间采样一致性。

当前 Swin 分支不会强制 `224x224` resize，而是直接处理训练/推理 patch 的原始尺寸，例如 `320x256`、`384x384`、`512x512`。如果输入尺寸不能被 32 整除，只做 padding，不做缩放。这一点对结构光条纹的相位保持非常重要。

大图仍不建议整图训练。推荐从真实大图裁剪 patch 训练，推理时使用 sliding-window，并让推理 patch 尺寸尽量与训练 `IMAGE_SIZE` 一致。

## 目录说明

```text
.
├── Y_FFC_unified.py                 # 顶层统一模型、损失、增强、训练工具
├── train.py                         # 当前主要训练脚本
├── process_images.py                # 推理脚本，支持滑窗 patch 推理
├── X-FFC_V3_模型完整技术文档.md       # 当前模型技术文档
├── decoupled_yffc/                  # 拆分模块版 X-FFC-Swin
├── phase_analysis/                  # 相位分析、ROI、灰度剖面、可视化工具
├── comporison_model/                # HDRNet、DCUNet 等对比模型
├── data/ data2/ data3/              # 本地训练数据目录
├── experiments/                     # 训练输出目录
└── processed_results/               # 推理输出目录
```

注意：`comporison_model` 目录名按仓库现状保留，不要为了拼写顺手重命名。

## 数据组织

当前 `train.py` 默认读取：

```text
data2/
  inputs/
  targets/
```

其中：

- `inputs/`：含高反光条纹输入图像；
- `targets/`：对应去高光或漫反射参考图像；
- 数据集会优先按同名匹配，也兼容 `input_xxx -> target_xxx`、数字后缀等常见命名方式。

如果要切换到 `data/` 或 `data3/`，直接修改 `train.py` 中 `main()` 的配置区：

```python
INPUT_DIR = 'data2/inputs'
TARGET_DIR = 'data2/targets'
```

## 训练

直接运行：

```powershell
python train.py
```

`train.py` 采用集中配置区，主要默认参数包括：

- `EPOCHS = 50`
- `BATCH_SIZE = 3`
- `ACCUMULATION_STEPS = 3`
- `BASE_LR = 2e-4`
- `IMAGE_SIZE = (128, 160)`
- `USE_EMA = True`
- `USE_HIGHLIGHT_AUG = True`
- `HIGHLIGHT_AUG_P = 1.0`
- `INPUT_DIR = 'data2/inputs'`
- `TARGET_DIR = 'data2/targets'`

当前默认 `IMAGE_SIZE` 仍是 `(128,160)`，这是较轻量的训练配置。使用 Swin 后，建议在显存允许时提升到至少 `(320,256)`，或者使用 `(384,384)`、`(512,512)`，原因是：

- Swin 的窗口注意力需要足够空间上下文；
- 过小 patch 容易截断条纹周期和高光结构；
- 更大 patch 更接近真实推理时的局部统计；
- 尽量选择 H/W 能被 32 整除的尺寸，减少 padding。

高反光模拟增强默认开启：

```python
USE_HIGHLIGHT_AUG = True
HIGHLIGHT_AUG_P = 1.0
```

验证集不会叠加该增强，用于观察真实恢复效果。

训练输出默认保存到：

```text
experiments/yffc_v2_<timestamp>/
  checkpoints/
  models/
  training_curves/
  val_visualization/
  training_summary.txt
```

## 推理

推理入口：

```powershell
python process_images.py
```

使用前修改 `process_images.py` 中 `main()` 的配置区：

```python
INPUT_PATH = r'你的单张图像或文件夹路径'
MODEL_PATH = r'训练得到的模型权重路径'
OUTPUT_DIR = 'processed_results'
DEVICE = 'cuda'
```

当前推理脚本会用：

```python
YFFC(swin_pretrained=False)
```

原因是推理 checkpoint 已经包含训练后的 Swin 权重，不需要再下载 ImageNet 预训练权重，因此离线推理更稳。

滑窗推理配置：

```python
TRAIN_IMAGE_SIZE = (128, 160)
PATCH_OVERLAP = 64
```

`TRAIN_IMAGE_SIZE` 应尽量与训练时的 `IMAGE_SIZE` 保持一致。也就是说，如果训练时改成 `(320,256)`，推理这里也应同步改成 `(320,256)`。这样可以减少训练和推理分辨率分布不一致带来的伪影。

输出通常包括：

- `*_diffuse.png`：最终漫反射/去高光结果；
- `*_specular.png`：估计的镜面反射分量；
- `*_gradient.png`：辅助梯度预测；
- `*_visualization.png`：输入与输出对比图；
- `processed_results/all_diffuse/`：集中收集的漫反射结果。

## 相位分析

去高光后可以继续使用 `phase_analysis/` 做下游分析。

常用入口：

```powershell
python phase_analysis\integrated_absolute_phase_tool.py
```

主要工具：

- `phase_analysis/integrated_absolute_phase_tool.py`
  - ROI、掩码、三频绝对相位、2D/3D 可视化的一站式流程。
- `phase_analysis/fringe_gray_analysis.py`
  - 灰度剖面和 ROI 对比分析。
- `phase_analysis/three_frequency_phase_visualization.py`
  - 三频相位读取、展开和可视化。
- `phase_analysis/README.md`
  - 相位分析子目录说明。

## 拆分模块版

`decoupled_yffc/` 是当前 X-FFC-Swin 的拆分版实现，适合阅读和后续模块化维护。

常用检查：

```powershell
cd decoupled_yffc
python -c "import torch; from yffc import YFFC; m=YFFC(swin_pretrained=False); x=torch.randn(1,1,320,256); y=m(x); print({k:v.shape for k,v in y.items()})"
```

顶层 `Y_FFC_unified.py` 仍是当前训练脚本直接依赖的主实现；拆分版用于保持结构清晰和工程化同步。

## 对比模型

`comporison_model/` 中包含对比实验模型：

- `HDRNet.py`
- `DCUNet.py`
- `train_hdrnet.py`
- `train_dcunet.py`

这些脚本用于横向实验对比，不是当前 X-FFC-Swin 主训练流程的一部分。

## 运行环境

仓库当前没有显式 `requirements.txt`。从源码导入看，至少需要：

- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `matplotlib`
- `tqdm`
- `opencv-python`
- `scipy`

参考安装：

```powershell
pip install torch torchvision numpy pillow matplotlib tqdm opencv-python scipy
```

建议：

- Python 3.8+
- 有 CUDA 时优先使用 GPU
- `torchvision` 版本需要包含 `torchvision.models.swin_t`

## 快速验证

静态检查：

```powershell
python -m py_compile Y_FFC_unified.py train.py process_images.py
```

模型前向 smoke test：

```powershell
python -c "import torch; from Y_FFC_unified import YFFC; m=YFFC(swin_pretrained=False); x=torch.randn(1,1,320,256); y=m(x); print({k:v.shape for k,v in y.items()})"
```

非 32 整除输入测试：

```powershell
python -c "import torch; from Y_FFC_unified import YFFC; m=YFFC(swin_pretrained=False); x=torch.randn(1,1,300,260); y=m(x); print({k:v.shape for k,v in y.items()})"
```

期望 `r/g/b` 都与输入尺寸一致。

## 维护建议

- 优先维护顶层 `Y_FFC_unified.py`、`train.py`、`process_images.py`。
- 同步架构改动时，再更新 `decoupled_yffc/` 和 `X-FFC_V3_模型完整技术文档.md`。
- 训练和推理脚本目前都偏“手动改配置区”的用法，做复现实验时应记录 `IMAGE_SIZE`、checkpoint、数据目录和增强配置。
- 不要把 `experiments/`、`processed_results/`、`outputs/`、`checkpoints/`、`logs/` 等产物目录当作源码目录维护。
- 旧 ViT checkpoint 不再兼容当前 Swin 结构；当前训练和推理都以新 X-FFC-Swin checkpoint 为准。
