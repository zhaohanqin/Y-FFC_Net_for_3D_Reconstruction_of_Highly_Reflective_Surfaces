# Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces

## 项目概述

该仓库围绕高反光条纹图像的预处理、去高光重建与相位分析展开，核心目标是为高反射表面的三维重建提供更稳定的输入与分析工具。仓库主体分为四块：

- `Y_FFC_unified.py` / `train.py`：Y-FFC 主模型、损失函数、数据集、训练工具与训练入口。
- `process_images.py`：使用训练好的 Y-FFC 权重对单张图像或整批图像做去高光推理。
- `phase_analysis/`：三频相位计算、ROI 掩码生成、灰度剖面与相位可视化工具。
- `comporison_model/`：HDRNet、DCUNet 等对比模型及其训练脚本。

如果你第一次进入这个仓库，建议按下面顺序阅读：

1. 先看 [train.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/train.py) 理解默认训练流程与数据组织。
2. 再看 [process_images.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/process_images.py) 了解模型推理输出。
3. 需要做相位恢复与可视化时，再进入 [phase_analysis/README.md](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/phase_analysis/README.md)。

## 仓库快速导览

### 顶层目录

- `data/`
  默认训练数据目录，包含 `inputs/` 与 `targets/` 两个子目录。当前仓库内两侧各有 4802 张样本，脚本默认按文件名排序后一一配对。
- `comporison_model/`
  对比实验目录，包含 `HDRNet.py`、`DCUNet.py` 及对应训练脚本。
- `phase_analysis/`
  相位分析工具集，覆盖灰度剖面、三频绝对相位、ROI 掩码与 3D 可视化。
- `integrated_results/`
  已生成的相位分析结果目录，主要是掩码、相位数据和可视化图片，不是核心源码目录。
- `__pycache__/`
  Python 缓存目录，可忽略。

### 顶层关键文件

- [train.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/train.py)
  当前主训练脚本。默认从 `data/inputs` 和 `data/targets` 读取样本，输出到带时间戳的 `experiments/yffc_v2_*` 目录。
- [Y_FFC_unified.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/Y_FFC_unified.py)
  统一实现文件，包含 Y-FFC 网络结构、FFC 模块、数据集、损失、训练/测试工具，并带有命令行入口。
- [process_images.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/process_images.py)
  推理脚本。读取训练好的 checkpoint，输出去高光结果、镜面分量、梯度图和可视化对比图。
- [点云误差比较文件.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/点云误差比较文件.py)
  从文件名判断，主要用于点云误差对比分析，属于后处理/评估辅助脚本。
- [X-FFC_V3_模型完整技术文档.md](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/X-FFC_V3_模型完整技术文档.md)
  模型设计说明文档。
- [Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces.pdf](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces.pdf)
  论文或项目说明 PDF。
- [KD-Tree详解.md](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/KD-Tree详解.md)
  独立知识文档，与主训练链路关系较弱。

## 核心流程

### 1. 数据准备

训练脚本默认使用如下目录结构：

```text
data/
  inputs/
  targets/
```

- `inputs/`：含高反光条纹图像。
- `targets/`：对应的目标图像，通常是去高光后的参考结果。
- `train.py` 中的数据集会优先尝试同名配对，也兼容 `input_xxx -> target_xxx` 这种命名。

### 2. 训练 Y-FFC 主模型

[train.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/train.py) 采用“集中参数配置区”风格，常用默认值包括：

- 输入尺寸：`(128, 160)`
- 默认数据目录：`data/inputs`、`data/targets`
- 输出目录：`experiments/yffc_v2_<timestamp>`
- 训练产物：`models/`、`training_curves/`、`val_visualization/`、`training_summary.txt`

直接运行：

```powershell
python train.py
```

如果你希望用命令行方式驱动统一脚本，也可以优先看 [Y_FFC_unified.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/Y_FFC_unified.py) 的 CLI 入口；从源码可见，`train` 子命令至少要求 `--input_dir` 和 `--target_dir`。

### 3. 用训练权重做去高光推理

[process_images.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/process_images.py) 通过修改 `main()` 中的参数区运行，核心配置包括：

- `INPUT_PATH`：单张图像或整个文件夹路径
- `MODEL_PATH`：训练输出的模型权重
- `OUTPUT_DIR`：推理结果保存目录
- `PREFER_EMA`：如果 checkpoint 中有 EMA 权重，则优先加载

直接运行：

```powershell
python process_images.py
```

输出通常包括：

- `*_diffuse.png`：去高光后的主结果
- `*_specular.png`：提取出的镜面反射分量
- `*_gradient.png`：梯度分量
- `*_visualization.png`：输入与输出对比图

### 4. 相位分析与可视化

`phase_analysis/` 适合在去高光之后继续做三频绝对相位恢复和结果分析。更完整的主链路是：

1. 用 [process_images.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/process_images.py) 生成更干净的条纹图像。
2. 用 [phase_analysis/integrated_absolute_phase_tool.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/phase_analysis/integrated_absolute_phase_tool.py) 做 ROI、掩码、绝对相位与可视化。
3. 按需使用 [phase_analysis/fringe_gray_analysis.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/phase_analysis/fringe_gray_analysis.py) 或 [phase_analysis/three_frequency_phase_visualization.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/phase_analysis/three_frequency_phase_visualization.py) 做单项分析。

## 目录与关键文件说明

### 主模型与训练

- [Y_FFC_unified.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/Y_FFC_unified.py)
  仓库最重要的逻辑文件。集成了 FFC 频域模块、ViT 编码器分支、镜面高光感知模块、数据增强、损失函数与训练辅助能力。适合作为“整体实现入口”阅读。
- [train.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/train.py)
  基于 `Y_FFC_unified.py` 的工程化训练脚本，增加了更完整的日志、可视化、EMA、动态损失权重、分层学习率等训练管理逻辑。

### 推理与结果生成

- [process_images.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/process_images.py)
  负责加载模型并批量输出去高光结果，是从训练产物走向下游处理的关键桥梁。

### 相位分析工具

- [phase_analysis/integrated_absolute_phase_tool.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/phase_analysis/integrated_absolute_phase_tool.py)
  一站式流程脚本，整合 ROI 选择、掩码生成、三频绝对相位计算、结果保存和 2D/3D 可视化。
- [phase_analysis/three_frequency_phase_visualization.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/phase_analysis/three_frequency_phase_visualization.py)
  更聚焦于三频相位读取、展开和可视化。
- [phase_analysis/fringe_gray_analysis.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/phase_analysis/fringe_gray_analysis.py)
  灰度剖面与 ROI 对比分析工具，适合观察条纹强度变化。
- [phase_analysis/README.md](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/phase_analysis/README.md)
  该子目录的详细使用说明。

### 对比模型

- [comporison_model/HDRNet.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/comporison_model/HDRNet.py)
  单通道 U-Net 风格网络实现，用作对比基线之一。
- [comporison_model/DCUNet.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/comporison_model/DCUNet.py)
  DC-UNet 结构实现，用于高反光去除对比实验。
- [comporison_model/train_hdrnet.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/comporison_model/train_hdrnet.py)
  HDRNet 训练脚本。
- [comporison_model/train_dcunet.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/comporison_model/train_dcunet.py)
  DCUNet 训练脚本。
- [comporison_model/Y_FFC_unified.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/comporison_model/Y_FFC_unified.py)
  对比实验目录内保留的一份统一实现副本，通常不应优先于顶层版本维护。

## 文件之间的依赖与关系

- `train.py` 依赖 `Y_FFC_unified.py` 中的模型、增强和损失定义。
- `process_images.py` 依赖 `Y_FFC_unified.py` 中的 `YFFC` 网络定义，并读取训练阶段保存的 checkpoint。
- `phase_analysis/` 与训练脚本基本解耦，更像是下游分析工具链。
- `comporison_model/*.py` 与主模型训练流程并行存在，用于横向实验对比，而不是主流程必经步骤。

## 结果目录说明

### `integrated_results/`

该目录是已生成结果的汇总示例，不应视为源码目录。当前结构大致为：

- `mask/`：ROI 掩码，如 `mask.bmp`、`mask.npy`
- `phase_data/`：绝对相位、包裹相位、差频相位及 `mat/npy` 数据
- `visualization/`：线剖面图、相位伪彩色图、3D 表面图

这类目录适合保留少量样例用于复现说明，但不建议在 README 中逐个展开文件。

## 运行环境

仓库没有看到显式的 `requirements.txt`。从源码导入看，至少需要以下 Python 包：

- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `matplotlib`
- `tqdm`
- `opencv-python`
- `scipy`

建议环境：

- Python 3.8+
- 有 CUDA 时优先使用 GPU 训练/推理

一个可参考的安装方式：

```powershell
pip install torch torchvision numpy pillow matplotlib tqdm opencv-python scipy
```

## 最小使用路径

### 训练

```powershell
python train.py
```

### 推理

先在 [process_images.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/process_images.py) 里改好 `INPUT_PATH` 和 `MODEL_PATH`，再运行：

```powershell
python process_images.py
```

### 相位分析

先在 [phase_analysis/integrated_absolute_phase_tool.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/phase_analysis/integrated_absolute_phase_tool.py) 里改好 `PROJECTION_FOLDER`、`FREQ_COMBINATION`、`OUTPUT_DIR`，再运行：

```powershell
python phase_analysis\integrated_absolute_phase_tool.py
```

## 维护建议

- 优先维护顶层的 [Y_FFC_unified.py](/E:/code/Y-FFC_Net_for_3D_Reconstruction_of_Highly_Reflective_Surfaces/Y_FFC_unified.py)，避免与 `comporison_model/` 内副本漂移。
- 训练脚本和推理脚本目前都偏“手动改参数区”的用法，做复现实验时建议记录当次配置。
- `integrated_results/`、未来的 `experiments/`、`processed_results/` 都更接近输出目录，新增逻辑时不要把核心代码混放进去。
