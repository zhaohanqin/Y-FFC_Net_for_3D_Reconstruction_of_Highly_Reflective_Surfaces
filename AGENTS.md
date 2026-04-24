# AGENTS.md

## 工作方式

- 修改前先给出简短计划；如果只是只读排查，也说明将读取哪些上下文。
- 如果不确定 API、配置项、路径或文件关系，先用 `rg` / `rg --files` 搜索仓库。
- 默认用中文沟通，解释保持简洁，步骤和命令尽量可直接复制。
- 不要把 secrets、tokens、private keys、`.env` 内容或任何凭据写入代码、日志或回复。
- 不要添加 analytics、telemetry 或额外网络请求，除非用户明确要求。
- 仓库里可能有用户未提交的改动；不要回滚或覆盖无关改动。

## 项目概览

这是一个面向高反光表面三维重建的 Python / PyTorch 项目，核心链路是条纹图像去高光、模型训练、推理处理和相位分析。

主要入口：

- `train.py`：当前主要训练脚本，导入 `Y_FFC_unified.py` 中的 `YFFC`、损失、增强、EMA、scheduler 等组件。
- `Y_FFC_unified.py`：统一模型实现，包含 FFC 模块、ViT/CNN 融合、损失函数、数据增强、数据集、训练工具和 CLI。
- `process_images.py`：加载 checkpoint，对单张或批量图像做去高光推理并保存结果。
- `rotate_dataset_images.py`：数据集图像旋转或扩展为正方形的辅助脚本。
- `phase_analysis/`：相位恢复、ROI 掩码、灰度剖面、三频相位可视化等下游分析工具。
- `comporison_model/`：HDRNet、DCUNet 等对比模型与训练脚本。目录名保留现状，不要顺手重命名。

常见数据与输出目录：

- `data/`、`data2/`、`data3/`：本地数据目录，通常不应把新增数据提交进仓库。
- `experiments/`：训练输出目录，包含模型、曲线、验证可视化和训练摘要，已在 `.gitignore` 中忽略。
- `processed_results/`、`outputs/`、`results/`、`checkpoints/`、`logs/`：结果、日志或中间产物目录，通常不要作为源码改动提交。
- `integrated_results/`：已有相位分析结果示例/产物目录，改动前先确认是否确实需要。

## 常用命令

```powershell
python train.py
```

```powershell
python process_images.py
```

```powershell
python rotate_dataset_images.py
```

```powershell
python phase_analysis\integrated_absolute_phase_tool.py
```

仓库没有明确的测试套件或 `requirements.txt`。从源码导入看，常见依赖包括 `torch`、`torchvision`、`numpy`、`Pillow`、`matplotlib`、`tqdm`、`opencv-python`、`scipy`。

## 代码维护注意事项

- 优先维护根目录的 `Y_FFC_unified.py` 和 `train.py`；`comporison_model/` 中的副本更偏对比实验，不要无意中把它当作主实现。
- 训练与推理脚本当前多为“文件顶部/`main()` 内集中配置参数”的风格；改配置时保持这种局部、清晰的方式，除非用户要求重构 CLI。
- 图像尺寸在项目里常见 `(H, W)` 与 PIL 的 `(W, H)` 转换，改数据集或预处理逻辑时要特别核对。
- `train.py` 的 `ImprovedFringeDataset` 会按同名、`input_` -> `target_`、数字后缀等规则匹配目标图；改文件命名逻辑时要保持兼容。
- 模型权重、训练曲线、图像结果、`.npy/.npz/.mat`、压缩包和原始数据默认视为产物，不要写入源码说明之外的提交。
- 如果需要验证训练代码，优先做轻量级静态检查或小样本 smoke test；完整训练可能耗时并依赖 GPU/本地数据。

## 回复格式偏好

- 代码改动完成后，给出短摘要和变更文件列表。
- 调试问题时，按“假设、尝试过什么、最小修复”说明。
- 如果测试无法运行，明确说明原因和未覆盖风险。
