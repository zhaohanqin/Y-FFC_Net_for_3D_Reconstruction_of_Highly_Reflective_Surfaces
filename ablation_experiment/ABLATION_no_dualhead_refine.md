# 消融实验 D：去除双头输出 + 残差精修头（ABLATION_no_dualhead_refine）

**文件**：`Y_FFC_no_dualhead_refine.py` / `train_no_dualhead_refine.py`
**对照**：完整 YFFC（baseline `train.py`，建议加同种子 42 重跑）

## 为什么这么做

完整模型（V3.2）的输出端由三件事构成：①主解码器额外输出直接漫反射头 `r_direct`；②可学习系数融合两条路径 r_fused = α·r_direct + (1−α)·clamp(x−b)；③`ResidualRefinementHead`（3 通道输入、含残差块、±0.15 幅度）做最终像素级精修。其设计动机是补救纯减法链 r = clamp(x−b) 的两类结构性失败：`b` **高估**时 x−b 被 clamp 抹零、暗区条纹信息不可逆丢失；`b` **低估**时高光残留无任何修正通路。

本实验把这三件事整体移除：主解码器换为 `MainDecoderSingleHead`（删除 r_direct 头与 alpha 参数，特征干逐行不变），网络删除精修头，输出端退化为旧式纯减法链 r = clamp(x−b, 0, 1)。模型文件自带数值自检：前向输出严格满足 r == clamp(x−b)。

## 这样有什么效果（如何解读）

- 高光区残留增多、暗区条纹细节丢失（`r_focal_hl` 子损失偏高、高光区 MAE 上升、固定可视化样本中 |Pred−GT| 热图在高光位置发亮）→ 证明双头 + 精修正是补救减法链脆弱性的有效机制；
- 训练曲线波动加大、对 `b` 的误差更敏感 → 额外证明该设计带来训练稳健性收益；
- 基本不变 → 说明 `b` 的预测精度已足够高、输出端冗余，可裁剪以简化部署。

## 为什么能与完整模型严格区分（体现结构优越性）

1. **唯一变量是输出端结构**：编码器双分支、全部 FFC、梯度辅助臂（g/s/enh/三路门控全部保留）、SpecularAwareAttention、主解码器特征干逐行不变；
2. **损失口径一致**：`l_r_focal_hl / l_r_smooth` 直接监督 r 本身、与 r 的形成方式无关，原样保留；`l_r_dual_cons` 因 `pred_r_direct=None` 自动为 0——这是结构前提消失的必然结果，属于消融本身而非额外改动，因此训练文件**无需修改权重调度器**；
3. 这是四组实验中与"减法链脆弱性"假设对应最直接的一组：若完整模型在高光区与暗区两端同时占优，即可把优越性精确归因到"双头融合 + 残差精修"，而非编码器或先验分支。

## 可选二级实验（进一步拆分贡献）

- **D-1 只删精修头**（保留双头融合）：在 YFFC.forward 中令 pred_diffuse = r_fused；
- **D-2 只删双头**（保留精修头）：用本文件的 MainDecoderSingleHead + 精修头作用于 r_sub。

D 与 D-1/D-2 的差值可分别量化"融合"与"精修"各自的贡献。

## 训练方式

```bash
python train_no_dualhead_refine.py
```

注意：用 `process_images.py` 评测该消融模型时，其 TTA 路径中的双路融合与 refinement_head 调用需对应跳过（输出字典无 `r_direct`，且模型无 `refinement_head` 属性；`train.py` 内的 `_tta_forward` 已做 hasattr 保护，推理脚本如有同名逻辑请同样保护）。
