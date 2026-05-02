# VA 与权重消融实验 Checklist

## 目标

验证在当前最终模型 `InCTRL + PQA + TA` 基础上，不同分支权重和监督权重是否能进一步提升跨域少样本工业异常检测性能。

当前结论已经比较明确：直接开启 VA 没有稳定优于 No-VA final。因此后续实验分成两类：

1. **主线实验**：围绕当前最佳 No-VA 配置，微调 `IMAGE / PATCH / PQA / TEXT` 融合权重和关键 loss 权重。
2. **补充实验**：继续验证 VA 是否存在很窄的甜点区间，或作为论文负结果/对照实验。

核心问题：

1. 当前 No-VA final 是否还可以通过残差、PQA、TA 权重微调提升？
2. `PATCH_WEIGHT` 或 `PQA_WEIGHT` 上调是否能改善 VisA 高敏类别和 AITEX 纹理域？
3. `MASK_WEIGHT` 下降是否能降低源域像素监督带来的跨域纹理过拟合？
4. VA visual-text 分支是否只能作为极小辅助项，还是应彻底关闭？
5. AdaptCLIP 原版等权融合思想是否适合迁移到当前 InCTRL-Adapt 框架？

---

## 当前最佳基准：No-VA final

注意：这里的 No-VA final 指的是 **VA 不参与 score/loss，但保留原始 visual-side/TA 交替训练调度**。后续复现实验应检查 `train_history.csv` 中 phase 是否为 `visual/text` 交替，而不是 `single`。

### 配置

| 配置项 | 数值 | 说明 |
| --- | ---: | --- |
| `FUSION.IMAGE_WEIGHT` | 0.35 | InCTRL 全局残差图像级分支 |
| `FUSION.PATCH_WEIGHT` | 0.25 | InCTRL patch 残差分支 |
| `FUSION.PQA_WEIGHT` | 0.25 | PQA 全局分支 |
| `FUSION.TEXT_WEIGHT` | 0.15 | TA 文本分支 |
| `FUSION.VISUAL_WEIGHT` | 0.00 | VA visual-text 分支关闭 |
| `LOSS.IMAGE_WEIGHT` | 1.0 | 图像残差头监督 |
| `LOSS.PQA_WEIGHT` | 0.5 | PQA 全局监督 |
| `LOSS.MASK_WEIGHT` | 1.0 | PQA 像素级分割监督 |
| `LOSS.TEXT_WEIGHT` | 0.0 | TA 无独立 CE loss，仅通过 final loss 训练 |
| `LOSS.TEXT_MASK_WEIGHT` | 0.0 | TA map 无 mask loss |
| `LOSS.VISUAL_WEIGHT` | 0.0 | VA 无独立 CE loss |
| `LOSS.VISUAL_MASK_WEIGHT` | 0.0 | VA map 无 mask loss |

### 已记录结果

| 实验 | Shot | Dataset | AUROC mean ± std | InCTRL baseline | Delta | 状态 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| No-VA final | 2 | VisA | 0.9030 ± 0.0015 | 0.858 | +0.0450 | 已完成 |
| No-VA final | 2 | AITEX | 0.7994 ± 0.0058 | 0.761 | +0.0384 | 已完成 |
| No-VA final | 2 | ELPV | 0.8601 ± 0.0077 | 0.839 | +0.0211 | 已完成 |
| No-VA final | 4 | VisA | 0.8916 ± 0.0021 | 0.877 | +0.0146 | 已完成 |
| No-VA final | 4 | AITEX | 0.7999 ± 0.0007 | 0.790 | +0.0099 | 已完成 |
| No-VA final | 4 | ELPV | 0.8705 ± 0.0032 | 0.846 | +0.0245 | 已完成 |
| No-VA final | 8 | VisA | 0.9060 ± 0.0018 | 0.887 | +0.0190 | 已完成 |
| No-VA final | 8 | AITEX | 0.8003 ± 0.0034 | 0.806 | -0.0057 | 已完成 |
| No-VA final | 8 | ELPV | 0.8815 ± 0.0023 | 0.872 | +0.0095 | 已完成 |

命令模板

```bash
python train_local.py \
  --train_dataset mvtec \
  --test_dataset visa/aitex/elpv \
  --shot 2 \
  --max_epoch 15 \
  --steps_per_epoch 100 \
  --output_dir results/ablation_no_va_final_2shot_15ep \
  FUSION.IMAGE_WEIGHT 0.35 \
  FUSION.PATCH_WEIGHT 0.25 \
  FUSION.PQA_WEIGHT 0.25 \
  FUSION.TEXT_WEIGHT 0.15 \
  FUSION.VISUAL_WEIGHT 0.00 \
  LOSS.IMAGE_WEIGHT 1.0 \
  LOSS.PQA_WEIGHT 0.5 \
  LOSS.MASK_WEIGHT 1.0 \
  LOSS.TEXT_WEIGHT 0.0 \
  LOSS.TEXT_MASK_WEIGHT 0.0 \
  LOSS.VISUAL_WEIGHT 0.0 \
  LOSS.VISUAL_MASK_WEIGHT 0.0
``` 

### 复现检查：恢复交替调度后的 No-VA rerun

该结果用于检查恢复 `visual/text` 交替调度后，No-VA final 是否能回到旧记录水平。它不覆盖上方多种子 No-VA final 基准。

| Shot | Dataset | AUROC | AUPR | vs 旧 No-VA final | vs InCTRL baseline | 状态 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2 | VisA | 0.9012 | 0.9137 | -0.0018 vs 0.9030 | +0.0432 vs 0.858 | 已完成 | 基本接近旧 No-VA final |
| 2 | AITEX | 0.7909 | 0.5677 | -0.0085 vs 0.7994 | +0.0299 vs 0.761 | 已完成 | 明显高于 single rerun，但仍低于旧 No-VA |
| 2 | ELPV | 0.8432 | 0.9248 | -0.0169 vs 0.8601 | +0.0042 vs 0.839 | 已完成 | 没有回到旧 No-VA，是主要差异来源 |
| 2 | **MEAN** | **0.8451** | **0.8021** | **-0.0091 vs 0.8542** | **+0.0258 vs 0.8193** | 已完成 | 部分复现，但不应覆盖旧基准 |

VisA per-category 结果：

| Category | AUROC | AUPR |
| --- | ---: | ---: |
| candle | 0.9636 | 0.9669 |
| capsules | 0.8612 | 0.9212 |
| cashew | 0.9634 | 0.9824 |
| chewinggum | 0.9804 | 0.9916 |
| fryum | 0.9356 | 0.9739 |
| macaroni1 | 0.9012 | 0.9110 |
| macaroni2 | 0.7925 | 0.8008 |
| pcb1 | 0.8377 | 0.8644 |
| pcb2 | 0.8271 | 0.8369 |
| pcb3 | 0.8901 | 0.8825 |
| pcb4 | 0.8732 | 0.8385 |
| pipe_fryum | 0.9888 | 0.9947 |
| **MEAN** | **0.9012** | **0.9137** |

复现结论：

- 恢复交替调度后，VisA 从 single rerun 的 0.8800 回到 0.9012，基本接近旧 No-VA final 的 0.9030。
- AITEX 从 single rerun 的 0.7760 回到 0.7909，也明显恢复，但仍低于旧 No-VA final 的 0.7994。
- ELPV 只有 0.8432，低于旧 No-VA final 的 0.8601，也低于 A1-A4 中的部分结果，是这次复现不完全的主要来源。
- 三域平均 AUROC 为 0.8451，低于旧 No-VA final 平均 0.8542，但高于 single rerun 平均 0.8334，说明交替调度确实恢复了一部分性能。
- 该结果支持“交替调度重要”，但也说明当前复现还存在 seed、few-shot 采样、epoch 数、checkpoint 选择或评估样本差异，需要进一步核查。
- 复现实验必须检查 `train_history.csv` 的 `phase` 列是否为 `visual/text` 交替，以及实际训练 epoch 是否与旧实验一致。



### 已知结论

- VisA 和 ELPV 上 2/4/8-shot 均稳定高于原始 InCTRL baseline。
- AITEX 上 2/4-shot 高于 baseline，但 8-shot 低于 baseline。
- VisA 4-shot 低于 2-shot 和 8-shot，且多种子 std 很小，说明不是 few-shot 采样噪声。
- AITEX 8-shot 稳定低于 baseline，说明纺织纹理跨域泛化仍有退化。
- 当前模型实际主要由 InCTRL 残差主干、PQA 和 TA 贡献，VA 分支保留但不参与最终分数。

---

## 已完成实验归档

### C1：VA-small

目标：以最保守方式开启 VA，测试 VA visual-text 分支是否能在不破坏 InCTRL 残差主干的情况下提供增益。

| 配置项 | 数值 |
| --- | ---: |
| `FUSION.IMAGE_WEIGHT` | 0.32 |
| `FUSION.PATCH_WEIGHT` | 0.23 |
| `FUSION.PQA_WEIGHT` | 0.25 |
| `FUSION.TEXT_WEIGHT` | 0.15 |
| `FUSION.VISUAL_WEIGHT` | 0.05 |
| `LOSS.VISUAL_WEIGHT` | 0.2 |
| `LOSS.VISUAL_MASK_WEIGHT` | 0.0 |

| Shot | Dataset | AUROC | AUPR | FPR | FNR | vs No-VA | vs InCTRL baseline | 状态 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | VisA | 0.9001 | 0.9137 | TBD | TBD | -0.0029 vs 0.9030 | +0.0421 vs 0.858 | 已完成 | 略低于 No-VA，多数类别仍较高 |
| 2 | AITEX | 0.7974 | 0.5703 | TBD | TBD | -0.0020 vs 0.7994 | +0.0364 vs 0.761 | 已完成 | 略低于 No-VA，高于原始 InCTRL 2-shot baseline |
| 2 | ELPV | 0.8532 | 0.9300 | TBD | TBD | -0.0069 vs 0.8601 | +0.0142 vs 0.839 | 已完成 | 低于 No-VA，高于原始 InCTRL 2-shot baseline |

VisA per-category 结果：

| Category | AUROC | AUPR |
| --- | ---: | ---: |
| candle | 0.9639 | 0.9675 |
| capsules | 0.8445 | 0.9132 |
| cashew | 0.9548 | 0.9787 |
| chewinggum | 0.9810 | 0.9919 |
| fryum | 0.9422 | 0.9772 |
| macaroni1 | 0.8786 | 0.8915 |
| macaroni2 | 0.7850 | 0.7829 |
| pcb1 | 0.8313 | 0.8558 |
| pcb2 | 0.8490 | 0.8509 |
| pcb3 | 0.8838 | 0.8820 |
| pcb4 | 0.8968 | 0.8781 |
| pipe_fryum | 0.9898 | 0.9950 |
| **MEAN** | **0.9001** | **0.9137** |

结论：

- VisA 2-shot AUROC 为 0.9001，低于 No-VA final 的 0.9030，差值约 -0.0029。
- AITEX 2-shot AUROC 为 0.7974，低于 No-VA final 的 0.7994，差值约 -0.0020。
- ELPV 2-shot AUROC 为 0.8532，低于 No-VA final 的 0.8601，差值约 -0.0069。
- 三个目标域平均 AUROC 为 0.8502，低于 No-VA final 的 0.8542，差值约 -0.0039。
- VA-small 没有带来稳定收益，不建议扩展到 4/8-shot。

### C2：VA-strong

目标：测试较强 VA 权重是否导致过拟合或误报上升，作为“复杂适配器风险”的证据。

| 配置项 | 数值 |
| --- | ---: |
| `FUSION.IMAGE_WEIGHT` | 0.28 |
| `FUSION.PATCH_WEIGHT` | 0.20 |
| `FUSION.PQA_WEIGHT` | 0.25 |
| `FUSION.TEXT_WEIGHT` | 0.15 |
| `FUSION.VISUAL_WEIGHT` | 0.12 |
| `LOSS.VISUAL_WEIGHT` | 0.2 |
| `LOSS.VISUAL_MASK_WEIGHT` | 0.0 |

| Shot | Dataset | AUROC | AUPR | FPR | FNR | vs No-VA | vs VA-small | 状态 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | VisA | 0.8904 | 0.9061 | TBD | TBD | -0.0126 vs 0.9030 | -0.0097 vs 0.9001 | 已完成 | 明显低于 No-VA 和 VA-small |
| 2 | AITEX | 0.8005 | 0.5753 | TBD | TBD | +0.0011 vs 0.7994 | +0.0031 vs 0.7974 | 已完成 | 略高于 No-VA 和 VA-small |
| 2 | ELPV | 0.8574 | 0.9338 | TBD | TBD | -0.0027 vs 0.8601 | +0.0042 vs 0.8532 | 已完成 | 低于 No-VA，略高于 VA-small |

VisA per-category 结果：

| Category | AUROC | AUPR |
| --- | ---: | ---: |
| candle | 0.9690 | 0.9715 |
| capsules | 0.8282 | 0.9062 |
| cashew | 0.9474 | 0.9776 |
| chewinggum | 0.9806 | 0.9918 |
| fryum | 0.9438 | 0.9780 |
| macaroni1 | 0.8586 | 0.8790 |
| macaroni2 | 0.7952 | 0.7958 |
| pcb1 | 0.7839 | 0.8084 |
| pcb2 | 0.8062 | 0.8197 |
| pcb3 | 0.8970 | 0.8914 |
| pcb4 | 0.8849 | 0.8586 |
| pipe_fryum | 0.9902 | 0.9955 |
| **MEAN** | **0.8904** | **0.9061** |

结论：

- VA-strong 三域平均 AUROC 为 0.8494，低于 No-VA final 的 0.8542，也低于 VA-small 的 0.8502。
- VisA 从 No-VA 的 0.9030 下降到 0.8904，下降约 -0.0126，是主要退化来源。
- AITEX 从 No-VA 的 0.7994 小幅上升到 0.8005，但增益仅 +0.0011，不足以抵消 VisA 退化。
- 当前结果支持“VA 权重越大越容易破坏 VisA 泛化”的判断，VA 不适合成为强分支。

---

## 待进行实验

### 主线优先级

优先跑 `A1 -> A2 -> A3 -> A4`。每个实验先跑 2-shot，目标域为 VisA / AITEX / ELPV。只有 2-shot 相对 No-VA final 有收益或非常接近，才扩展到 4/8-shot 与多种子。

### A1：patch_up

假设：patch 残差比分支图像级 residual 更稳，提高 `PATCH_WEIGHT` 可能改善 VisA 高敏类别和 AITEX 纹理域。

| 配置项 | 数值 |
| --- | ---: |
| `FUSION.IMAGE_WEIGHT` | 0.30 |
| `FUSION.PATCH_WEIGHT` | 0.30 |
| `FUSION.PQA_WEIGHT` | 0.25 |
| `FUSION.TEXT_WEIGHT` | 0.15 |
| `FUSION.VISUAL_WEIGHT` | 0.00 |
| `LOSS.IMAGE_WEIGHT` | 1.0 |
| `LOSS.PQA_WEIGHT` | 0.5 |
| `LOSS.MASK_WEIGHT` | 1.0 |
| `LOSS.TEXT_WEIGHT` | 0.0 |
| `LOSS.TEXT_MASK_WEIGHT` | 0.0 |
| `LOSS.VISUAL_WEIGHT` | 0.0 |
| `LOSS.VISUAL_MASK_WEIGHT` | 0.0 |

命令模板：

```bash
python train_local.py \
  --train_dataset mvtec \
  --test_dataset visa/aitex/elpv \
  --shot 2 \
  --max_epoch 15 \
  --steps_per_epoch 100 \
  --output_dir results/ablation_patch_up_2shot_15ep \
  FUSION.IMAGE_WEIGHT 0.30 \
  FUSION.PATCH_WEIGHT 0.30 \
  FUSION.PQA_WEIGHT 0.25 \
  FUSION.TEXT_WEIGHT 0.15 \
  FUSION.VISUAL_WEIGHT 0.00
```

记录：

| Shot | Dataset | AUROC | AUPR | FPR | FNR | vs No-VA | vs InCTRL baseline | 状态 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | VisA | 0.8832 | 0.8989 | TBD | TBD | -0.0198 vs 0.9030 | +0.0252 vs 0.858 | 已完成 | 明显低于 No-VA，pcb/macaroni 类别退化明显 |
| 2 | AITEX | 0.7792 | 0.5593 | TBD | TBD | -0.0202 vs 0.7994 | +0.0182 vs 0.761 | 已完成 | 低于 No-VA，纹理域没有受益 |
| 2 | ELPV | 0.8469 | 0.9254 | TBD | TBD | -0.0132 vs 0.8601 | +0.0079 vs 0.839 | 已完成 | 低于 No-VA，仅略高于原始 InCTRL baseline |

VisA per-category 结果：

| Category | AUROC | AUPR |
| --- | ---: | ---: |
| candle | 0.9651 | 0.9683 |
| capsules | 0.8275 | 0.9032 |
| cashew | 0.9456 | 0.9764 |
| chewinggum | 0.9774 | 0.9911 |
| fryum | 0.9366 | 0.9744 |
| macaroni1 | 0.8505 | 0.8733 |
| macaroni2 | 0.7828 | 0.8055 |
| pcb1 | 0.7958 | 0.8197 |
| pcb2 | 0.7915 | 0.8044 |
| pcb3 | 0.8813 | 0.8755 |
| pcb4 | 0.8570 | 0.8002 |
| pipe_fryum | 0.9870 | 0.9945 |
| **MEAN** | **0.8832** | **0.8989** |

A1 初步结论：

- 三域平均 AUROC 为 0.8364，低于 No-VA final 2-shot 平均 0.8542，差值约 -0.0178。
- VisA 从 0.9030 下降到 0.8832，说明简单把 `PATCH_WEIGHT` 从 0.25 提到 0.30 会削弱整体图像级判别。
- AITEX 从 0.7994 下降到 0.7792，说明更高 patch 残差权重没有改善纹理域，反而可能放大跨域纹理差异。
- ELPV 从 0.8601 下降到 0.8469，也没有表现出定位/patch 证据增强带来的收益。
- VisA 高敏类别中 `pcb1`、`pcb2`、`pcb4` 和 `macaroni1` 较弱，A1 没有修复这些类别，反而整体低于 No-VA。
- 判定为负结果，不建议扩展到 4/8-shot 或多种子；优先继续 A2：`pqa_up`。

判定：

- A1 未达到扩展条件，停止该方向。
- 下一步优先跑 A2，验证把权重转给 PQA 是否比转给 patch residual 更稳。

### A2：pqa_up

假设：PQA 全局分支可能比图像级 residual 更抗跨域，把 0.05 权重从 image 挪给 PQA 可能提升稳定性。

| 配置项 | 数值 |
| --- | ---: |
| `FUSION.IMAGE_WEIGHT` | 0.30 |
| `FUSION.PATCH_WEIGHT` | 0.25 |
| `FUSION.PQA_WEIGHT` | 0.30 |
| `FUSION.TEXT_WEIGHT` | 0.15 |
| `FUSION.VISUAL_WEIGHT` | 0.00 |
| `LOSS.IMAGE_WEIGHT` | 1.0 |
| `LOSS.PQA_WEIGHT` | 0.5 |
| `LOSS.MASK_WEIGHT` | 1.0 |
| `LOSS.TEXT_WEIGHT` | 0.0 |
| `LOSS.TEXT_MASK_WEIGHT` | 0.0 |
| `LOSS.VISUAL_WEIGHT` | 0.0 |
| `LOSS.VISUAL_MASK_WEIGHT` | 0.0 |

命令模板：

```bash
python train_local.py \
  --train_dataset mvtec \
  --test_dataset visa/aitex/elpv \
  --shot 2 \
  --max_epoch 15 \
  --steps_per_epoch 100 \
  --output_dir results/ablation_pqa_up_2shot_15ep \
  FUSION.IMAGE_WEIGHT 0.30 \
  FUSION.PATCH_WEIGHT 0.25 \
  FUSION.PQA_WEIGHT 0.30 \
  FUSION.TEXT_WEIGHT 0.15 \
  FUSION.VISUAL_WEIGHT 0.00
```

记录：

| Shot | Dataset | AUROC | AUPR | FPR | FNR | vs No-VA | vs InCTRL baseline | 状态 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | VisA | 0.8858 | 0.9018 | TBD | TBD | -0.0172 vs 0.9030 | +0.0278 vs 0.858 | 已完成 | 略高于 A1，但仍明显低于 No-VA |
| 2 | AITEX | 0.7780 | 0.5615 | TBD | TBD | -0.0214 vs 0.7994 | +0.0170 vs 0.761 | 已完成 | 低于 No-VA，也略低于 A1 |
| 2 | ELPV | 0.8498 | 0.9267 | TBD | TBD | -0.0103 vs 0.8601 | +0.0108 vs 0.839 | 已完成 | 略高于 A1，但仍低于 No-VA |

VisA per-category 结果：

| Category | AUROC | AUPR |
| --- | ---: | ---: |
| candle | 0.9647 | 0.9680 |
| capsules | 0.8317 | 0.9059 |
| cashew | 0.9408 | 0.9746 |
| chewinggum | 0.9764 | 0.9908 |
| fryum | 0.9336 | 0.9732 |
| macaroni1 | 0.8627 | 0.8836 |
| macaroni2 | 0.7852 | 0.8085 |
| pcb1 | 0.8025 | 0.8281 |
| pcb2 | 0.7968 | 0.8072 |
| pcb3 | 0.8846 | 0.8781 |
| pcb4 | 0.8626 | 0.8092 |
| pipe_fryum | 0.9884 | 0.9949 |
| **MEAN** | **0.8858** | **0.9018** |

A2 初步结论：

- 三域平均 AUROC 为 0.8379，低于 No-VA final 2-shot 平均 0.8542，差值约 -0.0163。
- A2 比 A1 的三域平均 0.8364 略高约 +0.0015，但这个差距很小，不能说明 PQA 上调解决了问题。
- VisA 从 No-VA 的 0.9030 下降到 0.8858，虽然比 A1 的 0.8832 略好，但仍是明显退化。
- AITEX 从 No-VA 的 0.7994 下降到 0.7780，且比 A1 的 0.7792 更低，说明 PQA 上调也没有改善纹理域。
- ELPV 从 No-VA 的 0.8601 下降到 0.8498，只是略高于 A1 的 0.8469。
- VisA 的 `pcb1`、`pcb2`、`pcb4`、`macaroni2` 仍然是低分区域，A2 没有修复 No-VA 需要关注的高敏类别。
- 判定为负结果，不建议扩展到 4/8-shot 或多种子；优先继续 A3：`mask_down`。

判定：

- A2 未达到扩展条件，停止该方向。
- A1/A2 连续负结果说明：简单从 `IMAGE_WEIGHT` 挪权重给 patch 或 PQA 都会破坏当前 No-VA 平衡。
- 下一步优先跑 A3，验证是否不是融合权重问题，而是 `MASK_WEIGHT=1.0` 的监督强度问题。

### A3：mask_down

假设：`LOSS.MASK_WEIGHT=1.0` 可能让 PQA 像素分割监督偏向源域纹理，适度下降到 0.75 可能提升跨域泛化。

| 配置项 | 数值 |
| --- | ---: |
| `FUSION.IMAGE_WEIGHT` | 0.35 |
| `FUSION.PATCH_WEIGHT` | 0.25 |
| `FUSION.PQA_WEIGHT` | 0.25 |
| `FUSION.TEXT_WEIGHT` | 0.15 |
| `FUSION.VISUAL_WEIGHT` | 0.00 |
| `LOSS.IMAGE_WEIGHT` | 1.0 |
| `LOSS.PQA_WEIGHT` | 0.5 |
| `LOSS.MASK_WEIGHT` | 0.75 |
| `LOSS.TEXT_WEIGHT` | 0.0 |
| `LOSS.TEXT_MASK_WEIGHT` | 0.0 |
| `LOSS.VISUAL_WEIGHT` | 0.0 |
| `LOSS.VISUAL_MASK_WEIGHT` | 0.0 |

命令模板：

```bash
python train_local.py \
  --train_dataset mvtec \
  --test_dataset visa/aitex/elpv \
  --shot 2 \
  --max_epoch 15 \
  --steps_per_epoch 100 \
  --output_dir results/ablation_mask_down_2shot_15ep \
  FUSION.IMAGE_WEIGHT 0.35 \
  FUSION.PATCH_WEIGHT 0.25 \
  FUSION.PQA_WEIGHT 0.25 \
  FUSION.TEXT_WEIGHT 0.15 \
  FUSION.VISUAL_WEIGHT 0.00 \
  LOSS.MASK_WEIGHT 0.75
```

记录：

| Shot | Dataset | AUROC | AUPR | Pixel AUROC/PRO | FPR | vs No-VA | vs InCTRL baseline | 状态 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | VisA | 0.8859 | 0.9018 | TBD | TBD | -0.0171 vs 0.9030 | +0.0279 vs 0.858 | 已完成 | 与 A2 接近，但仍明显低于 No-VA |
| 2 | AITEX | 0.7613 | 0.5502 | TBD | TBD | -0.0381 vs 0.7994 | +0.0003 vs 0.761 | 已完成 | 几乎退回原始 InCTRL baseline，是主要退化来源 |
| 2 | ELPV | 0.8502 | 0.9290 | TBD | TBD | -0.0099 vs 0.8601 | +0.0112 vs 0.839 | 已完成 | 略高于 A2，但仍低于 No-VA |

VisA per-category 结果：

| Category | AUROC | AUPR |
| --- | ---: | ---: |
| candle | 0.9662 | 0.9695 |
| capsules | 0.8358 | 0.9078 |
| cashew | 0.9364 | 0.9726 |
| chewinggum | 0.9788 | 0.9916 |
| fryum | 0.9346 | 0.9733 |
| macaroni1 | 0.8722 | 0.8912 |
| macaroni2 | 0.7851 | 0.8051 |
| pcb1 | 0.8016 | 0.8228 |
| pcb2 | 0.7996 | 0.8053 |
| pcb3 | 0.8839 | 0.8782 |
| pcb4 | 0.8532 | 0.8112 |
| pipe_fryum | 0.9840 | 0.9932 |
| **MEAN** | **0.8859** | **0.9018** |

A3 初步结论：

- 三域平均 AUROC 为 0.8325，低于 No-VA final 2-shot 平均 0.8542，差值约 -0.0217。
- VisA 从 No-VA 的 0.9030 下降到 0.8859，与 A2 的 0.8858 几乎持平，说明降低 `MASK_WEIGHT` 没有改善 VisA。
- AITEX 从 No-VA 的 0.7994 下降到 0.7613，几乎回到原始 InCTRL 2-shot baseline 0.761，是 A3 最大退化来源。
- ELPV 从 No-VA 的 0.8601 下降到 0.8502，略高于 A2 的 0.8498，但仍没有达到 No-VA。
- A1/A2/A3 连续低于 No-VA，说明当前最优配置的 `IMAGE/PATCH/PQA/TEXT` 融合权重和 `MASK_WEIGHT=1.0` 都不宜轻易下调或转移。
- 判定为负结果，不建议扩展到 4/8-shot 或多种子；继续跑 A4：`tiny_text_ce` 作为最后一个主线局部消融。

判定：

- A3 未达到扩展条件，停止该方向。
- `LOSS.MASK_WEIGHT=0.75` 没有带来跨域收益，当前应保留 `LOSS.MASK_WEIGHT=1.0`。
- 下一步优先跑 A4，验证极小文本 CE 是否能改善 TA 分支校准。

### A4：tiny_text_ce

假设：TA 当前没有独立 CE loss，只通过 final loss 间接训练；给极小 `TEXT_WEIGHT=0.05` 可能改善 text logit 校准，但需要防止 TA 与残差/PQA 分支竞争。

| 配置项 | 数值 |
| --- | ---: |
| `FUSION.IMAGE_WEIGHT` | 0.35 |
| `FUSION.PATCH_WEIGHT` | 0.25 |
| `FUSION.PQA_WEIGHT` | 0.25 |
| `FUSION.TEXT_WEIGHT` | 0.15 |
| `FUSION.VISUAL_WEIGHT` | 0.00 |
| `LOSS.IMAGE_WEIGHT` | 1.0 |
| `LOSS.PQA_WEIGHT` | 0.5 |
| `LOSS.MASK_WEIGHT` | 1.0 |
| `LOSS.TEXT_WEIGHT` | 0.05 |
| `LOSS.TEXT_MASK_WEIGHT` | 0.0 |
| `LOSS.VISUAL_WEIGHT` | 0.0 |
| `LOSS.VISUAL_MASK_WEIGHT` | 0.0 |

命令模板：

```bash
python train_local.py \
  --train_dataset mvtec \
  --test_dataset visa/aitex/elpv \
  --shot 2 \
  --max_epoch 15 \
  --steps_per_epoch 100 \
  --output_dir results/ablation_tiny_text_ce_2shot_15ep \
  FUSION.IMAGE_WEIGHT 0.35 \
  FUSION.PATCH_WEIGHT 0.25 \
  FUSION.PQA_WEIGHT 0.25 \
  FUSION.TEXT_WEIGHT 0.15 \
  FUSION.VISUAL_WEIGHT 0.00 \
  LOSS.TEXT_WEIGHT 0.05
```

记录：

| Shot | Dataset | AUROC | AUPR | FPR | FNR | vs No-VA | vs InCTRL baseline | 状态 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | VisA | 0.8870 | 0.9027 | TBD | TBD | -0.0160 vs 0.9030 | +0.0290 vs 0.858 | 已完成 | 略高于 A2/A3，但仍明显低于 No-VA |
| 2 | AITEX | 0.7657 | 0.5532 | TBD | TBD | -0.0337 vs 0.7994 | +0.0047 vs 0.761 | 已完成 | 低于 No-VA，仅略高于原始 InCTRL baseline |
| 2 | ELPV | 0.8538 | 0.9309 | TBD | TBD | -0.0063 vs 0.8601 | +0.0148 vs 0.839 | 已完成 | 是 A1-A4 中 ELPV 最高，但仍低于 No-VA |

VisA per-category 结果：

| Category | AUROC | AUPR |
| --- | ---: | ---: |
| candle | 0.9662 | 0.9692 |
| capsules | 0.8302 | 0.9049 |
| cashew | 0.9442 | 0.9760 |
| chewinggum | 0.9784 | 0.9914 |
| fryum | 0.9332 | 0.9727 |
| macaroni1 | 0.8817 | 0.8973 |
| macaroni2 | 0.7860 | 0.8066 |
| pcb1 | 0.7905 | 0.8123 |
| pcb2 | 0.8024 | 0.8069 |
| pcb3 | 0.8849 | 0.8805 |
| pcb4 | 0.8615 | 0.8214 |
| pipe_fryum | 0.9844 | 0.9934 |
| **MEAN** | **0.8870** | **0.9027** |

A4 初步结论：

- 三域平均 AUROC 为 0.8355，低于 No-VA final 2-shot 平均 0.8542，差值约 -0.0187。
- A4 比 A3 的三域平均 0.8325 略高约 +0.0030，但仍低于 A2 的 0.8379，也明显低于 No-VA。
- VisA 从 No-VA 的 0.9030 下降到 0.8870；`macaroni1` 有一定改善，但 `pcb1`、`pcb2`、`pcb4` 仍然偏弱。
- AITEX 从 No-VA 的 0.7994 下降到 0.7657，仅比原始 InCTRL 2-shot baseline 0.761 高 +0.0047，说明极小文本 CE 没有改善纹理域泛化。
- ELPV 从 No-VA 的 0.8601 下降到 0.8538，是 A1-A4 中最高的 ELPV，但不足以抵消 VisA/AITEX 退化。
- 判定为负结果；`LOSS.TEXT_WEIGHT=0.05` 没有带来稳定收益，当前应保持 `LOSS.TEXT_WEIGHT=0.0`。

判定：

- A4 未达到扩展条件，停止该方向。
- 不建议继续测试 `LOSS.TEXT_WEIGHT=0.10`，因为 0.05 已经导致 AITEX 明显低于 No-VA。
- A1-A4 主线局部消融全部低于 No-VA final，当前最佳配置应保持不变。

### A1-A4 主线总结

| 实验 | VisA AUROC | AITEX AUROC | ELPV AUROC | 平均 AUROC | vs No-VA 平均 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| No-VA final | 0.9030 | 0.7994 | 0.8601 | 0.8542 | 0.0000 | 当前最佳 |
| A1 patch_up | 0.8832 | 0.7792 | 0.8469 | 0.8364 | -0.0178 | 负结果 |
| A2 pqa_up | 0.8858 | 0.7780 | 0.8498 | 0.8379 | -0.0163 | 负结果 |
| A3 mask_down | 0.8859 | 0.7613 | 0.8502 | 0.8325 | -0.0217 | 负结果 |
| A4 tiny_text_ce | 0.8870 | 0.7657 | 0.8538 | 0.8355 | -0.0187 | 负结果 |

结论：A1-A4 都没有超过 No-VA final，说明当前最优配置的融合权重、`MASK_WEIGHT=1.0` 和 `TEXT_WEIGHT=0.0` 都应保留。后续若继续消融，应优先作为论文负结果补充或改做结构性诊断，而不是继续在这组权重附近盲目搜索。

---

## 补充待进行实验：VA 对照

这些实验优先级低于 A1-A4。它们主要用于确认 VA 负结果、补足论文对照或回答 reviewer-style 问题。

### V1：VA-mid

目标：测试稍强 VA 是否能进一步提升 VisA，同时观察 AITEX/ELPV 误报风险。

| 配置项 | 数值 |
| --- | ---: |
| `FUSION.IMAGE_WEIGHT` | 0.30 |
| `FUSION.PATCH_WEIGHT` | 0.22 |
| `FUSION.PQA_WEIGHT` | 0.25 |
| `FUSION.TEXT_WEIGHT` | 0.15 |
| `FUSION.VISUAL_WEIGHT` | 0.08 |
| `LOSS.VISUAL_WEIGHT` | 0.2 |
| `LOSS.VISUAL_MASK_WEIGHT` | 0.0 |

命令模板：

```bash
python train_local.py \
  --train_dataset mvtec \
  --test_dataset visa/aitex/elpv \
  --shot 2 \
  --max_epoch 15 \
  --steps_per_epoch 100 \
  --output_dir results/ablation_va_mid_2shot_15ep \
  FUSION.IMAGE_WEIGHT 0.30 \
  FUSION.PATCH_WEIGHT 0.22 \
  FUSION.PQA_WEIGHT 0.25 \
  FUSION.TEXT_WEIGHT 0.15 \
  FUSION.VISUAL_WEIGHT 0.08 \
  LOSS.VISUAL_WEIGHT 0.2 \
  LOSS.VISUAL_MASK_WEIGHT 0.0
```

记录：

| Shot | Dataset | AUROC | AUPR | FPR | FNR | vs No-VA | vs VA-small | 状态 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | VisA | TBD | TBD | TBD | TBD | TBD | TBD | 未跑 |  |
| 2 | AITEX | TBD | TBD | TBD | TBD | TBD | TBD | 未跑 |  |
| 2 | ELPV | TBD | TBD | TBD | TBD | TBD | TBD | 未跑 |  |

### V2：VA-mask-small

目标：验证 `VISUAL_MASK_WEIGHT` 是否能提升 VA 局部图质量，或是否会引入跨域误报。

| 配置项 | 数值 |
| --- | ---: |
| `FUSION.IMAGE_WEIGHT` | 0.32 |
| `FUSION.PATCH_WEIGHT` | 0.23 |
| `FUSION.PQA_WEIGHT` | 0.25 |
| `FUSION.TEXT_WEIGHT` | 0.15 |
| `FUSION.VISUAL_WEIGHT` | 0.05 |
| `LOSS.VISUAL_WEIGHT` | 0.2 |
| `LOSS.VISUAL_MASK_WEIGHT` | 0.05 |

命令模板：

```bash
python train_local.py \
  --train_dataset mvtec \
  --test_dataset visa/aitex/elpv \
  --shot 2 \
  --max_epoch 15 \
  --steps_per_epoch 100 \
  --output_dir results/ablation_va_mask_small_2shot_15ep \
  FUSION.IMAGE_WEIGHT 0.32 \
  FUSION.PATCH_WEIGHT 0.23 \
  FUSION.PQA_WEIGHT 0.25 \
  FUSION.TEXT_WEIGHT 0.15 \
  FUSION.VISUAL_WEIGHT 0.05 \
  LOSS.VISUAL_WEIGHT 0.2 \
  LOSS.VISUAL_MASK_WEIGHT 0.05
```

记录：

| Shot | Dataset | AUROC | AUPR | Pixel AUROC/PRO | FPR | vs VA-small | 状态 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | VisA | TBD | TBD | TBD | TBD | TBD | 未跑 |  |
| 2 | AITEX | TBD | TBD | TBD | TBD | TBD | 未跑 |  |
| 2 | ELPV | TBD | TBD | TBD | TBD | TBD | 未跑 |  |

### V3：AdaptCLIP-style 等权融合

目标：验证 AdaptCLIP 原版“VA/TA/PQA 等权融合”思想在 InCTRL 残差主干存在时是否仍然成立。

| 配置项 | 数值 |
| --- | ---: |
| `FUSION.IMAGE_WEIGHT` | 0.20 |
| `FUSION.PATCH_WEIGHT` | 0.20 |
| `FUSION.PQA_WEIGHT` | 0.20 |
| `FUSION.TEXT_WEIGHT` | 0.20 |
| `FUSION.VISUAL_WEIGHT` | 0.20 |
| `LOSS.VISUAL_WEIGHT` | 0.5 |
| `LOSS.VISUAL_MASK_WEIGHT` | 0.0 |

命令模板：

```bash
python train_local.py \
  --train_dataset mvtec \
  --test_dataset visa/aitex/elpv \
  --shot 2 \
  --max_epoch 15 \
  --steps_per_epoch 100 \
  --output_dir results/ablation_adaptclip_style_2shot_15ep \
  FUSION.IMAGE_WEIGHT 0.20 \
  FUSION.PATCH_WEIGHT 0.20 \
  FUSION.PQA_WEIGHT 0.20 \
  FUSION.TEXT_WEIGHT 0.20 \
  FUSION.VISUAL_WEIGHT 0.20 \
  LOSS.VISUAL_WEIGHT 0.5 \
  LOSS.VISUAL_MASK_WEIGHT 0.0
```

记录：

| Shot | Dataset | AUROC | AUPR | FPR | FNR | vs No-VA | 状态 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | VisA | TBD | TBD | TBD | TBD | TBD | 未跑 |  |
| 2 | AITEX | TBD | TBD | TBD | TBD | TBD | 未跑 |  |
| 2 | ELPV | TBD | TBD | TBD | TBD | TBD | 未跑 |  |

### V4：VA-tiny，可选

目标：确认 VA 是否存在比 VA-small 更窄的极低权重甜点区间。

| 配置项 | 数值 |
| --- | ---: |
| `FUSION.IMAGE_WEIGHT` | 0.34 |
| `FUSION.PATCH_WEIGHT` | 0.24 |
| `FUSION.PQA_WEIGHT` | 0.25 |
| `FUSION.TEXT_WEIGHT` | 0.15 |
| `FUSION.VISUAL_WEIGHT` | 0.02 |
| `LOSS.VISUAL_WEIGHT` | 0.05 |
| `LOSS.VISUAL_MASK_WEIGHT` | 0.0 |

命令模板：

```bash
python train_local.py \
  --train_dataset mvtec \
  --test_dataset visa/aitex/elpv \
  --shot 2 \
  --max_epoch 15 \
  --steps_per_epoch 100 \
  --output_dir results/ablation_va_tiny_2shot_15ep \
  FUSION.IMAGE_WEIGHT 0.34 \
  FUSION.PATCH_WEIGHT 0.24 \
  FUSION.PQA_WEIGHT 0.25 \
  FUSION.TEXT_WEIGHT 0.15 \
  FUSION.VISUAL_WEIGHT 0.02 \
  LOSS.VISUAL_WEIGHT 0.05 \
  LOSS.VISUAL_MASK_WEIGHT 0.0
```

记录：

| Shot | Dataset | AUROC | AUPR | FPR | FNR | vs No-VA | vs VA-small | 状态 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | VisA | TBD | TBD | TBD | TBD | TBD | TBD | 未跑 |  |
| 2 | AITEX | TBD | TBD | TBD | TBD | TBD | TBD | 未跑 |  |
| 2 | ELPV | TBD | TBD | TBD | TBD | TBD | TBD | 未跑 |  |

---

## 扩展实验：多 shot 与多 seed

若 A1-A4 中任一配置在 2-shot 表现接近或优于 No-VA，则扩展：

```bash
for SHOT in 2 4 8; do
  python train_local.py \
    --train_dataset mvtec \
    --test_dataset visa/aitex/elpv \
    --shot $SHOT \
    --max_epoch 15 \
    --steps_per_epoch 100 \
    --output_dir results/<experiment_name>_${SHOT}shot_15ep \
    <CONFIG_OVERRIDES>
done
```

多种子复验优先级：

1. A1-A4 中 2-shot 三域平均最好的配置。
2. AITEX 8-shot，因为当前 No-VA 已低于 baseline。
3. 高敏感类别：pcb1、pcb2、pcb3、macaroni2。
4. 若所有 A1-A4 都低于 No-VA，则保留 No-VA final 作为最终配置。

---

## 总结果汇总表

| 实验 | 类型 | 关键改动 | Shot | VisA AUROC | AITEX AUROC | ELPV AUROC | 平均 AUROC | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| No-VA final | 基准 | VA=0，当前最佳权重 | 2 | 0.9030 | 0.7994 | 0.8601 | 0.8542 | 当前最佳 2-shot 已知配置 |
| No-VA rerun | 复现检查 | 恢复 visual/text 交替调度后重跑 | 2 | 0.9012 | 0.7909 | 0.8432 | 0.8451 | 部分复现，ELPV 未回到旧 No-VA |
| No-VA final | 基准 | VA=0，当前最佳权重 | 4 | 0.8916 | 0.7999 | 0.8705 | 0.8540 | VisA 4-shot 非单调 |
| No-VA final | 基准 | VA=0，当前最佳权重 | 8 | 0.9060 | 0.8003 | 0.8815 | 0.8626 | AITEX 8-shot 低于原始 baseline |
| VA-small | 已完成 | VA fusion=0.05, VA CE=0.2 | 2 | 0.9001 | 0.7974 | 0.8532 | 0.8502 | 三域均略低于 No-VA，未见稳定收益 |
| VA-strong | 已完成 | VA fusion=0.12, VA CE=0.2 | 2 | 0.8904 | 0.8005 | 0.8574 | 0.8494 | AITEX 略升，但 VisA 明显下降 |
| A1 patch_up | 已完成 | IMAGE 0.35 -> 0.30, PATCH 0.25 -> 0.30 | 2 | 0.8832 | 0.7792 | 0.8469 | 0.8364 | 负结果，三域均低于 No-VA，不扩展 |
| A2 pqa_up | 已完成 | IMAGE 0.35 -> 0.30, PQA 0.25 -> 0.30 | 2 | 0.8858 | 0.7780 | 0.8498 | 0.8379 | 负结果，略好于 A1 但仍明显低于 No-VA |
| A3 mask_down | 已完成 | MASK loss 1.0 -> 0.75 | 2 | 0.8859 | 0.7613 | 0.8502 | 0.8325 | 负结果，AITEX 几乎退回原始 baseline |
| A4 tiny_text_ce | 已完成 | TEXT CE loss 0.0 -> 0.05 | 2 | 0.8870 | 0.7657 | 0.8538 | 0.8355 | 负结果，略高于 A3 但仍明显低于 No-VA |
| V1 VA-mid | 待跑 | VA fusion=0.08, VA CE=0.2 | 2 | TBD | TBD | TBD | TBD | VA 补充对照 |
| V2 VA-mask-small | 待跑 | VA mask loss=0.05 | 2 | TBD | TBD | TBD | TBD | VA 补充对照 |
| V3 AdaptCLIP-style | 待跑 | 五分支等权融合 | 2 | TBD | TBD | TBD | TBD | VA/AdaptCLIP 负结果对照 |
| V4 VA-tiny | 可选 | VA fusion=0.02, VA CE=0.05 | 2 | TBD | TBD | TBD | TBD | 仅验证极小 VA 甜点区间 |

---

## 论文可用结论模板

### 若 A1/A2 提升

在 InCTRL 残差主干和 PQA 已提供稳定异常证据的情况下，性能瓶颈不在于额外引入 VA，而在于已有残差证据与 PQA 证据的权重分配。适度提高 patch residual 或 PQA 分支权重，可以在不增加复杂 adapter 竞争的前提下提升跨域泛化。

### 若 A3 提升

降低 PQA 像素级 mask 监督后，跨域 image-level AUROC 得到改善，说明过强像素监督可能增强源域纹理拟合而损害目标域泛化。最终模型应在定位质量和跨域分类性能之间选择更稳的 mask loss 权重。

### 若 A4 提升

极小文本 CE 监督可以改善 TA 分支校准，但该监督必须保持低权重，避免 text branch 与 reference-based residual / PQA 分支形成无约束竞争。

### 若 VA 继续不提升

实验表明，AdaptCLIP 风格 VA 在纯视觉-语言 adapter 框架中有效，但在 InCTRL 残差主干已经提供强局部差异信号时，VA visual-text 分支未带来稳定跨域增益。考虑到 VA local 分支在消融中表现出退化风险，最终模型关闭 VA 融合权重，以保留 InCTRL 残差和 PQA 的稳定性。

### 若 VA-mask 导致退化

VA 的像素图来自 patch-level visual-text similarity，而非专门的分割 decoder。对该分支施加强 mask supervision 会增强源域纹理拟合，但可能削弱跨域泛化。相比之下，PQA segmentation head 更适合作为主要像素级监督分支。

---

## Checklist

### 已完成

- [x] 记录 No-VA final 的 VisA 2/4/8-shot 多种子 AUROC。
- [x] 记录 No-VA final 的 AITEX 8-shot 多种子 AUROC。
- [x] 补查或补跑 No-VA 2-shot AITEX/ELPV，用于严格直接对照。
- [x] 跑 C1：VA-small 2-shot。
- [x] 跑 C2：VA-strong 2-shot。
- [x] 将已完成实验从待跑区归档到“已完成实验归档”。

### 待跑主线

- [x] 跑 A1：patch_up 2-shot。
- [x] 跑 A2：pqa_up 2-shot。
- [x] 跑 A3：mask_down 2-shot。
- [x] 跑 A4：tiny_text_ce 2-shot。
- [x] 根据 A1-A4 结果决定是否扩展 4/8-shot：均不扩展，保留 No-VA final。
- [ ] 对最佳配置进行多种子复验。

### 待跑补充

- [ ] 跑 V1：VA-mid 2-shot。
- [ ] 跑 V2：VA-mask-small 2-shot。
- [ ] 跑 V3：AdaptCLIP-style 等权融合对照。
- [ ] 可选跑 V4：VA-tiny 2-shot。

### 汇总与写作

- [ ] 汇总 `test_results.csv` 到本文件表格。
- [ ] 生成 AUROC/AUPR/FPR 对比图。
- [ ] 更新论文实验章节和附录复现说明。
