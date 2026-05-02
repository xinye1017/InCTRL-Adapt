# VA 权重消融实验 Checklist

## 目标

验证在当前最终模型 `InCTRL + PQA + TA` 基础上，引入 `Visual Adapter (VA)` 是否能提升跨域少样本工业异常检测性能，并评估不同 VA 图像级权重、VA loss 权重和 VA mask loss 权重对 AUROC、AUPR、误报率和像素级定位的影响。

核心问题：

1. VA visual-text 分支是否能在 InCTRL 残差主干和 PQA 已存在时提供额外增益？
2. VA 的合理融合权重应是多少？
3. `VISUAL_MASK_WEIGHT` 是否会提升像素级定位，还是增加跨域误报？
4. AdaptCLIP 原版等权融合思想是否适合迁移到 InCTRL-Adapt 框架？

---

## 当前已完成实验：无 VA 最终模型

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
| No-VA final | 2 | VisA | 0.9030 ± 0.0015 | 0.858 | +0.0450 | ✅ 已完成 |
| No-VA final | 2 | AITEX | 0.7994 ± 0.0058 | 0.761 | +0.0384 | ✅ 已完成 |
| No-VA final | 2 | ELPV | 0.8601 ± 0.0077 | 0.839 | +0.0211 | ✅ 已完成 |
| No-VA final | 4 | VisA | 0.8916 ± 0.0021 | 0.877 | +0.0146 | ✅ 已完成 |
| No-VA final | 4 | AITEX | 0.7999 ± 0.0007 | 0.790 | +0.0099 | ✅ 已完成 |
| No-VA final | 4 | ELPV | 0.8705 ± 0.0032 | 0.846 | +0.0245 | ✅ 已完成 |
| No-VA final | 8 | VisA | 0.9060 ± 0.0018 | 0.887 | +0.0190 | ✅ 已完成 |
| No-VA final | 8 | AITEX | 0.8003 ± 0.0034 | 0.806 | -0.0057 | ✅ 已完成 |
| No-VA final | 8 | ELPV | 0.8815 ± 0.0023 | 0.872 | +0.0095 | ✅ 已完成 |

### 已知结论

- VisA 和 ELPV 上 2/4/8-shot 均稳定高于原始 InCTRL baseline。
- AITEX 上 2/4-shot 高于 baseline，但 8-shot 低于 baseline。
- VisA 4-shot 低于 2-shot 和 8-shot，且多种子 std 很小，说明不是 few-shot 采样噪声。
- AITEX 8-shot 稳定低于 baseline，说明纺织纹理跨域泛化仍有退化。
- 当前模型实际主要由 InCTRL 残差主干、PQA 和 TA 贡献，VA 分支保留但不参与最终分数。

---

## 实验执行原则

1. 优先保持 `PQA_WEIGHT=0.25` 和 `TEXT_WEIGHT=0.15` 不变，因为当前 VisA 提升主要来自稳定残差主干 + PQA + TA。
2. VA 权重从小到大递增，避免一次性采用 AdaptCLIP 式等权融合导致误报升高。
3. 先跑 `VISUAL_MASK_WEIGHT=0.0`，确认 VA 图像级分支是否有收益；再单独测试 VA mask loss。
4. 每个配置至少先跑 2-shot；若 2-shot 有收益或结论关键，再扩展到 4/8-shot。
5. 每个 smoke 或正式结果必须与 `reports/original_inctrl_baseline.md` 中对应 baseline 比较。

---

## 待跑实验列表

### E1：VA-small，推荐第一优先级

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

命令模板：

```bash
python train_local.py \
  --train_dataset mvtec \
  --test_dataset visa/aitex/elpv \
  --shot 2 \
  --max_epoch 15 \
  --steps_per_epoch 100 \
  --output_dir results/ablation_va_small_2shot_15ep \
  FUSION.IMAGE_WEIGHT 0.32 \
  FUSION.PATCH_WEIGHT 0.23 \
  FUSION.PQA_WEIGHT 0.25 \
  FUSION.TEXT_WEIGHT 0.15 \
  FUSION.VISUAL_WEIGHT 0.05 \
  LOSS.VISUAL_WEIGHT 0.2 \
  LOSS.VISUAL_MASK_WEIGHT 0.0
```

记录：

| Shot | Dataset | AUROC | AUPR | FPR | FNR | vs No-VA | vs InCTRL baseline | 状态 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | VisA | 0.9001 | 0.9137 | TBD | TBD | -0.0029 vs 0.903 | +0.0421 vs 0.858 | ✅ 已完成 | 略低于 No-VA，多数类别仍较高 |
| 2 | AITEX | 0.7974 | 0.5703 | TBD | TBD | -0.0020 vs 0.7994 | +0.0364 vs 0.761 | ✅ 已完成 | 略低于 No-VA，高于原始 InCTRL 2-shot baseline |
| 2 | ELPV | 0.8532 | 0.9300 | TBD | TBD | -0.0069 vs 0.8601 | +0.0142 vs 0.839 | ✅ 已完成 | 低于 No-VA，高于原始 InCTRL 2-shot baseline |

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

E1 初步结论：

- VisA 2-shot AUROC 为 0.9001，低于 No-VA final 的 0.9030，差值约 -0.0029。
- AITEX 2-shot AUROC 为 0.7974，低于 No-VA final 的 0.7994，差值约 -0.0020。
- ELPV 2-shot AUROC 为 0.8532，低于 No-VA final 的 0.8601，差值约 -0.0069。
- 三个目标域平均 AUROC 为 0.8502，低于 No-VA final 的 0.8542，差值约 -0.0039。
- 该结果说明 VA-small 没有带来稳定收益；后续 VA-mid/VA-strong 更适合作为确认“VA 权重增大是否进一步退化或是否存在甜点区间”的补充实验。

判定：

- 若 VisA AUROC ≥ No-VA 2-shot 的 0.903，且 AITEX/ELPV 不显著下降，则扩展到 4/8-shot。
- 若 VisA 提升但 AITEX 明显下降，记录为“VA 对纹理域有负迁移风险”。
- 若三域均下降，说明当前 InCTRL 主干下不适合引入 VA。

---

### E2：VA-mid，中等 VA 权重

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
| 2 | VisA | TBD | TBD | TBD | TBD | TBD | TBD | ⬜ 未跑 |  |
| 2 | AITEX | TBD | TBD | TBD | TBD | TBD | TBD | ⬜ 未跑 |  |
| 2 | ELPV | TBD | TBD | TBD | TBD | TBD | TBD | ⬜ 未跑 |  |

判定：

- 若 VA-mid 优于 VA-small，说明 VA 权重不足可能限制增益。
- 若 VA-mid 低于 VA-small，说明 VA 应保持小权重。

---

### E3：VA-strong，VA 权重上限测试

目标：测试较强 VA 权重是否导致过拟合或误报上升，作为论文中“复杂适配器风险”的证据。

| 配置项 | 数值 |
| --- | ---: |
| `FUSION.IMAGE_WEIGHT` | 0.28 |
| `FUSION.PATCH_WEIGHT` | 0.20 |
| `FUSION.PQA_WEIGHT` | 0.25 |
| `FUSION.TEXT_WEIGHT` | 0.15 |
| `FUSION.VISUAL_WEIGHT` | 0.12 |
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
  --output_dir results/ablation_va_strong_2shot_15ep \
  FUSION.IMAGE_WEIGHT 0.28 \
  FUSION.PATCH_WEIGHT 0.20 \
  FUSION.PQA_WEIGHT 0.25 \
  FUSION.TEXT_WEIGHT 0.15 \
  FUSION.VISUAL_WEIGHT 0.12 \
  LOSS.VISUAL_WEIGHT 0.2 \
  LOSS.VISUAL_MASK_WEIGHT 0.0
```

记录：

| Shot | Dataset | AUROC | AUPR | FPR | FNR | vs No-VA | vs VA-small | 状态 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | VisA | 0.8904 | 0.9061 | TBD | TBD | -0.0126 vs 0.9030 | -0.0097 vs 0.9001 | ✅ 已完成 | 明显低于 No-VA 和 VA-small |
| 2 | AITEX | 0.8005 | 0.5753 | TBD | TBD | +0.0011 vs 0.7994 | +0.0031 vs 0.7974 | ✅ 已完成 | 略高于 No-VA 和 VA-small |
| 2 | ELPV | 0.8574 | 0.9338 | TBD | TBD | -0.0027 vs 0.8601 | +0.0042 vs 0.8532 | ✅ 已完成 | 低于 No-VA，略高于 VA-small |

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

E3 初步结论：

- VA-strong 三域平均 AUROC 为 0.8494，低于 No-VA final 的 0.8542，也低于 VA-small 的 0.8502。
- VisA 从 No-VA 的 0.9030 下降到 0.8904，下降约 -0.0126，是主要退化来源。
- AITEX 从 No-VA 的 0.7994 小幅上升到 0.8005，但增益仅 +0.0011，不足以抵消 VisA 退化。
- ELPV 低于 No-VA，但高于 VA-small，说明较强 VA 权重可能对个别域有轻微帮助，但整体不稳定。
- 当前结果支持“VA 权重越大越容易破坏 VisA 泛化”的判断，VA 不适合成为强分支。

判定：

- 若 strong 比 small/mid 差，说明 VA 适合低权重辅助，不适合主导分数。
- 若 strong 更好，需要进一步跑 4/8-shot 和多种子验证。

---

### E4：VA-mask-small，轻量 VA 像素监督

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
| 2 | VisA | TBD | TBD | TBD | TBD | TBD | ⬜ 未跑 |  |
| 2 | AITEX | TBD | TBD | TBD | TBD | TBD | ⬜ 未跑 |  |
| 2 | ELPV | TBD | TBD | TBD | TBD | TBD | ⬜ 未跑 |  |

判定：

- 若 image AUROC 不变但 pixel map 更清晰，可作为定位增强证据。
- 若 FPR 明显上升，说明 VA local map 对跨域纹理过敏，`VISUAL_MASK_WEIGHT` 应保持 0。

---

### E5：AdaptCLIP-style 等权融合，对照实验

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
| 2 | VisA | TBD | TBD | TBD | TBD | TBD | ⬜ 未跑 |  |
| 2 | AITEX | TBD | TBD | TBD | TBD | TBD | ⬜ 未跑 |  |
| 2 | ELPV | TBD | TBD | TBD | TBD | TBD | ⬜ 未跑 |  |

判定：

- 若等权融合下降，说明 InCTRL-Adapt 不应简单照搬 AdaptCLIP 的 adapter 等权融合。
- 若等权融合提升，说明 VA/TA/PQA 的均衡协同仍有价值，需要进一步做 4/8-shot 和多种子。

---

## 扩展实验：多 shot 与多 seed

若 E1/E2/E3 中任一配置在 2-shot 表现接近或优于 No-VA，则扩展：

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

1. 最佳 VA 配置的 VisA 2/4/8-shot。
2. AITEX 8-shot，因为当前 No-VA 已低于 baseline。
3. 高敏感类别：pcb1、pcb2、pcb3、macaroni2。

---

## 总结果汇总表

| 实验 | VA fusion | VA CE loss | VA mask loss | Shot | VisA AUROC | AITEX AUROC | ELPV AUROC | 平均 AUROC | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| No-VA final | 0.00 | 0.0 | 0.0 | 2 | 0.9030 | 0.7994 | 0.8601 | 0.8542 | 当前最佳 2-shot 已知配置 |
| No-VA final | 0.00 | 0.0 | 0.0 | 4 | 0.8916 | 0.7999 | 0.8705 | 0.8540 | VisA 4-shot 非单调 |
| No-VA final | 0.00 | 0.0 | 0.0 | 8 | 0.9060 | 0.8003 | 0.8815 | 0.8626 | AITEX 8-shot 低于原始 baseline |
| VA-small | 0.05 | 0.2 | 0.0 | 2 | 0.9001 | 0.7974 | 0.8532 | 0.8502 | 三域均略低于 No-VA，未见稳定收益 |
| VA-mid | 0.08 | 0.2 | 0.0 | 2 | TBD | TBD | TBD | TBD | 待验证 |
| VA-strong | 0.12 | 0.2 | 0.0 | 2 | 0.8904 | 0.8005 | 0.8574 | 0.8494 | AITEX 略升，但 VisA 明显下降，整体低于 No-VA 和 VA-small |
| VA-mask-small | 0.05 | 0.2 | 0.05 | 2 | TBD | TBD | TBD | TBD | 待验证 |
| AdaptCLIP-style | 0.20 | 0.5 | 0.0 | 2 | TBD | TBD | TBD | TBD | 待验证 |

---

## 论文可用结论模板

### 若 VA-small 提升

在 InCTRL 残差主干和 PQA 已提供稳定异常证据的情况下，低权重 VA visual-text 分支能够作为辅助语义校准项带来额外提升。但较大 VA 权重或 VA mask loss 可能增加跨域纹理误报，因此本文采用受控融合而非 AdaptCLIP 原版等权融合。

### 若 VA-small 不提升

实验表明，AdaptCLIP 风格 VA 在纯视觉-语言 adapter 框架中有效，但在 InCTRL 残差主干已经提供强局部差异信号时，VA visual-text 分支未带来稳定跨域增益。考虑到 VA local 分支在早期消融中表现出较高误报率，最终模型关闭 VA 融合权重，以保留 InCTRL 残差和 PQA 的稳定性。

### 若 VA-mask 导致退化

VA 的像素图来自 patch-level visual-text similarity，而非专门的分割 decoder。对该分支施加强 mask supervision 会增强源域纹理拟合，但可能削弱跨域泛化。相比之下，PQA segmentation head 更适合作为主要像素级监督分支。

---

## Checklist

- [x] 记录 No-VA final 的 VisA 2/4/8-shot 多种子 AUROC。
- [x] 记录 No-VA final 的 AITEX 8-shot 多种子 AUROC。
- [x] 跑 E1：VA-small 2-shot。
- [x] 补查或补跑 No-VA 2-shot AITEX/ELPV，用于 E1 严格直接对照。
- [ ] 根据 E1 结果决定是否扩展 4/8-shot。
- [ ] 跑 E2：VA-mid 2-shot。
- [x] 跑 E3：VA-strong 2-shot。
- [ ] 若 E1 有潜力，跑 E4：VA-mask-small 2-shot。
- [ ] 跑 E5：AdaptCLIP-style 等权融合对照。
- [ ] 汇总 `test_results.csv` 到本文件表格。
- [ ] 对最佳配置进行多种子复验。
- [ ] 生成 AUROC/AUPR/FPR 对比图。
- [ ] 更新论文实验章节和附录复现说明。
