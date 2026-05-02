# No-VA 交替训练调度分析

## 背景

No-VA final 的配置中，VA 分支不参与最终分数和监督：

| 配置项 | 数值 |
| --- | ---: |
| `FUSION.VISUAL_WEIGHT` | 0.0 |
| `LOSS.VISUAL_WEIGHT` | 0.0 |
| `LOSS.VISUAL_MASK_WEIGHT` | 0.0 |

但旧版训练逻辑仍然启用交替训练，只要模型同时存在 visual-side 参数和 TA 文本参数。一次改成 `single` 单阶段训练后，No-VA rerun 从旧记录的三域平均 AUROC `0.8542` 下降到 `0.8334` 左右，说明高分 No-VA final 不只是分数融合配置，也依赖交替训练调度。

## 源码路径

### 1. 交替开关

`engine_IC.py` 中，交替训练由 `_should_use_alternating_training(...)` 决定：

```python
return (
    _has_parameter_list(model, "get_visual_parameters")
    and _has_parameter_list(model, "get_text_parameters")
)
```

也就是说，它不再看 VA 权重是否为 0。只要 visual-side 和 text-side 参数都存在，就保留原始交替调度。

### 2. 两组参数分别是什么

`open_clip/inctrl_adapt.py` 中，visual-side 参数包括：

```python
if self.use_visual_adapter:
    params.extend(self.visual_adapter.parameters())
params.extend(self.image_head.parameters())
if self.use_pqa:
    params.extend(self.pq_adapter.parameters())
if self.patch_text_projection is not None:
    params.extend(self.patch_text_projection.parameters())
```

text-side 参数只包括 TA 文本分支：

```python
return list(self.text_branch.parameters())
```

因此在 No-VA final 中，虽然 VA 不参与 score/loss，但 `visual` phase 实际训练的是：

- `image_head`
- `pq_adapter`
- `patch_text_projection`
- `visual_adapter` 参数会在 optimizer 里，但 VA branch inactive 时基本没有有效梯度

`text` phase 训练的是：

- `text_branch` / TA prompt learner 参数

### 3. phase 如何冻结参数

`set_train_phase(...)` 根据 phase 设置 `requires_grad`：

| phase | visual-side 参数 | TA 参数 |
| --- | --- | --- |
| `visual` | 可训练 | 冻结 |
| `text` | 冻结 | 可训练 |
| `single` | 可训练 | 可训练 |

训练循环中每个 epoch 交替：

```python
phase = "visual" if cur_epoch % 2 == 0 else "text"
```

## 为什么它像训练调度正则化

### 1. 它不是 VA 正则化

VA 分支是否参与 forward 由 `_visual_branch_is_active()` 控制：

```python
return fusion_visual_w > 0.0 or loss_visual_w > 0.0 or loss_visual_mask_w > 0.0
```

在 No-VA final 中这三个权重全为 0，所以 VA visual-text branch 返回 zero outputs，不进入 final score，也没有 VA loss。

因此 No-VA final 的收益不是来自 VA 分支提供额外异常证据。

### 2. 它分离了两类会竞争 final loss 的参数

最终图像级分数同时融合：

```python
image_logit, patch_logit, pqa_logit, text_logit
```

loss 中始终有 `final` BCE；同时还有 image、PQA、mask 等辅助监督。默认 `LOSS.TEXT_WEIGHT=0.0`，所以 TA 主要通过 final loss 间接学习。

如果使用 `single` 单阶段训练，TA、PQA、image head、patch projection 会同时沿着 final loss 更新。这样容易出现两个问题：

1. TA 文本分支和 reference-based residual/PQA 分支同时抢同一个 final score 的解释权。
2. `patch_text_projection` 既影响 TA map/logit 输入，又属于 visual-side 参数；单阶段训练会让桥接投影和 TA prompt 同时移动，优化目标更不稳定。

交替训练把这个过程拆成近似 coordinate descent：

- visual phase：TA 固定，更新 image/PQA/projection，让 reference-based 证据先适配当前 TA。
- text phase：image/PQA/projection 固定，更新 TA，让文本分支适配当前 residual/PQA 证据。

这种分阶段更新降低了分支间的同步漂移，效果上类似训练调度正则化。

### 3. 现有实验支持这个解释

旧 No-VA final 2-shot：

| Dataset | AUROC |
| --- | ---: |
| VisA | 0.9030 |
| AITEX | 0.7994 |
| ELPV | 0.8601 |
| 平均 | 0.8542 |

改成 `single` 后重新跑 No-VA：

| Dataset | AUROC |
| --- | ---: |
| VisA | 0.8800 |
| AITEX | 0.7760 |
| ELPV | 0.8442 |
| 平均 | 0.8334 |

差值：

| Dataset | Delta |
| --- | ---: |
| VisA | -0.0230 |
| AITEX | -0.0234 |
| ELPV | -0.0159 |
| 平均 | -0.0208 |

这个下降幅度和 A1-A4 局部权重消融的退化同量级，说明 `single` 不是一个等价训练实现，而是改变了关键训练机制。

## 对实验命名的修正

后续应把当前最佳配置命名为：

```text
No-VA final + alternating visual-side/TA schedule
```

而不是只写 No-VA final。因为 “No-VA” 只说明 score/loss 不使用 VA，不能说明训练调度。

## 复现实验检查项

跑 No-VA final 时，建议检查：

```bash
head -n 8 results/ablation_no_va_final_2shot_15ep/train_history.csv
```

期望看到 `phase` 列为 `visual/text/visual/text...`，而不是 `single`。

如果看到 `single`，这不是旧 No-VA final 的复现，而是新的单阶段训练消融。

## 结论

在当前代码和实验结果下，应该保留原始交替训练调度。即使 VA 分支关闭，交替学习仍然通过分离 visual-side residual/PQA/projection 更新与 TA prompt 更新，提供了有效的训练调度正则化。

因此当前最佳主线应保持：

| 项 | 设置 |
| --- | --- |
| VA score/loss | 关闭 |
| visual-side/TA alternating schedule | 保留 |
| `LOSS.TEXT_WEIGHT` | 0.0 |
| `LOSS.MASK_WEIGHT` | 1.0 |
| fusion weights | `0.35 / 0.25 / 0.25 / 0.15 / 0.0` |
