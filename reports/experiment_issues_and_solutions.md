# 实验挫折与解决方案汇总

> 自动生成于 2026-04-30，基于开发过程中的对话记录。

---

## 1. Few-shot 采样随机性导致结果不可复现

**问题**：对比不同 shot 数的实验结果时，发现 4-shot 有时反而低于 2-shot（VisA），AITEX 8-shot 低于 baseline。怀疑是 few-shot 参考图的随机抽样引入的噪声，但无法验证——采样种子写死了。

**根因**：`IC_dataset_new.py` 中 `rng = np.random.default_rng(42)` 硬编码，无法切换种子重复实验。

**解决方案**：
- 修改 `IC_dataset_new.py` → 新增 `few_shot_seed` 参数
- 修改 `datasets/build.py` / `datasets/loader.py` → 透传 `cfg.FEW_SHOT_SEED`
- 创建 `tools/eval_multi_seed.py` → 批量跑 3 个种子（42, 123, 7），输出 mean±std
- 默认 seed=42 保持向后兼容

---

## 2. 4-shot 性能低于 2-shot（VisA）

**问题**：直觉上 shot 数越多越好，但 VisA 上 4-shot AUROC 持续低于 2-shot。

**多种子验证结果**：

| Shot | AUROC (mean±std) |
|------|-----------------|
| 2-shot | **0.903 ± 0.002** |
| 4-shot | 0.892 ± 0.002 |
| 8-shot | 0.906 ± 0.002 |

**结论**：三个种子下稳定复现，std 极小（~0.002），**排除采样噪声**。这是模型本身在 4-shot 配置下的真实表现下降，可能与 4-shot 训练时的优化路径有关，不是随机性问题。

---

## 3. AITEX 8-shot 低于 baseline

**问题**：唯一低于 InCTRL baseline 的配置——AITEX 8-shot 的 Δ = -0.006。

**多种子验证结果**：

| Seed | AUROC |
|------|-------|
| 42 | 0.7983 |
| 123 | 0.7974 |
| 7 | 0.8051 |
| **mean±std** | **0.800 ± 0.003** |

**结论**：std=0.003，三个种子一致低于 baseline（0.806）。不是采样问题，是模型泛化问题——8-shot 模型在 AITEX 这个特殊纺织品数据集上的跨域迁移有退化。

---

## 4. Seed 敏感度因类别差异巨大

**问题**：不同类别对 few-shot 采样的敏感程度差异很大。

**发现**（来自 seed sensitivity 散点图）：
- **高敏感**：pcb2（std~0.020）、pcb1（std~0.019）、pcb3（std~0.015）、macaroni2（std~0.011）
- **低敏感**：candle、pipe_fryum、chewinggum（std < 0.003）

**规律**：细粒度纹理类别（PCB 电路板、macaroni）对参考图选择更敏感，结构简单的类别几乎不受影响。

---

## 5. InCTRLAdapt 的 few-shot 加载方式混淆

**问题**：项目中有两套 few-shot 机制，容易搞混。

| | 基线 InCTRL | InCTRLAdapt |
|---|---|---|
| few-shot 来源 | `.pt` 文件（预采样） | 数据加载器实时采样 |
| 生成脚本 | `sample_few_shot.py` | 不需要 |
| 消费者 | `test_baseline.py` / `test_all_models.py` | `IC_dataset_new.py` |
| 种子控制 | `PYTHONHASHSEED` + base_seed | `few_shot_seed` 参数 |

**厘清**：`tools/eval_multi_seed.py` 与 `tools/sample_few_shot.py` 完全无关。InCTRLAdapt 在数据加载器内实时从 normal 样本池采样，不读任何 `.pt` 文件。

---

## 6. 像素级分割能力的疑虑

**问题**：训练配置中 `TEXT_MASK_WEIGHT=0.0` 和 `VISUAL_MASK_WEIGHT=0.0`，担心模型没有像素级检测能力。

**厘清**：模型**具备像素级能力**。关键监督来自 PQA seg head（`MASK_WEIGHT=1.0`），text/visual map 只是辅助分支。`final_map` 融合了：

| 分支 | 融合权重 | 有无监督 |
|------|---------|---------|
| `residual_map_up`（patch 残差） | 0.4 | 无监督（天然有效） |
| `pqa_map`（PQA seg head） | 0.4 | **有 mask 监督** ✅ |
| `align_score_map`（多层对齐） | — | 无监督 |
| `text_map` | 0.0（自动跳过） | 未启用 |
| `visual_map` | 0.0（自动跳过） | 未启用 |

模型代码中有保护逻辑：当 `MASK_WEIGHT > 0` 但 `TEXT_MASK_WEIGHT = 0` 时，text_map 和 visual_map 自动不参与 `final_map` 融合。

---

## 总结表

| # | 问题 | 性质 | 状态 |
|---|------|------|------|
| 1 | 种子不可配置 | 工程缺陷 | ✅ 已修复（`few_shot_seed` 参数化） |
| 2 | 4-shot < 2-shot (VisA) | 模型真实行为 | ✅ 已确认非噪声（3 seeds 验证） |
| 3 | AITEX 8-shot < baseline | 跨域泛化退化 | ✅ 已确认非噪声（3 seeds 验证） |
| 4 | 类别间敏感度差异 | 数据特性 | ✅ 已量化（散点图） |
| 5 | 两套 few-shot 机制混淆 | 架构理解 | ✅ 已厘清 |
| 6 | 像素级能力疑虑 | 配置误解 | ✅ 已澄清（PQA seg head 有监督） |

---

## 相关产物

| 文件 | 用途 |
|------|------|
| `tools/eval_multi_seed.py` | 多种子批量评估脚本 |
| `tools/plot_multi_seed.py` | 多种子结果可视化（6 张图） |
| `tools/visualize_anomaly_map.py` | 像素级异常热力图可视化 |
| `reports/multi_seed_figures/` | 多种子可视化图表输出目录 |
| `results/multi_seed/` | 多种子评估 CSV/JSON 结果 |
