# InCTRL x AdaptCLIP x InCTRL PQA Fused 经验文档

## 1. 文档目的

这份文档给后续 Agent 用，目标是快速搞清楚三件事：

1. 原版 InCTRL 在做什么。
2. 仓库内 AdaptCLIP 参考实现的 PQA 在做什么。
3. 当前主线 inctrl_pqa_fused 实际保留了什么、改了什么、风险在哪里。

本文严格基于当前代码，不依赖外部假设。

---

## 2. 快速定位

### 2.1 关键文件

- 原版 InCTRL 主体：open_clip/model.py
- AdaptCLIP 参考实现：open_clip/adaptclip.py
- 当前主线融合模型：open_clip/inctrl_pqa_fused.py
- 当前损失实现：open_clip/inctrl_pqa_losses.py
- 训练引擎：engine_IC.py

### 2.2 推荐阅读顺序

1. 先看 open_clip/model.py 的 InCTRL forward，理解原始 pipeline。
2. 再看 open_clip/adaptclip.py 的 PQAdapter，理解 PQA 参考设计。
3. 最后看 open_clip/inctrl_pqa_fused.py 的 InCTRLPQA.forward，对照差异。
4. 再看 open_clip/inctrl_pqa_losses.py，确认哪些分支实际被监督。

---

## 3. 原版 InCTRL 核心逻辑

## 3.1 输入与分支

原版 InCTRL forward 在 open_clip/model.py 中，核心分成三路：

1. patch 残差图分支（few-shot 最近邻）。
2. 图像参考分支（token 差分）。
3. 文本相似度分支（normal/anomaly 文本模板）。

## 3.2 关键实现点

- patch 最近邻残差：
  - s = 0.5 * (1 - tmp @ tmp_n.T) 后取最小值。
- patch 图聚合：
  - 对层取均值得到 patch_ref_map，再取 max 得到 fg_score。
- 文本分支：
  - 使用 get_texts 生成 normal/anomaly 模板，CLIP 余弦后 softmax 取 anomaly 概率。
- 图像分支：
  - token_ref = mean(Adapter(prompt)) - Adapter(query)，再经 diff_head_ref。
- 最终分数：
  - final_score = (hl_score + fg_score) / 2。

---

## 4. AdaptCLIP 参考中 PQA 的关键机制

仓库内 open_clip/adaptclip.py 的 PQAdapter 体现了参考 PQA 思路：

1. 对 query patch 与 prompt patch 做对齐（按相似度找对应）。
2. 构造融合特征：
   - context=True 时，fusion_patch_feat = query + abs(query - aligned_prompt)。
3. local head 输出分割概率图。
4. global head 对融合特征做池化后输出图像级分类 logit。

参考实现的 global 池化是 mean + topk mean 的固定配方。

---

## 5. inctrl_pqa_fused 当前实现结论

## 5.1 总体定位

当前实现不是“完整复刻 AdaptCLIP 全体系”，而是“在 InCTRL 主干上接入 PQA 核心能力并做工程化融合”。

## 5.2 已接入内容

1. patch 对齐与残差计算。
2. PQA 上下文融合（query + beta * abs(query - aligned_prompt)）。
3. local segmentation head（2 通道 logit + softmax）。
4. global image head（MLP 输出 2 类差值 logit）。
5. 与 InCTRL 残差图做 patch map 融合。
6. final logit 层面再融合 base 与 pqa 两路得分。

## 5.3 未完整接入的 AdaptCLIP 组件

1. 未接入 AdaptCLIP 的 TextualAdapter 训练式文本分支。
2. 未接入 AdaptCLIP 的 VisualAdapter 路径。
3. global pooling 未采用 AdaptCLIP 参考的 mean + topk mean 固定策略，而是改成了 GAP/GMP 可学习加权。

---

## 6. 异常分数公式对照

## 6.1 原版 InCTRL

记：

- d_l,i：第 l 层第 i 个 patch 的最近邻残差。
- M_i：跨层聚合后的 patch 残差。
- f：patch 前景分数（max）。
- t：文本 anomaly 分数。
- r：图像参考分支分数。
- h：holistic 分数。

公式：

- d_l,i = min_j (1 - <q_l,i, p_l,j>) / 2
- M_i = mean_l(d_l,i)
- f = max_i(M_i)
- h = DiffHead(M + t + r)
- s_final = (h + f) / 2

## 6.2 当前 inctrl_pqa_fused

记：

- M_raw：原始 InCTRL 风格残差图聚合。
- M_base：当前 PQAdapter 输出的 residual_maps 聚合。
- M_pqa：PQA patch 分数图聚合。
- M_hybrid：base 与 pqa 的可学习融合图。
- l_h：holistic_logit。
- l_m：max_patch_logit。
- l_base：0.5 * (l_h + l_m)。
- l_pqa：PQA 图像级 logit。

公式：

- M_hybrid = w0 * M_base + w1 * M_pqa
- l_h = HolisticHead(M_hybrid + image_score + text_score)
- l_base = 0.5 * (l_h + logit(max(M_hybrid)))
- l_final = a0 * l_base + a1 * l_pqa
- s_final = sigmoid(l_final)

这意味着当前版本比原版多了一层图像级融合，PQA 从辅助信号变成最终分数直接参与项。

---

## 7. 当前训练监督的真实情况

以 open_clip/inctrl_pqa_losses.py 为准：

1. final_logit 有监督（必需）。
2. image_logit 可监督（由 image_loss_weight 控制）。
3. pqa_logit 可监督（由 pqa_loss_weight 控制）。
4. pqa_local_logits 只有在提供 masks 时才有分割监督。

注意：无 masks 的数据上，PQA local 分支没有直接像素级监督。

---

## 8. 已确认的常见坑

## 8.1 导入路径坑（已修复）

如果 open_clip/__init__.py 还在导入已不存在的 open_clip.inctrl_pqa，会在 pytest 收集阶段直接报 ModuleNotFoundError。

当前应导出 open_clip.inctrl_pqa_fused.InCTRLPQA。

## 8.2 配置语义坑

GLOBAL_TOPK 参数仍被读取，但 PQAGlobalHead 实际前向走 GAP/GMP 可学习融合，不走 topk 池化。

结果是“配置项名”与“真实行为”可能不一致，后续实验记录要明确写清。

## 8.3 缓存批次语义坑

prompt_feature_cache 会扩展到整个 batch 使用。
如果 batch 混了不同类别，且 cache 不是按类别构建，语义上会错配。

## 8.4 接口兼容坑

原版 InCTRL forward 返回 tuple，当前主线返回 dict（默认）。
老调用方如果写死了解包方式会出问题。

---

## 9. 给后续 Agent 的执行建议

## 9.1 分析/改动前最小检查

1. 确认 open_clip/__init__.py 导出的是 inctrl_pqa_fused。
2. 确认训练入口是否走 engine_IC.py + inctrl_pqa_fused。
3. 确认 loss 配置里 pqa_loss_weight、image_loss_weight、mask_loss_weight。

## 9.2 回归测试建议（dexter 环境）

优先命令：

```bash
C:/Users/dex/miniconda3/Scripts/conda.exe run -n dexter --no-capture-output python -m pytest tests/test_inctrl_pqa_fused.py -q
```

如改动了损失：

```bash
C:/Users/dex/miniconda3/Scripts/conda.exe run -n dexter --no-capture-output python -m pytest tests/test_inctrl_pqa_losses.py -q
```

如改动较大，建议补：

```bash
C:/Users/dex/miniconda3/Scripts/conda.exe run -n dexter --no-capture-output python -m pytest tests -q
```

---

## 10. 一句话总结

当前 inctrl_pqa_fused 是“保留 InCTRL 核心几何与文本先验，接入 PQA 的对齐与分割/图像分支，并在 patch 与 logit 两级做可学习融合”的工程化主线，不是 AdaptCLIP 全量路径直译版。
