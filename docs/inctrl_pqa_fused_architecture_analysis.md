# InCTRLPQA 架构分析报告

> 文件：`open_clip/inctrl_pqa_fused.py`  
> 参照原文：`open_clip/model.py`（InCTRL）、`open_clip/adaptclip.py`（AdaptCLIP PQA）  
> 生成时间：2026-04-20  

---

## 1. 背景：两个原始架构的设计哲学

### 1.1 InCTRL（`model.py :: InCTRL`）

InCTRL 是一个 **few-shot + zero-shot 混合**的异常检测框架，核心思路是：

- 用冻结 CLIP 视觉塔（`VisionTransformer_Mul`）提取 **多层 patch 特征**（layers 7, 9, 11）
- 通过 patch 级别的余弦距离残差定位局部异常
- 同时利用 CLS token 的全局差异做图像级判断
- 用 WinCLIP 风格的文本原型（normal/anomaly 描述集合）做零样本文本分支
- 三路信号（patch 残差 + 图像残差 + 文本）线性融合成 holistic map，最终用 `TransformerBasicHead` 回归最终分数

**最终融合公式（原始 InCTRL）：**

$$s_\text{final} = \frac{s_\text{holistic} + s_\text{patch-max}}{2}$$

这是硬编码的等权平均，没有任何可学习的融合。

### 1.2 AdaptCLIP PQA（`adaptclip.py :: PQAdapter`）

AdaptCLIP 的 PQA 分支将异常检测转化为 **prompt-query 对比**问题：

- 对每个 query patch，找最近邻 prompt patch，然后构造 **上下文特征图**：`f_ctx = query + |query - aligned_prompt|`
- 用 Conv 解码器（`local_adapter`）生成 **像素级分割图**（2-class softmax map）
- 用 MLP（`global_adapter`）对全局 `(GAP + top-10 mean) / 2` 池化特征生成 **图像级二分类 logit**
- 多层特征分别计算后通过平均或逐层加权合并

**全局池化公式（原始 AdaptCLIP，硬编码）：**

$$\mathbf{f}_\text{global}^{(l)} = \frac{1}{2} \cdot \text{GAP}(\mathbf{C}^{(l)}) + \frac{1}{2} \cdot \text{TopK-Mean}_{k=10}(\mathbf{C}^{(l)})$$

---

## 2. InCTRLPQA 的融合策略

### 2.1 模块总览

```
CLIP ViT (frozen)
├── CLS token  ──────────────────────────────────── [image branch]
└── Patch tokens [layers l₁, l₂, l₃]
    │
    ├── patch_projection (Linear or Identity)
    │
    ├── PQAdapter (per layer)
    │   ├── _match_prompt_patches()  ──── InCTRL residual maps r^(l)
    │   ├── _build_context_map()     ──── AdaptCLIP context map C^(l)
    │   ├── PQAConvLocalHead^(l)     ──── PQA local segmentation logits
    │   └── PQAGlobalHead^(l)        ──── PQA global image logit ℓ_pqa^(l)
    │
    ├── layer_weights_logits   → pqa_layer_weights α  (for PQA signals)
    ├── patch_layer_weights_logits → patch_layer_weights γ (for residual maps)
    │
    ├── patch_map_fusion_logits → fused patch map M_fused
    │
    ├── _reduce_patch_map_to_logit() → patch_logit
    ├── ImageResidualHead            → image_logit
    ├── (frozen text tower)          → text_logit
    │
    └── ScalarFusionHead (4-way decision_head)
            [patch_logit, pqa_logit, image_logit, text_logit]
                        ↓
                    final_logit → final_score
```

---

## 3. 各分支数学公式

### 记号

| 符号 | 含义 |
|------|------|
| $B$ | batch size |
| $L$ | 选用的 ViT 层数（默认 3，层 7/9/11） |
| $N$ | 每张图 patch 数（$N = (H/p)^2$，如 240px/16px = 225） |
| $D$ | embedding 维度（如 ViT-L/14 = 768） |
| $K$ | few-shot prompt 图数量 |
| $\hat{\mathbf{v}}$ | 向量 $\mathbf{v}$ 的 L2 归一化 |

---

### 3.1 视觉特征提取

$$\mathbf{g}_q \in \mathbb{R}^{B \times D}, \quad \left\{\mathbf{P}_q^{(l)}\right\}_{l=1}^L \in \mathbb{R}^{B \times N \times D}$$

$$\mathbf{g}_p \in \mathbb{R}^{B \times K \times D}, \quad \left\{\mathbf{P}_p^{(l)}\right\}_{l=1}^L \in \mathbb{R}^{B \times K \times N \times D}$$

CLS token 作为全局特征，各层非 CLS patch token（去掉 position 0）作为局部特征。

---

### 3.2 Patch 对齐（InCTRL 最近邻匹配）

对每层 $l$，将所有 $K$ 个 prompt 的 patch 展平为 $\mathbf{P}_p^{(l)} \in \mathbb{R}^{B \times (KN) \times D}$，对每个 query patch $n$ 找最近邻：

$$m^*_{n} = \arg\max_{m} \cos\!\left(\mathbf{q}_{n}^{(l)},\, \mathbf{p}_{m}^{(l)}\right)$$

**InCTRL 残差分数**（越大越异常，取值 $[0, 1]$）：

$$r_n^{(l)} = \frac{1}{2}\left(1 - \max_m \cos\!\left(\hat{\mathbf{q}}_n^{(l)},\, \hat{\mathbf{p}}_m^{(l)}\right)\right)$$

对应代码：`_match_prompt_patches()` → `residual = 0.5 * (1.0 - max_cosine)`

---

### 3.3 上下文特征图（AdaptCLIP context map）

将对齐的 prompt patch 与 query patch 融合为上下文特征：

$$\mathbf{c}_n^{(l)} = \hat{\mathbf{q}}_n^{(l)} + \beta \left|\hat{\mathbf{q}}_n^{(l)} - \hat{\mathbf{p}}_{m^*_n}^{(l)}\right|$$

重整为 $\mathbf{C}^{(l)} \in \mathbb{R}^{B \times D \times \sqrt{N} \times \sqrt{N}}$，通过 **BatchNorm2d** 归一化。

> $\beta$ 为超参数（默认 1.0），控制差异项的权重。当 $\beta = 0$ 退化为纯 query 特征；$\beta = 1$ 对异常区域有最强响应。

对应代码：`_build_context_map()` → `context_feat = query + beta * abs(query - aligned_prompt)`

---

### 3.4 PQA 局部分割头

对每层 $l$：

$$\tilde{\mathbf{S}}^{(l)} = \text{ConvDecoder}^{(l)}\!\left(\mathbf{C}^{(l)}\right) \in \mathbb{R}^{B \times 2 \times H \times W}$$

（2× ConvTranspose2d 上采样到 image_size $\times$ image_size）

**像素级分割概率图**（可用于后处理或 Mask loss）：

$$\mathbf{S}_\text{local}^{(l)} = \text{softmax}\!\left(\tilde{\mathbf{S}}^{(l)},\, \text{dim}=1\right) \in [0,1]^{B \times 2 \times H \times W}$$

**下采样回 patch 粒度的分数图**（用于加权融合）：

$$s_{\text{pqa},n}^{(l)} = \text{softmax}\!\left(\text{AvgPool}_{N}\!\left(\tilde{\mathbf{S}}^{(l)}\right)\right)_{n,1} \in [0,1]^{B \times N}$$

对应代码：`_compute_local_outputs()` → `patch_scores_from_pool`

---

### 3.5 PQA 全局头（可学习 GAP/GMP 融合）

将上下文特征图展平为 $\mathbf{f}^{(l)} \in \mathbb{R}^{B \times N \times D}$，然后：

$$\text{GAP}^{(l)} = \frac{1}{N}\sum_n \mathbf{f}_n^{(l)}, \quad \text{GMP}^{(l)} = \max_n \mathbf{f}_n^{(l)}$$

可学习池化权重（区别于 AdaptCLIP 的硬编码 0.5/0.5）：

$$w_\text{gap},\, w_\text{gmp} = \text{softmax}(\boldsymbol{\theta}_\text{pool}) \quad \in \mathbb{R}^2$$

$$\mathbf{p}_\text{pool}^{(l)} = w_\text{gap} \cdot \text{GAP}^{(l)} + w_\text{gmp} \cdot \text{GMP}^{(l)}$$

通过 `MLPAdapter`（Linear→LayerNorm→ReLU→Linear→2 output）得到 2-class logit，取差值：

$$\ell_\text{pqa}^{(l)} = \text{MLP}^{(l)}\!\left(\mathbf{p}_\text{pool}^{(l)}\right)_{[1]} - \text{MLP}^{(l)}\!\left(\mathbf{p}_\text{pool}^{(l)}\right)_{[0]}$$

对应代码：`PQAGlobalHead.forward()`

---

### 3.6 层间加权聚合（解耦双权重）

> 关键设计：PQA 信号（patch map + global logit）与 InCTRL 残差 map 使用**独立**的层权重向量，不共享。

$$\boldsymbol{\alpha} = \text{softmax}(\boldsymbol{\theta}_\text{pqa-layer}) \in \mathbb{R}^L \quad \text{（PQA 层权重）}$$

$$\boldsymbol{\gamma} = \text{softmax}(\boldsymbol{\theta}_\text{patch-layer}) \in \mathbb{R}^L \quad \text{（InCTRL 残差层权重）}$$

**InCTRL 残差 patch map**（层加权平均，取值 $[0,1]$）：

$$\mathbf{R}_\text{inctrl} = \sum_{l=1}^L \gamma_l \cdot \mathbf{r}^{(l)} \in [0,1]^{B \times N}$$

**PQA patch map**（层加权平均）：

$$\mathbf{S}_\text{pqa} = \sum_{l=1}^L \alpha_l \cdot \mathbf{s}_\text{pqa}^{(l)} \in [0,1]^{B \times N}$$

**PQA global logit**：

$$\ell_\text{pqa} = \sum_{l=1}^L \alpha_l \cdot \ell_\text{pqa}^{(l)} \in \mathbb{R}^B$$

---

### 3.7 Patch Map 融合

可学习的双路 softmax 权重：

$$w_\text{inctrl},\, w_\text{pqa} = \text{softmax}(\boldsymbol{\theta}_\text{fuse}) \in \mathbb{R}^2$$

$$\mathbf{M}_\text{fused} = w_\text{inctrl} \cdot \mathbf{R}_\text{inctrl} + w_\text{pqa} \cdot \mathbf{S}_\text{pqa} \in [0,1]^{B \times N}$$

---

### 3.8 Patch Score → Logit 规约（`_reduce_patch_map_to_logit`）

使用 max 与 top-k mean 的等权组合，平衡局部极值灵敏度与整体异常面积：

$$s_\text{patch} = \frac{1}{2} \max_n M_{\text{fused},n} + \frac{1}{2} \cdot \frac{1}{k} \sum_{j=1}^k M_{\text{fused},\sigma(j)}$$

其中 $k = \min(10, N)$，$\sigma$ 为 patch 分数的降序排列。

转为无界 logit（逆 sigmoid）：

$$\ell_\text{patch} = \log\frac{s_\text{patch}}{1 - s_\text{patch}} \quad \text{(clamp 到 } [10^{-6},\, 1-10^{-6}]\text{)}$$

---

### 3.9 图像级残差分支（InCTRL Adapter + ImageResidualHead）

取所有 prompt 图的 CLS token 均值作为原型：

$$\bar{\mathbf{g}}_p = \text{normalize}\!\left(\frac{1}{K}\sum_{k=1}^K \mathbf{g}_{p,k}\right)$$

计算绝对差异向量：

$$\mathbf{e}_\text{img} = \left|\bar{\mathbf{g}}_p - \hat{\mathbf{g}}_q\right| \in \mathbb{R}^{B \times D}$$

通过 ResMLP-style Adapter（bottleneck ratio=4）增强：

$$\mathbf{e}'_\text{img} = \mathbf{e}_\text{img} + \text{ReLU}\!\left(\mathbf{W}_2\, \text{ReLU}\!\left(\mathbf{W}_1 \mathbf{e}_\text{img}\right)\right) \quad \mathbf{W}_1 \in \mathbb{R}^{D/4 \times D},\, \mathbf{W}_2 \in \mathbb{R}^{D \times D/4}$$

最终通过 LayerNorm + 2-layer MLP（GELU）得到标量 logit：

$$\ell_\text{image} = \text{Linear}\!\left(\text{GELU}\!\left(\text{Linear}\!\left(\text{LN}(\mathbf{e}'_\text{img})\right)\right)\right) \in \mathbb{R}^B$$

---

### 3.10 文本分支（WinCLIP 风格，冻结）

分别构造 normal / anomaly 文本原型（在多个描述模板下平均后归一化）：

$$\hat{\mathbf{t}}_+ = \text{normalize}\!\left(\frac{1}{|T_+|}\sum_{t \in T_+} \text{CLIP-Text}(t)\right), \quad \hat{\mathbf{t}}_- = \text{normalize}\!\left(\frac{1}{|T_-|}\sum_{t \in T_-} \text{CLIP-Text}(t)\right)$$

（$|T_+| = 7 \times 22 = 154$ 个 normal 描述，$|T_-| = 4 \times 22 = 88$ 个 anomaly 描述）

CLIP 温度缩放余弦相似度：

$$\ell_\text{normal} = 100 \cdot \hat{\mathbf{g}}_q \cdot \hat{\mathbf{t}}_+, \quad \ell_\text{anomaly} = 100 \cdot \hat{\mathbf{g}}_q \cdot \hat{\mathbf{t}}_-$$

文本分支 logit（log-odds 形式）：

$$\ell_\text{text} = \ell_\text{anomaly} - \ell_\text{normal}$$

文本分支分数（用于评估指标）：

$$s_\text{text} = \text{softmax}\!\left([\ell_\text{normal},\, \ell_\text{anomaly}]\right)_1$$

---

### 3.11 四路决策头（`ScalarFusionHead`，可学习融合）

将四路 logit 拼接为向量：

$$\boldsymbol{\ell} = [\ell_\text{patch},\, \ell_\text{pqa},\, \ell_\text{image},\, \ell_\text{text}] \in \mathbb{R}^{B \times 4}$$

通过 LayerNorm + Linear(4, H) + GELU + Linear(H, 1)（$H = \max(\text{hidden\_dim}/2, 16)$）：

$$\ell_\text{final} = \text{Linear}\!\left(\text{GELU}\!\left(\text{Linear}\!\left(\text{LN}(\boldsymbol{\ell})\right)\right)\right) \in \mathbb{R}^B$$

$$\boxed{s_\text{final} = \sigma(\ell_\text{final})}$$

---

## 4. 与原始架构的对比

### 4.1 InCTRL 对应关系

| InCTRL 组件 | InCTRLPQA 对应 | 变化 |
|---|---|---|
| `Adapter(CLS_q)` → `Adapter(CLS_p)` → diff → `diff_head_ref` | `ImageResidualHead(abs(norm_p - norm_q))` | 由适配后相减改为先差再适配；增加 ResMLP adapter 残差 |
| patch 余弦残差 `r_n = 0.5(1 - max_cosine)` | `_match_prompt_patches()` → `residual_maps` | **完全保留** |
| 三层平均 `mean_l(Fp_map)` | `Σ γ_l × r^(l)` | 改为**可学习**层权重加权 |
| holistic map = text + img_ref + patch_map | 删除 | 三路信号各自为独立 logit 输入 decision_head |
| `diff_head(holistic_map)` | `decision_head([ℓ_patch, ℓ_pqa, ℓ_img, ℓ_text])` | 4-way 可学习融合，取代硬编码 |
| `(hl_score + fg_score) / 2` | `σ(decision_head(...))` | **完全替换** |

### 4.2 AdaptCLIP PQA 对应关系

| AdaptCLIP 组件 | InCTRLPQA 对应 | 变化 |
|---|---|---|
| `min(1 - cosine)` patch 对齐 | `max(cosine)` → `r = 0.5(1 - max_cos)` | 数学等价（min dist = max sim），显式写成残差形式 |
| `fusion = q + \|q - p\|` | `_build_context_map(): q + β\|q - p\|` | 增加 $\beta$ 超参，可控差异权重 |
| `local_adapter` (conv) | `PQAConvLocalHead` (conv) | **直接对应**，架构完全相同 |
| `global_adapter` (BN-MLP) | `PQAGlobalHead` (LN-MLP + learnable GAP/GMP) | BN→LN（小 batch safe）；GAP/GMP 权重从固定 0.5/0.5 → **可学习** |
| 逐层独立 `global_adapter` | per-layer `PQAGlobalHead` | **完全保留**独立参数设计 |
| `(GAP + top10) / 2` | `w_gap × GAP + w_gmp × GMP` | top-k mean 改为 GMP，权重变为可学习 |
| 无层间融合 | `Σ α_l × (·)` | 新增**可学习**层权重 |

---

## 5. 融合质量评估

### 5.1 已充分融合的设计点 ✅

1. **多层 patch 特征**：继承 InCTRL 对 layers 7/9/11 的选取，并升级为可学习双权重（PQA 路和残差路各自独立）。
2. **patch 对齐机制**：InCTRL 的 nearest-neighbor cosine matching 完整保留，作为 `residual_maps` 输出。
3. **AdaptCLIP context map**：`query + β × |query - aligned|` 公式完整继承，用于驱动 PQA 分割头和全局头。
4. **局部分割头**：与 AdaptCLIP `local_adapter` 架构完全对应（Conv→BN→ReLU→2×Upsample）。
5. **全局池化升级**：将 AdaptCLIP 的硬编码 `(GAP + top10) / 2` 升级为可学习 `w_gap × GAP + w_gmp × GMP`，消除超参假设。
6. **文本分支保留**：InCTRL 的 WinCLIP 文本先验完整保留，包括 7 种 normal 状态 × 22 个模板 + 4 种 anomaly 状态 × 22 个模板。
7. **图像残差分支增强**：在 InCTRL 的 `|CLS_ref - CLS_query|` 基础上增加 ResMLP adapter，增强特征判别力。
8. **Patch Map 融合**：双路 softmax 权重（InCTRL 残差 vs PQA patch map）端到端可学习。
9. **Patch Score 规约**：`0.5 × max + 0.5 × top-k mean`（k=10）公式在极值敏感性与鲁棒性之间取得平衡。
10. **四路决策头**：取代 InCTRL 的硬编码等权平均，实现数据驱动的多路融合。

### 5.2 设计权衡与已知缺口 ⚠️

**① Holistic Map 的空间交互被打破**

InCTRL 原始设计将 `text_score + img_ref_score + patch_ref_map` 在空间维度相加，产生一张 **跨模态感知的 patch-level anomaly map**，再用 `TransformerBasicHead` 回归。这一设计允许文本和图像分支为每个 patch 局部补充上下文信号。

InCTRLPQA 将三者解耦为独立标量 logit，空间交互消失。**影响**：对于纹理异常明显但全局图像相似度不变的样本（如划痕），文本分支贡献的语义信息无法在 patch 级别增强定位。

**② 图像分支的 Adapter 顺序差异**

原始 InCTRL：`Adapter(CLS_q)` 和 `Adapter(CLS_p)` 在各自空间中适配后再相减，使差异计算在"适配空间"中更有意义。

InCTRLPQA：先计算 `|norm_p - norm_q|`，再在差异向量上施加 adapter。数学上这是两个不同的非线性变换路径。当 adapter 权重较大时，后者可能对"差异方向"建模不够精准。

**③ GMP ≠ Top-K Mean**

AdaptCLIP 的 `top10_mean` 是对前 10 个 spatial 位置的均值，具有一定鲁棒性。GMP（全局最大池化）退化为单点最大值，受噪声影响更大。在小批量或噪声较大的 patch 特征上，可学习权重可能收敛到 GMP 权重极小的退化解。

> 建议：考虑将 `PQAGlobalHead` 中的 GMP 替换为 `top-k mean`（k 可学习或固定为 10），与 `_reduce_patch_map_to_logit` 的设计保持一致。

**④ Beta 参数固定**

`_build_context_map()` 中的 $\beta$（控制 `|query - aligned|` 差异权重）目前是固定超参（默认 1.0）。若 $\beta$ 可学习（per-layer 或全局），模型可以在不同层上自适应地调整"差异敏感度"。

**⑤ 文本分支缺少可学习 prompt**

AdaptCLIP 设计了 `TextualAdapter` 用于 learnable prompt 上下文，当前 InCTRLPQA 的文本分支仍使用纯静态 WinCLIP 模板，在 fine-tuning 阶段无法自适应调整文本先验。在类别偏移（domain shift）场景下可能限制泛化能力。

**⑥ Layer Weights 的命名与语义略有歧义**

- `layer_weights_logits` → `pqa_layer_weights` $\boldsymbol{\alpha}$（驱动 PQA patch map 和 global logit）
- `patch_layer_weights_logits` → `patch_layer_weights` $\boldsymbol{\gamma}$（驱动 InCTRL 残差 map）

前者命名为 `layer_weights`，后者命名为 `patch_layer_weights`，实际含义相反（PQA 是 patch 级别的，InCTRL 也是 patch 级别的）。建议重命名为 `pqa_layer_weight_logits` / `residual_layer_weight_logits`。

---

## 6. 可训练参数汇总

| 参数组 | 模块 | 参数量估算（ViT-L/14, D=768, H=256） |
|---|---|---|
| Patch 投影 | `patch_projection` | 0（Identity 或 Linear D→D） |
| 图像分支 | `ImageResidualHead` | adapter: $2 \times 768 \times 192 ≈ 295$K；net: $768 \times 256 + 256 + 256 ≈ 197$K |
| 四路决策头 | `ScalarFusionHead(4, 128)` | $4 \times 128 + 128 + 128 ≈ 640$ 参数（极轻量） |
| PQA 适配器 | `PQAdapter`（3 层） | 局部头 ×3 + 全局头 ×3 + BN ×3 |
| 层权重 | `layer_weights_logits` | $L = 3$ 个标量 |
| 残差层权重 | `patch_layer_weights_logits` | $L = 3$ 个标量 |
| 融合权重 | `patch_map_fusion_logits` | 2 个标量 |

CLIP 视觉塔（~300M）、文本 transformer 全部冻结。

---

## 7. 结论

InCTRLPQA 在以下关键维度上**成功融合**了两个架构的优势：

- **保留 InCTRL 的核心竞争力**：多层 patch 残差（余弦距离）、WinCLIP 文本先验、全局 CLS 差异分支均完整继承。
- **充分采纳 AdaptCLIP PQA 的有效设计**：context map 构造、Conv 局部分割头、全局 MLP 头结构与原始 AdaptCLIP 完全对应，并升级了全局池化为可学习版本。
- **超越两者的架构升级**：解耦双路层权重、四路可学习决策头、patch score 规约公式，这些设计在两个原始论文中均未出现。

主要待改进点集中在：**图像分支 Adapter 顺序**、**GMP vs top-k mean**、**固定 beta**、以及**文本先验可学习化**。这四点均有明确的改进路径，且不影响当前的训练稳定性。
