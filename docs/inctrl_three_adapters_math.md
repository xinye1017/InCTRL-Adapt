# InCTRL 三适配器融合模型数学原理分析

本文档解释当前实现 `InCTRLWithAdapters` 的数学逻辑。重点不只是列公式，而是说明：

1. 哪些部分保留自 InCTRL
2. 哪些部分来自 AdaptCLIP
3. 这两者如何在当前代码中被统一

---

## 1. 目标与总体思想

我们要解决的问题是 few-shot visual anomaly detection：

- 给定一张查询图像 `x`
- 给定 `K` 张正常参考图像 `{p_k}_{k=1}^K`
- 给定该类别的文本 normal / anomaly 描述

模型需要输出异常分数 `S(x)`。

与原始 InCTRL 一样，当前模型认为异常证据主要来自三部分：

1. **局部残差**：query patch 是否偏离 normal prompt patch
2. **全局残差**：query global feature 是否偏离 normal prompt prototype
3. **文本先验**：query global feature 更像 normal 还是 anomaly text prototype

与原始 AdaptCLIP 一样，当前模型又进一步引入：

1. 文本侧的可学习 prompt context
2. 视觉侧的特征适配
3. prompt-query patch 对齐后的上下文增强局部建模

因此，本模型可以看成：

> InCTRL 的结构外壳 + AdaptCLIP 的适配与对齐能力。

---

## 2. 符号定义

记：

- `x`: 查询图像
- `{p_k}_{k=1}^K`: few-shot normal prompt images
- `f_v(.)`: 冻结 CLIP 视觉编码器
- `f_t(.)`: 冻结 CLIP 文本编码器

对于 query image：

- `g_q in R^D`: query global feature
- `q_l = {q_{l,i}}_{i=1}^N`: 第 `l` 个中间层的 patch features

对于 prompt images：

- `g_{p_k} in R^D`: 第 `k` 张 prompt image 的 global feature
- `p_{l,k} = {p_{l,k,j}}_{j=1}^N`: 第 `l` 层、第 `k` 张 prompt 的 patch features

其中：

- `D`: 特征维度，当前工程默认是 `640`
- `L`: 使用的 patch 层数，当前默认 `L = 3`
- `N`: patch 数，默认 `15 x 15 = 225`

---

## 3. 视觉编码与多层 patch 表示

视觉塔输出：

```math
(g_q, \{q_l\}_{l=1}^{L}) = f_v(x)
```

```math
(g_{p_k}, \{p_{l,k}\}_{l=1}^{L}) = f_v(p_k)
```

在代码里，对应：

- `query_global`
- `query_patch_tokens`
- `prompt_global`
- `prompt_patch_tokens`

当前实现还做了一个重要预处理：

1. 如 patch tokens 带 cls token，则剥离 cls token
2. 如 patch feature 不在统一投影空间，则投影到 `D`

即：

```math
\tilde{q}_{l,i} = \Pi(q_{l,i})
```

```math
\tilde{p}_{l,k,j} = \Pi(p_{l,k,j})
```

其中 `\Pi` 是 identity 或线性投影。

---

## 4. Visual Adapter：视觉特征适配

### 4.1 全局特征适配

当前 `VisualAdapter` 使用 residual bottleneck MLP：

```math
\hat{g}_q = A_g(g_q) = g_q + \Delta_g(g_q)
```

```math
\hat{g}_{p_k} = A_g(g_{p_k}) = g_{p_k} + \Delta_g(g_{p_k})
```

其中：

- `A_g(.)`: global adapter
- `\Delta_g(.)`: 两层 MLP 形成的残差修正

### 4.2 局部 patch 特征适配

对于每层 patch：

```math
\hat{q}_{l,i} = A_l(\tilde{q}_{l,i}) = \tilde{q}_{l,i} + \Delta_l(\tilde{q}_{l,i})
```

```math
\hat{p}_{l,k,j} = A_l(\tilde{p}_{l,k,j}) = \tilde{p}_{l,k,j} + \Delta_l(\tilde{p}_{l,k,j})
```

这里的关键点是：

- Visual Adapter 并不替代 residual learning
- 它只是把 residual learning 所用的特征变成任务适配后的特征

所以它仍然服从 InCTRL 的主结构。

---

## 5. Prompt-Query Adapter：局部对齐与上下文增强

这部分是当前实现最核心的数学改造。

## 5.1 多 shot prompt patch 展平

对于第 `l` 层，我们先把 `K` 张 prompt 的 patch 合并：

```math
P_l = \{\hat{p}_{l,k,j} \mid k = 1,\dots,K,\; j = 1,\dots,N\}
```

也可以记作一个大集合：

```math
P_l = \{u_{l,m}\}_{m=1}^{K N}
```

---

## 5.2 最近邻 patch 对齐

对每个 query patch `\hat{q}_{l,i}`，找最相似的 prompt patch：

```math
m^*(l,i) = \arg\max_m \cos(\hat{q}_{l,i}, u_{l,m})
```

记对应的对齐 prompt patch 为：

```math
\hat{p}^{*}_{l,i} = u_{l,m^*(l,i)}
```

当前代码使用的是归一化后内积：

```math
\cos(a,b) = \left\langle \frac{a}{\|a\|}, \frac{b}{\|b\|} \right\rangle
```

这与 InCTRL 的 nearest-neighbor residual 和 AdaptCLIP 的 prompt-query alignment 在归一化特征空间里是一致的。

---

## 5.3 纯残差项

当前实现中的纯残差定义为：

```math
r_{l,i} = \frac{1}{2}\left(1 - \cos(\hat{q}_{l,i}, \hat{p}^{*}_{l,i})\right)
```

注意：

- 理论设计里常写 `1 - cos`
- 当前代码实现是 `0.5 * (1 - cos)`，它可以看成把余弦距离压到 `[0, 1]` 更稳定的尺度上

若 `\hat{q}_{l,i}` 与对齐 prompt patch 越相似，则：

- `cos` 越大
- `r_{l,i}` 越小

若异常越明显，则：

- 对齐后仍不相似
- `r_{l,i}` 越大

---

## 5.4 上下文增强特征

这是 AdaptCLIP 的关键思想之一。

当前实现中：

```math
c_{l,i} = \hat{q}_{l,i} + \beta \cdot |\hat{q}_{l,i} - \hat{p}^{*}_{l,i}|
```

其中：

- `\beta` 是超参数
- `|\cdot|` 是逐元素绝对值

解释：

- `\hat{q}_{l,i}` 保留 query patch 本身的上下文语义
- `|\hat{q}_{l,i} - \hat{p}^{*}_{l,i}|` 显式编码“偏离 normal prompt 的幅值”
- 两者相加后，既保留了语义，又保留了异常差异

这就比单纯 residual 更丰富。

---

## 5.5 上下文异常响应

对上下文增强特征送入局部 head：

```math
s^{ctx}_{l,i} = h_l(c_{l,i})
```

其中：

- `h_l(.)` 是当前实现里的 `ContextResidualPatchHead`
- 输出范围经过 sigmoid 映射后在 `(0, 1)`

---

## 5.6 每层 patch anomaly evidence

最终每层 patch 证据定义为：

```math
e_{l,i} = \gamma_r \cdot r_{l,i} + \gamma_c \cdot s^{ctx}_{l,i}
```

其中：

- `\gamma_r`: 纯残差权重
- `\gamma_c`: 上下文 head 权重

这个公式是当前模型最重要的融合点。

它说明：

- InCTRL 提供 `r_{l,i}`
- AdaptCLIP 提供 `c_{l,i}` 和 `s^{ctx}_{l,i}`
- 最终局部异常证据不是纯 residual，也不是纯 learned head，而是二者组合

---

## 5.7 多层 patch 融合

InCTRL 原本就是多层 residual learning，所以这里不能只保留最后一层。

当前实现中：

```math
M(i) = \sum_{l=1}^{L} w_l \cdot e_{l,i}
```

默认情况下：

```math
w_l = \frac{1}{L}
```

若开启 `learnable_layer_weights=True`，则：

```math
w_l = \operatorname{softmax}(a_l)
```

最终得到：

```math
M = [M(1), M(2), \dots, M(N)]
```

它就是新的 hybrid patch anomaly map。

---

## 6. Image residual branch

当前模型保留了 InCTRL 的图像级 residual idea。

先对 prompt global 做 prototype 平均：

```math
\bar{g}_p = \frac{1}{K} \sum_{k=1}^{K} \hat{g}_{p_k}
```

再计算图像级残差：

```math
r_g = \bar{g}_p - \hat{g}_q
```

然后用图像级 head 输出：

```math
s_g = h_g(r_g)
```

其中：

- `h_g(.)` 是 `ImageResidualHead`
- `s_g in (0, 1)` 是 image-level anomaly score

这个分支负责补充：

- 整体结构偏移
- 语义级异常
- patch 分支难以稳定覆盖的全局异常证据

---

## 7. Text prior branch

## 7.1 文本 normal / anomaly 原型构建

对每个类别 `t`，文本分支生成：

- normal descriptors
- anomaly descriptors

然后通过可学习 context prompt 构造文本输入：

```math
z^{n}_{m} = f_t(\text{prompt}^{n}_{m})
```

```math
z^{a}_{m} = f_t(\text{prompt}^{a}_{m})
```

再对描述集合求均值，得到基础原型：

```math
\bar{z}^{n} = \frac{1}{M_n} \sum_m z^{n}_{m}
```

```math
\bar{z}^{a} = \frac{1}{M_a} \sum_m z^{a}_{m}
```

随后再做 residual projection：

```math
\hat{z}^{n} = A_t(\bar{z}^{n})
```

```math
\hat{z}^{a} = A_t(\bar{z}^{a})
```

其中 `A_t(.)` 是 `ResidualProjectionMLP`。

---

## 7.2 文本先验打分

当前实现中，先归一化：

```math
\tilde{g}_q = \frac{\hat{g}_q}{\|\hat{g}_q\|}
```

```math
\tilde{z}^{n} = \frac{\hat{z}^{n}}{\|\hat{z}^{n}\|}
```

```math
\tilde{z}^{a} = \frac{\hat{z}^{a}}{\|\hat{z}^{a}\|}
```

再计算：

```math
\ell_n = \tau \cdot \langle \tilde{g}_q, \tilde{z}^{n} \rangle
```

```math
\ell_a = \tau \cdot \langle \tilde{g}_q, \tilde{z}^{a} \rangle
```

其中：

- `\tau = exp(logit_scale)`

最后：

```math
s_t = \sigma(\ell_a - \ell_n)
```

这与 InCTRL 的 text prior spirit 一致：

- 文本分支不是主干分类器
- 它只是提供 query 更偏 normal 还是 anomaly 的先验概率

---

## 8. Holistic scoring 与最终分数

当前模型保留了 InCTRL 风格的整体打分逻辑。

## 8.1 融合输入

```math
H = M + \lambda_g s_g + \lambda_t s_t
```

其中：

- `M in R^N`: patch anomaly map
- `s_g in R`: image score
- `s_t in R`: text score

这里 `s_g` 和 `s_t` 通过 broadcast 加到每个 patch 位置。

---

## 8.2 holistic head

```math
s_h = h_h(H)
```

其中：

- `h_h(.)` 是 `HolisticScoringHead`
- 输出 `s_h in (0, 1)`

---

## 8.3 最大 patch 补偿

继承 InCTRL 思路：

```math
s_{max} = \max_i M(i)
```

这个项的意义是：

- holistic branch 擅长整体判断
- 但非常局部的小异常可能在整体 pooling 中被冲淡
- 所以需要一个 fine-grained compensation

---

## 8.4 最终异常分数

当前实现定义：

```math
S(x) = \operatorname{clip}(s_h + \alpha s_{max}, 0, 1)
```

其中：

- `\alpha` 是 max patch 补偿系数
- `clip` 在代码里是 `torch.clamp(..., min=0, max=1)`

这与原 InCTRL 的“整体打分 + 局部最大残差补偿”一脉相承。

---

## 9. 整体公式汇总

如果把整个模型压缩成一组核心公式，可以写成：

### 9.1 视觉适配

```math
\hat{g}_q = A_g(g_q), \qquad \hat{g}_{p_k} = A_g(g_{p_k})
```

```math
\hat{q}_{l,i} = A_l(q_{l,i}), \qquad \hat{p}_{l,k,j} = A_l(p_{l,k,j})
```

### 9.2 patch-level 对齐与上下文增强

```math
\hat{p}^{*}_{l,i} = \arg\max_{k,j}\cos(\hat{q}_{l,i}, \hat{p}_{l,k,j})
```

```math
r_{l,i} = \frac{1}{2}(1 - \cos(\hat{q}_{l,i}, \hat{p}^{*}_{l,i}))
```

```math
c_{l,i} = \hat{q}_{l,i} + \beta |\hat{q}_{l,i} - \hat{p}^{*}_{l,i}|
```

```math
e_{l,i} = \gamma_r r_{l,i} + \gamma_c h_l(c_{l,i})
```

```math
M(i) = \sum_{l=1}^{L} w_l e_{l,i}
```

### 9.3 image branch

```math
\bar{g}_p = \frac{1}{K}\sum_{k=1}^{K} \hat{g}_{p_k}
```

```math
s_g = h_g(\bar{g}_p - \hat{g}_q)
```

### 9.4 text branch

```math
\hat{z}^{n}, \hat{z}^{a} = \text{TextualAdapter}(t)
```

```math
s_t = \sigma\left(\tau \langle \tilde{g}_q, \tilde{z}^{a} \rangle - \tau \langle \tilde{g}_q, \tilde{z}^{n} \rangle\right)
```

### 9.5 holistic fusion

```math
H = M + \lambda_g s_g + \lambda_t s_t
```

```math
s_h = h_h(H)
```

```math
S(x) = \operatorname{clip}\left(s_h + \alpha \max_i M(i), 0, 1\right)
```

---

## 10. 为什么这仍然是 InCTRL，而不是另一个模型

这是一个很重要的问题。

如果只是把三适配器的输出简单平均：

```math
S = \frac{1}{4}(S_{inctrl} + S_{text} + S_{visual} + S_{pq})
```

那它就不是 InCTRL 主体增强版，而是 late-fusion ensemble。

当前实现没有这么做。

当前实现的结构本质上是：

1. **Patch branch** 仍是主局部异常统计量，只是纯 residual 被升级为 residual + context evidence
2. **Image branch** 仍是 residual-style global anomaly scoring
3. **Text branch** 仍是 prior branch，不抢 reference branch 的主导权
4. **最终输出** 仍由 holistic head + max patch compensation 形成

所以它的数学结构仍然是 InCTRL，只是每个 branch 的输入特征和局部统计量被三适配器增强了。

---

## 11. Alternating learning 的数学理解

## 11.1 参数分块

当前模型把参数分成两块：

### 视觉侧参数 `\theta_v`

包括：

- VisualAdapter
- PromptQueryAdapter
- ContextResidualPatchHead
- ImageResidualHead
- HolisticScoringHead
- patch projection

### 文本侧参数 `\theta_t`

包括：

- TextualAdapter 的 context tokens
- prototype adapter
- TextPriorHead 的 `logit_scale`

---

## 11.2 交替优化

设总损失为：

```math
\mathcal{L}(\theta_v, \theta_t)
```

则 alternating learning 可看作 block coordinate optimization：

### visual phase

```math
\theta_v \leftarrow \arg\min_{\theta_v} \mathcal{L}(\theta_v, \theta_t^{fixed})
```

### text phase

```math
\theta_t \leftarrow \arg\min_{\theta_t} \mathcal{L}(\theta_v^{fixed}, \theta_t)
```

当前代码用 `requires_grad` 强制实现这一点。

---

## 11.3 为什么不直接 joint optimize

如果文本和视觉适配同时激进更新，会有几个问题：

1. 文本原型与视觉特征会同时漂移，训练更不稳定
2. few-shot normal reference branch 的优势容易被“可学习分类器化”掩盖
3. 更容易在辅助训练域过拟合
4. 冻结 CLIP 主干的收益会被双侧 adapter 的共振部分抵消

所以 alternating learning 可以理解为：

> 在保持 backbone 冻结的前提下，降低两个适配空间同时扰动造成的耦合不稳定。

---

## 12. 复杂度分析

最重的部分来自 Prompt-Query 最近邻对齐。

对第 `l` 层：

- query patches: `N`
- prompt patches: `K N`
- 特征维度: `D`

相似度矩阵大小：

```math
R^{B \times N \times K N}
```

计算复杂度约为：

```math
\mathcal{O}(B \cdot N \cdot K N \cdot D)
```

多层总复杂度：

```math
\mathcal{O}(L \cdot B \cdot N \cdot K N \cdot D)
```

当前实现通过：

```python
similarity = torch.einsum("bnc,bmc->bnm", query_norm, prompt_norm)
```

完成 batched vectorized computation，避免了低效的逐 patch Python 循环。

---

## 13. 超参数的数学作用

### `beta`

控制上下文增强强度：

```math
c = q + \beta |q - p^*|
```

- 太小：更接近纯 residual
- 太大：容易让差异幅值盖过原始语义

### `gamma_r`

控制纯 residual 权重：

```math
e = \gamma_r r + \gamma_c s^{ctx}
```

- 大：更偏 InCTRL
- 小：更偏 AdaptCLIP 的 learned local response

### `gamma_c`

控制 context head 的权重。

### `lambda_g`

控制全局残差在 holistic fusion 中的作用。

### `lambda_t`

控制文本先验在 holistic fusion 中的作用。

### `alpha`

控制 fine-grained max patch compensation 强度。

---

## 14. 当前实现中需要特别注意的数学/工程边界

### 14.1 cls token 处理

当前代码默认 patch tokens 带 cls token，因此：

```math
N_{raw} = 226, \qquad N = 225
```

若 backbone 以后改为不返回 cls token，需要设置：

- `patch_has_cls_token=False`

否则 patch 数会错位。

### 14.2 patch feature 是否已经投影

当前工程默认：

- global feature 是 `embed_dim=640`
- patch feature 也在统一投影空间

若以后 backbone 返回的是投影前 patch feature（例如 `896`），则必须通过：

```math
\Pi: R^{896} \to R^{640}
```

把 patch branch 拉回与 global/text 同一空间。

### 14.3 normal / anomaly 文本模板数量不同

normal 模板集合与 anomaly 模板集合大小一般不相等，因此当前实现是先分别求平均：

```math
\bar{z}^{n} = \frac{1}{M_n} \sum z^{n}_m
```

```math
\bar{z}^{a} = \frac{1}{M_a} \sum z^{a}_m
```

这样不会因为模板数量多寡直接引入偏置。

### 14.4 `return_dict`

虽然接口保留了 `return_dict=True` 参数，但当前实现始终返回 dict。这是接口演化后的工程选择，不影响数学逻辑。

---

## 15. 一句话总结

当前 `InCTRLWithAdapters` 的数学本质可以写成：

> 用 InCTRL 的 holistic residual framework 组织异常证据，用 AdaptCLIP 的 Textual / Visual / Prompt-Query adapters 提升文本原型、视觉特征和局部对齐能力，再通过 alternating learning 降低双侧适配的耦合干扰。

如果压缩成最终总公式，可以写成：

```math
S(x)=\operatorname{clip}\left(
    h_h\left(
        \sum_{l=1}^{L} w_l \left[
            \gamma_r \cdot \frac{1-\cos(\hat{q}_{l,i}, \hat{p}^{*}_{l,i})}{2}
            + \gamma_c \cdot h_l\left(\hat{q}_{l,i} + \beta |\hat{q}_{l,i}-\hat{p}^{*}_{l,i}|\right)
        \right]
        + \lambda_g s_g
        + \lambda_t s_t
    \right)
    + \alpha \max_i M(i),
    0, 1
\right)
```

这就是当前实现的核心数学结构。
