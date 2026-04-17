# InCTRL 三适配器融合模型结构说明

本文档对应实现文件：

- `open_clip/inctrl_three_adapters.py`
- `engine_IC.py`
- `train_local.py`
- `tests/test_inctrl_three_adapters.py`

目标不是重新发明一个新模型，而是在 **保持 InCTRL 主体结构** 的前提下，把 AdaptCLIP 中最关键的三类适配机制嵌入进去：

1. `TextualAdapter`
2. `VisualAdapter`
3. `PromptQueryAdapter`

整个模型的名字是 `InCTRLWithAdapters`。

---

## 1. 这次改了什么

### 1.1 新增的核心模型文件

新增文件：

- `open_clip/inctrl_three_adapters.py`

其中实现了：

- `ResidualMLP`
- `ResidualProjectionMLP`
- `ContextResidualPatchHead`
- `ImageResidualHead`
- `HolisticScoringHead`
- `TextPriorHead`
- `VisualAdapter`
- `TextualAdapter`
- `PromptQueryAdapter`
- `InCTRLWithAdapters`

### 1.2 训练与推理接口改造

原本的 InCTRL 主要走旧接口：

```python
model(tokenizer, inputs, types, normal_list)
```

新模型改为更清晰的接口：

```python
outputs = model(
    query_image=query_image,
    prompt_images=prompt_images,
    normal_list=normal_list,
    obj_types=types,
    text_inputs=None,
    return_aux=False,
    return_dict=True,
)
```

其中：

- `query_image`: `[B, 3, H, W]`
- `prompt_images`: `[B, S, 3, H, W]`
- `normal_list`: `[S, 3, H, W]` 或 list of `[3, H, W]`，用于共享 prompt 兼容入口

### 1.3 训练脚本改造

#### `engine_IC.py`

- 改为实例化 `InCTRLWithAdapters`
- 把 dataloader 产出的 `inputs=[query, normal_1, ..., normal_s]` 显式拆成：
  - `query_image`
  - `prompt_images`
- 增加 phase-based alternating learning：
  - 偶数 epoch: `visual`
  - 奇数 epoch: `text`
- 为视觉参数和文本参数分别建立 optimizer

#### `train_local.py`

- `build_model()` 切到 `InCTRLWithAdapters`
- `build_optimizers()` 分别构建：
  - `visual_optimizer`
  - `text_optimizer`
- 训练循环显式调用：

```python
phase = "visual" if epoch % 2 == 0 else "text"
model.set_train_phase(phase)
```

### 1.4 测试补充

新增：

- `tests/test_inctrl_three_adapters.py`

验证了：

- 文本原型输出 shape
- Prompt-Query 最近邻对齐是否与 brute-force 一致
- `set_train_phase()` 的梯度开关逻辑
- `forward()` 在 `prompt_images` 与 `normal_list` 两种模式下是否稳定输出

### 1.5 兼容性修复

修复了一个现有 CPU 兼容问题：

`open_clip/transformer.py` 中原本有：

```python
out_attn = torch.zeros([H, H]).to('cuda')
```

现在改为：

```python
out_attn = torch.zeros([H, H], device=x.device)
```

这样小模型单元测试可以在 CPU 环境正常跑通。

---

## 2. 总体结构

新的结构仍然是 InCTRL 风格的三分支 + holistic fusion，只是每条分支都被 AdaptCLIP 机制增强了。

```text
Query Image
   |
   |---- CLIP Visual Tower ----> query global token
   |                          -> multi-layer patch tokens
   |
Prompt Images
   |
   |---- CLIP Visual Tower ----> prompt global token(s)
   |                          -> multi-layer prompt patch tokens

TextualAdapter --------------------> normal / anomaly text prototypes
VisualAdapter ---------------------> adapted global / local visual features
PromptQueryAdapter ----------------> aligned patch residual + context anomaly evidence

Patch Branch  -> hybrid patch map
Image Branch  -> image score
Text Branch   -> text score

hybrid patch map + lambda_g * image_score + lambda_t * text_score
                             |
                      HolisticScoringHead
                             |
                 holistic_score + alpha * max_patch_score
                             |
                         final_score
```

核心思想可以概括为一句话：

> InCTRL 决定外层骨架，AdaptCLIP 提供局部对齐、视觉适配、文本适配和交替学习能力。

---

## 3. 主模型 `InCTRLWithAdapters`

### 3.1 初始化逻辑

主类在 `open_clip/inctrl_three_adapters.py` 中定义：

```python
class InCTRLWithAdapters(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        vision_cfg,
        text_cfg,
        quick_gelu=False,
        cast_dtype=None,
        output_dict=True,
        patch_layers=(7, 9, 11),
        beta=1.0,
        gamma_r=0.5,
        gamma_c=0.5,
        lambda_g=1.0,
        lambda_t=1.0,
        alpha=0.5,
        use_text_adapter=True,
        use_visual_adapter=True,
        use_prompt_query_adapter=True,
        text_adapter_ctx_len=12,
        adapter_hidden_dim=256,
        patch_has_cls_token=True,
        feature_is_projected=True,
        learnable_layer_weights=False,
    ):
```

设计点：

- `patch_layers=(7, 9, 11)`：延续 InCTRL 的多层 patch learning
- `beta`：上下文增强残差强度
- `gamma_r / gamma_c`：纯残差与上下文打分的融合权重
- `lambda_g / lambda_t`：global branch 与 text branch 的 holistic 融合权重
- `alpha`：最大 patch 异常补偿项
- `patch_has_cls_token`：是否显式去掉 patch tokens 中的 cls token
- `feature_is_projected`：当前 patch/global 特征是否已经在同一投影空间

### 3.2 复用原 InCTRL backbone 的方式

视觉塔和文本塔不是重写，而是直接复用原项目构造函数：

```python
self.visual = _build_vision_tower_Mul(embed_dim, self.vision_cfg, quick_gelu, cast_dtype)
text = _build_text_tower(embed_dim, self.text_cfg, quick_gelu, cast_dtype)
```

这样做的好处：

- 保持与现有 InCTRL 权重加载兼容
- 保持 `strict=False` 时 backbone 权重可直接复用
- 不引入 AdaptCLIP 项目的额外复杂依赖

### 3.3 冻结 backbone

为了保持 CLIP 的原始视觉-语言对齐能力，backbone 默认冻结：

```python
for parameter in self.visual.parameters():
    parameter.requires_grad = False
for parameter in self.transformer.parameters():
    parameter.requires_grad = False
for parameter in self.token_embedding.parameters():
    parameter.requires_grad = False
for parameter in self.ln_final.parameters():
    parameter.requires_grad = False
self.positional_embedding.requires_grad = False
self.text_projection.requires_grad = False
```

这与 AdaptCLIP 的核心思想一致：训练的是 adapter，不是重训 CLIP 主干。

---

## 4. 适配器模块设计

## 4.1 `VisualAdapter`

定义：

```python
class VisualAdapter(nn.Module):
    def __init__(self, global_dim, local_dim, hidden_dim):
        self.global_adapter = ResidualMLP(global_dim, hidden_dim)
        self.local_adapter = ResidualMLP(local_dim, hidden_dim)
```

作用：

- `global_adapter`：适配 query / prompt 的全局 token
- `local_adapter`：适配多层 patch token

视觉适配发生在 residual learning 之前，而不是之后：

```python
adapted_query_global = self.visual_adapter.adapt_global(query_global)
adapted_prompt_global = self.visual_adapter.adapt_global(prompt_global)

adapted_query_patch_levels = self._adapt_patch_levels(query_patch_levels)
adapted_prompt_patch_levels = self._adapt_prompt_patch_levels(prompt_patch_levels)
```

这意味着：

- InCTRL 的 residual structure 没变
- 但 residual learning 用的是任务适配后的视觉表示

## 4.2 `TextualAdapter`

定义核心：

```python
class TextualAdapter(nn.Module):
    def __init__(..., context_length=12, hidden_dim=256):
        self.normal_ctx = nn.Parameter(torch.empty(context_length, context_dim))
        self.anomaly_ctx = nn.Parameter(torch.empty(context_length, context_dim))
        self.prototype_adapter = ResidualProjectionMLP(feature_dim, hidden_dim)
```

这里有两层设计：

### 第一层：learnable context tokens

对每个 normal / anomaly prompt，在文本 token embedding 中插入可学习上下文：

```python
prompt_prefix = " ".join(["X"] * self.context_length)
prompt_texts = [f"{prompt_prefix} {descriptor}" for descriptor in descriptors]
```

随后把 learnable context 与 token prefix/suffix 拼起来：

```python
prefix = token_embeddings[:, :1, :]
suffix = token_embeddings[:, 1 + self.context_length :, :]
learned_ctx = context_embeddings.unsqueeze(0).expand(len(descriptors), -1, -1)
prompted = torch.cat([prefix, learned_ctx, suffix], dim=1)
```

### 第二层：prototype residual adaptation

对编码后的 normal / anomaly prompt features 分别求平均，再通过 residual projection：

```python
normal_proto = self.prototype_adapter(normal_features.mean(dim=0, keepdim=True)).squeeze(0)
anomaly_proto = self.prototype_adapter(anomaly_features.mean(dim=0, keepdim=True)).squeeze(0)
```

这样做的结果是：

- prompt learning 发生在文本输入侧
- prototype adaptation 发生在文本特征侧
- 最终 normal/anomaly 原型仍然服务于 InCTRL 的 text prior branch

### 一个需要说明的实现细节

类里保留了：

```python
self.static_normal = [...]
self.static_anomaly = [...]
```

但当前实际 forward 中，描述词主入口仍然来自原 InCTRL 的 `get_texts(obj_type)`，或外部传入的 `text_inputs`。这说明当前代码更偏向：

- 保留 AdaptCLIP 风格的 prompt-learning 机制
- 同时继续沿用 InCTRL 原有的文本描述体系

这和“主体仍属于 InCTRL”的要求是一致的。

## 4.3 `PromptQueryAdapter`

这是三适配器里最关键的局部分支增强模块。

定义：

```python
class PromptQueryAdapter(nn.Module):
    def __init__(..., beta=1.0, gamma_r=0.5, gamma_c=0.5):
        self.patch_heads = nn.ModuleList(
            ContextResidualPatchHead(feature_dim, hidden_dim) for _ in range(num_layers)
        )
```

每层逻辑如下：

### Step 1：把 few-shot prompt patches 展平

```python
prompt_flat = prompt_level.reshape(batch_size, -1, dim)
```

这里原始输入是 `[B, S, N, C]`，展平后变成 `[B, S*N, C]`。

### Step 2：做向量化最近邻对齐

```python
query_norm = F.normalize(query_level, dim=-1)
prompt_norm = F.normalize(prompt_flat, dim=-1)
similarity = torch.einsum("bnc,bmc->bnm", query_norm, prompt_norm)
max_cosine, best_indices = similarity.max(dim=-1)
```

这是 per-patch nearest-neighbor alignment，没有写三层 Python for 循环。

### Step 3：构造纯残差

```python
residual = 0.5 * (1.0 - max_cosine)
```

### Step 4：构造上下文增强特征

```python
aligned_prompt = torch.gather(prompt_flat, 1, gather_index)
context_feat = query_level + self.beta * torch.abs(query_level - aligned_prompt)
```

### Step 5：局部上下文打分

```python
context_score = self.patch_heads[layer_idx](context_feat)
patch_evidence = self.gamma_r * residual + self.gamma_c * context_score
```

这一步非常关键：最终 patch branch 不再只靠 residual，而是 residual + context anomaly evidence 的融合。

---

## 5. 头部设计：如何保持 InCTRL 风格

## 5.1 `ImageResidualHead`

图像级分支输入是：

```python
prompt_global_proto = adapted_prompt_global.mean(dim=1)
image_residual = prompt_global_proto - adapted_query_global
image_score = self.image_head(image_residual)
```

这对应原 InCTRL 的 image-level residual learning，只不过输入特征现在先经过了 `VisualAdapter`。

## 5.2 `TextPriorHead`

输入：

- `adapted_query_global`
- `normal_proto`
- `anomaly_proto`

代码：

```python
normal_logit = scale * torch.sum(query_global * normal_proto, dim=-1)
anomaly_logit = scale * torch.sum(query_global * anomaly_proto, dim=-1)
text_score = torch.sigmoid(anomaly_logit - normal_logit)
```

含义：

- 文本分支仍然只是先验分支
- 它不会单独形成另一个最终分类器
- 它只提供 `text_score` 进入 holistic fusion

## 5.3 `HolisticScoringHead`

融合方式：

```python
holistic_input = (
    hybrid_patch_map
    + self.lambda_g * image_score.unsqueeze(-1)
    + self.lambda_t * text_score.unsqueeze(-1)
)
holistic_score = self.holistic_head(holistic_input)
max_patch_score = hybrid_patch_map.max(dim=-1).values
final_score = torch.clamp(holistic_score + self.alpha * max_patch_score, min=0.0, max=1.0)
```

这部分保留了 InCTRL 的两个核心思想：

1. holistic scoring
2. max residual compensation

也就是说：

- `holistic_score`：负责整体异常判断
- `max_patch_score`：负责对局部小异常做补偿

---

## 6. 前向传播流程

## 6.1 输入规范化

首先把 `prompt_images` / `normal_list` 统一成同一种 shape。

```python
prompt_images = self._coerce_prompt_images(query_image, prompt_images, normal_list)
```

支持两种模式：

### 模式 A：标准 few-shot batch

```python
query_image:  [B, 3, H, W]
prompt_images:[B, S, 3, H, W]
```

### 模式 B：共享 prompt 推理

```python
normal_list: [S, 3, H, W]
```

会被扩展成：

```python
[B, S, 3, H, W]
```

注意这里是 **当前 batch 内共享同一组 normal prompt**，没有做跨 shot cache 混用。

## 6.2 编码 query / prompt 图像

```python
query_global, query_patch_tokens = self._encode_visual_features(query_image)
prompt_global, prompt_patch_tokens = self._encode_visual_features(flat_prompt_images)
```

## 6.3 patch token 预处理

```python
query_patch_levels = self._prepare_patch_levels(...)
prompt_patch_levels = self._prepare_patch_levels(...)
```

功能包括：

- stack 多层 patch tokens
- 如有 cls token，则剥离一次
- 必要时做 patch projection
- 整理成：
  - query: `[B, L, N, C]`
  - prompt: `[B, L, S, N, C]`

## 6.4 视觉适配

```python
adapted_query_global = self.visual_adapter.adapt_global(query_global)
adapted_prompt_global = self.visual_adapter.adapt_global(prompt_global)
adapted_query_patch_levels = self._adapt_patch_levels(query_patch_levels)
adapted_prompt_patch_levels = self._adapt_prompt_patch_levels(prompt_patch_levels)
```

## 6.5 Prompt-Query 多层 patch 对齐

```python
pq_outputs = self.prompt_query_adapter(
    query_patch_levels=adapted_query_patch_levels,
    prompt_patch_levels=adapted_prompt_patch_levels,
)
```

随后按层融合：

```python
hybrid_patch_map = sum(
    weight * evidence
    for weight, evidence in zip(layer_weights, pq_outputs["patch_evidence_maps"])
)
```

## 6.6 image branch

```python
prompt_global_proto = adapted_prompt_global.mean(dim=1)
image_residual = prompt_global_proto - adapted_query_global
image_score = self.image_head(image_residual)
```

## 6.7 text branch

```python
normal_proto, anomaly_proto = self._build_adapted_text_prototypes(...)
text_score, text_aux = self.text_prior_head(adapted_query_global, normal_proto, anomaly_proto)
```

## 6.8 holistic fusion

```python
holistic_input = hybrid_patch_map + lambda_g * image_score + lambda_t * text_score
holistic_score = self.holistic_head(holistic_input)
final_score = clamp(holistic_score + alpha * max_patch_score)
```

---

## 7. 输出字典

当前 `forward()` 最终返回：

```python
{
    "final_score": final_score,
    "holistic_score": holistic_score,
    "image_score": image_score,
    "text_score": text_score,
    "patch_map": hybrid_patch_map,
    "max_patch_score": max_patch_score,
    "aux": aux,
}
```

其中：

- `patch_map`: `[B, N]`
- `aux["patch_map_2d"]`: `[B, H_patch, W_patch]`
- `aux["per_layer_patch_evidence"]`: 每层 patch anomaly evidence
- `aux["per_layer_residual"]`: 每层纯 residual
- `aux["per_layer_context_score"]`: 每层上下文 head 输出
- `aux["aligned_indices"]`: 最近邻 prompt patch 索引
- `aux["text_prototypes"]`: normal/anomaly text prototypes

---

## 8. 交替学习如何落地

## 8.1 参数分组接口

### 视觉侧参数

```python
def get_visual_parameters(self):
    modules = [
        self.patch_projection,
        self.visual_adapter,
        self.prompt_query_adapter,
        self.image_head,
        self.holistic_head,
    ]
```

### 文本侧参数

```python
def get_text_parameters(self):
    modules = [
        self.textual_adapter,
        self.text_prior_head,
    ]
```

## 8.2 相位切换

```python
def set_train_phase(self, phase):
    for parameter in self.get_visual_parameters():
        parameter.requires_grad = phase in {"visual", "joint"}
    for parameter in self.get_text_parameters():
        parameter.requires_grad = phase in {"text", "joint"}
```

同时 backbone 会始终保持冻结。

## 8.3 训练脚本中的用法

### `engine_IC.py`

```python
visual_optimizer, text_optimizer = _build_alternating_optimizers(model, lr=1e-3)
phase = "visual" if cur_epoch % 2 == 0 else "text"
base_model.set_train_phase(phase)
optimizer = visual_optimizer if phase == "visual" else text_optimizer
```

### `train_local.py`

```python
visual_optimizer, text_optimizer = build_optimizers(model, lr=lr, weight_decay=weight_decay)

for epoch in range(start_epoch, n_epochs):
    phase = "visual" if epoch % 2 == 0 else "text"
    model.set_train_phase(phase)
```

这就是 AdaptCLIP 式 alternating learning 的工程落地版本。

---

## 9. 对应代码片段总览

## 9.1 文本 prompt 编码

```python
def encode_text_prompted(self, prompts, tokenized_prompts, normalize=False):
    cast_dtype = self.transformer.get_cast_dtype()
    x = prompts.to(cast_dtype)
    x = x + self.positional_embedding[: x.shape[1]].to(cast_dtype)
    x = x.permute(1, 0, 2)
    x = self.transformer(x, attn_mask=self.attn_mask)
    x = x.permute(1, 0, 2)
    x = self.ln_final(x)
    x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
    return F.normalize(x, dim=-1) if normalize else x
```

## 9.2 Prompt-Query 最近邻对齐

```python
query_norm = F.normalize(query_level, dim=-1)
prompt_norm = F.normalize(prompt_flat, dim=-1)
similarity = torch.einsum("bnc,bmc->bnm", query_norm, prompt_norm)
max_cosine, best_indices = similarity.max(dim=-1)
residual = 0.5 * (1.0 - max_cosine)
```

## 9.3 上下文增强残差

```python
aligned_prompt = torch.gather(prompt_flat, 1, gather_index)
context_feat = query_level + self.beta * torch.abs(query_level - aligned_prompt)
context_score = self.patch_heads[layer_idx](context_feat)
patch_evidence = self.gamma_r * residual + self.gamma_c * context_score
```

## 9.4 holistic fusion

```python
holistic_input = hybrid_patch_map + self.lambda_g * image_score.unsqueeze(-1) + self.lambda_t * text_score.unsqueeze(-1)
holistic_score = self.holistic_head(holistic_input)
max_patch_score = hybrid_patch_map.max(dim=-1).values
final_score = torch.clamp(holistic_score + self.alpha * max_patch_score, min=0.0, max=1.0)
```

---

## 10. 当前实现与原设计要求的关系

### 已满足的点

- 保留 InCTRL 多层 patch residual 主体
- 保留 image residual branch
- 保留 text prior branch
- 保留 holistic fusion + max patch compensation
- 显式加入三适配器
- 提供 alternating learning 的参数分组和 phase 接口
- 支持 few-shot prompt image 作为 reference
- 支持 `prompt_images` 和 `normal_list` 两种入口

### 当前实现中值得注意的小差异

1. `TextualAdapter` 中定义了 `static_normal/static_anomaly`，但当前 prototype 构建实际主入口仍是 `get_texts(obj_type)` / `text_inputs`。
2. `forward(..., return_dict=True)` 目前始终返回 dict，`return_dict` 参数只是保留接口，不再切 tuple。
3. `residual` 采用的是 `0.5 * (1 - cosine)`，属于归一化余弦距离形式，而不是严格的 `1 - cosine`。
4. `get_visual_parameters()` 目前把 `patch_projection` 放在 visual phase；这是合理的，因为它服务于 patch branch 的视觉对齐。

---

## 11. 建议如何阅读代码

如果你想最快理解整个实现，建议按下面顺序读：

1. `InCTRLWithAdapters.__init__`
2. `TextualAdapter.build_prototypes`
3. `PromptQueryAdapter.forward`
4. `InCTRLWithAdapters.forward`
5. `set_train_phase / get_visual_parameters / get_text_parameters`
6. `engine_IC.py` 或 `train_local.py` 的 phase 切换逻辑

如果你要继续做训练和消融，建议重点盯这些超参数：

- `beta`
- `gamma_r`
- `gamma_c`
- `lambda_g`
- `lambda_t`
- `alpha`
- `text_adapter_ctx_len`
- `patch_layers`

这些参数基本对应了三适配器与 InCTRL 主体之间的耦合强度。
