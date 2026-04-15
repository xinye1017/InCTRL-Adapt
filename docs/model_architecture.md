# InCTRL 模型架构详细分析报告

> 基于 `open_clip/model.py` 的代码分析
> 分析日期: 2026-04-15

---

## 目录

1. [整体架构概览](#一整体架构概览)
2. [配置类](#二配置类)
3. [辅助函数](#三辅助函数)
4. [核心模型分析](#四核心模型详细分析)
5. [数据流全景图](#五数据流全景图)
6. [关键设计特点](#六关键设计特点)

---

## 一、整体架构概览

本项目基于 **CLIP (Contrastive Language-Image Pre-training)** 架构，在此基础上构建了 **InCTRL** 模型，用于**工业异常检测（Anomaly Detection）**。

核心思想：结合图像特征、文本特征、参考图像特征进行多模态异常评分。

### 类层次结构

```
nn.Module
├── CLIP              # 标准 CLIP 模型（基线参考）
├── InCTRL            # 核心异常检测模型（本文重点）
├── CustomTextCLIP    # 自定义文本 CLIP 变体
└── 辅助组件
    ├── TransformerBasicHead   # 3层MLP分类头
    └── Adapter                # Bottleneck特征适配器
```

### 模型关系图

```
┌─────────────────────────────────────────────────────┐
│                    InCTRL                            │
│                                                      │
│  ┌──────────────────┐    ┌───────────────────────┐  │
│  │ VisionTower_Mul   │    │  TextTransformer       │  │
│  │ (frozen ViT)     │    │  (frozen CLIP text)    │  │
│  │ out_layers=[7,9,11]│   │                        │  │
│  └────────┬─────────┘    └───────────┬───────────┘  │
│           │                          │               │
│  ┌────────▼─────────┐    ┌───────────▼───────────┐  │
│  │ Visual Adapter   │    │ Text Prompt Templates  │  │
│  │ / Simple Adapter │    │ (normal vs anomaly)    │  │
│  └────────┬─────────┘    └───────────┬───────────┘  │
│           │                          │               │
│  ┌────────▼──────────────────────────▼───────────┐  │
│  │           Multi-modal Fusion                   │  │
│  │   text_score + img_ref_score + Fp_map         │  │
│  └────────┬──────────────────────────────────────┘  │
│           │                                          │
│  ┌────────▼─────────┐                               │
│  │  TransformerBasicHead  → final_score             │  │
│  └────────────────────┘                               │
└─────────────────────────────────────────────────────┘
```

---

## 二、配置类

### 2.1 CLIPVisionCfg（视觉配置）

**位置**: `model.py` 第 36-61 行

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `layers` | `Union[Tuple[int,int,int,int], int]` | `12` | ViT 层数或 ResNet 层配置 |
| `width` | `int` | `768` | 特征维度 |
| `head_width` | `int` | `64` | Attention head 维度 |
| `mlp_ratio` | `float` | `4.0` | MLP 隐藏层扩展比例 |
| `patch_size` | `int` | `16` | 图像 patch 大小 |
| `image_size` | `Union[Tuple[int,int], int]` | `224` | 输入图像尺寸 |
| `ls_init_value` | `Optional[float]` | `None` | Layer Scale 初始值 |
| `patch_dropout` | `float` | `0.0` | Patch dropout 比例（论文推荐 0.5-0.75） |
| `input_patchnorm` | `bool` | `False` | 输入 patch normalization |
| `global_average_pool` | `bool` | `False` | 全局平均池化替代 CLS token |
| `attentional_pool` | `bool` | `False` | Attentional Pooler |
| `n_queries` | `int` | `256` | Attentional Pooler 查询数 |
| `attn_pooler_heads` | `int` | `8` | Attentional Pooler head 数 |
| `output_tokens` | `bool` | `True` | 是否输出 token |
| `timm_model_name` | `str` | `None` | timm 库模型名称 |
| `timm_model_pretrained` | `bool` | `False` | 是否使用 ImageNet 预训练权重 |
| `timm_pool` | `str` | `'avg'` | timm 特征池化方式 |
| `timm_proj` | `str` | `'linear'` | timm 输出投影方式 |
| `timm_proj_bias` | `bool` | `False` | timm 投影 bias |
| `timm_drop` | `float` | `0.0` | timm head dropout |
| `timm_drop_path` | `Optional[float]` | `None` | timm stochastic depth |

### 2.2 CLIPTextCfg（文本配置）

**位置**: `model.py` 第 63-79 行

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `context_length` | `int` | `77` | 文本序列最大长度 |
| `vocab_size` | `int` | `49408` | 词表大小（CLIP 默认） |
| `width` | `int` | `512` | Transformer 特征维度 |
| `heads` | `int` | `8` | Multi-head attention 头数 |
| `layers` | `int` | `12` | Transformer 层数 |
| `ls_init_value` | `Optional[float]` | `None` | Layer Scale 初始值 |
| `hf_model_name` | `str` | `None` | HuggingFace 模型名称 |
| `hf_tokenizer_name` | `str` | `None` | HuggingFace tokenizer 名称 |
| `hf_model_pretrained` | `bool` | `True` | 是否使用 HF 预训练权重 |
| `proj` | `str` | `'mlp'` | 文本投影方式 |
| `pooler_type` | `str` | `'mean_pooler'` | 池化类型 |
| `embed_cls` | `bool` | `False` | 是否在输入嵌入 CLS token |
| `pad_id` | `int` | `0` | Padding token ID |
| `output_tokens` | `bool` | `False` | 是否输出 token |

---

## 三、辅助函数

### 3.1 精度转换函数

```python
# 第 80-86 行
def get_cast_dtype(precision: str):
    """获取计算精度类型"""
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype

# 第 89-95 行
def get_input_dtype(precision: str):
    """获取输入数据精度类型"""
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype
```

### 3.2 文本模板系统

**状态描述模板**（第 97-101 行）:

```python
state_level = {
    "normal": [
        "{}", "flawless {}", "perfect {}", "unblemished {}",
        "{} without flaw", "{} without defect", "{} without damage"
    ],  # 7 种正常描述
    "anomaly": [
        "damaged {}", "{} with flaw", "{} with defect", "{} with damage"
    ]   # 4 种异常描述
}
```

**视角模板**（第 102-125 行）:

```python
template_level = [
    "a cropped photo of the {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of a {}.",
    "a dark photo of the {}.",
    "a jpeg corrupted photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of the {}.",
    "a photo of a {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for anomaly detection.",
    "a photo of the {} for anomaly detection."
]  # 共 22 种视角模板
```

### 3.3 get_texts() 函数

**位置**: 第 127-146 行

```python
def get_texts(obj_name):
    """
    根据物体类别生成正常/异常描述文本

    Args:
        obj_name: 物体类别名称

    Returns:
        normal_texts: list[str], 正常描述文本列表
        anomaly_texts:  list[str], 异常描述文本列表
    """
    # CIFAR-10 类别使用简化模板
    l = ["airplane", "automobile", "bird",
         "cat", "deer", "dog", "frog", "horse", "ship", "truck", "animal"]

    if obj_name in l:
        normal_texts = ["a photo of " + obj_name + " for anomaly detection."]
        anomaly_texts = ["a photo without " + obj_name + " for anomaly detection."]
    else:
        # 工业类别: 状态描述 x 视角模板 = 7x22=154 (normal), 4x22=88 (anomaly)
        normal_states = [s.format(obj_name) for s in state_level["normal"]]
        anomaly_states = [s.format(obj_name) for s in state_level["anomaly"]]
        normal_texts = [t.format(state) for state in normal_states for t in template_level]
        anomaly_texts = [t.format(state) for state in anomaly_states for t in template_level]

    return normal_texts, anomaly_texts
```

### 3.4 视觉塔构建函数

**`_build_vision_tower()`**（第 149-207 行）:

根据配置选择三种视觉骨干网络之一：

| 骨干网络 | 触发条件 | 说明 |
|----------|----------|------|
| `TimmModel` | `vision_cfg.timm_model_name` 不为空 | 使用 timm 库中的预训练模型 |
| `ModifiedResNet` | `vision_cfg.layers` 为 tuple/list | 自定义 Modified ResNet |
| `VisionTransformer` | 默认情况 | 标准 ViT，使用 `LayerNormFp32`（混合精度时）或 `LayerNorm` |

**`_build_vision_tower_Mul()`**（第 209-267 行）:

与 `_build_vision_tower` 结构相同，唯一区别是使用 **`VisionTransformer_Mul`**（支持多层特征输出的 ViT 变体）。

**`_build_text_tower()`**（第 269-305 行）:

| 编码器 | 触发条件 | 说明 |
|--------|----------|------|
| `HFTextEncoder` | `text_cfg.hf_model_name` 不为空 | HuggingFace 预训练文本编码器 |
| `TextTransformer` | 默认情况 | 自定义 CLIP 风格 Transformer |

---

## 四、核心模型详细分析

### 4.1 CLIP 类（基线参考）

**位置**: 第 307-375 行

#### 初始化

```python
class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,                    # 嵌入维度
            vision_cfg: CLIPVisionCfg,         # 视觉配置
            text_cfg: CLIPTextCfg,             # 文本配置
            quick_gelu: bool = False,          # 是否使用快速 GELU
            cast_dtype: Optional[torch.dtype] = None,  # 计算精度
            output_dict: bool = False,         # 是否以字典形式输出
    ):
```

#### 组件清单

| 组件 | 类型 | 说明 |
|------|------|------|
| `self.visual` | `VisionTransformer` / `ModifiedResNet` / `TimmModel` | 视觉编码器 |
| `self.transformer` | `TextTransformer` | 文本 Transformer |
| `self.context_length` | `int` | 文本最大长度（77） |
| `self.vocab_size` | `int` | 词表大小（49408） |
| `self.token_embedding` | `nn.Embedding` | Token → Embedding |
| `self.positional_embedding` | `nn.Parameter` | 位置编码 |
| `self.ln_final` | `LayerNorm` | 最终 LayerNorm |
| `self.text_projection` | `nn.Parameter` | 文本投影矩阵 |
| `self.attn_mask` | `Buffer` | 注意力掩码（非持久化） |
| `self.logit_scale` | `nn.Parameter` | 可学习温度参数，初始化为 `log(1/0.07)` |

#### 关键方法

**`encode_image()`**（第 344-346 行）:

```python
def encode_image(self, image, normalize: bool = False):
    """
    Args:
        image: Tensor, shape [b, 3, H, W]
        normalize: 是否进行 L2 归一化

    Returns:
        features: Tensor, shape [b, embed_dim]
    """
    features = self.visual(image)
    return F.normalize(features, dim=-1) if normalize else features
```

**`encode_text()`**（第 348-360 行）:

```python
def encode_text(self, text, normalize: bool = False):
    """
    Args:
        text: Tensor, shape [b, 77]，tokenized 文本
        normalize: 是否进行 L2 归一化

    Returns:
        features: Tensor, shape [b, embed_dim]
    """
    cast_dtype = self.transformer.get_cast_dtype()

    # 1. Token Embedding: [b, 77] → [b, 77, d_model]
    x = self.token_embedding(text).to(cast_dtype)

    # 2. 位置编码
    x = x + self.positional_embedding.to(cast_dtype)

    # 3. Transformer: NLD → LND → LND → NLD
    x = x.permute(1, 0, 2)  # [b, 77, d] → [77, b, d]
    x = self.transformer(x, attn_mask=self.attn_mask)
    x = x.permute(1, 0, 2)  # [77, b, d] → [b, 77, d]

    # 4. Final LayerNorm
    x = self.ln_final(x)  # [b, 77, d_model]

    # 5. 提取 EOT token 特征 + 投影
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    return F.normalize(x, dim=-1) if normalize else x
```

**`forward()`**（第 362-375 行）:

```python
def forward(self, image=None, text=None):
    image_features = self.encode_image(image, normalize=True) if image is not None else None
    text_features = self.encode_text(text, normalize=True) if text is not None else None

    if self.output_dict:
        return {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp()
        }
    return image_features, text_features, self.logit_scale.exp()
```

---

### 4.2 InCTRL 类（核心异常检测模型）

**位置**: 第 420-609 行

#### 初始化

```python
class InCTRL(nn.Module):
    def __init__(
            self,
            args,                               # 命令行参数/配置对象
            embed_dim: int,                     # 嵌入维度
            vision_cfg: CLIPVisionCfg,          # 视觉配置
            text_cfg: CLIPTextCfg,              # 文本配置
            quick_gelu: bool = False,           # 是否使用快速 GELU
            cast_dtype: Optional[torch.dtype] = None,  # 计算精度
            output_dict: bool = False,          # 是否以字典形式输出
    ):
```

#### 组件清单

| 组件 | 类型 | 输入→输出 | 参数量估计 | 可训练 |
|------|------|-----------|------------|--------|
| `self.visual` | `VisionTransformer_Mul` | [b,3,240,240] → [token, Fp_list, Fp] | ~86M (ViT-B/16) | **否**（冻结） |
| `self.transformer` | `TextTransformer` | [b,77] → [b,77,d_model] | ~63M | **否**（冻结） |
| `self.adapter` | `Adapter(640, 4)` | [b,640] → [b,160] → [b,640] | 640×160 + 160×640 = 204,800 | **是** |
| `self.diff_head` | `TransformerBasicHead(225, 1)` | [b,225] → [b,1] | 225×128 + 128×64 + 64×1 = 37,120 | **是** |
| `self.diff_head_ref` | `TransformerBasicHead(640, 1)` | [b,640] → [b,1] | 640×128 + 128×64 + 64×1 = 90,368 | **是** |
| `self.visual_adapter` | `VisualAdapter`（可选） | 适配全局/局部特征 | 取决于配置 | **是** |

**总可训练参数量**: 约 332K（不含 `visual_adapter`），相对于 CLIP 的 ~150M 总参数量，仅训练 **~0.2%** 的参数。

#### 冻结策略（第 462-466 行）

```python
# 视觉编码器：完全冻结
for p in self.visual.parameters():
    p.requires_grad = False

# 文本编码器：完全冻结
for p in text.parameters():
    p.requires_grad = False
```

#### encode_image() 方法

**位置**: 第 468-470 行

```python
def encode_image(self, image, out_layers: list = [7, 9, 11], normalize: bool = False):
    """
    提取多层视觉特征

    Args:
        image: Tensor, shape [b, 3, 240, 240]
        out_layers: list[int], 要提取特征的层索引，默认 [7, 9, 11]
        normalize: 是否进行 L2 归一化

    Returns:
        token: Tensor, 全局 CLS token 特征
        Fp_list: list[Tensor], 中间层 patch 特征列表
        Fp: Tensor, 最终 patch 特征
    """
    features = self.visual.forward(image, out_layers)
    return F.normalize(features, dim=-1) if normalize else features
```

#### encode_text() 方法

**位置**: 第 472-483 行

与标准 `CLIP.encode_text()` 完全相同，实现文本编码流程：
1. Token embedding → 位置编码
2. Transformer 前向传播
3. EOT token 提取 + 投影
4. 可选 L2 归一化

#### forward() 方法 — 完整前向传播详解

**位置**: 第 485-609 行

```python
def forward(self, tokenizer, image, text=None, normal_list=None):
    """
    Args:
        tokenizer: 文本分词器函数
        image: list/tuple，image[0] 为查询图像，image[1:] 为参考（正常）图像
        text: list/tuple/tensor，物体类别标识
        normal_list: list，可选的正常样本列表

    Returns:
        final_score: Tensor, shape [b]，综合异常分数
        img_ref_score: Tensor, shape [b]，参考图像差异分数
    """
```

---

##### 阶段 1: 图像预处理与参考图像准备（第 486-507 行）

```python
device = next(self.parameters()).device
use_non_blocking = device.type == "cuda"

# 路径 A: normal_list 为 None（默认）
if normal_list is None:
    img = image[0]                          # [b, 3, 240, 240] 查询图像
    normal_image = image[1:]                # [n, 3, 240, 240] 参考图像
    normal_image = torch.stack(normal_image)
    shot, b, _, _, _ = normal_image.shape   # shot = 参考图像数量
    normal_image = normal_image.reshape(-1, 3, 240, 240)  # [shot*b, 3, 240, 240]

# 路径 B: 使用提供的 normal_list
else:
    img = image[0]
    normal_image = normal_list
    normal_image = torch.stack(normal_image)
    normal_image = normal_image.unsqueeze(1)
    b = len(img)
    normal_image = normal_image.repeat(1, b, 1, 1, 1)  # [n, b, 3, 240, 240]
    shot, _, _, _, _ = normal_image.shape
    normal_image = normal_image.reshape(-1, 3, 240, 240)
```

**关键说明**:
- 查询图像: `[b, 3, 240, 240]` — batch 个待检测图像
- 参考图像: `[shot*b, 3, 240, 240]` — 每个查询对应 `shot` 个正常样本
- 输入图像尺寸固定为 **240x240**

---

##### 阶段 2: 视觉特征提取（第 509-519 行）

```python
# 查询图像特征提取
token, Fp_list, Fp = self.encode_image(img, normalize=False)

# 参考图像特征提取
token_n, Fp_list_n, Fp_n = self.encode_image(normal_image, normalize=False)

# 堆叠多层特征
Fp_list = torch.stack(Fp_list)        # [3, b, 226, dim] — 226 = 1(cls) + 225(patches)
Fp_list_n = torch.stack(Fp_list_n)    # [3, shot*b, 226, dim]

# 去除 CLS token，只保留 patch 特征
Fp_list = Fp_list[:, :, 1:, :]        # [3, b, 225, dim]
Fp_list_n = Fp_list_n[:, :, 1:, :]    # [3, shot*b, 225, dim]

# 重新整理形状
Fp_list = Fp_list.reshape(b, 3, 225, -1)        # [b, 3, 225, dim]
Fp_list_n = Fp_list_n.reshape(b, 3, 225 * shot, -1)  # [b, 3, 225*shot, dim]

# 参考图像的 CLS token
token_n = token_n.reshape(b, shot, -1)  # [b, shot, dim]
```

**张量形状总结**:

| 张量 | 形状 | 说明 |
|------|------|------|
| `token` | `[b, dim]` | 查询图像全局特征 |
| `Fp_list` | `[b, 3, 225, dim]` | 查询图像 3 层 patch 特征 |
| `token_n` | `[b, shot, dim]` | 参考图像全局特征 |
| `Fp_list_n` | `[b, 3, 225*shot, dim]` | 参考图像 3 层 patch 特征 |
| `225` | `15x15` | patch 数量（240/16=15，去除 CLS） |

---

##### 阶段 3: 特征适配与参考差异计算（第 524-546 行）

**路径 A: 使用 VisualAdapter（如果启用）**:

```python
# 全局特征适配
token_ad = self.visual_adapter.adapt_global(token)     # [b, dim]
token_n = self.visual_adapter.adapt_global(token_n)    # [b*shot, dim]
token_n = torch.mean(token_n, dim=1)                   # [b, dim] — 平均参考特征

# 计算差异向量
token_ref = token_n - token_ad                         # [b, dim]

# 全局异常评分
img_ref_score = self.diff_head_ref(token_ref)          # [b, 1]

# 局部 patch 特征适配
adapted_Fp_list = torch.stack(
    self.visual_adapter.adapt_local(list(Fp_list.unbind(dim=1))), dim=1
)  # [b, 3, 225, adapted_dim]

adapted_Fp_list_n = torch.stack(
    self.visual_adapter.adapt_local(list(Fp_list_n.unbind(dim=1))), dim=1
)  # [b, 3, 225*shot, adapted_dim]
```

**路径 B: 使用简单 Adapter（默认）**:

```python
# 通过 bottleneck adapter
token_ad = self.adapter.forward(token)        # [b, 640] → [b, 160] → [b, 640]
token_n = self.adapter.forward(token_n)       # [b, shot, 640] → [b, shot, 640]
token_n = torch.mean(token_n, dim=1)          # [b, 640]
token_ref = token_n - token_ad                # [b, 640]

# 全局异常评分
img_ref_score = self.diff_head_ref(token_ref) # [b, 1]

# patch 特征不做适配
adapted_Fp_list = Fp_list
adapted_Fp_list_n = Fp_list_n
```

**Adapter 内部结构**:

```
Adapter(640, reduction=4):
    Linear(640, 160) → ReLU → Linear(160, 640) → ReLU
```

---

##### 阶段 4: Patch 级异常评分（第 548-560 行）

```python
b = token.shape[0]           # batch size
n_layers = adapted_Fp_list.shape[1]  # 3 (层数)

Fp_map_per_layer = []
for j in range(n_layers):
    # L2 归一化 patch 特征
    tmp_x = F.normalize(adapted_Fp_list[:, j, :, :], dim=-1)    # [b, 225, dim]
    tmp_n = F.normalize(adapted_Fp_list_n[:, j, :, :], dim=-1)  # [b, 225*shot, dim]

    # 计算余弦距离: 0.5 * (1 - cos_sim) ∈ [0, 1]
    sim = 0.5 * (1 - torch.bmm(tmp_x, tmp_n.transpose(-2, -1)))  # [b, 225, 225*shot]

    # 每个 patch 取与参考 patch 的最小距离
    Fp_map_per_layer.append(sim.min(dim=-1).values)  # [b, 225]

# 三层特征取平均
Fp_map = torch.stack(Fp_map_per_layer, dim=0).mean(dim=0)  # [b, 225]

# 取最大 patch 异常分数
max_diff_score = Fp_map.max(dim=1).values  # [b]
```

**算法可视化**:

```
查询 Patch P_i        参考 Patch R_j
  [dim]                 [dim]
     │                    │
     └──── cos_sim ───────┘
            │
     0.5 * (1 - cos_sim)  → 距离 d(i,j)
            │
     min_j d(i,j)  → Patch i 的异常分数
            │
     max_i score_i  → 图像最大异常分数
```

---

##### 阶段 5: 文本编码与语义评分（第 562-601 行）

```python
# 路径 A: text 是 tuple/list（字符串类别名）
if isinstance(text, (tuple, list)):
    unique_type_names = sorted(set(text))
    text_list = list(text)

    pos_features_map = {}  # 正常特征缓存
    neg_features_map = {}  # 异常特征缓存

    for type_name in unique_type_names:
        # 生成文本模板
        normal_texts, anomaly_texts = get_texts(type_name.replace('_', " "))

        # 编码文本特征
        pos_features = self.encode_text(tokenizer(normal_texts).to(token.device))
        neg_features = self.encode_text(tokenizer(anomaly_texts).to(token.device))

        # 平均多个模板的特征
        pos_features_map[type_name] = F.normalize(pos_features, dim=-1).mean(dim=0)  # [dim]
        neg_features_map[type_name] = F.normalize(neg_features, dim=-1).mean(dim=0)  # [dim]

# 路径 B: text 是 tensor（类别索引）
else:
    unique_type_indices = sorted(set(text.tolist()))
    text_list = text.tolist()

    pos_feature = F.normalize(
        self.encode_text(tokenizer(["a photo for anomaly detection"]).to(token.device)),
        dim=-1,
    )  # [1, dim]

    neg_feature = F.normalize(
        self.encode_text(tokenizer(["a photo without anomaly for anomaly detection"]).to(token.device)),
        dim=-1,
    )  # [1, dim]

    pos_features_map = {i: pos_feature[0] for i in unique_type_indices}
    neg_features_map = {i: neg_feature[0] for i in unique_type_indices}

# 组装每个样本的正/负文本特征
pos_batch = torch.stack([pos_features_map[t] for t in text_list])  # [b, dim]
neg_batch = torch.stack([neg_features_map[t] for t in text_list])  # [b, dim]

# 计算查询图像与文本的相似度
token_norm = F.normalize(token, dim=-1)  # [b, dim]
pos_sim = (token_norm * pos_batch).sum(dim=-1)  # [b] — 与"正常"文本相似度
neg_sim = (token_norm * neg_batch).sum(dim=-1)  # [b] — 与"异常"文本相似度

# 二分类 logits → softmax → 异常概率
logits = 100 * torch.stack([pos_sim, neg_sim], dim=-1)  # [b, 2]
text_score = logits.softmax(dim=-1)[:, 1].unsqueeze(1)  # [b, 1]
```

**文本特征维度总结**:

| 类别 | 文本数量 | 说明 |
|------|----------|------|
| CIFAR-10 | 1 normal + 1 anomaly | 简化模板 |
| 工业类别 | 154 normal + 88 anomaly | 完整模板（状态×视角） |

---

##### 阶段 6: 多模态融合与最终评分（第 603-609 行）

```python
# 三种信号相加（广播机制）
# text_score: [b, 1] + img_ref_score: [b, 1] + Fp_map: [b, 225]
# → holistic_map: [b, 225]
holistic_map = text_score + img_ref_score + Fp_map

# 通过 MLP 分类头处理 holistic map
hl_score = self.diff_head.forward(holistic_map).squeeze(1)  # [b]

# 最终分数: 融合 MLP 输出 + patch 最大异常分，取平均
final_score = (hl_score + max_diff_score) / 2  # [b]

return final_score, img_ref_score.squeeze(1)
```

---

### 4.3 TransformerBasicHead 分类头

**位置**: 第 377-403 行

```python
class TransformerBasicHead(nn.Module):
    """
    Basic Transformer Head. No pool.

    3层全连接网络，每层使用 BatchNorm1d 和 ReLU，最后用 Sigmoid
    """

    def __init__(self, dim_in, num_classes):
        """
        Args:
            dim_in: 输入特征维度
            num_classes: 输出类别数
        """
        super(TransformerBasicHead, self).__init__()
        self.projection1 = nn.Linear(dim_in, 128, bias=True)
        self.projection2 = nn.Linear(128, 64, bias=True)
        self.projection3 = nn.Linear(64, num_classes, bias=True)
        self.bn1 = nn.BatchNorm1d(dim_in)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.projection1(x)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)
        x = self.projection2(x)
        x = F.relu(x, inplace=True)
        x = self.bn3(x)
        x = self.projection3(x)
        return torch.sigmoid(x)
```

**网络结构**:

```
输入 [*, dim_in]
    │
    ▼
Linear(dim_in → 128, bias=True)
    │
    ▼
ReLU(inplace)
    │
    ▼
BatchNorm1d(128)
    │
    ▼
Linear(128 → 64, bias=True)
    │
    ▼
ReLU(inplace)
    │
    ▼
BatchNorm1d(64)
    │
    ▼
Linear(64 → num_classes, bias=True)
    │
    ▼
Sigmoid → 输出 [*, num_classes]
```

**注意**: `bn1` 被创建但从未使用。

### 4.4 Adapter 适配器

**位置**: 第 405-417 行

```python
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        """
        Bottleneck adapter: c_in → c_in//reduction → c_in

        Args:
            c_in: 输入/输出特征维度
            reduction: 瓶颈压缩比例，默认 4
        """
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
```

**网络结构**:

```
输入 [*, c_in]
    │
    ▼
Linear(c_in → c_in//reduction, bias=False)
    │
    ▼
ReLU(inplace)
    │
    ▼
Linear(c_in//reduction → c_in, bias=False)
    │
    ▼
ReLU(inplace)
    │
    ▼
输出 [*, c_in]
```

### 4.5 CustomTextCLIP 变体

**位置**: 第 611-664 行

与标准 `CLIP` 的主要区别：
- 文本编码器封装为 `self.text`（而非直接展开组件）
- 支持独立的 `lock_text_tower()` 方法
- 文本编码直接调用 `self.text(text)`

---

## 五、数据流全景图

### 完整前向传播数据流

```
输入: (查询图像 [b,3,240,240], 参考图像 [shot,3,240,240], 类别 text)
│
├══════════════════════════════════════════════════════════╗
│                                                          ║
│  ┌──────────────────────── 视觉编码 ───────────────────┐  ║
│  │                                                     │  ║
│  │  VisionTransformer_Mul (frozen)                     │  ║
│  │  out_layers = [7, 9, 11]                            │  ║
│  │                                                     │  ║
│  │  查询图像 ──→ token [b,dim]                         │  ║
│  │               Fp_list [b,3,225,dim]                 │  ║
│  │               Fp [b,225,dim]                        │  ║
│  │                                                     │  ║
│  │  参考图像 ──→ token_n [b,shot,dim]                  │  ║
│  │               Fp_list_n [b,3,225*shot,dim]          │  ║
│  └─────────────────────────────────────────────────────┘  ║
│                                                          ║
│  ┌──────────────────── 特征适配 ──────────────────────┐  ║
│  │                                                     │  ║
│  │  token ──→ adapter ──→ token_ad [b,dim]            │  ║
│  │  token_n ──→ adapter ──→ mean ──→ token_n [b,dim]  │  ║
│  │                                                     │  ║
│  │  token_ref = token_n - token_ad  [b,dim]           │  ║
│  │                                                     │  ║
│  │  token_ref ──→ diff_head_ref ──→ img_ref_score     │  ║
│  │                                    [b,1]           │  ║
│  └─────────────────────────────────────────────────────┘  ║
│                                                          ║
│  ┌────────────────── Patch相似度分析 ─────────────────┐  ║
│  │                                                     │  ║
│  │  Fp_list     ──→ normalize ─┐                       │  ║
│  │  Fp_list_n   ──→ normalize ─┤                       │  ║
│  │                             ▼                       │  ║
│  │                     bmm → cos_distance              │  ║
│  │                     [b, 225, 225*shot]              │  ║
│  │                             │                       │  ║
│  │                     min(dim=-1) → per-layer map     │  ║
│  │                     mean(layers) → Fp_map [b,225]   │  ║
│  │                     max(dim=-1) → max_diff_score    │  ║
│  └─────────────────────────────────────────────────────┘  ║
│                                                          ║
│  ┌─────────────────── 文本编码 ───────────────────────┐  ║
│  │                                                     │  ║
│  │  text ──→ get_texts() ──→ normal_texts (154条)      │  ║
│  │                            anomaly_texts (88条)     │  ║
│  │                                                     │  ║
│  │  encode_text(normal_texts) → mean → pos_batch [b,dim]│║
│  │  encode_text(anomaly_texts) → mean → neg_batch [b,dim]│║
│  │                                                     │  ║
│  │  token · pos_batch → pos_sim [b]                   │  ║
│  │  token · neg_batch → neg_sim [b]                   │  ║
│  │                                                     │  ║
│  │  softmax(100 * [pos_sim, neg_sim])[:,1]            │  ║
│  │                              → text_score [b,1]    │  ║
│  └─────────────────────────────────────────────────────┘  ║
│                                                          ║
│  ┌─────────────────── 多模态融合 ─────────────────────┐  ║
│  │                                                     │  ║
│  │  holistic_map = text_score [b,1]                   │  ║
│  │                     + img_ref_score [b,1]          │  ║
│  │                     + Fp_map [b,225]               │  ║
│  │                   = [b, 225]                       │  ║
│  │                                                     │  ║
│  │  holistic_map ──→ diff_head ──→ hl_score [b]       │  ║
│  │                                                     │  ║
│  │  final_score = (hl_score + max_diff_score) / 2     │  ║
│  │                                                     │  ║
│  │  return (final_score, img_ref_score)               │  ║
│  └─────────────────────────────────────────────────────┘  ║
│                                                          ║
╚══════════════════════════════════════════════════════════╝
```

### 张量形状追踪

| 阶段 | 张量 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `img` | `[b, 3, 240, 240]` | 查询图像 |
| 输入 | `normal_image` | `[shot*b, 3, 240, 240]` | 参考图像（展平后） |
| 编码 | `token` | `[b, dim]` | 查询全局特征 |
| 编码 | `Fp_list` | `[b, 3, 225, dim]` | 查询3层patch特征 |
| 编码 | `token_n` | `[b, shot, dim]` | 参考全局特征 |
| 编码 | `Fp_list_n` | `[b, 3, 225*shot, dim]` | 参考3层patch特征 |
| 适配 | `token_ad` | `[b, dim]` | 适配后查询特征 |
| 适配 | `token_ref` | `[b, dim]` | 参考-查询差异 |
| 评分 | `img_ref_score` | `[b, 1]` | 全局参考异常分 |
| 评分 | `Fp_map` | `[b, 225]` | Patch级异常图 |
| 评分 | `max_diff_score` | `[b]` | 最大patch异常分 |
| 文本 | `pos_batch` | `[b, dim]` | 正常文本特征 |
| 文本 | `neg_batch` | `[b, dim]` | 异常文本特征 |
| 文本 | `text_score` | `[b, 1]` | 文本异常概率 |
| 融合 | `holistic_map` | `[b, 225]` | 多模态异常图 |
| 输出 | `final_score` | `[b]` | 最终异常分数 |
| 输出 | `img_ref_score` | `[b]` | 参考差异分数 |

---

## 六、关键设计特点

### 6.1 零样本/少样本异常检测

- **CLIP 骨干完全冻结**：仅训练轻量适配器和分类头
- **可训练参数占比 < 0.2%**：约 332K vs 150M 总参数
- **无需异常样本训练**：依赖 CLIP 的零样本能力和正常参考样本

### 6.2 三模态融合策略

模型融合三种独立的异常检测信号：

| 信号 | 来源 | 维度 | 物理含义 |
|------|------|------|----------|
| `text_score` | 文本语义匹配 | [b, 1] | 图像与"异常"文本的语义相似度 |
| `img_ref_score` | 全局参考差异 | [b, 1] | 查询与正常参考的全局特征差异 |
| `Fp_map` | 局部patch对比 | [b, 225] | 每个patch与正常参考的细粒度差异 |

### 6.3 分层Patch分析

- 使用 ViT 的 **3个中间层**（第7、9、11层）的patch特征
- 每层独立计算余弦距离，最后取平均
- 好处：浅层捕捉纹理/边缘异常，深层捕捉语义异常

### 6.4 模板增强文本提示

| 类别 | 正常文本数 | 异常文本数 | 组合方式 |
|------|------------|------------|----------|
| CIFAR-10 | 1 | 1 | 固定模板 |
| 工业类别 | 154 (7状态 × 22视角) | 88 (4状态 × 22视角) | 交叉组合 |

多个模板编码后取平均，提高对提示词的鲁棒性。

### 6.5 可选 VisualAdapter

- **默认模式**: 简单 `Adapter`（bottleneck MLP），仅适配全局特征
- **增强模式**: `VisualAdapter` 同时适配全局和局部patch特征
- 通过 `args.VISUAL_ADAPTER.ENABLE` 控制

### 6.6 余弦距离度量

```
distance = 0.5 * (1 - cos_similarity)
```

- 范围: [0, 1]
- 0 = 完全相同，1 = 完全相反
- 对特征幅度不敏感，更适合跨模态比较

---

## 七、模型变体对比

| 特性 | CLIP | InCTRL | CustomTextCLIP |
|------|------|--------|----------------|
| 视觉编码器 | VisionTransformer | VisionTransformer_Mul | VisionTransformer |
| 文本编码器 | 展开组件 | 展开组件 | 封装为 self.text |
| 冻结策略 | 无 | visual + text 冻结 | 可选 lock |
| 输出 | (img_feat, text_feat, scale) | (final_score, img_ref_score) | 同 CLIP |
| 额外组件 | 无 | adapter, diff_head, diff_head_ref | 无 |
| 用途 | 预训练/推理 | 异常检测训练/推理 | 自定义文本 CLIP |

---

## 八、训练脚本参考

相关文件: `train_va_2_4_8.py`

- 训练仅优化 `adapter`, `diff_head`, `diff_head_ref`, 和可选的 `visual_adapter`
- 使用正常样本作为参考（few-shot setting）
- 损失函数未在 `model.py` 中定义，由训练脚本指定

---

*报告生成于 2026-04-15，基于 `open_clip/model.py` 的完整代码分析。*
