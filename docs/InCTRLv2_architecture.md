# InCTRL v2 架构笔记

> **核心升级**：InCTRL v1 只在视觉残差空间判别异常；v2 在此基础上引入 DASL（判别式语义分支）和 OASL（一类正常性分支），形成 "视觉残差 + 语义判别 + 正常性一类建模" 的双分支框架。

---

## 1. 整体框架

```
                        ┌─────────────────────────────────────┐
                        │       Frozen CLIP Backbone           │
                        │  (Visual Encoder + Text Encoder)     │
                        └──────────┬──────────────┬───────────┘
                                   │              │
                            visual features   text prototypes
                          (class token + patch)  (F_n, F_a)
                                   │              │
              ┌────────────────────┼──────────────┼────────────────┐
              │                    │              │                │
        ┌─────▼──────┐     ┌──────▼──────┐ ┌─────▼──────┐  ┌─────▼──────┐
        │  Adapter ψ  │     │  Adapter φ₁ │ │  Adapter φ₂ │  │   InCTRL   │
        │ (image-level)│     │   (DASL)    │ │   (OASL)    │  │  residual  │
        └─────┬──────┘     └──────┬──────┘ └─────┬──────┘  │  learning  │
              │                    │              │         └─────┬──────┘
              │                    │              │               │
         F_x = I_x - I_p    S_a (semantic   Ŝ_a (one-class     M_x (patch
         → s_I (η head)      anomaly map)   anomaly map)     residual map)
              │                    │              │               │
              │              ┌─────▼──────┐ ┌─────▼──────┐       │
              │              │ M_p = ½(M_x │ │ M_n = ½(M_x│       │
              │              │    + S_a)   │ │    + Ŝ_a)  │       │
              │              └─────┬──────┘ └─────┬──────┘       │
              │                    │              │               │
              │              DASL 主分支       OASL 辅助分支          │
              │                    │              │               │
         s_q (text score)          │              │          s_p = max(M_x)
              │                    │              │               │
              ▼                    ▼              ▼               ▼
     ┌────────────────┐   ┌───────────────────────────┐   ┌──────────┐
     │ s(x) = (1-α)·  │   │ M' = (1-β)·Φ(M_p)        │   │          │
     │ (s_I+s_q)/2    │   │     + β·Φ(M_n)            │   │          │
     │ + α·s_p        │   │ (pixel-level anomaly map)  │   │          │
     └────────────────┘   └───────────────────────────┘   └──────────┘
      image-level score        pixel-level map
```

---

## 2. 与 v1 代码的逐模块映射

### 2.1 已有模块（可复用）

| v2 概念 | v1 代码位置 | 说明 |
|---------|-----------|------|
| Frozen CLIP Visual Encoder | `InCTRL.__init__` → `self.visual` (VisionTransformer_Mul) | 冻结，`p.requires_grad = False` |
| Frozen CLIP Text Encoder | `InCTRL.__init__` → `self.transformer/token_embedding/...` | 冻结 |
| Image-level Adapter ψ | `self.adapter = Adapter(640, 4)` | 对 class token 降维-升维 |
| Image-level residual F_x = I_x - I_p | `forward()` L529: `token_ref = token_n - token_ad` | 注意 v1 方向是 `normal - query` |
| Image-level residual head η | `self.diff_head_ref = TransformerBasicHead(640, 1)` | 输出 s_I |
| Patch-level residual M_x | `forward()` L538-554: 三层 NN cosine distance | `0.5*(1 - cos_sim).min()` |
| Text prototype F_n / F_a | `forward()` L558-577: `get_texts()` → `encode_text()` → mean → normalize | 已有 normal/abnormal 模板 |
| Image-level text score s_q | `forward()` L576: `(100 * img @ text_feat.T).softmax()` | softmax 取 abnormal 概率 |
| BinaryFocalLoss | `binary_focal_loss.py` | 可复用于 L_I |
| Multi-layer patch tokens | `VisionTransformer_Mul.forward()` → `patch_tokens` (layers 7,9,11) | 3 层输出 |

### 2.2 v2 新增模块（已实现）

| v2 新模块 | 代码位置 | 说明 |
|----------|---------|------|
| **PatchTextAdapter** | `model.py` L443-455 | 896→224→640, GELU, L2-norm 输出 |
| **Adapter φ₁** (DASL) | `InCTRLv2.__init__` → `self.phi1` | `PatchTextAdapter(896, 640)` |
| **Adapter φ₂** (OASL) | `InCTRLv2.__init__` → `self.phi2` | 独立参数，不与 φ₁ 共享 |
| **Pixel-level upsampling Φ** | `InCTRLv2._upsample_map()` | bilinear 15×15 → 240×240 |
| **Text prototype cache** | `InCTRLv2._build_text_prototypes()` | `@torch.no_grad()`, dict 缓存 |
| **Vectorized patch residual** | `InCTRLv2._compute_patch_residual()` | `bmm` 替代三重 for loop |
| **Semantic maps** | `InCTRLv2._compute_semantic_maps()` | `einsum` 逐层 softmax，多层平均 |
| **Inference fusion** | `InCTRLv2.forward()` L909-915 | `M' = (1-β)·Mp + β·Mn` |
| **DiceLoss** | `losses_inctrlv2.py` | soft dice, smooth=1.0 |
| **BinaryFocalLossProb** | `losses_inctrlv2.py` | 概率输入版 focal loss |
| **Composite losses** | `losses_inctrlv2.py` → `compute_inctrlv2_loss()` | L_DASL + L_OASL |
| **Legacy interface** | `InCTRLv2.forward_legacy()` | 兼容 v1 `(score, sI)` 返回 |

---

## 3. DASL 详解

### 3.1 设计目标

在 InCTRL 的视觉残差之外，学习一个 CLIP 文本语义引导的 "normal/abnormal 判别空间"。同时看两件事：
1. query 和 few-shot 正常提示图像之间的视觉残差
2. query 的 patch 特征更接近 F_n 还是 F_a

### 3.2 输入

| 输入 | 含义 |
|------|------|
| 查询图像 x | 正常或异常 |
| Few-shot 正常样本提示 P_I | 辅助训练集模拟的正常提示 |
| 图像级标签 y_x | 正常/异常二分类 |
| 像素级 mask G_x | 异常分割监督 |
| 正常文本提示 P_T^n | "a photo of a flawless [c]" 等 |
| 异常文本提示 P_T^a | "a photo of a [c] with flaw" 等 |

### 3.3 计算流程

**Step 1: 图像级残差** (继承自 v1)

```
I_p = mean(ψ(f_v(x'_k)))        # few-shot normal prototype
I_x = ψ(f_v(x))                 # query adapted feature
F_x = I_x - I_p                 # residual
s_I = η(F_x)                    # image-level residual score
```

对应 v1 代码：
```python
# model.py L526-529
token_ad = self.adapter.forward(token)        # I_x
token_n = self.adapter.forward(token_n)       # ψ(normal)
token_n = torch.mean(token_n, dim=1)          # I_p
token_ref = token_n - token_ad                # F_x (注意 v1 方向)
# L581
img_ref_score = self.diff_head_ref.forward(token_ref)  # s_I
```

**Step 2: Patch 级残差** (继承自 v1)

```
M_x^l(i,j) = 1 - ⟨F_v^l(i,j), h(F_v^l(i,j)|P_I)⟩
M_x = mean_l(M_x^l)
```

对应 v1 代码：
```python
# model.py L538-554，三层 patch token，逐 patch 找最近邻
s = (0.5 * (1 - (tmp @ tmp_n.T))).min(dim=1).values
```

**Step 3: 文本原型** (v1 已有)

```
F_n = mean(f_t(p_i)) / ||...||    # normal text prototype
F_a = mean(f_t(p_j)) / ||...||    # abnormal text prototype
```

对应 v1 代码：
```python
# model.py L564-574
pos_features = self.encode_text(pos_features)   # → F_n
neg_features = self.encode_text(neg_features)   # → F_a
```

**Step 4: 图像级 semantic score** (v1 已有)

```
s_q = softmax(⟨f_v(x), F_a⟩) / Σ_c softmax(⟨f_v(x), F_c⟩)
```

**Step 5: 图像级最终分数** (v2 改进)

```
s(x) = (1-α) · (s_I + s_q) / 2 + α · s_p(x)
```

其中 `s_p(x) = max(M_x)`。v1 的融合方式不同：
```python
# v1: model.py L583-588
holistic_map = text_score + img_ref_score + patch_ref_map  # 直接加
hl_score = self.diff_head.forward(holistic_map)            # 过 MLP
final_score = (hl_score + fg_score) / 2                    # 和 max patch 平均
```

**Step 6: Patch 级 semantic anomaly map** (v2 新增)

```
S_a^l(i,j) = softmax(⟨φ₁(F_v^l(i,j)), F_a⟩)   # 每个 patch 的 abnormality 概率
S_a = mean_l(S_a^l)                              # 多层平均
M_p = (M_x + S_a) / 2                            # DASL 主分支 anomaly map
```

### 3.4 DASL 损失

```
L_DASL = L_P + L_I

L_I = FocalLoss(s(x), y_x)                              # 图像级

L_P = FocalLoss([Φ(S_n), Φ(S_a)], G_x)                  # 双通道 focal
    + DiceLoss(Φ(S_a), G_x)                              # S_a 直接约束
    + DiceLoss(Φ(M_p), G_x)                              # 融合图约束
```

---

## 4. OASL 详解

### 4.1 设计动机

异常形态跨域变化巨大，但正常性相对稳定。OASL **仅用正常样本**学习泛化的 normality manifold，提升跨域稳定性。

### 4.2 结构

与 DASL 结构对称，但：
- 使用独立 Adapter **φ₂**（不与 φ₁ 共享参数）
- 仅用正常样本训练（G_x 全为 0）

```
Ŝ_a^l(i,j) = softmax(⟨φ₂(F̂_v^l(i,j)), F_a⟩)
Ŝ_a = mean_l(Ŝ_a^l)
M_n = (M_x + Ŝ_a) / 2     # normality-aware anomaly map
```

### 4.3 为什么参数不共享

| 模块 | 学什么 |
|------|--------|
| DASL / φ₁ | 正常 vs 异常的判别边界（需要异常样本） |
| OASL / φ₂ | 只由正常样本定义的 normality manifold |

共享会导致两个目标互相干扰。

### 4.4 OASL 损失

```
L_OASL = FocalLoss([Φ(Ŝ_n), Φ(Ŝ_a)], G_x)   # G_x 全 0
       + DiceLoss(Φ(Ŝ_a), G_x)
       + DiceLoss(Φ(M_n), G_x)
```

约束含义：对正常样本，所有 patch 都应被压到 normal side。

---

## 5. 推理融合

```
# 像素级
M' = (1-β) · Φ(M_p) + β · Φ(M_n)

# 图像级
s(x') = (1-α) · (s_I + s_q) / 2 + α · max(M_x)
```

论文推荐超参：**α ≈ 0.5, β ≈ 0.75**

---

## 6. DASL / OASL 互补性分析

| 场景 | 只有 v1 residual | + DASL | + OASL |
|------|----------------|--------|--------|
| query 和 prompt 外观差异大但正常 | 误报 ↑ | 语义判断为 normal → 抑制 | normality manifold 进一步抑制 |
| 真实异常但和 prompt 差异小 | 漏报 ↑ | semantic map 捕获异常语义 | — |
| DASL 对正常纹理过敏 | — | false positive | OASL normality signal 压制 |

---

## 7. 训练总损失

```
L_total = L_DASL + L_OASL
        = (L_P_dasl + L_I) + L_P_oasl
```

DASL 使用正常+异常混合 batch；OASL 使用仅正常 batch。

---

## 8. 实现进度

### 8.1 模型层 (`open_clip/model.py`) — ✅ 完成

- [x] `PatchTextAdapter` 类 (896→640 with GELU + L2-norm)
- [x] `InCTRLv2` 类，保留原 `InCTRL` 不动
- [x] `self.phi1` / `self.phi2` — DASL / OASL 独立 patch adapter
- [x] `_compute_patch_residual()` — 向量化 batch 矩阵乘法
- [x] `_build_text_prototypes()` — 带缓存的文本原型构造
- [x] `_compute_semantic_maps()` — DASL/OASL 共用逻辑，adapter 参数独立
- [x] `_compute_semantic_image_score()` — DASL s_q
- [x] `_upsample_map()` — bilinear 上采样
- [x] `forward()` 返回完整 dict (score/maps/intermediates)
- [x] `forward_legacy()` — 兼容 v1 接口

### 8.2 损失层 (`losses_inctrlv2.py`) — ✅ 完成

- [x] `BinaryFocalLossProb` — 概率输入的 focal loss
- [x] `DiceLoss` — soft dice loss
- [x] `pixel_focal_loss_2class()` — [Sn, Sa] 双通道 focal
- [x] `compute_dasl_loss()` — L_I + L_P
- [x] `compute_oasl_loss()` — L_OASL (zero mask)
- [x] `compute_inctrlv2_loss()` — 总 loss，支持 lambda 权重

### 8.3 待完成 — 数据层 / 训练引擎

- [ ] 数据集返回 pixel-level mask G_x
- [ ] 训练集区分 normal-only batch (OASL) vs mixed batch (DASL)
- [ ] `engine_IC.py` 训练循环适配 v2 loss
- [ ] `engine_IC.py` 评估支持 pixel-level AUROC / PRO
- [ ] 命令行参数扩展 (alpha, beta, lambda_dasl, lambda_oasl)
- [ ] anomaly map 可视化工具

---

## 9. 关键维度对比表

| 设计点 | InCTRL v1 | InCTRL v2 |
|--------|-----------|-----------|
| 任务 | anomaly detection (image-level) | detection + segmentation |
| 核心依据 | visual residual | visual residual + semantic priors |
| 分支结构 | 单分支 | DASL 主分支 + OASL 辅助分支 |
| 文本先验 | 仅图像级 text score | 图像级 score + patch 级 semantic map |
| 像素监督 | 无 | 使用 pixel-level mask |
| 正常性建模 | 无显式 normality manifold | OASL 显式 one-class normality |
| 可训练参数 | ψ, η, diff_head | ψ, η, φ₁, φ₂ (+ 分割头如需要) |
| 推理输出 | image-level score | image-level score + pixel-level anomaly map |
