# InCTRL 训练原理详解

## 1. 核心思想：In-context Residual Learning

InCTRL 的目标是训练一个**通用的异常检测模型**，给定少量正常样本作为参考（few-shot normal samples），能够判断查询图像是否为异常。

**核心假设：** 异常图像与正常样本之间的特征残差大于正常图像与正常样本之间的特征残差。

---

## 2. 训练数据组织：打包式输入

### 2.1 数据集结构

每个训练样本由**一张查询图像 + N 张同类正常参考图像**组成：

```python
# datasets/IC_dataset_new.py 第 132-155 行
def __getitem__(self, index):
    sample = self.image[index]
    image = self._load_image(sample['image_path'])      # 查询图像
    label = sample['target']                            # 0=正常, 1=异常

    shot_indices = self.precomputed_shot_indices[index] # 预采样的正常图索引

    image_list = [image]                               # 第一个是查询图
    for normal_idx in shot_indices:
        n_img = self._load_image(self.normal_samples[normal_idx]['image_path'])
        image_list.append(n_img)                        # 后面是 shot 张正常参考图

    return image_list, image_type, label
```

### 2.2 不同 Shot 数的输入结构

| Shot 数 | 输入图像数量 | 结构示例 |
|---------|-------------|---------|
| 2-shot | 3 张 | `[query_img, normal_1, normal_2]` |
| 4-shot | 5 张 | `[query_img, normal_1, normal_2, normal_3, normal_4]` |
| 8-shot | 9 张 | `[query_img, normal_1, ..., normal_8]` |

### 2.3 采样策略

```python
# 每个样本预计算 shot 索引，固定 seed 保证可复现
rng = np.random.default_rng(42)
for idx, sample in enumerate(self.image):
    sample_type = sample['type']
    normal_indices = self.type_to_normal_indices[sample_type]
    if len(normal_indices) < shot:
        shot_indices = list(range(len(self.normal_samples)))
    else:
        shot_indices = rng.choice(normal_indices, size=shot, replace=False).tolist()
    self.precomputed_shot_indices.append(shot_indices)
```

**关键点：** 同类样本中随机采样，确保参考图与查询图属于同一类别。

---

## 3. 模型 Forward 流程

### 3.1 图像特征提取

```python
# open_clip/model.py 第 511-530 行
def forward(self, tokenizer, image, ...):
    if normal_list == None:
        img = image[0].cuda()           # 查询图像 [B, 3, 240, 240]
        normal_image = image[1:]        # N 张正常参考图
        normal_image = torch.stack(normal_image)  # [shot, B, 3, 240, 240]
        shot, b, _, _, _ = normal_image.shape
        normal_image = normal_image.reshape(-1, 3, 240, 240)  # [shot*B, 3, 240, 240]
    else:
        # 测试时可以从缓存加载 normal features
        ...

    # 分别提取查询图和正常图的特征
    token, Fp_list, Fp = self.encode_image(img)              # 查询图特征
    token_n, Fp_list_n, Fp_n = self.encode_image(normal_image)  # 正常图特征
```

`encode_image` 返回三个输出：
- `token`: 全局特征 [B, embed_dim=640]
- `Fp_list`: 3 层 transformer 的 patch 特征 [3, B, 225, dim]
- `Fp`: 最后一层 patch 特征 [B, 225, dim]

### 3.2 全局特征适配（Adapter / Visual Adapter）

```python
# open_clip/model.py 第 543-549 行
if self.visual_adapter is not None:
    # Visual Adapter: 瓶颈残差结构
    token_ad = self.visual_adapter.adapt_global(token)
    token_n = self.visual_adapter.adapt_global(token_n)
    token_n = torch.mean(token_n, dim=1)         # 对 shot 维度取平均
    token_ref = token_n - token_ad               # 特征差异向量
else:
    # 原始 Adapter: 简单 MLP
    token_ad = self.adapter.forward(token)
    token_n = self.adapter.forward(token_n)
    token_n = torch.mean(token_n, dim=1)
    token_ref = token_n - token_ad
```

**核心计算：** 查询图的特征与正常样本平均特征的差异。

### 3.3 局部 Patch 特征适配

```python
# open_clip/model.py 第 551-561 行
# 适配后的 patch 特征用于 patch 级别相似度计算
Fp_list = Fp_list[:, :, 1:, :]      # 去掉 cls token
Fp_list_n = Fp_list_n[:, :, 1:, :]  # [3, B, 225, dim]

# Visual Adapter 对多层 patch 分别处理
Fp_list_3d = [Fp_list[:, layer_idx, :, :] for layer_idx in range(3)]
adapted_Fp_list = self.visual_adapter.adapt_local(Fp_list_3d)
adapted_Fp_list = torch.stack(adapted_Fp_list, dim=1)  # [B, 3, 225, dim]
```

### 3.4 Patch 级别异常计算

```python
# open_clip/model.py 第 583-606 行
for j in range(n_layers):  # 遍历 3 层 transformer
    tmp_x = adapted_Fp_list[:, j, :, :]        # 查询图 patch 特征 [B, 225, dim]
    tmp_n = adapted_Fp_list_n[:, j, :, :]      # 正常图 patch 特征 [B, 225*shot, dim]

    # 归一化
    tmp_x_norm = tmp_x / tmp_x.norm(dim=-1, keepdim=True)
    tmp_n_norm = tmp_n / tmp_n.norm(dim=-1, keepdim=True)

    # 计算余弦相似度矩阵
    sim = 0.5 * (1 - torch.bmm(tmp_x_norm, tmp_n_norm.transpose(-2, -1)))

    # 每个 patch 与所有正常 patch 的最小相似度
    min_dist = sim.min(dim=-1).values  # [B, 225]
    Fp_map_per_layer.append(min_dist)

# 多层平均
Fp_map = torch.stack(Fp_map_per_layer, dim=0).mean(dim=0)  # [B, 225]
```

**核心计算：** 对每个 patch，找到与所有正常样本 patch 的最小相似度。

### 3.5 文本特征融合

```python
# open_clip/model.py 第 608-667 行
# 获取正常/异常文本描述的特征
all_normal_texts = ["a photo of normal " + type for type in unique_types]
all_anomaly_texts = ["a photo of anomaly " + type for type in unique_types]
pos_features = self.encode_text(tokenizer(all_normal_texts))   # 正常文本特征
neg_features = self.encode_text(tokenizer(all_anomaly_texts))   # 异常文本特征

# 计算文本引导的异常分数
text_score = (100 * token_norm @ torch.cat([pos_f, neg_f], dim=0).T).softmax(dim=-1)
```

### 3.6 多路分数融合

```python
# open_clip/model.py 第 669-676 行
# 三路分数融合
holistic_map = text_score + img_ref_score + patch_ref_map
hl_score = self.diff_head.forward(holistic_map)  # 全局异常分数
max_diff_score = Fp_map.max(dim=1).values        # patch 最大异常分数
final_score = (hl_score + max_diff_score) / 2
```

**三路分数：**
1. `text_score`: 文本语义引导的异常判断
2. `img_ref_score`: 全局图像特征的残差
3. `patch_ref_map`: 局部 patch 的最小相似度

---

## 4. 损失函数

```python
# engine_IC.py 第 59-64 行
preds, preds2 = model(tokenizer, inputs, types, None)
loss_fun = BinaryFocalLoss()
loss = loss_fun(preds, labels.float()) + loss_fun(preds2, labels.float())
loss.backward()
```

模型同时输出两个分数（`preds`, `preds2`），可能是：
- 不同异常头（全局 vs patch）
- 不同融合方式的输出

使用 **BinaryFocalLoss** 而非普通 BCE，对难分样本给予更高权重。

---

## 5. Shot 数对训练的影响

### 5.1 训练时的 Shot 参数

| Shot 数 | 输入结构 | 任务特点 |
|---------|---------|---------|
| 2-shot | 3 张图 | 少量参考，模型需从极少信息判断异常 |
| 4-shot | 5 张图 | 中等参考，平衡性能与效率 |
| 8-shot | 9 张图 | 丰富参考，正常分布估计更准确 |

### 5.2 特征融合方式

```python
token_n = torch.mean(token_n, dim=1)  # 对 shot 维度取平均
```

- **更多 shot**：正常估计更鲁棒，但异常信号可能被稀释
- **更少 shot**：对每个参考图更敏感，但估计方差更大

### 5.3 训练-测试分布一致性

**原文 InCTRL 的做法：** 每个 shot 数单独训练一个模型。

**原因：**
1. 训练时 shot 数决定了模型处理的参考图数量
2. 训练和测试的 shot 数一致时，输入分布匹配
3. 不同 shot 数的模型学到的特征适配模式可能不同

**当前 InCTRL+VA 的做法：** 只训练 2-shot 模型，测试时用 2/4/8-shot 评估。

**这种做法的合理性：**
- 2-shot 模型在更少参考下学到的"比较能力"理论上可以泛化到更多参考
- 但可能存在一定程度的分布偏移

---

## 6. 完整训练流程总结

```
训练数据
    │
    ▼
[query_img, normal_1, ..., normal_N]  ← N 由 shot 参数决定
    │
    ▼
encode_image(query) → token, Fp_list
encode_image(normal) → token_n, Fp_list_n
    │
    ▼
特征适配 (Adapter / Visual Adapter)
    │
    ├─→ 全局特征差异 → img_ref_score
    ├─→ patch 相似度 → patch_ref_map
    └─→ 文本特征融合 → text_score
    │
    ▼
三路分数融合 → final_score
    │
    ▼
BinaryFocalLoss(preds, labels)
    │
    ▼
backward()
```

---

## 7. 与测试流程的区别

| 阶段 | Shot 参数作用 | Normal Image 来源 |
|------|--------------|------------------|
| 训练 | 决定每批输入的参考图数量 | 从训练集同一类别中采样 |
| 测试 | 决定 few-shot 样本数量 | 从预存的 few-shot samples 加载 |

**关键区别：**
- 训练时：参考图从当前 batch 的同类样本中随机采样
- 测试时：参考图来自预存的 few-shot 样本文件（`few-shot samples/{ds}/{ds}/{shot}/{cat}.pt`）
