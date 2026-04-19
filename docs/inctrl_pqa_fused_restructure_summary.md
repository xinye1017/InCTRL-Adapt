# InCTRL PQA Fused Restructure Summary

## 本次改动做了什么

本次任务完成了 `InCTRLPQA` 的结构重整，目标是保留 InCTRL 的三路证据形态，同时把 PQA 收敛成唯一的 patch branch。

核心改动如下：

### 1. 简化 fused 模型主干
- 文件：`open_clip/inctrl_pqa_fused.py`
- 模型定义：`InCTRLPQA`
- 结果：
  - 删除了旧的 `holistic_head` 路径
  - `decision_head` 从 5 路输入缩减为 4 路输入
  - 最终融合输入固定为：
    - `patch_logit`
    - `pqa_logit`
    - `image_logit`
    - `text_logit`

### 2. 统一 patch branch 归属
- 文件：`open_clip/inctrl_pqa_fused.py`
- 模块定义：`PQAdapter`
- 结果：
  - `PQAdapter.forward()` 现在只返回一组清晰的 patch payload：
    - `inctrl_patch_maps`
    - `pqa_patch_maps`
    - `pqa_global_logits`
    - `pqa_local_logits`
    - `aligned_indices`
    - `aligned_prompt_features`
    - `layer_weights`
  - 删除了重复/重叠的 patch 输出，例如旧的 `patch_logits`、`patch_scores`、`residual_maps` 等
  - 顶层 `InCTRLPQA.forward()` 不再单独走 `_compute_patch_residuals()` 旧路径，而是直接从 `PQAdapter` 聚合 patch 证据

### 3. 简化 forward 输出契约
- 顶层主输出现在收敛为：
  - `final_score`
  - `final_logit`
  - `patch_score`
  - `patch_logit`
  - `pqa_score`
  - `pqa_logit`
  - `image_score`
  - `image_logit`
  - `text_score`
  - `text_logit`
  - `fused_patch_map`
  - `pqa_local_logits`
  - `patch_map_fusion_weights`
  - `aux`
- 删除了旧的重复字段，例如：
  - `base_score`
  - `base_logit`
  - `holistic_score`
  - `holistic_logit`
  - `base_patch_map`
  - `hybrid_patch_map`
  - `raw_base_patch_map`
  - 各种 `max_*` 旧派生量

### 4. patch map 到 patch logit 的约简逻辑
- 新增 `_reduce_patch_map_to_logit()`
- 逻辑：
  - `patch_score = 0.5 * max + 0.5 * topk_mean`
  - 再通过 `_score_to_logit()` 转成 `patch_logit`
- 目的：
  - 保留异常峰值敏感性
  - 同时降低单个 noisy patch 的偶然影响

### 5. loss contract 对齐
- 文件：`open_clip/inctrl_pqa_losses.py`
- 结果：`compute_training_loss()` 只依赖简化后的输出：
  - `final_logit`
  - `pqa_logit`
  - `image_logit`（可选）
  - `pqa_local_logits`
- 不再依赖：
  - `base_logit`
  - `holistic_logit`
- 当前 loss 项：
  - `final_loss`
  - `pqa_loss`
  - `image_loss`
  - `pqa_mask_loss`
  - `total_loss`

### 6. evaluator 分支指标同步
- 文件：`train_local.py`
- `evaluate()` 里的 branch metrics 已切到新接口：
  - `patch`
  - `text`
  - `pqa`
  - `image`
- 不再读取旧的：
  - `base`
  - `holistic`
  - `max_patch`
  - `raw_max_patch`

### 7. cache 行为回归保护
- `build_prompt_feature_cache()` 保持输出不变：
  - `prompt_global`
  - `prompt_patch_levels`
  - `num_shots`
- `build_text_prototype_cache()` 保持输出不变：
  - `normal_proto`
  - `anomaly_proto`
- 已为这两个 cache builder 补了 round-trip 回归测试

### 8. 接口安全性补丁
- `prompt_images` 的 4D / 5D 输入行为现在更严格
- `return_dict=False` 和 `return_dict=None` 的行为已补测试
- `forward()` 现在会显式校验：
  - `obj_types` 长度是否等于 query batch size
  - `text_prototype_cache` 的 batch 维是否为 `1` 或 `batch_size`
- `train_local.py` 中 few-shot `.pt` 文件加载补了：
  - `torch.load(..., map_location="cpu")`
  - 避免不同设备之间缓存加载失败

## 本次修改涉及的主要文件

### 模型定义
- `open_clip/inctrl_pqa_fused.py`

### 训练 loss
- `open_clip/inctrl_pqa_losses.py`

### 训练入口
- `train_local.py`

### 测试
- `tests/test_inctrl_pqa_fused.py`
- `tests/test_inctrl_pqa_losses.py`

## 下次训练应该使用哪个入口

下次继续训练/实验，建议直接使用：

- 训练入口文件：`train_local.py`
- 主入口位置：`train_local.py:761`
- 实际训练主流程函数：`run_experiment()`，位置 `train_local.py:386`
- 模型构建函数：`build_model()`，位置 `train_local.py:195`

也就是说，下次直接从这个脚本启动：

```bash
python train_local.py \
  --train-datasets mvtec visa \
  --train-shot 4 \
  --eval-shots 2 4 8 \
  --batch-size 48 \
  --epochs 10
```

如果要从 checkpoint 恢复：

```bash
python train_local.py \
  --resume checkpoints/trained_on_mvtec/final_model_shot_4.pt \
  --start-epoch 10
```

## 当前使用的模型是哪一个 py 里定义的

当前训练脚本使用的模型类是：

- 模型类：`InCTRLPQA`
- 定义文件：`open_clip/inctrl_pqa_fused.py`
- 训练脚本里导入位置：`train_local.py:28`
- 训练脚本里实例化位置：`train_local.py:202`

也就是说，下次训练实际跑的是：

- `open_clip/inctrl_pqa_fused.py` 里的 `InCTRLPQA`

不是 `open_clip/inctrl_three_adapters.py`。

## 当前训练/评估路径说明

训练时：
- `train_local.py -> build_model() -> InCTRLPQA`
- `train_local.py -> compute_training_loss()` 使用 `open_clip/inctrl_pqa_losses.py`

评估时：
- `train_local.py -> build_cached_prompt_features()`
- `train_local.py -> build_cached_text_prototypes()`
- `train_local.py -> evaluate()`

## 建议的下次使用方式

如果下次是继续这个 fused 方向，默认就走：

1. `train_local.py`
2. `open_clip/inctrl_pqa_fused.py` 里的 `InCTRLPQA`
3. `open_clip/inctrl_pqa_losses.py` 里的 `compute_training_loss()`

如果后续要对比旧三分支版本，再单独回看：
- `open_clip/inctrl_three_adapters.py`

但当前这次重构完成后，主训练入口已经对齐到 fused 版本。
