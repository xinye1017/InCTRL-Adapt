# No-VA 多种子评估结果汇总

来源：`/Users/xinye/Downloads/multi_seed_eval.tar.gz`

解压位置：`results/multi_seed/`

包含文件：

- `summary.csv`：跨数据集 mean±std 汇总
- `all_models_multi_seed.csv`：逐 seed、逐类别结果
- `all_results.json`：完整 JSON 结果

## 实验定义

该结果对应当前 No-VA final 配置，即：

| 分支 | 权重 | 状态 |
| --- | ---: | --- |
| InCTRL image residual | 0.35 | 启用 |
| InCTRL patch residual | 0.25 | 启用 |
| PQA | 0.25 | 启用 |
| TA | 0.15 | 启用 |
| VA | 0.00 | 不参与最终融合 |

Few-shot seeds：`[42, 123, 7]`

---

## MVTec → VisA/AITEX/ELPV 跨域结果

| Shot | Dataset | AUROC mean±std | AUPR mean±std | InCTRL baseline AUROC | Delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | VisA | 0.9030 ± 0.0015 | 0.9165 ± 0.0019 | 0.858 | +0.0450 |
| 2 | AITEX | 0.7994 ± 0.0058 | 0.5737 ± 0.0072 | 0.761 | +0.0384 |
| 2 | ELPV | 0.8601 ± 0.0077 | 0.9322 ± 0.0032 | 0.839 | +0.0211 |
| 4 | VisA | 0.8916 ± 0.0021 | 0.9065 ± 0.0021 | 0.877 | +0.0146 |
| 4 | AITEX | 0.7999 ± 0.0007 | 0.5491 ± 0.0069 | 0.790 | +0.0099 |
| 4 | ELPV | 0.8705 ± 0.0032 | 0.9387 ± 0.0016 | 0.846 | +0.0245 |
| 8 | VisA | 0.9060 ± 0.0018 | 0.9207 ± 0.0019 | 0.887 | +0.0190 |
| 8 | AITEX | 0.8003 ± 0.0034 | 0.4949 ± 0.0116 | 0.806 | -0.0057 |
| 8 | ELPV | 0.8815 ± 0.0023 | 0.9431 ± 0.0011 | 0.872 | +0.0095 |

## 关键观察

- No-VA final 在 VisA 2/4/8-shot 均高于原始 InCTRL baseline。
- No-VA final 在 AITEX 2/4-shot 高于 baseline，但 8-shot 低于 baseline `-0.0057`。
- No-VA final 在 ELPV 2/4/8-shot 均高于 baseline。
- 2-shot 三个目标域平均 AUROC 为 `0.8542`。
- VA-small(E1) 2-shot 三个目标域平均 AUROC 为 `0.8502`，比 No-VA 低约 `-0.0039`。

## VA-small(E1) 直接对照

| Dataset | No-VA 2-shot AUROC | VA-small 2-shot AUROC | Delta | No-VA AUPR | VA-small AUPR | AUPR Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VisA | 0.9030 | 0.9001 | -0.0029 | 0.9165 | 0.9137 | -0.0028 |
| AITEX | 0.7994 | 0.7974 | -0.0020 | 0.5737 | 0.5703 | -0.0034 |
| ELPV | 0.8601 | 0.8532 | -0.0069 | 0.9322 | 0.9300 | -0.0022 |
| **Mean** | **0.8542** | **0.8502** | **-0.0039** | **0.8075** | **0.8047** | **-0.0028** |

结论：E1 的 VA-small 配置在 2-shot 下未超过 No-VA final；三个目标域 AUROC 和 AUPR 均略低。当前证据支持保留 No-VA final 作为更稳配置，同时继续用 VA-mid/VA-strong/VA-mask 做补充验证。
