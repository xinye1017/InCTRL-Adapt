#!/usr/bin/env python3
"""临时诊断脚本：打印逐类结果和对比"""
import sys
sys.path.insert(0, r"d:/Data/Downloads/InCTRL")
import test_all_models as t
import json

transform = t.get_transform()
tokenizer = t.open_clip.get_tokenizer("ViT-B-16-plus-240")
shot = 4

all_details = {}
for variant in ["Baseline", "VA"]:
    spec = t.MODEL_VARIANTS[variant]
    model = t.build_model(spec["enable_va"], t.resolve_checkpoint_file(shot, variant), t.DEVICE)
    r = t.evaluate_model_on_dataset(model, tokenizer, transform, "visa", shot, t.DEVICE)
    all_details[variant] = r
    del model

# 打印逐类结果
for variant in ["Baseline", "VA"]:
    r = all_details[variant]
    print(f"=== {variant} shot={shot} VISA ===")
    for d in r["details"]:
        print(f"  {d['category']:16s} AUROC={d['auroc']:.4f} AP={d['ap']:.4f} n={d['n_samples']}")
    print(f"  macro AUROC={r['auroc']:.4f}  macro AP={r['ap']:.4f}")
    print(f"  micro AUROC={r['micro_auroc']:.4f}  micro AP={r['micro_ap']:.4f}")
    print()

# 对比
print("=== VA - Baseline (delta) ===")
for d1, d2 in zip(all_details["VA"]["details"], all_details["Baseline"]["details"]):
    assert d1["category"] == d2["category"]
    cat = d1["category"]
    d_auroc = d1["auroc"] - d2["auroc"]
    d_ap = d1["ap"] - d2["ap"]
    print(f"  {cat:16s} delta_auroc={d_auroc:+.4f}  delta_ap={d_ap:+.4f}")
