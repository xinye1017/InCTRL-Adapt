#!/usr/bin/env python3
"""Inspect VA checkpoint weights - fixed version"""
import sys, torch
sys.path.insert(0, 'd:/Data/Downloads/InCTRL')

ckpt = torch.load('d:/Data/Downloads/InCTRL/checkpoints/InCTRL_trained_on_MVTec_VA/4/checkpoint', map_location='cpu')

def inspect_key(name, t):
    if t.dtype.is_floating_point:
        return f'  {name}: shape={t.shape}, mean={t.mean().item():.6f}, std={t.std().item():.6f}'
    else:
        return f'  {name}: shape={t.shape}, dtype={t.dtype}'

# Visual Adapter keys
va_keys = [k for k in ckpt if 'visual_adapter' in k]
print('=== Visual Adapter Keys ===')
for k in va_keys:
    print(inspect_key(k, ckpt[k]))

# Adapter keys (global)
adapter_keys = [k for k in ckpt if 'adapter' in k and 'visual_adapter' not in k]
print('\n=== Global Adapter Keys ===')
for k in adapter_keys:
    print(inspect_key(k, ckpt[k]))

# Diff head keys
diff_keys = [k for k in ckpt if 'diff_head' in k]
print('\n=== Diff Head Keys ===')
for k in diff_keys:
    print(inspect_key(k, ckpt[k]))

# Check last layer of VA adapters (should be near-zero if training didn't break identity)
print('\n=== Last Layer Magnitudes (residual path) ===')
for k in va_keys:
    if 'fc.3.weight' in k or 'fc.5.weight' in k or k.endswith('.fc.3.weight') or k.endswith('.fc.5.weight'):
        t = ckpt[k]
        if t.dtype.is_floating_point:
            max_abs = t.abs().max().item()
            l2 = t.norm().item()
            print(f'  {k}: max|w|={max_abs:.6f}, L2={l2:.6f}')

# Also check the local adapter last layer
print('\n=== Checking all Linear weights in adapters ===')
for k in va_keys:
    if 'weight' in k and ckpt[k].dim() == 2:
        t = ckpt[k]
        max_abs = t.abs().max().item()
        l2 = t.norm().item()
        print(f'  {k}: max|w|={max_abs:.6f}, L2={l2:.4f}')

# Count total trainable params
trainable = sum(v.numel() for k, v in ckpt.items()
                if 'visual_adapter' in k or ('adapter' in k and 'visual' not in k) or 'diff_head' in k)
print(f'\nTotal trainable params (approx): {trainable:,}')
