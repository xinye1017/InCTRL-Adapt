import torch

path2 = 'd:/Data/Downloads/InCTRL/checkpoints/InCTRL_trained_on_MVTec/2/checkpoint.pyth'
path4 = 'd:/Data/Downloads/InCTRL/checkpoints/InCTRL_trained_on_MVTec/4/checkpoint.pyth'
path8 = 'd:/Data/Downloads/InCTRL/checkpoints/InCTRL_trained_on_MVTec/8/checkpoint.pyth'
ckpt2 = torch.load(path2, map_location='cpu')
ckpt4 = torch.load(path4, map_location='cpu')
ckpt8 = torch.load(path8, map_location='cpu')

keys = sorted(ckpt2.keys())
# Show keys with diff > 0.01 between shot 2 and shot 4
print('=== Keys with significant diff (shot 2 vs shot 4) ===')
count = 0
for k in keys:
    v2 = ckpt2[k]
    v4 = ckpt4[k]
    if v2.dtype.is_floating_point:
        d24 = (v2 - v4).abs().max().item()
        if d24 > 0.01:
            count += 1
            print(f'  {k}: shape={v2.shape}, max_diff={d24:.6f}')
if count == 0:
    print('  (no keys with diff > 0.01)')

# Also check which keys are identical
print('\n=== Keys that are EXACTLY identical across all 3 shots ===')
identical_keys = []
for k in keys:
    v2 = ckpt2[k]
    v4 = ckpt4[k]
    v8 = ckpt8[k]
    if v2.dtype.is_floating_point:
        d24 = (v2 - v4).abs().max().item()
        d28 = (v2 - v8).abs().max().item()
        if d24 < 1e-6 and d28 < 1e-6:
            identical_keys.append(k)
print(f'  Count: {len(identical_keys)}/{len(keys)}')
print(f'  Examples: {identical_keys[:10]}')
