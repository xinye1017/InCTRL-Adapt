import torch

from open_clip.visual_adapter import ResidualMLPAdapter, VisualAdapter


def test_residual_mlp_adapter_preserves_shape_and_starts_as_identity():
    torch.manual_seed(7)
    adapter = ResidualMLPAdapter(dim=8, reduction=4, zero_init=True)
    x = torch.randn(2, 5, 8)

    y = adapter(x)

    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)


def test_visual_adapter_updates_global_and_patch_tokens_when_not_zero_init():
    torch.manual_seed(11)
    adapter = VisualAdapter(dim=8, reduction=4, zero_init=False)
    global_feat = torch.randn(2, 8)
    patch_feats = [torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 4, 8)]

    adapted_global, adapted_patches = adapter(global_feat, patch_feats)

    assert adapted_global.shape == global_feat.shape
    assert len(adapted_patches) == 3
    assert adapted_patches[0].shape == patch_feats[0].shape
    assert not torch.allclose(adapted_global, global_feat)
    assert not torch.allclose(adapted_patches[0], patch_feats[0])


def test_visual_adapter_can_use_different_global_and_patch_dims():
    adapter = VisualAdapter(dim=8, local_dim=12, reduction=4, zero_init=True)
    global_feat = torch.randn(2, 8)
    patch_feats = [torch.randn(2, 4, 12)]

    adapted_global, adapted_patches = adapter(global_feat, patch_feats)

    assert adapted_global.shape == global_feat.shape
    assert adapted_patches[0].shape == patch_feats[0].shape
