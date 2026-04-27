import math

import torch

from open_clip.pqa_adapter import PQAdapter, align_prompt_to_query


def test_align_prompt_to_query_matches_best_prompt_token():
    torch.manual_seed(42)
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # [B=1, N=2, C=2]
    prompts = torch.tensor([[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]])  # [B=1, M=3, C=2]

    aligned, indices, residual_map = align_prompt_to_query(query, prompts)

    assert aligned.shape == query.shape
    assert indices.shape == (1, 2)
    assert residual_map.shape == (1, 2)
    # Verify indices are valid (within range of prompt tokens)
    assert indices.min() >= 0 and indices.max() < prompts.shape[1]
    # Residual map should be in [0, 1] range (cosine distance / 2)
    assert residual_map.min() >= 0 and residual_map.max() <= 1


def test_pq_adapter_output_shapes_and_range():
    torch.manual_seed(7)
    adapter = PQAdapter(dim=16, hidden_dim=32, image_size=64, topk=5, beta=0.5)

    batch, patches, dim = 2, 16, 16  # 16 patches = 4x4 grid
    query_tokens = torch.randn(batch, patches, dim)
    prompt_tokens = torch.randn(batch, patches * 3, dim)  # 3-shot

    out = adapter(query_tokens, prompt_tokens)

    assert out["pqa_seg_logits"].shape == (batch, 1, 64, 64)
    assert out["pqa_logit"].shape == (batch,)
    assert out["pqa_score"].shape == (batch,)
    assert out["pqa_patch_map"].shape == (batch, patches)
    assert 0 <= out["pqa_score"].min() <= out["pqa_score"].max() <= 1


def test_pq_adapter_without_seg_head_returns_alignment_residual_and_zero_logits():
    torch.manual_seed(7)
    adapter = PQAdapter(dim=16, hidden_dim=32, image_size=64, topk=5, beta=0.5, enable_seg_head=False)

    batch, patches, dim = 2, 16, 16
    query_tokens = torch.randn(batch, patches, dim)
    prompt_tokens = torch.randn(batch, patches * 3, dim)

    out = adapter(query_tokens, prompt_tokens)

    # Seg logits and logit should be zero
    assert torch.equal(out["pqa_seg_logits"], torch.zeros(batch, 1, 64, 64))
    assert torch.equal(out["pqa_logit"], torch.zeros(batch))
    assert torch.allclose(out["pqa_score"], torch.full((batch,), 0.5))
    # pqa_patch_map should fall back to alignment residual
    assert out["pqa_patch_map"].shape == (batch, patches)
    assert out["pqa_patch_map"].min() >= 0 and out["pqa_patch_map"].max() <= 1


def test_pq_adapter_beta_controls_context_blending():
    torch.manual_seed(11)
    adapter_low_beta = PQAdapter(dim=8, hidden_dim=16, image_size=32, beta=0.1)
    adapter_high_beta = PQAdapter(dim=8, hidden_dim=16, image_size=32, beta=2.0)

    query = torch.randn(1, 4, 8)
    prompt = torch.randn(1, 12, 8)

    out_low = adapter_low_beta(query, prompt)
    out_high = adapter_high_beta(query, prompt)

    assert not torch.allclose(out_low["pqa_score"], out_high["pqa_score"])
