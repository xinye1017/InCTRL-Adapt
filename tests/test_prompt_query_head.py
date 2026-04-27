import torch

from open_clip.prompt_query_head import PromptQuerySegHead, align_prompt_to_query


def test_align_prompt_to_query_selects_nearest_prompt_patch():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    p = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]])

    aligned, indices, residual_map = align_prompt_to_query(q, p)

    assert indices.tolist() == [[1, 0]]
    assert torch.allclose(aligned, q)
    assert torch.allclose(residual_map, torch.zeros(1, 2), atol=1e-6)


def test_prompt_query_seg_head_outputs_segmentation_and_image_logit():
    torch.manual_seed(23)
    head = PromptQuerySegHead(dim=8, hidden_dim=4, image_size=32, topk=2)
    query = torch.randn(2, 4, 8)
    prompt = torch.randn(2, 8, 8)

    outputs = head(query, prompt)

    assert outputs["pqa_seg_logits"].shape == (2, 1, 32, 32)
    assert outputs["pqa_logit"].shape == (2,)
    assert outputs["pqa_score"].shape == (2,)
    assert outputs["pqa_patch_map"].shape == (2, 4)
    assert outputs["inctrl_patch_map"].shape == (2, 4)
    assert outputs["context_tokens"].shape == (2, 4, 8)
    assert outputs["aligned_prompt_tokens"].shape == (2, 4, 8)
    assert outputs["aligned_indices"].shape == (2, 4)
