import torch
import torch.nn.functional as F
from torch import nn

from open_clip.adaptclip_textual_adapter import AdaptCLIPTextualAdapter, BinaryPromptLearner


def fake_tokenizer(text):
    if isinstance(text, str):
        return torch.arange(77, dtype=torch.long).unsqueeze(0)
    return torch.arange(77, dtype=torch.long).unsqueeze(0).expand(len(text), -1)


def fake_encode_text_prompted(prompts, tokenized_prompts, normalize=False):
    feats = prompts.mean(dim=1)
    return F.normalize(feats, dim=-1) if normalize else feats


def test_binary_prompt_learner_creates_two_learnable_contexts():
    learner = BinaryPromptLearner(ctx_dim=8, n_ctx=12)

    assert learner.ctx_pos.shape == (1, 12, 8)
    assert learner.ctx_neg.shape == (1, 12, 8)
    assert learner.ctx_pos.requires_grad is True
    assert learner.ctx_neg.requires_grad is True


def test_prompt_learner_lazily_builds_prompt_embeddings():
    learner = BinaryPromptLearner(ctx_dim=8, n_ctx=4)
    token_embedding = nn.Embedding(100, 8)

    prompts, tokenized = learner(
        batch_size=3,
        device=torch.device("cpu"),
        token_embedding=token_embedding,
        tokenizer=fake_tokenizer,
    )

    assert prompts.shape == (6, 77, 8)
    assert tokenized.shape == (6, 77)
    assert learner.is_initialized is True


def test_adaptclip_textual_adapter_outputs_expected_shapes():
    adapter = AdaptCLIPTextualAdapter(ctx_dim=8, image_size=32, n_ctx=4, logit_scale=10.0)
    token_embedding = nn.Embedding(100, 8)
    global_feat = torch.randn(2, 8)
    patch_feat = torch.randn(2, 4, 8)

    outputs = adapter(
        encode_text_prompted=fake_encode_text_prompted,
        token_embedding=token_embedding,
        tokenizer=fake_tokenizer,
        global_feat=global_feat,
        patch_feat=patch_feat,
    )

    assert outputs["text_features"].shape == (2, 2, 8)
    assert outputs["text_logits"].shape == (2, 2)
    assert outputs["text_logit"].shape == (2,)
    assert outputs["text_score"].shape == (2,)
    assert outputs["patch_text_logits"].shape == (2, 4, 2)
    assert outputs["text_map_logits"].shape == (2, 1, 32, 32)
    assert outputs["text_map"].shape == (2, 1, 32, 32)


def test_only_ctx_parameters_require_grad_in_text_branch():
    adapter = AdaptCLIPTextualAdapter(ctx_dim=8, image_size=32, n_ctx=4)
    trainable = {name for name, p in adapter.named_parameters() if p.requires_grad}
    all_params = {name for name, p in adapter.named_parameters()}

    assert trainable == {"prompt_learner.ctx_pos", "prompt_learner.ctx_neg"}
    assert len(all_params) >= 2
